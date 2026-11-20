"""A2: generate a held-out width-64 Linear/SiLU graph and measure its actual depth.

--out NEW [--generate-only] or --replay RUN. A rejected cell or a shallow
measurement is evidence, never a successful 14--16-rescale observation.
"""
import argparse
import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import phase2_multip_ablation as ab
import phase2_multip_deep as source

SCHEMA = 'snabf.a2.middepth.v1'
OPS = {'source', 'mask', 'coefficient', 'add', 'add_plain', 'drop', 'baby',
       'giant', 'mul', 'relin', 'rescale'}


def exp64(x):
    # libm.exp differs by an ulp for some held-out inputs on Windows/Linux.
    # Decimal.exp rounds to nearest; enclose it before converting to binary64.
    value = Decimal.from_float(x)
    with localcontext() as ctx:
        for ctx.prec in (48, 96, 192):
            y = ctx.exp(value)
            lo, hi = float(ctx.next_minus(y)), float(ctx.next_plus(y))
            if lo == hi:
                return lo
    raise ValueError('ambiguous binary64 exponential')


def generate(layers, seed, input_seed, trials, input_bound=1.0):
    source.check(type(layers) is int and layers in (4, 5), 'expected 4 or 5 linear layers')
    source.check(all(type(i) is int and 0 <= i < 2**64 for i in (seed, input_seed))
                 and type(trials) is int and 2 <= trials <= 4096, 'invalid seeds/trial count')
    src = source.source_for(layers, 64, seed)
    data, peaks = source.corpus_for_source(src, layers, 64, trials, input_seed, input_bound=input_bound, exp_fn=exp64)
    data = source.packed(data)
    meta = dict(schema=SCHEMA, width=64, linear_layers=layers, silu=layers-1,
                seed=seed, input_seed=input_seed, trials=trials, input_bound=input_bound, exp='decimal-enclosed-binary64',
                source_sha256=source.sha(src), corpus_sha256=source.sha(data),
                preactivation_absmax=peaks)
    return src, data, meta


def depth(g):
    """Path depth, not the number of all rescale instructions in a parallel DAG."""
    q = g['epochs'][0]['q']
    nodes = g['instructions']
    source.check(len(g['epochs']) == 1 and nodes and q, 'one nonempty epoch required')
    r, drops, starts = [], [], []
    for i, e in enumerate(nodes):
        op, level, parents = e['op'], e['Level'], e['inputs']
        source.check(op in OPS and type(level) is int and 0 <= level < len(q) and
                     all(type(j) is int and 0 <= j < i for j in parents), 'unknown op, level or parent')
        if op == 'source':
            source.check(not parents and level == len(q)-1, 'unexpected source level')
            starts.append(level); r.append(0); drops.append(0)
            continue
        source.check(bool(parents), 'missing arithmetic inputs')
        steps = [nodes[j]['Level']-level for j in parents]
        if op in ('drop', 'rescale'):
            source.check(len(steps) == 1 and (steps[0] == 1 if op == 'rescale' else steps[0] > 0),
                         'invalid drop/rescale transition')
        else:
            source.check(all(d == 0 for d in steps), 'implicit level change')
        r.append(max(r[j] for j in parents) + int(op == 'rescale'))
        drops.append(max(drops[j] + (s if op == 'drop' else 0) for j, s in zip(parents, steps)))
    sink = g['sink']
    source.check(type(sink) is int and 0 <= sink < len(nodes) and len(starts) == 1, 'invalid sink or source count')
    return dict(input_level=starts[0], output_level=nodes[sink]['Level'],
                consumed_levels=starts[0]-nodes[sink]['Level'], critical_rescales=r[sink],
                max_path_drop_limbs=drops[sink], q_count=len(q))


def assess(root):
    old = source.read(root/'source.json')
    src, data, meta = generate(old['linear_layers'], old['seed'], old['input_seed'], old['trials'], old['input_bound'])
    source.check(meta == old and (root/'source.mlir').read_bytes() == src and (root/'corpus.json').read_bytes() == data,
                 'source/corpus not reproducible')
    run = root/'ablation'
    rec = ab.replay(run)
    source.check(rec['schema'] == ab.V3 and rec['joint_max_k'] >= 2 and rec['trials'] == old['trials']
                 and rec['oracle'] == 'external' and rec['input_bound'] == meta['input_bound'] and (run/'source.mlir').read_bytes() == src
                 and (run/'external-corpus.json').read_bytes() == data and rec['precision_bits'] == 12
                 and rec['security_bits'] == 128 and set(rec['cells']) == {'1:narrow','2:narrow'}, 'wrong ablation policy or source')
    cells, single_admission_ok = {}, False
    for name, row in rec['cells'].items():
        d = run/('a'+name.replace(':','-'))
        if 'error' in row:
            ledger = row.get('admission', {})
            # Only an actual parameter/security infeasibility is a valid single-P
            # negative control. Bootstrap rejection, dependency and harness failures are not.
            accepted_reasons = ('Pass 3 infeasible under requested scale, precision and QP bounds', '[SNABF_SECURITY_INFEASIBLE]')
            rejected = [v for v in ledger.values() if v['status'] == 'compiler_rejected']
            negative = (set(ledger) == set(ab.ARMS) and bool(rejected)
                        and all(v['status'] in ('compiled','compiler_rejected','not_run') for v in ledger.values())
                        and all(v['reason'] in accepted_reasons for v in rejected)
                        and all(v['log'] in ('joint.log','no_trunc.log') and (d/v['log']).is_file()
                                and v['reason'] in (d/v['log']).read_text(errors='replace') for v in rejected))
            cells[name] = dict(measured=False, parameter_infeasible=negative, admission=ledger, error=row['error'])
            if name == '1:narrow':
                single_admission_ok = negative
            continue
        arms = {}
        for arm in ab.ARMS:
            p = source.read(d/arm/'program.json')
            actual_depth = depth(p)
            source.check(actual_depth['q_count'] == len(row['arms'][arm]['q']), 'unbound emitted Q')
            # ab.replay has already compared every decrypted output level to the sink.
            active_k = [i['K'] for i in p['instructions'] if i['K']]
            h = row['arms'][arm]
            arms[arm] = dict(depth=actual_depth, selected_prefixes=active_k, counters=h['actual'],
                             linf=h['linf'], mse=h['mse'], eval_ns=h['eval_ns'], serialized_evk_bytes=h['serialized_evaluation_key_bytes'])
        passed = row['precision_pass'] and all(a['linf'] < 2**-12 for a in arms.values())
        cells[name] = dict(measured=True, precision_pass=passed, arms=arms)
        if name == '1:narrow':
            single_admission_ok = passed
    two = cells['2:narrow']
    qualified = bool(two['measured'] and two['precision_pass'] and all(
        14 <= two['arms'][a]['depth']['critical_rescales'] <= 16 for a in ('joint','matched_exact')))
    return dict(schema=SCHEMA, source_sha256=meta['source_sha256'], input_bound=meta['input_bound'], log_n_requested=rec['log_n'],
                q_requested=rec['q_count'], cells=cells, middepth_measured=qualified,
                matrix_pass=qualified and single_admission_ok, probabilistic_cert_issued=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--out',type=Path); m.add_argument('--replay',type=Path)
    p.add_argument('--generate-only',action='store_true')
    p.add_argument('--layers',type=int,choices=(4,5),default=4)
    p.add_argument('--seed',type=int,default=20261120); p.add_argument('--input-seed',type=int,default=314159)
    p.add_argument('--trials',type=int,default=64); p.add_argument('--q-count',type=int,default=16)
    p.add_argument('--input-bound',type=float,choices=(0.5,1.0),default=1.0,
                   help='explicit public input box; changes every corpus row and the certified precondition')
    p.add_argument('--log-n',type=int,choices=(15,16),default=15)
    p.add_argument('--security-cache',type=Path); p.add_argument('--snabf-opt',type=Path)
    args = p.parse_args()
    if args.replay:
        if args.generate_only: p.error('--generate-only requires --out')
        fresh = assess(args.replay)
        source.check(fresh == source.read(args.replay/'middepth.json'), 'middepth report mismatch')
    else:
        src, data, meta = generate(args.layers,args.seed,args.input_seed,args.trials,args.input_bound)
        d = args.out.resolve(); d.mkdir(parents=True,exist_ok=False)
        (d/'source.mlir').write_bytes(src); (d/'corpus.json').write_bytes(data); (d/'source.json').write_bytes(source.packed(meta))
        if args.generate_only:
            print('Generated source and cleartext oracle only; no measured depth.'); return 0
        ab.execute(argparse.Namespace(mlir=d/'source.mlir',corpus=d/'corpus.json',out=d/'ablation',
            snabf_opt=args.snabf_opt,security_cache=args.security_cache,log_n=args.log_n,q_count=args.q_count,
            min_slots=4096,input_bound=args.input_bound,trials=args.trials,cell=['1:narrow','2:narrow'],joint_max_k=2))
        fresh = assess(d)
        (d/'middepth.json').write_bytes(source.packed(fresh))
    print(json.dumps({k:fresh[k] for k in ('middepth_measured','matrix_pass')}))
    return 0 if fresh['matrix_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
