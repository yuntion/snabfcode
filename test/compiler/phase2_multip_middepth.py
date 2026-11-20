"""A2: generate a held-out width-64 Linear/SiLU graph and measure its actual depth.

--out NEW [--generate-only] or --replay RUN. A rejected cell or a shallow
measurement is evidence, never a successful 14--16-rescale observation.
Set SNABF_PASS3_TRACE_REJECTIONS=1 on a new run to include candidate-level
evidence from rejected source searches. Old logs cannot acquire new causes.
"""
import argparse
import json
import math
import sys
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path

import phase2_multip_ablation as ab
import phase2_multip_deep as source

SCHEMA = 'snabf.a2.middepth.v1'
REJECTION = 'snabf.pass3.candidate-rejection.v1'
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


@lru_cache(maxsize=4096)
def prime_u64(n):
    """Deterministic uint64 Miller-Rabin, with PrimeInstantiator.cpp's bases."""
    if not 2 <= n < (1 << 64):
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = (d & -d).bit_length() - 1
    d >>= s
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x in (1, n-1):
            continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1:
                break
        else:
            return False
    return True


def rejection_records(lines, *, n, q_count, alpha, max_k, margin_bits=2):
    """Replay only candidate evidence, never an attribution for the whole arm."""
    rows = []

    def uint(s):
        source.check(isinstance(s, str) and s.isascii() and s.isdecimal() and str(int(s)) == s,
                     'noncanonical rejection integer')
        return int(s)

    def unique(pairs):
        obj = dict(pairs)
        source.check(len(obj) == len(pairs), 'duplicate candidate rejection field')
        return obj

    for line in lines:
        # Native JSON lines are mixed with MLIR diagnostics. JSON object member
        # order is not semantic; don't lose a record if its schema moves.
        if not line.lstrip().startswith('{') or 'snabf.pass3.candidate-rejection.' not in line:
            continue
        r = json.loads(line, object_pairs_hook=unique)
        if not str(r.get('schema', '')).startswith('snabf.pass3.candidate-rejection.'):
            continue
        source.check(r['schema'] == REJECTION and
                     [uint(r[k]) for k in ('n','q_count','alpha','max_k')] == [n,q_count,alpha,max_k],
                     'wrong rejection schema, cell or arm')
        stage, status, attempt = r['stage'], r['status'], uint(r['round'])
        source.check((stage == 'model_bounds' and attempt == 0) or
                     (stage == 'solver' and status == 'infeasible' and 1 <= attempt <= 12) or
                     (stage == 'concrete_replay' and status == 'rejected' and 1 <= attempt <= 12) or
                     (stage == 'refinement' and status == 'round_limit' and attempt == 12), 'wrong rejection stage')
        has_basis = any(k in r for k in ('q', 'p', 'basis_source'))
        source.check(has_basis or stage != 'concrete_replay', 'missing concrete rejection basis')
        if has_basis:
            origin = 'concrete_replay' if stage == 'concrete_replay' else 'profile_input'
            source.check(r.get('basis_source') == origin and isinstance(r.get('q'), list) and
                         isinstance(r.get('p'), list), 'wrong rejection basis source')
            q, p = [list(map(uint, r[key])) for key in ('q', 'p')]
            source.check(len(q) == q_count and len(p) == alpha and len(set(q+p)) == len(q+p) and
                         all(3 <= x < (1 << 60) and x % (2*n) == 1 for x in q+p), 'wrong rejection NTT basis')
            source.check(all(prime_u64(x) for x in q+p), 'composite rejection modulus')
        if r['reason'] == 'unknown':
            source.check('witness' not in r, 'unknown reason with a witness')
        else:
            source.check(stage == 'concrete_replay' and r['reason'] == 'aux_block_margin',
                         'unsupported candidate attribution')
            w = r['witness']
            first, end, bits = (uint(w[k]) for k in ('block_first', 'block_end_exclusive', 'margin_bits'))
            source.check(first < len(q) and first % alpha == 0 and end == min(first+alpha, len(q)) and
                         bits == margin_bits, 'wrong original block or threshold')
            actual, required = math.prod(p), math.prod(q[first:end]) << bits
            source.check(actual == uint(w['p_product']) and required == uint(w['required_product']) and
                         actual < required, 'auxiliary block rejection does not replay')
        rows.append(r)
    return rows


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
            # ab.replay checked the log digests. Do not infer a cause for an
            # unexecuted arm, from a generic message, or from diagnostic_tail.
            for arm in ('joint', 'no_trunc'):
                v = ledger.get(arm, {})
                if v.get('status') != 'compiler_rejected' or v.get('reason') != accepted_reasons[0]:
                    continue
                log = d / (arm+'.log')
                if v.get('log') == log.name and log.is_file():
                    with log.open(encoding='utf-8', errors='replace') as f:
                        rows = rejection_records(f, n=1 << rec['log_n'], q_count=rec['q_count'],
                                                 alpha=int(name.split(':')[0]), max_k=rec['joint_max_k'] if arm=='joint' else 0)
                    if rows:
                        cells[name].setdefault('candidate_rejections', {})[arm] = rows
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
    p.add_argument('--cost-profile',type=Path); p.add_argument('--cost-hardware-id')
    args = p.parse_args()
    if bool(args.cost_profile) != bool(args.cost_hardware_id): p.error('supply both measured pricing options')
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
            min_slots=4096,input_bound=args.input_bound,trials=args.trials,cell=['1:narrow','2:narrow'],joint_max_k=2,
            cost_profile=args.cost_profile,cost_hardware_id=args.cost_hardware_id))
        fresh = assess(d)
        (d/'middepth.json').write_bytes(source.packed(fresh))
    print(json.dumps({k:fresh[k] for k in ('middepth_measured','matrix_pass')}))
    return 0 if fresh['matrix_pass'] else 1


if __name__ == '__main__':
    sys.exit(main())
