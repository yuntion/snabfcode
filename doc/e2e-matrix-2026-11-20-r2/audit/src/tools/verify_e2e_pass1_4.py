"""Independently audit one E2E run, or the preregistered reuse64 128/512 pair."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
import sys

# The independent oracle is stdlib-only. Do not initialize the HEIR front end
# (or require its Python bindings) just to read a saved evidence directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'frontend' / 'snabf_frontend'))
from pass2_evidence import Oracle, TOLERANCE, TRIALS, load, require, save, sha
from e2e_matrix_evidence import paired, scenario_contract


def seals(run, program):
    seal = program['emit_sha256']
    require(program['emitter'] == 'snabf.direct-go.v1' and re.fullmatch('[0-9a-f]{64}', seal), 'invalid emitter receipt')
    def ir_seals(name):
        return re.findall(r'snabf\.emit_sha256 = "([0-9a-f]{64})"', (run/'work'/name).read_text())
    for f in ('checked.mlir', 'lowered.mlir'):
        require(ir_seals(f) == [seal], 'missing/conflicting checker receipt in '+f)
    code = (run/'out/model.go').read_text()
    require(re.findall(r'^const PlanSHA256 = "([0-9a-f]{64})"$', code, re.M) == [seal], 'Go receipt mismatch')
    return {'precheck_receipts': ir_seals('planned.mlir'), 'emit_sha256': seal}


def keys(program, search, observed):
    epochs = {e['n']: e for e in program['epochs']}
    src, dst = program['source_ring'], program['target_ring']
    mixed = src != dst
    require(set(epochs) == {src, dst}, 'epoch set mismatch')
    expected = {(n, 'galois', k) for n,e in epochs.items() for k in e['galois']}
    expected.update((i['n'], 'relin', 0) for i in program['instructions'] if i['op'] == 'relin')
    expected.add((src, 'public', 0))
    if mixed:
        expected.add((src, 'down', 0))
    inventory = Counter((k['N'], k['Role'], k['Galois']) for k in observed)
    require(inventory == Counter(expected), 'measured key inventory disagrees with program')
    sizes = Counter()
    for k in observed:
        e, role = epochs[k['N']], k['Role']
        require(k['Bytes'] > 0 and k['KeyID'] == e['key_id'], 'invalid measured key or domain')
        if role == 'down':
            c = search['conversion']
            require((k['LevelQ'], k['LevelP'], k['Bytes'], k['TargetKeyID']) ==
                    (c['switch_level_q'], c['switch_level_p'], c['switch_bytes'], epochs[dst]['key_id']), 'conversion key mismatch')
        else:
            require((k['LevelQ'], k['LevelP']) == (len(e['q'])-1, len(e['p'])-1), 'measured key level')
            if role == 'galois':
                require(k['Bytes'] == e['key_bytes'], 'measured Galois key size')
        sizes[role] += k['Bytes']
    require(sizes['galois'] == program['galois_bytes'] and sizes['down'] == program.get('switch_bytes', 0), 'serialized key totals')
    return {r:sizes[r] for r in ('galois', 'down', 'relin', 'public')}


def measured(run, corpus, program, metrics):
    width, slots = program['width'], program['slots']
    require(0 < width <= slots and slots % width == 0 and len(corpus) == TRIALS, 'corpus/period dimensions')
    sink = program['instructions'][program['sink']]
    require(sink['n'] == program['target_ring'], 'wrong sink domain')
    count = misses = 0
    peak = 0.
    times, scales, ss = [], [], []
    with (run/'out/trials.jsonl').open() as stream:
        for i,(line, case) in enumerate(zip(stream, corpus, strict=True)):
            row = json.loads(line)
            truth, got = case['truth'], row['decoded64']
            require(i == row['trial'] == case['trial'] and len(truth) == width and
                    row['truth_one_period'] == truth and len(got) == slots, f'trial {i}: order/slot coverage')
            require(row['ring'] == sink['n'] and row['level'] == sink['Level'] and
                    math.isclose(row['log_scale'], sink['physical_log_scale'], rel_tol=0, abs_tol=1e-7), f'trial {i}: measured ABI')
            err = [abs(y - truth[j % width]) for j,y in enumerate(got)]
            require(all(map(math.isfinite, err)), f'trial {i}: nonfinite slot error')
            count += slots
            misses += sum(e > TOLERANCE for e in err)
            ss.append(math.fsum(e*e for e in err))
            peak = max(peak, max(err))
            require(type(row['online_ns']) is int and row['online_ns'] > 0, 'invalid Eval timing')
            times.append(row['online_ns']/1e6)
            scales.append(row['log_scale'])
    mse = math.fsum(ss)/count
    require((metrics['trials'],metrics['logical_outputs'],metrics['physical_encoded_slots_checked'],metrics['misses']) ==
            (len(times),width*TRIALS,count,misses) and len(times) == TRIALS, 'runner totals do not replay')
    require((metrics['actual_output_ring'],metrics['actual_level'],metrics['actual_log_scale']) ==
            (sink['n'],sink['Level'],scales[-1]), 'runner output ABI does not replay')
    require(peak == metrics['linf'] and math.isclose(mse,metrics['mse'],rel_tol=1e-9) and
            math.isclose(statistics.mean(times),metrics['online_ms'],rel_tol=1e-12), 'runner norms or timing do not replay')
    require(misses == 0, f'{misses}/{count} physical slots exceed {TOLERANCE}, max={peak}')
    return dict(trials=len(times), logical_outputs=width*len(times), slots_per_trial=slots, physical_slots=count,
                misses=misses, linf=peak, mse=mse, tolerance=TOLERANCE, actual_level=sink['Level'], actual_ring=sink['n'],
                actual_log_scale_range=[min(scales),max(scales)], online_mean_ms=statistics.mean(times), online_median_ms=statistics.median(times))


def verify(run, *, oracle_witness=None):
    inputs, out = run/'inputs', run/'out'
    config = dict(line.split('=',1) for line in (run/'env.snapshot').read_text().splitlines())
    files = {'source.mlir','estimates.json','corpus.json','security.json','profile.json'}
    require(config['MODE'] in ('source','pass3'), 'unknown input mode')
    if config['MODE'] == 'pass3':
        files.add('checkpoint.mlir')
    if (inputs/'pass3-options.json').is_file():
        files.add('pass3-options.json')
    locked = {f:sha(inputs/f) for f in sorted(files)}
    manifest = dict(line.split('  ',1)[::-1] for line in (run/'inputs.sha256').read_text().splitlines())
    require(manifest == {'inputs/'+f:h for f,h in locked.items()}, 'locked inputs differ from manifest')
    corpus = load(inputs/'corpus.json')
    oracle = Oracle((inputs/'source.mlir').read_text())
    if oracle_witness is None:
        require(oracle.corpus() == corpus, 'source oracle differs from frozen corpus')
    else:
        from e2e_oracle_witness import replay
        replay(oracle,corpus,locked,oracle_witness)
    program, search = load(out/'program.json'), load(out/'search.json')
    receipt = seals(run, program)
    require(program['ring_search'] == {k:v for k,v in search.items() if k != 'conversion'}, 'search and emitter disagree')
    profile, sec = load(inputs/'profile.json'), load(inputs/'security.json')
    src, dst = program['source_ring'], program['target_ring']
    require(dst == search['target'] and (src != dst) == (search['outcome'] == 'ring_down'), 'planned ring decision')
    epoch = next(e for e in program['epochs'] if e['n'] == src)
    require((src,epoch['q'],epoch['p'],epoch['scale']) == (1<<profile['logN'],profile['Q'],profile['P'],profile['logDefaultScale']), 'source concrete profile mismatch')
    cache = load(inputs/'estimates.json')
    require((cache['source_ring'],cache['source']['q'],cache['source']['p']) == (src,epoch['q'],epoch['p']) and
            sec['ring'] == src and sec['estimator'] == cache['estimator'], 'frozen security domain mismatch')
    for label,mod in [('Q', math.prod(epoch['q'])),('QP',math.prod(epoch['q']+epoch['p']))]:
        r = sec['instances'][label]
        require(int(r['modulus']) == mod and min(r['attacks_log2_rop'].values()) >= 128, 'source security proof input mismatch')
    policy = search['policy']
    require(policy['source_security_bits'] == policy['target_security_bits'] == 128 and
            policy['switching_key_security_bits'] >= max(128,int(config['SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS'])), 'weakened security floor')
    mixed = src != dst
    require([d['role'] for d in search['key_domains']] == (['source','target','switching_key_ext'] if mixed else ['source']) and
            sum(i['op'] == 'ring_down' for i in program['instructions']) == int(mixed), 'execution/domain mismatch')
    if mixed:
        c, ext = search['conversion'], search['key_domains'][2]
        require(c == program['ring_decision'], 'conversion audit mismatch')
        for label,n in [('switch_QP_large_ring',src),('switch_QP_small_secret',dst)]:
            s = c['security'][label]
            require(s['n'] == n and int(s['modulus']) == math.prod(ext['q']+ext['p']) and
                    min(s['usvp'],s['dual_hybrid']) >= policy['switching_key_security_bits'], 'unverified switching-key domain')
    metrics = load(out/'metrics.json')
    name, work = config.get('SCENARIO','inference'),config.get('WORKLOAD','')
    scenario_contract(name,work,locked['source.mlir'],oracle,program,search)
    is_pair = name == 'equiv-single-ring'
    evidence = {f:sha(out/f) for f in ('search.json','program.json','model.go','metrics.json','trials.jsonl')}
    if is_pair:
        evidence.update({'legacy/'+f:sha(run/'work/legacy'/f) for f in ('model.go','audit.json','checked.mlir','translated.go.txt')})
    return dict(mode=config['MODE'], strategy=config['STRATEGY'], scenario=name,workload=work,decode_domain='target' if mixed else 'source',
                live_ciphertexts=len(search['baseline_cut']['live_values']), conversion_keys=sum(k['Role'] == 'down' for k in metrics['keys']),
                inputs=locked, policy=policy, outcome=search['outcome'],
                source_ring=src, target_ring=dst, reasons=[r['reason'] for r in search['rounds']], **receipt,
                checkpoint_sha256=sha(run/'work/checkpoint.mlir'), tools_sha256=sha(run/'toolchain.sha256'),
                observed_key_bytes=keys(program,search,metrics['keys']), measurements=(paired if is_pair else measured)(run,corpus,program,metrics),
                evidence_sha256=evidence)


def reuse64_pair(a, b, *, oracle_witness=None):
    x,y = verify(a,oracle_witness=oracle_witness),verify(b,oracle_witness=oracle_witness)
    require(a.resolve() != b.resolve() and x['mode'] == y['mode'] and x['strategy'] == y['strategy'] == 'adaptive', 'not independent adaptive runs')
    for field in ('inputs','tools_sha256'):
        require(x[field] == y[field], 'pair differs in '+field)
    if x['mode'] == 'pass3':
        require(x['checkpoint_sha256'] == y['checkpoint_sha256'], 'replay checkpoint differs')
        upstream = {'kind':'byte_identical_checkpoint', 'sha256':x['checkpoint_sha256']}
    else:
        p3 = sha(a/'work/pass3.json')
        require(p3 == sha(b/'work/pass3.json'), 'concrete Pass 3 audit differs')
        # External HEIR hashes Types/Attributes when naming a private client.pack.
        # Compare a copy in memory only. No normalized text is ever fed to a
        # compiler or used as a checker receipt. Any other difference must fail.
        def comparable(run):
            ir = (run/'work/checkpoint.mlir').read_text()
            names = re.findall(r'(?m)^  func\.func private @(_assign_layout_[0-9]+)\(\) attributes \{heir\.interface = \{[^\n]*?roles = \["client\.pack"\]', ir)
            require(len(names) == 1, 'unrecognized private HEIR pack helper')
            name = re.escape(names[0])
            ir = re.sub(r'(@)'+name+r'(?=\(\))', r'\1__paired_client_pack', ir)
            ir, n = re.subn(r'(snabf\.emit_sha256 = ")[a-f0-9]{64}(")',r'\1<per-run receipt>\2',ir)
            require(n == 1, 'missing Pass 3 checker receipt')
            return ir
        require(comparable(a) == comparable(b), 'upstream SSA differs beyond private client.pack naming and its receipt')
        upstream = {'kind':'independent_source_compilations', 'concrete_pass3_sha256':p3,
                    'checkpoint_comparison':'identical modulo private HEIR client.pack symbol and its checker receipt'}
    p,q = dict(x['policy']),dict(y['policy'])
    require((p.pop('switching_key_security_bits'),q.pop('switching_key_security_bits')) == (128,512) and p == q, 'unexpected policy delta')
    require((x['source_ring'],x['target_ring'],y['source_ring'],y['target_ring']) == (16384,8192,16384,16384) and
            x['outcome'] == 'ring_down' and y['outcome'] == 'retain_source_ring', 'expected reuse64 positive and source-ring fallback')
    rejects = [r for r in load(b/'out/search.json')['rounds'] if r['reason'] == 'switching_key_ext_modulus_security_rejected']
    def bits(k): return min(k['usvp'],k['dual_hybrid'])
    require(any(bits(r['security']['switch_QP_small_secret']) < 512 and bits(r['security']['QP']) >= 128 for r in rejects), 'not a conversion-specific refusal')
    return dict(positive_run=str(a),strict_run=str(b), only_policy_change='switching_key_security_bits:128->512', upstream=upstream,positive=x,strict=y)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--reuse64-pair',type=Path,metavar='STRICT_RUN')
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    result = reuse64_pair(a.run,a.reuse64_pair) if a.reuse64_pair else verify(a.run)
    if a.output:
        save(a.output,result)
    print(json.dumps(result if a.reuse64_pair else {k:result[k] for k in ('outcome','source_ring','target_ring','observed_key_bytes','measurements')},allow_nan=False))
