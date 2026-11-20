"""Checks specific to named matrix experiments; no evaluator or HEIR bindings."""
import json
import math
from pathlib import Path
import re
import statistics

from pass2_evidence import TOLERANCE, TRIALS, load, require, sha


def scenario_contract(name, work, source_hash, oracle, p, search):
    src, dst = p['source_ring'], p['target_ring']
    if name == 'inference':
        return
    registered = load(Path(__file__).resolve().parents[2]/'test/pass4/matrix_sources.json')
    require(registered.get(work) == source_hash, 'named workload does not match frozen source')
    policy = search['policy']
    if name == 'equiv-single-ring':
        require(work in ('control','strided64','reuse64','residual') and src == dst == policy['min_ring'] and
                search['outcome'] == 'retain_source_ring' and not search['rounds'], 'paired comparison is not fixed single-ring')
        return
    live = search['baseline_cut']['live_values']
    require(len({v['id'] for v in live}) == len(live) and all(v['uses'] and v['ring_dim'] == src and
            len(v['q']) == v['active_level']+1 and v['key_domain'] for v in live), 'invalid live ciphertext contract')
    floor, rounds = policy['switching_key_security_bits'], search['rounds']
    if name == 'cross-residual':
        require(work == 'cross_residual' and oracle.cross_residual and floor == 128 and len(live) == 2 and src == dst and rounds and
                all(r['reason'] == 'residual_bypass_crosses_cut_unconverted' for r in rounds), 'crossing residual not rejected')
        require(any(v['producer'] == 'snabf.lintrans' and any(u['role'] == 'unconverted' and u.get('kind') == 'ckks.add' for u in v['uses']) for v in live) and
                any(v['producer'] == 'snabf.poly_eval' and any(u['role'] == 'main' and u['user'] == 'snabf.lintrans' for u in v['uses']) for v in live), 'not the live h / SiLU(h) pair')
        return
    require(name in ('reuse64-128','reuse64-512') and work == 'reuse64' and len(live) == 1 and src == 16384, 'incorrect reuse64 scenario')
    if name == 'reuse64-128':
        require(floor == 128 and dst == 8192 and search['outcome'] == 'ring_down', 'legal descent not taken')
    else:
        require(floor == 512 and src == dst and any(r['reason'] == 'switching_key_ext_modulus_security_rejected' for r in rounds), 'security refusal not reproduced')


def paired(run, corpus, program, metrics):
    out, old = run/'out', run/'work/legacy'
    require(sha(old/'planned.mlir') == sha(run/'work/checkpoint.mlir'), 'legacy input differs from common checkpoint')
    legacy = load(old/'audit.json')
    routes = {r['site']:r for r in program['routes']}
    require(len(legacy['backend']) == len(routes) and {r['site'] for r in legacy['backend']} == set(routes), 'different route sites')
    for r in legacy['backend']:
        d = routes[r['site']]
        require((r['Level'],1 << r['LogN'],[x for term in r['Terms'] for x in term]) ==
                (d['level'],d['n'],d['selected']['terms']), 'different per-arm rotations or key domain')
    width, slots = program['width'], program['slots']
    sink = program['instructions'][program['sink']]
    require(len(corpus) == TRIALS and 0 < width <= slots and slots % width == 0 and
            legacy['key_bytes'] == program['galois_bytes'] and len(legacy['sinks']) == 1 and legacy['sinks'][0]['level'] == sink['Level'], 'incompatible legacy ABI')
    require(metrics['same_ciphertext_and_keys'] is True and metrics['untimed_probe_evals'] == 2, 'unexpected comparison protocol')
    expected_ids = {str(k['Galois']):k['Bytes'] for k in metrics['keys'] if k['Role'] == 'galois'}
    require(metrics['galois_bytes_by_id'] == expected_ids and {str(k['galois']) for k in legacy['keys']} == set(expected_ids), 'paired key inventory')
    shared = metrics['shared_eval_keys_sha256']
    require(re.fullmatch('[0-9a-f]{64}', shared), 'missing shared evaluator fingerprint')
    peaks, sums, miss = [0.]*3,[[],[],[]],[0]*3
    ns, scales, inputs = [[],[]],[[],[]],set()
    with (out/'trials.jsonl').open() as stream:
        for j,(line,case) in enumerate(zip(stream,corpus,strict=True)):
            row = json.loads(line)
            x, y, truth = row['decoded_direct'],row['decoded_legacy'],case['truth']
            require(row['trial'] == case['trial'] == j and len(x) == len(y) == slots and len(truth) == width, 'paired slot coverage')
            digest = row['input_sha256']
            require(re.fullmatch('[0-9a-f]{64}',digest) and digest not in inputs and row['copy_sha256_direct_legacy'] == [digest,digest] and
                    row['shared_eval_keys_sha256'] == shared and row['eval_order'] == [j%2,1-j%2], 'ciphertext/key sharing or crossover order mismatch')
            inputs.add(digest)
            abi = row['output_direct_legacy']
            require(len(abi) == len(row['online_ns']) == 2, 'missing per-arm measurements')
            for arm,(phys,t) in enumerate(zip(abi,row['online_ns'],strict=True)):
                require(phys['N'] == sink['n'] == program['source_ring'] == program['target_ring'] and phys['Level'] == sink['Level'] and
                        math.isclose(phys['LogScale'],sink['physical_log_scale'],rel_tol=0,abs_tol=1e-7) and type(t) is int and t > 0, 'paired timing or output ABI mismatch')
                ns[arm].append(t); scales[arm].append(phys['LogScale'])
            errors = ([abs(a-truth[k%width]) for k,a in enumerate(x)], [abs(b-truth[k%width]) for k,b in enumerate(y)],
                      [abs(a-b) for a,b in zip(x,y,strict=True)])
            for k,err in enumerate(errors):
                require(all(map(math.isfinite,err)), 'nonfinite paired output')
                miss[k] += sum(e > TOLERANCE for e in err); peaks[k] = max(peaks[k],max(err)); sums[k].append(math.fsum(e*e for e in err))
    n = len(inputs)*slots
    mses = [math.fsum(t)/n for t in sums]
    require(miss == [0,0,0], f'threshold exceeded (direct, legacy, paired): {miss}')
    require((metrics['trials'],metrics['slots_per_arm']) == (TRIALS,n) and peaks == [*metrics['linf_direct_legacy'],metrics['paired_linf']] and
            all(math.isclose(a,b,rel_tol=1e-9) for a,b in zip(mses,[*metrics['mse_direct_legacy'],metrics['paired_mse']],strict=True)), 'paired runner norms do not replay')
    times = [statistics.mean(v)/1e6 for v in ns]
    require(all(math.isclose(a,b,rel_tol=1e-12) for a,b in zip(times,metrics['online_ms_direct_legacy'],strict=True)), 'paired timing mismatch')
    return dict(trials=TRIALS, measured_evaluations=2*TRIALS, untimed_probe_evaluations=2, slots_per_trial=slots, physical_slots_per_arm=n,
                logical_outputs_per_arm=TRIALS*width,tolerance=TOLERANCE,actual_ring=sink['n'],actual_level=sink['Level'],
                actual_log_scale_ranges=[[min(v),max(v)] for v in scales],online_mean_ms_direct_legacy=times,
                truth_linf_direct_legacy=peaks[:2],truth_mse_direct_legacy=mses[:2],truth_misses_direct_legacy=miss[:2],
                paired_linf=peaks[2],paired_mse=mses[2],paired_misses=miss[2],encryption_count=len(inputs),shared_eval_keys_sha256=shared)
