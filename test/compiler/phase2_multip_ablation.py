"""A1 fixed-block ablation, one caller-supplied source per run (Linux).

python3 test/compiler/phase2_multip_ablation.py --mlir MODEL.mlir --out NEW_DIR
Use --cell 2:narrow --trials 1 for a smoke run; by default all four cells use
the same 64-case source oracle. For a different graph (e.g. a deep stack), pass
--corpus JSON: {"source_sha256": "...", "cases": [{"x": [...], "truth": [...]}]}.
--replay DIR recomputes evidence without keys, a solver or an estimator.
Production heterogeneous cells require an explicit HEIR integer-scale anchor on
all three arms. Widths in scale_profile come from ordered concrete Q, not MILP
targets; nonbase_q_spread_bits excludes q[0]. Old narrow receipts replay as v1.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'src/frontend/snabf_frontend'))
from pass2_evidence import Oracle, TOLERANCE, load, require, save, sha

ARMS = ('no_trunc', 'joint', 'matched_exact')
CELLS = ('1:narrow', '1:heterogeneous', '2:narrow', '2:heterogeneous')
V1, V2 = 'snabf.a1.multip.v1', 'snabf.a1.multip.v2'
TEMPLATE = Path(__file__).with_name('phase2_multip_trials.go.in')
ADAPTER = '''package planned
import (
    "github.com/tuneinsight/lattigo/v6/core/rlwe"
    epoch "snabf.local/compiler/src/backend/ringswitch/epochs"
)
func EvalWith(r *epoch.Runtime, in *rlwe.Ciphertext) (*rlwe.Ciphertext, error) { return evaluate(r, in) }
'''


def corpus_for(src, external, n, bound):
    if external:
        data = load(external)
        require(data['source_sha256'] == sha(src), 'external source-oracle SHA-256 mismatch')
        rows = data['cases']
    else:
        rows = Oracle(src.read_text(encoding='utf8')).corpus()
    require(0 < n <= len(rows), 'not enough source-oracle trials')
    corpus = []
    for i, row in enumerate(rows[:n]):
        x, y = row['x'], row['truth']
        require(x and y and len(x) == len(y) and all(math.isfinite(v) for v in [*x, *y])
                and max(map(abs, x)) <= bound, f'invalid corpus row {i}')
        corpus.append(dict(trial=i, x=x, truth=y))
    return corpus


class StageFailure(ValueError):
    def __init__(self, label, log, code):
        super().__init__(f'{label}: exit {code}; see {log.name}')
        text = log.read_text(errors='replace')
        known = ('Pass 3 infeasible under requested scale, precision and QP bounds',
                 '[SNABF_UNSUPPORTED_OP] op=ckks.bootstrap reason=no_lossless_encrypted_import',
                 'delta scale is negative',
                 '[SNABF_SECURITY_INFEASIBLE]', '[SNABF_SPECIALIZATION_CYCLE]', '[SNABF_SPECIALIZATION_LIMIT]')
        reason = next((x for x in known if x in text), None)
        self.receipt = dict(status='compiler_rejected' if reason else 'execution_failed',
                            stage=label, exit_code=code, log=log.name, reason=reason or str(self))
        if '[SNABF_SECURITY_ORACLE_UNAVAILABLE]' in text or 'FileNotFoundError:' in text or '[Errno 2] No such file or directory' in text:
            self.receipt['status'] = 'dependency_unavailable'
        self.receipt['diagnostic_tail'] = text.splitlines()[-6:]


def failure(exc):
    if isinstance(exc, StageFailure):
        return exc.receipt
    return dict(status='dependency_unavailable' if isinstance(exc, FileNotFoundError) else 'validation_failed', reason=str(exc))


def run_logged(cell, label, argv, cwd, env):
    args = list(map(str, argv))
    with (cell/'commands.jsonl').open('a', encoding='utf8') as f:
        f.write(json.dumps(dict(stage=label, cwd=str(cwd), argv=args))+'\n')
    with (cell/(label+'.log')).open('w', encoding='utf8') as f:
        proc = subprocess.run(args, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise StageFailure(label, cell/(label+'.log'), proc.returncode)


def scale_profile(p, hetero):
    o, q, scale = p['options'], p['q'], p['scale']
    require(o.get('heterogeneous_scales') is hetero and o.get('heir_scale_anchor') is hetero,
            f'expected explicit boolean heterogeneous_scales={hetero} and heir_scale_anchor={hetero} in native audit')
    require(type(scale) is int and 0 < scale < 64 and len(q) >= 2 and
            all(type(v) is int and 1 < v < (1 << 64) and v % 2 == 1 for v in q), 'invalid concrete Q/scale')
    # Positive HEIR widths round to scale in [scale-1/2, scale+1/2).
    # Integer squares avoid accepting a prime just outside a floating log2 boundary.
    if hetero:
        for i, v in enumerate(q[1:], 1):
            require((1 << (2*scale-1)) <= v*v < (1 << (2*scale+1)),
                    f'q[{i}]={v} is outside the HEIR integer scale bin {scale}')
    lo, hi = min(q[1:]), max(q[1:])
    center = 1 << scale
    # log1p keeps the spread even when two log2(q_i) round to the same float.
    return dict(heir_scale_anchor=o['heir_scale_anchor'], default_log_scale=scale,
                q_log2_bits=[math.log2(v) for v in q], q_bit_lengths=[v.bit_length() for v in q],
                nonbase_q_spread_bits=math.log1p((hi-lo)/lo)/math.log(2),
                nonbase_q_max_abs_delta_from_scale_bits=max(abs(math.log1p((v-center)/center)/math.log(2)) for v in q[1:]))


def check_plan(d, alpha, hetero, policy, *, legacy_narrow=False):
    p = load(d/'work/parameters.json')
    o = p['options']
    require(len(p['p']) == alpha == o['gadget_alpha'] and o['heterogeneous_scales'] is hetero and o['max_k'] == policy
            and o['tolerance'] == TOLERANCE, 'requested Pass 3 options not present in audit')
    if not (legacy_narrow and not hetero and 'heir_scale_anchor' not in o):
        scale_profile(p, hetero)
    require(all(0 <= e['K'] <= policy for e in p['edges']), 'out-of-policy selected truncation')
    security = load(d/'work/source-security.json')
    require((security['ring'], security['ordered_q'], security['ordered_p'], security['inventory']) ==
            (p['ring'], p['q'], p['p'], p['public_instances']), 'security receipt not bound to ordered Q/P and inventory')
    q = math.prod(p['q'])
    mods = dict(Q=q, QP=q*math.prod(p['p']), envelope=1 << math.ceil(p['security_cap']))
    minima = {}
    for role, mod in mods.items():
        r = security['instances'][role]
        attacks = r['attacks_log2_rop']
        require(int(r['modulus']) == mod and {'usvp', 'dual_hybrid'} <= attacks.keys() and
                all(math.isfinite(v) and v >= 128 for v in attacks.values()), 'unsafe or unbound '+role)
        minima[role] = min(attacks.values())
    return p, minima


def program_contract(d, p):
    g = load(d/'program.json')
    require(len(g['epochs']) == 1 and g['source_ring'] == g['target_ring'] == p['ring']
            and 'ring_decision' not in g and g['ring_search']['outcome'] == 'retain_source_ring', 'not keep-source')
    e = g['epochs'][0]
    require((e['n'], e['q'], e['p'], e['scale']) == (p['ring'], p['q'], p['p'], p['scale']), 'emitted basis/scale mismatch')
    require(f'const PlanSHA256 = "{g["emit_sha256"]}"' in (d/'model.go').read_text(), 'stale emitted model')
    require([(e['ID'], e['K']) for e in g['instructions']] == [(e['ID'], e['K']) for e in p['edges']], 'emitter changed sites/K')
    return g


def shared_contract(cell, joint, exact):
    require(sha(cell/'joint/work/source-security.json') == sha(cell/'matched_exact/work/source-security.json'),
            'joint/matched security receipts differ')
    for k in ('epochs', 'slots', 'width', 'sink', 'source_ring', 'target_ring', 'galois_bytes'):
        require(joint[k] == exact[k], 'joint/matched incompatible '+k)
    clean = lambda g: [{k: v for k, v in row.items() if k != 'K'} for row in g['instructions']]
    require(clean(joint) == clean(exact), 'joint/matched plaintexts, operations, ordering or physical ABI differ')
    require(all(e['K'] == 0 for e in exact['instructions']), 'matched arm is not exact')


def compiled_receipt(d, p, g, hetero):
    return dict(status='compiled', emit_sha256=g['emit_sha256'], security_receipt_sha256=sha(d/'work/source-security.json'),
                scale_profile=scale_profile(p, hetero))


def compile_arms(cell, a, src, alpha, hetero, env, opt):
    receipts = {}

    def independent(arm, k):
        cmd = [sys.executable, ROOT/'src/tools/snabf_compile.py', '--mlir', src, '--out-dir', cell/arm,
               '--gadget-alpha', alpha, '--max-k', k, '--mode', 'keep-source', '--input-bound', a.input_bound,
               '--precision-bits', 12, '--security-bits', 128, '--cut-profile', 'single-live-v1', '--emit-go',
               '--log-n', a.log_n, '--q-count', a.q_count, '--min-slots', a.min_slots, '--snabf-opt', opt]
        if hetero:
            cmd.append('--heterogeneous-scales')
        if a.security_cache:
            cmd.extend(['--security-cache', a.security_cache.resolve()])
        run_logged(cell, arm, cmd, ROOT, env)
        p, _ = check_plan(cell/arm, alpha, hetero, k)
        g = program_contract(cell/arm, p)
        c = load(cell/arm/'compile.json')
        require(c['mode'] == 'keep-source' and c['security_bits'] == 128 and c['precision_bits'] == 12 and
                c['emit_sha256'] == g['emit_sha256'], 'incorrect compiler receipt')
        return p, g

    for arm, k in zip(ARMS[:2], (0, 1)):
        try:
            p, g = independent(arm, k)
            receipts[arm] = compiled_receipt(cell/arm, p, g, hetero)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            receipts[arm] = failure(exc)
    if receipts['joint']['status'] != 'compiled':
        receipts['matched_exact'] = dict(status='not_run', reason='joint was not admitted')
    else:
        try:
            receipts['matched_exact'] = matched_exact(cell, p, g, alpha, hetero, env, opt)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            receipts['matched_exact'] = failure(exc)
    save(cell/'admission.json', receipts)
    return receipts


def fixed_policy(o):
    return (f'--snabf-pass3=fixed-exact=true gadget-alpha={o["gadget_alpha"]} max-k=0 '
            f'heterogeneous-scales={str(o["heterogeneous_scales"]).lower()} heir-scale-anchor={str(o["heir_scale_anchor"]).lower()} '
            f'epsilon={o["tolerance"]} security-cap={o["security_cap"]} min-base={o["min_base"]} '
            f'min-scale={o["min_scale"]} max-scale={o["max_scale"]} guard={o["guard"]} '
            'source-arithmetic-fraction=1 security=source-security.json audit=parameters.json')


def matched_exact(cell, p, g, alpha, hetero, env, opt):
    jw = cell/'joint/work'
    matched = cell/'matched_exact'
    w = matched/'work'
    w.mkdir(parents=True)
    shutil.copyfile(jw/'source-security.json', w/'source-security.json')
    policy = fixed_policy(p['options'])  # Reuse the audited joint policy, not CLI defaults.
    def native(stage, *args):
        run_logged(cell, 'matched_'+stage, [opt, '--allow-unregistered-dialect', '--verify-each',
                   '--mlir-print-op-on-diagnostic=false', *args], w, env)
    native('fixed', policy, jw/'selected.mlir', '-o', 'screened.mlir')
    fixed, _ = check_plan(matched, alpha, hetero, 0)
    require(fixed['options'] == dict(p['options'], max_k=0), 'fixed-exact changed joint Pass 3 policy')
    require(all(fixed[k] == p[k] for k in ('ring', 'q', 'p', 'scale', 'key_id', 'public_instances', 'security_cap')), 'fixed-exact changed joint domain')
    native('pass4', f'--snabf-pass4=min-ring={p["ring"]} cut-profile=single-live-v1 audit=search.json', 'screened.mlir', '-o', 'planned.mlir')
    native('check', '--snabf-check-plan', 'planned.mlir', '-o', 'checked.mlir')
    native('lower', '--snabf-lower-ckks=direct-go=true', 'checked.mlir', '-o', 'lowered.mlir')
    native('emit', '--snabf-bind-lattigo=ring-go=../model.go audit=../program.json', 'lowered.mlir', '-o', os.devnull)
    emitted = program_contract(matched, fixed)
    shared_contract(cell, g, emitted)
    save(matched/'matched.json', dict(joint_selected_sha256=sha(jw/'selected.mlir'), security_receipt_sha256=sha(w/'source-security.json'),
                                      fixed_exact=True, keep_source=True, key_group='paired', fixed_pass3_policy=policy,
                                      heir_scale_anchor=fixed['options']['heir_scale_anchor']))
    return compiled_receipt(matched, fixed, emitted, hetero)


def prepare_runner(cell):
    # Independent module under the output directory; emitted .go files remain
    # byte-for-byte unchanged. No test packages/files are added to the checkout.
    run = cell/'runner'
    run.mkdir()
    mod = (ROOT/'go.mod').read_text()
    header, rest = mod.split('\n', 1)
    require(header == 'module snabf.local/compiler', 'unexpected compiler module')
    (run/'go.mod').write_text('module snabf.local/ablation\n'+rest+'\nrequire snabf.local/compiler v0.0.0\n'
                              +'replace snabf.local/compiler => '+json.dumps(str(ROOT), ensure_ascii=False)+'\n')
    shutil.copyfile(ROOT/'go.sum', run/'go.sum')
    shutil.copyfile(TEMPLATE, run/'multip_test.go')
    for arm in ARMS:
        d = run/arm
        d.mkdir()
        shutil.copyfile(cell/arm/'model.go', d/'model.go')
        (d/'adapter.go').write_text(ADAPTER)
    return run


def verify_events(events, program, alpha):
    # Plans are only a coverage oracle. Never add a planned number to a total.
    expected = {e['ID']: e for e in program['instructions'] if e['K'] > 0}
    seen = set()
    fields = ('TruncatedLimbsK', 'FullBlocksSkipped', 'PartialBlockZeroLimbs', 'ActiveExternalProducts',
              'MeasuredLimbNTTsSaved', 'MeasuredLimbINTTsSaved', 'MeasuredSingleLimbNTTsSaved')
    totals = dict.fromkeys(fields, 0)
    for v in events:
        site = v['ID']
        require(site in expected and site not in seen, 'unknown or duplicate actual site: '+site)
        seen.add(site)
        k, l = expected[site]['K'], expected[site]['Level']
        beta = (l+alpha)//alpha
        require((v['LevelQ'], v['Alpha'], v['TruncatedLimbsK'], v['OriginalBlocks'], v['FullBlocksSkipped'],
                 v['PartialBlockZeroLimbs'], v['ActiveExternalProducts']) ==
                (l, alpha, k, beta, k//alpha, k%alpha, beta-k//alpha), 'physical block mismatch: '+site)
        f = v['NativeForwardNTT']-v['ForwardNTT']
        b = v['NativeInverseNTT']-v['InverseNTT']
        require((f, b, f+b) == tuple(v[k] for k in fields[-3:]), 'non-measured transform accounting: '+site)
        for key in fields:
            totals[key] += v[key]
    require(seen == expected.keys(), 'missing observed gadget events')
    return dict(scope='one untimed replay; counters cover truncated gadget sites only', observed_sites=len(events), **totals)


def replay_admission(cell, alpha, hetero):
    ledger = load(cell/'admission.json')
    require(set(ledger) == set(ARMS), 'incomplete admission ledger')
    plans, programs = {}, {}
    for arm, k in zip(ARMS, (0, 1, 0)):
        if ledger[arm]['status'] != 'compiled':
            continue
        d = cell/arm
        p, _ = check_plan(d, alpha, hetero, k)
        g = program_contract(d, p)
        require(ledger[arm] == compiled_receipt(d, p, g, hetero), 'stale admission/scale evidence: '+arm)
        plans[arm], programs[arm] = p, g
    if 'matched_exact' in plans:
        require('joint' in plans, 'matched arm has no admitted joint')
        j, m = plans['joint'], plans['matched_exact']
        shared_contract(cell, programs['joint'], programs['matched_exact'])
        require(m['options'] == dict(j['options'], max_k=0) and all(j[k] == m[k] for k in
                ('ring', 'q', 'p', 'scale', 'key_id', 'public_instances', 'security_cap')), 'stale matched domain/policy')
        receipt = load(cell/'matched_exact/matched.json')
        policy = fixed_policy(j['options'])
        require(receipt == dict(joint_selected_sha256=sha(cell/'joint/work/selected.mlir'),
                security_receipt_sha256=ledger['joint']['security_receipt_sha256'], fixed_exact=True,
                keep_source=True, key_group='paired', fixed_pass3_policy=policy, heir_scale_anchor=hetero), 'stale matched receipt')
        commands = [json.loads(line) for line in (cell/'commands.jsonl').read_text().splitlines()]
        fixed = [c['argv'] for c in commands if c['stage'] == 'matched_fixed']
        require(len(fixed) == 1 and [v for v in fixed[0] if v.startswith('--snabf-pass3=')] == [policy] and
                fixed[0][-3].endswith('/joint/work/selected.mlir') and fixed[0][-2:] == ['-o', 'screened.mlir'],
                'fixed-exact command does not replay the joint selection and audited anchor')
    return ledger


def summary(cell, alpha, hetero, corpus, *, legacy=False):
    observations = load(cell/'observed.json')
    out, progs, times = {}, {}, {arm: [] for arm in ARMS}
    errors = {arm: [0.0, 0.0, 0, 0] for arm in (*ARMS, 'paired')}
    for name, k in zip(ARMS, (0, 1, 0)):
        d = cell/name
        p, security = check_plan(d, alpha, hetero, k, legacy_narrow=legacy)
        g = progs[name] = program_contract(d, p)
        obs = observations['observed'][name]
        group = 'independent' if name == 'no_trunc' else 'paired'
        require(obs['key_group'] == group and obs['trial'] == 0 and obs['emit_sha256'] == g['emit_sha256'], 'unbound runtime observation')
        bound = observations['bindings'][group]
        require((bound['n'], bound['q'], bound['p'], bound['scale']) == (p['ring'], p['q'], p['p'], p['scale']), 'bound runtime parameters differ')
        keys, ep = bound['keys'], g['epochs'][0]
        roles = [('public', 0)]+([('relin', 0)] if any(e['op']=='relin' for e in g['instructions']) else [])
        roles += [('galois', v) for v in ep['galois']]
        require(sorted(roles) == sorted((r['Role'], r['Galois']) for r in keys) and all(
            r['KeyID'] == ep['key_id'] and r['N'] == p['ring'] and r['LevelQ'] == len(p['q'])-1 and
            r['LevelP'] == alpha-1 and r['Bytes'] > 0 for r in keys), 'actual serialized keys do not cover screened domain')
        require(sum(r['Bytes'] for r in keys if r['Role']=='galois') == g['galois_bytes'], 'stale serialized Galois inventory')
        out[name] = dict(n=p['ring'], q=p['q'], p=p['p'], log2_QP=math.log2(math.prod(p['q']+p['p'])),
                         security_min_log2_rop=security, receipt_sha256=sha(d/'work/source-security.json'),
                         key_group=group, evaluation_keys_sha256=bound['evaluation_keys_sha256'],
                         serialized_bytes_with_public=sum(r['Bytes'] for r in keys),
                         serialized_evaluation_key_bytes=sum(r['Bytes'] for r in keys if r['Role']!='public'),
                         actual=verify_events(obs['events'], g, alpha))
        if not legacy:
            out[name]['scale_profile'] = scale_profile(p, hetero)
    shared_contract(cell, progs['joint'], progs['matched_exact'])
    m = load(cell/'matched_exact/matched.json')
    require(m['joint_selected_sha256'] == sha(cell/'joint/work/selected.mlir') and
            m['security_receipt_sha256'] == out['joint']['receipt_sha256'], 'fixed replay lost its input/receipt binding')

    def compare(key, v, truth):
        z = errors[key]
        for i, y in enumerate(v):
            d = abs(y-truth[i%len(truth)])
            require(math.isfinite(d), 'nonfinite decoded error')
            z[0] = max(z[0], d)
            z[1] += d*d
            z[2] += 1
            z[3] += int(d > TOLERANCE)

    with (cell/'trials.jsonl').open() as f:
        for sample, line in zip(corpus, f, strict=True):
            trial = json.loads(line)
            rows = trial['arms']
            require(trial['trial'] == sample['trial'] and set(rows) == set(ARMS), 'wrong trial inventory')
            require(rows['joint']['input_sha256'] == rows['matched_exact']['input_sha256'] != rows['no_trunc']['input_sha256'],
                    'pairing must be same joint ciphertext, separate baseline ciphertext')
            for name, r in rows.items():
                g = progs[name]
                sink = g['instructions'][g['sink']]
                require(r['key_group'] == out[name]['key_group'] and r['eval_ns'] > 0 and len(r['input_sha256']) == 64 and
                        len(r['decoded']) == g['slots'] and len(sample['truth']) == g['width'] and r['level'] == sink['Level'] and
                        abs(r['log_scale']-sink['physical_log_scale']) <= 1e-7, 'wrong trial ABI')
                if trial['trial'] == 0:
                    require(r['input_sha256'] == observations['observed'][name]['input_sha256'], 'diagnostic used a different input')
                compare(name, r['decoded'], sample['truth'])
                times[name].append(r['eval_ns'])
            compare('paired', rows['joint']['decoded'], rows['matched_exact']['decoded'])
    for name in ARMS:
        t = sorted(times[name])
        peak, ss, n, misses = errors[name]
        out[name].update(linf=peak, mse=ss/n, slot_checks=n, violations=misses,
                         eval_ns=dict(mean=statistics.fmean(t), median=statistics.median(t), min=t[0],
                                      p95=t[math.ceil(.95*len(t))-1], max=t[-1], samples=times[name]))
    peak, ss, n, _ = errors['paired']
    result = dict(alpha=alpha, scales='heterogeneous' if hetero else 'narrow', trials=len(corpus), arms=out,
                pair=dict(arms=['joint','matched_exact'], same_runtime_and_ciphertext=True, linf=peak, mse=ss/n),
                threshold=TOLERANCE, precision_pass=all(out[a]['violations']==0 for a in ARMS), probabilistic_cert_issued=False)
    if not legacy:
        # The baseline solves independently; never state that it selected the joint center unless it did.
        centers = {arm: out[arm]['scale_profile']['default_log_scale'] for arm in ARMS}
        result['scale_policy'] = dict(heir_scale_anchor=hetero, default_log_scales=centers,
                                      same_integer_center=len(set(centers.values())) == 1)
    return result


def digest_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def manifest(root):
    return {p.relative_to(root).as_posix(): digest_file(p) for p in sorted(root.rglob('*')) if p.is_file() and p != root/'hashes.json'}


def replay(root):
    require(load(root/'hashes.json') == manifest(root), 'manifest member/digest mismatch')
    record = load(root/'results.json')
    require(record['schema'] in (V1, V2) and len(record['requested_cells']) == len(record['cells']) and
            set(record['requested_cells']) == set(record['cells']) <= set(CELLS), 'unknown schema or incomplete cell inventory')
    require(record['source_sha256'] == sha(root/'source.mlir') and record['corpus_sha256'] == sha(root/'corpus.json'), 'changed source or corpus')
    corpus = load(root/'corpus.json')
    if record['oracle'] == 'source':
        require(corpus_for(root/'source.mlir', None, len(corpus), record['input_bound']) == corpus, 'source oracle mismatch')
    else:
        require(corpus_for(root/'source.mlir', root/'external-corpus.json', len(corpus), record['input_bound']) == corpus, 'external corpus mismatch')
    for name, row in record['cells'].items():
        alpha, scale = name.split(':')
        cell = root/('a'+alpha+'-'+scale)
        if record['schema'] == V2 and (cell/'admission.json').is_file():
            ledger = replay_admission(cell, int(alpha), scale=='heterogeneous')
            require('error' in row or all(v['status']=='compiled' for v in ledger.values()), 'measurement without full admission')
        if 'error' in row:
            if (cell/'admission.json').is_file():
                require(row['admission'] == load(cell/'admission.json'), 'admission ledger mismatch')
            continue
        require((cell/'admission.json').is_file(), 'missing admission for measured cell')
        require(row == summary(cell, int(alpha), scale=='heterogeneous', corpus, legacy=record['schema']==V1),
                'measurement replay mismatch: '+name)
    return record


def execute(a):
    src = a.mlir.resolve(strict=True)
    corpus = corpus_for(src, a.corpus, a.trials, a.input_bound)
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(src, root/'source.mlir')
    if a.corpus:
        shutil.copyfile(a.corpus, root/'external-corpus.json')
    save(root/'corpus.json', corpus)
    ext = Path(os.getenv('SNABF_EXTERN', ROOT/'extern')).resolve()
    opt = (a.snabf_opt or Path(os.getenv('SNABF_OPT', ROOT/'build/native/ir/snabf-opt'))).resolve()
    sdk = Path(os.getenv('SNABF_MLIR_ROOT', ext/'mlir24/usr/lib/llvm-24'))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', GOPROXY='off', GOTOOLCHAIN='local', GOWORK='off',
               GOPATH=str(ext/'gopath'), GOCACHE=os.getenv('GOCACHE', str(ext/'go-cache')), GOFLAGS='')
    env.pop('SNABF_PASS4_ESTIMATES', None)
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, [sdk/'lib', sdk.parent/'x86_64-linux-gnu', ext/'highs-install/lib', env.get('LD_LIBRARY_PATH','')]))
    requested = a.cell or list(CELLS)
    require(len(requested)==len(set(requested)), 'duplicate cell')
    record = dict(schema=V2, requested_cells=requested, source_sha256=sha(src), corpus_sha256=sha(root/'corpus.json'),
                  log_n=a.log_n, q_count=a.q_count, min_slots=a.min_slots, security_bits=128, precision_bits=12,
                  input_bound=a.input_bound, trials=len(corpus), oracle='external' if a.corpus else 'source',
                  tools={str(x): digest_file(x) for x in (opt, ext/'go/bin/go', ROOT/'src/tools/snabf_compile.py', TEMPLATE, Path(__file__)) if x.is_file()},
                  timing='generated Eval only; matched uses joint runtime, keys, plaintexts and ciphertext; one separate observed replay', cells={})
    for name in requested:
        alpha, scale = name.split(':')
        cell = root/('a'+alpha+'-'+scale)
        cell.mkdir()
        try:
            admission = compile_arms(cell, a, root/'source.mlir', int(alpha), scale=='heterogeneous', env, opt)
            require(all(r['status']=='compiled' for r in admission.values()), 'incomplete admission; see admission.json')
            run = prepare_runner(cell)
            run_logged(cell, 'encrypted', [ext/'go/bin/go', 'test', '-mod=readonly', '-count=1', '-timeout', '0', '-v', '.'], run,
                       dict(env, SNABF_A1_CELL=str(cell), SNABF_A1_CORPUS=str(root/'corpus.json')))
            record['cells'][name] = summary(cell, int(alpha), scale=='heterogeneous', corpus)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            record['cells'][name] = dict(error=str(exc), failure=failure(exc), measured=False,
                                         admission=load(cell/'admission.json') if (cell/'admission.json').exists() else {})
        save(root/'results.json', record)
        print(name, record['cells'][name].get('error', 'measured'), flush=True)
    save(root/'hashes.json', manifest(root))
    return replay(root)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mlir', type=Path)
    p.add_argument('--out', '--out-dir', type=Path)
    p.add_argument('--replay', type=Path)
    p.add_argument('--cell', action='append', choices=CELLS)
    p.add_argument('--corpus', type=Path)
    p.add_argument('--trials', type=int, default=64)
    p.add_argument('--input-bound', type=float, default=1.0)
    p.add_argument('--log-n', type=int, default=15)
    p.add_argument('--q-count', type=int, default=8)
    p.add_argument('--min-slots', type=int, default=4096)
    p.add_argument('--security-cache', type=Path)
    p.add_argument('--snabf-opt', type=Path)
    a = p.parse_args()
    if a.replay and not (a.mlir or a.out):
        result = replay(a.replay.resolve())
    elif a.mlir and a.out and not a.replay:
        result = execute(a)
    else:
        p.error('use --mlir MODEL --out NEW_DIR or --replay DIR')
    return int(not all(row.get('precision_pass', False) for row in result['cells'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
