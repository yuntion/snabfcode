"""A-1.1 diagnostic gate, NOT a feasibility, precision or security approval.

Linux: python3 -B phase2_a11_17q_attribution.py --prior EXTRACTED_A2A \
    --opt /trusted/build/ir/snabf-opt --out NEW_DIR
Offline (including Windows): replace --opt ... --out ... with --replay DIR.
The prior is the released A2a-17Q-56ddc7b archive. No HEIR rerun, external
package, key, solver relaxation, or fabrication of a physical basis is needed.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import phase2_17q_probe as frozen
from a11_magnitude_replay import read_selected
from pass3_isolation import need

PRIOR = '826b89125a63197f4027db93682e0d05be22c932d53d5c658b214d6518c0bf8c'
ARMS = [(a, k, name) for a in (1, 2) for k, name in ((0, 'no_trunc'), (2, 'joint'))]
SOURCES = ('src/compiler/pass3/ParameterMILP.cpp', 'src/compiler/pass3/ParameterMILP.h',
           'src/compiler/ir/Pass3IR.cpp', 'test/compiler/a11_magnitude_replay.py',
           'test/compiler/phase2_17q_probe.py', 'test/compiler/phase2_a11_17q_attribution.py')
sha, packed = frozen.sha, frozen.packed


def digest(data):
    return hashlib.sha256(data).hexdigest()


def trace_lines(lines):
    return [json.loads(line) for line in lines if line.startswith(b'{') and b'candidate-rejection' in line]


def transcript(path, header, flag):
    rows, module, denied = [], None, False
    stream = hashlib.sha256()
    with gzip.open(path, 'rb') as f:
        for line in f:
            stream.update(line)
            if line.startswith(b'module attributes') and module is None:
                module = digest(line)
            elif b': error: '+frozen.FAIL.encode() in line:
                denied = True
            elif line.startswith(b'{'):
                rows.extend(trace_lines([line]))
    need(denied and module == header and len(rows) == 11*int(flag), 'wrong native denial/module/trace count')
    return rows, dict(transcript_sha256=stream.hexdigest(), module_line_sha256=module, trace_rows=len(rows))


def verify_sample(r, original, dag, d):
    t = r['isolation']
    expr = t['base_expression']
    # On the frozen input the ONLY change allowed in the entire native record
    # is the new trace child. On affine inputs preserve the same search policy.
    old = {k: v for k, v in r.items() if k != 'isolation'}
    need(old == {k: v for k, v in original.items() if k != 'isolation'} and int(r['scale_bits']) == d,
         'candidate/round/admission changed')
    if dag.mode == 'naive':
        need({k: v for k, v in t.items() if k != 'base_expression'} == original['isolation'],
             'not the original construction guard/DAG')
    need(t['kind'] == 'base_width_v1' and [float(t[x]) for x in ('min_base','guard_bits','exclusive_ceiling')] == [20,1,59],
         'base width policy changed')
    need(set(expr) == {'schema','ir_interval_mode','magnitude_transfers','level_zero_terms'} and
         expr['schema'] == 'snabf.pass3.base-expression.v1' and expr['ir_interval_mode'] == dag.mode,
         'unknown expression or wrong IR interval mode')
    edges, values = t['edges'], expr['magnitude_transfers']
    need(len(edges) == len(values) == len(dag.edges), 'incomplete magnitude transfers')
    base = []
    for i, (e, v, src) in enumerate(zip(edges, values, dag.edges)):
        need((e['id'], e['op'], int(e['level']), list(map(int,e.get('inputs',[])))) ==
             (src['id'],src['op'],src['level'],src['inputs']), 'unbound edge: ' + src['id'])
        need({k: float(x) for k,x in v.items()} == src['transfer'], 'magnitude or literal: ' + src['id'])
        if src['level'] == 0:
            need(float(e['magnitude']) == dag.mag(i), 'base magnitude mismatch')
            base.append(i)
    need(len(base) == len(expr['level_zero_terms']) == 1 and dag.edges[base[0]]['id'] == 'lin3.rescale',
         'not the isolated level-zero edge')
    i = base[0]
    e = dag.edges[i]
    log_b = math.log2(max(1., dag.mag(i)))
    nominal = e['cls'] * d
    raw = nominal + 1 + log_b
    lower = math.floor(raw) - .25
    expected = dict(edge=i, scale_class=e['cls'], nominal_scale_bits=nominal, guard_bits=1,
                    log2_clamped_decoded_abs_bits=log_b, before_floor_bits=raw,
                    floor_delta_bits=math.floor(raw)-raw, offset_bits=-.25, lower_bits=lower,
                    nominal_numerator_bits=dag.edges[e['inputs'][0]]['cls']*d,
                    nominal_drop_bits=-d, dropped_q_index=e['level']+1)
    row = expr['level_zero_terms'][0]
    need(set(row) == set(expected), 'missing or invented guard operand')
    for k, v in expected.items():
        # Only log2 is a libm operation; the frozen points are far from a
        # floor boundary. Use the native stored pre-floor sum for its delta.
        if k in ('before_floor_bits', 'log2_clamped_decoded_abs_bits'):
            ok = abs(float(row[k])-v) <= math.ulp(v)
        elif k == 'floor_delta_bits':
            ok = float(row[k]) == math.floor(raw)-float(row['before_floor_bits'])
        else:
            ok = float(row[k]) == v
        need(ok, 'wrong signed guard term: ' + k)
    certain = nominal + math.frexp(max(1., dag.mag(i)))[1] - .25  # +1 guard, -1 from frexp
    need(lower == certain == float(t['builder_lower']) and lower >= 59 and
         expected['nominal_numerator_bits']-d == nominal, 'guard did not independently exclude')
    return dict(scale_bits=d, nominal_numerator_bits=expected['nominal_numerator_bits'],
                nominal_drop_bits=-d, magnitude_bits=log_b, guard_bits=1, before_floor_bits=raw,
                floor_delta_bits=math.floor(raw)-raw, offset_bits=-.25, b0_bits=lower, without_edge_bits=20)


def tamper_tests(sample, original, dag):
    def check(label, edit):
        bad = copy.deepcopy(sample)
        edit(bad['isolation']['base_expression'])
        try:
            verify_sample(bad, original, dag, int(sample['scale_bits']))
        except ValueError:
            return label
        raise AssertionError('accepted tamper: '+label)

    def flip_literal(t):
        v = next(v for v in t['magnitude_transfers'] if float(v.get('signed_literal',0)) < 0)
        v['signed_literal'] = str(-float(v['signed_literal']))  # same abs and same sink!

    def change_table(t):
        v = next(v for v in t['magnitude_transfers'] if 'abs_weight' in v)
        v['abs_weight'] = str(math.nextafter(float(v['abs_weight']), math.inf))

    return [check('same_abs_opposite_signed_coefficient', flip_literal),
            check('table_maximum_one_ulp', change_table),
            check('rescale_wrong_sign', lambda t: t['level_zero_terms'][0].update(nominal_drop_bits='36')),
            check('invented_error_term', lambda t: t['level_zero_terms'][0].update(err_bound='0'))]


def replan(opt, source, path, mode):
    # Exercise the real Pass 2 option and the existing independent plan check.
    args = [opt, '--allow-unregistered-dialect', '--verify-each', '--mlir-print-op-on-diagnostic=false',
            '--snabf-pass2-candidates=depth=16 epsilon=0.000244140625 max-output-error=0.000244140625 interval-mode='+mode,
            '--snabf-plan', '--snabf-check-plan', source, '-o', path]
    subprocess.run(args, cwd=path.parent, check=True)


def run_arm(opt, selected, out, alpha, k):
    out.mkdir()
    receipts = {}
    with selected.open('rb') as f:
        header = digest(next(f))
    for flag in ('1', '0'):
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            w = Path(tmp)
            (w/'selected.mlir').symlink_to(selected)
            path = out/f'trace-{flag}.log.gz'
            with tempfile.TemporaryFile(dir=w) as log:
                p = subprocess.run([opt, *frozen.native_args(alpha,k)], cwd=w, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, SNABF_PASS3_TRACE_REJECTIONS=flag))
                log.seek(0)
                with gzip.GzipFile(str(path), 'wb', compresslevel=1, mtime=0) as saved:
                    shutil.copyfileobj(log, saved)
            need(p.returncode == 1 and list(w.iterdir()) == [w/'selected.mlir'],
                 f'not a clean original Pass 3 denial: a{alpha}-k{k} trace={flag}')
            _, fields = transcript(path, header, flag)
            receipts[flag] = dict(exit=p.returncode, proposal=False, audit=False, **fields)
    return receipts


def produce(opt, prior, out):
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(opt_sha256=sha(opt), source_file_sha256={s:sha(frozen.ROOT/s) for s in SOURCES}, arms={})
    for a, k, _ in ARMS:
        source = prior/'capture'/f'a{a}-k{k}'/'selected.mlir'
        # A shared module header is not a shared complete input. Replan each
        # capture separately; even the two k arms can have different bytes.
        with tempfile.TemporaryDirectory(dir=out) as tmp:
            p = Path(tmp)/'naive.mlir'
            replan(opt, source, p, 'naive')
            marker = b'snabf.interval_mode = "naive", '
            data = p.read_bytes()
            need(data.count(marker) == 1 and data.replace(marker,b'',1) == source.read_bytes(),
                 'naive replan changed the frozen selection')
        affine = out/f'affine-a{a}-k{k}.mlir'
        replan(opt, source, affine, 'affine')
        for mode, path in [('naive',source),('affine',affine)]:
            arm = f'{mode}-a{a}-k{k}'
            receipt['arms'][arm] = run_arm(opt,path,out/arm,a,k)
    (out/'runs.json').write_bytes(packed(receipt))


def replay(prior, out, old):
    runs = json.loads((out/'runs.json').read_bytes())
    report = dict(schema='snabf.a11.17q-attribution.v1', prior_diagnosis_sha256=PRIOR,
                  source_sha256=old['source_sha256'], corpus_sha256=old['corpus_sha256'],
                  runs_sha256=sha(out/'runs.json'), opt_sha256=runs['opt_sha256'], arms={},
                  stage='model_bounds/base_width', solver_called=False, concrete_basis=False,
                  graph_infeasibility_proven=False, production_matrix_pass=False)
    seen = {'naive': [], 'affine': []}
    delta, log_delta = [], []
    need(set(runs['arms']) == {f'{mode}-a{a}-k{k}' for mode in seen for a,k,_ in ARMS}, 'wrong arm inventory')
    for a, k, name in ARMS:
        prior_arm = old['arms'][f'a{a}/{name}']
        with (prior/'probe/ablation'/f'a{a}-narrow'/(name+'.log')).open('rb') as f:
            original = trace_lines(f)
        for mode in seen:
            arm = f'{mode}-a{a}-k{k}'
            path = prior/'capture'/f'a{a}-k{k}'/'selected.mlir' if mode == 'naive' else out/f'{arm}.mlir'
            dag = read_selected(path)
            need(dag.mode == mode, 'wrong planned input')
            with path.open('rb') as f:
                header = digest(next(f))
            run = runs['arms'][arm]
            need(set(run) == {'0','1'}, 'missing production control')
            for flag in ('0', '1'):
                rows, fields = transcript(out/arm/f'trace-{flag}.log.gz',header,flag)
                need(run[flag] == dict(exit=1, proposal=False, audit=False, **fields), 'wrong production receipt')
            need(len(original) == len(rows) == 11, 'incomplete arm')
            attempts = []
            for d, r, ref in zip(range(36,47), rows, original):
                attempts.append(verify_sample(r,ref,dag,d))
                seen[mode].append((a,k,d))
            if mode == 'naive':
                need(sha(path) == prior_arm['selected_sha256'] and
                     [x['b0_bits'] for x in attempts] == [x['builder_lower'] for x in prior_arm['attempts']],
                     'not the prior 11 bounds')
            if mode == 'naive' and (a,k) == (1,0):
                report['tamper_rejections'] = tamper_tests(rows[0],original[0],dag)
            report['arms'][arm] = dict(selected_sha256=sha(path), prior_trace_sha256=prior_arm['trace_sha256'],
                                      trace_sha256=digest(b''.join(map(packed,rows))), edges=len(dag.edges),
                                      stages=dag.stages, attempts=attempts)
        naive, affine = [report['arms'][f'{mode}-a{a}-k{k}'] for mode in ('naive','affine')]
        def widths(arm):
            return [hi-lo for s in arm['stages'] if 'input_interval' in s for lo,hi in [s['input_interval']]]
        need(any(y < x for x,y in zip(widths(naive),widths(affine))), 'no actual affine domain tightening')
        for x,y in zip(naive['attempts'],affine['attempts']):
            delta.append(y['b0_bits']-x['b0_bits'])
            log_delta.append(y['magnitude_bits']-x['magnitude_bits'])
    for attempts in seen.values():
        frozen.frozen_inventory(attempts)
    report['affine_minus_naive'] = dict(paired_candidates=len(delta), b0_bits_range=[min(delta),max(delta)],
                                        log2_magnitude_bits_range=[min(log_delta),max(log_delta)])
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--opt', type=Path)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--out', type=Path); m.add_argument('--replay', type=Path)
    a = ap.parse_args()
    if bool(a.out) != bool(a.opt):
        ap.error('--opt required only with --out')
    prior, out = a.prior.resolve(strict=True), (a.out or a.replay).resolve()
    need(sha(prior/'diagnosis.json') == PRIOR, 'not the released A2-a 44-candidate diagnosis')
    old = frozen.replay(prior)
    need(old == json.loads((prior/'diagnosis.json').read_bytes()), 'prior did not independently replay')
    if a.out:
        produce(a.opt.resolve(strict=True),prior,out)
    result = replay(prior,out,old)
    report = out/'attribution.json'
    if a.out:
        report.write_bytes(packed(result))
    else:
        need(report.read_bytes() == packed(result), 'stale attribution report')
    print('A11_DIAGNOSTIC_PASS naive=44 affine=44 trace_off_denials=8 proposals=0 highs_called=0',
          'sha256='+sha(report))


if __name__ == '__main__':
    main()
