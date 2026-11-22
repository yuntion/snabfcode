"""Offline two-graph B2-i regression. Pass --native to generate reports or replay an existing --reports directory."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/tools'))
from b2i_noise_replay import centered_linear, constant
from replay_noise_audit import binding_plan, independent, load, need, same
from test_exact_scale_source import replace_encoding

CASES = {'residual': (59, 6, 4, -10186237938), 'single-gelu': (62, 7, 3, -32058679686)}


def run(native, args, cwd, deny=False):
    env = os.environ.copy(); env.pop('SNABF_PASS4_ESTIMATES', None)
    proc = subprocess.run([native, '--allow-unregistered-dialect', '--verify-each', *args], cwd=cwd, env=env,
                          capture_output=True, text=True)
    need((proc.returncode != 0 and 'noise certificate denied' in proc.stderr) if deny else proc.returncode == 0,
         'unexpected native outcome: ' + proc.stderr[:700])
    return proc.returncode


def case(name, inputs, reports, tmp, native, go):
    tmp.mkdir(); d = inputs/name; ir = d/'checked.mlir'; program = d/'program.json'; baseline = d/'audit.json'
    report = reports/(name+'.json'); denial = reports/(name+'-denied.json')
    if native:
        def audit(path, flags='', deny=False):
            return run(native, [f'--snabf-check-plan=noise-audit={path}'+flags, ir, '-o', os.devnull], tmp, deny)
        audit('default.json')
        need((tmp/'default.json').read_bytes() == baseline.read_bytes(), 'legacy audit byte drift')
        audit('new.json', ' b2i-local=true'); shutil.copyfile(tmp/'new.json', report)
        code = audit('denied.json', ' b2i-local=true require-noise-cert=true', True)
        shutil.copyfile(tmp/'denied.json', denial)
    else:
        code = None

    def replay(proposal=None, *, checked=ir, params=program, basis=d/'basis.json', witness=d/'source.mlir'):
        return independent(checked, params, proposal, baseline, b2i=True, source_witness=witness, basis=basis)
    result = replay(report); before, adds, drops, c = CASES[name]
    need(result['missing_before'] == before and result['missing'] == before-1 and
         result['cleared'] == ['node/poly0.n2.pt0/plain_encoding_at_output'], 'wrong local reduction')
    remaining = result['missing_contract_terms']
    need(sum(x.endswith('/add') for x in remaining) == adds and sum(x.endswith('/level_reduce') for x in remaining) == drops
         and 'node/poly0.n2.pt0/mul_plain' in remaining, 'waived centered ciphertext lift')
    fresh = load(report); denied = load(denial)
    need(fresh['mode'] == 'engineering' and same(denied, dict(fresh, mode='probability')) and
         replay(denial)['matched'], 'denial not replayable')
    def proved(x): return next(t for t in x['b2i_local']['transfers'] if t['formula_ref'])
    need(proved(fresh)['encoded_constant'] == str(c) and proved(fresh)['lhs_gain'] == str(c), 'signed gain')
    negatives = []
    def reject(label, fn, message):
        try: fn()
        except ValueError as exc:
            need(message in str(exc), f'{label}: wrong refusal: {exc}'); negatives.append(label); return
        raise AssertionError('accepted '+label)
    for field, value in [('lhs_gain',str(-c)), ('delta_pt_signed','0'), ('local_bias_units','0'),
                         ('argument_sha256','0'*64), ('active_q_product','97')]:
        bad = copy.deepcopy(fresh); proved(bad)[field] = value
        path = tmp/'forged.json'; path.write_text(json.dumps(bad))
        reject(field, lambda: replay(path), 'source-derived B2-i production audit differs')
    bad = copy.deepcopy(fresh); site = 'node/poly0.n2.pt0/mul_plain'
    bad['missing_contract_terms'].remove(site)
    next(t for t in bad['sites'] if t['site_id']==site)['formula_ref'] = 'claimed-no-wrap'
    next(t for t in bad['b2i_local']['transfers'] if t['site_id']==site).update(formula_ref='claimed-no-wrap', blocker=None)
    path.write_text(json.dumps(bad)); reject('coordinated-wrap-waiver', lambda: replay(path), 'B2-i production audit differs')
    bad = copy.deepcopy(fresh); bad.update(common_event='independent', total_delta_valid=0, probabilistic_cert_issued=True)
    bad['sinks'][0].update(certified_bound_units=0, margin_passed=True)
    path.write_text(json.dumps(bad)); reject('invented-joint-law-and-margin', lambda: replay(path), 'B2-i production audit differs')
    q = load(d/'basis.json'); q['Q'][:2] = reversed(q['Q'][:2]); path.write_text(json.dumps(q))
    reject('reordered-basis', lambda: replay(basis=path), 'source/profile/domain mismatch')
    reject('no-original', lambda: replay(witness=None), 'needs both')
    # Resynchronize both legacy receipts with the changed value. The checked
    # activation DAG and independently bound public stage bits still disagree.
    p = load(program); val = next(e for e in p['instructions'] if e['ID']=='poly0.n2.pt0')['values']
    val[0] = math.nextafter(val[0], math.inf); p['encoding_receipts'] = binding_plan(p)
    path.write_text(json.dumps(p)); changed = tmp/'tampered.mlir'
    changed.write_text(replace_encoding(ir.read_text(), p['encoding_receipts']))
    reject('synchronized-public-bit-and-receipts', lambda: replay(checked=changed, params=path), 'source-derived instruction')

    if go:
        run(native, ['--snabf-lower-ckks=direct-go=true', ir, '-o', 'lower.mlir'], tmp)
        run(native, ['--snabf-bind-lattigo=ring-go=model.go audit=program.json', 'lower.mlir', '-o', os.devnull], tmp)
        need(same(load(tmp/'program.json'), load(program)), 'physical program changed on re-emission')
        probe = tmp/'b2i_plain_test.go'; shutil.copyfile(Path(__file__).parent/'testdata'/probe.name, probe)
        env = dict(os.environ, GOPROXY='off', GOSUMDB='off', GOTOOLCHAIN='local', GOVCS='*:off', GOWORK='off',
                   GOTELEMETRY='off', B2I_AUDIT=str(report))
        subprocess.run([go, 'test', '-mod=readonly', '-count=1', '-run', '^TestB2IActualBoundPlaintext$', '-v',
                        tmp/'model.go', probe], cwd=ROOT, env=env, check=True)
    print(json.dumps(dict(graph=name, before=before, cleared=result['cleared'], remaining=len(remaining),
                          denial_exit=code, negative_cases=negatives, total_delta_valid=fresh['total_delta_valid'],
                          certified_bound=fresh['sinks'][0]['certified_bound_units'], issued=False)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ('fixtures','reports','native','go'):
        ap.add_argument('--'+flag, type=Path, required=flag in ('fixtures','reports'))
    args = ap.parse_args(); need(not args.go or args.native, 'runtime probe needs --native')
    # Arithmetic-only counterexamples, not production bounds or encodings.
    need(centered_linear(97, ['u','v'], [1,1], {'u':48,'v':1}) == (49,False) and
         centered_linear(97, ['u','u'], [1,-1], {'u':48}) == (0,True), 'centered lift guard')
    from fractions import Fraction as Q
    for w, want in [(0.5,1), (-0.5,-1), (2.5,3), (math.nextafter(1.5, 0),1)]:
        need(constant([w], dict(branch='embed_double_f64',encoder_prec_bits=53,slots=16,realized_eff_scale='1'),
                      [193]) == (want,Q(want)-Q(w)), 'wrong fixed point rounding')
    reports = args.reports.resolve()
    if args.native: reports.mkdir(parents=True, exist_ok=False)  # never overwrite evidence
    with tempfile.TemporaryDirectory(prefix='snabf-b2i-check-') as t:
        for name in CASES:
            case(name, args.fixtures.resolve(), reports, Path(t)/name,
                 args.native.resolve() if args.native else None, args.go.resolve() if args.go else None)
