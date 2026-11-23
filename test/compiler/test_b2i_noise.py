"""Offline two-graph B2-i regression. Pass --native to generate reports or replay an existing --reports directory."""
import argparse
import copy
from fractions import Fraction as Q
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
from replay_noise_audit import binding_plan, independent, load, need, same
from test_exact_scale_source import replace_encoding
from diagonal_slot_probe import check_slots

CASES = {'residual': (59, 6, 4, -10186237938), 'single-gelu': (62, 7, 3, -32058679686)}
DC_SITES = {f'node/{node}/plain_encoding_at_output' for node in ('poly0.n2.pt0','poly0.n2.constant','poly0.n4.pt0')}
DIAGONALS = {'residual': {'lin0.mask0','lin0.mask6','lin0.mask8','lin1.mask0','lin1.mask7','lin1.mask9'},
             'single-gelu': {f'lin{i}.mask{j}' for i in range(2) for j in range(4)}}


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

    def replay(proposal=None, *, checked=ir, params=program, basis=d/'basis.json', witness=d/'source.mlir', guard=True):
        return independent(checked, params, proposal, baseline, b2i=True, source_witness=witness, basis=basis, guarded_diagonals=guard)
    result = replay(report); before, adds, drops, c = CASES[name]
    masks = DIAGONALS[name]; period = 64 if name == 'residual' else 4
    released = DC_SITES | {f'node/{node}/plain_encoding_at_output' for node in masks}
    need(result['missing_before'] == before and result['missing'] == before-len(released) and
         set(result['cleared']) == released, 'wrong local reduction')
    remaining = result['missing_contract_terms']
    need(sum(x.endswith('/add') for x in remaining) == adds and sum(x.endswith('/level_reduce') for x in remaining) == drops
         and {'node/poly0.n2.pt0/mul_plain', 'node/poly0.n2.constant/add_plain', 'node/poly0.n4.pt0/mul_plain'} <= set(remaining),
         'waived centered ciphertext lift')
    fresh = load(report); denied = load(denial)
    need(fresh['mode'] == 'engineering' and same(denied, dict(fresh, mode='probability')) and
         replay(denial)['matched'], 'denial not replayable')
    def transfer(x, site): return next(t for t in x['b2i_local']['transfers'] if t['site_id'] == site)
    def proved(x): return transfer(x, 'node/poly0.n2.pt0/plain_encoding_at_output')
    need(proved(fresh)['encoded_constant'] == str(c) and proved(fresh)['lhs_gain'] == str(c), 'signed gain')
    wide = 'node/poly0.n4.pt0/plain_encoding_at_output'
    for site in DC_SITES - {proved(fresh)['site_id']}:
        row = transfer(fresh, site)
        need(row['formula_ref'] == 'b2i/dc-word-bounded-ntt-v6.2.0' and
             int(row['encoded_constant']) < int(row['noncanonical_ntt_word_upper']) < 2**64, 'unproved CRT word bound')
    for node in masks:
        row = transfer(fresh, f'node/{node}/plain_encoding_at_output'); v = row['periodic_diagonal']
        op = transfer(fresh, f'node/{node}/mul_plain')
        need(row['formula_ref'] == 'b2i/guarded-periodic-real-diagonal-v1' and v['period'] == period and
             v['coefficient_stride']*2*period == 16384 and 2*int(v['centered_coeff_abs']) < int(row['active_q_product']) and
             Q(v['slot_delta_inf']) > 0 and row['delta_pt_signed'] is None and row['lhs_gain'] is None and
             op['site_id'] in remaining and op['blocker'] == 'need_tighter_centered_decryption_coefficients' and
             2*int(op['candidate_coeff_abs']) >= int(op['active_q_product']), 'false signed gain or ciphertext lift')
    negatives = []
    def reject(label, fn, message):
        try: fn()
        except ValueError as exc:
            need(message in str(exc), f'{label}: wrong refusal: {exc}'); negatives.append(label); return
        raise AssertionError('accepted '+label)
    need(replay(guard=False)['missing'] == before-len(DC_SITES), 'retrofitted historical target')
    reject('guardless-target', lambda: replay(report, guard=False), 'B2-i production audit differs')
    for field, value in [('lhs_gain',str(-c)), ('delta_pt_signed','0'), ('local_bias_units','0'),
                         ('argument_sha256','0'*64), ('active_q_product','97')]:
        bad = copy.deepcopy(fresh); proved(bad)[field] = value
        path = tmp/'forged.json'; path.write_text(json.dumps(bad))
        reject(field, lambda: replay(path), 'source-derived B2-i production audit differs')
    for site, field, value in [(wide, 'noncanonical_ntt_word_upper', '0'),
                               ('node/poly0.n4.pt0/mul_plain', 'dc_center_correction', '0')]:
        bad = copy.deepcopy(fresh); transfer(bad, site)[field] = value
        path.write_text(json.dumps(bad)); reject(field, lambda: replay(path), 'B2-i production audit differs')
    maskrow = 'node/lin0.mask0/plain_encoding_at_output'
    for field, val in [('period',2), ('coefficient_stride',1), ('centered_coeff_abs','0'), ('slot_delta_inf','0'),
                       ('requires_backend','epochs.RequireExactEncoding')]:
        bad = copy.deepcopy(fresh); transfer(bad,maskrow)['periodic_diagonal'][field] = val
        path.write_text(json.dumps(bad)); reject('diagonal/'+field, lambda: replay(path), 'B2-i production audit differs')
    bad = copy.deepcopy(fresh); del transfer(bad,maskrow)['periodic_diagonal']
    path.write_text(json.dumps(bad)); reject('missing-diagonal-binding', lambda: replay(path), 'B2-i production audit differs')
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
    for node, offset in [('poly0.n2.pt0', 0), ('lin0.mask0', 17)]:
        p = load(program); val = next(e for e in p['instructions'] if e['ID']==node)['values']
        val[offset] = math.nextafter(val[offset], math.inf); p['encoding_receipts'] = binding_plan(p)
        path.write_text(json.dumps(p)); changed = tmp/'tampered.mlir'
        changed.write_text(replace_encoding(ir.read_text(), p['encoding_receipts']))
        reject('synchronized-public-bit-and-receipts/'+node, lambda: replay(checked=changed, params=path), 'source-derived instruction')

    if go:
        run(native, ['--snabf-lower-ckks=direct-go=true', ir, '-o', 'lower.mlir'], tmp)
        run(native, ['--snabf-bind-lattigo=ring-go=model.go audit=program.json', 'lower.mlir', '-o', os.devnull], tmp)
        need(same(load(tmp/'program.json'), load(program)), 'physical program changed on re-emission')
        need('}, evaluate, epoch.RequireDiagonalEncodingV1)' in (tmp/'model.go').read_text(), 'legacy unguarded model')
        probe = tmp/'b2i_plain_test.go'; shutil.copyfile(Path(__file__).parent/'testdata'/probe.name, probe)
        env = dict(os.environ, GOPROXY='off', GOSUMDB='off', GOTOOLCHAIN='local', GOVCS='*:off', GOWORK='off',
                   GOTELEMETRY='off', B2I_AUDIT=str(report), B2I_PUBLIC_POLYS=str(tmp/'public-polys.json'))
        subprocess.run([go, 'test', '-mod=readonly', '-count=1', '-run', '^TestB2IActualBoundPlaintext$', '-v',
                        tmp/'model.go', probe], cwd=ROOT, env=env, check=True)
        old = tmp/'old.go'; old.write_text((tmp/'model.go').read_text().replace('epoch.RequireDiagonalEncodingV1)', 'epoch.RequireExactEncoding)'))
        down = subprocess.run([go,'test','-mod=readonly','-count=1','-run','^TestB2IActualBoundPlaintext$',old,probe],
                              cwd=ROOT, env=env, capture_output=True, text=True)
        need(down.returncode != 0 and 'model did not enforce the real root guard' in down.stdout, 'accepted a downgraded runtime')
        negatives.append('compiled-old-policy-with-same-seal')
        polys = load(tmp/'public-polys.json'); need(set(polys) == masks, 'missing actual diagonal')
        edges = {e['node_id']:e for e in fresh['edges']}; args = {x['ID']:x for x in load(program)['instructions']}
        for node in sorted(masks):
            v = transfer(fresh, f'node/{node}/plain_encoding_at_output')['periodic_diagonal']
            words, vals, e, bound = polys[node], args[node]['values'], Q(edges[node]['effective_encoder_scale']), Q(v['slot_delta_inf'])
            need(len(words) == 2*period, 'wrong actual polynomial support')
            check_slots(words, vals, e, bound)
            if node == 'lin0.mask0':
                wrong = words.copy(); wrong[0] += 1+math.ceil(2*bound)
                reject('altered-actual-public-polynomial', lambda: check_slots(wrong, vals, e, bound), 'exceeds slot error')
    print(json.dumps(dict(graph=name, pre_b2i=before, previous=before-len(DC_SITES), diagonal_cleared=sorted(masks), remaining=len(remaining),
                          denial_exit=code, negative_cases=negatives, total_delta_valid=fresh['total_delta_valid'],
                          certified_bound=fresh['sinks'][0]['certified_bound_units'], issued=False)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ('fixtures','reports','native','go'):
        ap.add_argument('--'+flag, type=Path, required=flag in ('fixtures','reports'))
    args = ap.parse_args(); need(not args.go or args.native, 'runtime probe needs --native')
    reports = args.reports.resolve()
    if args.native: reports.mkdir(parents=True, exist_ok=False)  # never overwrite evidence
    with tempfile.TemporaryDirectory(prefix='snabf-b2i-check-') as t:
        for name in CASES:
            case(name, args.fixtures.resolve(), reports, Path(t)/name,
                 args.native.resolve() if args.native else None, args.go.resolve() if args.go else None)
