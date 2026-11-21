"""Offline periodic-boundary rejection gate. Use LF fixtures from git archive.

Run with --opt NEW --baseline-opt BASE_5c301ff --out EMPTY_DIRECTORY.
Exercises the real planner/checker/lowering on screened R3 checkpoints. No
source compilation, ciphertext execution, or full-slot admission is claimed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess


FIXTURES = Path(__file__).resolve().parents[2] / 'doc/e2e-matrix-2026-11-20-r3/frozen'
PINS = {
    'cross_residual': ('44ea9571af3f9f507ffa2f6c5191113ed4a94c7d68ffd53315f2f6e54818bfd6',
                       '1a921108ddfe0d87cd25078d06ccccbc5418acd0930cbe6b652fd18ba28c17a8'),
    'reuse64': ('96e2cd0005ea6b41dc41f022eabcdfe42891724289f307aa372aeda78baf6662',
                '9e9f827151e8b4175641ad97791553afde1e2221fd055ae621dbe1d22165df6a'),
}
GATES = ('--snabf-check-plan', '--snabf-lower-ckks=direct-go=true')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(x, y):
    require(math.isclose(x, y, rel_tol=2e-12, abs_tol=2e-18), f'{x} != {y}')


def replay(proposal, unreserved):
    f = proposal['cut_margin_feedback']
    (r,) = f['rounds']
    require(f['proposal'] == proposal['fingerprint'] and not f['authorizes_conversion'] and
            f['round_limit'] == 4 and r['basis_origin'] == 'screened_checkpoint' and
            r['basis'] == {k: unreserved[k] for k in ('p', 'q', 'scale')}, 'different checkpoint/proposal')
    ports = [(v['edge_id'], u['user_index'], u['operand'], u['role']) for v in f['bindings'] for u in v['uses']]
    require(ports == [('poly0.n6.add', 10, 0, 'main'), ('lin0.rescale', 13, 1, 'bypass')], 'wrong SSA uses')
    eps = f['epsilon']
    close(eps, 2**-12)
    require(len(r['terms']) == len(ports) == len({t['down'] for t in r['terms']}), 'incomplete cut')
    # Evaluate forward in three coordinates (suffix intercept, main, bypass).
    # Production derives the joint row by reverse substitution.
    downs = {t['down']: (j, t) for j, t in enumerate(r['terms'])}
    floors, affine = [], []
    for i, row in enumerate(r['rows']):
        local = eps * (row['fixed_over_epsilon'] + min(row.get('increments_over_epsilon', [0])))
        inputs = list(zip(row['parents'], row['gains'], strict=True))
        require(all(0 <= k < i and w >= 0 for k, w in inputs), 'invalid predecessor or gain')
        floors.append(local + sum(w*floors[k] for k, w in inputs))
        if i not in downs:
            v = [local, 0., 0.]
            for k, w in inputs:
                for j in range(3):
                    v[j] += affine[k][j]*w
        else:
            j, term = downs[i]
            port = f['bindings'][j]
            require(term['binding'] == j and row['op'] == 'ring_down' and row['ks'] == [0] and
                    term['source_edge'] == port['edge_id'], 'unbound Down')
            (a, gain), = inputs
            if a != term['producer']:
                align = r['rows'][a]
                require(align['op'] == 'drop' and align['parents'] == [term['producer']] and
                        align['gains'] == [1.] and floors[a] == floors[term['producer']], 'nontrivial alignment')
            # Independent one-P Down arithmetic from the original integer basis.
            n = port['ring_dim']
            key = 10*n*n*sum(q+1 for q in unreserved['q'][:proposal['level']+1])/unreserved['p'][0]
            rounding = .5*n*(1+min(n, 2*math.sqrt(n*math.log(4*n*2**40))))
            close(local, (key+rounding)/2**port['physical_log_scale'])
            v = [local, 0., 0.]
            v[j+1] = gain
        affine.append(v)
    suffix, *weights = affine[f['sink']]
    close(suffix, r['suffix_floor'])
    demand = 0.
    for t, w in zip(r['terms'], weights, strict=True):
        close(w, t['gain'])
        close(floors[t['producer']], t['input_floor'])
        demand += w*floors[t['producer']]
    budget = eps-f['approximation']-suffix
    close(demand, r['joint_demand']); close(budget, r['joint_budget'])
    close(floors[f['sink']], r['sink_floor']); close(suffix+demand, r['sink_floor'])
    close(demand-budget, r['signed_deficit'])
    require(demand > budget > 0, 'no transferable deficit')
    return dict(level=proposal['level'], rows=len(floors), use_points=ports, fingerprint=r['fingerprint'],
                joint_budget=budget, demand=demand, signed_deficit=demand-budget, stop=f['stop'],
                source_search=r.get('source_search'), rounds=len(f['rounds']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('opt', 'baseline-opt', 'out'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--fixtures', type=Path, default=FIXTURES)
    a = p.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    require(not any(out.iterdir()), 'use a new evidence directory')
    facts = dict(schema='snabf.periodic-cut-margin-gate.v1', binary_sha256=sha(a.opt),
                 baseline_binary_sha256=sha(a.baseline_opt), fixtures=PINS, legacy={}, negative=[])

    def run(label, fixture, stage, src=None, opt=None, expected=None, overrides=None):
        root = a.fixtures/fixture
        env = dict(os.environ, SNABF_PASS4_ESTIMATES=str(root/'estimates.json'))
        env.pop('SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS', None)
        env.update(overrides or {})
        dst = out/(label+'.mlir')
        args = [opt or a.opt, '--allow-unregistered-dialect', '--verify-each', '--mlir-print-op-on-diagnostic=false',
                *([stage] if stage else []), src or root/'checkpoint.mlir', '-o', dst]
        r = subprocess.run(args, env=env, capture_output=True, text=True)
        why = next((line for line in r.stderr.splitlines() if 'error:' in line), r.stderr[:500])
        if expected:
            require(r.returncode > 0 and expected in why and not dst.exists(), label+': '+why)
            facts['negative'].append(dict(case=label, reason=expected, returncode=r.returncode))
        else:
            require(r.returncode == 0 and dst.is_file(), label+': '+why)
        return dst

    def plan(label, fixture, profile, **kw):
        audit = out/(label+'.json')
        path = run(label, fixture, f'--snabf-pass4=cut-profile={profile} audit={audit}', **kw)
        return path, json.loads(audit.read_text())

    cross, control = 'cross_residual', 'reuse64'
    for fx, pins in PINS.items():
        for name, pin in zip(('checkpoint.mlir', 'estimates.json'), pins):
            require(sha(a.fixtures/fx/name) == pin, 'not pinned LF fixture: '+fx+'/'+name)
        for profile in ('single-live-v1', 'dual-live-v1'):
            label = fx+'-'+profile
            old, _ = plan(label+'-base', fx, profile, opt=a.baseline_opt)
            new, j = plan(label, fx, profile)
            require(old.read_bytes() == new.read_bytes() and sha(old.with_suffix('.json')) == sha(new.with_suffix('.json')),
                    'legacy bytes changed: '+label)
            facts['legacy'][label] = dict(mlir_sha256=sha(new), audit_sha256=sha(new.with_suffix('.json')), outcome=j['outcome'])

    opts = json.loads((a.fixtures/cross/'pass3-options.json').read_text())
    audit = out/'unreserved-source.json'
    run('unreserved-source', cross, f'--snabf-pass3=audit={audit} source-arithmetic-fraction=1')
    source = json.loads(audit.read_text())
    require(source['options'] == opts and source['source_arithmetic_fraction'] == source['requested_source_arithmetic_fraction'] == 1,
            'not the original unreserved source search')
    profile = 'dual-live-feedback-v1'
    sealed, j = plan('feedback-cross', cross, profile)
    require(j['outcome'] == 'retain_source_ring' and not j.get('conversion') and
            [(r['target'],r['level'],r['reason']) for r in j['rounds']] ==
            [(8192,2,'boundary_milp_infeasible'),(8192,1,'boundary_milp_infeasible'),
             (4096,2,'period_mismatch'),(4096,1,'period_mismatch')], 'wrong boundary search')
    receipts = [replay(r, source) for r in j['rounds'] if 'cut_margin_feedback' in r]
    facts['projections'] = receipts
    l2, l1 = receipts
    require(l2['stop'] == 'no_source_candidate_under_original_limits' and l2['source_search']['solver_runs'] > 0 and
            l2['source_search']['models_with_cut_rows'] > 0 and
            l1['stop'] == 'switching_key_ext_modulus_estimate_unavailable' and l1['source_search'] is None,
            'wrong feedback ownership or screen')
    mixed, pos = plan('feedback-reuse', control, profile)
    require(pos['outcome'] == 'ring_down' and pos['target'] == 8192, 'no real positive control')
    facts['positive_control'] = dict(graph=control, target=pos['target'], outcome=pos['outcome'])
    sealed = run('checked-cross', cross, GATES[0], sealed)
    mixed = run('checked-reuse', control, GATES[0], mixed)
    run('lowered-cross', cross, GATES[1], sealed); run('lowered-reuse', control, GATES[1], mixed)
    run('cross-force', cross, '--snabf-pass4=cut-profile='+profile+' force-down=true', expected='SNABF_FORCE_DOWN_REJECTED')
    hard = {'SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS':'512'}
    _, hard_cross = plan('hard-cross', cross, profile, overrides=hard)
    require(all('source_search' not in v for r in hard_cross['rounds'] if 'cut_margin_feedback' in r
                for v in r['cut_margin_feedback']['rounds']), 'source tried to repair a rejected security instance')
    _, hard_reuse = plan('hard-reuse', control, profile, overrides=hard)
    require(hard_reuse['outcome'] == 'retain_source_ring', '512-bit screen bypassed')
    run('hard-force', control, '--snabf-pass4=cut-profile='+profile+' force-down=true', overrides=hard,
        expected='SNABF_FORCE_DOWN_REJECTED')
    for k, gate in enumerate(GATES):
        run(f'hard-sealed-{k}', control, gate, mixed, overrides=hard, expected='stale_or_weakened_pass4_policy')
        run(f'missing-evidence-{k}', cross, gate, sealed, overrides={'SNABF_PASS4_ESTIMATES':str(out/'absent.json')},
            expected='missing_external_estimator_evidence')
    # Each edit is syntactically valid and confined to the derived receipt.
    text = sealed.read_text(); at = text.index('cut_margin_feedback = ')
    edits = [('budget', r'(joint_budget = )[-+\w.]+', r'\g<1>0.000000e+00'),
             ('fingerprint', r'(fingerprint = ")[a-f0-9]{64}"', r'\g<1>'+'0'*64+'"'),
             ('use', r'role = "bypass"', 'role = "main"'),
             ('count', r'(solver_runs = )\d+', r'\g<1>0')]
    for name, pat, sub in edits:
        tail, n = re.subn(pat, sub, text[at:], count=1)
        require(n == 1 and tail != text[at:], 'missing tamper target: '+name)
        bad = out/('edited-'+name+'.mlir'); bad.write_text(text[:at]+tail)
        run('parse-'+name, cross, None, bad)
        for k, gate in enumerate(GATES):
            run(f'{name}-{k}', cross, gate, bad, expected='stale_or_forged_ring_search')
    (out/'acceptance.json').write_text(json.dumps(facts, indent=2)+'\n')
    print(json.dumps(facts, indent=2))


if __name__ == '__main__':
    main()
