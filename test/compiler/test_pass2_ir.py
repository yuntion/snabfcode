"""Black-box source provenance, fixed point, sink feedback and tamper regression."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def ensure(ok, message):
    if not ok:
        raise AssertionError(message)


def transform(text, pattern, new):
    changed, n = re.subn(pattern, lambda _: new, text, count=1)
    ensure(n == 1 and changed != text, f'test mutation did not apply: {pattern[:100]}')
    return changed


def op_line(text, op):
    lines = [line for line in text.splitlines() if f'"snabf.{op}"(' in line]
    ensure(len(lines) == 1, f'expected one {op}, found {len(lines)}')
    return lines[0]


def check_work(opt, heir, folder):
    def invoke(program, text, *args, reject=None):
        p = subprocess.run([*program, *args], input=text, text=True, capture_output=True)
        if reject is None:
            ensure(p.returncode == 0, p.stderr[:1800])
            return p.stdout
        ensure(p.returncode > 0 and reject in p.stderr, f'expected diagnostic {reject!r}; got exit={p.returncode}: {p.stderr[:1000]}')

    def native(text, *args, **kw):
        return invoke([opt, '--allow-unregistered-dialect', '--verify-each'], text, *args, **kw)

    source = (folder / 'input.mlir').read_text()
    invalid = transform(source, r'math\.exp %neg : f32', 'math.exp %in : f32')
    generic = invoke([heir], invalid, '--mlir-print-op-generic', '--mlir-print-local-scope')
    native(generic, '--snabf-mark-activations', reject='no source')
    marked = (folder / 'marked.mlir').read_text()
    ensure('snabf.activation_sources' in marked and 'semantics = "silu"' in marked, 'missing source semantics')
    ensure('snabf.activation_sources' in (folder / 'original-ckks.mlir').read_text(), 'HEIR lost source marker')
    imported = (folder / 'imported.mlir').read_text()
    ensure('activation = "silu"' in op_line(imported, 'poly_eval'), 'lost marked activation')
    # Do not rely on the presence of the old backend sigmoid as proof of its semantics.
    unmarked = transform(imported, r'snabf\.activation_sources = \[\{[^}]+\}\], ?', '')
    native(unmarked, '--snabf-pass2-candidates', reject='source semantics')

    witness = json.loads((folder / 'budget-search.json').read_bytes())['precision_witness']
    loose = (folder / 'economy/selected.mlir').read_text()
    tight = (folder / witness / 'selected.mlir').read_text()
    for name, text in [('economy', loose), (witness, tight)]:
        ensure(native(text, '--snabf-check-plan') == text, f'{name}: check changed selected IR')
        ensure(native(text, '--snabf-plan', '--snabf-check-plan') == text, f'{name}: plan not idempotent')

    sink = op_line(loose, 'sink')
    tight_sink = op_line(tight, 'sink')
    field = r'max_approx_error = [^ ,}]+ : f64'
    match = re.search(field, tight_sink)
    ensure(match, 'missing tight precision')
    changed = loose.replace(sink, transform(sink, field, match[0]))
    native(changed, '--snabf-check-plan', reject='violates sink contract')
    ensure(native(changed, '--snabf-plan', '--snabf-check-plan') == tight,
           'single edited sink did not reproduce fresh precision build')
    impossible = loose.replace(sink, transform(sink, r'min_output_level = \d+ : i64', 'min_output_level = 99 : i64'))
    native(impossible, '--snabf-plan', reject='no joint SiLU/route plan')

    poly = op_line(loose, 'poly_eval')
    frontier, selected = poly.split('poly_selected = ', 1)
    stale_bound = transform(frontier, r'activation_error = [^ ,}]+ : f64', 'activation_error = 1.0 : f64')
    stale_dag = transform(selected, r'weights = array<f64: [^,>]+', 'weights = array<f64: 1.234567')
    for damaged, diagnostic in [(stale_bound + 'poly_selected = ' + selected, 'stale Pass 2 frontier'),
                                (frontier + 'poly_selected = ' + stale_dag, 'stale selected full DAG')]:
        for phase in ['--snabf-check-plan', '--snabf-lower-ckks']:
            native(loose.replace(poly, damaged), phase, reject=diagnostic)
    facts = {'source_provenance': True, 'plan_idempotence': ['economy', witness],
             'precision_feedback_matches_fresh': True, 'impossible_level_rejected': True,
             'tamper_rejected_by': ['snabf-check-plan', 'snabf-lower-ckks'],
             'tampered_fields': ['poly_frontier.activation_error', 'poly_selected.nodes.weights'],
             'replanned_sha256': hashlib.sha256(tight.encode()).hexdigest()}
    if folder.name == 'residual':
        for text in [imported, loose, tight]:
            step = op_line(text, 'ckks_step')
            m = re.search(r'"snabf\.ckks_step"\((%\w+), (%\w+)\).*?kind = "ckks.add"', step)
            ensure(m and m[1] != m[2], 'residual must join distinct live SSA operands')
        for name in ('heir', 'economy', witness):
            ensure('.Add(' in (folder / name / 'model.go').read_text(), 'missing emitted ciphertext add')
        facts['residual_distinct_ssa_add'] = True
    return facts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--snabf-opt', type=Path)
    p.add_argument('--heir-bin', type=Path)
    p.add_argument('--workloads', nargs='+', default=['control', 'strided64', 'reuse64', 'residual'])
    p.add_argument('--report', type=Path, help='alternate report path when testing an existing partial run')
    a = p.parse_args()
    opt = a.snabf_opt or a.root / 'build/native/ir/snabf-opt'
    heir = (a.heir_bin or a.root / 'extern/heir-release/heir') / 'heir-opt'
    facts = {}
    for work in a.workloads:
        facts[work] = check_work(opt, heir, a.out / work)
        print(f'{work}: source, check, fixed point, precision-only feedback, infeasibility and certificate/DAG tamper PASS', flush=True)
    (a.report or a.out / 'ir-tests.json').write_text(json.dumps(facts, indent=2) + '\n')


if __name__ == '__main__':
    main()
