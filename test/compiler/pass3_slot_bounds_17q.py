"""Compare the real 17Q pre-solve guard on ONE existing server-resident A2-a capture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from a11_magnitude_replay import read_selected
from phase2_17q_probe import native_args


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opt', type=Path, required=True)
    ap.add_argument('--selected', type=Path, required=True)
    ap.add_argument('--alpha', type=int, choices=(1, 2), required=True)
    ap.add_argument('--max-k', type=int, choices=(0, 2), required=True)
    ap.add_argument('--out', type=Path, required=True, help='fresh private output directory')
    args = ap.parse_args()
    selected = args.selected.resolve(strict=True)
    opt = args.opt.resolve(strict=True)
    # Independently read the source-bound binary32 rows, not a compiler audit.
    dag = read_selected(selected)
    first = dag.stages[0]
    need(first['sum_diagonal_max_abs'] == 32.7568359375 and first['max_slot_sum_abs'] == 1.,
         'not the documented 17Q table')
    frozen = native_args(args.alpha, args.max_k)
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    if dag.mode != 'affine':
        # A different arm can have the same first-layer norm and topology.
        # Bind the actual input bytes and parameter options to the original tap.
        receipt = json.loads(selected.with_name('call.json').read_text())
        need(dag.mode == 'naive' and receipt['inputs']['selected.mlir'] == digest and receipt['argv'] == frozen,
             'selected.mlir or requested arm differs from the frozen tap receipt')
    args.out.mkdir(parents=True, exist_ok=False)
    report = {'selected_sha256': digest,
              'interval_mode': dag.mode, 'alpha': args.alpha, 'max_k': args.max_k, 'arms': {}}
    for mode in ('legacy', 'co-slot-v1'):
        out = args.out/mode
        out.mkdir()
        argv = frozen.copy()
        argv[argv.index('selected.mlir')] = str(selected)
        i = next(i for i, arg in enumerate(argv) if arg.startswith('--snabf-pass3='))
        argv[i] += f' magnitude-mode={mode} magnitude-audit=bounds.json'
        with (out/'solver.log').open('wb') as log:
            p = subprocess.run([opt, *argv], cwd=out, stdout=log, stderr=subprocess.STDOUT)
        solved = p.returncode == 0
        if not solved:
            need(p.returncode == 1 and 'Pass 3 infeasible under requested scale, precision and QP bounds'
                 in (out/'solver.log').read_text(), 'unexpected failure (see solver.log)')
        need((out/'parameters.json').exists() == solved, 'proposal and solver status differ')
        audit = json.loads((out/'bounds.json').read_text())
        need(audit['schema'] == 'snabf.pass3.ideal-magnitude.v1' and audit['min_scale'] == 36 and
             audit['max_scale'] == 46 and audit['magnitude_mode'] == mode and audit['interval_mode'] == dag.mode,
             'wrong audit mode or exponent range')
        edges = audit['edges']
        need([e['id'] for e in edges] == [e['id'] for e in dag.edges], 'expanded SSA topology changed')
        lin0 = next(e['magnitude'] for e in edges if e['id'] == 'lin0.rescale')
        lin3 = edges[-1]
        need(lin3['id'] == 'lin3.rescale' and lin3['level'] == 0, 'not the final 17Q rescale')
        terms, lower = lin3['base_width_term_bits'], audit['base_width_lower_bits']
        if mode == 'legacy':
            need(not solved and lin0 == .5*first['sum_diagonal_max_abs'] and
                 lower == terms == [d+256.75 for d in range(36,47)], 'missing frozen base-width rejection')
        else:
            need(lin0 == .5*first['max_slot_sum_abs'] and len(terms) == len(lower) == 11 and
                 all(t <= b < 59. for t,b in zip(terms,lower)), '17Q base-width obstruction not cleared')
        report['arms'][mode] = dict(first_decoded_abs=lin0, final_decoded_abs=lin3['magnitude'],
                                    final_guard_bits=terms, base_lower_bits=lower, parameter_proposal=solved)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
