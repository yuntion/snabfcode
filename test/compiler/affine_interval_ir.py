"""Replay a pre-Pass-3 selected MLIR as a compatibility fixture, not a new graph.

Use a canonical selected.mlir from the C1 single/mixed or SiLU regression.
The opt-in arm replans from source-bound tables. It is not a ciphertext test.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--opt', type=Path, required=True)
    ap.add_argument('--reference-opt', type=Path, help='clean release baseline; compare its fresh plan byte for byte')
    ap.add_argument('--selected', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='new, private evidence directory')
    ap.add_argument('--require-tightened', action='store_true', help='require a strictly narrower DP input')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    decisions = {}

    def native(text, *flags, reject=None, binary=None):
        p = subprocess.run([binary or args.opt, '--allow-unregistered-dialect', '--verify-each',
                            '--mlir-print-op-on-diagnostic=false', *flags],
                           input=text, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if reject:
            check(p.returncode > 0 and reject.encode() in p.stderr,
                  f'expected {reject}, rc={p.returncode}: {p.stderr[:700]!r}')
        else:
            check(p.returncode == 0, p.stderr[:1600])
        return p.stdout

    check_plan, plan = '--snabf-check-plan', '--snabf-plan'
    selected = args.selected.read_bytes()
    canonical = native(selected)
    check(b'snabf.pass3 =' not in canonical and b'snabf.interval_mode' not in canonical,
          'supply a legacy pre-Pass-3 selected.mlir')
    # Compare complete canonical IR (candidate order, certificates, selected
    # DAG, routes, budgets), not just the winner template name.
    fresh = native(canonical, plan, check_plan)
    decisions['archive_fixed_point'] = fresh == canonical
    if args.reference_opt:
        ref = native(selected, plan, check_plan, binary=args.reference_opt)
        check(fresh == ref, 'default plan differs from clean reference compiler')
        decisions['reference_plan_sha256'] = hashlib.sha256(ref).hexdigest()
        canonical = fresh
    else:
        check(fresh == canonical, 'legacy plan changed')
    decisions['legacy_selected_sha256'] = hashlib.sha256(canonical).hexdigest()
    check(b'affine_slots =' not in canonical, 'opt-in metadata leaked into legacy mode')
    hdr = b'module attributes {'
    check(canonical.count(hdr) == 1, 'unexpected serialized module header')

    def mode(name):
        return canonical.replace(hdr, hdr + b'snabf.interval_mode = "' + name + b'", ', 1)

    explicit_naive = native(mode(b'naive'), plan, check_plan)
    check(explicit_naive.replace(b'snabf.interval_mode = "naive", ', b'') == canonical,
          'explicit naive changed the selected IR')
    affine = native(mode(b'affine'), plan, check_plan)
    check(native(affine, check_plan) == affine, 'affine replay changed the selected IR')
    check(b'affine_slots =' in affine and b'affine-logical-slots.v1' in affine, 'missing per-slot proof')

    def activation_inputs(text):
        result = []
        for line in text.splitlines():
            if b'"snabf.poly_eval"(' in line:
                ranges = set(re.findall(rb'input_interval = array<f64: ([^>]+)>', line))
                check(len(ranges) == 1, 'candidates disagree on the admitted input domain')
                result.append(tuple(float(t) for t in ranges.pop().split(b',')))
        check(bool(result), 'fixture contains no activation')
        return result

    old, new = activation_inputs(canonical), activation_inputs(affine)
    check(len(old) == len(new), 'activation count changed')
    decisions['legacy_and_affine_dp_inputs'] = list(zip(old, new))
    if args.require_tightened:
        check(any(hi-lo < b-a for (a,b),(lo,hi) in zip(old,new)), 'no tighter DP input domain')
    # Overwrite the saved per-slot lower endpoint, leaving the true inputs,
    # tables, activation budgets and Pareto frontier untouched.
    bad, n = re.subn(rb'(?<![\w])lo = array<f64: [^,>]+', b'lo = array<f64: -9.87654321e+02', affine, count=1)
    check(n == 1 and bad != affine, 'could not tamper saved slot projection')
    native(bad, check_plan, reject='stale budget')
    bad_domain, n = re.subn(rb'input_interval = array<f64: [^,>]+',
                            b'input_interval = array<f64: -9.87654321e+02', affine, count=1)
    check(n == 1 and bad_domain != affine, 'could not tamper candidate input domain')
    native(bad_domain, check_plan, reject='stale Pass 2 frontier')
    native(affine.replace(b'snabf.interval_mode = "affine"', b'snabf.interval_mode = "typo"'),
           check_plan, reject='invalid snabf.interval_mode')
    decisions['stale_slot_and_unknown_mode_rejected'] = True
    decisions['stale_candidate_domain_rejected'] = True
    decisions['affine_sha256'] = hashlib.sha256(affine).hexdigest()
    for name, data in [('legacy.mlir',canonical), ('affine.mlir',affine), ('tampered.mlir',bad)]:
        (args.out/name).write_bytes(data)
    (args.out/'result.json').write_text(json.dumps(decisions,indent=2)+'\n',encoding='utf-8')
    (args.out/'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in sorted(args.out.iterdir())), encoding='ascii')
    print(json.dumps(decisions), flush=True)


if __name__ == '__main__':
    main()
