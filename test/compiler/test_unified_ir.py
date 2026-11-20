"""Black-box registration, SSA and invalidation tests on the real reuse64 module."""

import json
from pathlib import Path
import re
import subprocess
import sys


def main(root, out=None, opt=None):
    opt = opt or root / 'build/native/ir/snabf-opt'
    out = out or root / 'build/unified'
    d = out/'reuse64'
    exe = [opt, '--allow-unregistered-dialect', '--verify-each']
    def run(text, *args, error=None):
        p = subprocess.run([*exe, *args], input=text, text=True, capture_output=True)
        if error is None:
            assert p.returncode == 0, p.stderr[:1500]
            return p.stdout
        assert p.returncode != 0 and error in p.stderr, (error, p.stderr[:600])

    available = subprocess.check_output([opt, '--show-dialects'], text=True)
    assert ',snabf,' in available and ',lattigo,' in available
    src = (d/'imported.mlir').read_text()
    joint = (d/'joint/selected.mlir').read_text()
    cap = (d/'cap2/selected.mlir').read_text()
    assert run(joint, '--snabf-check-plan') == joint
    assert run(joint, '--snabf-plan') == joint
    assert run(cap, '--snabf-plan') == cap
    # Check the branch actually has two operands from different SSA producers.
    pairs = re.findall(r'"snabf.ckks_step"\((%\w+), (%\w+)\) <\{kind = "ckks.mul"', src)
    assert len(pairs) == 1 and pairs[0][0] != pairs[0][1]
    assert src.count('"snabf.lintrans"') == 2 and src.count('"snabf.poly_eval"') == 1

    # Turn a valid, previously materialized plan into a stale one solely by
    # tightening a downstream operation attribute. The next solve must fix it.
    changed, n = re.subn(r'("snabf.sink"\([^)]*\) \{)', r'\1rotation_path_limit = 2 : i64, ', joint)
    assert n == 1
    run(changed, '--snabf-check-plan', error='exceeds sink rotation budget')
    assert run(changed, '--snabf-plan') == cap
    run(changed.replace('rotation_path_limit = 2', 'rotation_path_limit = 1'), '--snabf-plan', error='no optimal subgroup routing solution')
    run(joint.replace('rotation_path = 0 : i64', 'rotation_path = 9 : i64', 1), '--snabf-lower-ckks', error='stale budget')
    run(joint.replace('16384, 7, @e01>', '16384, 7, @missing>'), '--snabf-plan', error='profile')
    run(joint.replace('16384, 7, @e01>', '16000, 7, @e01>'), error='power-of-two')
    stale, n = re.subn(r'(selected = \{)b_v',r'\1bad_order',joint,count=1); assert n
    run(stale, '--snabf-lower-ckks', error='invalid selected tower')
    stale, n = re.subn(r'(route_charge = \{amortized_key_bytes = )[^ ]+',r'\g<1>0.0',joint,count=1); assert n
    run(stale,'--snabf-check-plan',error='stale site key/work charge')
    run(cap.replace('reason = "ssa_rotation_path"','reason = "feasible"'), '--snabf-check-plan',error='stale route feedback trace')
    run(joint.replace('input_bound = 1.000000e+00', 'input_bound = 5.000000e-01'), '--snabf-check-plan', error='stale budget')
    # Check that the nonlinear op itself controls materialization, not a stale copy in native_props.
    poly = next(s for s in joint.splitlines() if '"snabf.poly_eval"' in s)
    edited = poly.replace('0.49999999999999994', '0.375', 1)
    assert edited != poly
    refreshed = run(joint.replace(poly, edited), '--snabf-plan')
    lowered = run(refreshed, '--snabf-lower-ckks', '--mlir-print-op-generic')
    assert re.search(r'"kernel.eval_chebyshev"[^\n]*coefficients = \[3.750000e-01', lowered)
    report = json.loads((d/'plan.json').read_text())
    for arm, rounds, keys in [('joint', 1, 3), ('independent', 1, 4), ('cap2', 2, 4)]:
        assert report[arm]['rounds'] == rounds and len(report[arm]['keys']) == keys
    # Mutate one real prepared table's SSA binding, not the selected plan.
    ckks = (d/'ckks.mlir').read_text()
    store = next(line for line in ckks.splitlines(keepends=True)
                 if '"preprocessing.store"' in line and '!kernel.prepared_linear_transform<' in line)
    key_bytes, rem = divmod(report['joint']['key_bytes'], len(report['joint']['keys']))
    assert key_bytes > 0 and rem == 0  # one physical key family in this fixture
    importer = f'--snabf-import-heir=key-bytes={key_bytes}'
    run(ckks, importer)
    run(ckks.replace(store, store + store, 1), importer, error='ambiguous HEIR preprocessing slot')
    run(ckks.replace(store, '', 1), importer, error='unbound HEIR table slot')
    # Verify concrete profile packing against the original HEIR argument, including
    # changes which leave the number of slots and all coefficients untouched.
    layout = next(s for s in joint.splitlines() if '"snabf.profile"' in s)
    wrong = layout.replace('(-i0 + slot) mod 64', '(-i0 + slot + 1) mod 64', 1)
    assert layout != wrong
    run(joint.replace(layout, wrong), '--snabf-check-plan', error='stale route pricing')
    run(joint.replace(layout, wrong), '--snabf-plan', error='profile layout')
    # Reuse the original HEIR bootstrap and attach the Eq. (3) calibration to
    # the operation itself. The C++ pass takes the SSA level and live layout.
    fixture = root/'test/fixtures'
    base = (fixture/'pass1_bootstrap.mlir').read_text()
    record = (fixture/'pass1_boot_record.mlirattr').read_text().strip()
    raw = base.replace('{snabf_input_level', '{snabf.boot_costs = '+record+', snabf_input_level')
    heir = root/'extern/heir-release/heir/heir-opt'
    def boot(module, error=None):
        generic = subprocess.run([heir, '--lattigo-configure-crypto-context', '--mlir-print-op-generic', '--mlir-print-local-scope'],
                                 input=module, text=True, capture_output=True, check=True).stdout
        return run(generic, '--pass-pipeline=builtin.module(snabf-pass1-candidates,snabf-plan,snabf-check-plan)', error=error)
    boot(base, error='missing or invalid snabf.boot_costs')
    profitable = boot(raw)
    (out/'boot-selected.mlir').write_text(profitable)
    assert 'reason = "profitable"' in profitable and 'margin = 5.000000e+00' in profitable and 'rs_variance = 5.000000e+00' in profitable
    run(profitable, '--snabf-lower-ckks', error='no executable lowering')
    for old, new, reason in [('down_cost = 10.0', 'down_cost = 15.0', 'tie'),
        ('down_cost = 10.0', 'down_cost = 16.0', 'negative_gain'), ('0.000732421875', '1.0', 'downstream_budget'),
        ('snabf_input_level = 3', 'snabf_input_level = 0', 'no_input_level_for_switch'), ('layout_verified = true','layout_verified = false','incompatible_packing')]:
        assert f'reason = "{reason}"' in boot(raw.replace(old, new))
    assert 'reason = "incompatible_packing"' in boot(raw.replace('slot_period = 4096','slot_period = 8192'))
    assert 'reason = "incompatible_packing"' in boot(raw.replace(' {snabf.slot_period = 4096 : i64}',''))
    run(profitable.replace('margin = 5.000000e+00','margin = 6.000000e+00'), '--snabf-check-plan',error='stale ring-switch decision')
    # A Pass 4 consumer can post a cap tied to the actual input packing;
    # this models layout feedback, not execution of a conversion.
    line = next(s for s in joint.splitlines() if '"snabf.lintrans"' in s)
    layout = re.search(r'layout = (.*?), physical_scope = ', line).group(1)
    feedback = ('snabf.route_feedback = {source = "pass4_layout", profile = @e01, ring = 16384 : i64, '
                f'layout = {layout}, max_serial_rotations = 1 : i64}}')
    posted = joint.replace(line,line.replace('route_charge =',feedback+', route_charge =',1))
    run(posted,'--snabf-check-plan',error='stale route pricing')
    replanned=run(posted,'--snabf-plan','--snabf-check-plan'); assert run(replanned,'--snabf-plan')==replanned
    assert re.search(r'bytes = 41946496 : i64',replanned)
    (out/'layout-feedback.mlir').write_text(replanned)
    run(posted.replace('profile = @e01, ring = 16384 : i64','profile = @e01, ring = 8192 : i64'), '--snabf-plan',error='current physical layout')
    # All mathematical subgroups survive executable-route compression, and
    # dense pricing is separately selectable instead of silently taxing zeros.
    dense=run(src.replace('snabf.route_pricing = "occupied"','snabf.route_pricing = "dense_tower"'),'--snabf-plan','--snabf-check-plan')
    (out/'dense-tower.mlir').write_text(dense)
    assert dense != joint and 'pricing = "dense_tower"' in dense
    print('PASS: complete tower catalog, charge/feedback tamper, Pass 4 layout contract, dense pricing')
    print('PASS: native Pass 1 Eq. (3), positive/tie/negative/variance/level/layout branches and planning-only boundary')
    print('PASS: native dialect registration, SSA branch, round-trip, fixed point, stale-plan rejection, sink feedback, infeasibility, profile/type checks and IR-guided lowering')


if __name__ == '__main__':
    main(*(Path(a) for a in sys.argv[1:]))
