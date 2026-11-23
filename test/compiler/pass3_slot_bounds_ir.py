"""Exercise the real legacy Pass 3 extractor, fixed replay and checker. No keys/HEIR binary."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def fixture():
    # This is a public synthetic *post-import* ABI, not a security fixture.
    q = [140858119421953, 2013265921, 1811939329, 270155612161, 270155972609, 270155284481,
         270156300289, 270154924033, 270154498049, 270157479937]
    aux = 288230376150630401
    level = len(q) - 1
    chain = ', '.join(map(str, q))
    elements = ', '.join(f'{v} : i64' for v in q)
    masks = []
    for d in range(33):
        row = [0.] * 64
        row[d] = -775/1024 if d == 32 else 1.
        masks.append(row)
    selector = [0.] * 64
    selector[63], selector[59] = 2., 1.  # offset 37 reads slots 36 (zero) and 32
    operations = [(list(range(33)), masks), ([37], [selector]), ([0], [[-1.] * 64])]
    layout = ('#tensor_ext.original_type<originalType = tensor<64xf32>, layout = '
              '#tensor_ext.layout<"{ [i0] -> [ct, slot] : ct = 0 and (-i0 + slot) mod 64 '
              '= 0 and 0 <= i0 <= 63 and 0 <= slot <= 63 }">>')
    lines = [f'#layout = {layout}',
        '!cipher = !lwe.lwe_ciphertext<plaintext_space = <ring = <polynomialModulus = <1 + x**128>>, '
        'encoding = #lwe.inverse_canonical_encoding<scaling_factor = 38>>, key = <>, '
        f'modulus_chain = <elements = <{elements}>, current = {level}>>',
        f'!v = !snabf.ct<tensor<64xf32>, 128, {level}, @p>',
        f'!mask = !kernel.prepared_linear_transform<level = {level}, slots = 64, log_bsgs_ratio = 0>',
        'module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 7, '
        f'Q = [{chain}], P = [{aux}], logDefaultScale = 38>, scheme.requested_slot_count = 64 : i64, '
        'snabf.activation_sources = [{function = "probe", semantics = "silu"}], snabf.giant = 1.0 : f64, '
        'snabf.hoist = 0.625 : f64, snabf.key_units_per_mib = 1.0 : f64} {',
        '"snabf.profile"() <{sym_name = "p", ring = 128 : i64, '
        f'q = array<i64: {chain}>, p = array<i64: {aux}>, default_log_scale = 38 : i64, '
        'key_id = "probe__configure:<>", layout = #layout, slots = 64 : i64, key_bytes = 128 : i64}> : () -> ()',
        'func.func @prepare() -> i64 {', '%h = "preprocessing.empty"() : () -> i64']
    for i, (offsets, rows) in enumerate(operations):
        ty = f'tensor<{len(rows)}x64xf32>'
        indices = ', '.join(map(str, offsets))
        lines += [f'%w{i} = arith.constant dense<{json.dumps(rows)}> : {ty}',
                  f'%t{i} = "kernel.prepare_linear_transform"(%w{i}) '
                  f'{{diagonal_indices = array<i64: {indices}>}} : ({ty}) -> !mask',
                  f'"preprocessing.store"(%t{i}, %h) {{site_id = {i} : i32}} : (!mask,i64) -> ()']
    lines += ['return %h : i64', '}',
              'func.func @eval(%x: tensor<1x!cipher> {tensor_ext.original_type = #layout}, %h: i64) '
              '-> !cipher attributes {heir.interface = {func_name = "probe"}} {', '%idx = arith.constant 0 : index',
              '%arg = tensor.extract %x[%idx] : tensor<1x!cipher>',
              '%x0 = "snabf.source"(%arg) {input_bound = 0.5 : f64, native = !cipher} : (!cipher) -> !v']
    for i, (offsets, _) in enumerate(operations):
        indices = ', '.join(map(str, offsets))
        lines += [f'%t{i} = "preprocessing.load"(%h) {{site_id = {i} : i32}} : (i64) -> !mask',
                  f'%x{i+1} = "snabf.lintrans"(%x{i}, %t{i}) '
                  f'<{{diagonals = array<i64: {indices}>}}> {{native = !cipher}} : (!v,!mask) -> !v']
    lines += ['%zero = "snabf.ckks_step"(%x2,%x3) <{kind = "ckks.add"}> '
              '{native = !cipher, native_props = {}} : (!v,!v) -> !v',
              '%act = "snabf.poly_eval"(%x2) <{activation = "silu", coefficients = []}> '
              '{native = !cipher} : (!v) -> !v',
              '%sum = "snabf.ckks_step"(%act,%zero) <{kind = "ckks.add"}> '
              '{native = !cipher, native_props = {}} : (!v,!v) -> !v',
              '%out = "snabf.sink"(%sum) : (!v) -> !cipher', 'return %out : !cipher', '}',
              'func.func @probe(%x: tensor<1x!cipher>) -> !cipher {', '%h = func.call @prepare() : () -> i64',
              '%y = func.call @eval(%x,%h) : (tensor<1x!cipher>,i64) -> !cipher', 'return %y : !cipher', '}', '}']
    return '\n'.join(lines) + '\n'


def run(opt, work):
    def native(text, *flags, reject=None):
        p = subprocess.run([opt, '--allow-unregistered-dialect', '--verify-each', '--mlir-print-op-on-diagnostic=false',
                            *flags], cwd=work, input=text, text=True, capture_output=True)
        if reject is None:
            check(p.returncode == 0, p.stderr[:2000])
        else:
            check(p.returncode == 1 and reject in p.stderr, (reject, p.returncode, p.stderr[:2000]))
        return p.stdout

    def select(interval, text):
        return native(text, f'--snabf-pass2-candidates=interval-mode={interval} depth=5 epsilon=0.001 template-family=direct',
                      '--snabf-plan', '--snabf-check-plan')

    fixed = '--snabf-pass3=fixed-exact=true max-k=0 epsilon=4 security-cap=512'
    def proposal(text, mode):
        flags = fixed + ' audit=plan.json' + (' magnitude-mode='+mode if mode else '')
        ir = native(text, flags)
        plan = json.loads((work/'plan.json').read_text())
        return ir, plan, {e['ID']: e for e in plan['edges']}

    check_plan = '--snabf-check-plan'
    missing_screen = 'missing or invalid snabf.security_screen'
    selected = select('naive', fixture())
    default, _, old = proposal(selected, None)
    legacy, _, _ = proposal(selected, 'legacy')
    check(default == legacy and 'magnitude_mode' not in default, 'legacy record changed')
    corrected, plan, new = proposal(selected, 'co-slot-v1')
    check(plan['options']['magnitude_mode'] == 'co-slot-v1', 'unspecified replay policy')
    check(old['lin0.rescale']['magnitude'] == 16.37841796875 and new['lin0.rescale']['magnitude'] == .5,
          'did not fix the diagonal-max counterexample in native IR')
    check(new['lin1.rescale']['magnitude'] == .5*775/1024 and new['step0']['magnitude'] == 775/1024,
          'second tower/add lost per-slot parents')
    check(new['lin1.mask37']['gains'] == [2.] and set(old) == set(new), 'changed mask gain or topology')
    check(not any('linear_noise_entry' in e or 'inherited_noise_gain' in e for e in old.values()),
          'legacy inherited an unrequested noise transfer')
    check(new['lin0.rescale']['inherited_noise_gain'] == 1 and
          plan['edges'][new['lin0.rescale']['linear_noise_entry']]['ID'] == 'input' and
          plan['edges'][new['lin1.rescale']['linear_noise_entry']]['ID'] == 'lin0.rescale',
          'real IR did not reset the error unit box at each linear entrance')
    for name, e in new.items():
        if e['op'] == 'mul':
            a,b = (plan['edges'][i]['magnitude'] for i in e['inputs'])
            check(e['gains'] == [b,a], 'multiply needs the opposite global operand bounds')
        else:
            check(e['gains'] == old[name]['gains'], 'co-slot gain leaked into '+name)
    # The real checker must replay both modes to the independent security gate.
    # Supply no fabricated security evidence; neither proposal is sealed.
    for text in (default, corrected):
        native(text, check_plan, reject=missing_screen)
    marker = 'magnitude_mode = "co-slot-v1", '
    check(corrected.count(marker) >= 2, 'missing module/profile mode copies')
    native(corrected.replace(marker,''), check_plan, reject='stale Pass 3 site, scale or error transfer')
    native(corrected.replace('"co-slot-v1"','"unknown"'), check_plan, reject='unknown recorded Pass 3 magnitude mode')
    import re
    wrong_gain, count = re.subn(r'inherited_noise_gain = [^ ]+ : f64', 'inherited_noise_gain = 0.0 : f64', corrected)
    check(count >= 2, 'missing both copies of recomputed inherited gain')
    native(wrong_gain, check_plan, reject='stale Pass 3 site, scale or error transfer')

    affine = select('affine', fixture())
    _, _, projected = proposal(affine, 'co-slot-v1')
    check(projected['step0']['magnitude'] == 0 and projected['lin2.mask0']['magnitude'] == new['lin1.rescale']['magnitude'],
          'the recomputed affine result should prove x + (-x) == 0, not erase its parent')
    # Same widths and divisibility, different layout. Scalar legacy planning
    # can handle this fixture, but co-slot-v1 must not assert this lane map.
    wrong = select('naive', fixture().replace('(-i0 + slot)', '(i0 + slot)'))
    native(wrong, fixed+' magnitude-mode=co-slot-v1', reject='unproven periodic source layout')


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        run(Path(sys.argv[1]).resolve(strict=True), Path(tmp))
