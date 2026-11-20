"""Exercise the source lemma on sealed checkpoint(s); never write into an archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from replay import ROOT, load, need, pre_moddown, runtime_identity, same, source_identity


def refused(fn, reason):
    try:
        fn()
    except ValueError as exc:
        need(reason in str(exc), str(exc))
        return
    raise AssertionError('accepted: ' + reason)


def support_tests(module, stage):
    large = 180143985094819841  # prime not representable in binary64
    r = pre_moddown(8, [97, large], [241, 337])
    # A coefficient oracle: with s=u=all ones and e_pk=e1=(19,-19,...),
    # each wrapped term changes sign. This tests support, not likelihood.
    err = [19] + [-19]*7
    need(r['combined_abs'] == 2*(err[0]-sum(err[1:]))+19 and
         r['centered_modulus'] == str(97*large*241), 'convolution sign, integer precision or P level')
    refused(lambda: pre_moddown(8, [97], []), 'unsupported public-key QP')
    refused(lambda: pre_moddown(8, [17], [241]), 'small-norm basis extension')
    # Keep the module path/version and the rest of its files, change only Xs.
    copy = stage / 'changed-module'
    shutil.copytree(module, copy)
    path = copy / 'core/rlwe/security.go'
    original = path.read_bytes()
    modified = original.replace(b'Ternary{P: 2 / 3.0}', b'Ternary{H: 64}')
    need(modified != original, 'wrong pinned source fixture')
    path.chmod(0o600)  # real Go module-cache files are read-only
    path.write_bytes(modified)
    refused(lambda: source_identity(copy), 'unsupported Lattigo sampler source')
    # The reviewed Bind function alone is not enough: another file can change
    # the global defaults before it runs. Only alter a private source copy.
    copy = stage / 'changed-runtime'
    shutil.copytree(ROOT/'src/backend', copy/'src/backend')
    for name in ('go.mod', 'go.sum'):
        shutil.copyfile(ROOT/name, copy/name)
    runtime_identity(copy)
    (copy/'src/backend/ringswitch/epochs/override.go').write_text(
        'package epochs\nimport "github.com/tuneinsight/lattigo/v6/core/rlwe"\nfunc init() { rlwe.DefaultXe.Bound = 64 }\n')
    refused(lambda: runtime_identity(copy), 'unreviewed direct Go runtime')


def exercise(native, module, checkpoint, stage):
    checked, params = checkpoint / 'checked.mlir', checkpoint / 'program.json'
    cli = [sys.executable, '-B', Path(__file__).with_name('replay.py').resolve(),
           '--snabf-opt', native, '--lattigo-src', module]

    def run(ir=checked, program=params, compare=None, reason=None):
        cmd = [*cli, '--checked', ir, '--params', program]
        if compare:
            cmd.extend(['--compare', compare])
        proc = subprocess.run(cmd, cwd=stage, text=True, capture_output=True)
        if reason:
            reasons = (reason,) if isinstance(reason, str) else reason
            need(proc.returncode > 0 and any(r in proc.stderr for r in reasons) and not proc.stdout,
                 'wrong refusal: ' + proc.stderr[:500])
            return
        need(proc.returncode == 0, proc.stderr[:500])
        return json.loads(proc.stdout)

    receipt = run()
    need(receipt['probabilistic_cert_issued'] is False and receipt['key_scope']['proven_independent_draws'] == [] and
         receipt['emitted_go_sha256'] == hashlib.sha256((checkpoint/'model.go').read_bytes()).hexdigest(),
         'wrong code identity or support promoted to a probability proof')
    candidate = stage / 'candidate.json'
    candidate.write_text(json.dumps(receipt))
    need(same(run(compare=candidate), receipt), 'valid receipt did not replay')
    for field, value in [('key_scope', dict(receipt['key_scope'], proven_independent_draws=['slot0', 'slot1'])),
                         ('support', dict(receipt['support'], combined_abs=receipt['support']['combined_abs']-1)),
                         ('fixed_real_input', {}), ('probabilistic_cert_issued', True)]:
        candidate.write_text(json.dumps(dict(receipt, **{field: value})))
        run(compare=candidate, reason='receipt differs from recomputation')
    program = load(params)
    program['epochs'][0]['Xs'] = {'H': 64}
    candidate.write_text(json.dumps(program))
    run(program=candidate, reason='sidecar differs from native re-emission')
    lines = checked.read_text().splitlines(keepends=True)
    idx, = [i for i, line in enumerate(lines) if ' = "snabf.source"(' in line]
    lines[idx], count = re.subn(r'\binput_bound = [\w.+-]+ : f64', 'input_bound = 5.000000e-01 : f64', lines[idx])
    need(count == 1, 'expected declared source bound')
    changed = stage / 'changed.mlir'
    changed.write_text(''.join(lines))
    run(ir=changed, reason=('stale_emit_fingerprint', 'stale Pass 2 frontier or certificate'))
    text, count = re.subn(r'(snabf\.emit_sha256 = ")([0-9a-f])(?=[0-9a-f]{63}")',
                          lambda m: m[1] + ('1' if m[2] == '0' else '0'), checked.read_text())
    need(count == 1, 'expected exactly one seal')
    changed.write_text(text)
    run(ir=changed, reason='stale_emit_fingerprint')

    for mode, archive in [('engineering', 'engineering.json'), ('probability', 'denied.json')]:
        old = load(checkpoint / archive)
        opt = f"--snabf-check-plan=noise-audit={archive} precision-bits={old['precision_bits']}"
        if mode == 'probability':
            opt += ' require-noise-cert=true'
        proc = subprocess.run([native, '--allow-unregistered-dialect', '--verify-each', '--mlir-print-op-on-diagnostic=false',
                               opt, checked, '-o', os.devnull], cwd=stage, text=True, capture_output=True)
        need(proc.returncode == 0 if mode == 'engineering' else proc.returncode > 0 and 'noise certificate denied' in proc.stderr,
             'wrong native gate: ' + proc.stderr[:500])
        audit = load(stage / archive)
        need(same(audit, old) and audit['mode'] == mode and audit['probabilistic_cert_issued'] is False and
             'common/validated_joint_law_and_domain' in audit['missing_contract_terms'], 'archive audit changed')
    print(f"{checkpoint.name}: support={receipt['support']['combined_abs']}, edges={len(audit['edges'])}, "
          f"sites={len(audit['sites'])}, missing={len(audit['missing_contract_terms'])}; archived audits match")


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--snabf-opt', type=Path, required=True)
    cli.add_argument('--lattigo-src', type=Path, required=True)
    cli.add_argument('--checkpoint', type=Path, action='append', required=True)
    args = cli.parse_args()
    with tempfile.TemporaryDirectory(prefix='snabf-b2-test-') as tmp:
        module = args.lattigo_src.resolve(strict=True)
        support_tests(module, Path(tmp))
        for i, folder in enumerate(args.checkpoint):
            stage = Path(tmp) / str(i)
            stage.mkdir()
            exercise(args.snabf_opt.resolve(strict=True), module, folder.resolve(strict=True), stage)
