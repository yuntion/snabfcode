"""Exercise the explicit legacy-to-fresh transition, never editing the R3 archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/tools'))
from replay_noise_audit import Attributes, need


SEAL = re.compile(r'snabf\.emit_sha256 = "([0-9a-f]{64})"')
RECEIPTS = 'snabf.encoding_receipts = '
CHECK, LOWER = '--snabf-check-plan', '--snabf-lower-ckks=direct-go=true'


def exercise(opt, fixture):
    source = fixture / 'checkpoint.mlir'
    data = source.read_bytes()
    legacy = data.decode()
    old, = SEAL.findall(legacy)
    need(RECEIPTS not in legacy, 'expected an unbound R3 checkpoint')
    estimates = fixture / 'estimates.json'
    n = json.loads(estimates.read_bytes())['source_ring']
    env = dict(os.environ, SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS='128', SNABF_PASS4_ESTIMATES=str(estimates))
    with tempfile.TemporaryDirectory(prefix='snabf-b2-replan-') as tmp:
        root = Path(tmp)
        search_audit = root / 'rejected-search.json'
        replan = '--snabf-pass4=audit=' + str(search_audit)
        bind = '--snabf-bind-lattigo=ring-go=model.go audit=program.json'
        negatives = 0

        def run(label, ir, *flags, reject=None):
            nonlocal negatives
            out = root / (label + '.mlir')
            p = subprocess.run([opt, '--allow-unregistered-dialect', '--verify-each', '--mlir-print-op-on-diagnostic=false',
                                *flags, ir, '-o', out], cwd=root, env=env, capture_output=True, text=True)
            if reject is None:
                need(p.returncode == 0 and out.is_file(), label + ': ' + p.stderr[:600])
            else:
                need(p.returncode > 0 and reject in p.stderr and not out.exists() and not search_audit.exists(),
                     label + ' did not fail closed: ' + p.stderr[:600])
                negatives += 1
            return out

        def mutate(name, txt):
            f = root / (name + '-input.mlir')
            f.write_text(txt)
            return f

        for label, flag in [('check', CHECK), ('cert', CHECK + '=require-noise-cert=true'), ('direct', LOWER), ('bind', bind)]:
            run(label, source, flag, reject='missing_encoding_scale_binding')
        need(not (root / 'model.go').exists() and not (root / 'program.json').exists(), 'published unbound artifacts')

        run('stale-seal', mutate('sha', legacy.replace(old, '0' * 64, 1)), replan, reject='stale_emit_fingerprint')
        run('foreign-attr', mutate('attr', SEAL.sub(r'b2.changed = true, \g<0>', legacy, count=1)),
            replan, reject='stale_emit_fingerprint')
        # Lowering is excluded from the old seal; this fails only if compared to the actual concrete program.
        run('forged-legacy-lowering', mutate('lowering', SEAL.sub(r'snabf.ckks_direct = {}, \g<0>', legacy, count=1)),
            replan, reject='stale_epoch_lowering')

        planned = run('replanned', source, '--snabf-pass4')
        text = planned.read_text()
        need('snabf.ring_search = ' in text and not SEAL.search(text) and RECEIPTS not in text,
             'a replan must return a new, unsealed plan')
        run('unsealed-lower', planned, LOWER, reject='snabf.emit_sha256')
        checked = run('checked', planned, CHECK)
        checked_text = checked.read_text()
        new, = SEAL.findall(checked_text)
        need(new != old and RECEIPTS in checked_text, 'new graph did not receive a fresh binding')
        lowered = run('lowered', checked, LOWER)
        run('denied-cert', checked, CHECK + '=noise-audit=denied.json require-noise-cert=true',
            reject='noise certificate denied; see denied.json')
        report = json.loads((root / 'denied.json').read_bytes())
        need(report['errors'] == [] and report['probabilistic_cert_issued'] is False and report['missing_contract_terms'],
             'expected a missing-contract refusal')

        start = checked_text.index(RECEIPTS)
        parser = Attributes(checked_text, start + len(RECEIPTS))
        parser.value()
        need(checked_text[parser.pos:parser.pos + 2] == ', ', 'unexpected receipt syntax')
        run('stripped-bound', mutate('strip', checked_text[:start] + checked_text[parser.pos + 2:]),
            replan, reject='stale_emit_fingerprint')
        damaged, edits = re.subn(r'(snabf\.ckks_epochs = \{cut_level = )-?\d+', r'\g<1>99', lowered.read_text(), count=1)
        need(edits == 1, 'expected the checked epoch lowering')
        run('stale-bound-lowering', mutate('epoch', damaged), replan, reject='stale_epoch_lowering')
        # A valid, already-lowered input may be replanned under a genuinely new policy.
        again = run('rechecked', lowered, f'--snabf-pass4=min-ring={n}', CHECK)
        text = again.read_text()
        need(RECEIPTS in text and 'snabf.ckks_epochs = ' not in text and SEAL.findall(text) not in ([], [new]),
             'the previous lowering or seal survived a change of policy')
        need(source.read_bytes() == data, 'modified the frozen checkpoint')
        return dict(case=fixture.name, frozen_sha256=hashlib.sha256(data).hexdigest(), new_seal=new, denials=negatives)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snabf-opt', type=Path, required=True)
    p.add_argument('--r3', type=Path, default=ROOT / 'doc/e2e-matrix-2026-11-20-r3')
    args = p.parse_args()
    for case in ('cross_residual', 'reuse64'):
        print(json.dumps(exercise(args.snabf_opt.resolve(), args.r3.resolve() / 'frozen' / case)), flush=True)
