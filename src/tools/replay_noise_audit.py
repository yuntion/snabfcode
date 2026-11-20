"""Recompute an engineering noise audit from the checked MLIR, never its cached edge bounds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def replay(package, native, precision_bits):
    package = package.resolve(strict=True)
    native = native.resolve(strict=True)
    ir = (package / 'work/checked.mlir').resolve(strict=True)
    expected = json.loads((package / 'audits/noise_cert_audit.json').read_bytes())
    env = os.environ.copy()
    env.pop('SNABF_PASS4_ESTIMATES', None)
    evidence = package / 'work/domains.json'
    if evidence.is_file():
        env['SNABF_PASS4_ESTIMATES'] = str(evidence)
    with tempfile.TemporaryDirectory(prefix='snabf-noise-replay-') as stage:
        subprocess.run([native, '--allow-unregistered-dialect', '--verify-each',
                        f'--snabf-check-plan=noise-audit=fresh.json precision-bits={precision_bits}',
                        ir, '-o', os.devnull], cwd=stage, env=env, check=True)
        fresh = json.loads((Path(stage) / 'fresh.json').read_bytes())
    # Canonical comparison also distinguishes bool from int, unlike Python dict equality.
    def canonical(value):
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))
    if canonical(expected) != canonical(fresh):
        raise ValueError('stored audit differs from the independently replayed engineering audit')
    return fresh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--snabf-opt', required=True, type=Path)
    parser.add_argument('--precision-bits', type=int, default=12)
    args = parser.parse_args()
    try:
        report = replay(args.package, args.snabf_opt, args.precision_bits)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps({'matched': True, 'mode': report['mode'], 'sinks': len(report['sinks']),
                      'probabilistic_cert_issued': report['probabilistic_cert_issued']}))
