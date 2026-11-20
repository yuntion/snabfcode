"""Archive measured inference results alongside the exact workload and plan hashes."""

import argparse
import hashlib
import json
from pathlib import Path


def collect(root, log):
    rows = {}
    for line in log.read_text().splitlines():
        if 'SNABF_PASS1 {' not in line:
            continue
        v = json.loads(line.split('SNABF_PASS1 ', 1)[1]); key = (v['workload'], v['arm'])
        if key in rows: raise ValueError(f'duplicate run {key}')
        rows[key] = v
    expected = {(w,a) for w in ('control','strided64','reuse64') for a in ('heir','independent','joint')}
    if rows.keys() != expected: raise ValueError('missing workload/arm')
    evidence = {'runs': list(rows.values()), 'workloads': {}}
    for w in sorted({w for w,_ in rows}):
        manifest = json.loads((root/w/'heir/manifest.json').read_text())
        manifest['plan_sha256'] = hashlib.sha256((root/w/'plan.json').read_bytes()).hexdigest()
        evidence['workloads'][w] = manifest
    boot = json.loads((root/'boot/plan.json').read_text())['bootstrap']['sites']['bootstrap:%ct_0']
    if not boot['enabled'] or boot['margin'] != 5 or boot['rs_variance'] != 5:
        raise ValueError(f'unexpected HEIR bootstrap planning outcome: {boot}')
    evidence['bootstrap_plan_only'] = boot
    (root/'metrics.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(f'Saved {len(rows)} encrypted runs and the HEIR bootstrap decision to {root}/metrics.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('log', type=Path)
    args = parser.parse_args(); collect(args.root, args.log)
