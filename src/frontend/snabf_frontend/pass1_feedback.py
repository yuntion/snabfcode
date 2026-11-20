"""Repeat the residual precision-feedback experiment with the current Pass 1."""
import argparse
from pathlib import Path
from .pass2 import Pipeline, TOLERANCE, write_harness
from .pass2_evidence import require


def main(a):
    a.extra_rotation_cap = None
    pp = Pipeline(a)
    d = pp.out / 'residual'
    oracle = pp.source('residual', d)
    src = pp.import_ckks('residual', d)
    # Same public sink constraints as the archived Pass 2 acceptance, not
    # a polynomial/rung selector. The lowering and keys are freshly generated.
    arms = {'economy': TOLERANCE, 'precision': 3.4614778033368946e-9}
    reports = [pp.plan(d, arm, src, 0, cap) for arm, cap in arms.items()]
    a0, a1 = [sorted(r['route_sites'], key=lambda s: s['site'])[1] for r in reports]
    require(a0['level'] != a1['level'] and a0['selected']['terms'] != a1['selected']['terms'],
            'residual: sink precision did not change second-layer level and rotation program')
    require(any(e['reason'] == 'backward_ssa' for e in reports[1]['decisions']), 'missing backward SSA feedback')
    write_harness(pp.root, pp.out, {'residual': {'arms': list(arms), 'width': oracle.width}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','out','heir-bin','snabf-opt','goemit'):
        p.add_argument('--'+name, type=Path, required=True)
    main(p.parse_args())
