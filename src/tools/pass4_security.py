"""Cache measurements, not admission decisions. Thresholds are replayed in C++."""
import argparse
import functools
import json
import math
import subprocess
import sys
from pathlib import Path

from pass3_security import PIN


def build_cache(plan, sizes, slots, instance):
    source = plan['ring']
    q_chain, p_chain = plan['q'], plan['p']
    if (slots <= 0 or slots > source // 2 or not q_chain or not p_chain
            or any(q < 3 or q % (2 * source) != 1 for q in q_chain + p_chain)
            or len(set(q_chain + p_chain)) != len(q_chain) + len(p_chain)):
        raise ValueError('invalid concrete source basis or slot count')
    profiles = {}
    for key, shape in sizes.items():
        n, level = map(int, key.split(':'))
        if n < 4 or n >= source or n & (n - 1) or source % n or not 0 <= level < len(q_chain):
            raise ValueError(f'invalid proposed epoch: {key}')
        if n // 2 < slots:
            continue
        lq, lp = shape['key_level_q'], shape['key_level_p']
        if (lq != level or lp != len(p_chain) - 1
                or shape['switch_bytes'] <= 0 or shape['key_bytes'] <= 0):
            raise ValueError(f'keyshape did not measure the proposed switching basis: {key}')
        q, p = q_chain[:level + 1], p_chain
        eq, ep = q_chain[:lq + 1], p_chain[:lp + 1]
        Q, QP = math.prod(q), math.prod(q) * math.prod(p)
        ext = math.prod(eq) * math.prod(ep)
        profiles[key] = dict(q=q, p=p, **shape, Q=instance(n, Q), QP=instance(n, QP),
                             switch_basis=dict(q=eq, p=ep, physical_ring=source, embedded_secret_ring=n),
                             switch_QP_small_secret=instance(n, ext), switch_QP_large_ring=instance(source, ext))
    full = math.prod(q_chain)
    return dict(schema='snabf.security-domains.v2', estimator=PIN,
                model='rough-classical-unbounded-ternary-sigma3.2', source_ring=source,
                source=dict(q=q_chain, p=p_chain, Q=instance(source, full), QP=instance(source, full * math.prod(p_chain))),
                profiles=profiles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for k in ('plan', 'sizes', 'output', 'estimator'):
        parser.add_argument(k, type=Path)
    parser.add_argument('--slots', type=int, required=True)
    args = parser.parse_args()
    root = args.estimator.resolve()
    if subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() != PIN:
        raise ValueError('unrecognized estimator revision')
    sys.path.insert(0, str(root))
    from estimator import LWE, ND
    from sage.all import infinity, log

    @functools.lru_cache(None)
    def instance(n, modulus):
        par = LWE.Parameters(n=n, q=modulus, Xs=ND.Uniform(-1, 1), Xe=ND.DiscreteGaussian(3.2), m=infinity)
        raw = LWE.estimate.rough(par, quiet=True, catch_exceptions=False)
        attacks = {str(k): float(log(v['rop'], 2)) for k, v in raw.items()}
        if not {'usvp', 'dual_hybrid'}.issubset(attacks) or any(math.isnan(v) or v <= 0 for v in attacks.values()):
            raise ValueError('missing or invalid concrete attack estimate')
        # No finite attack returned is not a security certificate.
        return dict(n=n, modulus=str(modulus), **{k: (v if math.isfinite(v) else 'inf') for k, v in attacks.items()})

    data = build_cache(json.loads(args.plan.read_text()), json.loads(args.sizes.read_text()), args.slots, instance)
    args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
