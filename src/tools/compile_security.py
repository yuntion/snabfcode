"""Build Pass 3/4 evidence in the formats replayed by the native gates."""
import math

from pass4_security import build_cache
from security_cache import ERROR, MODEL, PIN, SECRET, digest, measure, request


def source_evidence(plan, cache, estimator):
    n, q, p = plan['ring'], plan['q'], plan['p']
    Q = math.prod(q)
    moduli = dict(Q=Q, QP=Q*math.prod(p), envelope=1 << math.ceil(plan['security_cap']))
    reqs = {k:request('source.'+k, n, q, p, mod) for k,mod in moduli.items()}
    costs = measure(reqs.values(), cache, estimator)
    instances = {k:dict(modulus=r['modulus'], attacks_log2_rop=costs[digest(r)]) for k,r in reqs.items()}
    # Emit measurements, never a cached verdict. C++ independently binds the
    # public inventory, exact Q/QP and envelope and replays the threshold.
    return dict(ring=n, estimator=PIN, model=MODEL, secret=SECRET, error=ERROR, samples='unbounded',
                inventory=plan['public_instances'], instances=instances)


def domain_evidence(plan, sizes, cache, estimator):
    data = build_cache(plan, sizes, plan['requested_slots'], lambda n,m:dict(n=n,modulus=str(m)))
    rows = []
    def bind(role, group, fields):
        for label in fields:
            row = group[label]
            r = request(role+'.'+label,row['n'],group['q'],group['p'],int(row['modulus']))
            if label.startswith('switch_'):
                r['switch_basis'] = group['switch_basis']
            rows.append((row,r))
    bind('source', data['source'], ('Q','QP'))
    for group in data['profiles'].values():
        bind('target', group, ('Q','QP','switch_QP_small_secret','switch_QP_large_ring'))
    costs = measure([r for _,r in rows], cache, estimator)
    for row,r in rows: row.update(costs[digest(r)])
    return data
