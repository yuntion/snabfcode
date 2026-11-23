"""Independent check on actual public coefficients. Never used by the audit.

No Decode, libm, or measured tolerance. Evaluate C at all distinct slot
embeddings by integer interval arithmetic and use periodicity for the rest.
"""
from fractions import Fraction as Q
from functools import lru_cache
from math import isqrt


@lru_cache(None)
def powers(p):
    f = 1 << 112
    cos = (0, 0)  # primitive fourth root is i; halve the angle log2(p) times
    for _ in range(p.bit_length()-1):
        l, h = cos
        sin = isqrt((f-h)*f//2), 1+isqrt((f-l)*f//2)
        cos = isqrt((f+l)*f//2), 1+isqrt((f+h)*f//2)
    def mul(a,b):
        v = [x*y for x in a for y in b]
        return min(v)//f, -(-max(v)//f)
    table = [((f,f),(0,0))]
    for _ in range(1, 4*p):
        a, b = table[-1]
        ac, bd, ad, bc = mul(a,cos), mul(b,sin), mul(a,sin), mul(b,cos)
        table.append(((ac[0]-bd[1], ac[1]-bd[0]), (ad[0]+bc[0], ad[1]+bc[1])))
    return f, table


def check_slots(words, values, scale, bound):
    p = len(words)//2
    assert 2 <= p <= 64 and not p & (p-1) and len(words) == 2*p and pow(5,p,4*p) == 1
    assert all(x == values[i%p] for i,x in enumerate(values))
    f, table = powers(p)
    for i in range(p):
        lo = [0,0]; hi = [0,0]; rot = pow(5,i,4*p)
        for k,c in enumerate(words):
            for j,(a,b) in enumerate(table[(k*rot)%(4*p)]):
                lo[j] += min(c*a,c*b); hi[j] += max(c*a,c*b)
        ideal = f*scale*Q(values[i])
        re = max(abs(lo[0]-ideal), abs(hi[0]-ideal)); im = max(abs(lo[1]), abs(hi[1]))
        if re*re + im*im > (f*bound)**2:
            raise ValueError(f'actual public polynomial exceeds slot error at class {i}')
