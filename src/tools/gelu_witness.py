"""Portable C1 source oracle. No HEIR, runtime plan, libm erf or third-party module."""
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
from functools import lru_cache
import itertools
import json
import math
import random
import re

from pass2_evidence import f32, load, require, sha

PREC = 110
SCALE = 10**(PREC + 6)


@lru_cache(maxsize=1)
def sqrt_two_over_pi():
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239). Consecutive alternating
    # partial sums bracket each arctangent. isqrt then gives rational bounds.
    def atan(q):
        x, s = F(1, q), F(0)
        for n in itertools.count():
            t = x / (2*n+1)
            if abs(t) < F(1, SCALE):
                return sorted((s, s+t))
            s += t
            x /= -q*q
    a, b = atan(5), atan(239)
    lo, hi = 16*a[0]-4*b[1], 16*a[1]-4*b[0]
    def floor_root(v):
        return math.isqrt(v.numerator*SCALE*SCALE // v.denominator)
    return F(floor_root(2/hi), SCALE), F(floor_root(2/lo)+1, SCALE)


def erf_bounds(h):
    # erf(h/sqrt(2)) = sqrt(2/pi)*sum (-1)^n h^(2n+1)/(2^n n! (2n+1)).
    # Exact rational arithmetic, stop only in the decreasing alternating tail.
    x = abs(F.from_float(h))
    if not x:
        return F(0), F(0)
    require(x <= 4, 'outside C1 public erf witness domain')
    t, s = x, F(0)
    for n in itertools.count():
        if x*x <= 2*(n+1) and abs(t)*SCALE < x:
            lo, hi = sorted((s, s+t))
            c, d = sqrt_two_over_pi()
            return (lo*c, hi*d) if h > 0 else (-hi*d, -lo*c)
        s += t
        t *= -x*x*(2*n+1) / (2*(n+1)*(2*n+3))


def decimal_endpoint(v, rounding):
    with localcontext() as ctx:
        ctx.prec, ctx.rounding = PREC, rounding
        return Decimal(v.numerator) / Decimal(v.denominator)


@lru_cache(maxsize=8192)
def atom(kind, hx):
    h = float.fromhex(hx)
    require(math.isfinite(h), 'invalid transcendental input')
    if kind == 'gelu':
        a,b = erf_bounds(h)
        lo, hi = decimal_endpoint(a, ROUND_FLOOR), decimal_endpoint(b, ROUND_CEILING)
        if not h and hx.startswith('-'):
            lo = hi = Decimal('-0')
    elif kind == 'silu':
        with localcontext() as ctx:
            ctx.prec = PREC
            # Decimal.exp is correctly rounded; adjacent decimals enclose exp.
            y = Decimal.from_float(h).copy_negate().exp()
            lo, hi = y.next_minus(), y.next_plus()
    else:
        raise ValueError('unknown witness function')
    rounded = float(lo)
    require(rounded.hex() == float(hi).hex() and math.isfinite(rounded), 'unresolved binary64 rounding')
    require(max(abs(F(lo)-F(rounded)), abs(F(hi)-F(rounded))) <= F(math.ulp(rounded)), 'outside one binary64 ulp')
    return {'lo': str(lo), 'hi': str(hi), 'binary64': rounded.hex()}


SSA = r'%[A-Za-z_][A-Za-z_0-9]*'
NUM = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'


def recognize_region(lines, constants):
    """Independent expression tree for the public fixture grammar, not SSA names."""
    head = re.fullmatch(rf'\^bb0\(({SSA}): f32, ({SSA}): f32\):', lines[0])
    require(head and head[1] != head[2], 'invalid unary region arguments')
    env = dict(constants, **{head[1]:'x'})
    def expr(op, *args):
        if op in ('arith.addf','arith.mulf'):
            args = sorted(args, key=repr)
        return (op, *args)
    for line in lines[1:-1]:
        m = re.fullmatch(rf'({SSA}) = (math\.erf|math\.exp|arith\.(?:negf|addf|divf|mulf)) ({SSA})(?:, ({SSA}))? : f32', line)
        require(m and m[1] not in env, 'unrecognized scalar operation: '+line)
        args = [env[t] for t in (m[3],m[4]) if t]
        require(len(args)==(1 if m[2] in ('math.erf','math.exp','arith.negf') else 2), 'invalid scalar arity')
        env[m[1]] = expr(m[2],*args)
    ret = re.fullmatch(rf'linalg.yield ({SSA}) : f32',lines[-1]); require(ret, 'missing scalar return')
    x = 'x'
    silu = expr('arith.mulf',x,expr('arith.divf',1.,expr('arith.addf',1.,expr('math.exp',expr('arith.negf',x)))))
    cdf = expr('arith.addf',1.,expr('math.erf',expr('arith.mulf',x,float.fromhex('0x1.6a09e6p-1'))))
    gelus = [expr('arith.mulf',a,expr('arith.mulf',b,c)) for a,b,c in itertools.permutations((x,0.5,cdf))]
    y = env[ret[1]]
    if y == silu: return 'silu'
    if y in gelus: return 'gelu'
    raise ValueError('unknown source activation; erf exact required')


class Source:
    """Fail-closed reader for the two C1 test sources (not a general MLIR frontend)."""
    def __init__(self, text):
        lines = iter(s.strip() for s in text.splitlines() if s.strip())
        require(next(lines) == '#id = affine_map<(i) -> (i)>' and next(lines) == 'module {', 'expected static module')
        header = re.fullmatch(rf'func.func @\w+\(({SSA}): tensor<(\d+)xf32> \{{secret.secret\}}\) -> tensor<\2xf32> \{{',next(lines))
        require(header and 0 < int(header[2]) <= 64, 'invalid or unsupported source ABI')
        x, self.width = header[1], int(header[2]); self.x = x
        v, w = f'tensor<{self.width}xf32>', f'tensor<{self.width}x{self.width}xf32>'
        scalars, matrices, empties, zeros, defined = {}, {}, set(), set(), {x}
        self.nodes, topo = {}, {x:'x'}
        for line in lines:
            ret = re.fullmatch(rf'return ({SSA}) : {v}', line)
            if ret:
                self.sink = ret[1]
                require(list(lines)==['}','}'], 'unexpected operations after return')
                break
            op = re.fullmatch(rf'({SSA}) = (.*)',line); require(op, 'unsupported source statement: '+line)
            dst, rhs = op.groups(); require(dst not in defined,'duplicate SSA definition'); defined.add(dst)
            if m := re.fullmatch(rf'arith.constant ({NUM}) : f32',rhs):
                scalars[dst] = f32(float(m[1]))
            elif m := re.fullmatch(rf'arith.constant dense<(\[.*\])> : {w}',rhs):
                mat = [[f32(float(e)) for e in row] for row in json.loads(m[1])]
                require(len(mat)==self.width and all(len(r)==self.width and all(map(math.isfinite,r)) for r in mat),'invalid matrix')
                matrices[dst] = mat
            elif rhs == f'tensor.empty() : {v}':
                empties.add(dst)
            elif m := re.fullmatch(rf'linalg.fill ins\(({SSA}) : f32\) outs\(({SSA}) : {v}\) -> {v}',rhs):
                require(scalars[m[1]]==0 and m[2] in empties,'matvec needs zero init'); zeros.add(dst)
            elif m := re.fullmatch(rf'linalg.matvec ins\(({SSA}), ({SSA}) : {w}, {v}\) outs\(({SSA}) : {v}\) -> {v}',rhs):
                require(m[3] in zeros,'nonzero matvec init'); self.nodes[dst]=('linear',m[2],matrices[m[1]]); topo[dst]=('linear',topo[m[2]])
            elif m := re.fullmatch(rf'arith.addf ({SSA}), ({SSA}) : {v}',rhs):
                self.nodes[dst]=('add',m[1],m[2]); topo[dst]=('add',topo[m[1]],topo[m[2]])
            elif m := re.fullmatch(rf'linalg.generic \{{indexing_maps = \[#id, #id\], iterator_types = \["parallel"\], domain_lower = -1.0 : f64, domain_upper = 1.0 : f64\}} ins\(({SSA}) : {v}\) outs\(({SSA}) : {v}\) \{{',rhs):
                require(m[2] in empties,'invalid elementwise output'); body=[]
                for row in lines:
                    if row == f'}} -> {v}': break
                    body.append(row)
                else: raise ValueError('unterminated activation')
                kind = recognize_region(body,scalars); self.nodes[dst]=(kind,m[1]); topo[dst]=(kind,topo[m[1]])
            else: raise ValueError('unsupported C1 oracle operation: '+rhs)
        else: raise ValueError('missing source return')
        h=('linear','x'); s=('silu',h); g=('linear',('gelu',h))
        mixed = [('linear',('gelu',('linear',('add',*args)))) for args in ((s,h),(h,s))]
        result=topo[self.sink]
        require((result==g and len(self.nodes)==3) or (result in mixed and len(self.nodes)==6), 'unknown C1 graph topology')
        self.case = 'single' if result==g else 'mixed'

    def evaluate(self, x, atoms):
        values={self.x:x}
        for dst,(kind,arg,*rest) in self.nodes.items():
            h=values[arg]
            if kind=='linear':
                # Python 3.12 changed sum(float); left-fold explicitly, no FMA.
                y=[]
                for row in rest[0]:
                    total=0.
                    for a,b in zip(row,h,strict=True): total = total + a*b
                    y.append(total)
            elif kind=='add': y=[a+b for a,b in zip(h,values[rest[0]],strict=True)]
            else:
                y=[]
                for h0 in h:
                    hx=h0.hex(); cert=atom(kind,hx); atoms[kind][hx]=cert; ref=float.fromhex(cert['binary64'])
                    y.append(h0/(1.+ref) if kind=='silu' else (0.5*h0)*(1.+ref))
            values[dst]=y
        return values[self.sink]


def cases(width):
    rng=random.Random(314159)
    rows=[[0.]*width, [1.]*width, [-1.]*width, [(-1.)**j for j in range(width)]]
    rows += [[v if i==j else 0. for j in range(width)] for i in range(width) for v in (-1.,1.)]
    rows += [[f32(rng.uniform(-1,1)) for _ in range(width)] for _ in range(64-len(rows))]
    return rows


def evidence(source):
    graph=Source(source.read_text(encoding='utf-8')); atoms={'gelu':{},'silu':{}}
    corpus=[{'trial':i,'x':x,'truth':graph.evaluate(x,atoms)} for i,x in enumerate(cases(graph.width))]
    return corpus, dict(schema='snabf.c1.transcendental.v1', source_sha256=sha(source), graph=graph.case,
                        decimal_digits=PREC, erf_hidden_over_sqrt2=atoms['gelu'], exp_negative_hidden=atoms['silu'])


def verify_public(d):
    corpus, w = evidence(d/'source.mlir')
    # Canonical JSON also distinguishes signed zeros. Recompute every public
    # endpoint: changing a stored result, coefficient, or coverage cannot pass.
    def exact(a,b): return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)
    require(exact(corpus,load(d/'corpus.json')) and exact([c['x'] for c in corpus],load(d/'inputs.json')), 'source corpus mismatch')
    require(exact(dict(w,corpus_sha256=sha(d/'corpus.json')),load(d/'witness.json')), 'invalid or incomplete transcendental witness')
    return corpus,w
