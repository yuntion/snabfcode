"""Lattigo v6.2.0 scale semantics. No native calls, float logs as scale truth, or noise claims."""
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import math
import struct


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pow2(k):
    return Q(1 << k) if k >= 0 else Q(1, 1 << -k)


def rn(x, precision=128):
    x = Q(x); need(x > 0, 'nonpositive scale')
    k = x.numerator.bit_length() - x.denominator.bit_length()
    need(abs(k) < 16384, 'scale outside replay range')
    k -= x < pow2(k)
    unit = pow2(k - precision + 1)
    return round(x / unit) * unit  # Fraction round: ties to even


def encoding_scales(requested, precision):
    need(type(precision) is int and 53 <= precision <= 128, 'unsupported encoder precision')
    m = rn(requested)
    if precision > 53:
        return m, m
    subnormal = pow2(-1074)
    e = rn(m, 53) if m >= pow2(-1022) else round(m / subnormal) * subnormal
    need(pow2(-1022) <= e <= (2**53-1)*pow2(971), 'non-normal effective encoding scale')
    return m, e


def upper64(x):
    v = float(x)
    if Q(v) < x:
        v = math.nextafter(v, math.inf)
    need(math.isfinite(v), 'unrepresentable audit bound')
    return struct.pack('>d', v).hex()


def prime(n):
    # Deterministic Miller-Rabin on uint64, independent of the GMP/C++ predicate.
    if n < 2 or n >= 2**64 or n % 2 == 0:
        return n == 2
    d, s = n-1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0:
            continue
        v = pow(a, d, n)
        if v in (1, n-1):
            continue
        for _ in range(s-1):
            v = v*v % n
            if v == n-1:
                break
        else:
            return False
    return True


@lru_cache(maxsize=256)
def basis(n, q, p):
    need(type(n) is int and n >= 4 and n & (n-1) == 0 and q and p and
         len(set(q+p)) == len(q)+len(p) and
         all(type(v) is int and v % (2*n) == 1 and prime(v) for v in q+p), 'invalid ordered Q/P')


def encode(context, requested, words):
    """words are original public IEEE binary64 bytes in big-endian order (before tiling)."""
    n, level, slots = (context[k] for k in ('ring_n','level','slots'))
    q, p = tuple(context['q']), tuple(context['p'])
    basis(n, q, p)
    count = len(words)//8
    need(len(words) == 8*count and context['is_batched'] is True and context['arg_type'] == '[]float64' and
         type(context['arg_precision_bits']) is int and context['arg_precision_bits'] == 53 and
         type(context['operand_index']) is int and context['operand_index'] >= 0 and
         all(isinstance(context[k], str) and context[k] for k in ('key_id','site_id','call_id')) and
         type(level) is int and 0 <= level < len(q) and type(slots) is int and 0 < slots <= n//2 and
         slots & (slots-1) == 0 and context['dynamic_input'] is (not count) and (not count or slots % count == 0) and
         all(math.isfinite(v) for v, in struct.iter_unpack('>d', words)), 'unbound encoding call/branch/payload')
    prec = context['encoder_prec_bits']
    m, e = encoding_scales(requested, prec)
    digest = lambda b: hashlib.sha256(b).hexdigest()
    return dict(context, requested_scale=str(requested), realized_meta_scale=str(m), realized_eff_scale=str(e),
                branch='embed_double_f64' if prec == 53 else 'bigfloat_value',
                quantization_rule='batched-f64' if prec == 53 else 'batched-bigfloat',
                payload_count=count, payload_sha256=digest(words),
                argument_sha256=digest(words*(slots//count)) if count else None)


def state(m, e):
    need(m == rn(m) and e > 0, 'unrepresentable metadata or invalid effective gain')
    return Q(m), Q(e)


def multiply(a, b):
    ma, ea = state(*a); mb, eb = state(*b)
    return rn(ma*mb), ea*eb


def rescale(a, q, level, dropped, default_log, *, lowering=False, mul_depth_delta=0):
    m, e = state(*a)
    need(type(default_log) is int and 1 <= default_log <= 128, 'invalid precision mode')
    count = 1 if default_log <= 64 else 2
    if lowering:
        need(count == 1 and len(dropped) <= 1, 'UNSUPPORTED_MULTIPRIME_RESCALE')
    need(type(mul_depth_delta) is int and mul_depth_delta in ((0, 1) if lowering else (0,)), 'invalid depth')
    need(type(level) is int and count <= level < len(q) and len(dropped) == count and
         list(dropped) == [q[level-j] for j in range(count)], 'wrong level or ordered dropped_q')
    trace = []
    for v in dropped:
        need(type(v) is int and v > 1, 'invalid Q')
        m, e = rn(m/v), e/v
        trace.append(dict(q=v, metadata=str(m), effective=str(e)))
    return dict(in_level=level, out_level=level-count, levels_consumed=count, mul_depth_delta=mul_depth_delta,
                dropped_q=list(dropped), meta_before=str(a[0]), eff_before=str(a[1]),
                meta_after=str(m), eff_after=str(e), divisions=trace)


def join(op, a, b, ba, bb, *, upper_endpoints=True):
    ma, ea = state(*a); mb, eb = state(*b)
    need(op in ('add', 'sub') and all(v is None or v >= 0 for v in (ba, bb)), 'invalid join or magnitude')
    s = max(ma, mb); g = int(rn(s/min(ma, mb)))
    need(1 <= g <= 1 << 50, 'unsupported add integer')
    ga, gb = (g, 1) if ma < mb else (1, g)
    ea *= ga; eb *= gb
    left = ea-s; right = (s-eb) if op == 'sub' else (eb-s)
    miss = ba is None or bb is None
    enc = None if miss else abs(left)*ba + abs(right)*bb
    dec = None if miss else enc/s
    return dict(out_meta_scale=str(s), lhs_int_mul=ga, rhs_int_mul=gb,
                common_effective_scale=str(ea) if ea == eb else None,
                lhs_signed_effect=str(left), rhs_signed_effect=str(right),
                abs_lhs_residual=str(abs(left)), abs_rhs_residual=str(abs(right)),
                lhs_bound=None if ba is None else str(ba), rhs_bound=None if bb is None else str(bb),
                missing_operand_bounds=miss, encoded_bias=None if miss else str(enc),
                decoded_bias=None if miss else str(dec), encoded_bias_upper=None if miss or not upper_endpoints else upper64(enc),
                decoded_bias_upper=None if miss or not upper_endpoints else upper64(dec))


def fold(ct, target, context, words, old_requested, dropped):
    state(*ct); state(*target)
    need(not context['dynamic_input'] and context['operand_index'] == 1, 'fold needs existing public plaintext')
    old = encode(context, old_requested, words)
    def step(a):
        return rescale(a, context['q'], context['level'], dropped, context['default_log_scale'], lowering=True)
    step(ct)
    requested = Q(target[1])*dropped[0]/ct[1]
    rho = requested / Q(old['realized_eff_scale']); call = encode(context, requested, words)
    m, e = (Q(call[k]) for k in ('realized_meta_scale','realized_eff_scale'))
    out = step(multiply(ct,(m,e)))
    return dict(within_search_window=Q(1,2) <= rho <= 2, rho=str(rho),
                exact_representable=m == e == requested and (Q(out['meta_after']),Q(out['eff_after'])) == target,
                residual_scale_diff=str(Q(out['eff_after'])/target[1]-1), replacement=call, rescale=out)


class Trace:
    """A second reference convention, driven by the *same* verified SSA walk.

    On unequal gains the sum has NO homogeneous E. Keep the signed local
    decomposition, rebase the reference to M, and label all descendants.
    Never sum these diagnostics with the B2.1 metadata-reference bias ledger.
    """
    def __init__(self, domain, source_sha, seal):
        self.d, self.source_sha, self.seal = domain, source_sha, seal
        self.states, self.calls, self.edges = {}, [], []

    def record(self, op, id_, level, operands, meta, eff, values, expected_m, ideal):
        parents = [x[3] for x in operands]
        states = [self.states[x] for x in parents]
        ref_only = any(x[2] for x in states)
        row = dict(node_id=id_, op=op, inputs=parents, level=level, abs_bound=str(ideal))
        if op=='source':
            m,e = meta,eff
        else:
            a = states[0][:2]; m,e = a
            if op in ('mul','coefficient','mask'):
                m,e = multiply(a, states[1][:2] if op=='mul' else (meta,eff))
            elif op=='rescale':
                l=operands[0][2]; q=self.d['q']
                row['rescale']=rs=rescale(a, q, l, q[level+1:l+1][::-1], self.d['scale'])
                m,e=Q(rs['meta_after']),Q(rs['eff_after'])
            elif op in ('add','add_plain'):
                b=states[1][:2] if op=='add' else (meta,eff)
                bb=operands[1][1] if op=='add' else max(abs(Q(v)) for v in values)
                row['join']=j=join('add',a,b,operands[0][1],bb)
                m=Q(j['out_meta_scale']); common=j['common_effective_scale']
                if common is None: ref_only=True
                e=m if common is None else Q(common)
            else:
                need(op in ('drop','relin','baby','giant'), 'unknown effective gain transfer')
        need(m==expected_m and id_ not in self.states, 'metadata replay mismatch or repeated edge')
        row.update(metadata=str(m), effective=None if ref_only else str(e), reference_gain=str(e),
                   reference_only=ref_only)
        self.states[id_]=(m,e,ref_only); self.edges.append(row)

    def public(self):
        return dict(schema='snabf.exact-scale.v1', source_sha256=self.source_sha, emit_sha256=self.seal,
                    key_id=self.d['key_id'], ring_n=self.d['n'], ordered_q=self.d['q'], ordered_p=self.d['p'],
                    calls=self.calls, edges=self.edges, probabilistic_cert_issued=False)
