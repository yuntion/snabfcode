"""Deterministic local CKKS contracts. Inputs come from the source-bound SSA walk,
never from the proposed noise audit. This verifier does not assign a joint law.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
import hashlib
import math
import struct

from exact_scale_replay import encoding_scales, need, pow2, rn


@dataclass(frozen=True)
class DCEncoding:
    coefficient: int
    delta: Q
    word_upper: int | None


@dataclass(frozen=True)
class Diagonal:
    proof: dict
    peak: int


def diagonal(values, call, primes):
    slots, n = call['slots'], call['ring_n']
    if n > 65536 or slots > min(32768, n//2):
        return 'vector_ifft_not_proved'
    # Find the shortest *bitwise* tile, independently of the C++ half-fold.
    words = struct.pack('>'+str(len(values))+'d', *values)
    p = next((1 << k for k in range(1, 7) if 1 << k <= len(values) and
              words == words[:8 << k] * (len(values) >> k)), None)
    if not p:
        return 'vector_ifft_not_proved'
    w = max(map(abs, map(Q, values[:p])))
    e = Q(call['realized_eff_scale'])
    if not pow2(-300) <= w <= pow2(60) or not pow2(-300) <= e <= pow2(300):
        return 'unsupported_diagonal_range'
    err = w*(p.bit_length()-1)/2**44
    if e*(w+err) >= 2**48:
        return 'unsupported_diagonal_fixed_point_range'
    rounding = Q(9,16)
    c = math.floor(e*(w+err) + rounding)
    if 2*c >= math.prod(primes):
        return 'plaintext_centered_lift_unproved'
    if any(q >= 2**60 for q in primes):
        return 'ntt_modulus_outside_proved_range'
    stages = (2*slots).bit_length()-1
    peak = max(max(q,c) + 2*q*stages for q in primes)
    if peak >= 2**64:
        return 'unproved_ntt_word_range'
    return Diagonal(dict(requires_backend='epochs.RequireDiagonalEncodingV1', period=p, coefficient_stride=n//(2*p),
                         weight_abs=str(w), ifft_coeff_error=str(err), centered_coeff_abs=str(c),
                         slot_delta_inf=str(2*p*(e*err+rounding))), peak)


def constant(values, call, q, *, mask=False):
    """A source-bound bitwise-uniform public argument, or a scoped refusal."""
    words = b''.join(struct.pack('>d', x) for x in values)
    slots = call['slots']
    need(values and call['dynamic_input'] is False and call['payload_count'] == len(values) and
         slots > 0 and slots % len(values) == 0 and not slots & (slots-1) and
         call['payload_sha256'] == hashlib.sha256(words).hexdigest() and
         call['argument_sha256'] == hashlib.sha256(words * (slots // len(values))).hexdigest(),
         'different bound public encoding argument')
    need(tuple(q) == tuple(call['q'][:call['level']+1]) and call['is_batched'] is True and
         call['arg_type'] == '[]float64' and call['arg_precision_bits'] == 53 and
         tuple(map(str, encoding_scales(Q(call['requested_scale']), call['encoder_prec_bits']))) ==
         (call['realized_meta_scale'], call['realized_eff_scale']), 'different bound basis, branch or physical scale')
    if call['branch'] != 'embed_double_f64' or call['encoder_prec_bits'] != 53 or call['slots'] < 16:
        return 'unsupported_encoding_branch_or_slots'
    if words != words[:8] * len(values):
        return diagonal(values, call, q) if mask else 'vector_ifft_not_proved'
    w = Q(values[0]); product = abs(w) * Q(call['realized_eff_scale'])
    if w and (not pow2(-512) <= abs(w) <= pow2(100) or product < pow2(-512)) or product >= pow2(52):
        return 'unsupported_fixed_point_range'
    # On constant real tiles, the inverse FFT sends equal halves to (2w,0)
    # recursively, and its power-of-two normalization recovers w exactly.
    # CRT uses trunc(value+0.5), not ties-to-even on the final integer.
    h = Q(1, 2)
    c = int(rn((rn(product, 53) if product else 0) + h, 53))
    if c != int(rn(product + h, 53)):
        return 'ambiguous_fused_fixed_point'
    upper = None
    for p in q:
        if p >= 2**60:
            return 'ntt_modulus_outside_proved_range'
        raw = (0 if c == p else p - c % p) if w < 0 else c
        if raw >= p:
            # Bound the unmodified CRT word through each forward butterfly;
            # do not replace raw by raw % p at entry. The final Barrett reducer
            # accepts a single machine word, not an unbounded integer.
            size = 2 * slots
            while size > 1:
                raw += 2 * p
                if raw.bit_length() > 64:
                    return 'unproved_ntt_word_range'
                size //= 2
            upper = max(upper or 0, raw)
    if 2*c >= math.prod(q):
        return 'plaintext_centered_lift_unproved'
    c = -c if w < 0 else c
    return DCEncoding(c, c - w * Q(call['realized_eff_scale']), upper)


def centered_linear(q, parents, gains, bounds):
    # Sum signed gains of the SAME SSA value before applying triangle. Distinct
    # values use triangle even if they share a secret key. No stochastic model.
    combined = {}
    for name, gain in zip(parents, gains, strict=True):
        combined[name] = combined.get(name, 0) + gain
    candidate = sum(abs(gain)*bounds[name] for name, gain in combined.items())
    return candidate, 2*candidate < q


def transfers(trace, program, physical, *, guarded_diagonals=False):
    calls = {c['site_id']: c for c in trace.calls}
    instructions = {e['ID']: e for e in program['instructions']}
    edges = {e['node_id']: e for e in physical}
    bounds, dc, rows = {}, {}, []
    for edge in trace.edges:
        node, op, parents = (edge[k] for k in ('node_id', 'op', 'inputs'))
        primes = trace.d['q'][:edge['level']+1]
        modulus = math.prod(primes)
        bounds[node] = (modulus - 1)//2  # unconditional bound on an arbitrary centered residue

        def site(role):
            r = dict.fromkeys(('candidate_coeff_abs', 'formula_ref', 'blocker', 'lhs_gain', 'rhs_gain',
                               'argument_sha256', 'encoded_constant', 'sigma_pt_inf', 'delta_pt_signed', 'local_bias_units',
                               'dc_before_center', 'dc_after_center', 'dc_center_correction', 'noncanonical_ntt_word_upper'))
            r.update(site_id=f'node/{node}/{role}', inputs=parents, active_q_product=str(modulus))
            rows.append(r)
            return r

        def lift(r, candidate, exact_dc=None):
            if exact_dc is None and not candidate:
                exact_dc = 0
            if exact_dc is not None:
                candidate = abs(exact_dc)
                residue = exact_dc % modulus
                if 2 * residue > modulus:
                    residue -= modulus
                dc[node] = residue
                correction = residue - exact_dc
                bounds[node] = abs(residue)
                r.update(candidate_coeff_abs=str(candidate), dc_before_center=str(exact_dc), dc_after_center=str(residue),
                         dc_center_correction=str(correction), local_bias_units=str(abs(correction)),
                         formula_ref='b2i/exact-dc-recenter-v1' if correction else 'b2i/centered-linear-lift-v1')
                return
            r['candidate_coeff_abs'] = str(candidate)
            if 2*candidate < modulus:
                bounds[node] = candidate
                r.update(formula_ref='b2i/centered-linear-lift-v1', local_bias_units='0')
            else:
                r['blocker'] = 'need_tighter_centered_decryption_coefficients'

        if op in ('add', 'sub', 'neg', 'drop'):
            r = site('level_reduce' if op == 'drop' else op)
            if op in ('add', 'sub'):
                a, b = (int(edges[node]['multiplier_'+s]) for s in 'ab')
                gains = [a, b if op == 'add' else -b]
                r['rhs_gain'] = str(gains[1])
            else:
                gains = [-1 if op == 'neg' else 1]
            r['lhs_gain'] = str(gains[0])
            before = sum(dc[p]*k for p,k in zip(parents, gains, strict=True)) if all(p in dc for p in parents) else None
            candidate, _ = centered_linear(modulus, parents, gains, bounds)
            lift(r, candidate, before)
        elif op in ('mask', 'coefficient', 'add_plain'):
            add = op == 'add_plain'; p, = parents
            enc = site('plain_encoding_at_output')
            call = calls[node]
            need(call['call_id'] == 'encode:'+node and call['operand_index'] == 1 and call['slots'] == program['slots'] and
                 call['level'] == edge['level'] and tuple(call['q']) == tuple(trace.d['q']), 'different bound encoding use')
            enc['argument_sha256'] = call['argument_sha256']
            result = constant(instructions[node]['values'], call, primes, mask=guarded_diagonals and op == 'mask')
            r = site('add_plain' if add else 'mul_plain')
            if isinstance(result, str):
                enc['blocker'] = result
                r['blocker'] = 'unproved_plaintext_polynomial'
                continue
            if isinstance(result, Diagonal):
                v = result.proof
                bias = Q(edges[p]['physical_scale_exact']) * Q(edges[p]['ideal_abs_bound']) * Q(v['slot_delta_inf'])
                enc.update(periodic_diagonal=v, noncanonical_ntt_word_upper=str(result.peak),
                           formula_ref='b2i/guarded-periodic-real-diagonal-v1', local_bias_units=str(bias))
                candidate = bounds[p] * 2*v['period'] * int(v['centered_coeff_abs'])
                r['candidate_coeff_abs'] = str(candidate)
                if 2*candidate < modulus:
                    bounds[node] = candidate
                    r['blocker'] = 'unproved_vector_parent_error'
                else:
                    r['blocker'] = 'need_tighter_centered_decryption_coefficients'
                continue
            c, delta = result.coefficient, result.delta
            if add:
                gain, plain_gain = (int(edges[node]['multiplier_'+s]) for s in 'ab')
                bias = plain_gain * abs(delta)
                candidate = gain*bounds[p] + plain_gain*abs(c)
                before = gain*dc[p] + plain_gain*c if p in dc else None
                r['rhs_gain'] = str(plain_gain)
            else:
                gain = c  # includes delta*e; never treat this as fresh independent noise
                a, b = (Q(edges[p][key]) for key in ('physical_scale_exact', 'ideal_abs_bound'))
                bias = a*b*abs(delta)  # metadata mismatch is already charged by replay_noise_audit
                candidate = abs(c)*bounds[p]
                before = c*dc[p] if p in dc else None
            word = result.word_upper
            enc.update(encoded_constant=str(c), delta_pt_signed=str(delta), sigma_pt_inf=str(abs(c)),
                       noncanonical_ntt_word_upper=None if word is None else str(word),
                       formula_ref='b2i/constant-real-tile-v6.2.0' if word is None else 'b2i/dc-word-bounded-ntt-v6.2.0',
                       lhs_gain=str(gain), local_bias_units=str(bias))
            r['lhs_gain'] = str(gain)
            lift(r, candidate, before)
    return dict(schema='snabf.b2i.local.v2', transfers=rows)


def refusal(sink_scale, n, bits):
    """Independent single-sink production projection; allocation is not a measured margin."""
    need(type(bits) is int and 0 <= bits <= 16384, 'invalid sink precision')
    def lower(q):
        x = float(q)
        return math.nextafter(x, 0) if Q(x) > q else x
    s = Q(sink_scale); tail = 2.**-41
    return dict(schema='snabf.noise_cert.v1', precision_bits=bits, target_failure_prob=2.**-40,
                sum_delta_sinks=tail, total_delta_valid=None, common_event=None, common_event_proof=None,
                joint_independence_proof=None, validity_events=[], conservative_holder_terms=[], merged_same_source_terms=[],
                all_contracts_present=False, event_budget_passed=False, all_sinks_margin_passed=False,
                probabilistic_cert_issued=False, errors=[], sinks=[dict(sink_id='sink0', ring_n=n,
                physical_scale_exact=sink_scale, actual_physical_scale=lower(s), required_tolerance_units=lower(s*pow2(-bits)),
                allocated_delta_sink=tail, certified_bound_units=None, deterministic_bias_b=None,
                total_variance_proxy_v=None, margin_passed=False)])
