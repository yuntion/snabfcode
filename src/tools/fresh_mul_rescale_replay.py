"""Reconstruct same-key support from the sealed SSA and public call arguments.

The input rows here are *locally recomputed* B2-i encodings, not the audit
under comparison. This module never reads a proposed bound, key or execution.
"""
from fractions import Fraction as F
from math import prod


def reconstruct(trace, program, edges, local):
    d = trace.d
    n, width, slots = d['n'], program['width'], program['slots']
    if (len(d['p']) != 1 or len(d['q']) > 8 or not 1024 <= n <= 65536 or
            not 1 <= width <= 64 or width & (width - 1) or slots % width or
            not 16 <= slots <= n//2 or slots & (slots-1) or d['scale'] > 64 or max(d['q'] + d['p']) >= 1 << 50):
        return local
    p, = d['p']
    physical = {e['node_id']: e for e in edges}
    calls = {c['site_id']: c for c in trace.calls}
    ops = {e['ID']: e for e in program['instructions']}
    by_node = {}
    for row in local['transfers']:
        by_node.setdefault(row['site_id'].rsplit('/', 1)[0], []).append(row)
    rebuilt, coeff, error = [], {}, {}
    src = trace.edges[0]['node_id']
    for edge in trace.edges:
        node, op, parents, level = (edge[k] for k in ('node_id','op','inputs','level'))
        prefix = 'node/' + node
        rows = by_node.get(prefix, [])
        rebuilt.extend(rows)
        primes = d['q'][:level+1]
        modulus = prod(primes)
        def site(role):
            ident = prefix + '/' + role
            r = next((x for x in rows if x['site_id'] == ident), None)
            if r is None:
                r = dict.fromkeys(('candidate_coeff_abs','formula_ref','blocker','lhs_gain','rhs_gain',
                                   'argument_sha256','encoded_constant','sigma_pt_inf','delta_pt_signed','local_bias_units',
                                   'dc_before_center','dc_after_center','dc_center_correction','noncanonical_ntt_word_upper'))
                r.update(site_id=ident, inputs=parents, active_q_product=str(modulus)); rebuilt.append(r)
            m = dict(target='epochs.RequireFreshMulRescaleV1/EvalFresh', key_id=d['key_id'], P=str(p), source_id=src)
            r['fresh_degree_one'] = m
            return r, m
        e = physical[node]
        if op == 'source':
            r, m = site('encrypt')
            r['blocker'] = 'fresh_pk_support_is_not_a_conditional_law'
            c = calls[node]; E = F(c['realized_eff_scale']); A = F(e['physical_scale_exact'])
            if not (node == src and F(e['ideal_abs_bound']) == 1 and level == len(d['q'])-1 and
                    c['call_id'] == 'encode:'+node and c['level'] == level and c['key_id'] == d['key_id'] and
                    c['q'] == d['q'] and c['p'] == d['p'] and c['ring_n'] == n and c['slots'] == slots and
                    c['dynamic_input'] is True and c['operand_index'] == 0 and c['argument_sha256'] is None and
                    c['payload_count'] == 0 and c['branch'] == 'embed_double_f64' and c['arg_type'] == '[]float64' and
                    c['is_batched'] is True and c['arg_precision_bits'] == c['encoder_prec_bits'] == 53 and
                    F(c['realized_meta_scale']) == A and E == F(e['effective_encoder_scale']) and
                    1 <= c['default_log_scale'] < 47 and F(c['requested_scale']) == 2**c['default_log_scale'] == E == A):
                continue
            quant = F(9,16) + E*(width.bit_length()-1)/2**44
            plain = int(E + quant)
            # e_pk*u + e0 + e1*s minus the two centered ModDown remainders.
            key = (19*(2*n+1) + (1+n)*(p//2)) // p
            m.update(input_width=str(width), encoder_scale=str(E), plain_coeff_abs=str(plain), pk_noise_coeff_abs=str(key))
            if 2*(plain+key) < modulus:
                coeff[node] = plain+key; error[node] = 2*width*quant+n*key+abs(E-A)
                m.update(output_coeff_abs=str(coeff[node]), slot_error_abs=str(error[node]))
            continue
        a = parents[0]
        B = coeff.get(a)
        if op in ('baby','giant'):
            r, m = site('evk_switch'); m['evk'] = e['evk']
            r['blocker'] = 'rotation_support_is_not_an_evk_error_contract'
            k = ops[node]['K']
            if B is None or k not in (0, 1) or e['evk'] != f"{d['key_id']}:rotation:{ops[node]['rotation']}": continue
            digits, cut = sum(primes[k:]), prod(primes[:k])
            inc = n*(cut//2) + (n*19*digits+(n+1)*(p//2))//p
            m.update(parent_coeff_abs=str(B), retained_digit_sum_abs=str(digits), prefix_product=str(cut), switch_delta_coeff_abs=str(inc))
            if 2*(B+inc) < modulus:
                coeff[node] = B+inc; m['output_coeff_abs'] = str(coeff[node])
                if a in error:
                    error[node] = error[a]+n*inc; m['slot_error_abs'] = str(error[node])
        elif op in ('add','sub') and all(v in coeff for v in parents):
            b = sum(abs(int(e['multiplier_'+s]))*coeff[v] for s,v in zip('ab', parents, strict=True))
            if b <= modulus//2: coeff[node] = b
        elif op == 'rescale':
            r, m = site('rescale_round')
            levels = physical[a]['level']
            if levels != level+1:
                r['blocker'] = 'unbound_single_prime_rescale'; continue
            q = d['q'][levels]; remainder = (q//2)*(1+n)
            m.update(dropped_q=str(q), input_q_product=str(modulus*q), same_key_remainder_abs=str(remainder),
                     max_parent_coeff=str(q*(modulus//2+1)-1-remainder))
            if B is None:
                r['blocker'] = 'missing_same_key_degree_one_parent_support'; continue
            m['parent_coeff_abs'] = str(B)
            candidate = (B+remainder)//q; r['candidate_coeff_abs'] = str(candidate)
            if candidate > modulus//2:
                r['blocker'] = 'rescaled_centered_lift_unproved'; continue
            coeff[node] = candidate
            r.update(formula_ref='b2j/same-key-one-prime-rescale-v1', lhs_gain=str(F(1,q)),local_bias_units=str(F(n*remainder,q)))
        elif op in ('coefficient','mask') and rows and rows[0]['formula_ref']:
            enc = rows[0]; diag = enc.get('periodic_diagonal')
            if not diag:
                if B is not None and enc['encoded_constant'] is not None:
                    bound = B*abs(int(enc['encoded_constant']))
                    if bound <= modulus//2: coeff[node] = bound
                continue
            l1 = 2*diag['period']*int(diag['centered_coeff_abs'])
            if not l1: continue
            r, m = site('mul_plain'); r['candidate_coeff_abs'] = None
            c = min(l1,F(calls[node]['realized_eff_scale'])*F(diag['weight_abs'])+F(diag['slot_delta_inf']))
            m.update(plain_l1=str(l1), plain_slot_abs=str(c), max_parent_coeff=str((modulus//2)//l1))
            if B is None:
                r['blocker'] = 'missing_same_key_degree_one_parent_support'; continue
            m['parent_coeff_abs'] = str(B); candidate = B*l1; r['candidate_coeff_abs'] = str(candidate)
            if candidate > modulus//2:
                r['blocker'] = 'diagonal_product_centered_lift_unproved'; continue
            naive = n*B+F(physical[a]['physical_scale_exact'])*F(physical[a]['ideal_abs_bound'])
            delta = min(error.get(a,naive),naive)
            coeff[node] = candidate; m['parent_slot_error_abs'] = str(delta)
            r.update(blocker=None, formula_ref='b2j/same-key-diagonal-product-v1', local_bias_units=str(c*delta))
    return dict(local, transfers=rebuilt)
