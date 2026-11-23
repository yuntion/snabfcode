"""Equation (15) diagnostics. No fitted noise, inferred law or issuance path.

Single-charge projections are counterfactual: all other charges, including
reference/decode, are omitted. They prioritize proofs, not measured errors or
lower bounds on accuracy. Exact rationals drive the walk; decimals are display.
"""
import argparse
import json
import math
from fractions import Fraction as F
from pathlib import Path

from replay_noise_audit import checked_inputs, independent, need


def dry_run(d, comparison, *, deep=False):
    flags = {'common_support': True} if deep else {}
    result = independent(d/'checked.mlir', d/'program.json', comparison, d/'audit.json',
                         source_witness=d/'source.mlir', basis=d/'basis.json', b2i=True,
                         guarded_diagonals=True, mul_rescale=True, **flags)
    trace = result['exact_scale']; nodes = trace['edges']; last = nodes[-1]
    checked = checked_inputs(d/'checked.mlir')[0]
    need(checked['edges'][checked['sink']]['ID'] == last['node_id'] and last['op']=='rescale',
         'diagnostic requires a single terminal rescale')
    n, qs, ps = trace['ring_n'], trace['ordered_q'], trace['ordered_p']
    need(len(ps) == 1 and n > 1 and not n & (n-1), 'unsupported diagnostic basis')
    p = ps[0]; sink_scale = F(result['sink_scale']); tol = sink_scale/4096
    allocations = result['refusal']; sinks = allocations['sinks']
    need(len(sinks)==1 and sinks[0]['physical_scale_exact']==str(sink_scale), 'unbound diagnostic sink')
    sink = sinks[0]
    delta = F(sink['allocated_delta_sink']); limit = F(allocations['target_failure_prob'])
    by = {e['node_id']: e for e in nodes}; rows = {r['site_id']:r for r in result['b2i_local']['transfers']}
    calls = {c['site_id']:c for c in trace['calls']}
    # Input data for diagnostic attribution are source-bound, never proposed audit values.
    key_sites = {e['node_id']:e for e in result['obligations'] if e['obligation_kind']=='gadget_trunc'}
    emitted = {e['ID']:e for e in checked['edges']}
    charges = {}; gains = {}; products = {}
    def put(node, role, amount, status):
        charges[(node,role)] = (F(amount),status)
    for e in nodes:
        node, op, args, l = (e[k] for k in ('node_id','op','inputs','level'))
        def row(role): return rows.get('node/'+node+'/'+role,{})
        def status(role): return 'proved-local' if row(role).get('formula_ref') else 'candidate-if-lift-proved'
        need(op in {'source','baby','giant','relin','rescale','mask','coefficient','add_plain','mul','add','sub','drop','neg'}, 'unmodelled diagnostic operation: '+op)
        if op == 'source':
            support = row('encrypt')
            put(node,'fresh-input-support',support['centered_support']['error'] if deep else
                support['fresh_degree_one']['slot_error_abs'],'proved-support-only')
        def charge(role, fallback):
            return row(role)['local_bias_units'] if deep and row(role).get('centered_support') and row(role).get('formula_ref') else fallback
        if op in ('baby','giant','relin'):
            k = emitted[node]['K']; degree = 2 if op=='relin' else 1
            need(k in (0,1) and (node in key_sites)==bool(k),'unsupported/truncated-key binding')
            # Center_D(c_j)*s^j, no independence, and native singleton-P rows.
            if k: put(node,'gadget_trunc',charge('gadget_trunc',n**(degree+1)*(qs[0]//2)),status('gadget_trunc'))
            switch=(19*n*sum(qs[k:l+1])+(n+1)*(p//2))//p
            put(node,'evk_switch',charge('evk_switch',n*switch),status('evk_switch'))
        elif op=='rescale':
            q, = qs[l+1:by[args[0]]['level']+1]; gains[node] = F(1,q)
            put(node,'rescale_round',charge('rescale_round',F(n*(n+1)*(q//2),q)),status('rescale_round'))
        elif op in ('mask','coefficient','add_plain'):
            r = row('plain_encoding_at_output'); diag = r.get('periodic_diagonal')
            need(r.get('formula_ref') and r['local_bias_units'] is not None,'unproved public argument')
            put(node,'plain_encoding_at_output',r['local_bias_units'],'proved-local')
            gains[node] = (min(2*diag['period']*int(diag['centered_coeff_abs']),
                               F(calls[node]['realized_eff_scale'])*F(diag['weight_abs'])+F(diag['slot_delta_inf']))
                           if diag else abs(int(r['encoded_constant'])))
        elif op=='mul':
            products[node] = [F(by[a]['metadata'])*F(by[a]['abs_bound']) for a in args]
        # Reconstruct exact B1 metadata-reference residuals, NOT trace.reference_gain.
        m=F(e['metadata']); drift=F(0)
        if op=='source': drift=abs(F(calls[node]['realized_eff_scale'])-m)*F(e['abs_bound'])
        elif op in ('mul','mask','coefficient'):
            a=F(by[args[0]]['metadata']); b=F(by[args[1]]['metadata']) if op=='mul' else F(calls[node]['realized_eff_scale'])
            drift=abs(a*b-m)*F(e['abs_bound'])
        elif op=='rescale': drift=abs(F(by[args[0]]['metadata'])*gains[node]-m)*F(e['abs_bound'])
        elif op in ('add','sub','add_plain'):
            j=e['join']; a=by[args[0]]; ma,mb=j['lhs_int_mul'],j['rhs_int_mul']
            rhs=by[args[1]] if op!='add_plain' else None
            ib=F(rhs['abs_bound']) if rhs else F(e['abs_bound'])-F(a['abs_bound'])
            sb=F(rhs['metadata']) if rhs else F(calls[node]['realized_eff_scale'])
            drift=abs(ma*F(a['metadata'])-m)*F(a['abs_bound'])+abs(mb*sb-m)*ib
        if drift: put(node,'physical_scale_drift',drift,'proved-local')

    def walk(only=None):
        error, cross = {}, {}
        for e in nodes:
            node,op,args=(e[k] for k in ('node_id','op','inputs'))
            b=error[args[0]] if args else F(0)
            if op=='mul':
                v,w=products[node]; c=error[args[1]]; cross[node]=b*c; b=v*c+w*b+b*c
            elif op in ('mask','coefficient','rescale'): b*=gains[node]
            elif op in ('add','sub','add_plain'):
                j=e['join']; ma,mb=j['lhs_int_mul'],j['rhs_int_mul']*(-1 if op=='sub' else 1)
                if len(args)==2 and args[0]==args[1]: b*=abs(ma+mb)
                else:
                    b*=abs(ma)
                    if len(args)==2: b+=abs(mb)*error[args[1]]
            for site,(amount,_) in charges.items():
                if site[0]==node and (only is None or only==site): b+=amount
            error[node]=b
        return error[last['node_id']],cross
    # Only display log/sqrt/ratios as decimal; no gate uses these approximations.
    logterm=math.log(2*n/float(delta)); bias_cap=float(tol)/math.sqrt(2)
    ranked=[]
    for site,(_,s) in charges.items():
        impact,_=walk(site)
        ranked.append(dict(node=site[0],term=site[1],status=s,isolated_sink_units=str(impact),
                           one_charge_ratio_display=float(impact)/bias_cap))
    ranked.sort(key=lambda r:F(r['isolated_sink_units']),reverse=True)
    arithmetic,cross=walk()
    if deep:
        need(rows['node/'+last['node_id']+'/rescale_round'].get('centered_support',{}).get('error')==str(arithmetic),
             'aggregate attribution differs from the independently reconstructed sink support')
    return dict(schema='snabf.eq15.diagnostic.v1', graph_sha=result['seal'], profile='common-support' if deep else 'fc3ba4d',
                ring_n=n, ordered_q=qs, ordered_p=ps, key=trace['key_id'],source_domain_abs=nodes[0]['abs_bound'],
                sink_ideal_abs=last['abs_bound'], sink_scale=str(sink_scale), tolerance_units=str(tol),
                missing_count=result['missing'], missing=result['missing_contract_terms'],
                required_common_event=['for each fixed bounded input: authentic fresh execution and keys',
                                       'all active centered lifts, keyed projection and rescale identities simultaneously',
                                       'conditional component bias/MGFs or uniform absolute residuals for every charged site',
                                       'reference approximation and decode against the same sink'],
                holder='Merge identical signed source tokens first. Without a joint independence proof, sigma <= sum sigma_i; V <= (sum sigma_i)^2 on that same G.',
                ct_mul='I_x*e_y+I_y*e_x+e_x*e_y (I in physical units); bounded product is an absolute charge, not sub-Gaussian. A probabilistic truncation needs thresholds and a union charge for their complement on the joint execution.',
                product_cross_units={k:str(v) for k,v in cross.items()}, conditional_product_tail=None,
                delta_sink=str(delta),remaining_validity_budget=str(limit-delta),delta_valid=None,
                eq15='sqrt(2)*(b+sqrt(2*ln(2*N/delta_sink))*sigma) <= Delta_sink/4096',
                bias_only_cap_display=bias_cap, sigma_only_cap_display=float(tol)/(2*math.sqrt(logterm)),
                isolated_charge_ranking=ranked, arithmetic_envelope_units=str(arithmetic),
                arithmetic_envelope_eq15_ratio_display=float(arithmetic)/bias_cap,
                arithmetic_would_pass_zero_variance=(2*arithmetic**2 <= tol**2),
                arithmetic_shortfall_bits_display=math.log2(float(arithmetic)/bias_cap) if arithmetic else None,
                arithmetic_scope='support-only stress, conditional on all lifts; excludes reference/decode; not an error lower bound',
                actual_b=None,actual_V=None,certified_bound=None,actual_margin_to_target=None,issued=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True); p.add_argument('--compare',type=Path,required=True)
    p.add_argument('--common-support',action='store_true'); a=p.parse_args()
    print(json.dumps(dry_run(a.fixture,a.compare,deep=a.common_support),allow_nan=False))
