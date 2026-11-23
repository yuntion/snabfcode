"""Source-derived bounded degree-one/two arithmetic, with exact rational lifts.

The former proof rows passed here are locally reconstructed, not audit inputs.
Sufficient no-wrap tests use all odd roots, including unused sparse copies.
"""
from fractions import Fraction as F
from math import prod


def reconstruct(trace, program, physical, local):
    n, qs, ps = trace.d['n'], trace.d['q'], trace.d['p']
    ops={i['ID']:i for i in program['instructions']}
    ed={e['node_id']:e for e in physical}; src=trace.edges[0]['node_id']
    stores={}; answer=[]; old={}
    for r in local['transfers']:
        _,node,role=r['site_id'].split('/');old.setdefault(node,{})[role]=r
    kap=F(46341,65536); projected=n*kap
    for e in trace.edges:
        node,op,args,l=(e[k] for k in ('node_id','op','inputs','level'))
        q=prod(qs[:l+1]); h=q//2; pe=ed[node]; ideal=F(pe['physical_scale_exact'])*F(pe['ideal_abs_bound'])
        rows=old.get(node,{}); facts=dict(key_id=trace.d['key_id'],source=src,target='fresh-v1/all-embeddings',ideal_units=str(ideal))
        def fact(**kw): facts.update({k:str(v) for k,v in kw.items()})
        def emit():
            # This is the physical obligation order, not the map's insertion order.
            order=('encode','encrypt','plain_encoding_at_output','mul_plain','add_plain','add','sub','neg','level_reduce',
                   'ct_mul_including_cross_term','permute_input_components','evk_switch','gadget_trunc','rescale_round')
            answer.extend(rows[k] for k in order if k in rows)
        if op=='source':
            r=rows.get('encrypt',{}); proof=r.get('fresh_degree_one',{})
            if 'output_coeff_abs' in proof and 'slot_error_abs' in proof:
                b=int(proof['output_coeff_abs']); pk=int(proof['pk_noise_coeff_abs'])
                if b >= h: emit();continue
                err=F(proof['slot_error_abs'])-(n-projected)*pk
                stores[node]=(1,b,ideal,err);fact(degree=1,coeff=b,error=err,real_projection_upper=kap)
                r['centered_support']=facts
            emit();continue
        if not all(a in stores for a in args): emit();continue
        deg,b,ia,ea=stores[args[0]];fact(lhs_coeff=b,lhs_error=ea,lhs_degree=deg)
        effects=[]
        def add(role,bias=0,lhs=None,rhs=None):effects.append((role,bias,lhs,rhs))
        if op in ('mul','add','sub'):
            db,c,ib,eb=stores[args[1]]
            if deg!=1 or db!=1:emit();continue
            fact(rhs_coeff=c,rhs_error=eb,rhs_degree=db)
            if op=='mul':
                raw=n*b*c;left,right,cross=ea*ib,eb*ia,ea*eb;err=left+right+cross;deg=2
                fact(left_term=left,right_term=right,cross=cross);add('ct_mul_including_cross_term',err)
            else:
                a,bb=int(pe['multiplier_a']),int(pe['multiplier_b'])*(1 if op=='add' else -1)
                if args[0]==args[1]:raw,err=abs(a+bb)*b,abs(a+bb)*ea
                else:raw,err=abs(a)*b+abs(bb)*c,abs(a)*ea+abs(bb)*eb
                add(op,0,a,bb)
        elif op in ('relin','baby','giant'):
            rel=op=='relin';k=ops[node]['K'];key=trace.d['key_id']+(':relin' if rel else ':rotation:'+str(ops[node]['rotation']))
            if deg!=(2 if rel else 1) or k not in (0,1) or (rel and k!=1) or len(ps)!=1 or pe['evk']!=key:emit();continue
            p,=ps;D=prod(qs[:k]);digit=sum(qs[k:l+1]);sw=(19*n*digit+(n+1)*(p//2))//p
            cut=n**deg*(D//2); ss=projected*sw; cs=F(n*n*(D//2),2)*(projected if rel else 1)
            raw=b+sw+cut;err=ea+ss+cs;deg=1
            fact(P=p,prefix_product=D,digit_sum=digit,switch_coeff=sw,cut_coeff=cut,evk=key,switch_slot=ss,cut_slot=cs)
            if not rel:add('permute_input_components',ea)
            add('evk_switch',ss)
            if k:add('gadget_trunc',cs)
        elif op=='rescale':
            if deg!=1 or ed[args[0]]['level']!=l+1:emit();continue
            p=qs[l+1];D=(n+1)*(p//2);res=F(p//2,p)*(projected+F(n*n,2));raw=F(b+D,p);err=ea/p+res
            fact(dropped_q=p,same_key_remainder=D,round_slot=res);add('rescale_round',res,F(1,p))
        elif op in ('neg','drop'):
            if deg!=1:emit();continue
            raw,err=b,ea;add('neg' if op=='neg' else 'level_reduce',0,-1 if op=='neg' else 1)
        elif op in ('coefficient','mask','add_plain'):
            enc=rows.get('plain_encoding_at_output',{});v=enc.get('periodic_diagonal')
            if deg!=1 or not enc.get('formula_ref') or enc['local_bias_units'] is None:emit();continue
            ebias=F(enc['local_bias_units'])
            if v and op=='mask':
                L=2*v['period']*int(v['centered_coeff_abs']);M=min(L,F(pe['effective_encoder_scale'])*F(v['weight_abs'])+F(v['slot_delta_inf']))
                raw=L*b;err=M*ea+ebias;fact(plain_l1=L,plain_slot_abs=M);add('mul_plain',M*ea)
            elif enc['encoded_constant'] is not None:
                c=int(enc['encoded_constant'])
                if op=='add_plain':
                    ma,mb=int(pe['multiplier_a']),int(pe['multiplier_b']);raw=abs(ma)*b+abs(mb*c);err=abs(ma)*ea+ebias;add('add_plain',0,ma,mb)
                else:raw=abs(c)*b;err=abs(c)*ea+ebias;add('mul_plain',0,c)
            else:emit();continue
        else:emit();continue
        err+=F(pe['local_scale_bias']); slot=ideal+err
        candidate=min(raw,slot).__floor__();fit=candidate<h
        fact(coefficient_route=raw,embedding_route=slot,coeff=candidate,error=err,degree=deg)
        if fit:stores[node]=(deg,candidate,ideal,err)
        for role,charge,lg,rg in effects:
            if role not in rows:
                r=dict.fromkeys(('argument_sha256','encoded_constant','sigma_pt_inf','delta_pt_signed','noncanonical_ntt_word_upper'))
                r.update(site_id='node/'+node+'/'+role, inputs=args, active_q_product=str(q));rows[role]=r
            r=rows[role];r.pop('fresh_degree_one',None)
            r.update(centered_support=dict(facts),candidate_coeff_abs=str(candidate),dc_before_center=None,dc_after_center=None,dc_center_correction=None,
                     lhs_gain=None if lg is None else str(lg),rhs_gain=None if rg is None else str(rg),local_bias_units=str(charge) if fit else None,
                     blocker=None if fit else 'unproved_common_centered_lift',formula_ref='b2k/centered-'+role+'-v1' if fit else None)
        emit()
    return dict(local,transfers=answer)
