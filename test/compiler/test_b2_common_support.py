"""Native fresh-target denials, independent reconstruction and adversarial evidence checks."""
import argparse
from collections import Counter
import copy
from fractions import Fraction as F
import json
from math import prod
import os
from pathlib import Path
import re
import subprocess

from test_b2i_noise import DIAGONALS, ROOT, independent, load, need, run, same
from noise_eq15_dry_run import dry_run


MISSING = {'node/input/encode', 'node/input/encrypt', 'common/validated_joint_law_and_domain',
           'sink/sink0/reference_approx_and_decode'}


def case(graph, fixtures, reports, native, go):
    d, out = fixtures/graph, reports/graph
    out.mkdir(mode=0o700)
    ir, program = d/'checked.mlir', d/'program.json'
    old, eng, denied = [out/(s+'.json') for s in ('previous','engineering','denied')]
    flags = ['--snabf-b2-mul-rescale','--snabf-b2-common-support']
    def audit(path, options, certify=False):
        cmd = [*options, '--snabf-check-plan=b2i-local=true noise-audit='+str(path)+
               (' require-noise-cert=true' if certify else ''), ir, '-o', os.devnull]
        return run(native,cmd,out,certify)
    def replay(path, **kw):
        opts = dict(source_witness=d/'source.mlir', basis=d/'basis.json', b2i=True, guarded_diagonals=True,
                    mul_rescale=True, common_support=True)
        checked, params = kw.pop('checked',ir), kw.pop('params',program)
        return independent(checked,params,path,d/'audit.json',**dict(opts,**kw))

    audit(old,flags[:1],True); audit(eng,flags); rc=audit(denied,flags,True)
    baseline = replay(old,common_support=False); result=replay(denied); replay(eng)
    report=load(denied); need(rc==1 and same(report,dict(load(eng),mode='probability')), 'not the same proof on refusal')
    previous=set(baseline['missing_contract_terms']); remaining=set(result['missing_contract_terms']); closed=previous-remaining
    ledger=result['obligations']; total={'residual':59,'single-gelu':62}[graph]
    need(len(previous)==45 and remaining==MISSING and previous>=remaining and len(ledger)==total and
         sum(x['status']=='proved_local' for x in ledger)==total-4,'obligation inventory drift')
    rows=lambda r: {t['site_id']:t for t in r['b2i_local']['transfers']}
    by=rows(report)
    need(all(f'node/{n}/mul_plain' in closed for n in DIAGONALS[graph] if n.startswith('lin1.')) and
         len([s for s in closed if s.endswith('/rescale_round')])==5, 'missing deep diagonal or one-prime rescale')
    expected=dict(add=6 if graph=='residual' else 7,add_plain=1,ct_mul_including_cross_term=3,evk_switch=8 if graph=='residual' else 7,
                  gadget_trunc=4 if graph=='residual' else 5,level_reduce=4 if graph=='residual' else 3,
                  mul_plain=5 if graph=='residual' else 6,permute_input_components=5 if graph=='residual' else 4,rescale_round=5)
    need(Counter(s.rsplit('/',1)[1] for s in closed)==expected,'wrong releases by primitive')
    trace=result['exact_scale']; n,qs,ps=trace['ring_n'],trace['ordered_q'],trace['ordered_p']
    st={}; embedding_only=[]; embedding_tighter=[]
    for e in trace['edges']:
        node,op,args,l=(e[k] for k in ('node_id','op','inputs','level'))
        receipts=[r for r in by.values() if r['site_id'].rsplit('/',1)[0]=='node/'+node and 'centered_support' in r]
        need(bool(receipts),'missing arithmetic ancestry: '+node)
        f=receipts[0]['centered_support']; st[node]=f; q=prod(qs[:l+1]); b=int(f['coeff'])
        need(all(row['centered_support']==f and int(row['active_q_product'])==q for row in receipts) and 2*b<q-1,'inactive Q or rival lifts')
        if not args: continue
        need(f['lhs_coeff']==st[args[0]]['coeff'] and f['lhs_error']==st[args[0]]['error'] and
             b==int(min(F(f['embedding_route']),F(f['coefficient_route']))),'forged or stale parent')
        if int(F(f['coefficient_route']))>=q//2: embedding_only.append(node)
        if F(f['embedding_route'])<F(f['coefficient_route']): embedding_tighter.append(node)
        if op=='mul':
            need(f['rhs_error']==st[args[1]]['error'] and f['degree']=='2' and
                 F(f['cross'])==F(f['lhs_error'])*F(f['rhs_error'])>0,'cross term lost')
        if op=='relin':
            need(f['lhs_degree']=='2' and f['degree']=='1' and int(f['P'])==ps[0] and
                 int(f['cut_coeff'])==n*n*(qs[0]//2),'truncation uses wrong key power')
    need(embedding_tighter,'inverse embedding never tightened a production parent')
    before=dry_run(d,old); after=dry_run(d,denied,deep=True)
    for tag,obj in [('before',before),('after',after)]: (out/('eq15-'+tag+'.json')).write_text(json.dumps(obj)+'\n')
    ratio=after['arithmetic_envelope_eq15_ratio_display']
    need(not after['arithmetic_would_pass_zero_variance'] and 1<ratio<before['arithmetic_envelope_eq15_ratio_display'] and
         after['certified_bound'] is after['actual_V'] is after['actual_b'] is after['delta_valid'] is None and
         not report['probabilistic_cert_issued'] and not report['event_budget_passed'] and not report['sinks'][0]['margin_passed'], 'false Eq15 conclusion')

    failed=[]; proposed=out/'proposed.json'
    def reject(label,fn):
        try: fn()
        except ValueError: failed.append(label);return
        raise AssertionError('accepted '+label)
    def wrong(label, mutate):
        bad=copy.deepcopy(report); mutate(bad); proposed.write_text(json.dumps(bad));reject(label,lambda:replay(proposed))
    def site(label, role, field, value, sub=True):
        def mutate(bad):
            r=rows(bad)[role]; (r['centered_support'] if sub else r)[field]=value
        wrong(label,mutate)
    deep='node/lin1.mask0/mul_plain'; square='node/poly0.n1.mul/ct_mul_including_cross_term'
    evk='node/poly0.n1.relin/evk_switch'; cut='node/poly0.n1.relin/gadget_trunc'; rs='node/lin1.rescale/rescale_round'
    for name,role,field,val in [('wrong identity',deep,'key_id','fresh-independent'),('missing origin',deep,'source',''),
        ('only selected slots',deep,'target','active-rows-only'),('borrowed parent',deep,'lhs_coeff','0'),
        ('wrong degree',square,'degree','1'),('drop cross',square,'cross','0'),('zero linear term',square,'left_term','0'),
        ('unrelated EVK',evk,'evk','other:relin'),('swapped P',evk,'P',str(qs[0])),('omitted digit',evk,'digit_sum','0'),
        ('wrong prefix',cut,'prefix_product','1'),('linearized key square',cut,'cut_coeff',str(n*(qs[0]//2))),
        ('zero switch noise',evk,'switch_slot','0'),('zero truncation',cut,'cut_slot','0'),('recycled level',rs,'dropped_q',str(qs[-1])),
        ('forgot secret rounding',rs,'same_key_remainder','1'),('imaginary independence',rs,'round_slot','1/2'),
        ('rounded-down projection','node/input/encrypt','real_projection_upper','46340/65536')]: site(name,role,field,val)
    site('full chain in place of active Q',deep,'active_q_product',str(prod(qs)),False)
    site('fake vector signed gain',deep,'lhs_gain','1',False)
    wrong('removed parent receipt',lambda b:rows(b)[deep].pop('centered_support'))
    def counterfeit(b):
        r=rows(b)['node/input/encrypt'];r.update(formula_ref='b2k/claimed-zero',local_bias_units='0',blocker=None)
        next(s for s in b['sites'] if s['site_id']==r['site_id'])['formula_ref']=r['formula_ref'];b['missing_contract_terms'].remove(r['site_id'])
    wrong('coordinated source waiver',counterfeit)
    wrong('invented event and allocation',lambda b:b.update(common_event='independent-per-node',total_delta_valid=0,event_budget_passed=True))
    wrong('invented sink pass',lambda b:b['sinks'][0].update(deterministic_bias_b=0,total_variance_proxy_v=0,certified_bound_units=0,margin_passed=True))
    reject('old file on deep mode',lambda:replay(old));reject('new file on old mode',lambda:replay(denied,common_support=False))
    reject('no guarded target',lambda:replay(denied,mul_rescale=False));reject('no source',lambda:replay(denied,source_witness=None))
    for label,key,change in [('bad public-key profile','basis',lambda b:b.update(encryptionType='sk')),
            ('wrong auxiliary prime','basis',lambda b:b.update(P=[qs[0]])),('reordered Q','basis',lambda b:b['Q'].reverse()),
            ('missing scale receipt','params',lambda b:b['encoding_receipts']['receipts'].pop(0)),
            ('stale effective scale','params',lambda b:b['encoding_receipts']['receipts'][0].update(effective_encoder_scale='1')),
            ('changed rotation','params',lambda b:next(i for i in b['instructions'] if i['op']=='baby').update(rotation=99))]:
        obj=load(d/('basis.json' if key=='basis' else 'program.json'));change(obj);proposed.write_text(json.dumps(obj));reject(label,lambda:replay(denied,**{key:proposed}))
    cmd=[native,'--allow-unregistered-dialect','--snabf-b2-common-support',
         '--snabf-check-plan=b2i-local=true noise-audit='+str(out/'wrong-mode.json'),ir,'-o',os.devnull]
    altered=out/'changed-domain.mlir'; text,count=re.subn(r'\binput_bound = [\w.+-]+ : f64','input_bound = 2.000000e+00 : f64',ir.read_text())
    need(count==1,'unexpected declared domain count');altered.write_text(text)
    reject('changed checked domain',lambda:replay(denied,checked=altered))
    other='residual' if graph=='single-gelu' else 'single-gelu'
    reject('other original source',lambda:replay(denied,source_witness=fixtures/other/'source.mlir'))
    bad=subprocess.run(cmd,capture_output=True,text=True);need(bad.returncode and 'common support requires' in bad.stderr,'native accepted unguarded mode');failed.append('unguarded native checker')
    proposed.unlink()
    # A square of a same-key residual need not have zero mean even if the residual
    # itself is centered. A probabilistic truncation cannot discard this example.
    need(sum(x*x for x in (-1,1))==2,'invalid cross witness')
    # A concrete ternary-key R=Z[X]/(X^8+1) witness: charging just r*s
    # for a relin cut misses a coefficient even though r and s are bounded by 1.
    def conv(a,b):
        c=[0]*len(a)
        for i,ai in enumerate(a):
            for j,bj in enumerate(b): c[(i+j)%len(c)]+=ai*bj*(1 if i+j<len(c) else -1)
        return c
    secret=[1]*8; first=conv(secret,secret); second=conv(first,secret)
    need(max(map(abs,first))==8 and max(map(abs,second))>8,'false linear cutoff counterexample')
    env=dict(os.environ,B2K_PROGRAM=str(program),GOPROXY='off',GOSUMDB='off',GOTOOLCHAIN='local',GOWORK='off',GOTELEMETRY='off')
    cmd=[go,'test','-count=1','-v','-mod=readonly',ROOT/'test/compiler/testdata/b2k_switch_test.go']
    proc=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True);(out/'switch-go.log').write_text(proc.stdout+proc.stderr)
    need(proc.returncode==0,'Go same-key probe failed: '+proc.stdout[-500:]+proc.stderr[-500:])
    print(json.dumps(dict(graph=graph,obligations=total,before=45,remaining=sorted(remaining),cleared=sorted(closed),
          by_primitive=expected,embedding_only=embedding_only,embedding_tighter=embedding_tighter,negative_cases=failed,
          denial_exit=rc,arithmetic_ratio=ratio,issued=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('fixtures','reports','native','go'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(); output=a.reports.resolve();output.mkdir(mode=0o700,exist_ok=False)
    for graph in DIAGONALS:case(graph,a.fixtures.resolve(),output,a.native.resolve(),a.go.resolve())
