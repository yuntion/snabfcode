"""Server-only guarded DirectGo regression; --reports must be a new directory."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction as F

from test_b2i_noise import DIAGONALS, ROOT, independent, load, need, run, same


def case(graph, fixture, data, native, go):
    d, out = fixture/graph, data/graph
    out.mkdir()
    ir, program = d/'checked.mlir', d/'program.json'
    def replay(path, **kw):
        args = dict(source_witness=d/'source.mlir',basis=d/'basis.json',b2i=True,guarded_diagonals=True,mul_rescale=True)
        args.update(kw)
        return independent(ir,program,path,d/'audit.json',**args)
    def audit(file, fresh=True, deny=False):
        opt = ['--snabf-b2-mul-rescale'] if fresh else []
        return run(native,[*opt,'--snabf-check-plan=b2i-local=true noise-audit='+file+(' require-noise-cert=true' if deny else ''),ir,'-o',os.devnull],out,deny)
    audit('release.json',False);audit('new.json'); code=audit('denied.json',deny=True)
    previous = replay(out/'release.json',mul_rescale=False); now=replay(out/'new.json'); denied=replay(out/'denied.json')
    audit_data=load(out/'new.json'); need(same(load(out/'denied.json'),dict(audit_data,mode='probability')), 'denial changed the proof')
    masks=DIAGONALS[graph]; first={n for n in masks if n.startswith('lin0.')}
    added={f'node/{n}/mul_plain' for n in first} | {f'node/{n}/rescale_round' for n in ('lin0.rescale','poly0.n4.rescale')}
    old,new=set(previous['missing_contract_terms']),set(now['missing_contract_terms'])
    need(old-new==added and not new-old and len(old)=={'residual':50,'single-gelu':51}[graph] and len(new)==45 and denied['matched'],'unexpected site inventory')
    def rows(obj):return {r['site_id']:r for r in obj['b2i_local']['transfers']}
    by=rows(audit_data)
    for n in masks:
        row=by[f'node/{n}/mul_plain']; m=row['fresh_degree_one']; L=int(m['plain_l1']); Q=int(row['active_q_product']); threshold=int(m['max_parent_coeff'])
        need(L>0 and 2*threshold*L<Q<=2*(threshold+1)*L and row['lhs_gain'] is None,'invalid vector threshold or signed gain')
        if n in first:
            need(int(row['candidate_coeff_abs'])==int(m['parent_coeff_abs'])*L and F(row['local_bias_units'])==F(m['plain_slot_abs'])*F(m['parent_slot_error_abs'])>0,'lost transformed parent')
        else: need(row['formula_ref'] is None and row['candidate_coeff_abs'] is None and row['blocker']=='missing_same_key_degree_one_parent_support','borrowed fresh support across activation')
    r=by['node/lin0.rescale/rescale_round']; m=r['fresh_degree_one']; q,Q=int(m['dropped_q']),int(r['active_q_product']); N=audit_data['domains'][0]['ring_n']; h=q//2
    need(int(m['same_key_remainder_abs'])==(N+1)*h and F(r['local_bias_units'])==F(N*(N+1)*h,q) and F(r['lhs_gain'])==F(1,q),'wrong same-key rescale bias')
    # Same actual primes, no secret bytes. A boundary input with s=1 wraps
    # despite a centered input and each component remainder individually <= h.
    v=(Q*q)//2; c0=-((q+1)//4); c1=v-c0
    center=lambda x,p:(x+p//2)%p-p//2
    u=(v-center(c0,q)-center(c1,q))//q
    need(u==Q//2+1 and center(u,Q)==-(Q//2) and v>int(m['max_parent_coeff']),'spurious rescale counterexample')
    # For s=1+...+X^(N-1), the last coefficient of r1*s has N positive summands.
    need(F(h+N*h,q)>F(1,2),'same-key half-unit shortcut not falsified')
    failed=[];tmp=out/'tampered.json'
    def refuse(label,fn):
        try: fn()
        except ValueError: failed.append(label);return
        raise AssertionError('accepted '+label)
    refuse('guarded-diagonal-only',lambda:replay(out/'new.json',mul_rescale=False))
    refuse('downgraded-guard',lambda:replay(out/'new.json',guarded_diagonals=False))
    refuse('old-target-as-fresh',lambda:replay(out/'release.json'))
    refuse('absent-source',lambda:replay(out/'new.json',source_witness=None))
    def tamper(label,site,key,v,sub=True):
        bad=copy.deepcopy(audit_data);r=rows(bad)[site]
        if sub:r=r['fresh_degree_one']
        r[key]=v;tmp.write_text(json.dumps(bad));refuse(label,lambda:replay(tmp))
    for key in ('target','key_id','P','source_id','pk_noise_coeff_abs','encoder_scale','input_width','plain_coeff_abs','slot_error_abs'):
        tamper('fresh/'+key,'node/input/encrypt',key,'0')
    for key in ('parent_coeff_abs','max_parent_coeff','plain_l1','plain_slot_abs','parent_slot_error_abs'):
        tamper('product/'+key,'node/lin0.mask0/mul_plain',key,'0')
    for key in ('dropped_q','input_q_product','same_key_remainder_abs','max_parent_coeff'):
        tamper('rescale/'+key,'node/lin0.rescale/rescale_round',key,'1')
    rotation=next(s for s in by if s.endswith('/evk_switch') and 'switch_delta_coeff_abs' in by[s].get('fresh_degree_one',{}))
    tamper('reused-key-as-independent',rotation,'switch_delta_coeff_abs','0')
    tamper('half-unit-rounding','node/lin0.rescale/rescale_round','local_bias_units','1/2',False)
    tamper('invented-signed-parent-gain','node/lin0.mask0/mul_plain','lhs_gain','1',False)
    deep='node/lin1.mask0/mul_plain';bad=copy.deepcopy(audit_data);br=rows(bad)[deep];br.update(formula_ref='b2j/same-key-diagonal-product-v1',blocker=None,local_bias_units='0')
    next(x for x in bad['sites'] if x['site_id']==deep)['formula_ref']=br['formula_ref'];bad['missing_contract_terms'].remove(deep)
    tmp.write_text(json.dumps(bad));refuse('coordinated-deep-wrap-waiver',lambda:replay(tmp))
    bad=copy.deepcopy(audit_data);bad.update(probabilistic_cert_issued=True,common_event='all-same-key-independent',total_delta_valid=0);bad['sinks'][0].update(margin_passed=True,certified_bound_units=0)
    tmp.write_text(json.dumps(bad));refuse('invented-event-and-margin',lambda:replay(tmp));tmp.unlink()
    need(not audit_data['probabilistic_cert_issued'] and audit_data['total_delta_valid'] is None and audit_data['sinks'][0]['certified_bound_units'] is None and 'common/validated_joint_law_and_domain' in new,'false issuance')
    run(native,['--snabf-lower-ckks=direct-go=true',ir,'-o','lower.mlir'],out)
    for opt,name in ((['--snabf-b2-mul-rescale'],'model'),([], 'previous')):
        run(native,[*opt,'--snabf-bind-lattigo=ring-go='+name+'.go audit='+name+'.json','lower.mlir','-o',os.devnull],out)
        need(same(load(out/(name+'.json')),load(program)),'emitter changed plan')
    probe=out/'b2j_fresh_test.go';shutil.copyfile(Path(__file__).parent/'testdata'/probe.name,probe)
    env=dict(os.environ,GOPROXY='off',GOSUMDB='off',GOTOOLCHAIN='local',GOWORK='off',GOTELEMETRY='off',B2J_PROGRAM=str(out/'model.json'))
    for name in ('model','previous'):
        proc=subprocess.run([go,'test','-mod=readonly','-count=1','-run','^TestB2JFresh$','-v',out/(name+'.go'),probe],cwd=ROOT,env=env,capture_output=True,text=True)
        (out/(name+'-go.log')).write_text(proc.stdout+proc.stderr)
        need(proc.returncode==0 if name=='model' else proc.returncode!=0 and 'model has no fresh entry' in proc.stdout,'unexpected Go result: '+proc.stdout[-900:]+proc.stderr[-900:])
    failed.append('compiled-old-model-same-seal')
    print(json.dumps(dict(graph=graph,previous=len(old),remaining=len(new),cleared=sorted(added),negative_cases=failed,denial_exit=code,same_key_wrap_witness=True,total_delta_valid=None,certified_bound=None,issued=False)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('fixtures','reports','native','go'):parser.add_argument('--'+k,type=Path,required=True)
    args=parser.parse_args();p=args.reports.resolve();p.mkdir(mode=0o700,exist_ok=False)
    for g in DIAGONALS:case(g,args.fixtures.resolve(),p,args.native.resolve(),args.go.resolve())
