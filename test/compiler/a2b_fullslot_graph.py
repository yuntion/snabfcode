"""Two full-slot ciphertext trials, three engineering arms; no public Pass 4 admission."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/tools/n2'))
from search import PIN, checked_attacks, digest, measure, need, ntt_prime, request, sha, store

MODES=('keep-source','force-down','adaptive')
EPS=2**-12


def read(path):
    return json.loads(path.read_bytes())


def instances(b):
    n,m,q,t,p=(b[k] for k in ('SourceN','TargetN','SourceQ','TargetQ','P'))
    need((n,m,b['CutLevel'],b['KeyLevelQ'],b['KeyLevelP'],len(q),len(t),len(p))==(16384,8192,2,1,0,6,2,1)
         and t==q[:2] and all(ntt_prime(v,n) for v in q+p) and len(set(q+p))==7,'wrong ordered full-slot basis')
    qs,qt,aux=math.prod(q),math.prod(t),math.prod(p)
    rows={k:request('a2b.'+k,ring,chain,p,product) for k,ring,chain,product in
          [('source.Q',n,q,qs),('source.QP',n,q,qs*aux),('target.Q',m,t,qt),('target.QP',m,t,qt*aux),
           ('switch.small',m,t,qt*aux),('switch.large',n,t,qt*aux)]}
    for k in ('switch.small','switch.large'):
        rows[k]['switch_basis']=dict(q=t,p=p,physical_ring=n,embedded_secret_ring=m)
    return rows


def screen(b,rows,values):
    need(set(values)==set(rows),'incomplete estimator evidence')
    low={}
    for k,r in rows.items():
        v=values[k]; need(v['request']==r,'stale measured instance: '+k)
        low[k]=min(checked_attacks(v['attacks'],r).values())
    need(min(low.values())>=128,'128-bit ordinary/switch gate failed')
    return dict(Basis=b,Estimator=PIN,Minimum=low)


def lanes(values, length):
    need(len(values)==length and all(isinstance(pair,list) and len(pair)==2 and
         all(type(v) in (float,int) and math.isfinite(v) for v in pair) for pair in values),'invalid decoded slots')
    return [complex(*v) for v in values]


def audit(out, cache=None, extent='full'):
    basis=read(out/'basis.json'); rows=instances(basis)
    details=read(out/'security-detail.json')
    gate=screen(basis,rows,details)
    if cache is not None:
        costs=measure(rows.values(),cache,None)
        need(all(details[k]['attacks']==costs[digest(req)] for k,req in rows.items()),
             'recorded attacks differ from the trusted measurement cache')
    need(gate==read(out/'screen.json'),'screen no longer matches exact requests')
    routing=read(out/'routing.json'); kb,sb=routing['keep_bytes'],routing['split_bytes']
    need(min(kb,sb)>0 and routing['mode']=={'keep-source':'keep-source','force-down':'force-down',
         'adaptive':'force-down' if sb<kb else 'keep-source'},'unsupported marginal key-byte decision')
    need(min(gate['Minimum'][k] for k in ('switch.small','switch.large'))<512 and
         routing['switch_512_force']=='reject' and routing['switch_512_adaptive']=='keep-source','512 control')
    full,half=basis['SourceN']//2,basis['TargetN']//2
    width=half+1 if extent=='tail-one' else full
    offsets=(0,1,half,half+1,full-1); coefficients=(.25,-.5,.125,.375,-.0625)
    def row(i):
        return [((i+k)%full,coefficients[j]*(1+(i+3*j)%5)/4) for j,k in enumerate(offsets) if i<width and (i+k)%full<width]
    result={mode:[] for mode in MODES}
    artifacts=['basis.json','screen.json','security-detail.json','routing.json','pass1-request.json','pass1-answer.json']
    for tr in range(2):
        path=f'trial{tr}-input.json'; data=read(out/path); artifacts.append(path)
        x=lanes(data['x'],full); cut=lanes(data['cut'],full)
        need(data.get('width',full)==width and x==[complex(((37*i+17*tr+13*(i//half))%251-125)/256) if i<width else 0j
             for i in range(full)],'source corpus or padding mismatch')
        pre=[(x[i].real+x[(i+1)%full].real)/8 for i in range(full)]
        truth=[v/(1+math.exp(-v)) for v in pre]
        errs=[abs(v-h) for v,h in zip(cut,truth)]
        ep=[max(errs[:half]),max(errs[half:])]
        need(max(abs(cut[i]-pre[i]*(.5+pre[i]/4)) for i in range(full))<EPS,'encrypted prefix deviates from quadratic')
        for mode in MODES:
            path=f'trial{tr}-{mode}.json'; a=read(out/path); artifacts.append(path)
            need(a['selected']==routing['mode'][mode] and a['cut_sha256']==data['cut_sha256'],'unbound ciphertext or mode')
            observed=a['observed']; values=lanes(a['slots'],full)
            if a['selected']=='force-down':
                need([(b['logical_start'],b['used_slots'],b['level']) for b in observed['Split']['outputs']]==
                     [(0,half,1),(half,width-half,1)],'unbound two-block split or last-block padding')
                need((observed['CompletedBlocks'],observed['CrossBlockAdds'],observed['TargetRescales'])==(4,2,2)
                     and all(observed['Split'][k]==2 for k in ('fold_rot_count','fold_add_count','downconv_count',
                             'mask_mul_count','rescale_count')),'missing split operation')
            else:
                need((observed['PlainProducts'],observed['BabyRotations'])==(5,4),'wrong source map')
            peak=0.; witness=None; minimum=EPS; sentinel=0.
            for i,got in enumerate(values):
                terms=row(i); ideal=sum(w*truth[j] for j,w in terms); reference=sum(w*cut[j] for j,w in terms)
                error=abs(got-ideal); fresh=abs(got-reference); gain=[sum(abs(w) for j,w in terms if j//half==h) for h in (0,1)]
                inherited=sum(g*e for g,e in zip(gain,ep)); slack=EPS-fresh-inherited
                # An a posteriori two-half inequality, not independent noise or a verified MILP row.
                need(error<EPS and error<=fresh+inherited+1e-14 and slack>0,'full-slot error or observed joint margin')
                peak=max(peak,error); sentinel=max(sentinel,abs(values[(i+half)%full]-ideal))
                if witness is None or slack<minimum:
                    minimum=slack; witness=dict(slot=i,gain=gain,decoded_cut_half_Linf=ep,
                        observed_suffix_residual=fresh,remaining_for_cut=EPS-fresh,consumed=inherited,slack=slack)
            need(sentinel>=EPS,'corpus cannot detect swapped output halves')
            result[mode].append(dict(trial=tr,peak_error=peak,minimum_observed_margin=witness,
                                     swapped_halves_fail=True,checked_slots=full))
    gate=dict(schema='snabf.a2b.runtime-graph.v1',source='u[i]=(x[i]+x[i+1])/8; h=SiLU(u); y=W*h',
        encrypted_activation='u*(1/2+u/4)',slots=full,trials=2,selected_modes=routing['mode'],
        ordinary_and_switch_minima=gate['Minimum'],marginal_key_bytes={'keep-source':kb,'force-down':sb},
        results=result,raw_sha256={name:sha(out/name) for name in artifacts},
        scope='Runtime-only three arms. Adaptive optimizes marginal serialized key bytes, not latency. Margins use observed decoded values, never authorize a plan. No public MLIR admission, lowering, general noise certificate or feedback re-solve.')
    if width<full: gate.update(logical_width=width,padding_slots_checked=(full-width)*len(MODES)*2)
    return gate


def run(args):
    out=args.out.resolve(); out.mkdir(mode=0o700,exist_ok=False)
    exe=args.go_test.resolve(strict=True); planner=args.pass1.resolve(strict=True)
    env=dict(os.environ,SNABF_A2B_RUN=str(out),SNABF_TEST_PASS1=str(planner),SNABF_A12_EVIDENCE=str(out),SNABF_A2B_EXTENT=args.extent)
    def probe(name):
        with (out/(name+'.log')).open('xb') as log:
            subprocess.run([exe,'-test.run=^'+name+'$','-test.v','-test.count=1'],env=env,cwd=out,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    probe('TestA2bGraphBasis'); basis=read(out/'basis.json'); rows=instances(basis)
    costs=measure(rows.values(),args.cache,args.estimator)
    detail={k:dict(request=v,attacks=costs[digest(v)]) for k,v in rows.items()}
    store(out/'security-detail.json',detail); store(out/'screen.json',screen(basis,rows,detail))
    probe('TestA2bGraphArms'); store(out/'gate.json',audit(out,args.cache,args.extent))
    store(out/'tools.json',dict(go_test_sha256=sha(exe),pass1_sha256=sha(planner)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--replay',type=Path); group.add_argument('--out',type=Path)
    for key in ('go-test','pass1','cache','estimator'): ap.add_argument('--'+key,type=Path)
    ap.add_argument('--extent',choices=('full','tail-one'),default='full')
    args=ap.parse_args()
    if args.replay:
        need(audit(args.replay,args.cache,args.extent)==read(args.replay/'gate.json'),'stale gate')
        print('Replayed all 49152 slot comparisons.')
    else:
        need(all((args.go_test,args.pass1,args.cache,args.estimator)),'missing server run argument'); run(args)
