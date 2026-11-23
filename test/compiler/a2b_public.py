"""Isolated, public MLIR two-result gate for the bounded neighbor/SiLU/diagonal graph."""
import argparse
import copy
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from a2b_fullslot_graph import MODES, instances, screen, lanes
from search import digest, measure, need, sha, store
from native_package import ROOT, pinned_go, unique_fields
from native_product import mlir_string
from a2b_bound import replay as replay_bound

S=4096
EPS=F(1,4096)


def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def read(p):
    return json.loads(p.read_bytes(),object_pairs_hook=unique_fields)


def source(width):
    ds=[]
    for j,(off,c) in enumerate(zip([0,1,S,S+1,2*S-1],[4,-8,2,6,-1])):
        ds.append(dict(offset=off,w64=[c*(1+(i+3*j)%5) if i<width and (i+off)%(2*S)<width else 0 for i in range(2*S)]))
    return dict(schema='snabf.a2b.neighbor-silu-linear.v1',input_width=width,output_width=width,input_linf='1/2',input_type='real-zero-padded',
                ops=['rotate(x,1)','u=(x+rot)/8','h=SiLU(u)','y=W*h'],diagonals=ds)


def carrier(path,**fields):
    path.write_text('module attributes {'+', '.join('snabf.a2b_'+k+' = '+mlir_string(v) for k,v in fields.items())+'} {}\n')


def invoke(opt,inp,flag,dest,expect=None):
    name=dest.with_suffix('');argv=list(map(str,[opt,inp,flag,'-o',dest]))
    store(name.with_suffix('.argv.json'),argv)
    p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    name.with_suffix('.log').write_bytes(p.stdout)
    if expect is None: need(p.returncode==0 and dest.exists(),f'{name}: {p.stdout[:350]!r}')
    else: need(p.returncode!=0 and expect.encode() in p.stdout,f'{name}: wrong negative refusal')


def rejections(opt,d,raw_source,plan,security):
    # Identity round-trip control: every bad plan differs only by the intended edits,
    # not by the canonical JSON spelling or a stale seal.
    raw=canonical(plan)
    need(raw==(d/'proposal.json').read_bytes(),'JSON serialization differs from the producer')
    control=d/'identity-resealed.mlir'
    carrier(control,source=raw_source,arm=b'force-down',stage=b'checked',plan=raw,seal=hashlib.sha256(raw).hexdigest().encode())
    controls={}
    for flag in ('snabf-check-plan','snabf-lower-ckks'):
        dest=d/('identity-'+flag+'.mlir')
        invoke(opt,control,f'--{flag}=native-security={security}',dest)
        controls[flag]=sha(dest)
    results={}
    for group in ('halves-tail','fold','quadrants','key-basis'):
        p=copy.deepcopy(plan)
        if group=='halves-tail':
            for k in ('split_outputs','outputs_down'):
                p[k].reverse();p[k][0]['used']=S
        elif group=='fold':p['fold'].update(offsets=[1,1],galois=5,mask_halves=[1,0])
        elif group=='quadrants':p['children'][1]=copy.deepcopy(p['children'][0])
        else:
            p['basis']['q'][:2]=reversed(p['basis']['q'][:2]);p['fold']['to']='source'
            for item in p['split_outputs']+p['outputs_down']:item.update(key_domain='source',q=p['basis']['q'][:2])
        raw=canonical(p);bad=d/(group+'.mlir');carrier(bad,source=raw_source,arm=b'force-down',stage=b'checked',plan=raw,seal=hashlib.sha256(raw).hexdigest().encode())
        for flag,why in (('snabf-check-plan','differs from source replay'),('snabf-lower-ckks','lowering source replay mismatch')):
            dest=d/(group+'-'+flag+'.mlir');invoke(opt,bad,f'--{flag}=native-security={security}',dest,why)
            results[group+'/'+flag]=sha(dest.with_suffix('.log'))
    return controls,results


def proof(p,original):
    f=p['feedback'];budget=F(f['submitted']);old=min(F(f['initial_'+a]['budget']) for a in ('keep','down'))
    need(0<budget<=old and f['runs'] and f['models'] and f['source_edge']==11,'invalid source cut receipt')
    for name in ('keep','down'):
        m=p[name];fresh,gain,ec,remaining,total=map(lambda k:F(m[k]),('fresh','gain','cut_error','budget','total'))
        need(fresh>0 and gain>0 and (EPS-fresh)/gain==remaining and fresh+gain*ec==total<EPS and ec<remaining and m['headroom'],'physical suffix replay')
        need(m==replay_bound(original,p['basis'],name=='down') and f['initial_'+name]==replay_bound(original,f['initial'],name=='down'),'independent source-to-sink replay')
    b=p['basis'];q=b['q'];aux=b['p'][0];g=F(2**(2*b['d']),q[3]);n=16384
    # Independent down KS in the larger ring, including mod-down rounding at the actual effective scale.
    need(F(p['down']['down_key_charge'])==(F(10*n*n*sum(x+1 for x in q[:2]),aux)+F(n*(n+1),2))/g,'bad switching charge')
    need((math.prod(q)*aux).bit_length()<=p['cap_bits'] and all(aux>=(1<<f['margin_bits'])*x for x in q),'wrong ordered basis')


def compile_arms(args,out,width):
    src=canonical(source(width));(out/'original.json').write_bytes(src)
    cells={};controls={};neg={}
    for mode in MODES:
        d=out/mode;d.mkdir();carrier(d/'source.mlir',source=src,arm=mode.encode(),stage=b'source')
        invoke(args.opt,d/'source.mlir','--snabf-pass4-fullslot=proposal='+str(d/'proposal.json'),d/'planned.mlir')
        p=read(d/'proposal.json');proof(p,json.loads(src));b=p['basis'];q,aux=b['q'],b['p']
        basis=dict(SourceN=16384,TargetN=8192,CutLevel=2,KeyLevelQ=1,KeyLevelP=0,SourceQ=q,TargetQ=q[:2],P=aux)
        rows=instances(basis);cost=measure(rows.values(),args.cache,args.estimator)
        detail={k:dict(request=v,attacks=cost[digest(v)]) for k,v in rows.items()}; minima=screen(basis,rows,detail)['Minimum']
        sec=d/'security.json';store(sec,dict(original_sha256=sha(out/'original.json'),arm=mode,rows=detail))
        check='--snabf-check-plan=native-security='+str(sec)+(' require-ring-down' if mode=='force-down' else '')
        invoke(args.opt,d/'planned.mlir',check,d/'checked.mlir');pkg=out/'module'/mode.replace('-','');pkg.parent.mkdir(exist_ok=True)
        invoke(args.opt,d/'checked.mlir',f'--snabf-lower-ckks=native-security={sec} native-output={pkg}',d/'lowered.mlir')
        need((pkg/'program.json').read_bytes()==(d/'proposal.json').read_bytes() and (pkg/'source.json').read_bytes()==src,'emitted program or immutable source mismatch')
        cells[mode]=dict(selected=p['selected'],basis=b,minimum=minima,ordered_plan=sha(pkg/'program.json'),model=sha(pkg/'model.go'),
          source=sha(pkg/'source.json'),initial=f_init(p),keep_bound=float(F(p['keep']['total'])),down_bound=float(F(p['down']['total'])),
          row=float(F(p['feedback']['submitted'])),models=p['feedback']['models'],runs=p['feedback']['runs'],cost=p['preference'])
        if mode=='force-down':controls,neg=rejections(args.opt,d,src,p,sec)
        if width==S+1:
            if mode in ('force-down','adaptive'):
                # Synthetic under-floor control, never reported as an estimator measurement.
                v=read(sec);v['rows']['switch.small']['attacks']['usvp']=127;weak=d/'control-127.json';store(weak,v)
                wc=f'--snabf-check-plan=native-security={weak}';wl=f'--snabf-lower-ckks=native-security={weak}'
                if mode=='force-down':
                    invoke(args.opt,d/'planned.mlir',wc,d/'force-refused.mlir','below 128 bits')
                    invoke(args.opt,d/'checked.mlir',wl,d/'force-lower-refused.mlir','below 128 bits')
                else:
                    invoke(args.opt,d/'planned.mlir',wc,d/'fallback.mlir')
                    pkg=d/'fallback-package';invoke(args.opt,d/'fallback.mlir',wl+f' native-output={pkg}',d/'fallback-lowered.mlir')
                    need(read(pkg/'program.json')['selected']=='keep-source','insecure switch did not fall back')
                    invoke(args.opt,d/'fallback.mlir',wc+' require-ring-down',d/'fallback-down-refused.mlir','missing required physical down')
    need(len({canonical(x['basis']) for x in cells.values()})==1,'three arms do not share parameters')
    return dict(cells=cells,controls=controls,rejections=neg)


def f_init(p):
    return dict(basis=p['feedback']['initial'],down_bound=float(F(p['feedback']['initial_down']['total'])))


def run_ciphertexts(out,width,report):
    # Compile the three untouched lowerer outputs against this tree's committed
    # production backend (no test-only fixture, decryptor or keygen in Eval).
    module=out/'module';go,env=pinned_go();env['GOMAXPROCS']='1'
    for f in ('go.mod','go.sum'):shutil.copyfile(ROOT/f,module/f)
    for package in ('','gadget','tower','ringswitch'):
        sub=Path('src/backend')/package;(module/sub).mkdir(parents=True)
        for f in (ROOT/sub).glob('*.go'):
            if not f.name.endswith('_test.go'):shutil.copyfile(f,module/sub/f.name)
    shutil.copyfile(Path(__file__).with_name('a2b_eval_test.go.in'),module/'graph_test.go')
    argv=[str(go),'test','-p','1','-mod=readonly','-count=1','-run','^TestLoweredArms$','-v','.']
    store(out/'go.argv.json',argv)
    with (out/'actual-go.log').open('xb') as log:subprocess.run(argv,cwd=module,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    original=read(out/'original.json');ds=original['diagonals'];peak={k:0. for k in MODES};pad_peak={k:0. for k in MODES};raw={}
    for trial in range(2):
        inp=module/f'input-{trial}.json';row=read(inp);raw[inp.name]=sha(inp);x=row['x']
        need(x==[((37*i+17*trial+13*(i//S))%251-125)/256 if i<width else 0 for i in range(2*S)],'wrong original corpus or tail')
        u=[(x[i]+x[(i+1)%(2*S)])/8 for i in range(2*S)];h=[z/(1+math.exp(-z)) for z in u]
        truth=[sum(r['w64'][i]*h[(i+r['offset'])%(2*S)]/64 for r in ds) for i in range(2*S)];need(all(v==0 for v in truth[width:]),'source output padding is not zero')
        cuts=set()
        for mode in MODES:
            pf=module/mode.replace('-','')/'program.json';p=read(pf);name=module/f'trial-{trial}-{mode}.json';item=read(name);raw[name.name]=sha(name)
            down=p['selected']=='force-down';ports=item['ports'];obs=item['trace'];cuts.add(obs['CutSHA256']);need(item['input_sha256']==row['input_sha256'] and obs['CutScale']==p['keep']['cut_scale'],'unbound prefix')
            expected=([(0,S,8192,0),(S,width-S,8192,0)] if down else [(0,width,16384,1)])
            need([(b['start'],b['used'],b['ring'],b['level']) for b in ports]==expected,'invalid physical output order/extent')
            if down:
                counts=obs['Down'];need((counts['CompletedBlocks'],counts['CrossBlockAdds'],counts['TargetRescales'])==(4,2,2) and all(counts['Split'][k]==2 for k in ('fold_rot_count','fold_add_count','downconv_count','mask_mul_count','rescale_count')),'missing executed split operation')
                for b,c in zip(p['split_outputs'],counts['Split']['outputs']):need(b['scale']==c['exact_scale'] and (b['start'],b['used'])==(c['logical_start'],c['used_slots']),'wrong live child')
            else:need(obs['Keep']['PlainProducts']==len(ds),'keep did not execute original matrix')
            for b in ports:
                need(b['scale']==p['down' if down else 'keep']['sink_scale'],'physical scale mismatch')
                for j,got in enumerate(lanes(b['slots'],b['ring']//2)):
                    i=b['start']+j;err=abs(got-truth[i]);need(err<float(EPS),'physical lane exceeded 2^-12');peak[mode]=max(peak[mode],err)
                    if i>=width:pad_peak[mode]=max(pad_peak[mode],err)
        need(len(cuts)==1,'three arms used different physical prefix results')
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    store(out/'gate.json',dict(schema='snabf.a2b.public-gate.v1',tree=git('rev-parse','HEAD^{tree}'),head=git('rev-parse','HEAD'),dirty=bool(git('status','--porcelain')),
        original=sha(out/'original.json'),entry=sha(out/'public-entry.json'),peak_error=peak,padding_peak=pad_peak,checked=2*len(MODES)*2*S,
        padding_checked=(2*S-width)*2*len(MODES),violations=0,raw_sha256=raw,compiler=report['compiler'],
        scope='Bounded real neighbor-SiLU-diagonal source. Conditional primitive allowances, not a probability certificate. Adaptive scores key coefficients, not latency.'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('opt','out','cache','estimator'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--width',choices=(S+1,2*S),type=int,required=True);ap.add_argument('--compile-only',action='store_true')
    args=ap.parse_args();args.opt=args.opt.resolve(strict=True);out=args.out.resolve();out.mkdir(mode=0o700,exist_ok=False)
    report=compile_arms(args,out,args.width);report['compiler']=sha(args.opt);store(out/'public-entry.json',report)
    if not args.compile_only:run_ciphertexts(out,args.width,report)


if __name__=='__main__':main()
