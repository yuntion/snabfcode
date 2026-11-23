"""Private Linux acceptance driver. The portable verifier has no compiler/Go dependency."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from native_relu_slot_replay import SOURCES, PORTS, corpus, replay
from test_native_relu import fixture
from product_native_n1 import (ROOT, Number, canonical, refusals, replace_attr, require, reseal,
                               save, sha, traced)
from native_package import _dependencies, pinned_go, snapshot


def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def compile_case(d,case,opt,cache):
    d.mkdir(mode=0o700); raw,ports=fixture(case);require(sha(raw)==SOURCES[case] and sha(ports)==PORTS[case],'original source/port')
    src=d/'original.mlir';spec=d/'ports.json';src.write_bytes(raw);spec.write_bytes(ports)
    link=(opt.parent/'CMakeFiles/snabf-native-opt.dir/link.txt').read_text();ldd=subprocess.check_output(['ldd',opt],text=True)
    require('heir' not in (link+ldd).lower() and 'not found' not in ldd and 'libMLIR.so' in ldd and 'libsnabf_native_product.a' in link and
            'mlir::heir::' not in subprocess.check_output(['nm','-C',opt],text=True),'unexpected native compiler dependency')
    (d/'link.txt').write_text(link);(d/'ldd.txt').write_text(ldd)
    argv=[sys.executable,'-B',str(ROOT/'src/tools/snabf_compile.py'),'--frontend','native-linalg','--emit-go','--mode','keep-source',
          '--mlir',str(src),'--input-spec',str(spec),'--input-bound','1','--log-n','16','--q-count','29','--gadget-alpha','2',
          '--interval-mode','affine','--min-slots','4096','--snabf-opt',str(opt),'--security-cache',str(cache),'--out-dir',str(d/'public')]
    trace=traced(argv,d,os.environ,d/'cli-trace');require(str(opt) in trace['executables'],'not the formal native optimizer')
    pub=d/'public';p=json.loads((pub/'module/planned/program.json').read_bytes());receipt=json.loads((pub/'compile.json').read_bytes())
    sec=json.loads((pub/'security.json').read_bytes());w,=p['relu_jacobian']; require(p['source']==sha(raw) and p['input_spec']==sha(ports) and
        p['emit_sha256']==receipt['emit_sha256'] and not p['probabilistic_cert_issued'] and 0<p['engineering_upper']<=2**-12 and
        len(p['p'])==2 and sec['q']==p['q'] and sec['p']==p['p'] and min(v for r in sec['instances'].values() for v in r['attacks'].values())>=128 and
        len(w['sources'])>0 and w['cells']>0 and (pub/'checked.mlir').is_file() and (pub/'lowered.mlir').is_file(),'invalid published evidence')
    old=argv.copy();old[old.index('native-linalg')]='legacy-heir';old[-1]=str(d/'legacy');r=subprocess.run(old,capture_output=True,text=True)
    (d/'legacy.log').write_text(r.stderr+r.stdout);require(r.returncode!=0 and 'requires explicit --frontend native-linalg' in (r.stderr+r.stdout) and not (d/'legacy').exists(),'legacy accepted native ABI')
    save(d/'compile-evidence.json',dict(head=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),binary_sha256=sha(opt.read_bytes()),
        seal=p['emit_sha256'],sink_upper=p['engineering_upper'],relu_upper=w['output_error'],injections=len(w['sources']),
        cells=w['cells'],physical_edges=len(w['edges']),join_intermediates=len(w['join_intermediates']),heir_execs=trace['forbidden_execs']))


def cipher_case(d,case):
    pub=d/'public'; module=d/'cipher-module';shutil.copytree(pub/'module',module)
    shutil.copyfile(Path(__file__).with_name('native_single_fixture_test.go.in'),module/'planned/cipher_test.go')
    go,env=pinned_go();env.update(GOMAXPROCS='4',GOMEMLIMIT='7GiB');save(d/'dependencies.json',_dependencies(go,env,module))
    save(d/'vectors.json',[dict(input=x,want=y) for x,y in corpus(case,(d/'original.mlir').read_bytes())]);before=snapshot(module)
    env.update(TEST_VECTORS=str(d/'vectors.json'),TEST_RESULT=str(d/'stats.json'),N1_RAW_JSONL=str(d/'raw.jsonl'))
    for mode,args in [('build',[]),('test',['-count=1','-v','-timeout=45m'])]:
        traced([go,mode,'-mod=readonly',*args,'./...'],module,env,d/('go-'+mode))
    require(snapshot(module)==before,'build mutated the independent package')
    s=json.loads((d/'stats.json').read_bytes());require(s['exceeded']==0 and s['trials']==64 and s['slots']==64*4096,'incomplete ciphertext data')


def negatives(d,opt):
    pub=d/'public';raw=(pub/'module/planned/program.json').read_bytes();security=json.loads((pub/'security.json').read_bytes());ir=(pub/'checked.mlir').read_text()
    dest=d/'forgeries';dest.mkdir();refusals(dest,opt,os.environ,raw,security,ir)
    rows=[];initial=json.loads(raw,parse_float=Number)
    for name in ('original-ssa','stage-chain','signed-weight','bypass','closed-half','missing-injection','jacobian-gain','intermediate-no-wrap'):
        p=copy.deepcopy(initial);site=p['relu_witness']['sites'][0];j=p['relu_jacobian'][0]
        if name=='original-ssa':site['source_lock']['x']='arg0'
        elif name=='stage-chain':site['stages'][1]['input']=0
        elif name=='signed-weight':
            old=site['stages'][0]['bits'][1];new='%016x'%(int(old,16)^1);site['stages'][0]['bits'][1]=new
            for e in site['dag']:
                if e['constant']==old:e['constant']=new
                e['weights']=[new if k==old else k for k in e['weights']]
        elif name=='bypass':site['dag'][-3]['inputs']=[site['stages'][-1]['output']]
        elif name=='closed-half':site['bernstein'][0]='0';site['affine_slots'][0][0]=0
        elif name=='missing-injection':j['sources'].pop()
        elif name=='jacobian-gain':j['sources'][0]['output_abs_partial']=0;j['output_error']=0
        else:j['join_intermediates'][0]['magnitude']=0;j['edges'][0]['decoded_error']=0
        p['emit_sha256']=reseal(p);require(p['emit_sha256']!=initial['emit_sha256'],'unchanged forgery')
        sub=dest/name;sub.mkdir();(sub/'checked.mlir').write_text(replace_attr(replace_attr(ir,'plan',canonical(p).encode()),'seal',p['emit_sha256'].encode()))
        save(sub/'security.json',security)
        why='SNABF_NATIVE_RELU_WITNESS' if name in ('original-ssa','stage-chain','signed-weight','bypass','closed-half') else 'stale native ScalePlan'
        for entry in ('check-plan','lower-ckks'):
            flag='--snabf-'+entry+'=native-security=security.json'+(' native-output=emitted' if entry=='lower-ckks' else '')
            args=[opt,'--mlir-print-op-on-diagnostic=false',flag,'checked.mlir','-o',entry+'.mlir'];r=subprocess.run(args,cwd=sub,capture_output=True,text=True,timeout=300)
            (sub/(entry+'.log')).write_text(r.stderr);require(r.returncode==1 and why in r.stderr and not (sub/(entry+'.mlir')).exists() and not (sub/'emitted').exists(),f'{name}/{entry}: {r.stderr[-300:]}')
            rows.append(dict(name=name,entry=entry,reason=why,seal=p['emit_sha256']))
    save(dest/'relu-refusals.json',rows)


def package(out):
    portable=out/'offline';portable.mkdir();m=dict(schema='snabf.c-relu-offline.v1',head=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),graphs={})
    for case in SOURCES:
        d=out/case;p=json.loads((d/'public/module/planned/program.json').read_bytes());e=json.loads((d/'compile-evidence.json').read_bytes())
        require(e['tree']==m['tree'] and e['head']==m['head'] and json.loads((d/'stats.json').read_bytes())['exceeded']==0,'mixed builds')
        for old,new in [('original.mlir','.mlir'),('ports.json','.ports.json'),('raw.jsonl','.jsonl')]: shutil.copyfile(d/old,portable/(case+new))
        v=p['edges'][p['sink']];m['graphs'][case]=dict(seal=p['emit_sha256'],scale=v['meta'],level=v['level'],ring=p['ring'],slots=p['slots'],raw_sha256=sha((d/'raw.jsonl').read_bytes()))
    save(portable/'manifest.json',m);mh=sha((portable/'manifest.json').read_bytes());save(out/'linux-replay.json',replay(portable,mh))
    with zipfile.ZipFile(out/'offline.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for path in sorted(portable.iterdir()):z.write(path,path.name)
    print('manifest',mh,'zip',sha((out/'offline.zip').read_bytes()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['compile','cipher','negative','package']);ap.add_argument('out',type=Path);ap.add_argument('opt',type=Path);ap.add_argument('cache',type=Path);ap.add_argument('case',choices=SOURCES,nargs='?');a=ap.parse_args()
    require(sys.platform=='linux' and not git('status','--porcelain'),'clean Linux source required');r=a.out.resolve(strict=True);opt=a.opt.resolve(strict=True)
    if a.action=='package':package(r)
    elif a.action=='compile':compile_case(r/a.case,a.case,opt,a.cache.resolve(strict=True))
    elif a.action=='cipher':cipher_case(r/a.case,a.case)
    else:negatives(r/a.case,opt)
