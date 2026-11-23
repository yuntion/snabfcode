"""Linux private gate for the public native CLI. No test compiler, prior plan or prior ciphertext."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from native_import import oracle

ROOT = oracle.ROOT
sys.path.insert(0, str(ROOT/'src/tools'))
from native_package import _dependencies, pinned_go, snapshot
from native_product import mlir_string
from verify_n1_equivalence import SOURCE, audit_exec, corpus, replay_batch


def require(ok, why):
    if not ok: raise ValueError(why)


def save(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, allow_nan=False, indent=2); f.write('\n')


def sha(b): return hashlib.sha256(b).hexdigest()


def traced(cmd, cwd, env, dest):
    p, report = audit_exec(cmd, cwd, env, trace_dir=dest)
    (dest/'stdout').write_text(p.stdout); (dest/'stderr').write_text(p.stderr)
    save(dest/'exec.json', report)
    require(p.returncode == 0, f'{dest}: process failed: '+p.stderr[-650:]+' '+p.stdout[-650:])
    require(not any(Path(v).name.startswith(('NativeArithmeticTest','snabf-native-import','snabf-native-linear'))
                    for v in report['executables']), 'test/inspection substitute in native execution')
    return report


class Number(str):
    """Preserve the emitter's exact spelling of JSON numeric literals, even across a coordinated reseal."""


def canonical(v):
    if type(v) is Number: return str(v)
    if isinstance(v, list): return '['+','.join(map(canonical,v))+']'
    if isinstance(v, dict): return '{'+','.join(json.dumps(k)+':'+canonical(v[k]) for k in sorted(v))+'}'
    return json.dumps(v, allow_nan=False, separators=(',',':'))


def reseal(p): return sha(canonical({k:v for k,v in p.items() if k!='emit_sha256'}).encode())


def reject_native_b2i(opt, published, dest):
    # b2i-local alone is meaningful to the DirectGo checker. An N1 module
    # must not silently pass by skipping that backend's local audit.
    target = dest/'b2i-checked.mlir'
    flag = '--snabf-check-plan=native-security=security.json b2i-local=true'
    p = subprocess.run([opt, '--mlir-print-op-on-diagnostic=false', flag, 'checked.mlir', '-o', target],
                       cwd=published, capture_output=True, text=True, timeout=300)
    (dest/'b2i-refusal.log').write_text(p.stderr)
    require(p.returncode == 1 and 'b2i-local requires DirectGo v1' in p.stderr and not target.exists(),
            'native N1 silently accepted or misdiagnosed the DirectGo B2i audit')


def replace_attr(ir, field, raw):
    pattern = rf'(snabf\.n1_{field} = )"(?:[^"\\]|\\.)*"'
    result, count = re.subn(pattern, lambda m:m[1]+mlir_string(raw), ir)
    require(count==1, 'not a native checked MLIR carrier: '+field)
    return result


def refusals(dest, opt, env, planbytes, proof, ir):
    initial = json.loads(planbytes, parse_float=Number); original = initial['emit_sha256']
    require(reseal(initial)==original, 'cannot recreate original LLVM seal')
    reports = []
    for fault in ('join','encoding-branch','coefficient-ulp','ordered-basis'):
        d = dest/fault; d.mkdir(); p = copy.deepcopy(initial); sc = copy.deepcopy(proof)
        es = p['edges']
        if fault=='join':
            j = next(e['joins'][0] for e in es if e.get('joins')); j['lhs_mul']+=1; j['bias']='0'
        elif fault=='encoding-branch':
            b = es[0]['plain']; require(b['meta']==b['effective'], 'need numerically equal scales')
            b.update(branch='bigfloat_value', precision=96)
        elif fault=='coefficient-ulp':
            b = next(e['plain'] for e in es if e['op']=='coefficient')
            b['bits'][0]=f'{int(b["bits"][0],16)^1:016x}'
            b['sha']=sha(bytes.fromhex(''.join(b['bits']))*(p['slots']//len(b['bits'])))
        else:
            p['q'].reverse(); sc['q'].reverse()
        p['emit_sha256']=reseal(p)
        require(p['emit_sha256']!=original, 'mutation did not change semantic seal')
        edited=replace_attr(replace_attr(ir,'plan',canonical(p).encode()),'seal',p['emit_sha256'].encode())
        (d/'checked.mlir').write_text(edited); save(d/'security.json',sc)
        for entry in ('check-plan','lower-ckks'):
            flag=f'--snabf-{entry}=native-security=security.json'+(' native-output=go' if entry=='lower-ckks' else '')
            target=d/(entry+'.mlir')
            cmd=[opt,'--mlir-print-op-on-diagnostic=false',flag,'checked.mlir','-o',target.name]
            result=subprocess.run(cmd,cwd=d,env=env,capture_output=True,text=True,timeout=300)
            (d/(entry+'.log')).write_text(result.stderr)
            why='wrong ordered estimator prime' if fault=='ordered-basis' else 'stale native ScalePlan'
            require(result.returncode==1 and why in result.stderr and not (d/'go').exists() and not target.exists(),
                    f'{fault}/{entry}: not a semantic refusal: '+result.stderr[-600:])
            reports.append(dict(fault=fault,entry=entry,exit=result.returncode,reason=why,resealed=p['emit_sha256']))
    save(dest/'refusals.json',reports)
    return len(reports)


def run(opt, cache, out):
    require(sys.platform=='linux' and not out.exists(), 'private fresh Linux output required')
    out.mkdir(mode=0o700); portable=out/'offline'; portable.mkdir()
    go, env = pinned_go(); summaries={}
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    require(not git('status','--porcelain'), 'uncommitted compiler/gate source')
    manifest=dict(schema='snabf.n1-offline.v1',head=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),graphs={})
    link=(opt.parent/'CMakeFiles/snabf-native-opt.dir/link.txt').read_text()
    ldd=subprocess.check_output(['ldd',opt],text=True)
    symbols=subprocess.check_output(['nm','-C',opt],text=True)
    require('heir' not in (link+ldd).lower() and 'not found' not in ldd and 'mlir::heir::' not in symbols and
            'libsnabf_native_product.a' in link and 'libMLIR.so' in ldd, 'unexpected compiler link dependencies')
    (out/'link.txt').write_text(link); (out/'ldd.txt').write_text(ldd)
    save(out/'compiler.json',dict(sha256=sha(opt.read_bytes()),heir_symbols=0,**{k:manifest[k] for k in ('head','tree')}))
    for label, digest in SOURCE.items():
        case='c1-single' if label=='single_gelu' else label
        d=out/label; d.mkdir(); src=d/'input.mlir'; port=d/'port.json'
        src.write_bytes(oracle.original(case)); port.write_bytes(oracle.spec_for(case))
        require(sha(src.read_bytes())==digest, 'original source mismatch')
        cmd=[sys.executable,'-B',ROOT/'src/tools/snabf_compile.py','--frontend','native-linalg','--emit-go',
             '--mode','keep-source','--input-bound','1','--mlir',src,'--input-spec',port,'--security-cache',cache,
             '--snabf-opt',opt,'--out-dir',d/'public']
        trace=traced(cmd,d,os.environ,d/'cli-trace'); require(str(opt) in trace['executables'], 'missing product MLIR process')
        pub=d/'public'; module=pub/'module'; planbytes=(module/'planned/program.json').read_bytes(); p=json.loads(planbytes)
        proof=json.loads((pub/'security.json').read_bytes()); release=json.loads((pub/'compile.json').read_bytes())
        require(p['source']==digest and p['emit_sha256']==release['emit_sha256'] and p['input_spec']==sha(port.read_bytes()) and
                release['selected_mode']=='keep-source' and 0<=p['engineering_upper']<=2**-12 and p['probabilistic_cert_issued'] is False,
                'wrong source/precision/type of release')
        rejected=refusals(d,opt,os.environ,planbytes,proof,(pub/'checked.mlir').read_text())
        reject_native_b2i(opt, pub, d)
        work=d/'cipher-module'; shutil.copytree(module,work)
        shutil.copyfile(Path(__file__).with_name('native_single_fixture_test.go.in'), work/'planned/cipher_test.go')
        deps=_dependencies(go,env,work); save(d/'go-deps.json',deps)
        before=snapshot(work); vectors=[dict(input=r['x'],want=r['truth']) for r in corpus(label,src)]
        save(d/'vectors.json',vectors)
        raw=portable/(label+'.jsonl'); local={**env,'TEST_VECTORS':str(d/'vectors.json'),'TEST_RESULT':str(d/'stats.json'),'N1_RAW_JSONL':str(raw)}
        for kind,args in [('build',[]),('test',['-count=1','-v'])]:
            traced([go,kind,'-mod=readonly',*args,'./...'],work,local,d/(kind+'-trace'))
        require(before==snapshot(work),'cipher run modified package')
        stats=json.loads((d/'stats.json').read_bytes()); n=p['slots']
        require(stats['trials']==64 and n==4096 and stats['slots']==64*n and stats['exceeded']==0 and
                0<=stats['max_error']<2**-12,'insufficient ciphertext evidence')
        sink=p['edges'][p['sink']]; shutil.copyfile(src, portable/(label+'.mlir'))
        manifest['graphs'][label]=dict(seal=p['emit_sha256'],ring=p['ring'],level=sink['level'],slots=n,scale=sink['meta'],raw_sha256=sha(raw.read_bytes()))
        summaries[label]=dict(**stats,refusals=rejected,source=digest,engineering_upper=p['engineering_upper'],
                              min_security=min(v for r in proof['instances'].values() for v in r['attacks'].values()))
        print(label, json.dumps(summaries[label]), flush=True)
    save(out/'linux.json',summaries); save(portable/'manifest.json',manifest)
    mh=sha((portable/'manifest.json').read_bytes()); save(out/'linux-replay.json',replay_batch(portable,mh))
    with zipfile.ZipFile(out/'offline.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for f in sorted(portable.iterdir()): z.write(f,f.name)
    print('manifest',mh,'zip',sha((out/'offline.zip').read_bytes()),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('opt','cache','out'): ap.add_argument(name,type=Path)
    args=ap.parse_args(); run(args.opt.resolve(strict=True),args.cache.resolve(strict=True),args.out.absolute())
