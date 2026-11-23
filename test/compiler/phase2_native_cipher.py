"""Private Linux N1 engineering gate: fresh source/check/emit, isolated module, true CKKS slots.

Requires phase2_native_security.py output for the same four original inputs.
Never changes the product CLI, saves keys, or reports a probability certificate.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys

import mpmath as mp
from native_import import oracle
from native_exec_trace import traced
sys.path.insert(0,str(oracle.ROOT/'src/tools'))
from native_package import _dependencies, pinned_go, publish_native_package


def require(ok,reason):
    if not ok: raise ValueError(reason)


def vectors(src, ports):
    # Always derive public truth from the original source literals, not emitted masks/coefficients.
    mp.mp.dps=75
    ref=oracle.expected(src,ports); width=ref['inputs'][0]['period']; rng=random.Random(57391)
    templates={}
    for n in ref['nodes']:
        if n['kind']=='constant' and n['shape']:
            fmt='>f' if n['element_type']=='f32' else '>d'
            templates[n['id']]=[mp.mpf(struct.unpack(fmt,bytes.fromhex(b[2:]))[0]) for b in n['bits']]
    trials=[]
    for seed in range(64):
        x=([[0.]*width,[-1.]*width,[1.]*width,[float(1-2*(j%2)) for j in range(width)]][seed]
           if seed<4 else [rng.uniform(-1,1) for _ in range(width)])
        vals=dict(templates,arg0=list(map(mp.mpf,x)))
        for n in ref['nodes']:
            k,id_,a=n['kind'],n['id'],n['inputs']
            if k=='matvec':
                w,v=vals[a[0]],vals[a[1]]
                vals[id_]=[mp.fdot(w[i*width:(i+1)*width],v) for i in range(width)]
            elif k=='add': vals[id_]=[u+v for u,v in zip(vals[a[0]],vals[a[1]])]
            elif k=='silu': vals[id_]=[t/(1+mp.exp(-t)) for t in vals[a[0]]]
            elif k=='gelu_exact_erf':
                lock=next(l for l in ref['activation_locks'] if l['site']==id_)
                inv=next(v for v in lock['literals'] if v['role']=='inv_sqrt2')
                c=mp.mpf(struct.unpack('>f' if inv['type']=='f32' else '>d',bytes.fromhex(inv['bits'][2:]))[0])
                vals[id_]=[t*(1+mp.erf(c*t))/2 for t in vals[a[0]]]
        trials.append({'input':x,'want':list(map(float,vals[ref['outputs'][0]['value']]))})
    return trials


def run(exe, evidence, out):
    out.mkdir(mode=0o700); go, env=pinned_go();summary={}
    for case,digest in oracle.HASHES.items():
        dest=out/case; dest.mkdir();pkg=dest/'module/planned';pkg.mkdir(parents=True)
        orig=evidence/case; raw=oracle.original(case);ports=oracle.spec_for(case)
        require((orig/'input.mlir').read_bytes()==raw and (orig/'port.json').read_bytes()==ports,'source bytes changed')
        proof=orig/'screen.json'; inputs=[exe,orig/'input.mlir',orig/'port.json'];plan=pkg/'program.json'
        def tool(args, path):
            with path.open('xb') as f:
                f.write(traced([*inputs,*args],str(path)+'.exec',180))
        tool(['--physical'],dest/'plan');plan.write_bytes((dest/'plan').read_bytes());spec=json.loads(plan.read_bytes());seal=spec['emit_sha256']
        require(spec['source']==digest and spec['input_spec']==hashlib.sha256(ports).hexdigest() and spec['engineering_upper']<=2**-12 and not spec['probabilistic_cert_issued'],'wrong physical gate')
        tool(['--check',plan,proof],dest/'checked');tool(['--emit',plan,proof],dest/'model');tool(['--runtime',plan,proof],dest/'runtime');
        shutil.copyfile(dest/'model',pkg/'model.go');shutil.copyfile(dest/'runtime',pkg/'native_runtime.go')
        # The semantic checker, unlike the package byte seal, accepts benign formatting.
        claimed=dest/'claim.json';claimed.write_text(json.dumps(spec,indent=3)+'\n');subprocess.run([*inputs,'--check',claimed,proof],check=True)
        rejects=0
        for fault in ('join','encoding-branch','ulp','basis'):
            r=copy.deepcopy(spec); sc=json.loads(proof.read_bytes()); es=r['edges']
            if fault=='join':
                j=next(e for e in es if e.get('joins'));j['joins'][0]['lhs_mul']+=1;j['joins'][0]['bias']='0'
            elif fault=='encoding-branch':
                b=es[0]['plain'];require(b['meta']==b['effective'],'source control not numerically equal');b.update(branch='bigfloat_value',precision=96)
            elif fault=='ulp':
                b=next(e['plain'] for e in es if e['op']=='coefficient');b['bits'][0]=f'{int(b["bits"][0],16)^1:016x}'
                # Refresh the coordinated tiled digest too; do not merely trip a stale hash.
                b['sha']=hashlib.sha256(bytes.fromhex(''.join(b['bits']))*(r['slots']//len(b['bits']))).hexdigest()
            else:r['q'].reverse();sc['q'].reverse()
            # An attacker can reseal edited JSON. Compiling the original remains authoritative.
            r['emit_sha256']=hashlib.sha256(json.dumps({k:v for k,v in r.items() if k!='emit_sha256'},sort_keys=True,separators=(',',':')).encode()).hexdigest()
            claimed.write_text(json.dumps(r));bad=dest/'bad-screen.json';bad.write_text(json.dumps(sc))
            for mode in ('--check','--emit'):
                res=subprocess.run([*inputs,mode,claimed,bad],capture_output=True,timeout=180)
                require(res.returncode==1 and not res.stdout and (b'stale native ScalePlan' in res.stderr or b'wrong ordered estimator prime' in res.stderr),f'{fault}/{mode}: {res.stderr!r}');rejects+=1
        receipt=publish_native_package(pkg,dest/'published',seal)
        require(receipt['program_file_sha256']==hashlib.sha256(plan.read_bytes()).hexdigest(),'published plan changed')
        for name in ('model.go','native_runtime.go','program.json'):
            require((pkg/name).read_bytes()==(dest/'published/planned'/name).read_bytes(),'cipher fixture differs from published bytes')
        for name in ('go.mod','go.sum'):shutil.copyfile(oracle.ROOT/name,pkg.parent/name)
        shutil.copyfile(Path(__file__).with_name('native_single_fixture_test.go.in'),pkg/'cipher_test.go')
        baseline={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.iterdir()}
        deps=_dependencies(go,env,pkg.parent)
        require(not any('heirckks' in v.lower() for v in deps),'legacy Go dependency')
        (dest/'dependencies.json').write_text(json.dumps(deps));(dest/'vectors.json').write_text(json.dumps(vectors(raw,ports),allow_nan=False))
        gate={**env,'TEST_VECTORS':str(dest/'vectors.json'),'TEST_RESULT':str(dest/'stats.json')}
        with (dest/'go.log').open('w') as log:
            for cmd in (['build','-mod=readonly','./...'],['test','-mod=readonly','-count=1','-v','./...']):
                res=subprocess.run([go,*cmd],env=gate,cwd=pkg.parent,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                require(not res.returncode, f'{case}: go {cmd[0]} failed; see {dest}/go.log')
        require(baseline=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in pkg.iterdir()},'package mutated')
        stats=json.loads((dest/'stats.json').read_bytes());require(stats['trials']==64 and stats['slots']==64*spec['slots'] and stats['exceeded']==0 and 0<=stats['max_error']<2**-12,'ciphertext gate failed')
        summary[case]=dict(**stats,dual_refusals=rejects,source_sha256=digest,emit_sha256=seal,engineering_upper=spec['engineering_upper']);print(case,summary[case],flush=True)
    (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('executable','evidence','out'):ap.add_argument(arg,type=Path)
    a=ap.parse_args();run(a.executable.resolve(strict=True),a.evidence.resolve(strict=True),a.out.resolve())
