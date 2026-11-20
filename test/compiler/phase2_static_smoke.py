"""Generate an unseen static residual graph and check actual decrypted slots.

Run from a configured Linux checkout. SNABF_OPT and SNABF_EXTERN select the
locked native tools. The compiler never reads this generator or its oracle.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def graph(seed):
    rng=random.Random(seed)
    matrices=[[[rng.choice((-8,-5,-2,1,3,4,6,7))/64 for _ in range(4)] for _ in range(4)] for _ in range(3)]
    v='tensor<4xf32>'; m='tensor<4x4xf32>'
    lines=['#id = affine_map<(i) -> (i)>','module {',f'func.func @f{seed:x}(%x: {v} {{secret.secret}}) -> {v} {{',
           '%zero = arith.constant 0.0 : f32','%one = arith.constant 1.0 : f32',f'%empty = tensor.empty() : {v}',
           f'%init = linalg.fill ins(%zero : f32) outs(%empty : {v}) -> {v}']
    source='%x'
    for i,weights in enumerate(matrices):
        lines += [f'%w{i} = arith.constant dense<{json.dumps(weights)}> : {m}',
                  f'%h{i} = linalg.matvec ins(%w{i}, {source} : {m}, {v}) outs(%init : {v}) -> {v}']
        if i!=2:
            lines += [f'%a{i} = linalg.generic {{indexing_maps = [#id, #id], iterator_types = ["parallel"], domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}} ins(%h{i} : {v}) outs(%empty : {v}) {{',
                      '^bb0(%in: f32, %old: f32):','%n = arith.negf %in : f32','%e = math.exp %n : f32',
                      '%d = arith.addf %e, %one : f32','%s = arith.divf %one, %d : f32','%p = arith.mulf %in, %s : f32',
                      'linalg.yield %p : f32',f'}} -> {v}']
            source=f'%a{i}'
    lines += [f'%y = arith.addf %h2, %h0 : {v}',f'return %y : {v}','}','}']
    def truth(x):
        h=x
        for i,w in enumerate(matrices):
            h=[sum(a*b for a,b in zip(row,h)) for row in w]
            if i==0: skip=h
            if i!=2: h=[a/(1+math.exp(-a)) for a in h]
        return [a+b for a,b in zip(h,skip)]
    inputs=[[0.,0.,0.,0.],[-1.,1.,-1.,1.],[rng.uniform(-1,1) for _ in range(4)]]
    return '\n'.join(lines)+'\n',[dict(x=x,y=truth(x)) for x in inputs]


def smoke(d,seed,cache):
    d.mkdir(exist_ok=False); src=d/'unseen.mlir'; model=d/'output'
    source,oracle=graph(seed); src.write_text(source)
    prod=[sys.executable,str(ROOT/'src/tools/snabf_compile.py'),'--mlir',str(src),'--out-dir',str(model),
          '--input-bound','1','--log-n','15','--q-count','16','--mode','keep-source','--emit-go']
    if cache: prod += ['--security-cache',str(cache.resolve())]
    # Prove unsupported source operators report the caller's source location.
    bad=d/'unsupported.mlir'; bad.write_text(source.replace('%zero =','%unused = tensor.dim %x, %c0 : tensor<4xf32>\n%zero =').replace('%unused =','%c0 = arith.constant 0 : index\n%unused ='))
    deny=prod.copy(); deny[deny.index(str(src))]=str(bad)
    r=subprocess.run(deny,capture_output=True,text=True)
    if r.returncode<=0 or '[SNABF_UNSUPPORTED_OP] op=tensor.dim' not in r.stderr or str(bad)+':' not in r.stderr or model.exists():
        raise RuntimeError('missing located unsupported-op rejection: '+r.stderr)
    subprocess.run(prod,check=True)
    p=json.loads((model/'work/parameters.json').read_text())
    polynomials={e['ID'].split('.')[0] for e in p['edges'] if e['ID'].startswith('poly')}
    linears={e['ID'].split('.')[0] for e in p['edges'] if e['ID'].startswith('lin')}
    if len(polynomials)!=2 or len(linears)!=3: raise RuntimeError('logical stages missing from concrete graph')
    (d/'oracle.json').write_text(json.dumps(oracle))
    gofile=d/'static_test.go'; shutil.copyfile(Path(__file__).with_name('phase2_slots.go.in'),gofile)
    shutil.copyfile(model/'model.go',d/'model.go')
    ext=Path(os.getenv('SNABF_EXTERN',ROOT/'extern')).resolve()
    env=dict(os.environ,GOPATH=str(ext/'gopath'),GOCACHE=os.getenv('GOCACHE',str(ext/'go-cache')),
             GOWORK='off',GOPROXY='off',GOTOOLCHAIN='local',SNABF_TEST_ORACLE=str(d/'oracle.json'))
    subprocess.run([ext/'go/bin/go','test','-mod=readonly','-v','-count=1',d/'model.go',gofile],check=True,cwd=ROOT,env=env)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--seed',type=int,default=2341)
    p.add_argument('--security-cache',type=Path); p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.out: smoke(a.out.resolve(),a.seed,a.security_cache)
    else:
        with tempfile.TemporaryDirectory() as d: smoke(Path(d)/'smoke',a.seed,a.security_cache)
