"""Offline only. Rebuild the two fixture functions from authenticated original bytes."""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys


SOURCES = {'single':'6bcbc730a6625fbe5bbe12bdfb1437779d8394dcd53542ea769279875d70fbb6',
           'mixed':'8b43325718f038a71941de9f110559547f1411a6b1b227ac29d7e9519fe81aa4'}
PORTS = {'single':'0e63bb0e98f072fc0bef8cbf4e55095c344c65c34fff1a4f4a49a7c715f0f185',
         'mixed':'abd860164e1234333ddab7b0b35c70ec6016a260e0bc0a0e617a0626d85032a1'}


def check(ok, why):
    if not ok: raise ValueError(why)


def sha(data): return hashlib.sha256(data).hexdigest()


def read(data):
    def unique(pairs):
        obj=dict(pairs); check(len(obj)==len(pairs),'duplicate JSON field'); return obj
    return json.loads(data,object_pairs_hook=unique)


def corpus(case, source):
    check(sha(source)==SOURCES[case], 'wrong original ReLU source')
    rows=re.findall(rb'%w[01] = arith.constant dense<([^>]+)>',source)
    check(len(rows)==2, 'expected two source matrices')
    a,b=[json.loads(row,parse_float=Decimal) for row in rows]
    def mv(m,x): return [sum(v*w for v,w in zip(row,x,strict=True)) for row in m]
    xs=[list(v) for v in itertools.product((-1,1),repeat=4)]+[[0]*4]
    xs += [[s*2.**-14 if i==j else 0 for i in range(4)] for s in (-1,1) for j in range(4)]
    for n in range(64-len(xs)):
        h=hashlib.sha256(f'snabf.native-relu.corpus.v1:{case}:{n}'.encode()).digest()
        xs.append([int.from_bytes(h[i:i+4],'big',signed=True)/2**31 for i in range(0,16,4)])
    check(len({tuple(x) for x in xs})==64,'repeated input')
    with localcontext() as ctx:
        ctx.prec=90
        for x in xs:
            h=mv(a,[Decimal(v) for v in x]); r=[max(Decimal(0),t) for t in h]
            if case=='mixed': r=[v+t/(1+(-t).exp()) for v,t in zip(r,h)]
            y=mv(b,r)
            if case=='mixed': y=[v+t for v,t in zip(y,h)]
            yield x,[float(v) for v in y]


def replay(batch, expected):
    raw=(batch/'manifest.json').read_bytes();check(sha(raw)==expected,'wrong manifest bytes');m=read(raw)
    check(m['schema']=='snabf.c-relu-offline.v1' and set(m['graphs'])==set(SOURCES),'incomplete batch')
    report={}
    for case,abi in m['graphs'].items():
        src=(batch/(case+'.mlir')).read_bytes(); ports=(batch/(case+'.ports.json')).read_bytes()
        path=batch/(case+'.jsonl');check(sha(path.read_bytes())==abi['raw_sha256'] and sha(ports)==PORTS[case],'wrong raw data/ports')
        check(abi['ring']==65536 and abi['slots']==4096 and type(abi['level']) is int and abi['level']>=0 and
              str(Fraction(abi['scale']))==abi['scale'] and Fraction(abi['scale'])>0 and re.fullmatch('[a-f0-9]{64}',abi['seal']),'invalid physical sink ABI')
        count=0; peak=0.;fields={'schema','trial','source_sha256','emit_sha256','x','ring','level','slots','scale','decoded','imag'}
        with path.open(encoding='ascii') as f:
            for i,((x,y),line) in enumerate(zip(corpus(case,src),f,strict=True)):
                r=read(line);check(set(r)==fields and r['schema']=='snabf.n1-complex-slots.v1' and r['emit_sha256']==abi['seal'] and
                    r['source_sha256']==SOURCES[case] and r['trial']==i and r['x']==x and all(type(v) in (int,float) for v in r['x']) and
                    all(type(r[k]) is int for k in ('ring','level','slots','trial')) and all(r[k]==abi[k] for k in ('ring','level','scale','slots')),'wrong trial binding')
                check(isinstance(r['decoded'],list) and isinstance(r['imag'],list) and len(r['decoded'])==abi['slots'],'bad slot count')
                for j,(re_,im) in enumerate(zip(r['decoded'],r['imag'],strict=True)):
                    check(type(re_) in (float,int) and type(im) in (float,int),'not numeric')
                    e=math.hypot(re_-y[j%4],im);check(math.isfinite(e) and e<2**-12,f'{case}/{i}/{j}: {e}')
                    peak=max(peak,e); count+=1
        check(count==64*abi['slots'],'incomplete slots');report[case]=dict(slots=count,linf=peak,exceeded=0,raw_sha256=abi['raw_sha256'])
    return dict(platform=sys.platform,head=m['head'],tree=m['tree'],manifest_sha256=expected,graphs=report)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('batch',type=Path);ap.add_argument('sha256');args=ap.parse_args()
    print(json.dumps(replay(args.batch,args.sha256),indent=2))
