"""Server-only: exercise the pinned estimator on the four native arithmetic proposals.

Keep this separate from SDK-only CTests. Never copy the measurement cache to a
release. The aggregate is an engineering gate, not a probability certificate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from native_import import oracle
from native_exec_trace import traced

sys.path.insert(0, str(oracle.ROOT/'src/tools'))
from security_cache import PIN, MODEL, SECRET, ERROR, digest, measure, request


def screen(exe, out, estimator):
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    requests, by_case = {}, {}
    for case, sha in oracle.HASHES.items():
        work = out/case; work.mkdir(mode=0o700)
        src, port = work/'input.mlir', work/'port.json'
        src.write_bytes(oracle.original(case)); port.write_bytes(oracle.spec_for(case))
        # Trace the real source parser, solver and their entire process tree.
        raw = traced([exe,src,port],work/'exec.log')
        p = json.loads(raw); q, aux = p['q'], p['p']; n = p['ring']
        if p['source'] != sha or p['arithmetic_bound'] + p['approximation'] > 2**-12:
            raise ValueError('source/precision mismatch')
        exact = math.prod(q)
        instances = {k:request('source.'+k,n,q,aux,m) for k,m in
                     [('Q',exact),('QP',exact*math.prod(aux)),('envelope',1 << math.ceil(p['security_cap']))]}
        requests.update((digest(r),r) for r in instances.values()); by_case[case] = instances
        (work/'basis.json').write_text(json.dumps(dict(source=sha,ring=n,q=q,p=aux,attempts=p['attempts'])))
    costs = measure(requests.values(),out/'measurements',estimator)
    result = {}
    for case, instances in by_case.items():
        values = {k:costs[digest(r)] for k,r in instances.items()}
        if not all(min(v.values()) >= 128 for v in values.values()):
            raise ValueError(f'128-bit ordinary-domain check failed: {case}: {values}')
        result[case] = dict(attacks=values, request_ids={k:digest(r) for k,r in instances.items()})
        basis = json.loads((out/case/'basis.json').read_text())
        evidence = dict(schema='snabf.native-single-security.v1', model=MODEL, secret=SECRET, error=ERROR,
                        estimator=PIN, samples='unbounded', instances={k:dict(modulus=r['modulus'],attacks=values[k])
                        for k,r in instances.items()}, **basis)
        (out/case/'screen.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(case, 'min classical estimate', min(min(v.values()) for v in values.values()), flush=True)
    (out/'security.json').write_text(json.dumps(dict(estimator=PIN, cases=result),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable',type=Path); parser.add_argument('out',type=Path)
    parser.add_argument('--estimator',type=Path,default=os.getenv('LATTICE_ESTIMATOR'))
    args = parser.parse_args(); screen(args.executable.resolve(strict=True),args.out,args.estimator)
