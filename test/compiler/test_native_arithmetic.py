"""Four source-bound clear DAG checks and real integer Pass 3 replays (not decryption)."""
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from native_import import oracle


def truth(raw, ports, seed):
    ref = oracle.expected(raw, ports)
    width = ref['inputs'][0]['period']
    data = {'arg0': [((j*17+seed*23) % 101-50)/50 for j in range(width)]}
    for n in ref['nodes']:
        k, name, args = n['kind'], n['id'], n['inputs']
        if k == 'constant' and n['shape']:
            fmt = '>f' if n['element_type'] == 'f32' else '>d'
            data[name] = [struct.unpack(fmt, bytes.fromhex(b[2:]))[0] for b in n['bits']]
        elif k == 'matvec':
            w, x = data[args[0]], data[args[1]]
            data[name] = [math.fsum(a*b for a, b in zip(w[i*width:(i+1)*width], x)) for i in range(width)]
        elif k == 'add':
            data[name] = [a+b for a,b in zip(data[args[0]],data[args[1]])]
        elif k == 'silu':
            data[name] = [x/(1+math.exp(-x)) for x in data[args[0]]]
        elif k == 'gelu_exact_erf':
            lock = next(l for l in ref['activation_locks'] if l['site'] == name)
            lit = next(v for v in lock['literals'] if v['role'] == 'inv_sqrt2')
            inv = struct.unpack('>f' if lit['type'] == 'f32' else '>d',bytes.fromhex(lit['bits'][2:]))[0]
            data[name] = [0.5*x*(1+math.erf(x*inv)) for x in data[args[0]]]
    return data[ref['outputs'][0]['value']]


def main(exe):
    with tempfile.TemporaryDirectory() as tmp:
        src, spec = Path(tmp)/'source.mlir', Path(tmp)/'inputs.json'
        for case, digest in oracle.HASHES.items():
            raw, ports = oracle.original(case), oracle.spec_for(case)
            src.write_bytes(raw); spec.write_bytes(ports)
            result = json.loads(subprocess.check_output([exe,src,spec], timeout=240))
            assert result['source'] == digest and result['Q_count'] == 8
            assert result['polys'] == (['gelu_direct'] if case == 'c1-single' else ['silu_direct'])
            assert len(result['outputs']) == 8
            worst = 0.
            for seed, row in enumerate(result['outputs']):
                y = truth(raw, ports, seed)
                assert len(row) == len(y) and all(math.isfinite(z) for z in row)
                worst = max(worst,max(abs(a-b) for a,b in zip(row,y)))
            assert worst <= result['approximation'] + 1e-13
            assert result['approximation'] + result['arithmetic_bound'] <= 2**-12
            print(f"{case}: edges={result['edges']} sink_level={result['sink_level']} clear_error={worst:.7g} "
                  f"approx={result['approximation']:.7g} concrete_arithmetic={result['arithmetic_bound']:.7g}")

        def refuse(r, s, diagnostic):
            src.write_bytes(r); spec.write_bytes(s)
            p = subprocess.run([exe,src,spec],capture_output=True,timeout=120)
            assert p.returncode == 1 and not p.stdout and diagnostic.encode() in p.stderr, p

        no_domain = json.loads(ports); del no_domain['inputs'][0]['domain']
        refuse(raw,json.dumps(no_domain).encode(),'no bound domain')
        narrowed = raw.replace(b'domain_upper = 1.0', b'domain_upper = 0.0')
        assert narrowed != raw
        refuse(narrowed,ports,'exceeds source declaration')


if __name__ == '__main__':
    main(sys.argv[1])
