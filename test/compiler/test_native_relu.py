"""Linux source/witness component gates. Never substitute these for a sealed plan or decryption."""
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from c2_relu_source import source
from native_relu_contract import need, read, replay


def fixture(case):
    mixed = case == 'mixed'
    raw = source(84613+int(mixed), mixed).encode('ascii')
    name = re.search(rb'func.func @(\w+)',raw)[1].decode('ascii')
    ports = dict(schema='snabf.native-input-spec.v1', inputs=[dict(func_symbol=name,arg_index=0,source_id='x',
                 classification='secret',domain=[-1,1])])
    return raw, (json.dumps(ports)+'\n').encode('ascii')


def variants(w):
    def change(path, value):
        out = copy.deepcopy(w); node = out['sites'][0]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return out
    yield 'original-ssa', change(['source_lock','x'], 'arg0')
    yield 'stage-order', change(['stages',1,'input'], 0)
    # Forge both the coefficient and the producing instruction by the same ULP.
    old = w['sites'][0]['stages'][0]['bits'][1]; word = '%016x' % (int(old,16)^1)
    c = change(['stages',0,'bits',1],word)
    for n in c['sites'][0]['dag']:
        if n['constant'] == old:
            n['constant'] = word
    yield 'coordinated-coefficient-ulp', c
    yield 'reconstruction', change(['dag',-3,'inputs'],[w['sites'][0]['stages'][-1]['output']])
    yield 'claimed-half-bound', change(['bernstein',1],'0')
    yield 'missing-affine-endpoint', change(['affine_slots',2,1],0.5)


def gate(folder, case, arithmetic, importer):
    raw, spec = fixture(case)
    src, ports, claim = [folder/n for n in ('source.mlir','ports.json','claim.json')]
    src.write_bytes(raw); ports.write_bytes(spec)
    def run(args):
        return subprocess.run(list(map(str,args)), capture_output=True, timeout=90)
    positive = run([arithmetic,src,ports,'--relu-contract'])
    need(positive.returncode==0,positive.stderr.decode())
    w = read(positive.stdout); replay(case,raw,spec,w)
    def recheck(value, label=None):
        claim.write_text(json.dumps(value),encoding='ascii')
        r = run([arithmetic,src,ports,'--relu-recheck',claim]); msg = r.stderr.decode()
        need(r.returncode == (1 if label else 0) and not r.stdout and
             (not label or '[SNABF_NATIVE_RELU_WITNESS]' in msg), f'{label}: {msg}')
    witness = {k:w[k] for k in ('schema','source','input_spec','sites')}; recheck(witness)
    bad_witnesses = []
    for k, forged in variants(witness):
        recheck(forged,k)
        try:
            replay(case,raw,spec,forged)
        except ValueError:
            pass
        else:
            raise AssertionError(f'offline replay accepted {k}')
        bad_witnesses.append(k)
    prefix = b'%p = arith.maximumf %in, %zero : f32'
    bad_sources = []
    for label, edited, diagnostic in [
        ('x-is-not-source', raw.replace(prefix,b'%p = arith.maximumf %in, %in : f32'), 'SNABF_NATIVE_RELU_SSA'),
        ('negative-zero',raw.replace(b'%one =',b'%negative = arith.constant -0.0 : f32\n%one =')
                            .replace(prefix,prefix.replace(b'%zero',b'%negative')), 'SNABF_NATIVE_RELU_SSA'),
        ('outs',raw.replace(prefix,prefix.replace(b'%in',b'%old')), 'reads outs'),
        ('yield',raw.replace(b'linalg.yield %p',b'linalg.yield %in',1), 'SNABF_NATIVE_RELU_SSA'),
        ('fastmath',raw.replace(prefix,prefix.replace(b' :',b' fastmath<nnan> :')), 'SNABF_NATIVE_FLOAT'),
        ('spoof-attribute',raw.replace(b'domain_lower =',b'snabf.activation = "relu", domain_lower =',1), 'SNABF_NATIVE_ATTR')]:
        need(edited!=raw, label); src.write_bytes(edited)
        r = run([importer,src,'--input-spec',ports]); msg = r.stderr.decode()
        need(r.returncode==1 and not r.stdout and diagnostic in msg,f'{label}: {msg}')
        bad_sources.append(label)
    print(f'{case}: original contract + offline proof passed; source refusals={bad_sources}; witness refusals={bad_witnesses}')


if __name__ == '__main__':
    for case in ('single','mixed'):
        with tempfile.TemporaryDirectory() as root:
            gate(Path(root),case,*sys.argv[1:])
