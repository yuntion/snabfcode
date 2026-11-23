"""Offline, fixture-specific source, DAG and rational closed-domain replay. No CKKS runner."""
import argparse
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from c2_relu_portable import portable_proof


SOURCE = {'single': '6bcbc730a6625fbe5bbe12bdfb1437779d8394dcd53542ea769279875d70fbb6',
          'mixed': '8b43325718f038a71941de9f110559547f1411a6b1b227ac29d7e9519fe81aa4'}
PORT = {'single': '0e63bb0e98f072fc0bef8cbf4e55095c344c65c34fff1a4f4a49a7c715f0f185',
        'mixed': 'abd860164e1234333ddab7b0b35c70ec6016a260e0bc0a0e617a0626d85032a1'}
COEFFICIENTS = '5d1c3d6f6572aed704ccc0fe51b0b0d9cce459209660f493dcf8ec365bbc968e'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(raw):
    def unique(pairs):
        d = dict(pairs)
        need(len(d) == len(pairs), 'duplicate JSON field')
        return d
    return json.loads(raw, object_pairs_hook=unique)


def exact(word):
    need(isinstance(word, str) and re.fullmatch('[a-f0-9]{16}', word), 'invalid binary64 word')
    return F(struct.unpack('>d', bytes.fromhex(word))[0])


def combine(a, b, product=False):
    c = {} if product else dict(a)
    for i, x in (a.items() if product else [(0, 1)]):
        for j, y in b.items():
            c[i+j] = c.get(i+j, 0) + x*y
    return {i: x for i, x in c.items() if x}


def check_dag(site):
    dag, stages = site['dag'], site['stages']
    need([s['degree'] for s in stages] == [7]*5+[5], 'wrong odd chain order/degrees')
    need(sha(b''.join(bytes.fromhex(w) for s in stages for w in s['bits'])) == COEFFICIENTS,
         'coefficient payload differs by at least one bit')
    zero = '0000000000000000'
    need(dag[0] == dict(op='input', inputs=[], weights=[], constant=zero, depth=0), 'wrong original input')
    for i, n in enumerate(dag[1:], 1):
        parents = n['inputs']
        need(parents and all(type(p) is int and 0 <= p < i for p in parents), 'non-SSA parent')
        depth = max(dag[p]['depth'] for p in parents) + (n['op'] in ('linear','mul'))
        need(n['depth'] == depth, 'wrong multiplicative depth')
    prev, proof_rows = 0, []
    for s in stages:
        end = s['output']
        need(s['input'] == prev and prev < end < len(dag), 'stage input is not previous stage output')
        env = {prev: {1: F(1)}}  # independent indeterminate for this stage only
        for i in range(prev+1, end+1):
            n = dag[i]
            need(all(p in env for p in n['inputs']), 'stage reads across a boundary')
            args = [env[p] for p in n['inputs']]
            c = exact(n['constant'])
            if n['op'] == 'mul':
                need(len(args) == 2 and not n['weights'] and not c, 'wrong multiply')
                val = combine(*args, product=True)
            elif n['op'] in ('linear', 'add'):
                if n['op'] == 'add':
                    need(not n['weights'], 'unexpected add weights')
                    weights = [1]*len(args)
                else:
                    weights = list(map(exact, n['weights']))
                need(len(args) == len(weights), 'wrong number of weights')
                val = {0: c} if c else {}
                for arg, k in zip(args, weights):
                    val = combine(val, {j: k*x for j, x in arg.items()})
            else:
                raise ValueError('wrong stage operation')
            env[i] = val
        row = list(map(exact, s['bits']))
        need(len(row) == s['degree']+1 and env[end] == {i:x for i,x in enumerate(row) if x},
             'DAG does not evaluate to the stated polynomial')
        proof_rows.append(dict(coefficients=row)); prev = end
    half, product, out = prev+1, prev+2, prev+3
    need(site['output'] == out == len(dag)-1 and
         [(n['op'], n['inputs'], n['weights'], n['constant']) for n in dag[half:]] ==
         [('linear',[0],['3fe0000000000000'],zero), ('mul',[half,prev],[],zero), ('add',[half,product],[],zero)],
         'reconstruction did not use the original x/2 twice')
    live = {out}
    for i in range(out, -1, -1):
        if i in live:
            live.update(dag[i]['inputs'])
    need(len(live) == len(dag), 'dead node')
    return proof_rows


def replay(case, raw, ports, w):
    src = raw.decode('ascii')
    need(w['schema'] == 'snabf.native-relu-witness.v1' and sha(raw) == w['source'] == SOURCE[case]
         and sha(ports) == w['input_spec'] == PORT[case], 'source/port byte binding')
    s, = w['sites']; lock = s['source_lock']; name = re.search(r'func.func @(\w+)\(%x:', src)[1]
    port, = read(ports)['inputs']
    need(port == dict(func_symbol=name, arg_index=0, source_id='x', classification='secret', domain=[-1,1]), 'input domain')
    need(lock['kind'] == 'relu' and 'relu_composite' == s['template'] and s['site'] == lock['site'] == lock['result'] == 'n6'
         and lock['x'] == 'n5' and lock['region_x'] == 'n6.arg0' and lock['region_result'] == 'n6.b0'
         and lock['origin'] == dict(kind='source', source_sha256=sha(raw), func_symbol=name, call_path=[], iteration_path=[]),
         'original SSA/site binding')
    b, = lock['body']
    need((b['id'], b['op'], b['inputs']) == ('n6.b0','arith.maximumf',['n6.arg0','n0']) and
         lock['literals'] == [dict(bits='0x00000000', id='n0', role='positive_zero', type='f32')], 'source ReLU operand/literal')
    def loc(needle):
        lines = src.splitlines()
        i, = [i for i, line in enumerate(lines) if needle in line]
        return dict(line=i+1, column=lines[i].index(' = ')+4)
    need(lock['location'] == loc('%relu = linalg.generic') and b['location'] == loc('arith.maximumf') and
         '%zero = arith.constant 0.0 : f32' in src and
         '%p = arith.maximumf %in, %zero : f32\nlinalg.yield %p : f32' in src, 'source operation/location')
    weights = json.loads(re.search(r'%w0 = arith.constant dense<([^>]+)>', src)[1], parse_float=F)
    radii = [sum(map(abs, row)) for row in weights]  # independent input symbols, exact signed row extrema
    slots = [[-x,x] for x in radii]
    axis = [min(x[0] for x in slots), max(x[1] for x in slots)]
    need(radii == [1]*4 and s['affine_slots'] == slots and s['axis'] == lock['declared_domain'] == axis,
         'incomplete per-slot source domain')
    rows = check_dag(s)
    mpfi, budget = s['mpfi160'], F(s['bernstein_budget'])
    need(len(mpfi) == 2 and all(math.isfinite(x) and 0 <= x <= budget for x in mpfi)
         and budget == F(1,1<<15), 'bad proof receipt/budget')
    proof = portable_proof(rows, axis, budget)
    halves = [proof['left'], proof['right']]
    need([x['closed_cells'] for x in halves] == s['cells'] and [x['upper'] for x in halves] == s['bernstein'],
         'full closed-half rational proof mismatch')
    return dict(case=case, source=sha(raw), ports=sha(ports), coefficient_bytes=COEFFICIENTS, dag_nodes=len(s['dag']),
                depth=s['dag'][-1]['depth'], affine_slots=s['affine_slots'], bernstein=proof,
                mpfi_server_receipt=mpfi, ciphertext_replayed=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('case', choices=SOURCE)
    for name in ('source', 'ports', 'contract'):
        ap.add_argument(name, type=Path)
    a = ap.parse_args()
    print(json.dumps(replay(a.case,a.source.read_bytes(),a.ports.read_bytes(),read(a.contract.read_bytes())),
                     sort_keys=True, allow_nan=False))
