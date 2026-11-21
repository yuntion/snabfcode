"""Independent read-only structural oracle for the frozen four-file grammar, not an MLIR frontend."""
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = '8e8f7ce374227a3d8b70a548329d0a2b55671cb5'
R3 = Path('doc/e2e-matrix-2026-11-20-r3/frozen')
HASHES = {
    'reuse64': 'c87b0bbd03a024e297088e877619003468bca01d50d95b1c99503297cd5ca8e1',
    'residual': 'c7e8e778ac39fbca30002d7ab4af7b57d130e9fdfe4be8ad491920b07348e1f2',
    'cross_residual': '8dcbe72b6f9b6950206bec2d8c4252214b7d1c8c28b3ab85e283baaf701992e7',
    'c1-single': '69f3b56c72c261a37a1c229ba8925fa6d07b6da2b3b39240cf4dcd3f7b294ec4',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def check(value, why):
    if not value:
        raise ValueError(why)


def read_receipt(raw):
    def unique(pairs):
        out = dict(pairs)
        check(len(out)==len(pairs), 'duplicate receipt key')
        return out
    return json.loads(raw, object_pairs_hook=unique)


def original(case, c1=None):
    if case == 'c1-single':
        data = (c1 or HERE/'c1-single.mlir').read_bytes()
    else:
        rel = R3/case/'source.mlir'
        data = (ROOT/rel).read_bytes()
        if sha(data) != HASHES[case]:
            # A Windows text checkout is not the archive. Read the immutable
            # blob as bytes, not through a terminal, codec or newline filter.
            data = subprocess.check_output(['git', 'cat-file', 'blob', f'{BASE}:{rel.as_posix()}'], cwd=ROOT)
    check(sha(data) == HASHES[case], f'{case}: not the frozen source bytes')
    return data


def spec_for(case):
    return (HERE/('c1.inputs.json' if case == 'c1-single' else 'r3.inputs.json')).read_bytes()


def words(values, ty):
    return ['0x' + struct.pack('>f' if ty == 'f32' else '>d', v).hex() for v in values]


def ssa(text):
    return re.findall(r'%[\w]+', text)


def expected(source, spec):
    """Reconstruct ordered edges and physical locations directly from the fixture bytes."""
    lines = source.decode('ascii').splitlines()
    port = json.loads(spec)['inputs'][0]
    symbol, arg, vector, width, ty = re.search(
        r'func\.func @([\w]+)\((%\w+): (tensor<(\d+)x(f32|f64)>)', '\n'.join(lines)).groups()
    check((port['func_symbol'], port['arg_index']) == (symbol, 0), 'wrong spec port')
    common = dict(func_symbol=symbol, logical_type=vector, layout='periodic', period=int(width))
    inputs = [dict(common, arg_index=0, id='arg0', source_id=port['source_id'], classification='secret',
                   secret_marker='{secret.secret}' in next(l for l in lines if 'func.func @' in l), domain=port.get('domain'))]
    ids = {arg: 'arg0'}
    constants, nodes, locks, outputs = {}, [], [], []
    def loc(i, token):
        return dict(line=i+1, column=lines[i].index(token)+1)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('return '):
            outputs.append(dict(common, result_index=0, value=ids[ssa(line)[0]], location=loc(i, 'return')))
        match = re.match(r'(%\w+) = ([\w.]+)\s*(.*)', line)
        if not match:
            i += 1
            continue
        var, op, tail = match.groups()
        id_ = f'n{len(nodes)}'
        node = dict(id=id_, op=op, inputs=[], logical_type=vector, location=loc(i, op))
        if op == 'arith.constant':
            val, _, t = tail.rpartition(' : ')
            if val.startswith('dense<'):
                a = json.loads(val[6:-1]); shape = [len(a), len(a[0])]; vals = [v for row in a for v in row]
            else:
                shape, vals = [], [float(val)]
            bits = words(vals, ty)
            node.update(kind='constant', shape=shape, bits=bits, logical_type=t, element_type=ty)
            if not shape:
                constants[id_] = bits[0]
        elif op == 'tensor.empty':
            node['kind'] = 'empty'
        elif op.startswith('linalg.'):
            header = tail
            while 'outs(' not in header:
                i += 1; header += ' ' + lines[i].strip()
            operands = [ids[v] for item in re.findall(r'(?:ins|outs)\(([^:]+):', header) for v in ssa(item)]
            node['inputs'] = operands
            if op == 'linalg.matvec':
                check(len(operands)==3, 'matvec arity'); node.update(kind='matvec', outs=operands[2])
            elif op == 'linalg.fill':
                check(constants[operands[0]] == words([0], ty)[0], 'source has nonzero fill'); node['kind'] = 'zero_fill'
            else:
                check(op == 'linalg.generic', op)
                bounds = [float(re.search(rf'domain_{k} = ([^ ]+)', header)[1]) for k in ('lower', 'upper')]
                i += 1
                x, old = ssa(lines[i]); local = {**ids, x: id_+'.arg0'}
                expr = {**constants, local[x]: 'x'}; body = []
                def term(opcode, args):
                    return (opcode, *sorted(args, key=repr)) if opcode in ('arith.mulf', 'arith.addf') else (opcode, *args)
                i += 1
                while 'linalg.yield' not in lines[i]:
                    lhs, expression = lines[i].strip().split(' = '); code, uses = expression.split(' ', 1)
                    check(old not in ssa(uses), 'output argument used')
                    args = [local[v] for v in ssa(uses)]; bid = id_+f'.b{len(body)}'
                    body.append(dict(id=bid, op=code, inputs=args, location=loc(i, code)))
                    expr[bid] = term(code, [expr[v] for v in args]); local[lhs] = bid; i += 1
                yielded = local[ssa(lines[i])[0]]
                one, half, inv = words([1, .5, float.fromhex('0x1.6a09e667f3bcdp-1')], ty)
                silu = term('arith.mulf', ['x', ('arith.divf', one, term('arith.addf', [one, ('math.exp', ('arith.negf', 'x'))]))])
                gelu = term('arith.mulf', [term('arith.mulf', ['x', half]), term('arith.addf', [one, ('math.erf', term('arith.mulf', ['x', inv]))])])
                check(expr[yielded] in (silu, gelu) and len(body)==5, 'wrong source activation tree')
                kind = 'silu' if expr[yielded]==silu else 'gelu_exact_erf'; node['kind'] = kind
                # Role IDs are recovered from the actual source use, not the receipt.
                captured = [(step['op'], val, constants[val]) for step in body for val in step['inputs'] if val in constants]
                queries = [('one','arith.divf',one), ('one','arith.addf',one)] if kind=='silu' else [
                    ('half','arith.mulf',half), ('one','arith.addf',one), ('inv_sqrt2','arith.mulf',inv)]
                literals = [dict(role=role, id=next(val for op_,val,bits in captured if (op_,bits)==(code,word)),
                                 type=ty, bits=word) for role,code,word in queries]
                locks.append(dict(site=id_, kind=kind, x=operands[0], result=id_, region_x=local[x], region_result=yielded,
                                  body=body, literals=literals, location=node['location'], declared_domain=bounds,
                                  origin=dict(kind='source', source_sha256=sha(source), func_symbol=symbol,
                                              call_path=[], iteration_path=[])))
        else:
            check(op == 'arith.addf', op); node.update(kind='add', inputs=[ids[v] for v in ssa(tail)])
        ids[var] = id_; nodes.append(node); i += 1
    return dict(schema='snabf.native-import.v1', frontend='native-linalg', stage='structural-inspection',
                source_sha256=sha(source), input_spec_sha256=sha(spec), func_symbol=symbol, inputs=inputs,
                outputs=outputs, nodes=nodes, activation_locks=locks)


def verify(source, spec, receipt):
    ref = expected(source, spec)
    def same(a, b):
        # Python equates false with 0. An index/edge field must not silently
        # become a boolean merely because its numerical value is zero.
        if isinstance(a,bool) or isinstance(b,bool): return type(a) is type(b) and a == b
        if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
        if isinstance(a,list): return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        return a == b
    check(set(receipt) == set(ref), 'receipt keys')
    for key in ref:
        check(same(ref[key],receipt[key]), f'not bound to source/spec: {key}')
