"""Unseen C2 sources. No run artifacts are required to regenerate either graph."""
import json
import random


def source(seed, mixed):
    rng = random.Random(seed)
    v, m = 'tensor<4xf32>', 'tensor<4x4xf32>'
    lines = ['#id = affine_map<(i) -> (i)>', 'module {',
             f'func.func @f{rng.getrandbits(60):x}(%x: {v} {{secret.secret}}) -> {v} {{',
             '%zero = arith.constant 0.0 : f32', '%one = arith.constant 1.0 : f32',
             f'%empty = tensor.empty() : {v}', f'%init = linalg.fill ins(%zero : f32) outs(%empty : {v}) -> {v}']

    def linear(i, x):
        # Every input row has l1 norm exactly one. Signed corners attain both
        # endpoints of the declared full source domain at the first activation.
        row = [1/16, 3/16, 5/16, 7/16]; rng.shuffle(row)
        row = [c*rng.choice((-1, 1)) for c in row]
        w = [row[-j:] + row[:-j] if j else row[:] for j in range(4)]
        lines.extend([f'%w{i} = arith.constant dense<{json.dumps(w)}> : {m}',
                      f'%l{i} = linalg.matvec ins(%w{i}, {x} : {m}, {v}) outs(%init : {v}) -> {v}'])
        return f'%l{i}'

    def activation(x, kind):
        lines.extend([f'%{kind} = linalg.generic {{indexing_maps = [#id, #id], iterator_types = ["parallel"], domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}} ins({x} : {v}) outs(%empty : {v}) {{',
                      '^bb0(%in: f32, %old: f32):'])
        if kind == 'relu': lines.append('%p = arith.maximumf %in, %zero : f32')
        else: lines.extend(['%n = arith.negf %in : f32', '%e = math.exp %n : f32', '%d = arith.addf %one, %e : f32',
                            '%s = arith.divf %one, %d : f32', '%p = arith.mulf %in, %s : f32'])
        lines.extend(['linalg.yield %p : f32', f'}} -> {v}'])
        return '%'+kind

    h = linear(0, '%x'); r = activation(h, 'relu')
    if mixed:
        s = activation(h, 'silu'); lines.append(f'%sum = arith.addf {r}, {s} : {v}'); r = '%sum'
    y = linear(1, r)
    if mixed: lines.append(f'%res = arith.addf {y}, {h} : {v}'); y = '%res'
    return '\n'.join([*lines, f'return {y} : {v}', '}', '}', ''])
