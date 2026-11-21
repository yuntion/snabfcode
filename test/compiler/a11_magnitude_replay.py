"""Independent, deliberately narrow reader of the real A2-a 17Q selected MLIR.

No native trace is an input to reconstruction. Follow bound store/load sites
to binary32 tables and read ONLY poly_selected.nodes, not the Pareto frontier,
budget intervals, error gains, or the native producer's output magnitudes.
"""
import array
from collections import defaultdict
import math
import re
import sys

from pass3_isolation import need


def match(pattern, text):
    m = re.search(pattern, text)
    need(m is not None, '17Q MLIR: missing ' + pattern)
    return m


def vector(text, field, kind):
    s = match(r'\b' + field + r' = array<' + kind + r'(?:: ([^>]*))?>', text)[1]
    return list(map(int if kind == 'i64' else float, s.split(', '))) if s else []


def integer(text, field):
    return int(match(r'\b' + field + r' = (\d+) : i(?:32|64)\b', text)[1])


def scalar(text, field):
    return float(match(r'\b' + field + r' = ([^ ]+) : f64\b', text)[1])


def ordered_sum(terms):
    # This is a replay of the binary64 accumulation, not Python 3.12's
    # compensated sum or a reassociated real-arithmetic polynomial.
    s = 0.0
    for term in terms:
        s += term
    return s


def table_norms(data, rows, slots):
    values = array.array('f')
    values.frombytes(bytes.fromhex(data))
    if sys.byteorder != 'little':
        values.byteswap()
    need((rows, slots) == (64, 4096) and values.itemsize == 4 and len(values) == rows * slots,
         'not the 17Q binary32 diagonal table')
    diagonals = [values[r*slots:(r+1)*slots] for r in range(rows)]
    maxima = [max(map(abs, row)) for row in diagonals]
    # A contrast, NOT used by this Pass 3 guard. These particular weights
    # are small dyadics, exactly summable in binary64.
    slot_norm = max(math.fsum(map(abs, column)) for column in zip(*diagonals))
    return maxima, slot_norm


class Replay:
    def __init__(self):
        self.edges, self.stages = [], []

    def put(self, name, op, level, parents, value, **operands):
        p = [self.edges[i]['cls'] for i in parents]
        if op == 'source':
            cls = 1
        elif op == 'mul':
            cls = sum(p)
        else:
            need(len(set(p)) == 1, 'incompatible nominal scales at ' + name)
            cls = p[0] + (op in ('mask', 'coefficient')) - (op == 'rescale')
        need(math.isfinite(value) and value >= 0 and level >= 0, 'invalid ideal magnitude or level')
        self.edges.append(dict(id=name, op=op, level=level, inputs=list(parents), cls=cls,
                               transfer=dict(decoded_abs=value, **operands)))
        return len(self.edges) - 1

    def mag(self, i):
        return self.edges[i]['transfer']['decoded_abs']

    def keep(self, name, op, level, i):
        return self.put(name, op, level, [i], self.mag(i))

    def add(self, name, level, args):
        if len(args) == 1:
            return args[0]
        return self.put(name, 'add', level, args, ordered_sum(self.mag(i) for i in args))

    def linear(self, site, a, line, weights, slot_norm, offsets):
        selected = line.split('selected = {', 1)[1]
        terms = vector(selected, 'terms', 'i64')
        need(terms[::3] == vector(line, 'diagonals', 'i64') == offsets and len(terms) == 3*len(weights),
             'unbound BSGS route/table rows')
        p = f'lin{site}.'
        level = self.edges[a]['level']
        babies, groups = {0: a}, defaultdict(list)
        for offset, baby, giant, w in zip(terms[::3], terms[1::3], terms[2::3], weights):
            if baby not in babies:
                babies[baby] = self.keep(p+f'baby{baby}', 'baby', level, a)
            j = babies[baby]
            groups[giant].append(self.put(p+f'mask{offset}', 'mask', level, [j], self.mag(j)*w, abs_weight=w))
        rotated = []
        for g in sorted(groups):
            v = self.add(p+f'coset{g}', level, groups[g])
            rotated.append(self.keep(p+f'giant{g}', 'giant', level, v) if g else v)
        merged = self.add(p+'merge', level, rotated)
        out = self.keep(p+'rescale', 'rescale', level-1, merged)
        self.stages.append(dict(stage=p+'rescale', input_decoded_abs=self.mag(a), decoded_abs=self.mag(out),
                                sum_diagonal_max_abs=ordered_sum(weights), max_slot_sum_abs=slot_norm))
        return out

    def poly(self, pi, a, line):
        sel = line.split('poly_selected = {', 1)[1]
        nodes = re.findall(r'\{([^{}]*)\}', match(r'\bnodes = \[([^\]]+)\]', sel)[1])
        level, values = self.edges[a]['level'], []
        for n, node in enumerate(nodes):
            op = match(r'\bop = "(\w+)"', node)[1]
            end = level - integer(node, 'depth')
            start = end + (op in ('mul', 'linear'))
            p = f'poly{pi}.n{n}.'
            args = []
            for j, ref in enumerate(vector(node, 'args', 'i64')):
                v = values[ref]
                args.append(v if self.edges[v]['level'] == start else self.keep(p+f'drop{j}', 'drop', start, v))
            if op == 'mul':
                x, y = args
                v = self.put(p+'mul', 'mul', start, args, self.mag(x)*self.mag(y))
                v = self.keep(p+'relin', 'relin', start, v)
            elif op == 'linear':
                weights = vector(node, 'weights', 'f64')
                need(len(weights) == len(args), 'polynomial arity')
                terms = [self.put(p+f'pt{j}', 'coefficient', start, [v], self.mag(v)*abs(c),
                                  abs_weight=abs(c), signed_literal=c) for j, (v,c) in enumerate(zip(args, weights))]
                v = self.add(p+'add', start, terms)
            elif op == 'add':
                v = self.add(p+'add', start, args)
            else:
                need(op == 'input' and n == 0 and not args, 'unknown polynomial node')
                v = a
            if op in ('mul', 'linear'):
                v = self.keep(p+'rescale', 'rescale', end, v)
            c = scalar(node, 'constant')
            if c:
                v = self.put(p+'constant', 'add_plain', end, [v], self.mag(v)+abs(c), signed_literal=c)
            values.append(v)
        out = values[integer(sel, 'output')]
        self.stages.append(dict(stage=f'poly{pi}', candidate=match(r'candidate_id = "([^"]+)"', sel)[1],
                                input_interval=vector(sel, 'input_interval', 'f64'),
                                selected_output_interval=vector(sel, 'output_interval', 'f64'),
                                input_decoded_abs=self.mag(a), decoded_abs=self.mag(out)))
        return out


def read_selected(path):
    run = Replay()
    tables, prepared, stores, loads, ct = {}, {}, {}, {}, {}
    func, pi = None, 0
    with path.open(encoding='utf-8') as f:
        header = next(f)
        need(header.startswith('module attributes {'), 'expected selected MLIR')
        mode = re.search(r'snabf.interval_mode = "(\w+)"', header)
        run.mode = mode[1] if mode else 'naive'
        for line in f:
            if 'func.func ' in line:
                func = match(r'func.func .*?@(\w+)', line)[1]
            if func == 'deep_stack__preprocessing':
                if 'arith.constant dense<"0x' in line:
                    m = match(r'(%\w+) = arith.constant dense<"0x([0-9A-Fa-f]+)"> : tensor<(\d+)x(\d+)xf32>', line)
                    tables[m[1]] = table_norms(m[2], int(m[3]), int(m[4]))
                elif '"kernel.prepare_linear_transform"' in line:
                    m = match(r'(%\w+) = "kernel.prepare_linear_transform"\((%\w+)\)', line)
                    prepared[m[1]] = (*tables[m[2]], vector(line, 'diagonal_indices', 'i64'))
                elif '"preprocessing.store"' in line:
                    v = match(r'"preprocessing.store"\((%\w+),', line)[1]
                    if v in prepared:
                        slot = integer(line, 'site_id')
                        need(slot not in stores, 'ambiguous prepared table')
                        stores[slot] = prepared[v]
            if func != 'deep_stack__preprocessed':
                continue
            if '"preprocessing.load"' in line:
                loads[match(r'(%\w+) = "preprocessing.load"', line)[1]] = integer(line, 'site_id')
            if ' = "snabf.' not in line:
                continue
            m = match(r'(%\w+) = "snabf\.(\w+)"\(([^)]+)\)', line)
            out, op, args = m[1], m[2], m[3].split(', ')
            if op == 'source':
                need(not ct and scalar(line, 'input_bound') == .5, 'wrong 17Q input')
                ct[out] = run.put('input', 'source', 16, [], scalar(line, 'input_bound'))
            elif op == 'lintrans':
                ct[out] = run.linear(integer(line, 'site'), ct[args[0]], line, *stores[loads[args[1]]])
            elif op == 'poly_eval':
                ct[out] = run.poly(pi, ct[args[0]], line)
                pi += 1
            else:
                need(op == 'sink' and ct[args[0]] == len(run.edges)-1, 'unbound 17Q sink/op')
    need(len(stores) == 4 and pi == 3 and run.edges[-1]['id'] == 'lin3.rescale' and
         run.edges[-1]['level'] == 0, 'wrong frozen source topology')
    return run
