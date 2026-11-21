"""Solver-free isolation of native Pass 3 candidate constraints, not an IIS."""
import math
from fractions import Fraction


def need(condition, message):
    if not condition:
        raise ValueError(message)


def uint(s):
    need(isinstance(s, str) and s.isascii() and s.isdecimal() and str(int(s)) == s,
         'noncanonical trace index')
    return int(s)


def finite(s):
    need(isinstance(s, str), 'expected a native binary64 string')
    x = float(s)
    need(math.isfinite(x), 'nonfinite trace scalar')
    return x


def base_width(r, t):
    need((r['stage'], r['status'], uint(r['round'])) == ('model_bounds', 'base_width', 0),
         'base evidence at the wrong stage')
    d, levels = uint(r['scale_bits']), uint(r['q_count'])
    floor, guard, ceiling = (finite(t[k]) for k in ('min_base', 'guard_bits', 'exclusive_ceiling'))
    need(24 <= d <= 52 and 20 <= floor <= 60 and 1 <= guard and 20 < ceiling <= 60,
         'invalid base width policy')
    classes, terms = [], [('min_base', floor)]
    edges = t['edges']
    need(isinstance(edges, list) and edges, 'missing scale DAG')
    unary = {'mask', 'coefficient', 'rescale', 'drop', 'relin', 'add_plain', 'baby', 'giant', 'ring_down'}
    rows = []
    for i, e in enumerate(edges):
        op, level = e['op'], uint(e['level'])
        p = list(map(uint, e.get('inputs', [])))
        need(level < levels and all(j < i for j in p) and
             ((op == 'source' and not p) or (op in unary and len(p) == 1) or
              (op == 'mul' and len(p) == 2) or (op == 'add' and p)), 'invalid scale DAG')
        if op == 'source': cls = 1
        elif op == 'mul': cls = sum(classes[j] for j in p)
        else: cls = classes[p[0]] + (op in ('mask', 'coefficient')) - (op == 'rescale')
        classes.append(cls)
        if level:
            need('magnitude' not in e, 'unexpected nonbase magnitude')
            continue
        mag = finite(e['magnitude'])
        need(mag >= 0, 'negative magnitude')
        # frexp gives exact floor(log2) of the input binary64, even just below a power
        # of two. For integral guard, isolate the exact mathematical row as well as
        # replaying the builder's floating floor; never certify a rounding-only miss.
        exponent = math.frexp(max(1., mag))[1] - 1
        lower = math.floor(cls*d + guard + math.log2(max(1., mag))) - .25
        certain_lower = cls*d + math.floor(guard) + exponent - .25
        terms.append((i, lower))
        rows.append(dict(edge=i, id=e['id'], scale_class=cls, floor_log2_magnitude=exponent,
                         magnitude=e['magnitude'], lower_bits=lower, certain_lower_bits=certain_lower,
                         excludes_alone=min(lower, certain_lower) >= ceiling))
    b0 = max(x for _, x in terms)
    need(b0 == finite(t['builder_lower']) and b0 >= ceiling, 'base bound does not replay')
    # Leave-one-out only tests this construction guard; it does not compile a relaxed graph.
    for row in rows:
        row['lower_without_this_edge'] = max(x for j, x in terms if j != row['edge'])
    return dict(scope='base_width_guard', solver_status='not_called', edges=len(edges),
                min_base_bits=floor, guard_bits=guard, min_base_excludes=floor >= ceiling,
                exclusive_ceiling=ceiling, builder_lower=b0,
                rows=rows, excluded=floor >= ceiling or any(x['excludes_alone'] for x in rows))


def bounds(pair):
    need(isinstance(pair, list) and len(pair) == 2, 'bad bound pair')
    lo, hi = (None if x == inf else Fraction.from_float(finite(x))
              for x, inf in zip(pair, ('-inf', 'inf')))
    need(lo is None or hi is None or lo <= hi, 'inverted bound pair')
    return lo, hi


def lp_rows(r, t):
    status = t['highs_model_status']
    expected = {'Infeasible': 'infeasible', 'Unknown': 'unknown'}.get(status, 'not_optimal')
    need(r['stage'] == 'solver' and 1 <= uint(r['round']) <= 12 and r['status'] == expected and
         status and status != 'Optimal', 'solver evidence at the wrong stage/status')
    nc, nr, qp = (uint(t[k]) for k in ('num_cols', 'num_rows', 'qp_cap_row'))
    cols, rows = t['columns'], t['rows']
    need(isinstance(cols, list) and len(cols) == nc and isinstance(rows, list) and len(rows) == nr
         and qp < nr, 'incomplete model snapshot')
    box = [bounds(c) for c in cols]
    excluded = []
    for i, row in enumerate(rows):
        need(isinstance(row, list) and len(row) >= 2 and len(row) % 2 == 0, 'malformed sparse row')
        lo, hi = bounds(row[:2])
        # Each row gets a FRESH copy of the same box, with all other rows removed
        # and integrality relaxed. Nonempty intersection says nothing about the MIP.
        left = right = Fraction(0)
        seen = set()
        for j, value in zip(row[2::2], row[3::2]):
            j, a = uint(j), Fraction.from_float(finite(value))
            need(j < nc and j not in seen and a != 0, 'invalid sparse column')
            seen.add(j)
            l, u = box[j] if a > 0 else box[j][::-1]
            left = None if left is None or l is None else left + a*l
            right = None if right is None or u is None else right + a*u
        if i == qp:
            need(lo is None and hi == Fraction.from_float(finite(r['security_cap_bits'])) and seen and
                 all(finite(v) == 1 for v in row[3::2]), 'unbound QP cap row')
        if hi is not None and left is not None and left > hi:
            excluded.append(dict(row=i, side='upper', extremum=str(left), bound=str(hi)))
        elif lo is not None and right is not None and right < lo:
            excluded.append(dict(row=i, side='lower', extremum=str(right), bound=str(lo)))
    return dict(scope='one_original_row_plus_continuous_column_box', solver_status=status,
                rows_checked=nr, qp_cap_row=qp, conflicts=excluded, excluded=bool(excluded))


def isolate(r):
    t = r.get('isolation')
    if t is None:
        return None  # never invent inputs for an old run
    need(r.get('reason') == 'unknown' and 'witness' not in r, 'mixed native and diagnostic evidence')
    if t.get('kind') == 'base_width_v1': return base_width(r, t)
    if t.get('kind') == 'individual_lp_rows_v1': return lp_rows(r, t)
    raise ValueError('unknown isolation schema')
