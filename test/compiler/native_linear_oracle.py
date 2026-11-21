"""Frozen-source oracle. No compiler bindings, HEIR artifacts, or ciphertext library."""
import hashlib
import json
import math
import operator
import struct
from collections import defaultdict

from native_import import oracle as src


def same(actual, expected, field):
    # Unlike Python equality this distinguishes false, 0 and 0.0.
    src.check(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True), field)


def signed_support(rows, slots, group):
    w = len(rows)
    by_d = {}
    for i, row in enumerate(rows):
        for j, word in enumerate(row):
            # Assign original entries to diagonals by their pair of coordinates.
            if int(word, 16) == 0:
                continue
            d = (j - i + w // 2) % w - w // 2
            zero = '0x' + '0' * (len(word) - 2)
            if d not in by_d:
                by_d[d] = [zero] * w
            by_d[d][i] = word
    return [dict(signed_offset=d, encoded_offset=d % slots, physical_offset=d % group, row_bits=by_d[d])
            for d in sorted(by_d, key=lambda d: d % group)]


def candidates(support, ring):
    group = ring // 2
    order = group // math.gcd(group, *support)
    out = []
    for rung in range(order.bit_length()):
        stride = group // (2**rung)
        terms = [[d, (d // stride) * stride, d % stride] for d in support]
        split = [rung, stride, 2**rung, order // (2**rung)]
        if out and out[-1]['terms'] == terms:
            out[-1]['equivalent_splits'].append(split)
        else:
            b, g = (sorted({term[i] for term in terms} - {0}) for i in (1, 2))
            out.append(dict(terms=terms, equivalent_splits=[split], baby_rotations=b, giant_rotations=g,
                            galois_keys=sorted({pow(5, step, 2 * ring) for step in b + g})))
    return out


def structure(raw, spec, receipt, log_n, slots):
    same(sorted(receipt), sorted(('schema', 'stage', 'inspection', 'layout', 'matvecs')), 'root keys')
    same(receipt['schema'], 'snabf.native-linear.v1', 'schema')
    same(receipt['stage'], 'pass1-route-candidates', 'stage')
    graph = src.expected(raw, spec)
    src.verify(raw, spec, receipt['inspection'])
    w = graph['inputs'][0]['period']
    ring = 1 << log_n
    same(receipt['layout'], dict(kind='periodic_single', rotate_direction='left', ring_dimension=ring,
         rotation_modulus=ring//2, encoded_slots=slots, logical_width=w, first_slot_for_row=list(range(w)),
         replica_stride=w, replica_count=slots//w), 'physical slot mapping')
    nodes = {n['id']: n for n in graph['nodes']}
    matrices = []
    expected_sites = []
    for node in graph['nodes']:
        if node['kind'] != 'matvec':
            continue
        weight, data, outs = node['inputs']
        literal = nodes[weight]
        src.check(literal['shape'] == [w, w], 'source weight shape')
        words = literal['bits']
        rows = [words[k:k+w] for k in range(0, w*w, w)]
        matrices.append(rows)
        ds = signed_support(rows, slots, ring // 2)
        src.check(nodes[outs]['kind'] == 'zero_fill', 'original matvec must use linalg.fill(+0)')
        expected_sites.append(dict(result_id=node['id'], input_id=data, weight_id=weight, outs_id=outs,
            outs_kind='positive-zero-fill', element_bits=int(literal['element_type'][1:]),
            weight_sha256=hashlib.sha256(b''.join(bytes.fromhex(x[2:]) for x in words)).hexdigest(),
            diagonals=ds, route_candidates=candidates([d['physical_offset'] for d in ds], ring)))
    same(len(receipt['matvecs']), len(expected_sites), 'site count')
    for got, ref in zip(receipt['matvecs'], expected_sites):
        same(sorted(got), sorted(ref), 'site fields')
        for field, value in ref.items():
            same(got[field], value, field)
    return matrices


def rotate_left(xs, d):
    d %= len(xs)
    return xs[d:] + xs[:d]


def basis(rows, cosets, slots):
    """Multiply each e_k using the physical baby/giant program and actual C++ masks."""
    w = len(rows)
    for k in range(w):
        out = [0.0] * slots
        for giant, terms in cosets.items():
            partial = [0.0] * slots
            for baby, mask in terms:
                # Rot_b(tile(e_k)) is 1 at exactly these encoded slots.
                for j in range((k-baby) % w, slots, w):
                    partial[j] += mask[j]
            out = list(map(operator.add, out, rotate_left(partial, giant)))
        truth = [row[k] for row in rows] * (slots // w)  # original matrix column, not a second diagonal algorithm
        src.check(out == truth, f'wrong output for basis e_{k}')
    return w * slots


def replay(receipt, matrices, packed, *, check_mask_bits=True):
    """Check encoded words separately from real-linear basis action (which cannot see -0)."""
    pos, stats = 0, dict(sites=0, bases=0, candidates=0, masks=0, slot_checks=0)
    slots = receipt['layout']['encoded_slots']
    for site, rows in zip(receipt['matvecs'], matrices):
        w = len(rows)
        length = site['element_bits'] // 8
        fmt = '>f' if length == 4 else '>d'
        raw_rows = [[bytes.fromhex(word[2:]) for word in row] for row in rows]
        real_rows = [[struct.unpack(fmt, word)[0] for word in row] for row in raw_rows]
        stats['sites'] += 1
        stats['bases'] += w
        for route in site['route_candidates']:
            cosets = defaultdict(list)
            for _, baby, giant in route['terms']:
                chunk = packed[pos:pos + slots*length]
                src.check(len(chunk) == slots*length, 'truncated mask stream')
                if check_mask_bits:
                    # At ciphertext slot j before Rot_g the result row is j-g;
                    # the input after Rot_b is j+b. Index the SOURCE matrix.
                    one = b''.join(raw_rows[(j-giant) % w][(j+baby) % w] for j in range(w))
                    src.check(chunk == one * (slots//w), 'wrong materialized mask bits')
                cosets[giant].append((baby, [v for v, in struct.iter_unpack(fmt, chunk)]))
                pos += len(chunk)
                stats['masks'] += 1
            stats['slot_checks'] += basis(real_rows, cosets, slots)
            stats['candidates'] += 1
    src.check(pos == len(packed), 'extra bytes in mask stream')
    return stats
