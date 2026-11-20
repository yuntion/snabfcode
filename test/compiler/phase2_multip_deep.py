"""Generate a held-out Linear/SiLU source and public binary64 oracle. No compiler dependencies.

--layers L counts dense Linear layers, with true SiLU after the first L-1. No
bias, skip, inverse pair, bootstrap or implied Q chain. --verify replays a bundle;
--levels-from optionally reads actual bound Q and decrypted-output levels from a
completed ablation on that exact source/corpus. Use a fresh output directory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import secrets
import struct


SCHEMA = 'snabf.a1.deep.v1'
WIDTHS = (4, 8, 16, 32, 64)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def packed(obj):
    return (json.dumps(obj, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n').encode('ascii')


def read(path):
    return json.loads(path.read_bytes())


class Stream:
    """Versioned, domain-separated SHA-256 words (independent of Python's sample/shuffle)."""
    def __init__(self, seed, label):
        def words():
            prefix = f'{SCHEMA}/{seed}/{label}/'.encode('ascii')
            i = 0
            while True:
                yield from struct.unpack('>8I', hashlib.sha256(prefix + str(i).encode('ascii')).digest())
                i += 1
        self.words = words()

    def below(self, n):
        limit = (1 << 32) - (1 << 32) % n
        while True:
            w = next(self.words)
            if w < limit:
                return w % n


def matrix(width, layer, seed):
    r = Stream(seed, f'weights/{layer}')
    perm = list(range(width))
    for i in range(width-1, 0, -1):
        j = r.below(i+1)
        perm[i], perm[j] = perm[j], perm[i]
    total = 1024 if layer == 0 else 1392  # exact row norm 1 or 87/64
    rows = []
    for pivot in perm:
        tail = total//8 + r.below(total//3 - total//8 + 1)
        cuts = {0, tail}
        while len(cuts) < width:
            cuts.add(1 + r.below(tail-1))
        cuts = sorted(cuts)
        row = [b-a for a, b in zip(cuts, cuts[1:])]
        row.insert(pivot, total-tail)
        rows.append([x/1024 for x in row])
    return rows


def source_for(layers, width, seed):
    v, m = f'tensor<{width}xf32>', f'tensor<{width}x{width}xf32>'
    lines = ['#id = affine_map<(i) -> (i)>', 'module {',
             f'  func.func @deep_stack(%x: {v} {{secret.secret}}) -> {v} {{',
             '    %zero = arith.constant 0.0 : f32', '    %one = arith.constant 1.0 : f32',
             f'    %empty = tensor.empty() : {v}', f'    %init = linalg.fill ins(%zero : f32) outs(%empty : {v}) -> {v}']
    live = '%x'
    for i in range(layers):
        w = json.dumps(matrix(width, i, seed), separators=(',', ':'))
        lines += [f'    %w{i} = arith.constant dense<{w}> : {m}',
                  f'    %h{i} = linalg.matvec ins(%w{i}, {live} : {m}, {v}) outs(%init : {v}) -> {v}']
        live = f'%h{i}'
        if i == layers-1:
            break
        lines += [f'    %a{i} = linalg.generic {{indexing_maps = [#id, #id], iterator_types = ["parallel"], '
                  f'domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}} ins({live} : {v}) outs(%empty : {v}) {{',
                  '      ^bb0(%in: f32, %ignored: f32):', '        %neg = arith.negf %in : f32',
                  '        %ex = math.exp %neg : f32', '        %den = arith.addf %ex, %one : f32',
                  '        %sig = arith.divf %one, %den : f32', '        %silu = arith.mulf %in, %sig : f32',
                  '        linalg.yield %silu : f32', f'    }} -> {v}']
        live = f'%a{i}'
    return ('\n'.join([*lines, f'    return {live} : {v}', '  }', '}'])+'\n').encode('ascii')


def source_weights(source, layers, width):
    # Read the actual f32 literals, not RNG state, approximants, masks or plans.
    rows = re.findall(r'^    %w(\d+) = arith\.constant dense<(\[.*\])> : tensor<(\d+)x(\d+)xf32>$',
                      source.decode('ascii'), re.M)
    check(len(rows) == layers, 'wrong source matrix count')
    result = []
    for i, (index, data, n, k) in enumerate(rows):
        w = json.loads(data)
        norm = 1.0 if i == 0 else 87/64
        check(int(index) == i and int(n) == int(k) == len(w) == width, 'wrong source shape')
        for row in w:
            check(len(row) == width and all(v > 0 and struct.unpack('<f', struct.pack('<f', v))[0] == v for v in row)
                  and math.fsum(row) == norm and max(row) > norm/2, 'source row lost its dyadic/domain guarantee')
        check(len({row.index(max(row)) for row in w}) == width, 'dominant columns must be a permutation')
        result.append(w)
    return result


def public_input(t, n, seed):
    if t in (0, 1):
        return [1.0 if t == 0 else -1.0] * n
    if t in (2, 3):
        return [1.0 if (j % 2 == 0 if t == 2 else j < n//2) else -1.0 for j in range(n)]
    if t in (4, 5):
        return [(1.0 if t == 4 else -1.0) if j == n//2 else 0.0 for j in range(n)]
    r = Stream(seed, f'input/{t}')
    return [(r.below((1 << 24)+1) - (1 << 23))/(1 << 23) for _ in range(n)]  # exact f32 in [-1,1]


def corpus_for_source(source, layers, width, trials, input_seed):
    matrices = source_weights(source, layers, width)
    cases, peaks = [], [0.0] * layers
    for t in range(trials):
        x = public_input(t, width, input_seed)
        h = x
        for i, w in enumerate(matrices):
            h = [math.fsum(c*v for c, v in zip(row, h, strict=True)) for row in w]
            peaks[i] = max(peaks[i], max(map(abs, h)))
            check(peaks[i] <= 1.0, 'observed value outside the declared full-box bound')
            if i != layers-1:
                h = [v/(1.0+math.exp(-v)) for v in h]
        cases.append(dict(x=x, truth=h))
    return dict(source_sha256=sha(source), cases=cases), peaks


def bundle(layers, width, seed, input_seed, trials):
    check(12 <= layers <= 16 and width in WIDTHS and 2 <= trials <= 4096, 'expected 12..16 layers, supported width, 2..4096 trials')
    check(all(type(v) is int and 0 <= v < (1 << 64) for v in (seed, input_seed)), 'seeds must be uint64')
    source = source_for(layers, width, seed)
    corpus, peaks = corpus_for_source(source, layers, width, trials, input_seed)
    data = packed(corpus)
    manifest = dict(schema=SCHEMA, seed=str(seed), input_seed=str(input_seed), trials=trials, width=width,
                    linear_layers=layers, silu_activations=layers-1, serial_source_ops=2*layers-1,
                    input_bound=1.0, true_silu_domain=[-1, 1], row_l1_norms=[1.0]+[87/64]*(layers-1),
                    domain_argument='|silu(x)| <= 1/(1+exp(-1)) for |x| <= 1, and 87/64 < 1+exp(-1).',
                    oracle='public f32 literals/inputs, binary64 matvec (fsum) and analytic x/(1+exp(-x)); no intermediate f32 rounding',
                    observed_corpus_preactivation_absmax=peaks, source_sha256=sha(source), corpus_sha256=sha(data),
                    actual_q_limb_count=None, actual_max_level_q=None,
                    q_note='No Q in source. --q-count is a compile request; read actual bound Q only from a measured run.')
    return source, data, manifest


def verify(d):
    old = read(d/'manifest.json')
    check(old['schema'] == SCHEMA, 'unknown deep bundle')
    src, data, manifest = bundle(old['linear_layers'], old['width'], int(old['seed']), int(old['input_seed']), old['trials'])
    check((d/'source.mlir').read_bytes() == src and (d/'corpus.json').read_bytes() == data and manifest == old,
          'source, corpus or provenance differs from independent regeneration')
    return old


def runtime_levels(run, manifest):
    record = read(run/'results.json')
    check(record['source_sha256'] == manifest['source_sha256'] == sha((run/'source.mlir').read_bytes()) and
          record['oracle'] == 'external' and sha((run/'external-corpus.json').read_bytes()) == manifest['corpus_sha256'],
          'ablation run is not bound to this exact deep source and external corpus')
    cells = {}
    for name, row in record['cells'].items():
        if 'error' in row:
            cells[name] = dict(measured=False, reason=row['error'], admission=row.get('admission', {}), actual_q_limb_count=None)
            continue
        d = run/('a'+name.replace(':', '-'))
        obs = read(d/'observed.json')
        arms = {}
        with (d/'trials.jsonl').open(encoding='utf8') as f:
            output_levels = {arm: set() for arm in ('no_trunc', 'joint', 'matched_exact')}
            count = 0
            for t in map(json.loads, f):
                for arm, levels in output_levels.items():
                    levels.add(t['arms'][arm]['level'])
                count += 1
        check(count == record['trials'] == row['trials'] and count > 0, 'missing observed trial levels')
        for arm, levels in output_levels.items():
            g = read(d/arm/'program.json')
            binding = obs['bindings'][obs['observed'][arm]['key_group']]
            q = binding['q']
            check(q == row['arms'][arm]['q'] == g['epochs'][0]['q'] and levels == {g['instructions'][g['sink']]['Level']} and
                  0 <= min(levels) < len(q), 'runtime Q or sink level disagrees with emitted ABI')
            arms[arm] = dict(actual_q_limb_count=len(q), actual_max_level_q=len(q)-1, observed_output_level_q=min(levels))
        cells[name] = dict(measured=True, precision_pass=row['precision_pass'], arms=arms)
    return dict(requested_q_count=record['q_count'], cells=cells)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path)
    mode.add_argument('--verify', type=Path)
    p.add_argument('--layers', type=int, default=12, choices=range(12, 17))
    p.add_argument('--width', type=int, default=8, choices=WIDTHS)
    p.add_argument('--seed', type=int, help='public weight seed; omitted: generate and record a fresh uint64')
    p.add_argument('--input-seed', type=int, default=314159)
    p.add_argument('--trials', type=int, default=64)
    p.add_argument('--levels-from', type=Path, help='with --verify: inspect an ablation run after its normal --replay')
    a = p.parse_args()
    if a.verify:
        receipt = verify(a.verify)
        if a.levels_from:
            receipt = runtime_levels(a.levels_from, receipt)
    else:
        if a.levels_from:
            p.error('--levels-from requires --verify')
        src, data, receipt = bundle(a.layers, a.width, secrets.randbits(64) if a.seed is None else a.seed, a.input_seed, a.trials)
        a.out.mkdir(parents=True, exist_ok=False)
        for name, payload in [('source.mlir', src), ('corpus.json', data), ('manifest.json', packed(receipt))]:
            (a.out/name).write_bytes(payload)
    print(json.dumps(receipt, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
