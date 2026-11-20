"""Portable replay of the producer libm's binary64 exp rounding, not its oracle outputs."""
from decimal import Decimal, localcontext
import json
import math

from pass2_evidence import Oracle, load, require, sha


def exact_corpus(a, b):
    # Unlike ==, canonical JSON distinguishes the sign of binary64 zero.
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return canonical(a) == canonical(b)


def freeze(inputs):
    src, data = inputs/'source.mlir', inputs/'corpus.json'
    corpus = load(data)
    require(exact_corpus(Oracle(src.read_text()).corpus(),corpus), 'producer must replay frozen truth byte-for-byte')
    values = {v.hex():math.exp(-v).hex() for row in corpus for v in row['hidden']}
    return dict(schema='snabf.exp64.v1',source_sha256=sha(src),corpus_sha256=sha(data),exp_neg_hidden=values)


def replay(oracle, corpus, locked, path):
    w = load(path)
    require((w['schema'],w['source_sha256'],w['corpus_sha256']) == ('snabf.exp64.v1',locked['source.mlir'],locked['corpus.json']), 'unbound exp witness')
    regenerated = oracle.corpus()  # public input generation and the first dense matvec must still match exactly
    seen = {v.hex() for row in regenerated for v in row['hidden']}
    encoded = w['exp_neg_hidden']
    require(set(encoded) == seen, 'incomplete or surplus exp evaluations')
    values = {}
    with localcontext() as ctx:
        ctx.prec = 90
        for raw,hx in encoded.items():
            h,e = float.fromhex(raw),float.fromhex(hx)
            require(math.isfinite(h) and math.isfinite(e) and e > 0 and e.hex() == hx, 'invalid exp witness')
            ref = Decimal.from_float(h).copy_negate().exp()
            # Decimal.exp is correctly rounded at the working precision. Add one
            # decimal ulp to the distance, so uncertainty cannot enlarge the bound.
            uncertainty = Decimal(1).scaleb(ref.adjusted()-ctx.prec+1)
            require(abs(Decimal.from_float(e)-ref)+uncertainty <= Decimal.from_float(math.ulp(float(ref))), 'exp witness outside one binary64 ulp')
            values[raw] = e
    for case in regenerated:
        h = case['hidden']
        case['truth'] = oracle.suffix(h,[v/(1+values[v.hex()]) for v in h])
    require(exact_corpus(regenerated,corpus), 'witness replay differs from exact frozen source truth')
