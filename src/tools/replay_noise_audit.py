"""Replay an engineering audit, the B2.1 ledger, or a source-bound exact scale trace."""
import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
from decimal import Decimal, localcontext
from fractions import Fraction as Q
from functools import lru_cache
import exact_scale_replay as exact
from exact_scale_replay import need, pow2, rn


def unique(pairs):
    result = dict(pairs)
    need(len(result) == len(pairs), 'duplicate dictionary key')
    return result


def load(path):
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_int=lambda x: -0.0 if x == '-0' else int(x),
                      parse_constant=lambda x: need(False, 'nonfinite JSON: ' + x))


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def f64(text):
    # MLIR hexadecimal f64 literals are raw bits (not hexadecimal real numbers).
    x = struct.unpack('>d', bytes.fromhex(text[2:]))[0] if text.startswith('0x') else float(text)
    need(math.isfinite(x), 'nonfinite f64')
    return x


class Attributes:
    """Only the printed builtin DictionaryAttr subset used by pass3 and the receipts.
    Unsupported assembly is an error; do not guess a value from a textual log.
    """
    token = re.compile(r'\s*("(?:[^"\\]|\\.)*"|[\w.+-]+|[][{}:,=<>])')

    def __init__(self, text, start):
        self.text, self.pos = text, start

    def pop(self, expected=None):
        m = self.token.match(self.text, self.pos)
        need(m and (expected is None or m[1] == expected), 'unsupported checked attribute at ' + str(self.pos))
        self.pos = m.end()
        return m[1]

    def at(self, token):
        m = self.token.match(self.text, self.pos)
        return bool(m and m[1] == token)

    def seq(self, end, read):
        out = []
        while not self.at(end):
            out.append(read())
            if not self.at(end):
                self.pop(',')
        self.pop(end)
        return out

    def entry(self):
        key = self.pop()
        if not self.at('='):
            return key, None  # builtin unit attribute
        self.pop('=')
        return key, self.value()

    def value(self):
        token = self.pop()
        if token == '{':
            return unique(self.seq('}', self.entry))
        if token == '[':
            return self.seq(']', self.value)
        if token.startswith('"'):
            # The direct ABI currently uses printable ASCII IDs. Reject other
            # spellings rather than misinterpret MLIR's byte escape as JSON.
            need('\\' not in token, 'unsupported escaped identifier')
            return token[1:-1]
        if token == 'array':
            self.pop('<'); typ = self.pop()
            need(typ in ('i64', 'f64'), 'unsupported dense array type')
            if self.at('>'):
                self.pop('>'); return []
            self.pop(':')
            return self.seq('>', lambda: (int if typ == 'i64' else f64)(self.pop()))
        if token in ('true', 'false'):
            return token == 'true'
        self.pop(':'); typ = self.pop()
        need(typ in ('i64', 'f64'), 'unsupported scalar type')
        return (int if typ == 'i64' else f64)(token)


def checked_inputs(path):
    lines = path.read_text(encoding='utf8').splitlines()
    need(lines and lines[0].startswith('module attributes {'), 'expected printed checked module')
    head = lines[0]

    def attr(key):
        # Only match actual module attribute names, not text inside a string.
        pattern = re.compile(r'"(?:[^"\\]|\\.)*"|(?<![\w.])' + re.escape(key) + r'\s*=\s*')
        hits = [m.end() for m in pattern.finditer(head) if not m[0].startswith('"')]
        need(len(hits) == 1, 'missing or duplicate checked ' + key)
        return Attributes(head, hits[0]).value()

    src = [s for s in lines[1:] if re.match(r'^\s*%[\w.]+ = "snabf\.source"\(', s)]
    need(len(src) == 1, 'unsupported input count')
    bounds = re.findall(r'\binput_bound = ([\w.+-]+) : f64', src[0]); need(len(bounds) == 1, 'missing declared input bound')
    b = Q(f64(bounds[0])); need(b >= 0, 'negative input bound')
    return attr('snabf.pass3'), attr('snabf.encoding_receipts'), attr('snabf.emit_sha256'), b


@lru_cache(maxsize=2048)
def exp2_128(log):
    need(math.isfinite(log) and -512 <= log <= 512, 'unsupported encoding exponent')
    if log == int(log):
        return pow2(int(log))
    # Decimal ln and exp are correctly rounded. Widen each endpoint by a
    # decimal ulp; only accept if both map to the same binary128 significand.
    x = Decimal.from_float(log)
    for digits in (80, 160, 320):
        with localcontext() as ctx:
            ctx.prec = digits
            mid = Decimal(2).ln(); a, b = mid.next_minus(), mid.next_plus()
            if x < 0:
                a, b = b, a
            lo = (x * a).next_minus().exp().next_minus()
            hi = (x * b).next_plus().exp().next_plus()
            l, h = rn(Q(lo)), rn(Q(hi))
            if l == h:
                return l
    raise ValueError('ambiguous independent exp2 rounding')


def binding_plan(p, calls=None):
    h = hashlib.sha256()

    def word(x):
        need(type(x) is int and -1 <= x < (1 << 64), 'invalid encoding program integer')
        h.update((x % (1 << 64)).to_bytes(8, 'big'))

    def text(s):
        b = s.encode(); word(len(b)); h.update(b)

    def vec(items):
        word(len(items))
        for x in items:
            word(x)

    text('snabf.encoding-input.v1')
    for k in ('source_ring', 'target_ring', 'width', 'slots', 'sink'):
        word(p[k])
    domains = {d['n']: d for d in p['epochs']}
    need(len(domains) == len(p['epochs']), 'duplicate domain')
    word(len(domains))
    for n, d in sorted(domains.items()):
        word(n); text(d['key_id']); word(d['scale']); vec(d['q']); vec(d['p'])
    ins = p['instructions']; word(len(ins)); rows = []
    for i, e in enumerate(ins):
        op = e['op']; text(e['ID']); text(op)
        for v in (e['n'], e['Level'], e['K'], e.get('rotation', 0), e.get('route_site', -1)):
            word(v)
        h.update(struct.pack('>d', e['physical_log_scale'])); vec(e['inputs'])
        vals = e.get('values', [])
        words = b''.join(struct.pack('>d', v) for v in vals)
        digest = hashlib.sha256(words).hexdigest()
        word(len(vals)); text(digest)
        d = domains[e['n']]; prec = max(53, d['scale'])
        need(all(0 <= j < i for j in e['inputs']) and 1 <= d['scale'] <= 128, 'unbound operand or encoder precision')
        if op in ('source', 'mask', 'coefficient', 'add_plain'):
            if op == 'source':
                need(i == 0 and not vals and not e['inputs'], 'invalid source')
                log = float(d['scale'])
            else:
                need(len(e['inputs']) == 1 and vals and p['slots'] % len(vals) == 0, 'invalid public payload')
                parent = float(ins[e['inputs'][0]]['physical_log_scale'])
                log = parent if op == 'add_plain' else float(e['physical_log_scale']) - parent
            requested = exp2_128(log)  # v1 passes this rounded rational to rlwe.NewScale
            meta, eff = exact.encoding_scales(requested, prec)
            if calls is not None:
                context = dict(call_id='encode:'+e['ID'], site_id=e['ID'], operand_index=0 if op=='source' else 1,
                               key_id=d['key_id'], ring_n=d['n'], q=d['q'], p=d['p'], default_log_scale=d['scale'],
                               level=e['Level'], slots=p['slots'], is_batched=True, arg_type='[]float64',
                               arg_precision_bits=53, encoder_prec_bits=prec, dynamic_input=op=='source')
                calls.append(dict(exact.encode(context, requested, words), log_argument_exact=str(Q(log))))
            rows.append(dict(index=i, node_id=e['ID'], call=op, key_id=d['key_id'], level=e['Level'], ring_n=e['n'],
                             precision=prec, branch='embedDouble/Scale.Float64' if prec == 53 else 'embedArbitrary/Scale.Value',
                             log_argument_exact=str(Q(log)), metadata_scale=str(meta), effective_encoder_scale=str(eff),
                             payload_count=len(vals), payload_sha256=digest))
        else:
            need(not vals, 'orphan payload')
    return dict(schema='encoding-receipt-v1', abi='lattigo/v6.2.0;batched-periodic-f64;mpfr-exp2-rn128;nearest-even',
                program_sha256=h.hexdigest(), receipts=rows)


ROLES = {'source': ('encode', 'encrypt'), 'add': ('add',), 'add_plain': ('plain_encoding_at_output', 'add_plain'),
         'drop': ('level_reduce',), 'rescale': ('rescale_round',), 'mul': ('ct_mul_including_cross_term',),
         'mask': ('plain_encoding_at_output', 'mul_plain'), 'coefficient': ('plain_encoding_at_output', 'mul_plain'),
         'baby': ('permute_input_components', 'evk_switch'), 'giant': ('permute_input_components', 'evk_switch'),
         'relin': ('evk_switch',)}


def independent(checked, params, compare=None, baseline=None, *, source_witness=None, basis=None, compare_scales=None,
                b2i=False, precision_bits=12):
    ir, checked_plan, seal, source_bound = checked_inputs(checked)
    p = load(params)
    need(re.fullmatch(r'[0-9a-f]{64}', seal) and p.get('emit_sha256') == seal, 'different checked/program seal')
    need(p.get('emitter') == 'snabf.direct-go.v1' and p['source_ring'] == p['target_ring'] == ir['ring'] and
         len(p['epochs']) == 1, 'B2.1 independent replay covers single-ring direct programs only')
    domain = p['epochs'][0]
    need(domain['n'] == ir['ring'] and all(same(domain[k], ir[k]) for k in ('q', 'p', 'key_id', 'scale')),
         'checked domain/Q/P mismatch')
    ins = p['instructions']
    need(len(ins) == len(ir['edges']) and 0 <= ir['sink'] == p['sink'] < len(ins), 'checked topology/sink mismatch')
    for e, a in zip(ins, ir['edges']):
        need(all(same(e[k], a[k]) for k in ('ID', 'op', 'Level', 'K', 'inputs')) and
             struct.pack('>d', e['physical_log_scale']) == struct.pack('>d', a['physical_log_scale']) and
             e['n'] == e['kernel_ring'] == ir['ring'], 'checked instruction mismatch: ' + e['ID'])
    need(bool(source_witness) == bool(basis) and (not (compare_scales or b2i) or source_witness),
         'source-bound replay needs both --source-witness and --basis')
    trace = None
    if source_witness:
        from scale_source_binding import bind
        raw = source_witness.read_bytes()
        p = bind(checked, raw, load(basis), p, Attributes, source_bound)
        ins = p['instructions']
        trace = exact.Trace(domain, hashlib.sha256(raw).hexdigest(), seal)
    plan = binding_plan(p, trace.calls if trace else None)
    need(same(plan, checked_plan) and same(plan, p.get('encoding_receipts')), 'missing or stale encoding scale binding')
    by_index = {r['index']: r for r in plan['receipts']}
    edges, sites, states, ledger = [], [], [], []
    missing = {'common/validated_joint_law_and_domain'}
    key = domain['key_id']

    def site(node, name, evk, sink=False):
        id_ = ('sink/' if sink else 'node/') + node + '/' + name
        sites.append(dict(site_id=id_, primitive=name, formula_ref=None)); missing.add(id_)
        ledger.append(dict(node_id=node, role='sink' if sink else 'node', occurrence=0, domain=ir['ring'],
                           key_id=evk if name in ('evk_switch', 'gadget_trunc') else key,
                           obligation_kind=name, status='missing'))

    def edge(e, id_, args, meta, eff, values):
        op, level, k = e['op'], e['Level'], e['K']
        need(op in ROLES and 0 <= level < len(ir['q']) and 0 <= k <= level, 'unbound operation or level at ' + id_)
        keyed = op in ('relin', 'baby', 'giant'); need(not k or keyed, 'unbound truncation')
        arity = 0 if op == 'source' else 2 if op in ('add', 'mul') else 1
        need(len(args) == arity, 'primitive arity mismatch')
        count = 1 if ir['scale'] <= 64 else 2
        for _, _, l, _ in args:
            need(l >= level if op == 'drop' else l - level == (count if op == 'rescale' else 0), 'level mismatch')
        evk = key + ':relin' if op == 'relin' else key + ':rotation:' + str(e['rotation']) if keyed else ''
        a, ba = args[0][:2] if args else (None, source_bound)
        da = db = 0; bb = 0; branch = ma = mb = None
        ideal = ba
        if op == 'source':
            s, da = meta, eff - meta
        elif op in ('mask', 'coefficient', 'mul'):
            b, bb = args[1][:2] if op == 'mul' else (meta, max(abs(Q(v)) for v in values))
            s = rn(a * b); ideal = ba * bb; ba = ideal
            da = a * (b if op == 'mul' else eff) - s
        elif op == 'rescale':
            l = args[0][2]
            row = exact.rescale((a,a), ir['q'], l, ir['q'][level+1:l+1][::-1], ir['scale'])
            s = Q(row['meta_after']); da = Q(row['eff_after']) - s
        elif op in ('add', 'add_plain'):
            b, bb = args[1][:2] if op == 'add' else (meta, max(abs(Q(v)) for v in values))
            # Existing ledger references ciphertext metadata and already carries
            # upstream discrepancies. Do not charge the effective trace again.
            j = exact.join('add',(a,a),(b,b if op=='add' else eff),ba,bb,upper_endpoints=False)
            ideal = ba + bb; s = Q(j['out_meta_scale']); ma,mb = j['lhs_int_mul'],j['rhs_int_mul']
            branch = 'equal' if a==b else 'multiply_left' if a<b else 'multiply_right'
            da,db = Q(j['lhs_signed_effect']),Q(j['rhs_signed_effect'])
        else:
            s = a
        txt = lambda x: None if x is None else str(x)
        edges.append(dict(node_id=id_, key_id=key, evk=evk, ring_n=ir['ring'], level=level, k=k,
                          physical_scale_exact=str(s), ideal_abs_bound=str(ideal), local_scale_bias=str(abs(da)*ba + abs(db)*bb),
                          metadata_scale=txt(meta), effective_encoder_scale=txt(eff), join_branch=branch,
                          multiplier_a=txt(ma), multiplier_b=txt(mb), scale_residual_a=str(da), scale_residual_b=str(db)))
        for role in ROLES[op] + (('gadget_trunc',) if k else ()):
            site(id_, role, evk)
        if trace:
            trace.record(op, id_, level, args, meta, eff, values, s, ideal)
        return s, ideal, level, id_

    for i, e in enumerate(ins):
        args = [states[j] for j in e['inputs']]
        if e['op'] == 'add' and len(args) > 2:
            acc = args[0]
            for j in range(1, len(args)-1):
                acc = edge(e, e['ID'] + '#add' + str(j), [acc, args[j]], None, None, [])
            args = [acc, args[-1]]
        r = by_index.get(i)
        states.append(edge(e, e['ID'], args, Q(r['metadata_scale']) if r else None,
                           Q(r['effective_encoder_scale']) if r else None, e.get('values', [])))
    site('sink0', 'reference_approx_and_decode', '', True)
    ledger.append(dict(node_id='common', role='common', occurrence=0, domain=ir['ring'], key_id=key,
                       obligation_kind='validated_joint_law_and_domain', status='missing'))
    sink_scale = str(states[p['sink']][0]); ids = [e['node_id'] for e in edges]
    need(len(set(ids)) == len(ids), 'duplicate edge IDs')
    result = dict(scope='B2.1-physical-only', seal=seal, program_sha256=plan['program_sha256'],
                  encoding_calls=len(by_index), effective_differs=sum(r['metadata_scale'] != r['effective_encoder_scale'] for r in by_index.values()),
                  edges=len(edges), sites=len(sites), sink_scale=sink_scale, missing=len(missing), obligations=ledger,
                  probabilistic_cert_issued=False)
    if trace:
        result['exact_scale'] = trace.public()
        if compare_scales:
            proposed = load(compare_scales)  # only after source-bound replay
            need(same(proposed, result['exact_scale']), 'source-derived exact scale trace differs')
            result['exact_scale_matched'] = True
    if b2i:
        from b2i_noise_replay import refusal, transfers
        local = transfers(trace, p, edges)
        releases = {r['site_id']: r['formula_ref'] for r in local['transfers'] if r['formula_ref']}
        need(releases.keys() <= missing, 'orphan local proof')
        release = refusal(sink_scale, ir['ring'], precision_bits)
        domains = [dict(key_id=key, ring_n=ir['ring'], q=list(map(str, ir['q'])), p=list(map(str, ir['p'])))]
        physical = dict(release, emit_sha256=seal, encoding_program_sha256=plan['program_sha256'],
                        domains=domains, edges=edges, sites=sites, missing_contract_terms=sorted(missing))
        def match(path, expected, label):
            observed = load(path)  # only after replay; no observed value feeds any bound
            mode = observed.get('mode')
            need(mode in ('engineering', 'probability') and same(observed, dict(expected, mode=mode)), label)
        if baseline is not None:
            match(baseline, physical, 'different pre-B2-i production baseline')
        for s, entry in zip(sites, ledger):
            if s['site_id'] in releases:
                s['formula_ref'] = releases[s['site_id']]
                entry['status'] = 'proved_local'
        missing.difference_update(releases)
        if compare is not None:
            match(compare, dict(physical, b2i_local=local, missing_contract_terms=sorted(missing)),
                  'source-derived B2-i production audit differs')
            result['matched'] = True
        result.update(scope='B2-i-source-bound', b2i_local=local, cleared=sorted(releases),
                      missing_before=result['missing'], missing=len(missing), missing_contract_terms=sorted(missing),
                      refusal=release)
        return result
    if compare is not None:
        audit = load(compare)  # Read ONLY after deriving every row, including folds and local scale bias.
        need(audit.get('emit_sha256') == seal and audit.get('encoding_program_sha256') == plan['program_sha256'], 'different audit/program identity')
        need(audit['errors'] == [] and audit['probabilistic_cert_issued'] is False and audit['all_contracts_present'] is False,
             'B2.1 is not a probability-certificate verifier')
        need(same(audit['edges'], edges) and same(audit['sites'], sites), 'native physical inventory or primitive obligations differ')
        need(same(audit['missing_contract_terms'], sorted(missing)), 'native missing contracts differ')
        need(len(audit['sinks']) == 1 and audit['sinks'][0]['sink_id'] == 'sink0' and
             audit['sinks'][0]['physical_scale_exact'] == sink_scale, 'sink scale mismatch')
        result['matched'] = True
    if baseline is not None:
        old = load(baseline)
        need(old['errors'] == [] and old['probabilistic_cert_issued'] is False and
             same(old['sites'], sites) and [e['node_id'] for e in old['edges']] == ids and
             same(old['domains'], [dict(key_id=key, ring_n=ir['ring'], q=list(map(str, ir['q'])), p=list(map(str, ir['p'])))]),
             'different B1 inventory or basis')
        was = set(old['missing_contract_terms']); resolved = sorted(was - missing)
        need(all(x.endswith('/physical_scale') for x in resolved) and not missing - was, 'unexpected B1 obligation change')
        result['baseline'] = dict(missing_before=len(was), resolved=resolved, missing_after=len(missing))
    return result


def replay(package, native, precision_bits):
    package = package.resolve(strict=True)
    native = native.resolve(strict=True)
    ir = (package / 'work/checked.mlir').resolve(strict=True)
    expected = json.loads((package / 'audits/noise_cert_audit.json').read_bytes())
    env = os.environ.copy()
    env.pop('SNABF_PASS4_ESTIMATES', None)
    evidence = package / 'work/domains.json'
    if evidence.is_file():
        env['SNABF_PASS4_ESTIMATES'] = str(evidence)
    with tempfile.TemporaryDirectory(prefix='snabf-noise-replay-') as stage:
        subprocess.run([native, '--allow-unregistered-dialect', '--verify-each',
                        f'--snabf-check-plan=noise-audit=fresh.json precision-bits={precision_bits}',
                        ir, '-o', os.devnull], cwd=stage, env=env, check=True)
        fresh = json.loads((Path(stage) / 'fresh.json').read_bytes())
    # Canonical comparison also distinguishes bool from int, unlike Python dict equality.
    def canonical(value):
        return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))
    if canonical(expected) != canonical(fresh):
        raise ValueError('stored audit differs from the independently replayed engineering audit')
    return fresh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', nargs='?', type=Path)
    parser.add_argument('--snabf-opt', type=Path)
    parser.add_argument('--independent', action='store_true', help='pure Python scale verification; never issues a noise certificate')
    parser.add_argument('--b2i', action='store_true', help='reconstruct local noise transfers; requires original source and ordered basis')
    parser.add_argument('--checked', type=Path)
    parser.add_argument('--params', type=Path, help='direct program.json with public coefficients, parameters and receipts')
    parser.add_argument('--compare', type=Path, help='optional native audit, read only at final comparison')
    parser.add_argument('--baseline', type=Path, help='optional matching frozen B1 audit (pre-B2-i physical audit for --b2i)')
    parser.add_argument('--source-witness', type=Path, help='original R3/C1 source MLIR (not lowered IR); opts into source-bound replay')
    parser.add_argument('--basis', type=Path, help='independent concrete profile.json with ordered Q/P, logN and logDefaultScale')
    parser.add_argument('--compare-scales', type=Path, help='compare a proposed exact_scale object, including all calls and steps')
    parser.add_argument('--precision-bits', type=int, default=12)
    args = parser.parse_args()
    try:
        if args.independent:
            need(args.checked and args.params and not args.package and not args.snabf_opt, 'use --independent --checked IR --params PROGRAM')
            summary = independent(args.checked, args.params, args.compare, args.baseline,
                                  source_witness=args.source_witness, basis=args.basis, compare_scales=args.compare_scales,
                                  b2i=args.b2i, precision_bits=args.precision_bits)
        else:
            need(args.package and args.snabf_opt and not any((args.checked, args.params, args.compare, args.baseline,
                 args.source_witness, args.basis, args.compare_scales, args.b2i)), 'use PACKAGE --snabf-opt PATH for native engineering replay')
            report = replay(args.package, args.snabf_opt, args.precision_bits)
            summary = dict(matched=True, mode=report['mode'], sinks=len(report['sinks']), probabilistic_cert_issued=report['probabilistic_cert_issued'])
    except (ValueError, TypeError, KeyError, IndexError, ArithmeticError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(summary, allow_nan=False))
