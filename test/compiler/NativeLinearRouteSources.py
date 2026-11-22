"""Four byte-locked native matvec route tests; optionally recheck saved witnesses offline."""
import argparse
import copy
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from native_import import oracle as src
from native_linear_oracle import candidates, replay, signed_support, structure


LOG_N, SLOTS = 14, 4096  # Explicit test domain, not inferred from a source tensor.


def denied(test, reason):
    try:
        test()
    except ValueError as e:
        src.check(reason in str(e), f'unexpected refusal: {e}; wanted {reason}')
        return 1
    raise ValueError(f'tamper survived: {reason}')


def receipt_faults(raw, spec, rec, rows, case):
    count = 0

    def mutate(fn, reason):
        other = copy.deepcopy(rec)
        fn(other)
        return denied(lambda: structure(raw, spec, other, LOG_N, SLOTS), reason)

    def swap_sites(r):
        a, b = r['matvecs']
        for key in ('weight_id', 'weight_sha256', 'diagonals', 'route_candidates'):
            a[key], b[key] = b[key], a[key]

    def transpose(r):
        site = r['matvecs'][0]
        m = list(zip(*rows[0]))
        flat = [v for row in m for v in row]
        next(n for n in r['inspection']['nodes'] if n['id'] == site['weight_id'])['bits'] = flat
        site['weight_sha256'] = src.sha(b''.join(bytes.fromhex(x[2:]) for x in flat))
        site['diagonals'] = signed_support(m, SLOTS, (1 << LOG_N)//2)
        site['route_candidates'] = candidates([d['physical_offset'] for d in site['diagonals']], 1 << LOG_N)

    def wrong_lift(r):
        site = r['matvecs'][0]
        for d in site['diagonals']:
            d['physical_offset'] = d['encoded_offset']  # same clear slots; WRONG key domain for d < 0
        site['route_candidates'] = candidates(sorted(d['physical_offset'] for d in site['diagonals']), 1 << LOG_N)

    def wrong_port(r):
        ports = json.loads(spec)
        ports['inputs'][0]['source_id'] = 'another-input'
        r['inspection']['inputs'][0]['source_id'] = 'another-input'
        r['inspection']['input_spec_sha256'] = src.sha(json.dumps(ports).encode())

    count += mutate(lambda r: r['layout']['first_slot_for_row'].reverse(), 'physical slot mapping')
    count += mutate(lambda r: r['layout'].update(replica_count=1), 'physical slot mapping')
    count += mutate(lambda r: r['layout'].update(rotation_modulus=SLOTS), 'physical slot mapping')
    count += mutate(swap_sites, 'weight_id')
    count += mutate(transpose, 'nodes')
    count += mutate(lambda r: r['matvecs'][1].update(input_id='arg0'), 'input_id')
    count += mutate(lambda r: r['matvecs'][0].update(outs_id='arg0'), 'outs_id')
    count += mutate(wrong_port, 'input_spec_sha256')
    count += mutate(lambda r: r['matvecs'][0]['route_candidates'].pop(), 'route_candidates')
    count += denied(lambda: structure(raw + b'\n', spec, rec, LOG_N, SLOTS), 'source_sha256')
    if case == 'c1-single':
        count += mutate(wrong_lift, 'diagonals')
        count += mutate(lambda r: r['matvecs'][0]['diagonals'][-1].update(signed_offset=3), 'diagonals')
    return count


def mask_faults(rec, rows, packed):
    # Disable the word-by-word check: the independent basis evaluation itself
    # must detect a wrong last replica and a positive-vs-negative giant error.
    def reject(p):
        return denied(lambda: replay(rec, rows, p, check_mask_bits=False), 'wrong output for basis')

    n = rec['matvecs'][0]['element_bits']//8
    broken = bytearray(packed)
    broken[(SLOTS-1)*n:SLOTS*n] = bytes.fromhex('42f60000' if n == 4 else '405ec00000000000')
    count = reject(broken)
    pos = 0
    for site in rec['matvecs']:
        by_phys = {d['physical_offset']: d['row_bits'] for d in site['diagonals']}
        for route in site['route_candidates']:
            for diagonal, _, giant in route['terms']:
                words = by_phys[diagonal]
                w = len(words)
                wrong = b''.join(bytes.fromhex(words[(j+giant) % w][2:]) for j in range(w)) * (SLOTS//w)
                if wrong != packed[pos:pos + len(wrong)]:
                    return count + reject(packed[:pos] + wrong + packed[pos+len(wrong):])
                pos += len(wrong)
    raise ValueError('no orientation-sensitive mask in fixture')


def live_faults(exe, root, raw, spec):
    source, ports = root/'bad.mlir', root/'bad.inputs.json'
    count = 0

    def reject(text, port_spec, options, diag, status=1):
        source.write_bytes(text)
        ports.write_bytes(port_spec)
        p = subprocess.run([exe, source, '--input-spec', ports, *options], capture_output=True, timeout=20)
        src.check(p.returncode == status and not p.stdout and diag.encode() in p.stderr, f'wrong CLI refusal: {p.stderr!r}')
        return 1

    for ln, e in ((1, 1), (31, 4096), (14, 8193), (14, 16384), (14, 3), (14, 0), (14, 2)):
        count += reject(raw, spec, ['--log-n', str(ln), '--encoded-slots', str(e)], 'SNABF_NATIVE_LINEAR_LAYOUT')
    args = ['--log-n', str(LOG_N), '--encoded-slots', str(SLOTS)]
    for wrong in (['--log-n', '14'], ['--encoded-slots', '4096'], args + ['--log-n', '13'], args + ['--stride', '2']):
        count += reject(raw, spec, wrong, 'usage:', 2)
    count += reject(raw.replace(b'outs(%init', b'outs(%x', 1), spec, args, 'SNABF_NATIVE_OUTS')
    count += reject(raw.replace(b'0.0 : f32', b'-0.0 : f32', 1), spec, args, 'SNABF_NATIVE_OUTS')
    count += reject(raw.replace(b'return %l1', b'%bad = math.sin %l1 : tensor<4xf32>\nreturn %l1'),
                    spec, args, 'SNABF_NATIVE_OP')
    count += reject(raw, b'{}', args, 'SNABF_NATIVE_INPUT_SPEC')
    return count


def run(args):
    totals = Counter()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for case in src.HASHES:
            raw, spec = src.original(case), src.spec_for(case)
            if args.offline:
                doc = (args.offline / (case+'.json')).read_bytes()
                masks = (args.offline / (case+'.masks')).read_bytes()
            else:
                source, ports = root/(case+'.mlir'), root/(case+'.inputs.json')
                source.write_bytes(raw)
                ports.write_bytes(spec)
                doc = subprocess.check_output([args.exe, source, '--input-spec', ports, '--log-n', str(LOG_N),
                                               '--encoded-slots', str(SLOTS)], timeout=30)
                masks = subprocess.check_output([args.probe, '--masks', source, ports, str(LOG_N), str(SLOTS)], timeout=30)
            rec = src.read_receipt(doc)
            rows = structure(raw, spec, rec, LOG_N, SLOTS)
            stats = replay(rec, rows, masks)
            stats['receipt_refusals'] = receipt_faults(raw, spec, rec, rows, case)
            stats['basis_refusals'] = mask_faults(rec, rows, masks)
            if case == 'c1-single' and not args.offline:
                stats['cli_refusals'] = live_faults(args.exe, root, raw, spec)
            if args.save:
                (args.save/(case+'.json')).write_bytes(doc)
                (args.save/(case+'.masks')).write_bytes(masks)
            print(f'{case}: {src.sha(raw)} {stats}', flush=True)
            totals.update(stats)
    print('total:', dict(totals), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('exe', nargs='?', type=Path)
    ap.add_argument('probe', nargs='?', type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--offline', type=Path, help='verify saved witnesses without executing either binary')
    mode.add_argument('--save', type=Path, help='write checked witnesses to a NEW directory')
    args = ap.parse_args()
    if (args.offline and (args.exe or args.probe)) or (not args.offline and not (args.exe and args.probe)):
        ap.error('pass EXE PROBE, or --offline DIR')
    if args.save:
        args.save.mkdir()
    run(args)
