"""Independent N1 evidence checks. Component receipts are not release certificates."""
import hashlib
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


# Digest bytes from the original R3 and C1 sources, not a normalized reprint.
SOURCE = {
    'reuse64': 'c87b0bbd03a024e297088e877619003468bca01d50d95b1c99503297cd5ca8e1',
    'residual': 'c7e8e778ac39fbca30002d7ab4af7b57d130e9fdfe4be8ad491920b07348e1f2',
    'cross_residual': '8dcbe72b6f9b6950206bec2d8c4252214b7d1c8c28b3ab85e283baaf701992e7',
    'single_gelu': '69f3b56c72c261a37a1c229ba8925fa6d07b6da2b3b39240cf4dcd3f7b294ec4',
}
BLOCKED = {'heir-opt', 'heir-translate'}


def source_digest(label, path):
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != SOURCE[label]:
        raise ValueError(f'{label}: original source bytes differ (actual SHA-256 {actual})')
    return actual


def audit_exec(argv, cwd, env=None, *, trace_dir=None):
    """Trace the entire Linux process tree, including failed exec attempts. Never infer from PATH alone."""
    tracer = shutil.which('strace')
    if sys.platform != 'linux' or not tracer:
        raise ValueError('Linux strace required; process tree not verified')
    root = Path(argv[0]).resolve(strict=True)
    if trace_dir is not None:
        Path(trace_dir).mkdir(mode=0o700)
    with (nullcontext(trace_dir) if trace_dir is not None else tempfile.TemporaryDirectory(prefix='n1-exec-')) as td:
        tmp = Path(td)
        # -ff writes a log per task, avoiding interleaved/unfinished exec lines.
        # -xx makes path decoding independent of the locale and quote escapes.
        cmd = [tracer, '-ff', '-qq', '-xx', '-s', '4096', '-e', 'trace=execve,execveat',
               '-o', str(tmp/'exec'), '--', str(root), *map(str, argv[1:])]
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors='replace')
        logs = list(tmp.glob('exec.*'))
        attempted, successful = [], []
        for log in logs:
            for line in log.read_text(encoding='utf-8', errors='strict').splitlines():
                if line.startswith(('--- ', '+++ ')):
                    continue
                m = re.fullmatch(r'execve\("((?:\\x[0-9a-f]{2})+)", .*\)\s+=\s+(-?\d+)(?: .*)?', line)
                if not m:
                    # Includes fd-based execveat and incomplete records: no proof of exclusion.
                    raise ValueError('unresolved exec record; process tree not verified')
                name = os.fsdecode(bytes.fromhex(m[1].replace('\\x', '')))
                path = Path(name)
                if path.name in BLOCKED or (path.is_absolute() and path.resolve().name in BLOCKED):
                    raise ValueError('forbidden HEIR subprocess: '+name)
                attempted.append(name)
                if int(m[2]) == 0:
                    if not path.is_absolute():
                        raise ValueError('relative successful exec cannot be resolved independently')
                    successful.append(path.resolve())
        if not logs or root not in successful:
            raise ValueError('no traced root exec; process tree not verified: '+p.stderr[-256:])
        return p, dict(schema='snabf.exec-trace.v1', trace_files=len(logs), attempts=len(attempted),
                       executables=sorted(set(attempted)), forbidden_execs=0)


def corpus(label, src):
    source_digest(label, src)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'frontend/snabf_frontend'))
    if label == 'single_gelu':
        from gelu_witness import evidence
        return evidence(src, source_literal=True)[0]
    from pass2_evidence import Oracle
    from gelu_witness import atom
    oracle = Oracle(src.read_text(encoding='utf-8'))
    rows = oracle.corpus()
    for row in rows:
        h = row['hidden']
        s = [v/(1+float.fromhex(atom('silu', v.hex())['binary64'])) for v in h]
        row['truth'] = oracle.suffix(h, s)
    return rows


def replay_slots(label, source, trials, *, seal, ring, level, slots, scale=None):
    """Offline only: verify runner rows against original source, not a printed error or truth vector."""
    if (not re.fullmatch(r'[0-9a-f]{64}', seal) or type(ring) is not int or ring < 2 or ring & (ring-1)
            or type(level) is not int or level < 0 or type(slots) is not int or not 1 <= slots <= ring//2):
        raise ValueError('invalid independently supplied sink ABI')
    rows = corpus(label, source)
    if len(rows) != 64 or slots % len(rows[0]['truth']):
        raise ValueError('wrong input coverage or unsupported periodic layout')
    def unique(pairs):
        d = dict(pairs)
        if len(d) != len(pairs): raise ValueError('duplicate observation field')
        return d
    peak = 0.0
    with Path(trials).open(encoding='utf-8') as f:
        for i, (line, case) in enumerate(zip(f, rows, strict=True)):
            r = json.loads(line, object_pairs_hook=unique)
            if (r['trial'] != i or r['emit_sha256'] != seal or r['source_sha256'] != SOURCE[label]
                    or r['ring'] != ring or r['level'] != level or len(r['decoded']) != slots
                    or r['x'] != case['x']):
                raise ValueError(f'trial {i}: wrong source, input, plan or sink ABI')
            truth = case['truth']
            if scale is not None:
                if (r.keys() != {'schema','emit_sha256','source_sha256','x','ring','level','slots','scale','trial','decoded','imag'}
                        or r['schema'] != 'snabf.n1-complex-slots.v1' or r['scale'] != scale
                        or any(type(r[k]) is not int for k in ('ring','level','trial','slots')) or r['slots'] != slots
                        or not isinstance(r['imag'], list) or len(r['imag']) != slots or not isinstance(r['decoded'], list)
                        or any(type(v) not in (float,int) for v in r['x'])):
                    raise ValueError(f'trial {i}: invalid complex slot record')
            for j, v in enumerate(r['decoded']):
                im = r['imag'][j] if scale is not None else 0.
                if type(v) not in (float, int) or type(im) not in (float,int):
                    raise ValueError('non-numeric decoded slot')
                err = math.hypot(v-truth[j % len(truth)], im)
                if not math.isfinite(err) or err >= 2**-12:
                    raise ValueError(f'trial {i} slot {j}: error {err} is not below 2^-12')
                peak = max(peak, err)
    return dict(schema='snabf.n1-slot-replay.v2' if scale is not None else 'snabf.n1-slot-replay.v1', graph=label, emit_sha256=seal,
                source_sha256=SOURCE[label], inputs=len(rows), slots_checked=len(rows)*slots, linf=peak)


def replay_batch(root, expected_manifest):
    """Offline only: verify the transferred bytes before interpreting any row."""
    root = Path(root)
    raw = (root/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest: raise ValueError('wrong batch digest')
    m = json.loads(raw)
    if m['schema'] != 'snabf.n1-offline.v1' or set(m['graphs']) != set(SOURCE): raise ValueError('incomplete batch')
    reports = {}
    for label, abi in m['graphs'].items():
        src, records = root/(label+'.mlir'), root/(label+'.jsonl')
        if hashlib.sha256(records.read_bytes()).hexdigest() != abi['raw_sha256']: raise ValueError('wrong JSONL digest: '+label)
        reports[label] = replay_slots(label, src, records, **{k:abi[k] for k in ('seal','ring','level','slots','scale')})
        reports[label]['raw_sha256'] = abi['raw_sha256']
    return dict(head=m['head'], tree=m['tree'], platform=sys.platform, manifest_sha256=expected_manifest, graphs=reports)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Offline original-source replay of a native N1 observation batch')
    ap.add_argument('batch', type=Path); ap.add_argument('manifest_sha256')
    args = ap.parse_args()
    print(json.dumps(replay_batch(args.batch, args.manifest_sha256), indent=2))
