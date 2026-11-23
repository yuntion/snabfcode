"""N1 four-source probe and offline slot replay. Inspection and release are separate gates."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/tools'))
from verify_n1_equivalence import SOURCE, audit_exec, replay_slots, source_digest


def store(path, record):
    path.write_bytes((json.dumps(record, indent=2, sort_keys=True, allow_nan=False)+'\n').encode())


def probe(a):
    r3, gelu, out = a.r3_frozen.resolve(strict=True), a.single_gelu.resolve(strict=True), a.out.absolute()
    paths = {g: (gelu if g == 'single_gelu' else r3/g/'source.mlir') for g in SOURCE}
    pins = {g: source_digest(g, p) for g, p in paths.items()}
    out.mkdir(parents=True, exist_ok=False)
    cases = {}
    for g, p in paths.items():
        case = out/g
        case.mkdir()
        symbol = 'f88280ebd6b8f89e' if g == 'single_gelu' else 'linear_silu_linear'
        spec = case/'input-spec.json'
        store(spec, dict(schema='snabf.native-input-spec.v1', inputs=[dict(func_symbol=symbol, arg_index=0,
                      source_id='x', classification='secret', domain=[-1,1])]))
        argv = [sys.executable, '-B', ROOT/'src/tools/snabf_compile.py', '--frontend','native-linalg',
                '--mlir', p, '--input-bound','1','--out-dir',case/'out','--snabf-opt', a.snabf_opt.absolute(),
                '--input-spec',spec]
        if a.mode == 'inspect':
            argv.append('--native-import-only')
        else:
            argv += ['--emit-go','--mode','keep-source']
            if a.security_cache is not None:
                argv += ['--security-cache',a.security_cache.absolute()]
        p, trace = audit_exec(argv, case)
        (case/'stdout.txt').write_bytes(p.stdout.encode())
        (case/'stderr.txt').write_bytes(p.stderr.encode())
        store(case/'exec.json', trace)
        result = dict(source_sha256=pins[g], exit_code=p.returncode, process_tree=trace,
                      compiler='refused' if p.returncode else 'returned', ciphertext_slots='not_run',
                      scale_replay='not_run', dual_entry_mutations='not_run')
        if a.mode == 'inspect' and not p.returncode:
            import hashlib
            receipt = json.loads((case/'out/import.json').read_bytes())
            if (receipt['stage'] != 'structural-inspection' or receipt['source_sha256'] != pins[g] or
                    receipt['input_spec_sha256'] != hashlib.sha256(spec.read_bytes()).hexdigest() or
                    receipt['func_symbol'] != symbol or list((case/'out').iterdir()) != [case/'out/import.json']):
                raise ValueError(g+': invalid inspection receipt')
            result['source_inspection'] = 'verified'
        elif p.returncode and (case/'out').exists():
            raise ValueError(g+': failed compilation published a directory')
        cases[g] = result
    # This command does not run the two offline OS verifiers or the four dual-entry
    # mutations. Even a compiler returning zero on four inputs is not N1 acceptance.
    store(out/'probe.json', dict(schema='snabf.n1-four-source-probe.v1', mode=a.mode, cases=cases,
                                 n1_release_verified=False))
    return 0 if all(c['exit_code'] == 0 for c in cases.values()) else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    s = p.add_subparsers(dest='mode', required=True)
    for op in ('inspect', 'compile'):
        g = s.add_parser(op)
        g.add_argument('--r3-frozen', type=Path, required=True)
        g.add_argument('--single-gelu', type=Path, required=True)
        g.add_argument('--snabf-opt', type=Path, required=True)
        g.add_argument('--out', type=Path, required=True)
        if op == 'compile': g.add_argument('--security-cache', type=Path)
    for op in ('slots','observe'):
        g = s.add_parser(op)
        g.add_argument('graph', choices=SOURCE)
        g.add_argument('source', type=Path)
        if op == 'slots':
            g.add_argument('jsonl', type=Path)
            g.add_argument('--scale', help='checked exact sink scale, required for complex product observations')
        else:
            g.add_argument('module', type=Path)
            g.add_argument('--out', required=True, type=Path)
            g.add_argument('--scale', required=True, help='exact sink metadata from independent replay')
        g.add_argument('--seal', required=True)
        for k in ('ring','level','slots'):
            g.add_argument('--'+k, required=True, type=int)
    a = p.parse_args()
    if a.mode == 'observe':
        from native_observations import observe
        print(json.dumps(observe(a.graph,a.source,a.module,a.out,seal=a.seal,ring=a.ring,
                                 level=a.level,slots=a.slots,scale=a.scale),allow_nan=False))
        return 0
    if a.mode == 'slots':
        print(json.dumps(replay_slots(a.graph, a.source, a.jsonl, seal=a.seal, ring=a.ring,
                                      level=a.level, slots=a.slots, scale=a.scale), allow_nan=False))
        return 0
    return probe(a)


if __name__ == '__main__':
    sys.exit(main())
