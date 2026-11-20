"""Publish a new, replayable pair of matrices; verify it with Python stdlib alone."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
from e2e_matrix import CASES, case_id, inspect
from e2e_oracle_witness import freeze
from verify_e2e_pass1_4 import load, require, save


AUDIT = ('src/tools/archive_e2e_matrix.py','src/tools/e2e_matrix.py','src/tools/verify_e2e_pass1_4.py',
         'src/tools/e2e_matrix_evidence.py','src/tools/e2e_oracle_witness.py','src/frontend/snabf_frontend/pass2_evidence.py','test/pass4/matrix_sources.json')


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def files(folder):
    return sorted(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.relative_to(folder).parts)


def member_name(s):
    path = PurePosixPath(s)
    require(s and not path.is_absolute() and '\\' not in s and ':' not in s and path.as_posix() == s and '..' not in path.parts, 'invalid archive path')
    return path


def verify_archive(folder):
    manifest = {}
    for line in (folder/'SHA256SUMS').read_text().splitlines():
        value,name = line.split('  ',1)
        require(name not in manifest, 'duplicate checksum: '+name)
        manifest[str(member_name(name))] = value
    observed = {p.relative_to(folder).as_posix():digest(p) for p in files(folder) if p != folder/'SHA256SUMS'}
    require(manifest == observed, 'outer archive checksums or member set differ')
    summary = load(folder/'acceptance.json')
    require(set(summary) == {'source','pass3'}, 'missing matrix')
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        with tarfile.open(folder/'evidence.tar.gz','r|gz') as tar:
            for item in tar:
                rel = member_name(item.name)
                require(item.isfile() and rel.parts[0] in summary, 'unexpected evidence member')
                target = root/rel; target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(item) as f, target.open('xb') as out:
                    shutil.copyfileobj(f,out)
        for mode in summary:
            replay = inspect(root/mode,witnesses=folder/'oracle-witness')
            require(replay == load(root/mode/'acceptance.json') == summary[mode], 'archived matrix differs: '+mode)
            for r in [*replay['planning'].values(),*replay['single_ring'].values()]:
                require(all(digest(folder/'frozen'/r['workload']/name) == h for name,h in r['inputs'].items()), 'reproduction inputs differ from executed inputs')
    return {k:v['equivalence'] for k,v in summary.items()}


def report(summary):
    lines = ['# Pass 1-4: planning and same-ciphertext emitter equivalence', '',
             'Fresh encrypted evaluations; threshold 2^-12 at every encoded slot, including periodic replicas. Times are online Eval, not key setup, encryption, copy hashing or decoding.', '']
    for mode,group in summary.items():
        lines += ['## '+mode, '', '| Planning scenario | Live ciphertexts | Decision or refusal | N | L / log2(scale) | Conversion keys / bytes | Truth Linf / MSE | Misses / slots | Mean Eval ms |',
                  '|---|---:|---|---|---|---|---|---:|---:|']
        for name,r in group['planning'].items():
            m = r['measurements']; reason = ('ring_down' if r['outcome']=='ring_down' else next(x for x in r['reasons'] if x in ('switching_key_ext_modulus_security_rejected','residual_bypass_crosses_cut_unconverted')))
            lines.append(f"| {name} | {r['live_ciphertexts']} | `{reason}` | {r['source_ring']} -> {r['target_ring']} | {m['actual_level']} / {m['actual_log_scale_range'][0]:.6f} | {r['conversion_keys']} / {r['observed_key_bytes']['down']:,} | {m['linf']:.8g} / {m['mse']:.8g} | {m['misses']} / {m['physical_slots']:,} | {m['online_mean_ms']:.2f} |")
        lines += ['', '| Fixed source ring | Inputs / timed Eval | N / L | Truth Linf direct / legacy | Paired Linf | Paired MSE | Truth misses direct / legacy | Paired misses |',
                  '|---|---|---|---|---:|---:|---|---:|']
        for name,r in group['single_ring'].items():
            m = r['measurements']; x,y = m['truth_linf_direct_legacy']; a,b = m['truth_misses_direct_legacy']
            lines.append(f"| {name} | {m['trials']} / {m['measured_evaluations']} | {m['actual_ring']} / {m['actual_level']} | {x:.8g} / {y:.8g} | {m['paired_linf']:.8g} | {m['paired_mse']:.8g} | {a} / {b} | {m['paired_misses']} |")
        s = group['equivalence']
        lines += ['', f"{s['inputs']} encrypted inputs; {s['timed_evaluations']} timed evaluations; {s['untimed_probe_evaluations']} additional, untimed gadget probes. {s['physical_slot_pairs']:,} slot pairs, maximum paired difference {s['paired_linf']:.9g}, slot-weighted paired MSE {s['paired_mse']:.9g}.", '']
    lines += ['## Interpretation and replay', '',
              '`source` recompiles the supplied matrices against locked Pass 3 search bounds and concrete parameters. `pass3` is checkpoint replay, not another independent set of source inputs. The same corpus is used across modes.',
              'Fixed-source-ring tests compare the native straight-line Go emitter with a freshly emitted historical HEIR/go-bind arm at the same upstream checkpoint. A single encryption is copied; both arms use the actual source evaluator and same-domain Decode. Mixed outputs are decrypted only in their own target domain.',
              'Periodic replicas are not independent samples. A per-slot pass is neither a failure-probability proof nor a speedup claim. Lattice-estimator screens are conditional classical estimates; the 512-bit floor applies only to the conversion key.', '',
              'Archive portability: Windows and Linux libm can differ in the last bit of exp. Each public exp(-h) binary64 witness was taken on the producer after an exact source/corpus replay. The offline checker independently bounds each witness against 90-digit Decimal.exp to one binary64 ulp, then reconstructs the entire frozen truth bit-for-bit. No tolerance is added to truth comparisons, decrypted slot errors, paired differences or the 2^-12 decision. Live source runs still use the original strict oracle. The four-gate negative record is in negative.json; raw Go, SSA, logs and slot pairs are in evidence.tar.gz.', '',
              'Archive-only verification (Python 3.10+, standard library; independent of HEIR, Go, the estimator and the build tree):', '',
              '```bash', 'python3 audit/src/tools/archive_e2e_matrix.py check .', '```', '',
              'On the compiler server, from this archive directory, rerun in fresh directories:', '',
              '```bash', 'python3 /home/fhecompiler/src/tools/e2e_matrix.py run --from source --inputs "$PWD/frozen" --out-root /home/fhecompiler/build/matrix-reproduce',
              'python3 /home/fhecompiler/src/tools/e2e_matrix.py run --from pass3 --inputs "$PWD/frozen" --out-root /home/fhecompiler/build/matrix-reproduce', '```', '']
    return '\n'.join(lines)


def publish(paths, output, negatives=None):
    root = Path(__file__).resolve().parents[2]
    require(not output.exists(), 'refusing to overwrite an archive')
    summary = {mode:inspect(path) for mode,path in paths.items()}
    for mode,s in summary.items():
        require(s['mode'] == mode and s == load(paths[mode]/'acceptance.json'), 'unaccepted source matrix')
    # Stage alongside destination, publish only after a complete extraction and replay.
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp)/'release'; stage.mkdir()
        for path in AUDIT:
            p = stage/'audit'/path; p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(root/path,p)
        if negatives:
            shutil.copyfile(negatives,stage/'negative.json')
        save(stage/'acceptance.json',summary)
        (stage/'README.md').write_text(report(summary),encoding='utf-8')
        (stage/'oracle-witness').mkdir()
        for scenario,net in CASES:
            d = stage/'frozen'/net
            if not d.exists():
                run, = (paths['pass3']/case_id(scenario,net)).glob('run.*'); shutil.copytree(run/'inputs',d)
                save(stage/'oracle-witness'/(net+'.json'),freeze(d))
        with tarfile.open(stage/'evidence.tar.gz','w:gz',compresslevel=4) as tar:
            for mode,path in paths.items():
                for p in files(path):
                    require(not p.is_symlink(),'unexpected symlink in evidence')
                    if p.name not in ('key-meter','model.a'):
                        tar.add(p,arcname=mode+'/'+p.relative_to(path).as_posix(),recursive=False)
        (stage/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(stage).as_posix()}\n' for p in files(stage)),encoding='ascii')
        verify_archive(stage)
        require(not output.exists(), 'archive destination appeared during verification')
        stage.rename(output)
    print('ARCHIVE VERIFIED:',output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); cmd = p.add_subparsers(dest='action',required=True)
    q = cmd.add_parser('publish')
    for name in ('source','pass3','out'): q.add_argument('--'+name,required=True,type=Path)
    q.add_argument('--negatives',type=Path)
    c = cmd.add_parser('check'); c.add_argument('folder',type=Path)
    a = p.parse_args()
    if a.action == 'publish': publish({'source':a.source,'pass3':a.pass3},a.out.resolve(),a.negatives)
    else: print('ARCHIVE VERIFIED:',verify_archive(a.folder.resolve()))
