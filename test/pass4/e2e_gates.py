"""Negative tests for the unified driver, using a successful source run as a fixture."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def main(src, out):
    root = Path(__file__).resolve().parents[2]
    verify = root/'src/tools/verify_e2e_pass1_4.py'
    check(src.is_dir() and not out.exists(), 'supply an accepted run and a fresh output path')
    out.mkdir(parents=True)
    # Mutate copies only. No certificate, trial or metric in the accepted run is edited.
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tampered = Path(tmp)
        for name in ('inputs','env.snapshot','inputs.sha256','toolchain.sha256','work'):
            (tampered/name).symlink_to(src/name)
        (tampered/'out').mkdir()
        for name in ('model.go','program.json','search.json','metrics.json'):
            (tampered/'out'/name).symlink_to(src/'out'/name)
        with (src/'out/trials.jsonl').open() as before, (tampered/'out/trials.jsonl').open('w') as after:
            row = json.loads(next(before))
            replica = len(row['truth_one_period'])
            row['decoded64'][replica] += 1.0
            after.write(json.dumps(row)+'\n')
            shutil.copyfileobj(before,after)
        result = subprocess.run([sys.executable,verify,tampered],capture_output=True,text=True)
        (out/'replica.log').write_text(result.stderr)
        check(result.returncode > 0 and 'runner totals do not replay' in result.stderr, 'corrupt replica not caught')
    # A mismatched evidence context is fatal, not a reason to issue a fallback.
    bad = json.loads((src/'inputs/estimates.json').read_text())
    bad['source_ring'] *= 2
    cache = out/'wrong-context.json'
    cache.write_text(json.dumps(bad))
    cmd = [root/'src/tools/run_e2e_pass1_4.sh', '--from','pass3','--mlir',src/'work/checkpoint.mlir',
           '--oracle-mlir',src/'inputs/source.mlir','--corpus',src/'inputs/corpus.json','--profile',src/'inputs/profile.json',
           '--security',src/'inputs/security.json','--estimates',cache,'--strategy','adaptive','--sw-key-bits','128','--out-root',out/'driver']
    res = subprocess.run(cmd,capture_output=True,text=True,env=dict(os.environ,SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS='512'))
    log = res.stdout+res.stderr
    (out/'context.log').write_text(log)
    run, = (out/'driver').glob('run.*')
    check(res.returncode > 0 and 'estimator_context_mismatch' in log and 'FAILED stage=pass4' in log, 'unexpected bad-cache outcome')
    check('SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS=512\n' in (run/'env.snapshot').read_text(), 'CLI weakened the process policy')
    check(not any((run/name).exists() for name in ('out/model.go','out/program.json','verified.json')) and 'DONE:' not in log,'failed run published output')
    result = dict(replica_index=replica, replica='runner totals do not replay', external_evidence='estimator_context_mismatch',
                  failed_stage='pass4', go_published=False, cli_floor=128, environment_floor=512, effective_process_floor=512)
    (out/'negative.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source_run',type=Path)
    p.add_argument('new_output',type=Path)
    a = p.parse_args()
    main(a.source_run.resolve(),a.new_output.resolve())
