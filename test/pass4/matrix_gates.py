"""Check matrix-only failure modes without modifying accepted artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/tools'))
from verify_e2e_pass1_4 import load, require, save, verify


def negative(run, name, edit, expected):
    with tempfile.TemporaryDirectory() as tmp:
        case = Path(tmp)
        for fn in ('inputs','inputs.sha256','env.snapshot','toolchain.sha256','work'):
            (case/fn).symlink_to(run/fn)
        (case/'out').mkdir()
        for fn in ('search.json','program.json','metrics.json','model.go'):
            (case/'out'/fn).symlink_to(run/'out'/fn)
        with (run/'out/trials.jsonl').open() as src, (case/'out/trials.jsonl').open('w') as dst:
            row = json.loads(next(src)); edit(row); dst.write(json.dumps(row)+'\n'); shutil.copyfileobj(src,dst)
        try:
            verify(case)
        except ValueError as err:
            require(expected in str(err), f'{name}: unexpected rejection: {err}')
            return {'case':name,'error':str(err)}
        raise AssertionError(name+' was accepted')


def main(run):
    require(verify(run)['scenario'] == 'equiv-single-ring', 'expected a verified paired run')
    truth = load(run/'inputs/corpus.json')[0]['truth']
    def only_pair(row):
        # First periodic replica, not in the logical-width prefix. Each output
        # is still within tolerance of truth, but the arms disagree by 1.5 eps.
        k = len(truth); off = .75*2**-12
        row['decoded_direct'][k] = truth[0]+off
        row['decoded_legacy'][k] = truth[0]-off
    return [negative(run,'paired_only_periodic_slot',only_pair,'(direct, legacy, paired): [0, 0, 1]'),
            negative(run,'unshared_ciphertext',lambda r:r['copy_sha256_direct_legacy'].__setitem__(1,'0'*64),'sharing or crossover'),
            negative(run,'wrong_order',lambda r:r.update(eval_order=[1,0]),'sharing or crossover'),
            negative(run,'legacy_wrong_ring',lambda r:r['output_direct_legacy'][1].update(N=8192),'output ABI mismatch')]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path); p.add_argument('--output',type=Path,required=True)
    args = p.parse_args(); results = main(args.run.resolve()); save(args.output,results)
    print(json.dumps(results))
