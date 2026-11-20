"""Run or independently replay the seven named Pass 1-4 regression cases."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile

from verify_e2e_pass1_4 import load, require, reuse64_pair, save, verify


NETWORKS = ('control','strided64','reuse64','residual')
CASES = [('reuse64-128','reuse64'),('reuse64-512','reuse64'),('cross-residual','cross_residual')]
CASES += [('equiv-single-ring',n) for n in NETWORKS]


def case_id(s,n):
    return 'equiv-'+n if s == 'equiv-single-ring' else s


def inspect(folder):
    plans, comparisons, paths, seen, modes = {}, {}, {}, set(), set()
    expected = {case_id(s,n) for s,n in CASES}
    require({p.name for p in folder.iterdir() if p.is_dir()} == expected, 'matrix case set differs from protocol')
    for scenario,net in CASES:
        label = case_id(scenario,net)
        children = list((folder/label).glob('run.*'))
        require(len(children) == 1, f'{label}: missing or duplicate run')
        run, = children
        require(run.resolve() not in seen and run.resolve().is_relative_to(folder.resolve()), 'external or duplicate run')
        seen.add(run.resolve()); paths[label] = run
        result = verify(run)
        require(result == load(run/'acceptance.json') == load(run/'verified.json') and
                (result['scenario'],result['workload']) == (scenario,net), 'archived result/scenario does not replay: '+label)
        modes.add(result['mode'])
        result['run'] = run.relative_to(folder).as_posix()
        if scenario == 'equiv-single-ring':
            comparisons[net] = result
        else:
            plans[label] = result
    require(len(modes) == 1, 'do not combine upstream compilation modes in one matrix')
    lock = reuse64_pair(paths['reuse64-128'],paths['reuse64-512'])['upstream']
    same = plans['reuse64-128']['inputs']
    require(comparisons['reuse64']['inputs'] == same, 'reuse64 planning and emitter arms used different frozen inputs')
    m = [r['measurements'] for r in comparisons.values()]
    summary = dict(inputs=sum(x['encryption_count'] for x in m), timed_evaluations=sum(x['measured_evaluations'] for x in m),
                   untimed_probe_evaluations=sum(x['untimed_probe_evaluations'] for x in m), physical_slot_pairs=sum(x['physical_slots_per_arm'] for x in m),
                   truth_slot_checks=sum(2*x['physical_slots_per_arm'] for x in m), paired_linf=max(x['paired_linf'] for x in m),
                   paired_mse=sum(x['paired_mse']*x['physical_slots_per_arm'] for x in m)/sum(x['physical_slots_per_arm'] for x in m),
                   truth_misses=sum(sum(x['truth_misses_direct_legacy']) for x in m), paired_misses=sum(x['paired_misses'] for x in m))
    return dict(schema='snabf.e2e.matrix.v1',mode=modes.pop(),reuse64_upstream=lock,planning=plans,single_ring=comparisons,equivalence=summary)


def execute(inputs, base, mode):
    base.mkdir(parents=True,exist_ok=True)
    folder = Path(tempfile.mkdtemp(dir=base,prefix='matrix.'))
    print('MATRIX_DIR='+str(folder),flush=True)
    driver = Path(__file__).with_name('run_e2e_pass1_4.sh').resolve()
    for scenario,net in CASES:
        d = inputs/net
        cmd = [driver,'--scenario',scenario,'--workload',net,'--from',mode,'--estimates',d/'estimates.json',
               '--pass3-options',d/'pass3-options.json','--out-root',folder/case_id(scenario,net)]
        if mode == 'source': cmd += ['--mlir',d/'source.mlir']
        else: cmd += ['--mlir',d/'checkpoint.mlir','--oracle-mlir',d/'source.mlir']
        subprocess.run(cmd,check=True)
    result = inspect(folder)
    save(folder/'acceptance.json',result)
    print(f'MATRIX VERIFIED: {folder}/acceptance.json',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action',required=True)
    run = sub.add_parser('run')
    run.add_argument('--from',dest='mode',choices=['source','pass3'],default='source')
    run.add_argument('--inputs',type=Path,required=True)
    run.add_argument('--out-root',type=Path,required=True)
    audit = sub.add_parser('verify')
    audit.add_argument('matrix',type=Path)
    a = p.parse_args()
    if a.action == 'run':
        execute(a.inputs.resolve(strict=True),a.out_root.resolve(),a.mode)
    else:
        result = inspect(a.matrix.resolve())
        require(result == load(a.matrix/'acceptance.json'), 'summary differs from independently replayed matrix')
        print('MATRIX VERIFIED:', a.matrix, result['equivalence'])
