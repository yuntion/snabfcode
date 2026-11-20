"""Build *checked* source programs and the historical go-bind comparator for A/B."""

import argparse
import shutil
import subprocess
import tarfile
from pathlib import Path

from .pass2 import Pipeline
from .pass2_evidence import load, require, save
from .pass3 import emit, parameter, profile, selected


def frozen(archive, member, dest):
    # Fixed list of regular files; no archive paths are used as destinations.
    with tarfile.open(archive) as ta:
        f = ta.getmember(member)
        require(f.isfile(), f'not a regular file: {f.name}')
        with ta.extractfile(f) as src, dest.open('wb') as out:
            shutil.copyfileobj(src, out)


def prepare_one(args, work):
    d = args.out / work
    d.mkdir()
    legacy = d / 'legacy'
    legacy.mkdir()
    archive = args.root / 'doc/pass3-results/evidence.tar.gz'
    if work in ('reuse64', 'residual'):
        for name in ('checked.mlir', 'plan.json', 'security.json', 'profile.json'):
            frozen(archive, f'{work}/joint/{name}', d / name)
        frozen(archive, f'{work}/corpus.json', d / 'corpus.json')
        pp = Pipeline(args, d / 'profile.json')
        pp.call(*pp.opt, '--snabf-plan', '--snabf-check-plan', d / 'checked.mlir', '-o', legacy / 'planned.mlir')
    else:
        parameters = load(args.root / 'test/fixtures/reference_profile.json')
        cap = 350.0
        seen = set()
        for i in range(10):
            pp, w, _ = selected(args, work, d / f'specialize{i}', parameters)
            policy = f'max-k=1 security-cap={cap}'
            # This dense layer needs more precision than the sparse fixtures.
            # Widen the search for q0, not the QP cap or the accuracy gate.
            if work == 'control':
                policy += ' min-base=20'
            plan = parameter(pp, w, d / f'proposal{i}', policy)
            proposed = profile(plan)
            if proposed != parameters:
                sig = (cap, tuple(proposed['Q']), tuple(proposed['P']), proposed['logDefaultScale'])
                require(sig not in seen, f'{work}: repeated parameter proposal')
                seen.add(sig)
                parameters = proposed
                continue
            try:
                pp.call('sage', '-python', args.root / 'src/tools/pass3_security.py',
                        d / f'proposal{i}/plan.json', d / f'proposal{i}/security.json')
            except subprocess.CalledProcessError as exc:
                if exc.returncode != 3:
                    raise
                new_cap = load(d / f'proposal{i}/security.json')['new_cap']
                require(new_cap < cap, 'security feedback did not tighten the envelope')
                cap = new_cap
                continue
            shutil.copyfile(d / f'proposal{i}/security.json', d / 'security.json')
            parameter(pp, w, legacy, policy, d / 'security.json')
            shutil.copyfile(legacy / 'plan.json', d / 'plan.json')
            shutil.copyfile(w / 'corpus.json', d / 'corpus.json')
            shutil.copyfile(w / 'input.mlir', d / 'source.mlir')
            save(d / 'profile.json', parameters)
            break
        else:
            raise ValueError(f'{work}: Pass 3 did not converge')
    # This is intentionally the independent historical comparator. Direct
    # emission consumes checked.mlir, not the translated/rewritten Go.
    emit(pp, legacy)
    shutil.copyfile(legacy / 'checked.mlir', d / 'input.mlir')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ('root', 'out', 'heir-bin', 'snabf-opt', 'goemit'):
        p.add_argument('--' + opt, type=Path, required=True)
    p.add_argument('workloads', nargs='+', choices=('control', 'strided64', 'reuse64', 'residual', 'cross_residual'))
    a = p.parse_args()
    a.root, a.out = a.root.resolve(), a.out.resolve()
    a.extra_rotation_cap = None
    for work in a.workloads:
        prepare_one(a, work)
