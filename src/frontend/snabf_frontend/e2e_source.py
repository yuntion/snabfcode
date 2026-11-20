"""Compile the caller's dense Linear -> SiLU -> Linear MLIR against a frozen profile.

Run inside a fresh evidence directory. No fixture generation, historical Go
emission, or reading old plans. A changed MILP solution needs a new profile
and estimator evidence; this locked comparison must not silently switch them.
"""
import argparse
import math
from pathlib import Path

from .pass2 import DEPTH, EPSILON, Pipeline
from .pass2_evidence import Oracle, load, require
from .pass3 import profile


def compile_source(a):
    source, profile_file = Path('inputs/source.mlir'), Path('inputs/profile.json')
    oracle = Oracle(source.read_text())
    require(oracle.corpus() == load(Path('inputs/corpus.json')), 'frozen corpus does not replay from source')
    pp = Pipeline(a, profile_file)  # go-bind key-size only; the emitter is the native Target/Lattigo.
    work = Path('work')
    pp.lower_source(source, work, oracle.residual or oracle.cross_residual)
    imported = pp.import_ckks('residual' if oracle.residual else 'source', work)
    pp.call(*pp.opt, '--snabf-pass1-candidates', imported, '-o', work/'pass1.mlir')
    pp.call(*pp.opt, f'--snabf-pass2-candidates=depth={DEPTH} epsilon={EPSILON} max-output-error={EPSILON}',
            work/'pass1.mlir', '-o', work/'pass2.mlir')
    pp.call(*pp.opt, '--snabf-plan', '--snabf-check-plan', work/'pass2.mlir', '-o', work/'selected.mlir')
    security = Path('inputs/security.json')
    # The estimator screened the actual Q, QP and the power-of-two envelope.
    envelope = int(load(security)['instances']['envelope']['modulus'])
    require(envelope > 1 and envelope & (envelope-1) == 0, 'invalid estimator envelope')
    cap = envelope.bit_length()-1
    options = {'max_k': 1, 'security_cap': cap}
    frozen_policy = Path('inputs/pass3-options.json')
    if frozen_policy.exists():
        options = load(frozen_policy)
        require(set(options) == {'min_scale','max_scale','min_base','max_k','tolerance','guard','security_cap'} and
                all(type(v) in (int,float) and math.isfinite(v) for v in options.values()) and
                math.ceil(options['security_cap']) == cap, 'invalid locked Pass 3 search bounds/envelope')
    opts = ' '.join(f'{k.replace("_","-") if k != "tolerance" else "epsilon"}={v}' for k,v in sorted(options.items()))
    opts += f' security={security} audit=work/pass3.json'
    pp.call(*pp.opt, '--snabf-pass3='+opts, work/'selected.mlir', '-o', work/'pass3.mlir')
    require(profile(load(work/'pass3.json')) == load(profile_file), 'Pass 3 differs from locked HEIR profile')
    pp.call(*pp.opt, '--snabf-check-plan', work/'pass3.mlir', '-o', work/'checkpoint.mlir')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'heir-bin', 'snabf-opt', 'goemit'):
        p.add_argument('--'+name, type=Path, required=True)
    p.set_defaults(out=Path.cwd(), extra_rotation_cap=None)
    compile_source(p.parse_args())
