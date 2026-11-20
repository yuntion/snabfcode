"""Regenerate the historical Go arm from the same verified single-ring checkpoint."""
import argparse
import os
from pathlib import Path
import shutil

from .pass2 import Pipeline
from .pass2_evidence import load, require
from .pass3 import emit


def prepare(a):
    direct = load(Path('out/program.json'))
    require(direct['source_ring'] == direct['target_ring'] == direct['ring_search']['policy']['min_ring']
            and not any(i['op'] == 'ring_down' for i in direct['instructions']), 'paired comparison requires fixed source ring')
    legacy = Path('work/legacy')
    legacy.mkdir()
    shutil.copyfile('work/checkpoint.mlir', legacy/'planned.mlir')
    pp = Pipeline(a, Path('inputs/profile.json'))
    pp.translate = Path(os.environ['HEIR_TRANSLATE'])
    emit(pp, legacy)  # native check -> CKKS -> HEIR/Lattigo -> historical go-bind


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','heir-bin','snabf-opt','goemit'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.set_defaults(out=Path.cwd(), extra_rotation_cap=None)
    prepare(parser.parse_args())
