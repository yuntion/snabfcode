import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

root, run, d, frozen = map(Path, sys.argv[1:])
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(p):
    lines = [line.split(None, 1) for line in p.read_text().splitlines() if line.strip()]
    result = {name: h for h, name in lines}
    assert len(result) == len(lines) and all(re.fullmatch('[0-9a-f]{64}', h) for h in result.values()), p
    return result

def check_files(files, parent):
    for name, expected in files.items():
        assert digest(parent/name) == expected, (parent, name)

doc = root/'doc/pass4-ab-results-2026-11-19'
archive = json.loads((doc/'sha256.json').read_text())['members']
delivery = manifest(doc/'snabf-p4-ab-delivered.sha256')
current = manifest(run/'source.sha256')
assert digest(run/'source.sha256') == archive['source.sha256']
check_files(delivery, root)
check_files(current, root)
names = [f'{case}/{name}' for case in ('reuse64','cross_residual','residual') for name in ('input.mlir','cache.json')]
fixtures = {k: archive[k] for k in names}
check_files(fixtures, run)
baseline = manifest(frozen)
assert len(baseline) == 7
final = {k: digest(root/k) for k in baseline}
assert all(final[k] == delivery[k] == current[k] for k in final)
changed = sorted(k for k in baseline if final[k] != baseline[k])
assert set(changed) == {'src/compiler/ir/Pass4IR.cpp','src/compiler/pass4/tests/replay_gates.py'}
scratch = d/'reverse-proof'; scratch.mkdir(mode=0o700)
for k in baseline:
    (scratch/k).parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/k,scratch/k)
patch = root/'build/pass4-delivery.lcFrMpgs/snabf-p4-ab-preload.patch'
for flags in (['--check','--reverse'],['--reverse']):
    subprocess.run(['git','apply',*flags,str(patch)], cwd=scratch, check=True)
check_files(baseline, scratch)
pin = subprocess.check_output(['git','-C',os.environ['LATTICE_ESTIMATOR'],'rev-parse','HEAD'],text=True).strip()
assert pin == 'ae45d740ceb138af87a4f92edc592a0ab0d4f43f'
assert all(json.loads((run/n).read_text())['estimator'] == pin for n in names if n.endswith('.json'))
opt = d/'native/ir/snabf-opt'
translator = Path(os.environ['HEIR_TRANSLATE'])
helper = root/'src/backend/ringswitch/epochs/heirckks/standard.go'
assert digest(helper) == '3ba17babf49e6153724b374ebc22232f047121a9094c841fa8512c8c8d3f3102'
print(json.dumps(dict(project=str(root),fresh_build=str(d),fixtures_dir=str(run),fixtures=fixtures,
    frozen_sha256=digest(frozen),frozen=baseline,delivered=final,changed_since_freeze=changed,
    source_files_checked=len(current),delivery_files_checked=len(delivery),reverse_patch=str(patch),
    reverse_patch_sha256=digest(patch),reverse_proof_matched=len(baseline),estimator=pin,
    estimator_mode='replay of pinned cache, no fresh estimator run',snabf_opt=str(opt),snabf_opt_sha256=digest(opt),
    heir_translate=str(translator),heir_translate_sha256=digest(translator),helper_sha256=digest(helper),
    go=subprocess.check_output(['go','version'], text=True).strip()), indent=2))
