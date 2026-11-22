"""Run the actual original-source CLI and pinned importer; never emit an execution receipt here."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
F = ROOT/'test/compiler/native_import'
source = F/'c1-single.mlir'
spec = F/'c1.inputs.json'
with tempfile.TemporaryDirectory() as d:
    out = Path(d)/'model'
    cmd = [sys.executable,'-B',ROOT/'src/tools/snabf_compile.py','--frontend','native-linalg',
           '--snabf-opt',sys.argv[1],'--mlir',source,'--out-dir',out,'--input-bound','1','--input-spec',spec]
    for flags, reason in ((['--native-import-only','--emit-go'],'SNABF_NATIVE_INSPECTION'),
                          (['--native-import-only','--mode','force-down'],'SNABF_NATIVE_INSPECTION'),
                          (['--native-import-only','--certify-failure-bits','40'],'SNABF_NATIVE_INSPECTION'),
                          (['--emit-go','--mode','force-down'],'SNABF_NATIVE_PROFILE')):
        p = subprocess.run(cmd+flags,capture_output=True)
        if p.returncode != 1 or reason.encode() not in p.stderr or out.exists():
            raise AssertionError((flags,p.returncode,p.stderr))
    subprocess.run(cmd+['--native-import-only'],check=True)
    r = json.loads((out/'import.json').read_bytes())
    if (r['stage'] != 'structural-inspection' or r['frontend'] != 'native-linalg' or
            r['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest() or
            r['input_spec_sha256'] != hashlib.sha256(spec.read_bytes()).hexdigest() or
            {p.name for p in out.iterdir()} != {'import.json'}):
        raise AssertionError('unbound positive inspection')
