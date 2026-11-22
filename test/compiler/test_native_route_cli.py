"""Exercise the native route-only public CLI, not an encrypted execution gate."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from native_import import oracle as src
from native_linear_oracle import structure


TOOLS = Path(__file__).resolve().parents[2]/'src/tools'
sys.path.insert(0, str(TOOLS))
from native_inspection import route_inspection


def main(opt):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        source, ports = root/'model.mlir', root/'ports.json'
        cli = [sys.executable, '-B', TOOLS/'snabf_compile.py', '--snabf-opt', opt, '--mlir', source,
               '--input-bound', '1', '--input-spec', ports]
        native = ['--frontend', 'native-linalg']
        for case in src.HASHES:
            raw, spec = src.original(case), src.spec_for(case)
            source.write_bytes(raw)
            ports.write_bytes(spec)
            out = root/case
            subprocess.run(cli+native+['--native-routes-only','4096','--out-dir',out], check=True)
            src.check([p.name for p in out.iterdir()] == ['routes.json'], 'inspector published a model or extra files')
            catalog = src.read_receipt((out/'routes.json').read_bytes())
            structure(raw, spec, catalog, 14, 4096)
        # Same encoded slot count but a different physical rotation group.
        out = root/'small-domain'
        subprocess.run(cli+native+['--native-routes-only','4096','--log-n','13','--out-dir',out], check=True)
        rec = src.read_receipt((out/'routes.json').read_bytes())
        structure(raw, spec, rec, 13, 4096)
        for updates in ({'rotation_modulus':8192}, {'encoded_slots':2048}, {'logical_width':True},
                        {'replica_count':1}, {'first_slot_for_row':[False,1,2,3]}):
            wrong = copy.deepcopy(rec)
            wrong['layout'].update(updates)
            try:
                route_inspection(wrong,13,4096)
            except ValueError as exc:
                src.check('SNABF_NATIVE_INSPECTION' in str(exc), 'wrong tamper diagnostic')
            else:
                raise AssertionError(f'accepted wrong domain {updates}')
        bad = root/'never-publish'
        opts = ['--native-routes-only','4096']
        for flags, msg in ((opts, 'SNABF_INVALID_OPTIONS'),
                           (native+['--native-routes-only','0'], 'SNABF_NATIVE_LINEAR_LAYOUT'),
                           (native+['--native-routes-only','3'], 'SNABF_NATIVE_LINEAR_LAYOUT'),
                           (native+opts+['--log-n','12'], 'SNABF_NATIVE_LINEAR_LAYOUT'),
                           (native+opts+['--min-slots','8192'], 'SNABF_NATIVE_INSPECTION'),
                           (native+opts+['--emit-go'], 'SNABF_NATIVE_INSPECTION'),
                           (native+opts+['--certify-failure-bits','40'], 'SNABF_NATIVE_INSPECTION'),
                           (native+opts+['--native-import-only'], 'not allowed with argument')):
            p = subprocess.run(cli+flags+['--out-dir',bad],capture_output=True)
            src.check(p.returncode != 0 and msg.encode() in p.stderr and not bad.exists(), flags)
        original = (out/'routes.json').read_bytes()
        p = subprocess.run(cli+native+opts+['--out-dir',out], capture_output=True)
        src.check(p.returncode and b'SNABF_OUTPUT_EXISTS' in p.stderr and
                  (out/'routes.json').read_bytes() == original, 'pre-existing receipt overwritten')
        ports.write_bytes(b'{"inputs": [], "inputs": '+json.dumps(json.loads(spec)['inputs']).encode()+b'}')
        p = subprocess.run(cli+native+opts+['--out-dir',bad],capture_output=True)
        src.check(p.returncode and b'duplicate JSON field' in p.stderr and not bad.exists(), 'ambiguous input spec')
        print('four byte-locked sources, two physical domains; 5 layout and 10 CLI refusals')


if __name__ == '__main__':
    main(sys.argv[1])
