"""Drive the registered C++ MLIR passes. JSON is a backend/audit export only."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .strided import strided_model


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def store(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('root', 'out', 'heir-bin', 'snabf-opt', 'goemit'):
        ap.add_argument('--' + arg, required=True, type=Path)
    args = ap.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    fixture = root / 'test/fixtures'
    profile = fixture / 'reference_profile.json'
    heir, translate = args.heir_bin / 'heir-opt', args.heir_bin / 'heir-translate'
    costs = json.loads((fixture / 'pass1_costs.json').read_text())
    size = int(subprocess.check_output([args.goemit, 'key-size', profile]))
    imp = (f'key-bytes={size} hoist={costs["hoisted_per_active_limb"]} '
           f'giant={costs["giant_per_active_limb"]} key-units-per-mib={costs["key_units_per_mib"]}')
    generic = ['--mlir-print-op-generic', '--mlir-print-local-scope']
    opt = [args.snabf_opt, '--allow-unregistered-dialect', '--verify-each']
    for work in ('control', 'strided64', 'reuse64'):
        d = out / work
        ref = d / 'heir'
        ref.mkdir(parents=True, exist_ok=True)
        source = (fixture / 'linear_silu_linear.mlir').read_text()
        if work != 'control':
            source = strided_model(source, json.loads((fixture / f'pass1_{work}.json').read_text()))
        model = ref / 'input.mlir'
        model.write_text(source)
        run(heir, '--annotate-module=backend=lattigo scheme=ckks', '--linalg-canonicalizations',
            f'--torch-linalg-to-ckks=min-slot-count=4096 parameter-file={profile}', *generic, model, '-o', d/'ckks.mlir')
        run(heir, '--scheme-to-lattigo', d/'ckks.mlir', '-o', ref/'lowered.mlir')
        run(translate, '--emit-lattigo', '--package-name=baseline', ref/'lowered.mlir', '-o', ref/'model.go')
        run(*opt, '--snabf-import-heir='+imp, d/'ckks.mlir', '-o', d/'imported.mlir')
        run(*opt, '--snabf-pass1-candidates', d/'imported.mlir', '-o', d/'candidates.mlir')
        report = {'bootstrap': {'status': 'not_applicable'}, 'input_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
        for arm in (('independent', 'joint', 'cap2') if work == 'reuse64' else ('independent', 'joint')):
            sub = d / arm
            sub.mkdir(exist_ok=True)
            imported = d / 'candidates.mlir'
            if arm == 'cap2':
                imported = sub / 'imported.mlir'
                run(*opt, '--snabf-import-heir='+imp+' rotation-path-limit=2', d/'ckks.mlir', '-o', imported)
            mode = '{independent=true}' if arm == 'independent' else ''
            run(*opt, f'--pass-pipeline=builtin.module(snabf-plan{mode},snabf-check-plan)', imported, '-o', sub/'selected.mlir')
            run(*opt, '--snabf-lower-ckks', *generic, sub/'selected.mlir', '-o', sub/'ckks.mlir')
            run(heir, '--scheme-to-lattigo', *generic, sub/'ckks.mlir', '-o', sub/'lattigo.mlir')
            run(*opt, '--snabf-bind-lattigo=audit='+str(sub/'audit.json'), *generic, sub/'lattigo.mlir', '-o', sub/'bound.mlir')
            run(heir, sub/'bound.mlir', '-o', sub/'verified.mlir')
            run(translate, '--emit-lattigo', f'--package-name={arm}', sub/'verified.mlir', '-o', sub/'translated.go.txt')
            audit = json.loads((sub/'audit.json').read_text())
            backend = audit.pop('backend')
            store(sub/'routes.json', backend)
            run(args.goemit, 'emit', sub/'routes.json', sub/'translated.go.txt', sub/'model.go')
            by_site = sorted(backend, key=lambda x: x['site'])
            audit['routes'] = [{'s': r['S'], 'terms': r['Terms']} for r in by_site]
            audit['selected_sha256'] = hashlib.sha256((sub/'selected.mlir').read_bytes()).hexdigest()
            report[arm] = audit
            report['regions'] = [{'level': r['Level'], 'diagonals': [t[0] for t in r['Terms']]} for r in by_site]
            print(f'{work}/{arm}: rounds={audit["rounds"]} keys={len(audit["keys"])} bytes={audit["key_bytes"]}', flush=True)
        store(d/'plan.json', report)


if __name__ == '__main__':
    main()
