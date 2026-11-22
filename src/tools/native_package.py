"""Build the N1 emitter's package in isolation. This is not a model or security verifier."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]
LATTIGO = 'github.com/tuneinsight/lattigo/v6'
# Runtime sources, not tests, archives, frozen models or HEIR-generated helpers.
BACKEND = ('src/backend', 'src/backend/gadget', 'src/backend/ringswitch',
           'src/backend/ringswitch/epochs')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def unique_fields(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError('[SNABF_NATIVE_PACKAGE] duplicate JSON field')
    return result


def pinned_go():
    ext = Path(os.environ.get('SNABF_EXTERN', ROOT/'extern')).resolve()
    go = (ext/'go'/'bin'/('go.exe' if os.name == 'nt' else 'go')).resolve(strict=True)
    # No ambient toolchain, workspace, compiler overlays, Go env file, or HEIR overrides.
    keep = ('PATH', 'HOME', 'USERPROFILE', 'SystemRoot', 'LOCALAPPDATA', 'TMP', 'TEMP',
            'TMPDIR', 'GOTMPDIR', 'GOCACHE', 'LD_LIBRARY_PATH')
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(GOPATH=str(ext/'gopath'), GOWORK='off', GOPROXY='off', GOTOOLCHAIN='local',
               GOENV='off', GOFLAGS='', GOSUMDB='off', CGO_ENABLED='0')
    return go, env


def snapshot(root):
    result = {}
    for p in root.rglob('*'):
        if p.is_symlink() or not (p.is_dir() or p.is_file()):
            raise ValueError(f'[SNABF_NATIVE_PACKAGE] non-regular entry: {p.name}')
        if p.is_file():
            result[p.relative_to(root).as_posix()] = digest(p.read_bytes())
    return result


def _go(go, env, cwd, *args):
    p = subprocess.run([go, *args], cwd=cwd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace')
    if p.returncode:
        raise ValueError(f'[SNABF_NATIVE_PACKAGE] go {args[0]}: {(p.stderr or p.stdout)[-4000:]}')
    return p.stdout


def _dependencies(go, env, root):
    keys = 'ImportPath,Dir,Name,Module,Standard,GoFiles,TestGoFiles,XTestGoFiles,EmbedFiles,TestEmbedFiles,XTestEmbedFiles'
    text = _go(go, env, root, 'list', '-mod=readonly', '-deps', '-test', '-json='+keys, './...')
    decoder = json.JSONDecoder()
    paths, lattigo = set(), False
    offset = 0
    while offset < len(text):
        if text[offset].isspace():
            offset += 1
            continue
        row, offset = decoder.raw_decode(text, offset)
        name = row['ImportPath']
        paths.add(name)
        files = [f for k in keys.split(',') if k.endswith('Files') for f in row.get(k, [])]
        if ('heirckks' in name.lower() or row.get('Name', '').lower() == 'heirckks' or
                any('standardckks' in f.lower() or 'heirckks' in f.lower() for f in files)):
            raise ValueError(f'[SNABF_NATIVE_PACKAGE] static HEIR helper: {name}')
        module = row.get('Module') or {}
        if module.get('Replace'):
            raise ValueError(f'[SNABF_NATIVE_PACKAGE] unexpected module replacement: {name}')
        if module.get('Path') == LATTIGO:
            if module.get('Version') != 'v6.2.0':
                raise ValueError('[SNABF_NATIVE_PACKAGE] wrong Lattigo version')
            # Do not let test-only or synthetic test-binary imports prove a runtime dependency.
            if ' [' not in name and not name.endswith('.test'):
                lattigo = True
    runtime = set(_go(go, env, root, 'list', '-mod=readonly', '-deps', './planned').splitlines())
    if (not lattigo or not any(p.startswith(LATTIGO+'/') for p in runtime) or
            'snabf.local/compiler/planned' not in runtime):
        raise ValueError('[SNABF_NATIVE_PACKAGE] expected planned package with pinned Lattigo dependency')
    return sorted(paths)


def publish_native_package(emitted, out, expected_seal):
    """Publish a checked emitter payload to a *new* module root; never modify the input.

    expected_seal comes from checked IR. It is not the SHA-256 of program.json.
    A successful return only attests package construction, not encrypted execution.
    """
    emitted, out = Path(emitted).resolve(strict=True), Path(out).absolute()
    if out.exists() or out.is_symlink():
        raise ValueError('[SNABF_OUTPUT_EXISTS] use a new package directory')
    if not isinstance(expected_seal, str) or not re.fullmatch('[0-9a-f]{64}', expected_seal):
        raise ValueError('[SNABF_NATIVE_PACKAGE] invalid checked IR seal')
    required = {'model.go', 'native_runtime.go', 'program.json'}
    children = {p.name for p in emitted.iterdir()}
    if not required <= children or any(n not in required and not re.fullmatch('[a-z][a-z0-9_]*_test[.]go', n) for n in children):
        raise ValueError('[SNABF_NATIVE_PACKAGE] incomplete or unexpected emitter payload')
    if any(p.is_symlink() or not p.is_file() for p in emitted.iterdir()):
        raise ValueError('[SNABF_NATIVE_PACKAGE] non-regular emitter payload')
    data = {n: (emitted/n).read_bytes() for n in children}
    plan = json.loads(data['program.json'], object_pairs_hook=unique_fields)
    if not isinstance(plan, dict) or plan.get('emit_sha256') != expected_seal:
        raise ValueError('[SNABF_NATIVE_PACKAGE] plan/IR seal mismatch')
    # Also catch a renamed byte-for-byte StandardCKKS helper, including one whose
    # package declaration alone was changed. Do not reject unrelated Lattigo APIs.
    static_helper_body = '85e98601745f8a9c2a319fd1577008a6d6b2697a8ce270f93e2911a2f5cb1982'
    for n in children - {'program.json'}:
        if digest(b'\n'.join(data[n].splitlines()[1:])) == static_helper_body:
            raise ValueError('[SNABF_NATIVE_PACKAGE] copied static HEIR helper')
    out.parent.mkdir(parents=True, exist_ok=True)
    go, env = pinned_go()
    with tempfile.TemporaryDirectory(prefix='.snabf-native-', dir=out.parent) as td:
        stage = Path(td)/'module'
        pkg = stage/'planned'
        pkg.mkdir(parents=True)
        for n, b in data.items():
            (pkg/n).write_bytes(b)
        for n in ('go.mod', 'go.sum'):
            shutil.copyfile(ROOT/n, stage/n)
        for rel in BACKEND:
            dest = stage/rel
            dest.mkdir(parents=True, exist_ok=True)
            for p in (ROOT/rel).glob('*.go'):
                if not p.name.endswith('_test.go'):
                    shutil.copyfile(p, dest/p.name)
        expected_version = re.search(r'^go ([0-9.]+)$', (stage/'go.mod').read_text(), re.M)[1]
        if _go(go, env, stage, 'env', 'GOVERSION').strip() != 'go'+expected_version:
            raise ValueError('[SNABF_NATIVE_PACKAGE] toolchain/version mismatch')
        before = snapshot(stage)
        # Run a Go assertion on the actual compiled constant, not a source substring.
        probe = pkg/'seal_n1_test.go'
        with probe.open('x', encoding='ascii') as f:
            f.write('package planned\nimport "testing"\nfunc TestN1CheckedSeal(t *testing.T) {\n'
                    f'const checked = "{expected_seal}"\nif PlanSHA256 != checked {{ t.Fatal("plan/Go seal mismatch") }}\n}}\n')
        paths = _dependencies(go, env, stage)
        _go(go, env, stage, 'build', '-mod=readonly', './...')
        _go(go, env, stage, 'test', '-mod=readonly', '-count=1', './...')
        probe.unlink()
        if snapshot(stage) != before:
            raise ValueError('[SNABF_NATIVE_PACKAGE] build/test mutated release files')
        receipt = dict(schema='snabf.native-package-build.v1', emit_sha256=expected_seal,
                       program_file_sha256=before['planned/program.json'], files=before,
                       toolchain='go'+expected_version, dependencies=paths)
        (stage/'package-build.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n', encoding='utf-8')
        (pkg/'program.json').chmod(0o444)
        if out.exists() or out.is_symlink():
            raise ValueError('[SNABF_OUTPUT_EXISTS] publication target appeared')
        stage.rename(out)
    return receipt
