"""A2-a: rebuild and rerun the real final-mid-q17 rejection probe, or replay it.

Linux: set SNABF_EXTERN and (if separate) HEIR_BIN_DIR. --out NEW snapshots
the checkout's actual source/test bytes and builds from the snapshot. --replay
DIR needs only Python; add --verify-native /trusted/snabf-opt to rerun both
trace-on and trace-off on the four captured inputs. Exit 0 is a diagnostic
gate, never a positive encrypted mid-depth or production feasibility receipt.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
import phase2_multip_middepth as mid
from pass3_isolation import need

ROOT = Path(__file__).resolve().parents[2]
SOURCE = 'b7b67e6f49deec16c7985c56cdee73e4e05be97776661df5a7caab86a3bcd805'
CORPUS = '6687715fcd8385a731cf88358bd03ce24eb2a9ae121e93ccb838566cf766eeb6'
# SHA-256 of the first complete diagnostic source line beginning 'module attributes'
# (including LF) in each arm of the final-mid-q17 archive.
MODULE = {1: 'f11c38fc75b3ef458cc67ecf042d495cf9aefd983d4ee228899e5a12e87a4585',
          2: '7df852bd080002db54bb78e905e3cbc13f82138ac31223f56e1de664e3ad4240'}
FAIL = 'Pass 3 infeasible under requested scale, precision and QP bounds'
sha = mid.ab.digest_file
read = mid.source.read


def packed(obj):
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def tree(path):
    return {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}


def snapshot(out):
    git = ['git', '-c', 'core.fileMode=false']
    # Worktrees made by Windows Git point to a drive-letter gitdir inside WSL.
    dot = ROOT/'.git'
    if sys.platform == 'linux' and dot.is_file():
        gitdir = dot.read_text().strip().removeprefix('gitdir: ')
        if len(gitdir) > 2 and gitdir[1] == ':':
            absolute = subprocess.check_output(['wslpath', '-u', gitdir], text=True).strip()
            git.extend(['--git-dir', absolute, '--work-tree', str(ROOT)])

    def query(*args):
        return subprocess.check_output([*git, *args], cwd=ROOT, env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'))

    paths = ['src', 'test/compiler', 'go.mod', 'go.sum']
    dest = out/'source'
    for b in query('ls-files', '-z', '--cached', '--others', '--exclude-standard', '--', *paths).split(b'\0'):
        if b:
            rel = Path(os.fsdecode(b))
            if not (ROOT/rel).is_file(): continue  # tracked deletion
            (dest/rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/rel, dest/rel)
    return dict(head=query('rev-parse', 'HEAD').decode().strip(),
                branch=query('rev-parse', '--abbrev-ref', 'HEAD').decode().strip(),
                changes=query('status', '--porcelain=v1', '-uall', '--', *paths).decode().splitlines(),
                files=tree(dest))


def native_args(alpha, k):
    cap = {1: 745, 2: 800}[alpha]
    return ['--allow-unregistered-dialect', '--verify-each', '--mlir-print-op-on-diagnostic=false',
            f'--snabf-pass3=epsilon=0.000244140625 min-base=20 security-cap={cap} '
            f'source-arithmetic-fraction=1.0 gadget-alpha={alpha} max-k={k} audit=parameters.json',
            'selected.mlir', '-o', 'proposal.mlir']


def run(out, jobs):
    out.mkdir(parents=True, exist_ok=False)
    src = out/'source'; build = out/'build'; opt = build/'ir/snabf-opt'
    ext = Path(os.environ['SNABF_EXTERN']).resolve(strict=True)
    sdk = Path(os.getenv('SNABF_MLIR_ROOT', ext/'mlir24/usr/lib/llvm-24')).resolve(strict=True)
    heir = Path(os.getenv('HEIR_BIN_DIR', ext/'heir-release/heir')).resolve(strict=True)
    prov = snapshot(out)
    (out/'capture').mkdir(); (out/'tmp').mkdir()
    tap = out/'native-tap'; shutil.copyfile(src/'test/compiler/a2_17q_native_tap.py', tap); tap.chmod(0o755)
    env = dict(os.environ, SNABF_EXTERN=str(ext), HEIR_BIN_DIR=str(heir), SNABF_MLIR_ROOT=str(sdk),
               PYTHONDONTWRITEBYTECODE='1', GOTELEMETRY='off', TMPDIR=str(out/'tmp'),
               GOCACHE=os.getenv('GOCACHE', str(out/'go-cache')), SNABF_PASS3_TRACE_REJECTIONS='1',
               SNABF_A2_REAL_OPT=str(opt), SNABF_A2_CAPTURE=str(out/'capture'),
               LD_LIBRARY_PATH=os.pathsep.join(map(str, [sdk/'lib', sdk.parent/'x86_64-linux-gnu',
                                                        ext/'highs-install/lib', os.getenv('LD_LIBRARY_PATH', '')])))
    def logged(label, argv):
        argv = list(map(str, argv)); prov.setdefault('commands', {})[label] = argv
        with (out/(label+'.log')).open('w') as log:
            rc = subprocess.run(argv, env=env, cwd=src, stdout=log, stderr=subprocess.STDOUT).returncode
        return rc
    need(logged('configure', ['cmake', '-S', src/'src/compiler', '-B', build,
         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG', '-DBUILD_TESTING=ON',
         f'-DCMAKE_PREFIX_PATH={ext}/highs-install', f'-DSNABF_MLIR_ROOT={sdk}', f'-DHEIR={ext}/heir']) == 0,
         'configure failed; see configure.log')
    need(logged('build', ['cmake', '--build', build, '--target', 'snabf-opt', 'Pass3Test', 'GadgetBlocksTest',
                          'Pass3CalibrationTest', '-j', jobs]) == 0, 'build failed; see build.log')
    prov.update(native_sha256=sha(opt), tap_sha256=sha(tap), heir_sha256=sha(heir/'heir-opt'))
    rc = logged('probe', [sys.executable, '-B', src/'test/compiler/phase2_multip_middepth.py',
                          '--out', out/'probe', '--snabf-opt', tap, '--layers', 4, '--log-n', 16, '--q-count', 17,
                          '--input-bound', 0.5, '--trials', 2, '--security-cache', out/'security-cache'])
    prov['probe_exit'] = rc
    (out/'build-receipt.json').write_bytes(packed(prov))
    need(rc == 1, 'not the archived negative probe; inspect probe.log')


def observed(log, alpha, k):
    with log.open('rb') as f:
        lines, module, rejected = [], None, False
        for line in f:
            if line.startswith(b'{') and b'"snabf.pass3.candidate-rejection.' in line:
                lines.append(line.decode('utf8'))
            elif module is None and line.startswith(b'module attributes'):
                module = hashlib.sha256(line).hexdigest()
            elif b': error: ' + FAIL.encode() in line:
                rejected = True
    need(rejected and module == MODULE[alpha], 'not the archived diagnostic module/exit')
    trace_sha = hashlib.sha256(''.join(lines).encode('utf8')).hexdigest()
    return mid.rejection_records(lines, n=65536, q_count=17, alpha=alpha, max_k=k), trace_sha


def frozen_inventory(attempts):
    expected = {(alpha, k, d) for alpha in (1, 2) for k in (0, 2) for d in range(36, 47)}
    need(len(attempts) == len(expected) == 44 and set(attempts) == expected, 'not the frozen 44 candidates')


def replay(out):
    prov = read(out/'build-receipt.json')
    need(prov['files'] == tree(out/'source') and prov['probe_exit'] == 1 and
         sha(out/'native-tap') == prov['tap_sha256'] == sha(out/'source/test/compiler/a2_17q_native_tap.py'),
         'changed source snapshot/tap or wrong exit')
    built = out/'build/ir/snabf-opt'
    if built.exists(): need(sha(built) == prov['native_sha256'], 'changed built executable')
    root = out/'probe'; ab = root/'ablation'
    evidence = mid.assess(root)  # recompute source, oracle, ablation manifest, admission and traces
    need(evidence == read(root/'middepth.json') and sha(root/'source.mlir') == SOURCE and sha(root/'corpus.json') == CORPUS
         and not evidence['middepth_measured'] and not evidence['matrix_pass'], 'wrong frozen 17Q input or measurement')
    record = read(ab/'results.json')
    need([record[n] for n in ('log_n','q_count','min_slots','input_bound','trials','joint_max_k')] ==
         [16,17,4096,.5,2,2] and 'measured_profile_sha256' not in record, 'wrong source probe policy')
    probe_cmd = prov['commands']['probe']
    # These names belong to the Linux producer, even when replaying on Windows.
    original_probe = PurePosixPath(probe_cmd[probe_cmd.index('--out')+1]); original_root = original_probe.parent
    need(original_probe.is_absolute(), 'relative producer path')
    compiler = original_root/'source/src/tools/snabf_compile.py'; tap = original_root/'native-tap'
    need(record['tools'][str(tap)] == prov['tap_sha256'] and
         record['tools'][str(compiler)] == prov['files']['src/tools/snabf_compile.py'], 'wrong driver or tap')
    result = dict(schema='snabf.a2.17q-diagnostic.v1', head=prov['head'], snapshot_sha256=hashlib.sha256(packed(prov['files'])).hexdigest(),
                  native_sha256=prov['native_sha256'], build_receipt_sha256=sha(out/'build-receipt.json'),
                  source_sha256=SOURCE, corpus_sha256=CORPUS, arms={}, graph_infeasibility_proven=False,
                  middepth_measured=False, production_matrix_pass=False)
    need({p.name for p in (out/'capture').iterdir()} == {f'a{a}-k{k}' for a in (1,2) for k in (0,2)},
         'unexpected/missing captured arm')
    inventory = []
    for alpha in (1, 2):
        cell = evidence['cells'][f'{alpha}:narrow']; dir_ = ab/f'a{alpha}-narrow'
        need(cell['admission']['matched_exact']['status'] == 'not_run' and not cell['measured'], 'wrong matched arm')
        cmds = [json.loads(line) for line in (dir_/'commands.jsonl').read_text().splitlines()]
        need([c['stage'] for c in cmds] == ['no_trunc','joint'], 'wrong source search inventory')
        for k, arm in ((0, 'no_trunc'), (2, 'joint')):
            log = dir_/(arm+'.log'); cap = out/'capture'/f'a{alpha}-k{k}'
            cmd = cmds[int(k != 0)]; target = original_probe/'ablation'
            expected = [probe_cmd[0], str(compiler), '--mlir', str(target/'source.mlir'), '--out-dir',
                        str(target/f'a{alpha}-narrow'/arm), '--gadget-alpha', str(alpha), '--max-k', str(k),
                        '--mode', 'keep-source', '--input-bound', '0.5', '--precision-bits', '12', '--security-bits',
                        '128', '--cut-profile', 'single-live-v1', '--emit-go', '--log-n', '16', '--q-count', '17',
                        '--min-slots', '4096', '--snabf-opt', str(tap), '--security-cache', str(original_root/'security-cache')]
            need(cmd['argv'] == expected and cmd['cwd'] == str(original_root/'source'), 'wrong source compiler invocation')
            call = read(cap/'call.json'); argv = native_args(alpha, k)
            need(call['argv'] == argv and call['trace'] == '1' and call['native_sha256'] == prov['native_sha256']
                 and call['inputs'] == {n:sha(cap/n) for n in ('selected.mlir','profile.json')}, 'unbound native invocation')
            profile = read(cap/'profile.json')
            need(profile['logN'] == 16 and len(profile['Q']) == 17 and len(profile['P']) == alpha, 'wrong seed profile')
            with (cap/'selected.mlir').open('rb') as f:
                first = next(f)
                need(hashlib.sha256(first).hexdigest() == MODULE[alpha], 'archived module line != captured native input')
                scheme = f"#ckks.scheme_param<logN = 16, Q = {profile['Q']}, P = {profile['P']}, logDefaultScale = {profile['logDefaultScale']}>"
                need(scheme.encode() in first[:4096], 'seed profile does not match captured module')
            rec = cell['admission'][arm]
            need(rec['status'] == 'compiler_rejected' and rec['reason'] == FAIL and rec['exit_code'] == 1 and rec['log'] == log.name,
                 'wrong compiler admission')
            rows, trace_sha = observed(log, alpha, k)
            need(rows == cell['candidate_rejections'][arm] and [int(r['scale_bits']) for r in rows] == list(range(36,47)),
                 'incomplete/wrong candidates')
            attempts = []
            for r in rows:
                x = r['constraint_isolation']
                need(r['tolerance'] == '0.000244140625' and int(r['security_cap_bits']) == {1:745,2:800}[alpha]
                     and x['scope'] == 'base_width_guard' and (x['min_base_bits'],x['guard_bits'],x['exclusive_ceiling']) == (20,1,59)
                     and x['solver_status'] == 'not_called'
                     and x['excluded'] and not x['min_base_excludes'] and all(j not in r for j in ('q','p','basis_source')),
                     'missing independent pre-solver exclusion or invented physical basis')
                attempts.append(dict(scale_bits=int(r['scale_bits']), **x))
                inventory.append((alpha, k, int(r['scale_bits'])))
            result['arms'][f'a{alpha}/{arm}'] = dict(log_sha256=sha(log), trace_sha256=trace_sha, module_line_sha256=MODULE[alpha],
                selected_sha256=call['inputs']['selected.mlir'], profile_sha256=call['inputs']['profile.json'], attempts=attempts)
    frozen_inventory(inventory)
    result['candidate_count'] = len(inventory)
    return result


def extends_archived_trace(log, archive):
    """Allow only A-1.1's additive diagnostic when checking an older binary's log."""
    def records(path):
        with path.open('rb') as f:
            return [json.loads(line) for line in f if line.startswith(b'{') and
                    b'"snabf.pass3.candidate-rejection.' in line]
    actual, expected = records(log), records(archive)
    if not actual or len(actual) != len(expected): return False
    for r, old in zip(actual, expected):
        extra = r.get('isolation', {}).pop('base_expression', None)
        if not isinstance(extra, dict) or extra.get('schema') != 'snabf.pass3.base-expression.v1' or r != old:
            return False
    return True


def verify_native(out, opt):
    for alpha in (1, 2):
        for k, arm in ((0,'no_trunc'),(2,'joint')):
            cap = out/'capture'/f'a{alpha}-k{k}'
            archive = out/'probe/ablation'/f'a{alpha}-narrow'/(arm+'.log')
            original = observed(archive, alpha, k)
            for trace in ('1','0'):
                with tempfile.TemporaryDirectory() as tmp:
                    w = Path(tmp); (w/'selected.mlir').symlink_to(cap/'selected.mlir')
                    with (w/'native.log').open('w') as log:
                        p = subprocess.run([opt, *native_args(alpha,k)], cwd=w,
                            env=dict(os.environ, SNABF_PASS3_TRACE_REJECTIONS=trace), stdout=log, stderr=subprocess.STDOUT)
                    log = w/'native.log'; actual = observed(log,alpha,k)
                    same_trace = ((actual == original or (actual[0] == original[0] and extends_archived_trace(log,archive)))
                                  if trace == '1' else actual == ([], hashlib.sha256(b'').hexdigest()))
                    need(p.returncode == 1 and not (w/'proposal.mlir').exists() and not (w/'parameters.json').exists() and same_trace,
                         'native replay mismatch')
    print('native replay: 4 traced + 4 production denials; same archived modules; no proposals')


def main():
    p = argparse.ArgumentParser(description=__doc__); m = p.add_mutually_exclusive_group(required=True)
    m.add_argument('--out', type=Path); m.add_argument('--replay', type=Path)
    p.add_argument('--verify-native', type=Path); p.add_argument('--jobs', type=int, default=4)
    a = p.parse_args(); out = (a.out or a.replay).resolve()
    if a.out: run(out, a.jobs)
    result = replay(out)
    if a.out: (out/'diagnosis.json').write_bytes(packed(result))
    else: need(read(out/'diagnosis.json') == result, 'stale diagnostic report')
    if a.verify_native: verify_native(out, a.verify_native.resolve(strict=True))
    print(f"diagnostic_gate=PASS arms={len(result['arms'])} candidates={result['candidate_count']} "
          "highs_called=0 middepth_measured=false production_matrix_pass=false")


if __name__ == '__main__': main()
