"""Mandatory N0 contract gates; all observations and built packages are temporary."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = [sys.executable, '-B']
REFUSAL = 'SNABF_NATIVE_FRONTEND_UNAVAILABLE'


def run(*args, cwd, data=None, error=None, env=None):
    proc = subprocess.run(args, cwd=cwd, input=data, capture_output=True, env=env)
    if error is None:
        if proc.returncode != 0:
            raise RuntimeError(f'{args}: {proc.stderr.decode(errors="replace")}')
    elif proc.returncode <= 0 or error.encode() not in proc.stderr:
        raise RuntimeError(f'expected refusal {error}: rc={proc.returncode}, {proc.stderr!r}')
    return proc.stdout


def contracts(opt, tmp):
    src = (HERE/'contracts.mlir').read_text(encoding='utf-8')
    def parse(s, *flags, error=None):
        return run(opt, *flags, '-', cwd=tmp, data=s.encode(), error=error)
    printed = parse(src)
    if not printed or parse(printed.decode()) != printed:
        raise AssertionError('parse/verify/print is not stable')
    changes = [
        ('arg_index=1', 'arg_index=0', 'result port'),
        ('arg_index=1', 'arg_name="y", arg_index=1', 'result port'),
        ('result_index=0', 'arg_index=0', 'result port'),
        ('slot_map=array<i64: 0,3>', 'slot_map=array<i64: 0,0>', 'slot map'),
        ('"32","1"', '"31","1"', 'requested product'),
        ('"32","1"', '"64","2"', 'reduced positive rational'),
        ('batched=true', 'batched=false', 'unsupported encoding branch'),
        ('branch="embed_double_f64"', 'branch="bigfloat_value"', 'encoding branch'),
        ('effective=#eight', 'effective=#snabf.exact_scale<"4503599627370497","562949953421312">', 'effective encoding scale'),
        ('site="add", encoding=#enc_add', 'site="add", encoding=#enc_mul', 'receipt site/operand mismatch'),
        ('dropped_q=array<i64: 449>', 'dropped_q=array<i64: 353>', 'execution order'),
        ('depth_delta=1:i64', 'depth_delta=0:i64', 'Tier 4'),
        ('encoded-one', 'scalar-one', 'Tier 4'),
        ('level_out=1:i64', 'level_out=0:i64', 'invalid levels'),
        ('array<i64: 0,1,2>,array<i64: 3,4,5>', 'array<i64: 0,1,3>,array<i64: 2,4,5>', 'cross-row'),
        ('edge_divisors=array<i64: 3,3>', 'edge_divisors=array<i64: 4,4>', 'edge divisor'),
        (', out_layout=#out} : (!x,tensor', '} : (!x,tensor', 'both layouts'),
    ]
    for old, new, why in changes:
        if old not in src:
            raise AssertionError('fixture drift: '+old)
        parse(src.replace(old,new), error=why)
    parse(src.replace('(%x,%table,%init)', '(%x,%table)').replace('(!x,tensor<2x6xf32>,tensor<2xf32>)','(!x,tensor<2x6xf32>)'), error='original typed outs')
    parse(src.replace('level_out=1:i64','level_out=0:i64').replace('levels_consumed=1:i64','levels_consumed=2:i64').replace('dropped_q=array<i64: 449>','dropped_q=array<i64: 449,353>'), error='multi-prime rescale unavailable')

    # Exercise all four coordination record variants without admitting execution.
    fold = src.replace('mode="exact_prime_match"', 'mode="fold_into_plain", encoding=#enc_mul, '
                       'uses=[{consumer="sub",operand_index=0:i64}], requested=#eight, realized=#eight, residual="0", '
                       'logical_payload_sha256="'+'b'*64+'"')
    parse(fold)
    parse(fold.replace('residual="0"','residual="1/1000"'), error='relative fold residual')
    mismatch = src.replace('mode="exact_prime_match"', 'mode="bounded_mismatch_bias", '
                           'lhs_abs_bound="1", rhs_abs_bound="0", lhs_multiplier="1", rhs_multiplier="1"')
    parse(mismatch)  # Absent allocated_bias denotes zero, not an unlimited budget.
    parse(mismatch.replace('lhs_abs_bound="1",',''), error='two-sided mismatch')

    # Even a *valid* declaration must not get sealed, lowered or published in N0.
    # Exercise a native op without the module tag, and a tag without any such op.
    for specimen in (src, src.replace(' attributes {snabf.frontend_provenance=#src}', ''),
                     'module attributes {snabf.frontend_provenance="native-linalg"} {}',
                     'module attributes {test.nested=[{x=#snabf.exact_scale<"1","1">}]} {}'):
        for flag in ('--snabf-check-plan', '--snabf-lower-ckks', '--snabf-lower-ckks=direct-go=true',
                     '--snabf-bind-lattigo=ring-go=model.go audit=program.json'):
            parse(specimen, flag, error=REFUSAL)
    if any(tmp.iterdir()):
        raise AssertionError('a refused pass emitted a file')
    run(*PY, ROOT/'test/compiler/test_phase2_entry.py', cwd=tmp)


def go_cmd():
    ext = Path(os.environ.get('SNABF_EXTERN', ROOT/'extern')).resolve()
    pinned = ext/'go/bin'/('go.exe' if os.name == 'nt' else 'go')
    exe = str(pinned) if pinned.is_file() else shutil.which('go')
    if not exe:
        raise RuntimeError('Go is required for RFC004 encoding/package gates')
    env = dict(os.environ, GOPROXY='off', GOWORK='off', GOTOOLCHAIN='local', GOFLAGS='')
    if (ext/'gopath').is_dir():
        env.setdefault('GOPATH', str(ext/'gopath'))
    return exe, env


def main(probe, opt):
    with tempfile.TemporaryDirectory(prefix='snabf-rfc004-') as td:
        tmp = Path(td)
        if probe == 'contracts':
            contracts(opt, tmp)
        elif probe in ('fold','layout'):
            run(*PY, HERE/(probe+'.py'), cwd=tmp)
        elif probe == 'encoding':
            go, env = go_cmd()
            observed = tmp/'observed.json'
            observed.write_bytes(run(go,'run','-mod=readonly',HERE/'testdata/encoding/main.go',cwd=ROOT,env=env))
            run(*PY,HERE/'encoding_oracle.py',observed,cwd=tmp)
        elif probe == 'package':
            go, env = go_cmd()
            pkg = tmp/'module'
            shutil.copytree(HERE/'testdata/package',pkg)
            run(go,'build','-mod=readonly','./planned',cwd=pkg,env=env)
            run(go,'test','-mod=readonly','-count=1','./planned',cwd=pkg,env=env)
            run(go,'build','-mod=readonly','model.go',cwd=pkg/'planned',env=env,error='undefined: nativeAdd')
        else:
            raise ValueError(probe)


if __name__ == '__main__':
    main(*sys.argv[1:])
