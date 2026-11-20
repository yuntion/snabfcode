"""Compile a caller-supplied static Linear/SiLU/Add graph. No benchmark inputs."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from compile_security import domain_evidence, source_evidence

ROOT = Path(__file__).resolve().parents[2]
GENERIC = ('--mlir-print-op-generic', '--mlir-print-local-scope')


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2)+'\n')


def compile_model(a):
    if a.security_bits != 128:
        raise ValueError('[SNABF_UNSUPPORTED_SECURITY_TARGET] ordinary domains currently replay 128 bits only')
    if a.certify_failure_bits is not None:
        raise ValueError('[SNABF_PROBABILISTIC_CERT_UNAVAILABLE] validated primitive and sink contracts required')
    if not (0 < a.input_bound < math.inf and 1 <= a.precision_bits <= 52 and
            128 <= a.switch_security_bits < 2**63 and 10 <= a.log_n <= 17 and
            2 <= a.q_count <= 32 and 1 <= a.min_slots <= 2**(a.log_n-1)):
        raise ValueError('[SNABF_INVALID_OPTIONS] invalid interval, ring, chain, slots or threshold')
    env = os.environ.copy()
    ext = Path(env.get('SNABF_EXTERN', ROOT/'extern')).resolve()
    sdk = Path(env.get('SNABF_MLIR_ROOT',ext/'mlir24/usr/lib/llvm-24')).resolve()
    heir = Path(env.get('HEIR_BIN_DIR', ext/'heir-release/heir')).resolve()/'heir-opt'
    native = (a.snabf_opt or Path(env.get('SNABF_OPT',ROOT/'build/native/ir/snabf-opt'))).resolve()
    go = ext/'go/bin/go'
    env.update(GOPATH=str(ext/'gopath'), GOCACHE=env.get('GOCACHE',str(ext/'go-cache')),
               GOPROXY='off', GOTOOLCHAIN='local', GOWORK='off', GOFLAGS='',
               LD_LIBRARY_PATH=os.pathsep.join(map(str,(sdk/'lib',sdk.parent/'x86_64-linux-gnu',ext/'highs-install/lib',env.get('LD_LIBRARY_PATH','')))))
    # Do not inherit an unrelated run's estimator export. Never weaken an
    # administrator-supplied switching-key floor.
    env.pop('SNABF_PASS4_ESTIMATES',None)
    floor = int(env.get('SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS','128'))
    if not 0 <= floor < 2**63: raise ValueError('[SNABF_INVALID_OPTIONS] process switching floor')
    env['SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS'] = str(max(a.switch_security_bits, floor))
    estimator = Path(env.get('LATTICE_ESTIMATOR',ext/'lattice-estimator'))
    src, out = a.mlir.resolve(strict=True), a.out_dir.absolute()
    if out.is_symlink() or out.exists(): raise ValueError('[SNABF_OUTPUT_EXISTS] use a new output directory')
    out.parent.mkdir(parents=True, exist_ok=True)
    eps = 2.0**-a.precision_bits
    with tempfile.TemporaryDirectory(prefix='.snabf-',dir=out.parent) as tmp:
        staging = Path(tmp); publish = staging/'publish'; publish.mkdir()
        w = publish/'work'; w.mkdir()
        def run(*argv, cwd=w, capture=False):
            return subprocess.run(list(map(str,argv)), cwd=cwd,env=env,check=True,text=True,
                                  stdout=subprocess.PIPE if capture else None).stdout
        def opt(*args):
            run(native,'--allow-unregistered-dialect','--verify-each','--mlir-print-op-on-diagnostic=false',*args)
        run(heir,*GENERIC,'--mlir-print-debuginfo',src,'-o','source.mlir')
        opt('--snabf-mark-activations=static-entry=true',*GENERIC,'source.mlir','-o','marked.mlir')
        def tool(pkg):
            exe=staging/Path(pkg).name
            run(go,'build','-mod=readonly','-o',exe,'./'+pkg,cwd=ROOT)
            return exe
        seed = tool('src/tools/seed_profile'); meter = tool('src/compiler/Target/Lattigo/go-bind')
        profile = json.loads(run(seed,'-log-n',a.log_n,'-q-count',a.q_count,capture=True))
        # Initial envelope is a search hint derived from the provisional basis;
        # it cannot authorize emission. A rejected estimate strictly tightens it.
        cap = math.prod(profile['Q']+profile['P']).bit_length()
        seen = set(); history = []
        for _ in range(16):
            state = (tuple(profile['Q']),tuple(profile['P']),profile['logDefaultScale'],cap)
            if state in seen: raise ValueError('[SNABF_SPECIALIZATION_CYCLE] repeated concrete profile')
            seen.add(state); write(w/'profile.json',profile)
            size = int(run(meter,'key-size','profile.json',capture=True))
            run(heir,'--annotate-module=backend=lattigo scheme=ckks','--linalg-canonicalizations',
                f'--torch-linalg-to-ckks=parameter-file=profile.json greedy-modulus-switch-after-mul=true greedy-level-budget={len(profile["Q"])-1} min-slot-count={a.min_slots}',
                *GENERIC,'marked.mlir','-o','heir.mlir')
            opt(f'--snabf-import-heir=input-bound={a.input_bound} key-bytes={size}',
                '--snabf-pass1-candidates',f'--snabf-pass2-candidates=depth={a.q_count-1} epsilon={eps} max-output-error={eps}',
                '--snabf-plan','--snabf-check-plan','heir.mlir','-o','selected.mlir')
            opts = f'--snabf-pass3=epsilon={eps} min-base=20 security-cap={cap}'
            opt(opts+' audit=parameters.json','selected.mlir','-o','proposal.mlir')
            plan=json.loads((w/'parameters.json').read_text())
            new = dict(profile, Q=plan['q'],P=plan['p'],logDefaultScale=plan['scale'])
            history.append(dict(q=new['Q'],p=new['P'],scale=new['logDefaultScale'],search_cap=cap))
            if new != profile: profile=new; continue
            report=source_evidence(plan,a.security_cache,estimator)
            unsafe=[int(r['modulus']) for r in report['instances'].values() if min(r['attacks_log2_rop'].values())<a.security_bits]
            if unsafe:
                qp=int(report['instances']['QP']['modulus'])
                if len(unsafe)==1 and unsafe[0]==1<<cap and qp.bit_length()<cap:
                    cap=qp.bit_length()  # remeasure this envelope, never infer approval
                else:
                    cap=min(cap-1,min(v.bit_length()-1 for v in unsafe)-8)
                if cap<32: raise ValueError('[SNABF_SECURITY_INFEASIBLE] exhausted search envelope')
                continue
            write(w/'source-security.json',report)
            opt(opts+' security=source-security.json','selected.mlir','-o','screened.mlir')
            break
        else: raise ValueError('[SNABF_SPECIALIZATION_LIMIT] no checked fixed point')
        p4 = '--snabf-pass4=audit=search.json'
        if a.mode == 'keep-source': p4+=f" min-ring={plan['ring']}"
        else:
            sizes=json.loads(run(tool('src/tools/pass4_keyshape'),'parameters.json',capture=True))
            write(w/'domains.json',domain_evidence(plan,sizes,a.security_cache,estimator))
            env['SNABF_PASS4_ESTIMATES']=str(w/'domains.json')
        opt(p4,'screened.mlir','-o','planned.mlir')
        check = '--snabf-check-plan' + ('=require-ring-down=true' if a.mode == 'force-down' else '')
        opt(check,'planned.mlir','-o','checked.mlir')
        search=json.loads((w/'search.json').read_text())
        result=dict(schema='snabf.static-compile.v1',mode=a.mode,outcome=search['outcome'],
                    source_ring=plan['ring'],target_ring=search['target'],security_bits=a.security_bits,
                    switching_key_security_bits=search['policy']['switching_key_security_bits'],
                    precision_bits=a.precision_bits,input_bound=a.input_bound,cut_profile=a.cut_profile,
                    verification='engineering',probabilistic_cert_issued=False, specialization=history)
        if a.emit_go:
            opt('--snabf-lower-ckks=direct-go=true','checked.mlir','-o','lowered.mlir')
            opt('--snabf-bind-lattigo=ring-go=../model.go audit=../program.json','lowered.mlir','-o',os.devnull)
            proof=json.loads((publish/'program.json').read_text())['emit_sha256']
            if f'const PlanSHA256 = "{proof}"' not in (publish/'model.go').read_text():
                raise ValueError('[SNABF_STALE_OUTPUT] Go/audit fingerprint mismatch')
            run(go,'build','-mod=readonly','-o',staging/'model.a',publish/'model.go',cwd=ROOT)
            result['emit_sha256']=proof
        write(publish/'compile.json',result)
        publish.rename(out)  # publish the entire checked artifact set, or none


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mlir',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--input-bound',type=float,required=True); p.add_argument('--precision-bits',type=int,default=12)
    p.add_argument('--security-bits','--security',type=int,default=128)
    p.add_argument('--switch-security-bits',type=int,default=128)
    p.add_argument('--mode',choices=['adaptive','keep-source','force-down'],default='adaptive')
    p.add_argument('--cut-profile',choices=['single-live-v1'],default='single-live-v1',help='currently emittable backend capability')
    p.add_argument('--certify-failure-bits',type=int); p.add_argument('--emit-go',action='store_true')
    p.add_argument('--security-cache',type=Path,default=Path.home()/'.cache/snabf/measure-v1')
    p.add_argument('--snabf-opt',type=Path); p.add_argument('--log-n',type=int,default=14)
    p.add_argument('--q-count',type=int,default=8); p.add_argument('--min-slots',type=int,default=4096)
    a=p.parse_args()
    try: compile_model(a)
    except (ValueError,OSError,subprocess.CalledProcessError) as e:
        p.exit(1,str(e)+'\n')


if __name__=='__main__': main()
