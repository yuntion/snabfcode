"""Compile a caller-supplied static Linear/SiLU/exact-GeLU/Add graph."""
import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile

from compile_security import domain_evidence, source_evidence

ROOT = Path(__file__).resolve().parents[2]
GENERIC = ('--mlir-print-op-generic', '--mlir-print-local-scope')


def ckks_pipeline(protect_erf, options):
    if not protect_erf:
        return ['--torch-linalg-to-ckks='+options]
    # Pinned HEIR linalgPreprocessingBuilder, except linalg-fuse-linear-ops:
    # absorbing 1/sqrt(2) into W would destroy the shared ORIGINAL x of GeLU.
    # No substitute activation or fabricated CKKS instructions enter HEIR.
    return ['--debug-validate-names', '--inline-activations', '--activation-canonicalizations',
            '--linalg-canonicalizations', '--drop-unit-dims', '--fold-constant-tensors',
            '--canonicalize', '--symbol-dce', '--sccp', '--cse', '--linalg-canonicalizations',
            '--reduction-canonicalizations', '--softmax-canonicalizations',
            '--mlir-to-ckks=enable-arithmetization=true '+options]


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False, indent=2)+'\n')


def scale_search_mode(mode, legacy_heterogeneous):
    if mode == 'narrow' and legacy_heterogeneous:
        raise ValueError('[SNABF_INVALID_OPTIONS] narrow conflicts with --heterogeneous-scales')
    return mode or ('heir-anchor' if legacy_heterogeneous else 'narrow')


def gadget_options(a):
    # Absence is distinct from explicit alpha=1 (the frozen v1 and fixed-block
    # v2 contracts respectively). 0 asks the native solver to search {1, 2}.
    options = []
    if a.gadget_alpha is not None:
        options.append(f'gadget-alpha={a.gadget_alpha}')
    if scale_search_mode(a.scale_mode, a.heterogeneous_scales) == 'heir-anchor':
        options.extend(('heterogeneous-scales=true', 'heir-scale-anchor=true'))
    if not (20 <= a.prime_bw_min < a.prime_bw_max <= 59 and 2 <= a.aux_prime_margin_bits <= 59):
        raise ValueError('[SNABF_INVALID_OPTIONS] invalid prime search bounds or block margin')
    if (a.prime_bw_min, a.prime_bw_max, a.aux_prime_margin_bits) != (24, 59, 2):
        options.extend((f'prime-bw-min={a.prime_bw_min}', f'prime-bw-max={a.prime_bw_max}',
                        f'aux-prime-margin-bits={a.aux_prime_margin_bits}'))
    if a.max_k is not None:
        if a.max_k < 0:
            raise ValueError('[SNABF_INVALID_OPTIONS] max-k must be nonnegative')
        options.append(f'max-k={a.max_k}')
    return ''.join(' ' + opt for opt in options)


def checked_pricing(a):
    if a.cost_profile is None and a.cost_hardware_id is None:
        return None
    if (a.cost_profile is None or not re.fullmatch('[0-9a-f]{64}', a.cost_hardware_id or '')
            or a.gadget_alpha is None or a.mode != 'keep-source'):
        raise ValueError('[SNABF_INVALID_OPTIONS] measured pricing needs --cost-profile, --cost-hardware-id, explicit --gadget-alpha and --mode keep-source; cross-ring costs are not calibrated')
    return a.cost_profile.resolve(strict=True)


def compile_model(a):
    if getattr(a, 'frontend', 'legacy-heir') != 'legacy-heir':
        raise ValueError('[SNABF_NATIVE_FRONTEND_UNAVAILABLE] native-linalg is contract-only in N0; no HEIR fallback')
    if a.security_bits != 128:
        raise ValueError('[SNABF_UNSUPPORTED_SECURITY_TARGET] ordinary domains currently replay 128 bits only')
    certification = a.certify_failure_bits is not None or a.certify_failure_prob is not None
    if certification and (a.certify_failure_bits not in (None,40) or
                          a.certify_failure_prob not in (None,2.0**-40) or not a.mlir.is_file()):
        raise ValueError('[SNABF_PROBABILISTIC_CERT_UNAVAILABLE] need a concrete source and supported target 2^-40')
    if not (0 < a.input_bound < math.inf and 1 <= a.precision_bits <= 52 and
            128 <= a.switch_security_bits < 2**63 and 10 <= a.log_n <= 17 and
            2 <= a.q_count <= 32 and 1 <= a.min_slots <= 2**(a.log_n-1)):
        raise ValueError('[SNABF_INVALID_OPTIONS] invalid interval, ring, chain, slots or threshold')
    gadget = gadget_options(a)
    cost_file = checked_pricing(a)
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
    fraction = a.source_arithmetic_fraction
    if fraction is None: fraction = 0.5 if a.cut_profile == 'dual-live-v1' else 1.0
    if not 0 < fraction <= 1: raise ValueError('[SNABF_INVALID_OPTIONS] source arithmetic fraction must be in (0,1]')
    with tempfile.TemporaryDirectory(prefix='.snabf-',dir=out.parent) as tmp:
        staging = Path(tmp); publish = staging/'publish'; publish.mkdir()
        w = publish/'work'; w.mkdir()
        if cost_file is not None:
            # Snapshot once so every specialization pass sees identical bytes.
            shutil.copyfile(cost_file, w/'gadget-profile.json')
            gadget += f' cost-profile=gadget-profile.json cost-hardware-id={a.cost_hardware_id}'
        interval = '' if a.interval_mode is None else ' interval-mode='+a.interval_mode
        def run(*argv, cwd=w, capture=False):
            return subprocess.run(list(map(str,argv)), cwd=cwd,env=env,check=True,text=True,
                                  stdout=subprocess.PIPE if capture else None).stdout
        def opt(*args):
            run(native,'--allow-unregistered-dialect','--verify-each','--mlir-print-op-on-diagnostic=false',*args)
        run(heir,*GENERIC,'--mlir-print-debuginfo',src,'-o','source.mlir')
        opt('--snabf-mark-activations=static-entry=true receipt=activations.json',*GENERIC,'source.mlir','-o','marked.mlir')
        marker = json.loads((w/'activations.json').read_text())
        if marker.get('schema')!='snabf.activation-pipeline.v1' or type(marker.get('protect_erf')) is not bool:
            raise ValueError('[SNABF_INVALID_ACTIVATION_RECEIPT] invalid marker output')
        protect_erf = marker['protect_erf']
        def tool(pkg):
            exe=staging/Path(pkg).name
            run(go,'build','-mod=readonly','-o',exe,'./'+pkg,cwd=ROOT)
            return exe
        seed = tool('src/tools/seed_profile'); meter = tool('src/compiler/Target/Lattigo/go-bind')
        seed_alpha = 2 if a.gadget_alpha == 0 else (a.gadget_alpha or 1)
        profile = json.loads(run(seed,'-log-n',a.log_n,'-q-count',a.q_count,'-p-count',seed_alpha,capture=True))
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
                *ckks_pipeline(protect_erf, f'parameter-file=profile.json greedy-modulus-switch-after-mul=true greedy-level-budget={len(profile["Q"])-1} min-slot-count={a.min_slots}'),
                *GENERIC,'marked.mlir','-o','heir.mlir')
            opt(f'--snabf-import-heir=input-bound={a.input_bound} key-bytes={size}',
                '--snabf-pass1-candidates',f'--snabf-pass2-candidates=depth={a.q_count-1} epsilon={eps} max-output-error={eps}'+interval,
                '--snabf-plan','--snabf-check-plan','heir.mlir','-o','selected.mlir')
            opts = f'--snabf-pass3=epsilon={eps} min-base=20 security-cap={cap} source-arithmetic-fraction={fraction}' + gadget
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
                elif 'gadget_alpha' in plan['options']:
                    # Tighten below the failing integer modulus; an arbitrary
                    # multi-bit jump can discard every remaining feasible basis.
                    cap=min(cap-1,min((v-1).bit_length()-1 for v in unsafe))
                else:
                    cap=min(cap-1,min(v.bit_length()-1 for v in unsafe)-8)
                if cap<32: raise ValueError('[SNABF_SECURITY_INFEASIBLE] exhausted search envelope')
                continue
            write(w/'source-security.json',report)
            opt(opts+' security=source-security.json','selected.mlir','-o','screened.mlir')
            break
        else: raise ValueError('[SNABF_SPECIALIZATION_LIMIT] no checked fixed point')
        p4 = f'--snabf-pass4=audit=search.json cut-profile={a.cut_profile}'
        if a.mode == 'force-down': p4 += ' force-down=true'
        if a.mode == 'keep-source': p4+=f" min-ring={plan['ring']}"
        else:
            sizes=json.loads(run(tool('src/tools/pass4_keyshape'),'parameters.json',capture=True))
            write(w/'domains.json',domain_evidence(plan,sizes,a.security_cache,estimator))
            env['SNABF_PASS4_ESTIMATES']=str(w/'domains.json')
        opt(p4,'screened.mlir','-o','planned.mlir')
        audits=publish/'audits'; audits.mkdir()
        noise=audits/'noise_cert_audit.json'
        check=f'--snabf-check-plan=noise-audit=../audits/noise_cert_audit.json precision-bits={a.precision_bits}'
        if a.mode=='force-down': check+=' require-ring-down=true'
        if certification: check+=' require-noise-cert=true'
        try:
            opt(check,'planned.mlir','-o','checked.mlir')
        except subprocess.CalledProcessError:
            if not certification or not noise.is_file(): raise
            # Preserve the denial, never a partially published code package.
            with tempfile.NamedTemporaryFile(dir=out.parent,prefix=out.name+'.noise-denied.',suffix='.json',delete=False) as f:
                f.write(noise.read_bytes())
            raise ValueError('[SNABF_PROBABILISTIC_CERT_UNAVAILABLE] audit: '+f.name) from None
        issued=json.loads(noise.read_text())['probabilistic_cert_issued']
        if issued is not certification: raise ValueError('[SNABF_NOISE_MODE_MISMATCH] inconsistent certificate result')
        search=json.loads((w/'search.json').read_text())
        result=dict(schema='snabf.static-compile.v1',mode=a.mode,outcome=search['outcome'],
                    source_ring=plan['ring'],target_ring=search['target'],security_bits=a.security_bits,
                    switching_key_security_bits=search['policy']['switching_key_security_bits'],
                    precision_bits=a.precision_bits,input_bound=a.input_bound,cut_profile=a.cut_profile,
                    verification='conditional' if issued else 'engineering',probabilistic_cert_issued=issued, specialization=history,
                    source_arithmetic_fraction=plan['source_arithmetic_fraction'])
        if a.interval_mode is not None:
            result['interval_mode'] = a.interval_mode
        if cost_file is not None:
            policy = plan['options']
            if (policy.get('cost_mode') != 'calibrated-sites-v1' or policy.get('cost_hardware_id') != a.cost_hardware_id
                    or not re.fullmatch('[0-9a-f]{64}', policy.get('cost_profile_sha256', ''))):
                raise ValueError('[SNABF_STALE_COST_PROFILE] native receipt does not bind measured pricing')
            result.update({k: policy[k] for k in ('cost_mode','cost_hardware_id','cost_profile_sha256')})
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
    p.add_argument('--frontend', choices=('legacy-heir', 'native-linalg'), default='legacy-heir',
                   help='legacy-heir is executable; native-linalg is reserved by RFC-004')
    p.add_argument('--mlir',type=Path,required=True); p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--input-bound',type=float,required=True); p.add_argument('--precision-bits',type=int,default=12)
    p.add_argument('--security-bits','--security',type=int,default=128)
    p.add_argument('--gadget-alpha', type=int, choices=range(5), help='explicit fixed-block v2 P count; 0 searches {1,2}; omit for legacy single-P')
    p.add_argument('--cost-profile',type=Path,help='measured gadget invocations on ordered Q/P, keep-source only')
    p.add_argument('--cost-hardware-id',help='expected hardware fingerprint, independently supplied by the operator')
    p.add_argument('--interval-mode',choices=('affine','naive'),help='Pass 2 candidate domains (default: legacy naive)')
    p.add_argument('--heterogeneous-scales', action='store_true', help='v2 heterogeneous search constrained to the pinned HEIR integer rescale bin')
    p.add_argument('--scale-mode', choices=('narrow','heir-anchor'),
                   help='narrow by default; heir-anchor is the explicit pinned-HEIR ablation')
    p.add_argument('--prime-bw-min', type=float, default=24, help='continuous width floor for non-base Q and P in v2')
    p.add_argument('--prime-bw-max', type=float, default=59, help='continuous prime width ceiling in v2')
    p.add_argument('--aux-prime-margin-bits', type=int, default=2, help='minimum bits of P headroom over each fixed Q block')
    p.add_argument('--max-k', type=int, help='nonnegative prefix limit; omit for native default')
    p.add_argument('--switch-security-bits',type=int,default=128)
    p.add_argument('--mode',choices=['adaptive','keep-source','force-down'],default='adaptive')
    p.add_argument('--cut-profile',choices=['single-live-v1','dual-live-v1'],default='single-live-v1',help='one periodic boundary; dual-live-v1 enables the terminal residual join')
    p.add_argument('--source-arithmetic-fraction',type=float,help='source search share of remaining arithmetic headroom (single: 1, dual: 0.5); never relaxes concrete acceptance')
    cert=p.add_mutually_exclusive_group()
    cert.add_argument('--certify-failure-bits',type=int)
    cert.add_argument('--certify-failure-prob','--certify-prob',type=float)
    p.add_argument('--emit-go',action='store_true')
    p.add_argument('--security-cache',type=Path,default=Path.home()/'.cache/snabf/measure-v1')
    p.add_argument('--snabf-opt',type=Path); p.add_argument('--log-n',type=int,default=14)
    p.add_argument('--q-count',type=int,default=8); p.add_argument('--min-slots',type=int,default=4096)
    a=p.parse_args()
    try: compile_model(a)
    except (ValueError,OSError,subprocess.CalledProcessError) as e:
        p.exit(1,str(e)+'\n')


if __name__=='__main__': main()
