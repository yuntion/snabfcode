"""C1: compile two unseen source graphs, then check every real decrypted slot."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src/tools'), str(ROOT/'src/frontend/snabf_frontend')]
from gelu_witness import evidence
from pass2_evidence import require, sha
from snabf_compile import GENERIC, ckks_pipeline
from verify_e2e_pass1_4 import c1_reconstruction, c1_selected, c1_verify


def source(seed, mixed):
    rng = random.Random(seed)
    v, m = 'tensor<4xf32>', 'tensor<4x4xf32>'
    ops = ['#id = affine_map<(i) -> (i)>', 'module {',
           f'func.func @f{rng.getrandbits(60):x}(%x: {v} {{secret.secret}}) -> {v} {{',
           '%zero = arith.constant 0.0 : f32', '%one = arith.constant 1.0 : f32',
           '%half = arith.constant 0.5 : f32', '%inv = arith.constant 0.7071067811865475244 : f32',
           f'%empty = tensor.empty() : {v}', f'%init = linalg.fill ins(%zero : f32) outs(%empty : {v}) -> {v}']

    def linear(i, arg):
        weights = [[rng.choice((-12, -9, -7, -3, 2, 5, 8, 13))/64 for _ in range(4)] for _ in range(4)]
        ops.extend([f'%w{i} = arith.constant dense<{json.dumps(weights)}> : {m}',
                    f'%l{i} = linalg.matvec ins(%w{i}, {arg} : {m}, {v}) outs(%init : {v}) -> {v}'])
        return f'%l{i}'

    def activation(arg, kind):
        out = f'%{kind}'
        ops.extend([f'{out} = linalg.generic {{indexing_maps = [#id, #id], iterator_types = ["parallel"], domain_lower = -1.0 : f64, domain_upper = 1.0 : f64}} ins({arg} : {v}) outs(%empty : {v}) {{',
                    '^bb0(%in: f32, %old: f32):'])
        if kind == 'silu':
            ops.extend(['%n = arith.negf %in : f32', '%e = math.exp %n : f32', '%d = arith.addf %one, %e : f32',
                        '%s = arith.divf %one, %d : f32', '%p = arith.mulf %in, %s : f32'])
        else:
            ops.extend(['%z = arith.mulf %in, %inv : f32', '%e = math.erf %z : f32', '%d = arith.addf %one, %e : f32',
                        '%s = arith.mulf %in, %half : f32', '%p = arith.mulf %d, %s : f32'])
        ops.extend(['linalg.yield %p : f32', f'}} -> {v}'])
        return out

    h = linear(0, '%x')
    if mixed:
        s = activation(h, 'silu')
        ops.append(f'%res = arith.addf {s}, {h} : {v}')
        h = linear(1, '%res')
    y = linear(2 if mixed else 1, activation(h, 'gelu'))
    return '\n'.join([*ops, f'return {y} : {v}', '}', '}', ''])


def store(path, value):
    # Protocol files have identical bytes on Windows and Linux (including LF).
    path.write_bytes((json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())


def prepare(root, seed):
    for case in ('single','mixed'):
        d=root/case; d.mkdir(); src=d/'source.mlir'; src.write_bytes(source(seed,case=='mixed').encode())
        corpus,w=evidence(src); require(w['graph']==case, 'oracle/source topology disagreement')
        store(d/'inputs.json',[t['x'] for t in corpus]); store(d/'corpus.json',corpus)
        store(d/'witness.json',dict(w,corpus_sha256=sha(d/'corpus.json')))


def check_reconstructions(root):
    """Replay real emitted cases, then corrupt one binding at a time in memory only."""
    report={}
    for case in ('single','mixed'):
        d=root/case
        model=d/'model' if (d/'model').is_dir() else d
        ir=model/'work/checked.mlir' if model!=d else d/'checked.mlir'
        sites=c1_selected(ir,case); program=json.loads((model/'program.json').read_bytes())
        c1_reconstruction(program,sites)
        edges=program['instructions']; ids={e['ID']:i for i,e in enumerate(edges)}; rejected=0

        def reject(label, changes, reason):
            nonlocal rejected
            trial=edges.copy()
            for index,fields in changes: trial[index]=dict(trial[index],**fields)
            try: c1_reconstruction(dict(program,instructions=trial),sites)
            except ValueError as exc:
                require(reason in str(exc),f'{case}/{label}: unexpected rejection: {exc}')
                rejected+=1
            else: raise AssertionError(f'{case}/{label}: altered program accepted')

        for i,s in enumerate(sites):
            def idx(k,end): return ids[f'poly{i}.n{s[k]}.{end}']
            half=idx('half','pt0'); half_rs=idx('half','rescale'); mul=idx('product','mul')
            relin=idx('product','relin'); prod_rs=idx('product','rescale'); out=idx('output','add')
            a,b=edges[mul]['inputs']; left,right=edges[out]['inputs']; upstream=ids[s['input_edge']]
            # Use an actual alignment edge from the emitted graph. No fixture opcode list represents a model.
            require(edges[a]['op']==edges[left]['op']=='drop','missing bypass/drop regression coverage')
            before=edges[left]['inputs'][0]; core=idx('core',s['core_end'])
            reject('wrong coefficient',[(half,{'values':[0.25]})],'coefficient or original x')
            reject('wrong x even when both branches agree',[(half,{'inputs':edges[upstream]['inputs']}),
                   (a,{'inputs':edges[upstream]['inputs']})],'coefficient or original x')
            reject('wrong product bypass',[(mul,{'inputs':[b,b]})],'coefficient or original x')
            reject('inner core edge',[(mul,{'inputs':[a,edges[core]['inputs'][0]]})],'selected core result')
            reject('false core opcode',[(core,{'op':'relin'})],'selected core result')
            for impostor in ('rescale','level_reduce'):
                reject('drop changed to '+impostor,[(left,{'op':impostor})],'unaligned final add')
            reject('drop with two inputs',[(left,{'inputs':[before,before]})],'expected drop with 1 operand')
            reject('cyclic drop',[(left,{'inputs':[left]})],'non-topological')
            reject('drop without descent',[(left,{'Level':edges[before]['Level']})],'drop must lower level')
            reject('scale-changing drop',[(left,{'physical_log_scale':edges[left]['physical_log_scale']+0.125})],'preserve physical scale')
            reject('wrong drop domain',[(left,{'kernel_ring':edges[left]['n']//2})],'invalid C1 ring')
            reject('wrong product scale',[(mul,{'physical_log_scale':edges[mul]['physical_log_scale']+0.125})],'unaligned product')
            reject('unbound relin',[(relin,{'inputs':[b]})],'unbound relinearization')
            reject('false half rescale',[(half_rs,{'inputs':[upstream]})],'rescale operand')
            reject('skipped relin',[(prod_rs,{'inputs':[mul]})],'rescale operand')
            reject('wrong Q rescale',[(prod_rs,{'physical_log_scale':edges[prod_rs]['physical_log_scale']+0.125})],'Q-based scale')
            reject('unaligned final level',[(out,{'inputs':[before,right]})],'unaligned final add')
            reject('unaligned final scale',[(out,{'physical_log_scale':edges[out]['physical_log_scale']+0.125})],'unaligned final add')
        report[case]=dict(checked_activations=len(sites),rejected_mutations=rejected,program_sha256=sha(model/'program.json'))
    return report


def native_frontend(root, env, heir, opt, seed, meter):
    def mark(src, dest, reject=None):
        generic=dest.with_suffix('.generic'); subprocess.run([heir,*GENERIC,'--mlir-print-debuginfo',src,'-o',generic],check=True,env=env)
        r=subprocess.run([opt,'--allow-unregistered-dialect','--verify-each','--mlir-print-op-on-diagnostic=false',
                          '--snabf-mark-activations=static-entry=true',*GENERIC,generic,'-o',dest],env=env,capture_output=True,text=True)
        if reject:
            require(r.returncode>0 and reject in r.stderr and not dest.exists(),f'{src.name}: wrong front-end rejection: {r.stderr}')
        else: require(r.returncode==0,r.stderr)
    neg=root/'source-tests'; neg.mkdir(); text=(root/'single/source.mlir').read_text()
    def variant(label, program, reject=None):
        src=neg/(label+'.mlir'); src.write_bytes(program.encode()); mark(src,neg/(label+'.marked'),reject)
    tagged=text.replace('domain_lower =','snabf.activation = "gelu", domain_lower =').replace('%p = arith.mulf %d, %s :','%p = arith.mulf %d, %s {snabf.activation = "gelu"} :')
    variant('markers',tagged)
    variant('divide',text.replace('0.7071067811865475244','1.4142135623730951').replace('arith.mulf %in, %inv','arith.divf %in, %inv'))
    variant('tanh-tagged',tagged.replace('math.erf','math.tanh'),'[SNABF_UNSUPPORTED_OP]')
    variant('wrong-scale',text.replace('0.7071067811865475244','0.70710683'),'[SNABF_UNSUPPORTED_OP]')
    variant('wrong-input',text.replace('%z = arith.mulf %in','%z = arith.mulf %old'),'[SNABF_UNSUPPORTED_OP]')
    variant('unknown',text.replace('%e = math.erf','%dead = math.sin %in : f32\n%e = math.erf'),'[SNABF_UNSUPPORTED_OP]')
    variant('approximate',tagged.replace('domain_lower =','approximate = "tanh", domain_lower ='),'approximate activation is not exact')
    # The same exact expansion also occurs as tensor arithmetic (without a linalg region).
    start=text.index('%gelu = linalg.generic'); end=text.index('} -> tensor<4xf32>',start)+len('} -> tensor<4xf32>')
    tensor='\n'.join(['%t1 = arith.constant dense<1.0> : tensor<4xf32>', '%th = arith.constant dense<0.5> : tensor<4xf32>',
                      '%tc = arith.constant dense<0.7071067811865475244> : tensor<4xf32>',
                      '%z = arith.mulf %l0, %tc : tensor<4xf32>', '%e = math.erf %z : tensor<4xf32>',
                      '%d = arith.addf %e, %t1 : tensor<4xf32>', '%s = arith.mulf %d, %th : tensor<4xf32>',
                      '%gelu = arith.mulf %l0, %s {snabf.activation = "gelu"} : tensor<4xf32>'])
    variant('tensor',text[:start]+tensor+text[end:]); variant('tensor-tanh',text[:start]+tensor.replace('math.erf','math.tanh')+text[end:],'[SNABF_UNSUPPORTED_OP]')
    for case in ('single','mixed'):
        d=root/case; mark(d/'source.mlir',d/'marked.mlir'); level=8 if case=='single' else 16
        with (d/'profile.json').open('wb') as f: subprocess.run([seed,'-log-n','14' if case=='single' else '15','-q-count',str(level)],env=env,check=True,stdout=f)
        size=int(subprocess.check_output([meter,'key-size',d/'profile.json'],env=env,text=True)); require(size>0,'no measured key size')
        options=f'parameter-file=profile.json greedy-modulus-switch-after-mul=true greedy-level-budget={level-1} min-slot-count=4096'
        subprocess.run([heir,'--annotate-module=backend=lattigo scheme=ckks','--linalg-canonicalizations',*ckks_pipeline(True,options),
                        *GENERIC,'marked.mlir','-o','heir.mlir'],check=True,cwd=d,env=env)
        native=[opt,'--allow-unregistered-dialect','--verify-each','--mlir-print-op-on-diagnostic=false']
        subprocess.run([*native,f'--snabf-import-heir=input-bound=1 key-bytes={size}','heir.mlir','-o','imported.mlir'],cwd=d,env=env,check=True)
        subprocess.run([*native,'--snabf-pass1-candidates',f'--snabf-pass2-candidates=depth={level-1} epsilon=0.000244140625 max-output-error=0.000244140625',
                        '--snabf-plan','imported.mlir','-o','selected.mlir'],cwd=d,env=env,check=True)
        subprocess.run([*native,'--snabf-check-plan','selected.mlir','-o','rechecked.mlir'],cwd=d,env=env,check=True)
        print(case, c1_selected(d/'rechecked.mlir',case), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    dest=parser.add_mutually_exclusive_group(required=True)
    dest.add_argument('--out',type=Path,help='new output directory')
    dest.add_argument('--check-reconstruction',type=Path,metavar='DIR',help='audit saved single/mixed program.json and checked.mlir, including negative controls; no Go or decrypt claim')
    parser.add_argument('--seed',type=int,default=2341); parser.add_argument('--security-cache',type=Path)
    mode=parser.add_mutually_exclusive_group(); mode.add_argument('--witness-only',action='store_true'); mode.add_argument('--frontend-only',action='store_true')
    a=parser.parse_args()
    if a.check_reconstruction:
        if a.frontend_only or a.witness_only: parser.error('reconstruction replay requires only the saved evidence directory')
        print(json.dumps(check_reconstructions(a.check_reconstruction),indent=2)); return
    root=a.out.resolve(); root.mkdir(parents=True,exist_ok=False); prepare(root,a.seed)
    if a.witness_only:
        print(json.dumps(c1_verify(root,public_only=True),indent=2)); return
    ext=Path(os.environ.get('SNABF_EXTERN',ROOT/'extern')).resolve()
    sdk=Path(os.environ.get('SNABF_MLIR_ROOT',ext/'mlir24/usr/lib/llvm-24')).resolve()
    opt=Path(os.environ.get('SNABF_OPT',ROOT/'build/native/ir/snabf-opt')).resolve()
    heir=Path(os.environ.get('HEIR_BIN_DIR',ext/'heir-release/heir')).resolve()/'heir-opt'
    go=ext/'go/bin/go'; env=dict(os.environ,GOPATH=str(ext/'gopath'),GOCACHE=os.environ.get('GOCACHE',str(root/'go-cache')),
        GOWORK='off',GOPROXY='off',GOTOOLCHAIN='local',GOFLAGS='',LD_LIBRARY_PATH=os.pathsep.join(map(str,(sdk/'lib',sdk.parent/'x86_64-linux-gnu',ext/'highs-install/lib',os.environ.get('LD_LIBRARY_PATH','')))))
    if a.frontend_only:
        tools=[]
        for pkg in ('src/tools/seed_profile','src/compiler/Target/Lattigo/go-bind'):
            exe=root/Path(pkg).name; subprocess.run([go,'build','-mod=readonly','-o',exe,'./'+pkg],cwd=ROOT,env=env,check=True); tools.append(exe)
        native_frontend(root,env,heir,opt,*tools); return
    for case in ('single','mixed'):
        d=root/case; model=d/'model'
        cmd=[sys.executable,ROOT/'src/tools/snabf_compile.py','--mlir',d/'source.mlir','--out-dir',model,'--snabf-opt',opt,
             '--input-bound','1','--precision-bits','12','--mode','keep-source','--emit-go',
             '--log-n','14' if case=='single' else '15','--q-count','8' if case=='single' else '16',
             '--security-cache',(a.security_cache or root/'security-cache').resolve()]
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        subprocess.run([opt,'--allow-unregistered-dialect','--verify-each','--snabf-check-plan',model/'work/planned.mlir','-o',d/'rechecked.mlir'],env=env,check=True)
        test=model/'c1_trials_test.go'; shutil.copyfile(Path(__file__).with_name('phase2_cut_trials.go.in'),test)
        subprocess.run([go,'test','-mod=readonly','-v','-count=1',model/'model.go',test],cwd=ROOT,env=dict(env,
                       SNABF_INPUTS=str(d/'inputs.json'),SNABF_KEYS=str(d/'keys.json'),SNABF_TRIALS=str(d/'trials.jsonl')),check=True)
    verified=c1_verify(root); check_reconstructions(root); store(root/'acceptance.json',verified); print(json.dumps(verified,indent=2))


if __name__=='__main__': main()
