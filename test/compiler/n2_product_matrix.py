"""Same-source four-cell native product gate. Linux, fresh private output, serial Eval."""
import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

from native_import import oracle

ROOT = oracle.ROOT
sys.path.insert(0, str(ROOT/'src/tools'))
from native_package import pinned_go, snapshot, unique_fields
from native_product import screen
from n2.search import digest, measure, need, ntt_prime, search_envelope, sha, store
from product_native_n1 import refusals
from verify_n1_equivalence import corpus, replay_slots

CELLS = [(1,'narrow'), (2,'wide'), (1,'wide'), (2,'narrow')]


def read(file):
    return json.loads(file.read_bytes(), object_pairs_hook=unique_fields)


def run(cmd, log, cwd, env=None):
    store(log.with_suffix('.argv.json'), list(map(str,cmd)))
    with log.open('xb') as output:
        subprocess.run(cmd, stdout=output, stderr=subprocess.STDOUT, env=env, cwd=cwd, check=True)


def verify(p, alpha, mode, cap):
    n,q,aux = p['ring'],p['q'],p['p']; P = math.prod(aux)
    need((n,p['slots'],len(q),len(aux)) == (32768,4096,8,alpha) and p['probabilistic_cert_issued'] is False and
         0 <= p['engineering_upper'] <= 2**-12 and p['n2_search'] == dict(alpha=alpha, scale_mode=mode,
         min_scale=42, max_scale=49, max_k=0, cap_bits=cap, heir_scale_anchor=False, prime_bw=[24,59],
         aux_margin_bits=2), 'changed product profile')
    need(all(ntt_prime(v,n) for v in q+aux) and len(set(q+aux))==8+alpha and math.prod(q)*P <= 1<<cap and
         all(P>=4*math.prod(q[k:k+alpha]) for k in range(0,8,alpha)), 'bad concrete ordered basis')
    operations=[]; key_rows=0
    for e in p['edges']:
        operations.append([e[k] for k in ('id','op','inputs','level','shift')]+[e.get('plain',{}).get('bits')])
        if e['op'] in ('baby','giant','relin'):
            active=q[:e['level']+1]
            digits=[math.prod(active[b:b+alpha]) for b in range(0,len(active),alpha)]
            need(Fraction(e['ks_before_round']) == Fraction(10*n*n*sum(x+1 for x in digits),P),
                 'wrong exact physical key-switch allowance')
            key_rows+=1
    need(key_rows>0,'no key-switch witness')
    return digest(operations)


def matrix(out, cache, opt, cpu):
    need(sys.platform=='linux' and cpu in os.sched_getaffinity(0),'expected an available Linux CPU')
    os.sched_setaffinity(0,{cpu}); out.mkdir(mode=0o700,exist_ok=False)
    src=ROOT/'test/compiler/native_import/c1-single.mlir'; spec=src.with_name('c1.inputs.json')
    label='single_gelu'; vec=out/'vectors.json'
    store(vec,[dict(input=r['x'],want=r['truth']) for r in corpus(label,src)])
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    need(not git('status','--porcelain'), 'benchmark requires a clean source tree')
    go,env=pinned_go(); env['GOMAXPROCS']='1'
    policy=dict(source=sha(src),ports=sha(spec),vectors=sha(vec), cpu=cpu,kernel=platform.release(),
                scheduler_nice=os.getpriority(os.PRIO_PROCESS,0),exclusive_cpu=False, gomaxprocs=1,
                order=[CELLS,list(reversed(CELLS))], inputs_per_block=32,warmups_per_block=3,
                keys='fresh random keys per arm/block; fresh public-key encryption per trial',
                boundary='wall time of bound.Eval only, including live checks and GC; no keygen, encoding, encryption, decryption or I/O',
                head=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),compiler=sha(opt))
    store(out/'protocol.json',policy)
    cells={}; dags=set(); cap=None
    for a,m in CELLS:
        tag=f'a{a}-{m}'; d=out/tag; d.mkdir(); pub=d/'public'
        run([sys.executable,'-B',ROOT/'src/tools/snabf_compile.py','--frontend','native-linalg','--emit-go',
             '--mode','keep-source','--snabf-opt',opt,'--mlir',src,'--input-spec',spec,'--input-bound','1',
             '--log-n','15','--q-count','8','--min-slots','4096','--gadget-alpha',str(a),'--scale-mode',m,
             '--max-k','0','--out-dir',pub,'--security-cache',cache],d/'public.log',d)
        envelope=search_envelope(32768,128,lambda r:measure(r,cache,None))
        need(read(pub/'search-envelope.json')==envelope,'unreplayable envelope')
        cap=envelope['cap_bits']; pkg=pub/'module'; planfile=pkg/'planned/program.json'; p=read(planfile)
        need(p['source']==sha(src) and p['input_spec']==sha(spec) and
             read(pub/'compile.json')['emit_sha256']==p['emit_sha256'],'unbound public source/plan')
        graph=verify(p,a,m,cap); dags.add(graph)
        proof=screen(dict(plan=p,security_cap=cap),cache,None)
        need(read(pub/'security.json')==proof,'estimator measurements differ')
        security={k:min(v['attacks'].values()) for k,v in proof['instances'].items()}
        need(min(security.values())>=128,'failed 128-bit gate')
        work=d/'test-module'; shutil.copytree(pkg,work)
        for f in ('native_single_fixture','n2_eval'):
            shutil.copyfile(Path(__file__).with_name(f+'_test.go.in'),work/'planned'/(f+'_test.go'))
        before=snapshot(work)
        run([go,'test','-mod=readonly','-c','-o',d/'eval.test','./planned'],d/'build.log',work,env)
        need(before==snapshot(work),'build mutated the checked package')
        cells[tag]=dict(q=p['q'],p=p['p'],scale=p['scale'],nonbase_spread_bits=math.log2(max(p['q'][1:])/min(p['q'][1:])),
            q_log2=math.log2(math.prod(p['q'])),qp_log2=math.log2(math.prod(p['p'])*math.prod(p['q'])),
            concrete_security=security,engineering_upper=p['engineering_upper'],ordered_basis_sha256=digest([p['q'],p['p']]),
            plan_sha256=sha(planfile), seal=p['emit_sha256'],dag=graph,binary=sha(d/'eval.test'))
        if (a,m)==(2,'wide'):
            cells[tag]['native_refusals']=refusals(d,opt,os.environ,planfile.read_bytes(),proof,(pub/'checked.mlir').read_text())
        print('published',tag,flush=True)
    need(len(dags)==1,'different arithmetic, public coefficients, routes or levels')
    store(out/'public-cells.json',cells)
    for start,order in [(0,CELLS),(32,reversed(CELLS))]:
        for a,m in order:
            tag=f'a{a}-{m}'; d=out/tag; block=d/f'block-{start}'; block.mkdir()
            def load(): return dict(monotonic_ns=time.monotonic_ns(),loadavg=os.getloadavg(),
                cpu_stat=next(row for row in Path('/proc/stat').read_text().splitlines() if row.startswith(f'cpu{cpu} ')))
            store(block/'before.json',load())
            run([d/'eval.test','-test.run=^TestN2EvalBlock$','-test.v','-test.count=1'],block/'eval.log',block,
                dict(env,N2_START=str(start),TEST_VECTORS=str(vec),N2_SLOTS=str(block/'slots.jsonl'),N2_TIMES=str(block/'times.jsonl')))
            store(block/'after.json',load()); print('measured',tag,start,flush=True)
    for tag,c in cells.items():
        d=out/tag; p=read(d/'public/module/planned/program.json'); times=[]
        with (d/'slots.jsonl').open('xb') as dest:
            for start in (0,32):
                block=d/f'block-{start}'
                with (block/'slots.jsonl').open('rb') as srcfile: shutil.copyfileobj(srcfile,dest)
                rows=[json.loads(line) for line in (block/'times.jsonl').read_bytes().splitlines()]
                need([r['trial'] for r in rows]==list(range(start,start+32)) and all(r['emit_sha256']==c['seal'] and
                     type(r['eval_ns']) is int and r['eval_ns']>0 for r in rows), 'incomplete or foreign timings')
                times.extend(r['eval_ns'] for r in rows)
        sink=p['edges'][p['sink']]; c['precision']=replay_slots(label,src,d/'slots.jsonl',seal=c['seal'],ring=p['ring'],
                                            level=sink['level'],slots=p['slots'],scale=sink['meta'])
        c.update(slots_sha256=sha(d/'slots.jsonl'),n_eval=64,p50_ms=statistics.median(times)/1e6,
                 p95_ms=sorted(times)[math.ceil(len(times)*.95)-1]/1e6,eval_ns=times)
    store(out/'gate.json',dict(schema='snabf.n2-product-eval.v1',protocol_sha256=sha(out/'protocol.json'),
          cells=cells,common_cap=cap,identical_dag=True,probabilistic_cert_issued=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('out','cache','opt'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--cpu',type=int,required=True); a=ap.parse_args()
    matrix(a.out.absolute(),a.cache.absolute(),a.opt.resolve(strict=True),a.cpu)
