"""DEV-2: three source compilations, independent keys, one common corpus and source Q/P.

Run on a configured Linux checkout with SNABF_EXTERN and SNABF_OPT. The only
model input is --source. The compiler never sees the oracle or prior plans.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/frontend/snabf_frontend'))
sys.path.insert(0, str(ROOT / 'src/tools'))
from pass2_evidence import Oracle, TOLERANCE, load, require, save, sha
from e2e_oracle_witness import freeze as witness_for, replay as verify_witness

MODES = ('keep-source', 'force-down', 'adaptive')


def verify_arm(d, mode, corpus):
    p, c = load(d/'program.json'), load(d/'compile.json')
    search = p['ring_search']; src, dst = p['source_ring'], p['target_ring']
    require(c['mode'] == mode and c['cut_profile'] == 'dual-live-v1' and c['emit_sha256'] == p['emit_sha256'] and
            c['verification'] == 'engineering' and c['probabilistic_cert_issued'] is False and
            c['security_bits'] == 128 and search['policy']['cut_profile'] == c['cut_profile'], 'wrong policy/receipt')
    require(len(search['baseline_cut']['live_values']) == 2, 'not a two-value crossing')
    uses = {(v['producer'], u['user'], u.get('kind')) for v in search['baseline_cut']['live_values'] for u in v['uses']}
    require(uses == {('snabf.poly_eval', 'snabf.lintrans', None), ('snabf.lintrans','snabf.ckks_step','ckks.add')}, 'wrong residual position')
    mixed = src != dst
    require((mode != 'keep-source' or not mixed) and (mode != 'force-down' or mixed), 'wrong execution domain')
    require(('ring_decision' in p) == mixed and search['outcome'] == ('ring_down' if mixed else 'retain_source_ring'), 'stale outcome')
    edges = p['instructions']; sink = edges[p['sink']]; ports = [edges[i] for i in sink['inputs']]
    downs = [edges[e['inputs'][0]]['ID'] for e in edges if e['op']=='ring_down']
    keys = [k for k in load(d/'keys.json') if k['Role']=='down']
    require(sink['op']=='add' and len(ports)==2 and len(keys)==int(mixed) and len(set(downs))==len(downs)==2*int(mixed), 'wrong physical inventory')
    if mixed:
        decision=p['ring_decision']; sw=keys[0]
        require(len(decision['operands'])==2 and decision['terminal_join']['edge']==sink['ID'] and sw['Bytes']==decision['switch_bytes'] and
                sw['KeyID']==decision['source_key'] and sw['TargetKeyID']==decision['target_key'] and sw['LevelQ']==decision['switch_level_q'] and
                sw['LevelP']==decision['switch_level_p'] and sw['N']==src and sw['Bytes']>0,'conversion key not bound to screened domain')
    errors=[]; times=[]; ntrial=0; thresholds=0; first_join=None; actual_calls=0
    with (d/'trials.jsonl').open() as f:
        for row, case in zip(f,corpus,strict=True):
            r=json.loads(row); obs=r['observation']; y=r['decoded']; calls=obs['Converted'] or []
            require(r['trial']==case['trial'] and len(y)==p['slots'] and r['n']==dst and r['level']==sink['Level'] and
                    abs(r['log_scale']-sink['physical_log_scale'])<=1e-8 and r['eval_ns']>0,'invalid output metadata')
            require(calls==downs and obs['JoinSite']==sink['ID'] and len(obs['Join'])==2,'missing runtime cut/join observation')
            for v,e in zip(obs['Join'],ports,strict=True):
                require(v['N']==dst and v['Level']==e['Level']==r['level'] and abs(v['LogScale']-e['physical_log_scale'])<=1e-8,'join port mismatch')
            require(abs(obs['Join'][0]['LogScale']-obs['Join'][1]['LogScale'])<=1e-8,'mismatched physical scales')
            for j,value in enumerate(y):
                delta=abs(value-case['truth'][j%len(case['truth'])]);require(math.isfinite(delta),'nonfinite decoded slot')
                errors.append(delta);thresholds+=delta>TOLERANCE
            first_join=first_join or obs['Join'];actual_calls+=len(calls);times.append(r['eval_ns']);ntrial+=1
    require(ntrial and thresholds==0,'true-SiLU slot threshold exceeded')
    return dict(mode=mode,source=src,target=dst,trials=ntrial,slot_checks=len(errors),violations=thresholds,
                linf=max(errors),mse=math.fsum(e*e for e in errors)/len(errors),eval_mean_ms=statistics.fmean(times)/1e6,
                join=first_join,output_level=r['level'],output_log_scale=r['log_scale'],conversions=actual_calls,
                switches_per_eval=actual_calls//ntrial,serialized_swk_bytes=sum(k['Bytes'] for k in keys),swk_count=len(keys),
                baseline_cost=search['baseline_cost'],chosen_cost=search['selected_cost'],emit_sha256=p['emit_sha256'])


def negatives(d, native, env):
    src=(d/'work/checked.mlir').read_text(); folder=d/'negative';folder.mkdir()
    g=re.compile(r'(?m)^\s*(%[\w.]+) = "snabf\.convert"\((%[\w.]+)\)[^\n]*$'); defs=list(g.finditer(src))
    require(len(defs)==2,'expected exactly two convertible SSA definitions')
    # Both conversion results have the same target type. Replacing the bypass
    # with the converted main value keeps the edit well typed, but incorrect.
    line=defs[1]
    join=re.compile(r'(?m)^(\s*%[\w.]+ = "snabf\.ckks_step"\()([^)]*)(\) <\{kind = "ckks\.add"\}>)')
    m=join.search(src,line.end()); require(m is not None,'terminal add missing')
    ports=m[2].split(', ');require(ports.count(line[1])==1,'bypass does not reach expected use')
    ports[ports.index(line[1])]=defs[0][1]
    missing=src[:line.start()]+src[line.end():m.start(2)]+', '.join(ports)+src[m.end(2):]
    sink=load(d/'program.json'); edges=sink['instructions']; last=edges[sink['sink']]
    port=edges[last['inputs'][-1]]['ID']; at=src.index('snabf.mixed_parameters = ')
    pat=r'(\bID = "'+re.escape(port)+r'"[^{}]*?physical_log_scale = )[-+\w.]+'
    edited,n=re.subn(pat,r'\g<1>9.900000e+01',src[at:],count=1);require(n==1,'missing concrete join port')
    results=[]
    for name,text in [('missing_bypass',missing),('stale_join_scale',src[:at]+edited)]:
        bad=folder/(name+'.mlir');bad.write_text(text)
        subprocess.run([native,'--allow-unregistered-dialect',bad,'-o',os.devnull],env=env,check=True,capture_output=True)
        for gate in ('--snabf-check-plan','--snabf-lower-ckks=direct-go=true'):
            out=folder/(name+gate.split('=')[0]+'.out'); argv=[native,'--allow-unregistered-dialect','--mlir-print-op-on-diagnostic=false',gate,bad,'-o',out]
            z=subprocess.run(argv,env=env,capture_output=True,text=True); why=next((v.partition('error: ')[2] for v in z.stderr.splitlines() if 'error: ' in v),'')
            require(z.returncode>0 and not out.exists() and (any(s in why for s in ('stale','materialized','recomputed','transfer','conversion','unconverted','budget','sink','inconsistent parameter shape'))),f'{name}: wrong rejection: {why}')
            results.append(dict(case=name,gate=gate,reason=why))
    return results


def replay(root):
    from pathlib import PurePosixPath
    manifest={}
    for line in (root/'SHA256SUMS').read_text().splitlines():
        digest, name=line.split('  ',1); path=PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and name not in manifest,'invalid manifest member')
        manifest[name]=digest
    actual={f.relative_to(root).as_posix():sha(f) for f in root.rglob('*') if f.is_file() and f!=root/'SHA256SUMS'}
    require(actual==manifest,'checksum or member set differs')
    a=load(root/'acceptance.json');source=root/'source.mlir';o=Oracle(source.read_text());require(o.cross_residual and not o.residual,'not the cross-residual graph')
    require(a['schema']=='snabf.dev2.cross.v2','expected portable DEV-2 evidence')
    corpus=load(root/'corpus.json')
    verify_witness(o,corpus,{'source.mlir':sha(source),'corpus.json':sha(root/'corpus.json')},root/'exp64.json')
    require(load(root/'inputs.json')==[c['x'] for c in corpus] and a['source_sha256']==sha(source) and a['inputs_sha256']==sha(root/'inputs.json'),'wrong inputs')
    require(a['arms']==[verify_arm(root/m,m,corpus) for m in MODES] and a['threshold']==TOLERANCE and a['probabilistic_cert_issued'] is False,'independent replay mismatch')
    require({(v['case'],v['gate']) for v in a['negatives']}=={(n,g) for n in ('missing_bypass','stale_join_scale') for g in ('--snabf-check-plan','--snabf-lower-ckks=direct-go=true')}, 'incomplete recorded negatives')
    print('Checksums, source oracle, three physical traces and all decoded slots verified.')


def run(a):
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    source=root/'source.mlir';shutil.copyfile(a.source,source)
    oracle=Oracle(source.read_text());require(oracle.cross_residual and not oracle.residual,'expected W1*SiLU(h)+h');corpus=oracle.corpus()
    save(root/'corpus.json',corpus); save(root/'exp64.json',witness_for(root))
    save(root/'inputs.json',[c['x'] for c in corpus]); results=[];basis=None
    ext=Path(os.getenv('SNABF_EXTERN',ROOT/'extern')).resolve();native=Path(os.getenv('SNABF_OPT',ROOT/'build/native/ir/snabf-opt')).resolve()
    env=dict(os.environ,GOPATH=str(ext/'gopath'),GOCACHE=os.getenv('GOCACHE',str(ext/'go-cache')),GOWORK='off',GOPROXY='off',GOTOOLCHAIN='local')
    for mode in MODES:
        d=root/mode; extra=['--security-cache',str(a.security_cache.resolve())] if a.security_cache else []
        argv=[sys.executable,ROOT/'src/tools/snabf_compile.py','--mlir',source,'--input-bound','1','--mode',mode,'--cut-profile','dual-live-v1','--emit-go','--out-dir',d,'--snabf-opt',native,*extra]
        with (root/(mode+'.log')).open('w') as log: subprocess.run(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        p=load(d/'program.json'); b=next(e for e in p['epochs'] if e['n']==p['source_ring']);curr=(b['n'],b['q'],b['p'],b['scale']);require(basis is None or curr==basis,'different source parameters');basis=curr
        test=d/'cut_trials_test.go';shutil.copyfile(Path(__file__).with_name('phase2_cut_trials.go.in'),test)
        with (d/'trials.log').open('w') as log:
            subprocess.run([ext/'go/bin/go','test','-mod=readonly','-count=1','-v',d/'model.go',test],cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT,
                           env=dict(env,SNABF_INPUTS=str(root/'inputs.json'),SNABF_KEYS=str(d/'keys.json'),SNABF_TRIALS=str(d/'trials.jsonl')))
        results.append(verify_arm(d,mode,corpus))
    sdk=Path(env.get('SNABF_MLIR_ROOT',ext/'mlir24/usr/lib/llvm-24'))
    neg=negatives(root/'force-down',native,dict(env,SNABF_PASS4_ESTIMATES=str(root/'force-down/work/domains.json'),LD_LIBRARY_PATH=os.pathsep.join(map(str,[sdk/'lib',sdk.parent/'x86_64-linux-gnu',ext/'highs-install/lib',env.get('LD_LIBRARY_PATH','')]))))
    evidence=dict(schema='snabf.dev2.cross.v2',source_sha256=sha(source),inputs_sha256=sha(root/'inputs.json'),threshold=TOLERANCE,
                  timing='Eval only; independent fresh keys per arm; physical slots include periodic copies',probabilistic_cert_issued=False,arms=results,negatives=neg)
    save(root/'acceptance.json',evidence)
    files=sorted(x for x in root.rglob('*') if x.is_file());(root/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(root)}\n' for f in files))
    replay(root); print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path);p.add_argument('--out',type=Path);p.add_argument('--security-cache',type=Path);p.add_argument('--replay',type=Path,help='verify an existing independent evidence directory, without compiling')
    a=p.parse_args()
    if a.replay and not (a.out or a.source): replay(a.replay.resolve())
    elif not a.replay and a.source and a.out: run(a)
    else: p.error('use --replay DIR or --source MODEL --out NEW_DIR')
