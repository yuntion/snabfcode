"""Independent regression checker for raw trials and stale native Pass 3 plans."""
import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

def load(p):return json.loads(p.read_text())
def need(b,why):
    if not b:raise ValueError(why)
def check(out,opt):
    all_stats={};negative=[]
    for w in ('reuse64','residual'):
        base=out/w;cases=load(base/'corpus.json');rows_by_arm={};all_stats[w]={}
        for arm in ('no_trunc','joint','matched_exact'):
            d=base/arm;p=load(d/'plan.json');s=load(d/'metrics.json');e=p['edges'][p['sink']];security=load(d/'security.json');rows=[json.loads(l) for l in (d/'trials.jsonl').read_text().splitlines()];rows_by_arm[arm]=rows
            need(len(rows)==len(cases)==64 and [r['trial'] for r in rows]==list(range(64)),f'{w}/{arm}: corpus')
            need(all(len(r['decoded'])==len(c['truth']) for r,c in zip(rows,cases)), 'logical width')
            err=[abs(y-v) for r,c in zip(rows,cases) for y,v in zip(r['decoded'],c['truth'])];need(all(math.isfinite(x) and x<=2**-12 for x in err),'precision')
            need(math.isclose(max(err),s['Linf'],abs_tol=1e-16) and math.isclose(sum(x*x for x in err)/len(err),s['MSE'],rel_tol=1e-12),'metrics mismatch')
            need(all(r['level']==e['Level'] and abs(r['log_scale']-e['physical_log_scale'])<1e-7 for r in rows),'ABI')
            events=s['Events'] or [];need(len(events)==len([x for x in p['edges'] if x['K']]),'runtime trace')
            need(math.prod(p['q']+p['p'])==int(security['instances']['QP']['modulus']),'security screened another modulus')
            all_stats[w][arm]={**s,'bits_qp':sum(math.log2(x) for x in p['q']+p['p']), 'scale':p['scale'],'q':p['q'],'p':p['p'],'min_qp_bits_security':min(security['instances']['QP']['attacks_log2_rop'].values()),'workload_sites':len(p['edges']),'cuts':p['tightening_cuts'],
                'envelope_security':min(security['instances']['envelope']['attacks_log2_rop'].values()),'skipped_rows':sum(v['RowsSkipped'] for v in events),'net_decomposition_ntt_saved':sum(v['NetDecompNTTSaved'] for v in events),'program_sha256':hashlib.sha256((d/'model.go').read_bytes()).hexdigest()}
        j=load(base/'joint/plan.json');x=load(base/'matched_exact/plan.json');need((j['q'],j['p'],j['scale'])==(x['q'],x['p'],x['scale']),'unmatched baseline')
        diffs=[abs(a-b) for jr,er in zip(rows_by_arm['joint'],rows_by_arm['matched_exact']) for a,b in zip(jr['decoded64'],er['decoded64'])];need(math.isclose(max(diffs),all_stats[w]['joint']['PairedLinf'],rel_tol=1e-12) and max(diffs)>0,'paired differential')
        signed=[a-b for jr,er in zip(rows_by_arm['joint'],rows_by_arm['matched_exact']) for a,b in zip(jr['decoded64'],er['decoded64'])];mean=sum(signed)/len(signed);all_stats[w]['joint']['paired_empirical_variance']=sum((v-mean)**2 for v in signed)/len(signed)
    checked=out/'reuse64/joint/checked.mlir';txt=checked.read_text()
    # Mutate BOTH copies of the native record. A shallow manifest equality
    # check alone cannot detect this; concrete arithmetic replay must reject.
    for key,expr,repl in [('wrong-k',r'\bK = 1 : i64','K = 999 : i64'),('fake-scale',r'physical_log_scale = [^,} ]+','physical_log_scale = 9.900000e+01')]:
        changed,count=re.subn(expr,repl,txt);need(count>1,'missing mutation anchors');path=out/(key+'.mlir');path.write_text(changed)
        for flag in ('--snabf-check-plan','--snabf-lower-ckks'):
            r=subprocess.run([opt,'--allow-unregistered-dialect',flag,path,'-o','/dev/null'],capture_output=True,text=True);need(r.returncode and ('illegal truncation' in r.stderr or 'stale Pass 3' in r.stderr),'corrupt plan accepted or parser-only failure');negative.append({'case':key,'pass':flag,'diagnostic':r.stderr.splitlines()[0][:500]})
    sweep=[]
    for eps in (0.0006,0.00032):
        path=(out/f'budget-{eps}.json').resolve();subprocess.run([opt,'--allow-unregistered-dialect',f'--snabf-pass3=min-scale=38 max-scale=38 min-base=30 epsilon={eps} audit={path}',out/'reuse64/initial/reuse64/selected.mlir','-o','/dev/null'],check=True)
        p=load(path);sweep.append(dict(epsilon=eps,p_bits=p['continuous_p'],sites=[e['ID'] for e in p['edges'] if e['K']]))
    need(set(sweep[1]['sites'])<set(sweep[0]['sites']) and sweep[1]['p_bits']>sweep[0]['p_bits'],'missing joint precision response')
    (out/'summary.json').write_text(json.dumps(dict(workloads=all_stats,negative=negative,compile_only_budget_sweep=sweep),indent=2)+'\n');print('PASS: paired ciphertexts, independent slot and parameter replay, stale-plan rejection, and joint budget sweep')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('out',type=Path);ap.add_argument('opt',type=Path);a=ap.parse_args();check(a.out,a.opt)
