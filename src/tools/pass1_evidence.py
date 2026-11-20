"""Bind frozen harnesses to a fresh run, and verify the native decision evidence."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(f):
    return json.loads(f.read_text())


def save(f, data):
    f.write_text(json.dumps(data, indent=2) + '\n')


def check(test, msg):
    if not test:
        raise ValueError(msg)


def overlay(out):
    rep = {}
    for folder, f, old in [('pass1','inference_test.go','build/pass1/'), ('unified','feedback_test.go','build/unified/')]:
        src = ROOT/'test'/folder/f
        dest = out/(folder+'.go.txt')
        dest.write_text(src.read_text().replace(old, out.relative_to(ROOT).as_posix()+'/'))
        rep[str(src)] = str(dest)
    save(out/'overlay.json', {'Replace':rep})


def sites(audit):
    result = sorted(audit['route_sites'], key=lambda s:s['site'])
    refs = Counter()
    work = share = 0.
    for s in result:
        r, c, ctx = s['selected'], s['charge'], s['context']
        refs.update(r['keys']);work += c['hoist_work'] + c['giant_work'] + c['key_cost'];share += c['amortized_key_bytes']
        h = ctx['group_order']//r['subgroup_stride']
        splits = [u for a in s['candidates'] for u in zip(*[iter(a['equivalent_splits'])]*4)]
        check(sorted(v[0] for v in splits)==list(range(h.bit_length())), 'incomplete tower')
        check(all(b==1<<i and b*g==h and stride*b==ctx['group_order'] for i,stride,b,g in splits),'wrong order/index')
        check(r['b_v']*r['g_v']==h and s['level']==ctx['input_level'], 'wrong rung or active level')
    check(refs=={k['galois']:k['refs'] for k in audit['keys']},'wrong global key refs')
    check(math.isclose(work,audit['objective']) and math.isclose(share,audit['key_bytes']), 'site charges do not sum to global cost')
    return result


def collect(out):
    rows = [json.loads(s) for s in re.findall(r'SNABF_PASS1 (\{.*\})',(out/'inference.log').read_text())]
    check(len(rows)==10 and len({(r['workload'],r['arm']) for r in rows})==10,'missing encrypted arm')
    for r in rows:
        m=r['metrics']; check(m['Linf']<=2**-12 and m['MSE']>=0 and m['Outputs']>0,'accuracy failed')
        if r['arm']!='heir':
            d=out/r['workload']/r['arm']; a=read(d/'audit.json'); s=sites(a)
            check([k['galois'] for k in a['keys']]==m['Galois'] and a['key_bytes']==m['SerializedGaloisBytes'],'wrong runtime key set')
            r.update(sites=[dict(level=t['level'],b=t['selected']['b_v'],g=t['selected']['g_v'],terms=t['selected']['terms'],charge=t['charge']) for t in s],
                     rounds=a['route_feedback']['rounds'],program_sha256=hashlib.sha256((d/'model.go').read_bytes()).hexdigest())
    p2={}
    for arm in ('economy','precision'):
        d=out/'pass2/residual'/arm; a=read(d/'audit.json'); actual=read(d/'runtime.json');s=sites(a);m=read(d/'accuracy.json')
        keys=actual['galois_keys'];check([k['galois'] for k in a['keys']]==[k['galois'] for k in keys] and sum(k['serialized_bytes'] for k in keys)==a['key_bytes'],'residual runtime inventory')
        check(m['precision_passed'] and m['logical_slots_checked']==4096,'residual accuracy')
        p2[arm]=dict(candidate=a['polynomials'][0]['selected']['candidate_id'],level=s[1]['level'],terms=s[1]['selected']['terms'],keys=keys,accuracy=m)
    p3={}
    for w in ('reuse64','residual'):
        p3[w]={}
        for arm in ('no_trunc','joint','matched_exact'):
            d=out/'pass3'/w/arm; a=read(d/'audit.json'); pars=read(d/'plan.json');s=sites(a); hist=read(next(d.glob('specialization*.json'))) if arm!='matched_exact' else []
            scope=s[0]['context']['physical_scope'];check(all(t['context']['physical_scope']==scope for t in s),'mixed final physical scopes')
            check(all(str(q) in scope for q in pars['q']+pars['p']),'route scope misses concrete Q/P')
            p3[w][arm]=dict(physical_scope=scope,key_bytes=a['key_bytes'],profiles_in_first_specialization=len(hist),metrics=read(d/'metrics.json'))
    boot=read(out/'periodic_boot/metrics.json');binding=read(out/'periodic_boot/audit.json')['bootstrap']
    check(binding['execution']=='periodic-n15-v1' and boot['linf']<=2**-12 and boot['slots']==512,'missing real periodic bootstrap results')
    save(out/'evidence.json',dict(pass1=rows,pass2_residual=p2,pass3=p3,periodic_bootstrap=dict(binding=binding,metrics=boot),pass4='layout_feedback_interface'))
    print('PASS: ciphertexts, complete tower/union/work records, Pass 2 precision feedback and concrete Pass 3 scopes')


if __name__=='__main__':
    action, dest=sys.argv[1:]; {'overlay':overlay,'collect':collect}[action](Path(dest).resolve())
