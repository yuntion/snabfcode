"""Accept only observed epochs and reject mutations at both native gates."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def check(net, opt):
    def read(arm,name): return json.loads((net/arm/name).read_text())
    for arm in ('baseline','adaptive','expensive'):
        m,p,a=read(arm,'metrics.json'),read(arm,'program.json'),read(arm,'search.json')
        expected=p['target_ring']; switched=expected!=p['source_ring']
        assert switched==(arm=='adaptive'), (net.name,arm,a)
        assert m['actual_output_ring']==expected==a['target'] and m['trials']==64 and m['misses']==0
        assert m['logical_outputs']==64*p['width'] and m['physical_encoded_slots_checked']==64*p['slots']
        assert m['actual_level']==p['instructions'][p['sink']]['Level'] and m['linf']<=2**-12
        if switched:
            c=a['conversion']; assert a['selected_cost']<a['baseline_cost'] and c['security_min_bits']>=128
            ev=[k for k in m['keys'] if k['Role']=='down'];assert len(ev)==1 and ev[0]['Bytes']==c['switch_bytes'] and ev[0]['LevelQ']==p['cut_level'] and ev[0]['LevelP']==c['switch_level_p']
            ix=next(j for j,i in enumerate(p['instructions']) if i['op']=='ring_down');assert p['instructions'][ix+1:] and all(i['n']==expected for i in p['instructions'][ix:])
        elif arm=='expensive': assert any(r['reason']=='no_net_benefit' for r in a['rounds'])
    a=net/'adaptive'; raw=(a/'planned.mlir').read_text();neg=net/'negative';neg.mkdir(exist_ok=True)
    def mutate(pattern,repl,text):
        value,count=re.subn(pattern,repl,text,count=1);assert count==1 and value!=text,pattern;return value
    def at(marker,pattern,repl,text=raw):
        left,sep,right=text.partition(marker);assert sep,marker;return left+sep+mutate(pattern,repl,right)
    cases={
        'period':at('provided = #snabf.slot_layout',r'period = \d+', 'period = 17'),
        'stride':at('required = #snabf.slot_layout',r'stride = \d+', 'stride = 3'),
        'target_ring':at('required = #snabf.slot_layout',r'ring_dim = \d+','ring_dim = 4096'),
        'unemittable':at('"snabf.convert"',r'"trace-invariant"','"split-repack"'),
        'switch_key':at('snabf.ring_decision =',r'switch_level_q = \d+','switch_level_q = 99'),
        'security_claim':at('snabf.ring_decision =',r'security_min_bits = [^ ]+', 'security_min_bits = 999.0'),
        'basis':at('snabf.mixed_parameters =',r'p = array<i64: \d+','p = array<i64: 17'),
        'gadget':at('snabf.mixed_parameters =',r'K = 0 :','K = 1 :'),
        'scale':at('snabf.mixed_parameters =',r'physical_log_scale = [^ ]+','physical_log_scale = 12.0'),
        'profit':at('snabf.ring_search =',r'selected_cost = [^ ]+','selected_cost = 0.0'),
    }
    entries=[]
    # Replan the same checked mixed SSA, then change a policy on that SSA.
    # Compare semantic choices, not paths or json serialization timestamps.
    for label,policy in [('repeat',''),('revert','switch-price=1000')]:
        path=neg/(label+'.json')
        subprocess.run([str(opt),'--allow-unregistered-dialect','--snabf-pass4='+policy+' audit='+str(path),'--snabf-check-plan',str(a/'planned.mlir'),'-o',os.devnull],check=True)
        record=json.loads(path.read_text());ref=read('adaptive' if label=='repeat' else 'expensive','search.json')
        assert (record['target'],record['selected_cost'])==(ref['target'],ref['selected_cost'])
    def denied(name,text,flags,env=None):
        f=neg/(name+'.mlir');f.write_text(text)
        for flag in flags:
            proc=subprocess.run([str(opt),'--allow-unregistered-dialect','--mlir-print-op-on-diagnostic=false',flag,str(f),'-o',os.devnull],capture_output=True,text=True,env=env)
            assert proc.returncode>0 and 'error:' in proc.stderr, (name,flag,proc.stderr[:500]);entries.append(dict(case=name,gate=flag.split('=')[0],diagnostic=proc.stderr.split('error:',1)[1].splitlines()[0].strip()))
    for k,v in cases.items():denied(k,v,['--snabf-check-plan','--snabf-lower-ckks'])
    env=os.environ.copy();env.pop('SNABF_PASS4_ESTIMATES',None);denied('no_external_evidence',raw,['--snabf-check-plan','--snabf-lower-ckks'],env)
    lowered=(a/'lowered.mlir').read_text();out=neg/'must-not-exist.go'
    denied('lowered_payload',at('snabf.ckks_epochs =',r'cut_level = \d+','cut_level = 99',lowered),['--snabf-bind-lattigo=ring-go='+str(out)])
    assert not out.exists();(neg/'results.json').write_text(json.dumps(entries,indent=2)+'\n');print(net.name,len(entries),'invalid plans rejected')
if __name__=='__main__':check(Path(sys.argv[1]),Path(sys.argv[2]))
