"""Native single-live-v1 regression on a checked W1*SiLU(h)+h checkpoint.

Run with SNABF_PASS4_ESTIMATES pointing at the matching frozen measurements.
This tests rejection, not the future two-value emitter.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def check(opt, checkpoint):
    native=[str(opt.resolve()),'--allow-unregistered-dialect','--verify-each','--mlir-print-op-on-diagnostic=false']
    with tempfile.TemporaryDirectory() as tmp:
        def run(*args,deny=None):
            p=subprocess.run([*native,*args],cwd=tmp,capture_output=True,text=True)
            if (deny is None and p.returncode) or (deny and (p.returncode<=0 or deny not in p.stderr)):
                raise RuntimeError(f'{args}: {p.returncode}: {p.stderr}')
        run('--snabf-pass4=audit=search.json','--snabf-check-plan',str(checkpoint.resolve()),'-o','checked.mlir')
        p=json.loads((Path(tmp)/'search.json').read_text()); live=p['baseline_cut']['live_values']
        code='residual_bypass_crosses_cut_unconverted'
        if not (p['outcome']=='retain_source_ring' and 'conversion' not in p and
                [d['role'] for d in p['key_domains']]==['source'] and len({v['id'] for v in live})==2 and
                {(v['producer'],u['user'],u['role'],u.get('kind')) for v in live for u in v['uses']}=={
                    ('snabf.lintrans','snabf.ckks_step','unconverted','ckks.add'),('snabf.poly_eval','snabf.lintrans','main',None)} and
                p['rounds'] and all(r['reason']==code and not r['checked'] for r in p['rounds'])):
            raise RuntimeError('not a verified two-value crossing on the one-value backend')
        run('--snabf-check-plan=require-ring-down=true','--snabf-lower-ckks','checked.mlir','-o','forced.mlir',
            deny='[SNABF_FORCE_DOWN_REJECTED] '+code)
        if (Path(tmp)/'forced.mlir').exists(): raise RuntimeError('failed forced plan published')
        print('adaptive: 2 distinct live values, retained source; force-down: '+code)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('snabf_opt',type=Path); p.add_argument('checkpoint',type=Path)
    a=p.parse_args(); check(a.snabf_opt,a.checkpoint)
