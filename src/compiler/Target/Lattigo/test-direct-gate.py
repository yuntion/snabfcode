"""Replay a real planned SSA fixture. No Go parsing or rewriting on the production path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def exercise(opt, fixture, pass4):
    with tempfile.TemporaryDirectory(prefix='snabf-direct-gate-') as tmp:
        root = Path(tmp)
        model, audit = root/'model.go', root/'audit.json'
        def call(src, *flags, out=Path(os.devnull), denied=None):
            p = subprocess.run([str(opt), '--allow-unregistered-dialect', '--mlir-print-op-on-diagnostic=false',
                                *flags, str(src), '-o', str(out)], capture_output=True, text=True)
            if denied is None:
                if p.returncode != 0: raise AssertionError(p.stderr[:600])
            elif p.returncode <= 0 or not any(s in p.stderr for s in denied):
                raise AssertionError(f'{flags}: expected {denied}; got {p.stderr[:400]}')

        bind = f'--snabf-bind-lattigo=ring-go={model} audit={audit}'
        checked, lowered = root/'checked.mlir', root/'lowered.mlir'
        call(fixture, *(['--snabf-pass4'] if pass4 else []), '--snabf-check-plan', out=checked)
        lower = '--snabf-lower-ckks' if pass4 else '--snabf-lower-ckks=direct-go'
        call(checked, lower, out=lowered)
        call(lowered, bind)
        program = json.loads(audit.read_bytes()); code = model.read_text()
        assert re.fullmatch('[0-9a-f]{64}', program['emit_sha256'])
        assert f'const PlanSHA256 = "{program["emit_sha256"]}"' in code
        assert 'epoch.Bind(epoch.Program{' in code and 'epoch.New([]byte' not in code
        edges = program['instructions']
        def count(op): return sum(e['op']==op for e in edges)
        for op, token in [('relin','gadget.RelinearizeNew('),('giant','gadget.Giant('),('ring_down','r.Down('),
                          ('mul','heir.Mul('),('drop','.DropLevelNew('),('rescale','heir.Rescale(')]:
            assert code.count(token) == count(op), (op, token)
        assert code.count('heir.MulPlain(') == count('mask') + count('coefficient')
        assert code.count('gadget.Babies(') == len({(e['route_site'],tuple(e['inputs'])) for e in edges if e['op']=='baby'})
        original=(model.read_bytes(),audit.read_bytes())
        call(lowered, bind); assert (model.read_bytes(),audit.read_bytes())==original

        def corrupt(src, pattern, repl, name):
            text,n=re.subn(pattern,repl,src.read_text(),count=1)
            assert n==1,name
            dest=root/(name+'.mlir'); dest.write_text(text);return dest

        seal=r'snabf\.emit_sha256 = "[0-9a-f]{64}"'
        no_check=corrupt(checked,seal,r'ignored.receipt = "unused"','no-check')
        call(no_check,lower,denied=['snabf.emit_sha256'])
        call(checked,bind,denied=['stale_epoch_lowering'])
        stale=corrupt(lowered,seal,'snabf.emit_sha256 = "'+'0'*64+'"','stale-seal')
        call(stale,bind,denied=['stale_emit_fingerprint'])
        # An edit outside the derived payload must invalidate the full-IR seal.
        stale=corrupt(lowered,seal,r'other.plan_data = "changed", \g<0>','changed-ir')
        call(stale,bind,denied=['stale_emit_fingerprint'])
        stale=corrupt(lowered,r'(snabf\.ckks_(?:epochs|direct) = \{cut_level = )-?\d+',r'\g<1>99','stale-lowering')
        call(stale,bind,denied=['stale_epoch_lowering'])
        call(lowered,f'--snabf-bind-lattigo=ring-go={model} audit={model}',denied=['aliased_go_and_audit_output'])
        link=root/'alias.json'; os.link(model,link)
        call(lowered,f'--snabf-bind-lattigo=ring-go={model} audit={link}',denied=['aliased_go_and_audit_output'])
        call(lowered,f'--snabf-bind-lattigo=ring-go={root} audit={audit}',denied=['unsafe_emitter_output'])
        call(lowered,f'--snabf-bind-lattigo=ring-go={model} audit={root}/missing/a',denied=['cannot_stage_output'])
        assert (model.read_bytes(),audit.read_bytes())==original and not list(root.rglob('*.tmp'))
        print(json.dumps(dict(sha256=program['emit_sha256'],source=program['source_ring'],target=program['target_ring'],
                              sites=len(edges),go_sha256=hashlib.sha256(original[0]).hexdigest(),negative_cases=9)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('opt',type=Path);parser.add_argument('fixture',type=Path)
    parser.add_argument('--pass4',action='store_true',help='run dynamic ring search before sealing')
    args=parser.parse_args();exercise(args.opt.resolve(),args.fixture.resolve(),args.pass4)
