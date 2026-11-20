"""Specialize HEIR with native Pass 3 proposals, then emit only a checked plan."""
import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from .pass2 import Pipeline, GENERIC, EPSILON
from .pass2_evidence import save, load, require


def profile(plan):
    return dict(logN=plan['ring'].bit_length()-1,Q=plan['q'],P=plan['p'],logDefaultScale=plan['scale'],
                encryptionType='pk',encryptionTechnique='standard')


def selected(args, work, target, pars):
    target.mkdir(); f=target/'profile.json'; save(f,pars)
    pp=Pipeline(args,f); w=target/work; oracle=pp.source(work,w); imported=pp.import_ckks(work,w)
    pp.call(*pp.opt, f'--snabf-pass2-candidates=depth=12 epsilon={EPSILON} max-output-error={EPSILON}',
            '--snabf-plan','--snabf-check-plan',imported,'-o',w/'selected.mlir')
    return pp,w,oracle


def parameter(pp,w,d,opts,security=None):
    d.mkdir(exist_ok=True); suffix=f' audit={d}/plan.json'+(f' security={security}' if security else '')
    pp.call(*pp.opt,'--snabf-pass3='+opts+suffix,w/'selected.mlir','-o',d/'planned.mlir')
    return load(d/'plan.json')


def emit(pp,d):
    pp.call(*pp.opt,'--snabf-check-plan',d/'planned.mlir','-o',d/'checked.mlir')
    pp.call(*pp.opt,'--snabf-lower-ckks',*GENERIC,d/'checked.mlir','-o',d/'ckks.mlir')
    pp.call(pp.heir,'--scheme-to-lattigo',*GENERIC,d/'ckks.mlir','-o',d/'lattigo.mlir')
    pp.call(*pp.opt,'--snabf-bind-lattigo=audit='+str(d/'audit.json'),*GENERIC,d/'lattigo.mlir','-o',d/'bound.mlir')
    pp.call(pp.heir,d/'bound.mlir','-o',d/'verified.mlir')
    pp.call(pp.translate,'--emit-lattigo','--package-name=planned',d/'verified.mlir','-o',d/'translated.go.txt')
    audit=load(d/'audit.json');save(d/'routes.json',audit['backend']);pp.call(pp.goemit,'emit',d/'routes.json',d/'translated.go.txt',d/'model.go')


def prepare(a):
    a.root=a.root.resolve();a.out=a.out.resolve();out=a.out
    stock=load(a.root/'test/fixtures/reference_profile.json');manifest={};cap=float(os.environ.get('SNABF_P3_INITIAL_CAP','350'))
    for work in ('reuse64','residual'):
        block=out/work;block.mkdir();pp0,w0,oracle=selected(a,work,block/'initial',stock)
        shutil.copyfile(w0/'corpus.json',block/'corpus.json');shutil.copyfile(w0/'input.mlir',block/'input.mlir')
        manifest[work]={}
        for arm in ('no_trunc','joint','matched_exact'):
            dest=block/arm;dest.mkdir();policy='max-k=0' if arm=='no_trunc' else ('fixed-exact=true' if arm=='matched_exact' else 'max-k=1');opts=f'{policy} security-cap={cap}'
            if arm!='matched_exact':
                feedback=[]
                for attempt in range(8):
                    pp,w,pars=pp0,w0,stock;history=[];opts=f'{policy} security-cap={cap}'
                    for i in range(6):
                        plan=parameter(pp,w,dest/f'proposal{attempt}-{i}',opts);new=profile(plan);history.append(new)
                        if new==pars:break
                        require(new not in history[:-1], 'Pass 3 profile/route feedback cycle')
                        pars=new;pp,w,_=selected(a,work,dest/f'specialize{attempt}-{i}',pars)
                    else:raise RuntimeError('Pass 3 specialization did not reach a fixed point')
                    save(dest/f'specialization{attempt}.json',history)
                    parameter(pp,w,dest,opts)
                    try:pp.call('sage','-python',a.root/'src/tools/pass3_security.py',dest/'plan.json',dest/'security.json')
                    except subprocess.CalledProcessError as ex:
                        if ex.returncode!=3:raise
                        rejected=load(dest/'security.json');need_cap=rejected['new_cap'];require(need_cap<cap,'non-tightening estimator feedback')
                        feedback.append({'previous_cap':cap,**rejected});save(dest/'security-feedback.json',feedback);cap=need_cap;continue
                    break
                else:raise RuntimeError('security refinement did not converge')
                if arm=='joint':joint=(pp,w,pars)
            else: pp,w,pars=joint
            plan=parameter(pp,w,dest,opts);require(profile(plan)==pars,'HEIR parameters do not match MILP')
            # A matched arm can reuse precisely the SAME security screen.
            if arm=='matched_exact':shutil.copyfile(block/'joint/security.json',dest/'security.json')
            parameter(pp,w,dest,opts,dest/'security.json');emit(pp,dest);save(dest/'profile.json',pars)
            manifest[work][arm]={'source':str(w/'selected.mlir'),'options':opts}
    save(out/'manifest.json',manifest)
    harness=out/'go_test';harness.mkdir();shutil.copyfile(a.root/'test/pass3/harness_test.go',harness/'harness_test.go')
    prefix='snabf.local/compiler/'+out.relative_to(a.root).as_posix();imports=[];lines=[]
    for work,arms in manifest.items():
        obj=[]
        for arm in arms:
            alias=f'm{len(imports)}';imports.append(f'{alias} "{prefix}/{work}/{arm}"');fn=alias+'.Linear_silu_linear__'
            obj.append('{"'+arm+'",'+','.join(fn+k for k in ('configure','preprocessing','preprocessed','encrypt__arg0','decrypt__result0'))+'}')
        lines.append('func Test_'+work+'(t *testing.T){experiment(t,"../'+work+'",[]program{'+','.join(obj)+'})}')
    (harness/'models_test.go').write_text('package pass3_test\nimport (\n"testing"\n'+'\n'.join(imports)+'\n)\n'+'\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('root','out','heir-bin','snabf-opt','goemit'):ap.add_argument('--'+n,type=Path,required=True)
    args=ap.parse_args();args.extra_rotation_cap=None;prepare(args)
