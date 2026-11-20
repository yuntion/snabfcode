"""Verify direct generation, same-ciphertext comparisons, and two real large-ring fallbacks."""
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from snabf_frontend.pass2_evidence import load, require, save, sha


def test(root, out, opt, keyshape):
    policy_var = 'SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS'
    env = os.environ.copy()
    env.pop(policy_var, None)
    env.pop('SNABF_PASS4_ESTIMATES', None)
    paired, boundary, negatives = {}, {}, []

    def native(src, dest, *passes, settings=None):
        subprocess.run([opt, '--allow-unregistered-dialect', '--mlir-print-op-on-diagnostic=false', *passes, src, '-o', dest], check=True, env=settings or env)

    def emit(net, dest, policy, settings):
        dest.mkdir()
        native(out/net/'input.mlir', dest/'planned.mlir', f'--snabf-pass4={policy} audit={dest}/search.json', settings=settings)
        native(dest/'planned.mlir', dest/'checked.mlir', '--snabf-check-plan', settings=settings)
        native(dest/'checked.mlir', dest/'lowered.mlir', '--snabf-lower-ckks', settings=settings)
        native(dest/'lowered.mlir', os.devnull, f'--snabf-bind-lattigo=ring-go={dest}/model.go audit={dest}/program.json', settings=settings)
        program = load(dest/'program.json')
        code = (dest/'model.go').read_text()
        require(program['emitter']=='snabf.direct-go.v1' and program['emit_sha256'] in code and
                'heir.MulPlain(' in code and 'gadget.RelinearizeNew(' in code and 'epoch.New([]byte' not in code, 'not the direct target')
        return load(dest/'search.json'), program

    module = next(line.split()[1] for line in (root/'go.mod').read_text().splitlines() if line.startswith('module '))
    for net in ('control','strided64','reuse64','residual'):
        d = out/net
        base_n = load(d/'plan.json')['ring']
        direct = d/'direct'
        search, program = emit(net, direct, f'min-ring={base_n}', env)
        require(program['source_ring']==program['target_ring']==base_n and search['target']==base_n, 'not a same-domain comparison')
        cmpdir=d/'paired'; cmpdir.mkdir()
        paths=[module+'/'+f.relative_to(root).as_posix() for f in (direct,d/'legacy')]
        (cmpdir/'adapters.go').write_text('package comparison\nimport (\n' + f'direct "{paths[0]}"\nlegacy "{paths[1]}"\n)\n' +
            'var newDirect = direct.New\nvar legacyConfigure = legacy.Linear_silu_linear__configure\nvar legacyPrepare = legacy.Linear_silu_linear__preprocessing\nvar legacyEval = legacy.Linear_silu_linear__preprocessed\n')
        shutil.copyfile(root/'test/pass4/paired_test.go',cmpdir/'paired_test.go')
        subprocess.run(['go','test','-mod=readonly','-count=1','-tags=snabf_generated','-timeout=30m','-v','.'],cwd=cmpdir,check=True,
            env=dict(env,SNABF_EVIDENCE=str(cmpdir),SNABF_CORPUS=str(d/'corpus.json'),SNABF_PROGRAM=str(direct/'program.json')))
        m=load(cmpdir/'metrics.json');require(m['trials']==64 and m['slots_per_arm']==64*program['slots'] and m['same_ciphertext_and_keys'],'incomplete paired trials')
        paired[net]=dict(corpus_sha256=sha(d/'corpus.json'),direct_go_sha256=sha(direct/'model.go'),legacy_go_sha256=sha(d/'legacy/model.go'),**m)

    # The security rejection reuses *exactly* the positive arm's measured
    # estimates and input; change only its minimum accepted switching-key cost.
    for net in ('reuse64','cross_residual','residual'):
        d=out/net
        with (d/'keyshape.json').open('w') as f: subprocess.run([keyshape,d/'plan.json'],check=True,stdout=f)
        slots=int(re.search(r'scheme.requested_slot_count = (\d+)',(d/'input.mlir').read_text())[1])
        subprocess.run(['sage','-python',root/'src/tools/pass4_security.py',d/'plan.json',d/'keyshape.json',d/'cache.json',env['LATTICE_ESTIMATOR'],'--slots',str(slots)],check=True)
    for name,net,floor in [('positive','reuse64',128),('bypass','cross_residual',128),('security','reuse64',512)]:
        d=out/net; dest=d/name
        settings=dict(env,SNABF_PASS4_ESTIMATES=str(d/'cache.json'),**{policy_var:str(floor)})
        s,p=emit(net,dest,'',settings)
        require(s['policy']['switching_key_security_bits']==floor and
                p['ring_search']=={k:v for k,v in s.items() if k!='conversion'},'search/emit audit mismatch')
        live=s['baseline_cut']['live_values'];roles=[x['role'] for x in s['key_domains']]
        for value in live:
            require(value['uses'] and value['key_domain'] and value['ring_dim']==p['source_ring'] and math.isfinite(value['physical_log_scale']) and
                    len(value['q'])==value['active_level']+1 and value['layout']['active_level']==value['active_level'],'incomplete live ciphertext contract')
        if name=='positive':
            require(len(live)==1 and roles==['source','target','switching_key_ext'] and s['target']<p['source_ring'],'expected legal downsizing')
            x=s['key_domains'][2]; ext=math.prod(x['q']+x['p']);sec=s['conversion']['security']
            for role,n in [('switch_QP_small_secret',s['target']),('switch_QP_large_ring',p['source_ring'])]:
                require(int(sec[role]['modulus'])==ext and sec[role]['n']==n and min(sec[role][k] for k in ('usvp','dual_hybrid'))>=floor,'unscreened switching domain')
        else:
            require(roles==['source'] and s['target']==p['source_ring'] and all(i['op']!='ring_down' for i in p['instructions']),'illegal fallback execution')
            if name=='bypass':
                require(len(live)==2 and s['rounds'] and all(r['reason']=='residual_bypass_crosses_cut_unconverted' for r in s['rounds']),'residual bypass not intercepted')
            else:
                rs=[r for r in s['rounds'] if r['reason']=='switching_key_ext_modulus_security_rejected'];require(rs,'missing conversion-specific refusal')
                require(any(min(r['security']['switch_QP_small_secret'][k] for k in ('usvp','dual_hybrid'))<floor and
                            min(r['security']['QP'][k] for k in ('usvp','dual_hybrid'))>=128 for r in rs),'not a conversion-only failure')
        shutil.copyfile(root/'test/pass4/encrypted_test.go',dest/'encrypted_test.go')
        subprocess.run(['go','test','-mod=readonly','-count=1','-timeout=30m','-v','model.go','encrypted_test.go'],cwd=dest,check=True,
                       env=dict(settings,SNABF_CORPUS=str(d/'corpus.json'),SNABF_EVIDENCE=str(dest)))
        m=load(dest/'metrics.json'); switches=[k for k in m['keys'] if k['Role']=='down']
        require(len(switches)==(name=='positive') and m['actual_output_ring']==s['target'] and m['trials']==64 and m['misses']==0,'incorrect actual execution domain')
        if switches:require(switches[0]['LevelQ']==s['conversion']['switch_level_q'] and switches[0]['Bytes']==s['conversion']['switch_bytes'],'unexpected physical switching key')
        boundary[name]=dict(cache_sha256=sha(d/'cache.json'),cut=s['baseline_cut'],policy=s['policy'],reasons=sorted(set(r['reason'] for r in s['rounds'])),**m)
        # Tamper only with saved IR, never the trusted estimator cache.
        raw=(dest/'checked.mlir').read_text()
        prefix, marker, suffix=raw.partition('snabf.ring_search = ');require(marker,'missing ring record')
        alterations={'live_key':(r'key_domain = "[^"]+"','key_domain = "forged"')}
        if name=='positive': alterations['switch_qp']=(r'qp_modulus = "\d+"','qp_modulus = "17"')
        if name=='security': alterations['refusal']=(r'reason = "switching_key_ext_modulus_security_rejected"','reason = "verified"')
        for key,(pattern,repl) in alterations.items():
            change,n=re.subn(pattern,repl,suffix,count=1);require(n==1 and change!=suffix,'missing tamper fixture')
            path=dest/(key+'.mlir');path.write_text(prefix+marker+change)
            for gate in ('--snabf-check-plan','--snabf-lower-ckks'):
                cmd=[opt,'--allow-unregistered-dialect','--mlir-print-op-on-diagnostic=false',gate,path,'-o',os.devnull]
                pr=subprocess.run(cmd,env=settings,capture_output=True,text=True)
                require(pr.returncode>0 and 'error:' in pr.stderr,'tampered plan admitted or crashed')
                negatives.append(dict(arm=name,case=key,gate=gate,reason=pr.stderr.split('error:')[1].splitlines()[0].strip()))
        # The checker seal is mandatory for Lowering even if math is unchanged.
        unsealed,n=re.subn(r'snabf.emit_sha256 = "[0-9a-f]{64}", ', '', raw);require(n==1,'missing emit seal')
        path=dest/'unsealed.mlir';path.write_text(unsealed)
        pr=subprocess.run([opt,'--allow-unregistered-dialect','--mlir-print-op-on-diagnostic=false','--snabf-lower-ckks',path,'-o',os.devnull],env=settings,capture_output=True,text=True)
        require(pr.returncode>0 and 'error:' in pr.stderr,'unsealed lowering accepted')
        negatives.append(dict(arm=name,case='missing_check',gate='--snabf-lower-ckks',reason=pr.stderr.split('error:')[1].splitlines()[0].strip()))
        altered=(dest/'lowered.mlir').read_text().replace(p['emit_sha256'],'0'*64)
        path=dest/'stale_seal.mlir';path.write_text(altered)
        nowhere=dest/'must-not-be-emitted.go'; report=dest/'must-not-be-emitted.json'
        pr=subprocess.run([opt,'--allow-unregistered-dialect','--mlir-print-op-on-diagnostic=false',f'--snabf-bind-lattigo=ring-go={nowhere} audit={report}',path,'-o',os.devnull],env=settings,capture_output=True,text=True)
        require(pr.returncode>0 and 'error:' in pr.stderr and not nowhere.exists() and not report.exists(),'invalid request published partial Go or audit')
        negatives.append(dict(arm=name,case='stale_seal',gate='--snabf-bind-lattigo',reason=pr.stderr.split('error:')[1].splitlines()[0].strip()))
        if name=='positive':
            strong=dict(settings,**{policy_var:'512'})
            pr=subprocess.run([opt,'--allow-unregistered-dialect','--mlir-print-op-on-diagnostic=false','--snabf-check-plan',dest/'checked.mlir','-o',os.devnull],env=strong,capture_output=True,text=True)
            require(pr.returncode>0 and 'stale_or_weakened_pass4_policy' in pr.stderr,'hardened environment accepted earlier weak plan')
            negatives.append(dict(arm=name,case='hardened_environment',gate='--snabf-check-plan',reason='stale_or_weakened_pass4_policy'))
        if name=='security':
            # A later process without the environment variable must retain 512.
            weak=dict(settings);weak.pop(policy_var)
            native(dest/'checked.mlir',os.devnull,'--snabf-check-plan',settings=weak)
            source=(d/'input.mlir').read_text()
            require('module attributes {' in source, 'expected checked module')
            floor_ir=dest/'ir-floor.mlir';floor_ir.write_text(source.replace('module attributes {','module attributes {snabf.switching_key_security_bits = 512 : i64, ',1))
            ir_log=dest/'ir-floor.json'
            native(floor_ir,os.devnull,f'--snabf-pass4=audit={ir_log}','--snabf-check-plan',settings=dict(weak,**{policy_var:'128'}))
            same=load(ir_log);require(same['policy']['switching_key_security_bits']==512 and same['outcome']=='retain_source_ring','source floor ignored')
    require(boundary['positive']['cache_sha256']==boundary['security']['cache_sha256'],'security cache drift')
    direct_gate=root/'src/compiler/Target/Lattigo/test-direct-gate.py'
    native_gates={}
    for group,flags in [('direct',[]),('mixed',['--pass4'])]:
        settings=dict(env,SNABF_PASS4_ESTIMATES=str(out/'reuse64/cache.json')) if flags else env
        gate_run=subprocess.run([sys.executable,direct_gate,opt,out/'reuse64/input.mlir',*flags],
                                check=True,env=settings,stdout=subprocess.PIPE,text=True)
        dest=out/f'gate-{group}.json';dest.write_text(gate_run.stdout)
        native_gates[group]=load(dest)
    subprocess.run([sys.executable,root/'src/compiler/pass4/tests/replay_gates.py',opt,out/'reuse64/input.mlir',
                    out/'reuse64/cache.json',out/'replay','--crossing',out/'cross_residual/input.mlir',
                    '--cross-estimates',out/'cross_residual/cache.json','--abi-shell',out/'residual/input.mlir',
                    '--shell-estimates',out/'residual/cache.json'],check=True,env=env)
    native_gates['replay']=load(out/'replay/gates.json')
    save(out/'acceptance.json',dict(paired=paired,boundary=boundary,negatives=negatives,native_gates=native_gates))


if __name__=='__main__':
    test(*(Path(x).resolve() for x in sys.argv[1:]))
