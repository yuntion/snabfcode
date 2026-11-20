"""Native EB-1/2/3 gates on real Pass 3 SSA and pinned estimator output (no ciphertext timing)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def exercise(opt, baseline, estimates, out, crossing=None, cross_estimates=None,
             abi_shell=None, shell_estimates=None):
    out.mkdir(parents=True, exist_ok=False)
    cache_digest = hashlib.sha256(estimates.read_bytes()).hexdigest()
    env = os.environ.copy()
    floor = 'SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS'
    env.pop(floor, None)
    env['SNABF_PASS4_ESTIMATES'] = str(estimates.resolve())
    strict_env = {**env, floor: '512'}
    events = []

    def run(source, flags, dest, settings, rejected=False):
        result = subprocess.run([str(opt), '--allow-unregistered-dialect', '--mlir-print-op-on-diagnostic=false',
                                 *flags, str(source), '-o', str(dest)], env=settings, capture_output=True, text=True)
        if rejected:
            assert result.returncode > 0 and 'error:' in result.stderr, (source, flags, result.stderr)
            events.append(dict(input=source.name, gate=flags[0], reason=result.stderr.split('error:', 1)[1].splitlines()[0].strip()))
        elif result.returncode != 0:
            raise RuntimeError(f'{source} {flags}: {result.stderr}')

    def plan(label, source, settings):
        audit, ir = out / (label+'.json'), out / (label+'.mlir')
        run(source, [f'--snabf-pass4=audit={audit}', '--snabf-check-plan'], ir, settings)
        return ir, json.loads(audit.read_text())

    # The same immutable measurements are used in both arms; only the floor moves.
    positive, good = plan('positive', baseline, env)
    strict, bad = plan('strict', baseline, strict_env)
    assert good['outcome'] == 'ring_down' and bad['outcome'] == 'retain_source_ring'
    assert [v['role'] for v in good['key_domains']] == ['source', 'target', 'switching_key_ext']
    assert [v['role'] for v in bad['key_domains']] == ['source'] and 'conversion' not in bad
    for record, bits in [(good,128),(bad,512)]:
        assert record['policy']['switching_key_security_bits'] == bits
        assert record['policy']['source_security_bits'] == record['policy']['target_security_bits'] == 128
    cut = good['baseline_cut']; assert len(cut['live_values']) == cut['ciphertext_use_count'] == 1
    v = cut['live_values'][0]
    assert v['q'] and v['p'] and len(v['q']) == v['active_level']+1
    assert v['key_domain'] == good['key_domains'][0]['key_id'] and v['uses'][0]['role'] == 'main'
    assert v['physical_log_scale'] == v['layout']['log2_scale']
    code = 'switching_key_ext_modulus_security_rejected'
    r = next(r for r in bad['rounds'] if r['reason'] == code)
    assert not r['checked'] and r['security']['switching_key_security_bits'] == 512
    for role in ('switch_QP_small_secret', 'switch_QP_large_ring'):
        assert r['security'][role]['modulus'] == r['key_domains'][2]['qp_modulus']
    min_attack = min(v for role in ('switch_QP_small_secret','switch_QP_large_ring') for k,v in r['security'][role].items()
                     if k in ('usvp','dual_hybrid') and isinstance(v,(float,int)))
    assert min_attack < 512
    run(strict, ['--snabf-check-plan','--snabf-lower-ckks'], out/'strict.lowered.mlir', env)
    for gate in ['--snabf-check-plan', '--snabf-lower-ckks']:
        run(positive, [gate], os.devnull, strict_env, True)
    _, hardened = plan('hardened', positive, strict_env)
    _, repeated = plan('repeated', strict, env)
    assert hardened == repeated == bad

    # Mutate MLIR only, never the estimator input. The asserted plans have
    # already passed check-plan, so lowering cannot dismiss them as unsealed.
    def mutation(label, ir, anchor, pattern, replacement, settings=env):
        left, marker, right = ir.read_text().partition(anchor)
        assert marker, anchor
        changed,count = re.subn(pattern,replacement,right,count=1)
        assert count == 1 and right != changed, label
        path=out/(label+'.mlir'); path.write_text(left+marker+changed)
        for gate in ['--snabf-check-plan','--snabf-lower-ckks']:
            run(path,[gate],os.devnull,settings,True)

    mutation('changed-live-scale',strict,'snabf.ring_search =',r'physical_log_scale = [^ ]+', 'physical_log_scale = 3.0')
    mutation('changed-use',positive,'snabf.ring_decision =',r'user_index = \d+', 'user_index = 900')
    modulus=good['key_domains'][2]['qp_modulus']
    # Scope to the conversion-key row, not an identical QP in the target domain.
    mutation('changed-switch-basis',positive,'snabf.ring_decision =',r'(key_kind = "ring_down"[^}]*?qp_modulus = ")'+modulus+'"', r'\g<1>17"')
    mutation('fake-rejection',strict,'snabf.ring_search =',r'reason = "'+code+'"', 'reason = "no_net_benefit"')
    mutation('weak-trace',strict,'snabf.ring_search =',r'switching_key_security_bits = 512', 'switching_key_security_bits = 128')
    mutation('weak-anchor',strict,'snabf.pass4_effective_switching_key_security_bits = ',r'^512', '128')
    mutation('missing-search',strict,'snabf.ring_',r'^search =', 'discarded_search =')
    mutation('false-attack',strict,'snabf.ring_search =',r'(switch_QP_large_ring = \{dual_hybrid = )[^ ]+', r'\g<1>999.0')
    no_cache = env.copy(); no_cache.pop('SNABF_PASS4_ESTIMATES')
    run(strict,['--snabf-check-plan'],os.devnull,no_cache,True)
    if crossing:
        cross_env = {**env, 'SNABF_PASS4_ESTIMATES':str((cross_estimates or estimates).resolve())}
        crossed, result = plan('crossing', crossing, cross_env)
        assert result['outcome'] == 'retain_source_ring' and 'conversion' not in result
        assert [d['role'] for d in result['key_domains']] == ['source']
        cross_code = 'residual_bypass_crosses_cut_unconverted'
        assert result['rounds'] and all(not r['checked'] and r['reason'] == cross_code for r in result['rounds'])
        uses = [u for v in result['baseline_cut']['live_values'] for u in v['uses']]
        assert len(uses) == result['baseline_cut']['ciphertext_use_count'] and {u['role'] for u in uses} == {'main','unconverted'}
        run(crossed,['--snabf-check-plan','--snabf-lower-ckks'],out/'cross.lowered.mlir',cross_env)
        mutation('fake-bypass',crossed,'snabf.ring_search =',r'reason = "'+cross_code+'"', 'reason = "no_net_benefit"',cross_env)
        # Scheduling a dead public preload on the suffix must not hide the
        # real residual. Conversely a missing store cannot be inferred public.
        lines=crossing.read_text().splitlines(keepends=True)
        sink=next(i for i,line in enumerate(lines) if '"snabf.sink"' in line)
        load=next(line for line in lines[:sink] if ' = "preprocessing.load"' in line and 'element_type = !lwe.lwe_plaintext<' in line)
        preload=re.sub(r'^(\s*)%[\w.]+ =',r'\g<1>%eb_preload =',load)
        good_path=out/'with-public-load.mlir'
        good_path.write_text(''.join(lines[:sink]+[preload]+lines[sink:]))
        _,public=plan('replayed-public-load',good_path,cross_env)
        assert public['target']==result['target'] and len(public['baseline_cut']['live_values'])==2
        assert any('literal_table_site' in u for v in public['baseline_cut']['public_values'] for u in v['uses'])
        bad_path=out/'unknown-public-load.mlir'
        forged,n=re.subn(r'site_id = \d+ : i32','site_id = 2147483647 : i32',preload)
        assert n==1
        bad_path.write_text(''.join(lines[:sink]+[forged]+lines[sink:]))
        run(bad_path,['--snabf-pass4'],os.devnull,cross_env,True)
        assert 'unclassified_live_nonpublic_value' in events[-1]['reason']
    if abi_shell:
        shell_env = {**env, 'SNABF_PASS4_ESTIMATES':str((shell_estimates or estimates).resolve())}
        _, adaptive = plan('shell-adaptive', abi_shell, shell_env)
        source_ring = adaptive['key_domains'][0]['ring_dim']
        fixed_out, fixed_audit = out/'shell-fixed.mlir',out/'shell-fixed.json'
        run(abi_shell, [f'--snabf-pass4=min-ring={source_ring} audit={fixed_audit}', '--snabf-check-plan', '--snabf-lower-ckks'], fixed_out, shell_env)
        fixed = json.loads(fixed_audit.read_text())
        assert fixed['target']==source_ring and adaptive['target']<source_ring
        for cut in (fixed['baseline_cut'],adaptive['baseline_cut'],adaptive['conversion']['cut']):
            assert len(cut['live_values']) == cut['ciphertext_use_count'] == 1
            storage, = cut['output_abi_storage']; use, = storage['uses']
            assert storage['fully_overwritten'] and storage['overwrite_index']==0 and storage['producer']=='tensor.empty'
            assert use['role']=='output_abi_storage' and use['user']=='tensor.insert' and use['operand']==1
        # Fixture mutations, not changes to the production frontend or topology.
        lines=abi_shell.read_text().splitlines(keepends=True)
        sink_i=next(i for i,line in enumerate(lines) if '"snabf.sink"' in line)
        ins_i=next(i for i in range(sink_i+1,len(lines)) if ' = tensor.insert ' in lines[i])
        insert=re.search(r'into (%[\w.]+)\[(%[\w.]+)\] : (tensor<.+>)\s*$',lines[ins_i]); assert insert
        dest,zero,tensor=insert.groups()
        arg_line=next(line for line in reversed(lines[:sink_i]) if ' = tensor.extract %arg0[' in line)
        arg=re.search(r'(%[\w.]+) = tensor.extract %arg0\[.*?\] : (tensor<.+>)\s*$',arg_line); assert arg
        raw,raw_type=arg.groups()
        variants = {
            'shell-read':f'    %eb_read = tensor.extract {dest}[{zero}] : {tensor}\n'+lines[ins_i],
            'shell-not-zero':'    %eb_one = arith.constant 1 : index\n'+lines[ins_i].replace(f'[{zero}]','[%eb_one]'),
            'raw-ct-capture':f'    %eb_capture = tensor.from_elements {raw} : {raw_type}\n'+lines[ins_i],
        }
        for label, replacement in variants.items():
            path=out/(label+'.mlir');path.write_text(''.join(lines[:ins_i]+[replacement]+lines[ins_i+1:]))
            for flag in ['--snabf-pass4',f'--snabf-pass4=min-ring={source_ring}']:
                run(path,[flag],os.devnull,shell_env,True)
                assert 'unclassified_live_nonpublic_value' in events[-1]['reason'], events[-1]
    assert hashlib.sha256(estimates.read_bytes()).hexdigest() == cache_digest
    summary=dict(positive=good['target'],strict=bad['target'],switch_min_bits=min_attack,
                 estimator_sha256=cache_digest,checked_crossing=bool(crossing),checked_output_shell=bool(abi_shell),denied=events)
    (out/'gates.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f"{len(events)} rejected gate attempts; retained N={bad['target']}; observed {min_attack:.4f} < 512; crossing={bool(crossing)}")


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('opt','baseline','estimates','out'): ap.add_argument(key,type=Path)
    ap.add_argument('--crossing',type=Path); ap.add_argument('--cross-estimates',type=Path)
    ap.add_argument('--abi-shell',type=Path); ap.add_argument('--shell-estimates',type=Path)
    exercise(**vars(ap.parse_args()))
