"""Freeze verified run artifacts (never reconstruct decisions from measurements)."""
import argparse
import collections
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def collect(run, out, ctest):
    evidence = load(run / 'evidence.json')
    if '100% tests passed' not in ctest.read_text():
        raise ValueError('missing successful C++ suite')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f'archive destination must be empty: {out}')
    metrics, candidates = [], []
    files = [run / n for n in ['commands.jsonl', 'run.log', 'ir-tests.json', 'protocol.json', 'toolchain.json', 'inference.log', 'evidence.json']]
    for group in ['workloads', 'additional_template_coverage']:
        for model, data in evidence[group].items():
            base = run / model
            files.extend(p for p in (base / n for n in ['input.mlir','corpus.json','budget-search.json','precision-comparison.json']) if p.is_file())
            for arm, s in data['arms'].items():
                d = base / arm
                audit = load(d / 'audit.json') if (d / 'audit.json').exists() else None
                keys = s['runtime']['galois_keys']
                row = dict(group=group, model=model, arm=arm, trials=s['trials'], slots=s['logical_slots_checked'],
                           keys=len(keys), key_bytes=sum(k['serialized_bytes'] for k in keys),
                           linf=s['decrypted_vs_true']['linf'], mse=s['decrypted_vs_true']['mse'],
                           mean_ms=s['eval_ns']['mean']/1e6, levels=s['level_pairs'], output_logscale=s['actual_output_log_scale_range'])
                corpus=load(base/'corpus.json')
                observed=[json.loads(line) for line in (d/'trials.jsonl').read_text().splitlines()]
                row['over_limit']=sum(abs(x-y)>2**-12 for a,z in zip(observed,corpus,strict=True)
                                      for x,y in zip(a['decrypted'],z['truth'],strict=True))
                if audit and row['over_limit']: raise ValueError(f'planned program exceeded tolerance: {d}')
                if audit:
                    p = audit['polynomials'][0]
                    c = p['selected']
                    rows = sorted(audit['backend'],key=lambda x:x['site'])
                    row.update(candidate=c['candidate_id'], frontier=len(p['frontier']), activation_error=c['activation_error'],
                               L_DAG=c['gain'], second_level=rows[1]['Level'], second_route=rows[1]['Terms'],
                               rotation_solve_rounds=audit['rounds'], decisions=dict(collections.Counter(v['reason'] for v in audit['decisions'])))
                    for a in p['frontier']:
                        candidates.append(dict(model=model,arm=arm,selected=a==c,id=a['candidate_id'], template=a['template'],depth=a['delta_level'],
                            ct_mul=a['ct_mults'], pt_mul=a['pt_mults'],act_error=a['activation_error'],core_error=a['core_error'],gain=a['gain']))
                metrics.append(row)
                for key in ['contract.json','audit.json','runtime.json','metrics.json','trials.jsonl','selected.mlir','model.go']:
                    f=d/key
                    if f.is_file():
                        digest=hashlib.sha256(f.read_bytes()).hexdigest()
                        if digest != data['files'][str(f.relative_to(run))]: raise ValueError(f'file changed after validation: {f}')
                        files.append(f)
    for f in ['ir-tests.json','inference.log','run.log']:
        shutil.copyfile(run/f,out/f)
    for arm in ['economy',evidence['workloads']['residual']['precision_witness']]:
        shutil.copyfile(run/'residual'/arm/'selected.mlir',out/f'residual-{arm}.mlir')
    shutil.copyfile(ctest,out/'ctest.log')
    for name,values in [('metrics',metrics),('frontiers',candidates)]:
        (out/f'{name}.json').write_text(json.dumps(values,indent=2)+'\n')
    archive=out/'ir-programs-and-trials.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for f in sorted(set(files)):
            tar.add(f,arcname=f.relative_to(run))
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    rows=['# Pass 2 实验结果', '',f'服务器原始运行目录：`{run}`。逐槽比较对象是 **float64 的真实 SiLU 网络**；门限固定为 `2^-12`。', '',
          '|模型|方案|选中 DAG|第二层输入|输入→输出|Galois 密钥|超限槽|L∞|MSE|在线均值 ms|', '|---|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        li,lo=r['levels'][0]
        rows.append(f"|{r['model']}|{r['arm']}|{r.get('candidate','HEIR 原生')}|{r.get('second_level','—')}|{li}→{lo}|{r['keys']}|{r['over_limit']}|{r['linf']:.4g}|{r['mse']:.4g}|{r['mean_ms']:.1f}|")
    rows.extend(['','## 按实验口径汇总',''])
    for label,subset in [('四个主模型（含 HEIR 对照）',[r for r in metrics if r['group']=='workloads']),
                         ('其中自由规划的 SN-ABF 方案',[r for r in metrics if r['group']=='workloads' and 'candidate' in r]),
                         ('额外模板覆盖（含独立对照）',[r for r in metrics if r['group']!='workloads'])]:
        if not subset: continue
        count=sum(r['slots'] for r in subset)
        rows.append(f"- {label}：{sum(r['trials'] for r in subset)} 次推理，{count} 个逻辑输出，超限 {sum(r['over_limit'] for r in subset)} 个；最大绝对误差 {max(r['linf'] for r in subset):.6g}，按输出数加权 MSE {sum(r['slots']*r['mse'] for r in subset)/count:.6g}。")
    rows += ['','常规 precision 组只收紧公开的确定性逼近误差合约。额外 forced 组检验指定模板的密文实现，不是跨模板自由选型证据。',
             '密钥字节为 Galois 密钥序列化之和，不是峰值内存；在线时间不含预处理、密钥生成、加密或解密。', '',
             '`metrics.json` 和 `frontiers.json` 为可读索引。压缩包包含完整预算证据、所选 MLIR、Go 程序、原始输入和逐槽解密输出，以及复现命令与哈希。',
             f'证据包 SHA-256：`{sha}`。','']
    (out/'results.md').write_text('\n'.join(rows),encoding='utf-8')
    print(f'Verified Pass 2 report: {out}/results.md')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path);ap.add_argument('out',type=Path);ap.add_argument('--ctest',required=True,type=Path)
    args=ap.parse_args();collect(args.run,args.out,args.ctest)
