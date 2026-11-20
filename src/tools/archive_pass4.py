"""Package a finished native Pass 4 experiment; never substitute older metrics."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def archive(run,out):
    load=lambda f: json.loads(f.read_text())
    rows=[]; decisions={}; rejected=0
    for network in ('reuse64','residual'):
        d=run/network; rejected+=len(load(d/'negative/results.json'))
        for arm in ('baseline','adaptive','expensive'):
            a=d/arm; m=load(a/'metrics.json'); s=load(a/'search.json');p=load(a/'program.json')
            if m['misses'] or m['trials']!=64 or m['actual_output_ring']!=s['target'] or (arm=='adaptive')!=(s['target']<p['source_ring']):raise ValueError(f'incomplete arm {a}')
            with (a/'trials.jsonl').open() as f:
                if sum(1 for _ in f)!=m['trials']:raise ValueError(f'missing trials {a}')
            row=dict(network=network,arm=arm,**m,predicted_work=s['selected_cost'],baseline_work=s['baseline_cost'],
                     galois_bytes=sum(k['Bytes'] for k in m['keys'] if k['Role']=='galois'),switch_bytes=sum(k['Bytes'] for k in m['keys'] if k['Role']=='down'))
            rows.append(row)
            if arm=='adaptive':decisions[network]=s['conversion']
    out.mkdir(parents=True,exist_ok=False)
    report=['# Pass 4 开发与验收记录','',
        '原生 `snabf-pass4` 在独立复核过的 Pass 3 输入上比较维持源环与末层 `lintrans` 前的降环。两组网络使用同一冻结语料；`residual` 的旁路在转换之前完成密文汇合。每个方案均重新生成并执行 Lattigo 程序。','',
        '|网络|策略|实际输出 N / L|log₂(scale)|最大绝对误差|MSE|平均在线 ms|Galois 字节|转换密钥字节|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:report.append(f"|{r['network']}|{r['arm']}|{r['actual_output_ring']} / {r['actual_level']}|{r['actual_log_scale']:.6f}|{r['linf']:.4g}|{r['mse']:.4g}|{r['online_ms']:.2f}|{r['galois_bytes']:,}|{r['switch_bytes']:,}|")
    report+=['','## 选型证据','']
    for n,c in decisions.items():
        report.append(f"* `{n}`：转换前后 N={c['provided']['ring_dim']}→{c['required']['ring_dim']}，活跃 L{c['switch_level_q']}，共有 {c['cut_edges']} 条活跃密文边；转换不消耗 Q 层。预测工作量 {c['baseline_cost']:.0f}→{c['cost']:.0f}，下降 {100*(1-c['cost']/c['baseline_cost']):.2f}%。转换密钥实际使用 Q[0:{c['switch_level_q']+1}] 与 P[0:{c['switch_level_p']+1}]。固定模型下，源端最低 {c['source_min_bits']:.2f} bit，目标端与转换密钥最低 {c['target_and_switch_min_bits']:.2f} bit。")
    report+=['',f"本轮共 {sum(r['trials'] for r in rows)} 次加密推理，{sum(r['logical_outputs'] for r in rows):,} 个逻辑输出；计入周期副本共 {sum(r['physical_encoded_slots_checked'] for r in rows):,} 个实际解码槽，均通过 2⁻¹²。密文每条指令均核对实际环维、level、scale 与编码槽数。{rejected} 个篡改/缺凭证的校验或发射尝试被拒绝。相同策略复算、从混合环改成昂贵转换策略回切均经回归。",'',
        '## 复现与解释','','在工程根目录执行 `bash src/tools/run_pass4.sh`。公开依赖为锁定的 MLIR/HEIR、HiGHS、Lattigo、Go 和 Sage lattice-estimator。旧归档仅提供输入图、参数和 64 组独立 float64 参考；新运行独占 build/pass4/run.*。', '',
        '* 顺序为布局与后端能力预检、Pass 2 候选和 Pass 1 密钥重新定价、固定已加密输入 Q/P 和物理 scale 的 Pass 3 边界 MILP 与具体参数回查、外部估计器复核。缩小环维不会产生与活跃密文不一致的新素数。',
        '* `baseline` 禁止提案，`adaptive` 自由比较，`expensive` 仅提高转换操作单价；目标值为带环维和活跃 RNS 层权重的工作量，不是毫秒。在线均值包含密文检查与转换，不含编码、密钥生成、预处理、加解密和磁盘记录。序列化字节来自真实密钥的 `MarshalBinary()`，不是峰值内存。',
        '* 当前可发射 ABI 是单密文周期布局、末层前降环、目标密钥输出。全占用独立多路复用、split/repack 及一般升环均不由该目标发射；不等于否定 Pass 1 独立的自举专用升环。安全数值是指定经典攻击模型的估计，不是量子安全或转换密钥依赖的证明；误差转移为条件界，不把一次数值回归当作全系统失败概率。', '',
        '完整证据包含每个方案的 `planned.mlir`、`lowered.mlir`、`model.go`、`program.json`、`search.json`、逐槽 `trials.jsonl` 和 `metrics.json`，以及参数实例、安全攻击结果、拒绝日志和源文件 SHA-256。']
    (out/'results.md').write_text('\n'.join(report)+'\n');(out/'summary.json').write_text(json.dumps(dict(rows=rows,decisions=decisions,rejected_checks=rejected),indent=2)+'\n')
    shutil.copyfile(run/'run.log',out/'run.log');files=sorted(p for p in run.rglob('*') if p.is_file() and not p.is_relative_to(out) and p.suffix in {'.go','.json','.jsonl','.mlir','.log','.sha256'} and not (p.parent.name=='negative' and p.suffix=='.mlir'))
    with tarfile.open(out/'evidence.tar.gz','w:gz',compresslevel=5) as z:
        for p in files:z.add(p,arcname=p.relative_to(run),recursive=False)
    hashes={p.relative_to(run).as_posix():digest(p) for p in files};(out/'sha256.json').write_text(json.dumps(dict(members=hashes,archive=digest(out/'evidence.tar.gz')),indent=2)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('output',type=Path);x=ap.parse_args();archive(x.run.resolve(),x.output.resolve())
