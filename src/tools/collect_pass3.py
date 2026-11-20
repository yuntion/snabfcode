"""Publish a checked Pass 3 run, including raw trials, generated programs and IR."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile


def collect(run, dest, root):
    data = json.loads((run / 'summary.json').read_text())
    dest.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run / 'summary.json', dest / 'summary.json')
    shutil.copy2(run / 'run.log', dest / 'run.log')
    prime = subprocess.check_output([root / 'build/pass3/native/Pass3Test'], text=True).strip()
    (dest / 'prime-cut.log').write_text(prime + '\n')
    files = [run / p for p in ('commands.jsonl', 'manifest.json', 'summary.json', 'run.log')]
    files.extend(sorted(run.glob('budget-*.json')))
    lines = ['# Pass 3 密文实验验收', '', f'服务器验收目录：`{run}`。', '',
             '两张图、三组方案、每组 64 个输入。每个输入检查 64 个逻辑槽。'
             '`no_trunc` 独立优化但禁止截断；`matched_exact` 固定联合方案 Q/P、密钥、输入密文与选中 DAG，只关闭截断。在线时间不含客户端加解密及预处理。', '',
             '| 模型 | 方案 | log₂(QP) | log₂Δ | 输出层 | 超限/槽 | L∞ | MSE | 在线均值 ms |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    detail=[]
    for w, arms in data['workloads'].items():
        files.extend(run / w / p for p in ('corpus.json', 'input.mlir'))
        for name, s in arms.items():
            d = run / w / name
            files.extend(d / p for p in ('plan.json', 'security.json', 'profile.json', 'checked.mlir',
                                          'routes.json', 'audit.json', 'model.go', 'metrics.json', 'trials.jsonl'))
            files.extend(sorted(d.glob('specialization*.json')))
            feedback=d/'security-feedback.json'
            if feedback.exists():files.append(feedback)
            lines.append(f"| {w} | `{name}` | {s['bits_qp']:.3f} | {s['scale']} | {s['OutputLevel']} | "
                         f"{s['Misses']}/{s['Slots']} | {s['Linf']:.4e} | {s['MSE']:.4e} | {s['OnlineMS']:.2f} |")
        s=arms['joint']; delta=s['bits_qp']-arms['no_trunc']['bits_qp']
        detail.append(f"- **{w}**：每次推理实际少做 {s['skipped_rows']} 行 gadget 外积；分解部分净少 {s['net_decomposition_ntt_saved']} 次单模数 NTT/INTT。"
                      f"命中 `{', '.join(e['ID'] for e in s['Events'])}`。同密文 `joint − matched_exact` 的最大槽差 {s['PairedLinf']:.4e}，"
                      f"经验方差 {s['paired_empirical_variance']:.4e}。相对独立无截断方案，QP 位宽变化 {delta:+.4f} bit；"
                      f"具体 QP 最低经典估计 {s['min_qp_bits_security']:.2f} bit。")
    lines.extend(['','## 工作量与截断误差','',*detail,'','## 规划器回归',''])
    for row in data['compile_only_budget_sweep']:
        lines.append(f"- 仅编译的同图预算扫描：ε={row['epsilon']}，P 目标 {row['p_bits']:.3f} bit，k=1 的站点为 `{', '.join(row['sites'])}`。")
    lines.extend(['- 独立修改计划的截断档位和物理 scale；校验与 lowering 各拒绝两种修改，共 4 项负例。',
                  f'- 真实 NTT 素数回切单测：`{prime}`。', '',
                  '## 复现材料', '', '[完整日志](run.log) · [机器可读结果](summary.json) · [证据包](evidence.tar.gz) · [SHA-256 清单](sha256.json)。', '',
                  '证据包包含同一运行的源模型、明文参考、每次推理输出、各方案的具象计划、已检查 MLIR、生成的 Go、估计器输入与攻击成本。'
                  '每个 `metrics.json` 的 `Events` 是一遍独立、不计时的诊断推理；外积和 NTT 数是每次推理的数，不乘以槽数。'
                  '模数容量与时延没有预设改善方向。数学归属与证书条件见[开发报告](../pass3-development.md)。'])
    (dest / 'results.md').write_text('\n'.join(lines)+'\n')
    with tarfile.open(dest/'evidence.tar.gz','w:gz') as tar:
        for p in files:tar.add(p, arcname=p.relative_to(run))
    hashes={str(f.relative_to(run)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    hashes['evidence.tar.gz']=hashlib.sha256((dest/'evidence.tar.gz').read_bytes()).hexdigest()
    (dest/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('dest',type=Path)
    a=ap.parse_args();collect(a.run.resolve(),a.dest.resolve(),Path(__file__).resolve().parents[2])
