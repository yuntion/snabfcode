"""Archive a successful native Pass 1 run (fresh destination, no rebuild)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive(run, out):
    evidence = json.loads((run/'evidence.json').read_text())
    rows, p2, p3 = evidence['pass1'], evidence['pass2_residual'], evidence['pass3']
    boot = evidence['periodic_bootstrap']; bm = boot['metrics']; obs = bm['observations']
    if not all((len(rows)==10,len(p3)==2,len(obs)==8,len(p2)==2)):
        raise ValueError('incomplete experiment')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(run/'evidence.json',out/'evidence.json')
    shutil.copyfile(run/'run.log',out/'run.log')
    files=sorted(f for f in run.rglob('*') if f.is_file() and f.suffix in {'.json','.jsonl','.mlir','.go','.txt','.log'} and not f.is_relative_to(out))
    with tarfile.open(out/'raw-evidence.tar.gz','w:gz',compresslevel=6) as tar:
        for f in files:tar.add(f,arcname=f.relative_to(run),recursive=False)
    src=sorted(f for name in ('src','test') for f in (ROOT/name).rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    (out/'source-sha256.json').write_text(json.dumps({f.relative_to(ROOT).as_posix():digest(f) for f in src+[ROOT/'go.mod',ROOT/'go.sum']},indent=2)+'\n')
    report = [f'''# Pass 1 原生实现与密文验收

本次服务器运行：`{run}`。所列数字取自本次执行，不合并旧报告的试次。论文依据：`body/methodology.tex`，III-B（式 2、3）和 III-F。

## 1. 开发交付

| 任务 | 实现 | 验收证据 |
|---|---|---|
| P1-1 子群塔与密钥共享 | `SubgroupTower` 独立生成所有指数为 2 的子群、非空陪集、提升后的自同构与密钥。`KeyReuseModel` 汇总固定方案，`RouteSelection` 负责 HiGHS。 | 穷举 Z₈ 的 256 种支持集（包括空、恒等、负偏移与重复偏移）；独立枚举验证物理并集最优值。 |
| P1-2 环切与自举 | `RingSwitch` 负责式 (3)；`NativeBootstrap` 绑定输入 SSA 的布局周期、活跃层级、占用和下游预算。 | 正收益、平局、负收益、方差、层级、布局分别测试；另有真实小环自举执行。 |
| P1-3 预算反馈 | `snabf-plan` 按当前层级定价并产生带 SSA 路径的逐轮记录。`snabf-check-plan` 在模块副本上重放，不依赖再跑一次 HiGHS。 | 输入层级、具体 Q/P 或布局更改触发重新规划；旧费用、旧塔层、伪造反馈记录、失效绑定均被拒绝。 |
| P1-4 后端与归档 | `snabf-lower-ckks`、`snabf-bind-lattigo`、Go 绑定与 Lattigo 按选定的分解执行。 | 对照真实浮点 SiLU，检查密钥 ID、序列化字节数、层级、scale，另跑残差网络与具体参数联合优化。 |

`src/compiler/pass1` 的 C++ 决策核由 CTest 直接测试；原生 `snabf-opt` 处理 HEIR/MLIR 的 SSA，Python 只驱动前端与黑盒回归。`src/backend/ringswitch` 执行环切；本套对照使用共享的 `src/compiler/Target/Lattigo/go-bind` 绑定已核验的决策，不选路。

## 2. 式 (2)：数学切分、实际工作量和物理密钥

计划保留 `equivalent_splits`（每项为 `s, stride, b_v, g_v`）、非空 `cosets`、内外旋转、每条对角线的分解和顺序路径深度。默认 `occupied` 按实际执行的非零旋转计费，跳过空陪集；`dense_tower` 单独实现正文的 `(b_v−1, g_v−1)` 整塔定价。只在前者合并指令完全相同的相邻塔层，各数学切分仍可恢复。

`reuse64` 在物理 Z₈₁₉₂ 中选路，编码 4096 槽，逻辑宽度 64。全图方案第一层 `7=6+1, 9=8+1`，与第二层复用旋转 6、8。第一层代表切分 `(4096,2)`，不是执行 4095 次 baby rotation：实际只有 baby `6,8` 和 giant `1`。两层分摊的存储分别为 20,973,248 B 和 10,486,624 B，总计 31,459,872 B。逐层方案独立选路，再按同样的物理并集统计。每次执行的 key-switch 仍计入运算费用，存储按 `(key ID, N, Q, P, Galois 元素)` 去重。

|工作负载|方案|试次/逻辑输出|密钥数|实际序列化 B|最大绝对误差|MSE|在线均值 ms|
|---|---|---:|---:|---:|---:|---:|---:|''']
    for r in rows:
        m=r['metrics'];report.append(f"|{r['workload']}|{r['arm']}|{m['Trials']}/{m['Outputs']}|{len(m['Galois'])}|{m['SerializedGaloisBytes']:,}|{m['Linf']:.5g}|{m['MSE']:.4g}|{m['MeanEvalMs']:.1f}|")
    report.append(f"\n三条固定参数主图共 {sum(r['metrics']['Trials'] for r in rows)} 个被统计的方案试次、{sum(r['metrics']['Outputs'] for r in rows):,} 个逻辑输出（含 cap2 的 4 次/256 个输出；其 HEIR 配对参照不另计）。全部通过 2⁻¹²，运行时 L7→L0，输出 log₂(scale) 为 {rows[0]['metrics']['OutputLogScale']:.9f}；每次亦核对 stock HEIR。在线时间不含密钥生成、预处理、加解密。密钥字节取 `MarshalBinary()`，不是峰值内存。")
    report.append('''
## 3. 反馈确实改变了程序

* `reuse64`：不加路径上限时 1 轮、3 把密钥；sink 上限 2 时，原路径超限，重放的第二轮改为直接旋转，4 把密钥；上限 1 不可行。重复规划相同，过期计划不能 lowering。
* `residual`：输入 MLIR、权重和环参数不变，仅将 sink 确定性逼近上限从 2⁻¹² 收紧到 3.4614778033368946e−9。下表是本轮密文，不是回填旧测量。

|策略|SiLU 候选|第二层输入|Galois 数/序列化 B|超限/槽数|最大绝对误差|MSE|
|---|---|---:|---:|---:|---:|---:|''')
    for arm,s in p2.items():
        a=s['accuracy'];report.append(f"|{arm}|`{s['candidate']}`|L{s['level']}|{len(s['keys'])}/{sum(k['serialized_bytes'] for k in s['keys']):,}|{a['slots_over_tolerance']}/{a['logical_slots_checked']}|{a['linf']:.5g}|{a['mse']:.4g}|")
    report.append('''
* Pass 3：在两个工作负载分别重新特化 HEIR，并用实际 Q/P 重建 Pass 1 的物理密钥作用域；运行 Sage 的固定版本 estimator 与实际参数回查。`matched_exact` 保持与 `joint` 相同的具体参数，是同密文配对对照。此处不把 estimator 的具体筛查表述为全系统失败概率证明。

|工作负载|参数/截断策略|实际输出层级 / log₂(scale)|超限/槽数|最大绝对误差|具体参数下的计划 Galois B|
|---|---|---:|---:|---:|---:|''')
    for work,arms in p3.items():
        for arm,entry in arms.items():
            m=entry['metrics'];report.append(f"|{work}|{arm}|L{m['OutputLevel']} / {m['ObservedOutputLogScale']:.3f}|{m['Misses']}/{m['Slots']}|{m['Linf']:.5g}|{entry['key_bytes']:,}|")
    ms=[statistics.mean(row['SegmentNS'][i] for row in obs)/1e6 for i in range(3)]
    report.append(f'''
## 4. 式 (3)：编译判定与真实自举

原 `16384→8192` 夹具只做编译判定，收益 40、罚金 35、净收益 5 **均为归一化回归费用，不是毫秒**。平局和负收益分别给 `tie`、`negative_gain`；只给槽数而缺少保值的周期布局不能放行。输入方差与额外切换方差分开，与下游容量/层级分别核对。

新增独立的周期布局 HEIR bootstrap 工作负载，不改变三条 SiLU 主图。它经过 `snabf-pass1-candidates → snabf-plan → snabf-check-plan → snabf-lower-ckks → snabf-bind-lattigo`；Go 绑定保留 HEIR 生成的 bootstrap 调用和配置函数，用检查过的 `periodic-n15-v1` evaluator 执行。降环和升环各用真实 evaluation key，中间调用 Lattigo CKKS bootstrap，而不是解密重加密。

| 项目 | 实测 |
|---|---|
| 四处物理环维 | {' → '.join(map(str,obs[0]['Rings']))} |
| 活跃 RNS 层级 | {' → '.join('L'+str(x) for x in obs[0]['Levels'])} |
| 四处 log₂(scale) | {', '.join(map(str,obs[0]['LogScales']))} |
| 校验范围 | {bm['trials']} 次加密、{bm['slots']} 个解密槽，L∞ = {bm['linf']:.5g}，MSE = {bm['mse']:.5g} |
| 分段在线均值 | 降环（含输入对齐）{ms[0]:.1f} ms；小环自举 {ms[1]:.1f} ms；升环 {ms[2]:.1f} ms |

支持的是单密文周期布局、显式的 N15 自举参数族。全占用多密文 `split-repack` 有成本与合法性判定，尚不发射；Pass 4 的布局反馈是接口测试，不是完整动态降环 Pass。没有同参数大环自举的在线时间对照，以上分段时间不构成加速比。

## 5. 复现与归档

```bash
cd /home/fhecompiler
bash src/tools/run_pass1.sh
# 将打印的新运行目录传给归档工具，目标目录必须尚不存在：
python3 src/tools/archive_pass1.py /home/fhecompiler/build/pass1-revision/<run> /home/fhecompiler/doc/<new-report>
```

[原始运行日志](run.log) · [机器可读指标](evidence.json) · [原始 IR、生成程序、逐次记录与安全回查](raw-evidence.tar.gz) · [源文件摘要](source-sha256.json)。原始包包含 {len(files)} 个文件，SHA-256 为 `{digest(out/'raw-evidence.tar.gz')}`。保留旧 Pass 1/2/3 实验目录不动。''')
    (out/'results.md').write_text('\n'.join(report)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();archive(args.run.resolve(),args.out.resolve())
