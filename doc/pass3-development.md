# Pass 3 开发报告：从 SSA 预算到实际执行的近似 gadget

2026-11-19。对应论文 III-D、III-F 和附录的条件误差模型。详细数字、运行路径与原始证据见 [Pass 3 验收结果](pass3-results/results.md)。本轮没有改动论文正文或覆盖 Pass 1/2 的历史数据。

## 1. 编译入口与实验问题

原生 `ModuleOp` pass `snabf-pass3` 在 `snabf-plan` 后读取同一张 SSA 图。对每个 `lintrans`，按选中塔的实际 baby、giant、旋转掩码和分组加法展开；多个 baby 共用输入分解，不按矩阵对角线数计作多次 key-switch。对 `poly_eval`，使用已选完整 DAG 及 `transferRecord` 的 relin 站点，包含旁路对齐、显式 relin/rescale、系数编码和残差合并。未实现的加密操作直接报错。第一批验收对象是 `reuse64` 和 `residual` 两个真实 `Linear → SiLU → Linear` 源模型。

求解产生参数提案；如提案与 HEIR 原参数不同，工程前端使用同一源模型和具体素数重新执行 HEIR、Pass 1 和 Pass 2，再评估原生 Pass 3，直到图与物理参数一致。这个步骤会重新生成预处理表、密钥尺寸和客户端 ABI，不是替换导出 Go 里的一行常量。候选在 `snabf.pass3` 和 `snabf.profile` 内关联。`snabf-check-plan` 与 lowering 执行同一套独立于 HiGHS 的具象重放，包括参数、站点、物理尺度、预算和公开密钥估计输入；有提案但没有匹配的实际参数和安全回查，不能发射。Python 只负责调度原生 passes、调用 Sage 和整理证据。

## 2. 后端执行的截断是什么

本轮后端族固定为一个辅助素数 P、纯 RNS 分解、`BaseTwoDecomposition=0`。在这个族里一个 Q 肢对应一整行 gadget。设 `D=∏_{j<k}q_j`。在系数域做 `r=center_D(c)`、`c'=c-r`。D 为奇数，中心代表唯一落在 `[-(D-1)/2,(D-1)/2]`；保留行也必须减去**同一个** r。于是 `c' mod q_j=0` 对所有 `j<k` 成立，后端不为这些行做基扩展、前向 NTT 和 gadget 外积。保留行仍在完整 QP 上运算，最后走 Lattigo 原生 ModDown。多肢路径用前缀乘积做一次 CRT；有正半模邻点的回归，避免错一个模数。`k=0` 沿用原生运算，已测试原生与封装 relin 的密文相等。

旋转中被修正的分量为 `c1`，遗漏信号为 `r·s` 再经自同构；重线性化中为 `c2`，遗漏信号为 `r·s²`。这解释了两者的截断容限为何不同。该构造是本实现的中心余数原型，不把普通 CRT 残基解释成位置进制的低位数字。实际执行记录给出 `RowsSkipped`、`RowsExecuted`、执行的前/逆 NTT、中心余数最大绝对值、一二阶矩和经验方差；原始密文与相同参数、相同密钥、相同输入密文的完整分解作第二种、独立的槽域差分观测。NTT 净差只统计分解变换，不含两臂公共的 ModDown、掩码乘或整段推理。

## 3. HiGHS、算术模型与具体素数

`src/compiler/pass3/` 是纯 C++ 实现。外层枚举后端可编码的整数默认 scale 指数，每个指数对应一个 HiGHS MILP：联合选择连续 Q/P 位宽、边对数尺度和各站点的 one-hot 截断档。约束包括乘法尺度和、rescale 减去实际被除素数的对数、旁路汇合、可调系数编码、各边模数容量、QP 位宽上界和 sink 误差。目标是有权的 gadget 行工作与位宽代价，不是预测毫秒数。固定路由以后密钥并集不因单纯跳过一行就减少，模型没有把跳过运算算成密钥字节节省。

对前缀乘积按 `sum(log2(q_j))` 计量。对指数和平方指数使用归一化分段割线**上界**；引入有界连续变量并用四条 McCormick 约束精确表示二元档位乘包络。上界只进入单调的噪声预算，因而凸分段线性 epigraph 不需要额外 SOS2 二元变量。这不等于对指数本身精确线性化。论文 III-D 中以逐肢方差和表达的一般形式，在这里替换成与当前 `center_D` 核相符的前缀残差界。模数容量用对数不等式直接检查，没有假设普适的 `q_l/2` 残差抵消阈值。

分量幅度满足 `|r_i|≤(D−1)/2`。在三元秘密的共同事件 `max_z |s(z)|≤S` 下，取 `S=min(N,2 sqrt(N ln(4N·2^40)))`；这是对各根实虚部应用 Hoeffding 并作并集界。于是槽的 canonical 截断扰动由 `N(D−1)S/2`（rotation）或 `N(D−1)S²/2`（relin）控制。条件绝对误差逐原语传播；乘法包含两条输入及误差乘积，汇合不假设独立。对任意条件绝对界 A，可记录均值幅度上界 A 和中心化 Hoeffding proxy `A²`。此处不是 `D²/12` 的独立均匀噪声假设。选择时直接使用绝对误差包络，**不把实测方差填入证书，也不把这份记录写成全系统 Theorem 1 的无条件失败概率**。量化/编码采用每次编码 canonical 误差不超过 N 的前提；本轮没有单独证明 Lattigo 浮点 FFT 对任意输入都满足此前提。Pass 2 真激活的确定性逼近误差另计。

实例化使用确定性 64 位 Miller–Rabin，从 `2^b` 附近依距离搜索 `1 mod 2N` 的全局互异素数。回查从实际 Q/P 重新计算每条物理尺度、中心余数幅度、普通切换密钥噪声项和输出容量，独立验证分支汇合。精确位宽越界后，重新约束相应前缀或总 QP，并重解。附带测试以两个实际相邻 NTT 素数构造目标，稳定触发“连续可行 → 近邻素数不够 → 位宽收紧 → 新素数通过”。

安全估计器固定 `ae45d740ceb138af87a4f92edc592a0ab0d4f43f`，经典 ADPS16/GSA uSVP 与 dual-hybrid、均匀三元秘密、σ=3.2、高斯估计模型、无界样本。分别检查包络、实际 Q 及公钥、Galois 和 relin 所用扩展 QP。估计器拒绝会保留原因并向编译器提交更紧的总位宽上界。这是指定攻击模型下的经典具体安全筛查，不是量子评估或对循环密钥假设的证明。

## 4. 代码与复现

| 入口 | 用途 |
|---|---|
| `src/backend/gadget/centered.go`、`src/backend/tower/lintrans.go` | 系数域零化、原生/近似 relin 和旋转、hoisting 与工作量记录 |
| `src/compiler/pass3/{GadgetNoiseContract,ParameterMILP,PrimeInstantiator,ConcreteVerifier}.*` | 算术合同、联合求解、真素数、独立回查 |
| `src/compiler/ir/Pass3IR.cpp`、`Passes.td`、`Lowering.cpp` | 原生 pass、SSA 图与已选候选提取、检查和 Lattigo 绑定 |
| `src/frontend/snabf_frontend/pass3.py` | 复用 HEIR 前端，用具体参数重新编译至稳定 |
| `test/compiler/Pass3Test.cpp`、`test/pass3/harness_test.go` | 真素数回切；配对密文、逐槽明文对照与在线计时 |

服务器：

```bash
cd /home/fhecompiler
LATTICE_ESTIMATOR=/home/zc_batch/build/deps/lattice-estimator SNABF_P3_INITIAL_CAP=390 bash src/tools/run_pass3.sh
```

每次生成独立的 `build/pass3/run.*`。常规运行不设置 `SNABF_P3_INITIAL_CAP` 时从 350 bit 开始；390 用于复现安全拒绝与重解。报告中的统计只取指定的验收运行，不合并调试批次。当前验证了 Pass 3 和上游 Pass 1/2 的联合链路；未将尚未纳入本实验的 Pass 4 降环作为结果。
