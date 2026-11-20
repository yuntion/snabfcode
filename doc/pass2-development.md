# 开发报告：原生 Pass 2、预算反馈与 CKKS 密文执行

2026-11-19。工程根目录：`/home/fhecompiler`。Pass 2 接入现有 `snabf` 方言、C++ `ModuleOp` Pass、MLIR `DataFlowSolver` 和 Pass 1 的 HiGHS 全图选路。没有引入旁路求解器或第二个推理运行时。结果见[逐模型、逐方案验收表](pass2-results/results.md)，含超限槽数、MSE、实测层级及在线耗时。

**本轮验收结论。** 四个主模型中自由规划的 8 个 SN-ABF 实验臂共 512 次真实加密推理，25,088 个逻辑输出均通过 `2^-12`；最大误差 `2.03173e-4`，按输出数加权的 MSE `1.10358e-9`。残差网络的前沿有 10 个完整 DAG。只收紧 sink 的确定性逼近预算，选中方案从 `silu_direct.d3.ps` 切到 `silu_direct.d7.ps`，第二层输入从 L2 变为 L1，旋转 `7,9` 从直接执行改为 `6+1,8+1`。实测 Galois 密钥从 4 把、41,946,496 字节变为 3 把、31,459,872 字节；在线均值从 114.3 ms 变为 145.1 ms。这里改善的是逼近证书和密钥开销，不是延迟或单次解密精度。

额外的较长链上，自由选择仍选直接模板，指定两阶段模板的 `(3,3)`、`(3,7)` 也均通过密文检验。唯一不达门限的是独立比较对象 `residual/heir`：4,096 个输出中 1,098 个超限，最大误差 `1.35715e-3`。对照结果和相同门限都已保留，没有算作通过。C++ 四项 CTest、四模型的 IR 固定点与篡改拒绝回归，以及原 Pass 1 的 IR 回读回归均通过。

## 论文原文与实现契约

核对了同级论文目录 `sn_abf_paper/main.tex` 实际编入的 `body/methodology.tex`、`body/appendix.tex`，未修改论文。III-C 给出奇 Chebyshev 候选次数 `{3,5,7,9,13,15}`、Remez 残差、PS 代价、Bellman 回溯单链，以及 canonical bias、已验证 DAG 随机契约的预算更新。III-F 描述向公共规划器提交替代 DAG、沿 SSA 反馈并在发射前复核。**双模板、`hull(I,A)`、分层代价 Pareto 前沿及显式重构，是本次任务对这两节的具体化；不是把当前 III-C 原文说成已经列出这些细节。**

`silu_direct` 逼近 `h(x)=sigmoid(x)-1/2`。`silu_tanh_rational` 先逼近 `tanh(x/4)` 再逼近 `u/(1+u²)`；两者参考误差均为零。该恒等式只用于定义参考函数，密文里没有除法。两条路径均发射 `x/2 + x·core`。输入域为 `|x|≤B` 时，**对激活预算筛选的是 `B·(ετ+e_k)`，不是 `e_k`**。GeLU 有直接奇核的数学模板及认证测试；这次源图识别和密文验收针对 SiLU。

## 实现和验收入口

|任务|源码位置|可以实际检查的内容|
|---|---|---|
|P2-1 模板和认证|`src/compiler/pass2/PolyTemplates.*`, `IntervalCertifier.*`|在参考域做 Remez；把系数转换为后端实际使用的 binary64 单项式系数后独立认证。MPFI/MPFR 有向舍入和分区间包络覆盖连续域，不以拟合点最大残差充当证书。|
|P2-2 DP 和 DAG|`src/compiler/pass2/CompositePolyDP.*`|同时维护理想轨迹 `I`、实际轨迹 `A`，在 `I` 上认证残差、在 `hull(I,A)` 上认证导数界，递推 `e'=L·e+ε`。保留可行前缀；只对完整 DAG 作非支配比较。|
|P2-3 原生 IR 和规划|`src/compiler/ir/{SNABFOps.td,Passes.td,ActivationRegions.cpp,PolyRecords.cpp,Planning.cpp,TransferRecords.cpp}`|先在 `linalg.generic` 中识别实际 SiLU 运算，再沿 HEIR 的 SSA 核实两分支同源、只剥离恒等掩码和旧层级对齐。`poly_eval` 的 ODS 属性携带前沿及选择。|
|P2-4 后端|`src/compiler/ir/{Lowering.cpp,Pass2Lowering.cpp}`、`src/compiler/pass1/goemit/`|完整拓扑节点生成 HEIR 密文操作；按实际 Q 素数解物理 scale 对齐，绑定预处理表和系数编码层级，生成 Go 并调用真正的 Lattigo 加密、推理、解密。|

1. **次数不是层数。** 记录论文的 PS 估算及实际节点、平方、密文乘法、重线性化、标量乘法、重缩放和对齐。比如直接 3 次的完整实现包括标量摊入与重构，实耗 4 层。所有候选包含逐阶段系数、`I/A/J`、局部 `ε/L`、递推值、完整 DAG、增益、输出区间和各层原语清单。
2. **候选不是单链。** 比较深度、误差、增益、输出包络及每一层的原语工作；不同层上的乘法不能仅按总数抵消。`Pass2DPTest` 在一个开放深度预算上用独立穷举对比完整候选 ID 集合，还检查改变层级价格能翻转实际代价次序。只有深度/误差/资源契约允许才剪枝。
3. **约束真的传到下游。** `snabf-plan` 把候选实际层数写到 `!snabf.ct`；`DataFlowSolver` 传播预算，`lintrans` 保存 `input_budget`。若 sink 的误差或层级不够，沿 SSA 上溯求必要约束，记录 `backward_ssa`；重新比较实际层级对应的 Pass 1 子群塔和全图密钥并集。相同层级向量可复用选路结果。`snabf-check-plan` 从源域、权重和 profile 重建前沿、选中 DAG、预算及密钥库存；老计划不能绕过 checker 发射。
4. **层级不只是元数据。** 在残差汇合处，对仍处于高层的那一次 SSA 使用显式发射 `ckks.level_reduce`。计划从完整 profile 指定的输入层级加密，后端同步更新输入编码、公钥环、实际客户端和求值 ABI；第二层的准备表按选定输入层重新构造。

## 随机预算接口

`src/compiler/pass2/NoiseTransfer.*` 对完整 DAG 列出 source、drop、pt_mul、mul、relin、rescale、add 等实际站点；`TransferRecords.cpp` 把它们序列化进每个候选。确定性逼近残差在输出加一次，按真实输出 scale 转换至 canonical 单位。相同 `x` 在多分支复用时，不假定独立：在共同有效事件上组合中心化分量的标准差代理，用复数误差绝对包络约束乘积余项。`Pass2NoiseTest` 含共享输入、误差单位、条件不匹配和缺失契约的独立用例。

可通过 profile 的 `validated_poly_noise` 为特定 site、候选、Q/P、key、输入层级和 DAG 提供已验证的逐原语增量契约。外部须给出共同有效事件、输入域、各站点的平均偏差界、分量方差代理、绝对误差包络和单位。任一前提缺失时 IR 保留完整符号转移，不填零。**本次 HEIR profile 没有这些实参。** 报告中的 `activation_error` 是实多项式的逼近证书；`approx_error` 是传播至该边的确定性逼近分量，实测解密误差是独立观测，都不是 Theorem 1 的全系统失败概率证明。

## 复现与读数

```bash
cd /home/fhecompiler
bash src/tools/accept_pass2.sh
```

该命令从 CMake 构建及 CTest 开始，创建新的 `build/pass2/accept.*/`，经过源图标记、HEIR 变换、候选生成、规划、复核、CKKS lowering、后端绑定、Go 编译和真密文验证，最后收集报告。独立基线仍使用 `src/tools/run_unified.sh`。必需环境为已固定的 HEIR、MLIR 24、HiGHS、Go/Lattigo 和 MPFI/MPFR/GMP 开发库。

固定每个实验臂 64 个输入，包含角点、结构化符号向量、脉冲和固定种子随机输入。密钥与加密随机性不固定。使用源矩阵和真实 SiLU 的 float64 输出逐槽比较，门限 `2^-12`；另存明文选中 DAG 的误差以分离逼近和密文求值误差。常规实验覆盖 `control`、`strided64`、`reuse64` 和带真实密文加法的残差网络；另用长链和显式模板限制覆盖两阶段，单独报告。输出误差预算从公开证书设置，不用解密结果挑选实现。

阅读顺序：[结果表](pass2-results/results.md) → [机器可读汇总](pass2-results/metrics.json) → [完整前沿索引](pass2-results/frontiers.json) → [残差宽预算的完整 IR](pass2-results/residual-economy.mlir) 与 [严预算 IR](pass2-results/residual-precision.mlir)。压缩包中同时有各方案的 `audit.json`、`selected.mlir`、`model.go`、输入语料及逐槽解密输出。Pass 3 可以直接读取这些预算转移做 Q/scale/gadget 参数选择，本报告没有把固定 profile 上的验证写成四个 Pass 全部完成。
