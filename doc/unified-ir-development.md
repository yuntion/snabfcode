# 开发报告：统一 SN-ABF IR、C++ Pass 与 Pass 1 迁移验收

2026-11-19，`/home/fhecompiler`。**Pass 1 正式入口现在是原生 MLIR Pass**。输入仍是原来的 secret/linalg 模型，选择发生在 HEIR 的 CKKS 语义层、进入 Lattigo 之前。`plan.json` 是选中 IR 的实验导出，不再作为求解器的输入。原来已验收的脚本和证据保留在独立的 `build/pass1`、`doc/pass1-results`。

## 方言在哪里

| 想看的内容 | 文件 |
|---|---|
| 注册名 `snabf`，参数化 `!snabf.ct`、`#snabf.budget` | [方言](../src/compiler/ir/SNABFDialect.td)、[类型和属性](../src/compiler/ir/SNABFTypes.td) |
| `profile` 符号、`lintrans`、`poly_eval`、`convert`、输入输出和 CKKS 运算 | [ODS 算子](../src/compiler/ir/SNABFOps.td)、[注册与校验](../src/compiler/ir/IR.cpp) |
| 哪个 Pass 作用在哪一级、有哪些选项、要装载哪些方言 | [Passes.td](../src/compiler/ir/Passes.td)、[C++ 实现和注册](../src/compiler/ir/Passes.cpp)、[原生驱动](../src/compiler/ir/Driver.cpp) |
| 论文公式 (2)、(3) 及其消费者 | [子群塔候选](../src/compiler/pass1/NativeRouting.cpp)、[环切自举判定](../src/compiler/pass1/RingSwitch.cpp)、[共享规划与预算传播](../src/compiler/ir/Planning.cpp) |
| 实际运行过的 IR | [reuse64 全图](unified-ir-results/reuse64-joint.mlir)、[同一输入收紧输出预算后](unified-ir-results/reuse64-cap2.mlir)、[独立 bootstrap 判定](unified-ir-results/boot-selected.mlir) |

这两个 `reuse64` 文件都是完整可重读的 MLIR，不是节选。查找 `"snabf.lintrans"` 可看到两个实际线性算子；查找 `"snabf.poly_eval"` 和 `kind = "ckks.mul"` 可看到多项式支路及它与输入旁路汇合的位置。`!snabf.ct<tensor<64xf32>, 16384, 7, @e01>` 表示 64 个逻辑值、环维 16384、零起始层号 7（8 个活跃 Q 素数）。`@e01` 指向真实 Q/P、默认 scale、密钥族和 HEIR 槽位映射。

## 为什么选 ModuleOp

共同基类由 TableGen 的 `Pass<..., "::mlir::ModuleOp">` 生成。求解要同时看到预处理函数、模型求值函数、profile 符号和全模块物理密钥。若按每个线性算子各跑一次局部 Pass，就无法让第二层影响第一层的路由。各 C++ 实现继承生成的 `impl::*Base`，实现 `runOnOperation()`；失败通过 MLIR 诊断和 `signalPassFailure()` 返回。

| 原生 Pass | 实际工作 |
|---|---|
| `snabf-import-heir` | 从真实 CKKS use-def 和预处理表的 `store/load` 绑定掩码、对角线与应用；保留 SiLU 两条支路。 |
| `snabf-pass1-candidates` | 在 `lintrans` 上发布子群塔分解。若有独立的 HEIR bootstrap 算子，则使用该算子的标定属性，写入公式 (3) 的编译决策。 |
| `snabf-plan` | C++ 直接对 MLIR 节点和 profile 枚举、刷新候选；HiGHS 对选中程序的密钥并集只计一次。通过 MLIR `DataFlowSolver` 传播预算，用不满足输出要求的 SSA 路径约束反向触发重算。 |
| `snabf-check-plan` | 独立重新计算选中分解、密钥引用计数、总字节、目标及预算。检查不重新选路；过期计划不能直接发射。 |
| `snabf-lower-ckks` | 通过 C++ `IRRewriter` 还原 HEIR 原语，把选中的分解和权重内容摘要带过 bufferization。多项式由当前 `poly_eval` 的系数生成。 |
| `snabf-bind-lattigo` | 以权重摘要、对角线、层和槽数核对后端准备点，按选中并集生成真实 Galois key 操作；Go 发射器只绑定该分解。 |

**依赖和工具边界。** `src/compiler/ir/HeirDialect.cmake` 直接从固定的 `extern/heir` 源码编译 Lattigo 方言、类型接口和验证器，和 `snabf` 一起注册在 `snabf-opt`。绑定 Pass 显式声明 `lattigo`、`func`、`memref` 依赖。官方发布的 `heir-opt` 负责 secret/linalg→CKKS 和 CKKS→Lattigo，`heir-translate` 负责 Go 翻译；没有修改官方可执行文件，也没有将不同 LLVM 快照的插件加载进去。两进程间传完整 MLIR。独立驱动把未链接的 HEIR CKKS 方言按 generic MLIR 保留，在边界读取固定版本的自定义类型字段；图结构、权重、SSA use-def 和 SN-ABF 规划不是从文本正则重建的。

## 决策是否真的改变密文程序

| `reuse64` 决策 | 第 1 层如何实现 7、9 | 两层最坏路径的顺序旋转次数 | 规划轮数 | 实际 Galois 密钥数 / 序列化字节 |
|---|---|---:|---:|---:|
| 原生 HEIR | 原生分解 | 不作为本规划器决策 | 不适用 | 5 / 52,433,120 |
| 逐层 | 直接旋转 7、9 | 2 | 1 | 4 / 41,946,496 |
| 全图 | `7=6+1`、`9=8+1` | 3 | 1 | 3 / 31,459,872 |
| 全图，输出要求 `rotation_path_limit=2` | 回到直接旋转 7、9 | 2 | 2 | 4 / 41,946,496 |

最后两行的输入模型、权重、层级、价格和候选生成规则不变。只改变 `sink` 合约：第一轮发现深度 3，沿 SSA 找到违反约束的路径，将该路径上的候选深度加入同一 HiGHS 模型，第二轮得到可行选择。再次运行规划，选中 IR 字节一致。将上限收紧为 1 会报告不可行。门限是 **Pass 1 顺序自同构的路径资源**，不是经验方差、总耗时或精度证书。

逐层方案指**每一层作决策时忽略其他层的共享反馈**；最终计量仍将实际生成的密钥按物理并集去重。`MarshalBinary()` 的字节数是序列化长度，不是峰值内存。全图比原生少 20,973,248 B，并不意味着同等比例加速。

## 验收与范围

三种模型、三种执行方案、每种 64 组输入，总共 **576 次密文推理，25,344 个逻辑输出**。复用原 Go 验收程序，只调整生成包和输入文件所在目录。对照是原始权重与真正的 float64 SiLU；逐槽检查 `2^-12`、实测层级 `7→0` 和方案间输出 scale。三个模型原生 / 逐层 / 全图的密钥数依次为 `2/3/2`、`3/4/3`、`5/4/3`。另对上限 2 的新程序运行 4 组输入、256 个输出，与原生 HEIR 逐槽比较，不计入上述 576 次。各臂 Linf、MSE、在线时延、密钥清单和反馈误差见[最终原始日志](unified-ir-results/inference.log)。

[IR 回归](unified-ir-results/ir-tests.log)覆盖注册、往返、两条 SiLU 支路、幂等重算、过期预算、错误 profile/布局/路由及更改多项式系数后的实际 lowering。三个模型均无 bootstrap。独立样例：`N=16384→8192`、`ℓ=3→2`、附加方差 5、节省 40、罚金 35，判定启用；平局、负收益、方差超额、层级不足、布局不合分别测试。选中但未实现的环切 bootstrap **不能发射**。

**本次交付的是统一 IR 和迁入其中的 Pass 1，不是四个优化算法都已完成。** 类型中的 level 和 scale 已核对；预算中的区间是所选明文算术轨迹的区间，`nominal_log_scale` 是 HEIR 标称值。未经验证的 canonical bias、variance proxy、decoded error 和 validity failure 留为未知。Pass 2 的复合多项式前沿、Pass 3 的参数与 gadget 联合求解和具体安全筛查、Pass 4 的动态转换选择仍需实现。`convert` 已注册并有静态契约，不作为本轮推理的已执行算子。下一步是在现有 `poly_eval` 节点上接 Pass 2 候选，让实际消耗的层级反馈给后层 `lintrans`。

## 在服务器复现

```bash
cd /home/fhecompiler
build/native/ir/snabf-opt --show-dialects
build/native/ir/snabf-opt --help | grep -- --snabf-
build/native/ir/snabf-opt --allow-unregistered-dialect --snabf-check-plan \
  doc/unified-ir-results/reuse64-joint.mlir -o /dev/null
bash src/tools/run_unified.sh
```

首次安装在 Ubuntu 22.04 x86-64 使用 `bash src/tools/bootstrap_linux.sh`、`bash src/tools/bootstrap_mlir.sh`。新增 SDK 校验 SHA-256 后解包至 `extern/mlir24`，不改系统 LLVM。完整中间文件在 `build/unified/<模型>/`，原有回归保留在 `build/pass1/`。
