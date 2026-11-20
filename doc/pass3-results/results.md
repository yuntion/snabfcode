# Pass 3 密文实验验收

服务器验收目录：`/home/fhecompiler/build/pass3/run.VUfjVOsG`。

两张图、三组方案、每组 64 个输入。每个输入检查 64 个逻辑槽。`no_trunc` 独立优化但禁止截断；`matched_exact` 固定联合方案 Q/P、密钥、输入密文与选中 DAG，只关闭截断。在线时间不含客户端加解密及预处理。

| 模型 | 方案 | log₂(QP) | log₂Δ | 输出层 | 超限/槽 | L∞ | MSE | 在线均值 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| reuse64 | `no_trunc` | 336.606 | 38 | 1 | 0/4096 | 9.6337e-07 | 2.6902e-14 | 186.80 |
| reuse64 | `joint` | 336.809 | 38 | 1 | 0/4096 | 9.6337e-07 | 2.6924e-14 | 186.73 |
| reuse64 | `matched_exact` | 336.809 | 38 | 1 | 0/4096 | 9.6337e-07 | 2.7010e-14 | 184.91 |
| residual | `no_trunc` | 342.500 | 39 | 1 | 0/4096 | 3.0343e-06 | 2.0475e-13 | 153.23 |
| residual | `joint` | 342.514 | 39 | 1 | 0/4096 | 3.0343e-06 | 2.0466e-13 | 151.12 |
| residual | `matched_exact` | 342.514 | 39 | 1 | 0/4096 | 3.0343e-06 | 2.0475e-13 | 154.01 |

## 工作量与截断误差

- **reuse64**：每次推理实际少做 4 行 gadget 外积；分解部分净少 28 次单模数 NTT/INTT。命中 `lin0.giant1, poly0.n1.relin, poly0.n3.relin, poly0.n5.relin`。同密文 `joint − matched_exact` 的最大槽差 4.5988e-08，经验方差 1.4241e-16。相对独立无截断方案，QP 位宽变化 +0.2036 bit；具体 QP 最低经典估计 136.95 bit。
- **residual**：每次推理实际少做 4 行 gadget 外积；分解部分净少 23 次单模数 NTT/INTT。命中 `poly0.n1.relin, poly0.n3.relin, poly0.n5.relin, lin1.giant1`。同密文 `joint − matched_exact` 的最大槽差 2.2336e-08，经验方差 2.3951e-17。相对独立无截断方案，QP 位宽变化 +0.0140 bit；具体 QP 最低经典估计 133.74 bit。

## 规划器回归

- 仅编译的同图预算扫描：ε=0.0006，P 目标 45.801 bit，k=1 的站点为 `lin0.giant1, poly0.n1.relin, poly0.n3.relin, poly0.n5.relin`。
- 仅编译的同图预算扫描：ε=0.00032，P 目标 50.448 bit，k=1 的站点为 `lin0.giant1, poly0.n1.relin, poly0.n3.relin`。
- 独立修改计划的截断档位和物理 scale；校验与 lowering 各拒绝两种修改，共 4 项负例。
- 真实 NTT 素数回切单测：`first prime=17179875329 refined=17203716097 concrete message/scale headroom at input; bit deficit=2.06379091594e-07`。

## 复现材料

[完整日志](run.log) · [机器可读结果](summary.json) · [证据包](evidence.tar.gz) · [SHA-256 清单](sha256.json)。

证据包包含同一运行的源模型、明文参考、每次推理输出、各方案的具象计划、已检查 MLIR、生成的 Go、估计器输入与攻击成本。每个 `metrics.json` 的 `Events` 是一遍独立、不计时的诊断推理；外积和 NTT 数是每次推理的数，不乘以槽数。模数容量与时延没有预设改善方向。数学归属与证书条件见[开发报告](../pass3-development.md)。
