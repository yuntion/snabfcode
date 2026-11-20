# Pass 1 旧 Python 入口清理与原生回归

2026-11-19，在 `10.15.0.9-l20server:/home/fhecompiler` 部署。同一份源码也同步到本机正式 `sn_abf_paper_code` 目录。远端为目录部署，不是 Git 远程仓库。变更补丁保存在本目录的 `pass1-native.patch`，已变更的存量文件摘要见 `updated.sha256`。

## 文件与入口

- 删除 `src/compiler/pass1/{heir_ir,compile,bootstrap}.py`、该目录的 `__pycache__` 和旧 `test/compiler/test_pass1.py`。没有把旧实现复制到另一个可调用的模块，历史报告和本次删前快照不参与编译。
- `src/tools/run_pass1.sh` 不再将编译器的 Pass 1 目录放到 Python 搜索路径，不再设置 `SNABF_PASS1_CLI`，不调用旧 unittest。脚本构建注册了 `snabf` 的原生 `snabf-opt`，执行 CTest、原生 MLIR 检查和 Go 密文回归；无 CTest 用例时报错。前端、测试驱动和证据归档仍可用 Python，Pass 1 决策只由 C++ 实现。独立 C++ JSON CLI 未删除，也没有取代正式 MLIR 路径。
- `Pass1MechanicsTest.cpp` 除已有的 256 种旋转支持集、密钥复用、平局和层级边界，另查方差从 5 增到 10 时罚金从 5 变为 15、净收益从 5 变为 −5，检查超占用、非法并行倍数和 split-repack 成本 −59。
- `test_unified_ir.py` 向真实 `snabf-opt` 输入有歧义/无绑定的 HEIR 预处理表、缺自举校准的真实模块，确认结构化错误。保留合法对照与既有过期计划、布局、下降发射和密文测试；不把旧文本识别器“一张表禁止多次读取”的限制当作 SSA 语义。

## 复现

```bash
cd /home/fhecompiler
b=$(mktemp -d "$PWD/build/pass1-native.XXXXXXXX")
SNABF_PASS1_BUILD="$b" bash /home/fhecompiler/src/tools/run_pass1.sh
bash /home/fhecompiler/doc/pass1-native-cleanup-2026-11-19/pass4-core.sh \
  "$b/native/ir/snabf-opt" "$b/pass4-core"
```

完整 Pass 1 密文实测与预算证据见同目录 `results.md`、`evidence.json`、`run.log` 与 `raw-evidence.tar.gz`。Pass 4 使用同一个新编译器，对冻结的三组 MLIR 和估计器缓存逐项校验；此次选择的是核心门禁，不把之前 704 次加密评估计入本轮。`pass4-core.log`、两份发射收据及 `pass4-replay.json` 给出逐项结果。单环/混环的 Go 与计划指纹，分别跟各自历史同域结果一致。辅助测试不充当生产实现。

本轮原始构建目录为 `/home/fhecompiler/build/pass1-native-clean.W2pb5Lzm`，原始 Pass 1 运行目录为其下的 `run.lA2p4ZDE`。清理前独立的 A/B 报告对应旧源码清单；其精确历史源文件保存在该构建目录的 `pre-clean-sources.tar.gz`，不要用旧的 153 文件清单判定本轮已删除的文件是否应回装。
