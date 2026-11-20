# 远端 A+B 联合门禁与 Pass 1 Python 依赖复核（2026-11-19）

远端 `10.15.0.9-l20server:/home/fhecompiler`，本轮全新构建和原始日志在 `/home/fhecompiler/build/pass4-joint-recheck.Wps2mqbI`。本机 WSL 只用于读原始冻结记录和对正式 Windows 工程的挂载视图做哈希核对。没有重跑整批密文推理或重新运行安全估计器，不把原报告的 704 次推理算为本轮结果。

## 四项 Pass 1 文件的实际用途

| 项目 | 当前用途 | 清理建议 |
|---|---|---|
| `heir_ir.py` | 历史 HEIR 文本检查；`test/compiler/test_pass1.py:4` 导入 `inspect, rewrite`。`rewrite` 主动拒绝发射。 | 不属于原生优化管线。先迁移兼容测试，再移出 Pass 1 目录，不能直接删。 |
| `bootstrap.py` | `heir_ir.py` 和旧单测用的薄适配器，把 JSON 转给 `snabf-pass1 --bootstrap`；判定在 `RingSwitch.cpp`。 | 与 `heir_ir.py` 一起放到兼容工具区；不是第二套自举实现。 |
| `compile.py` | 手动运行的 `--plan-only` 历史 JSON 对照入口；没有找到正式流水线或现有测试调用它。 | 可移出生产目录；需支持旧命令时保留为归档工具。 |
| `__pycache__/` | 两份 CPython 3.10 `.pyc`；不是源码、构建输入或交付凭证。 | 可清理并排除，未替用户删除。 |

远端 `src/tools/run_pass1.sh:19` 显式跑旧单测；新编的 C++ `snabf-pass1` 与上述两份 Python 适配代码配合，4/4 通过。原生 `snabf-opt` 链接 `NativeRouting.cpp` 和 `NativeBootstrap.cpp`。把旧工具迁到 `src/tools/legacy/pass1/` 时应同步更新脚本的 `PYTHONPATH`、测试和适配器默认 CLI 定位；现有历史证据保留，新快照另建清单。

## 交付文件与冻结点

- 三个新增文件 `SSACut.h`、`SSACutTest.cpp`、`replay_gates.py` 全在正式交付补丁和服务器。旧临时 Git 工作树仅对这三份执行了暂存，索引均为 `A`。远端是目录部署，没有声称向远端 Git 新增提交。
- 发布清单 39/39、上一轮运行的源码清单 153/153 逐项匹配。最终的七份 B 源码在 Windows 正式目录、临时 Git 工作副本、WSL 正式目录挂载视图和服务器一致。原始 `B-source.sha256` 冻结在联调前，直接比对为 5/7：`Pass4IR.cpp` 与 `replay_gates.py` 后来补了 `cross_residual` 的可证公开预加载识别及其反例。只在副本上反向应用已发布增量补丁，恢复 7/7 原始哈希；未改写原始冻结清单。完整值见 `inputs.json`、`local-sources.json` 与 `wsl-final.sha256`。

## 本次实跑

| 项目 | 结果 |
|---|---|
| A/B 原生联编 | Release 全新构建，包含 `EmitVerifier`、`RingEmitter`、`EmitGraph`、`DirectGo`、`Pass3IR`、`Pass4IR`、`SSACutTest`，CTest 8/8。 |
| B 侧 | `replay_gates.py` 28 次预期拒绝，含 crossing/ABI shell。相同 `reuse64` 缓存下，转换密钥门限 128 时取 N=8192；512 时因 `switching_key_ext_modulus_security_rejected` 保留 N=16384，源域和目标域策略均维持 128。缓存的最小相关经典估计约 165.0054 bit，不是全系统安全证明。 |
| HEIR 与 Go | 固定 translator 的 `--check` 字节相同；helper SHA-256 为 `3ba17babf49e6153724b374ebc22232f047121a9094c841fa8512c8c8d3f3102`。`go test -mod=readonly -count=1 ./src/backend/ringswitch/epochs/...` 通过，helper 子包自身无测试。 |
| A 侧 | `test-direct-gate.py` 在单环与 `--pass4` 各通过 9 个负例。两份 JSON 与先前归档逐字节相同；单环/混环分别 35/34 条指令。两者不是跨域同密文解密对照。 |
| 指定篡改 | 仅将固定/自适应两份 `snabf.ring_search` 里的 `fully_overwritten` 从 true 翻为 false；`check-plan`/`lower-ckks` 四次均返回 `stale_or_forged_ring_search` 且无输出，四个未篡改对照通过。 |
| 两文件发布 | 成功对照通过。普通用户在隔离粘滞目录中对最后发布的旧 `model.go` 无替换权限，实际得到 `cannot_publish_output: Permission denied`。两种场景下旧 Go 逐字节保留；有旧 JSON 时回滚为原字节但新 inode，原无 JSON 时恢复为不存在。两者都没有残留 `.tmp`。单文件 rename 与失败恢复不是两文件跨进程的原子事务，跨文件读者仍需核对 SHA。 |

夹具第一次因普通用户不能遍历正式 `extern` 而报 `libMLIR.so.24.0: cannot open shared object file`，没进入发射器；该次日志为 `extra.log`。只为夹具复制并哈希核对公开的 MLIR/LLVM 共享库，没有放宽任何部署目录权限；复测为 `extra-retry1.log`。原始粗估值来自已锁定的外部估计器缓存；各组负例有重叠，不合计为不同漏洞数。

## 重放

在相同远端部署环境执行 `bash /home/fhecompiler/doc/pass4-joint-recheck-2026-11-19/recheck.sh`。脚本新建一次性目录，不修改原始报告。标准门禁不需要切换用户；补充的真实文件系统发布失败测试需要能切到本机 `nobody` 并分配夹具文件属主的执行账户。
