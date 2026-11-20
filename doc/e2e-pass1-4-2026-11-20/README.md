# reuse64：Pass 1–4 统一入口、降环与安全回退实测

2026-11-20（新加坡时间），执行位置 `10.15.0.9-l20server:/home/fhecompiler`。本目录只归档这次新运行。冻结原始输入与估计器数据可以复用，密钥、加密、求值、解密和下列计时均为本次实测。工程为目录部署，不是一次 Git 推送。

## 验收结果

每行独占一个目录；串行执行，`JOBS=4`。每行 64 条输入、4,096 个逻辑输出、262,144 个实际编码槽（每条输入的 64 槽向量在 4,096 个物理槽中周期复制），超过 `2^-12` 的槽均为 0。

| 输入模式 | 转换密钥门槛 | 实际 N（输入→输出） | 输出 L / log₂(scale) | 最大绝对误差 | MSE | Eval 均值 / 中位数 (ms) |
|---|---:|---|---|---:|---:|---:|
| 原始 MLIR，重跑 P1–4 | 128 | 16384 → 8192 | 1 / 38.00000000000003 | 9.99634894860435e-7 | 2.7076409999963746e-14 | 157.669 / 152.538 |
| 同一原始 MLIR，重跑 P1–4 | 512 | 16384 → 16384 | 1 / 38.00000000000003 | 9.953690962016815e-7 | 2.7095312353654514e-14 | 190.487 / 184.811 |
| 冻结 Pass 3 检查点，重放 | 128 | 16384 → 8192 | 1 / 38.00000000000003 | 9.992549632326098e-7 | 2.7069904396507574e-14 | 158.778 / 152.331 |
| 同一检查点，重放 | 512 | 16384 → 16384 | 1 / 38.00000000000003 | 9.982956664933429e-7 | 2.709895377389808e-14 | 196.287 / 188.893 |

两种输入模式的密钥统计一致。128 组：Galois 密钥 4 把，共 **40,898,336 B**，下切换密钥 1 把 **3,146,080 B**。512 组：Galois 密钥 3 把，共 **56,627,808 B**，无下切换密钥。两组各另有序列化公钥 2,359,480 B、重线性化密钥 18,875,920 B。字节数是公钥材料的序列化大小，不是峰值内存；计时仅涵盖在线 `Eval`，不含建钥、加密或解密，不外推到其他负载。

`adaptive` 两组的源环与目标环普通域门槛始终为 128。512 只提高转换密钥扩展模数门槛，记录拒绝原因 `switching_key_ext_modulus_security_rejected`；回退没有发射 `ring_down`，也没有生成转换密钥。此处是锁定估计器模型的筛查和有限语料的精度实测，不是 512 位全系统安全或小概率失败界的证明。混环只在输出目标域解密，保大环只在源域解密，对齐同一明文真值，不跨域解读同一旧密文。

## 输入与开发记录

* `/home/fhecompiler/src/tools/run_e2e_pass1_4.sh` 为统一入口；`--from source` 真的读取 `--mlir`；`--from pass3` 要求另给原始 `--oracle-mlir`，只重放已具象化的上游。此验收支持已有 dense `Linear → SiLU → Linear` 语料契约及锁定具体 profile，不宣称任意模型/任意安全参数都能跑。`--strategy` 映射到已注册的原生 Pass 4 选项，未知错误不重试；`--sw-key-bits` 不得降低进程或 IR 中已有的门槛。
* `/home/fhecompiler/src/frontend/snabf_frontend/pass2.py` 抽出共享的 `lower_source`。`/home/fhecompiler/src/frontend/snabf_frontend/e2e_source.py` 从传入的矩阵源图，经标记、HEIR CKKS、导入、P1/P2 候选、共享规划/检查，再调用原生 P3 MILP 和具象参数/安全报告复核。要求重新求出的 Q/P/scale 与冻结 profile 相同。旧 Go 工具仅用于 `key-size` 实测，没有调用旧 `emit` 或已删除的 Pass 1 Python 编译脚本。
* 两组共同的源图、语料、profile、源域安全报告和多域估计缓存都按 SHA-256 锁定（`frozen.sha256`）。不在本轮重新跑 Sage 估计器；各轮原生复核检查证据对应的具体模数和环。两次源编译得到的 `work/pass3.json` 逐字节相同：`faaed3fdf172832b52df3687661ddef753fc46568ee0051a9470ee7eece6a328`。
* 源编译中固定版本 HEIR 用类型/属性哈希命名私有 `_assign_layout_*` 打包函数。独立进程之间名字可变，随之改变检查回执。配对程序仅在内存中的**对照副本**允许这一个私有符号及其回执不同，检查点剩余部分必须相同。各轮实际发射使用各自未修改的受检 IR；原始文件与哈希全部保留。另做了严格字节对照：两次重放的输入检查点以及重检后的检查点 SHA-256 均为 `96e2cd0005ea6b41dc41f022eabcdfe42891724289f307aa372aeda78baf6662`。
* 每一组都单独执行 `snabf-check-plan`、`snabf-lower-ckks`、`snabf-bind-lattigo`、`go build -mod=readonly` 以及 Lattigo v6.2.0 密文测试。`planned.mlir` 可能还保留上游回执；以 `work/checked.mlir` 为新回执锚点，必须和 `work/lowered.mlir`、`out/model.go`、`out/program.json` 一致。`program.json` 不作执行输入；Go build 输出是包归档，不冒充可执行文件。
* `/home/fhecompiler/src/tools/verify_e2e_pass1_4.py` 从原始矩阵和真正的 SiLU 重生成冻结语料中的 float64 真值，逐条读取 `out/trials.jsonl`，包括所有周期副本；独立核对误差、实际 N/L/scale、在线计时、密钥域和字节数。`/home/fhecompiler/test/pass4/e2e_gates.py` 将单个副本（槽 64）改动后被拒绝；错误估计器上下文在 `pass4` 阶段被拒绝，不发射 model/audit、无完成标记；CLI 要求 128 时也不能放松已有的 512 环境门槛。详见 `negative.json` 和 `logs/`。

## 复现与定位

在远端工程运行，无需引用任何历史 `build/run.*`，每次新建目录：

```bash
bash /home/fhecompiler/doc/e2e-pass1-4-2026-11-20/reproduce.sh source
bash /home/fhecompiler/doc/e2e-pass1-4-2026-11-20/reproduce.sh pass3
```

正式新鲜结果位于 `/home/fhecompiler/build/e2e-release-20261120/` 下：

| 模式 | 128 位 | 512 位 |
|---|---|---|
| 源编译 | `source/128/run.yUfmyIRM` | `source/512/run.wogS4pGV` |
| 检查点重放 | `pass3/128/run.2X7ZlxaM` | `pass3/512/run.yAHWUYuI` |

`pair-source.json`、`pair-pass3.json` 是配对复算结果，`logs/{source,pass3}-{128,512}.log` 是原始 stdout；`evidence.tar.gz` 含这四次运行的冻结输入、逐槽解密输出、逐阶段命令与日志、计划/检查/降低 IR、生成 Go 和 JSON。未打包可重建的 `.a` 或 key-meter 二进制，没有保存私钥。`SHA256SUMS` 校验此目录交付物；`code.sha256` 从工程根检查源码。打开旧证据不需要跑编译器或 HEIR Python：

```bash
cd /home/fhecompiler/doc/e2e-pass1-4-2026-11-20
sha256sum -c SHA256SUMS
d=$(mktemp -d); tar -xzf evidence.tar.gz -C "$d"
python3 /home/fhecompiler/src/tools/verify_e2e_pass1_4.py "$d/source/128/run.yUfmyIRM" --reuse64-pair "$d/source/512/run.wogS4pGV"
python3 /home/fhecompiler/src/tools/verify_e2e_pass1_4.py "$d/pass3/128/run.2X7ZlxaM" --reuse64-pair "$d/pass3/512/run.yAHWUYuI"
```
