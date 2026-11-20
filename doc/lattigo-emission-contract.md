# Lattigo 发射与数值对照契约

## 谁生成，谁执行

|对象|职责|不负责|
|---|---|---|
|`snabf-check-plan`|回放预算、具体 Q/P 与路由；对可发射的计划绑定当前 IR 指纹|猜测缺少的参数，或仅相信 `verified=true`|
|C++ `Target/Lattigo`|把具体密钥域、站点、截断、系数、层级与转换绑定成 Go 中的直接调用|解密、根据输出调参、对已生成 Go 作搜索替换|
|固定版本 HEIR|从独立的公共 MLIR 模块生成 `Add`、`Mul`、`MulPlain`、`Rescale` 等常规 CKKS 函数包|按模型选择路由、截断或降环|
|Go/Lattigo|创建和持有真实参数与密钥，执行加密、核、转换与解密，核对实际层级和 scale|解释审计 JSON、静默替换证书中指定的素数|
|历史 `go-bind emit`|重放已归档的 HEIR + 后绑定生成路径，供同域比较|作为新的模型发射器|

`ring-go=...` 是 `snabf-bind-lattigo` 指定输出文件的命令行选项，不是库名。使用环规划的模型按 `--snabf-check-plan` → `--snabf-lower-ckks` → `--snabf-bind-lattigo=ring-go=... audit=...` 生成 `package planned`；公共 helper 通过正常 Go import 引入，无需拼接模型源码。单独运行 Pass 3 而不运行 Pass 4 时，可用 `--snabf-lower-ckks=direct-go=true` 显式选择直出；默认的 HEIR Lowering 路径仍能复现历史结果。go-bind 的 `key-size` 只是测量工具，不能与其生成后改写子命令混为一谈。

## 哪些结果可以比较

1. 单环等价检验不只要求 N 相等。两臂必须使用相同 Q/P、scale、槽位布局及同一实际密钥。`planned.New()` 返回的 `SourceEvaluation()` 提供源域 evaluator/公开参数/encoder；只加密一次，复制密文输入给直出和旧版，分别在同域 `Decode`。先对独立 float64 真值逐槽比较，再记录两侧逐槽最大差。门限内一致不等于结果逐比特相同。
2. 混环方案的 `Decode` 在目标域。大环对照应在源域解密、小环计算在目标域解密，再按布局对齐同一组逻辑输出。绝不能把大环输出交给小环解密器。择优时的预测工作量也不是在线时间；实测在线值要单独记录。
3. 转换密钥门槛为 `max(128, snabf.switching_key_security_bits, SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS)`。固定模型下，源/目标常规参数仍按 128 bit 筛查。较严门槛随计划持久化；后续进程环境变宽不能降低已保存的门槛，环境变严需要重新规划。完整记录 QP 实例、物理环和嵌入秘密环，而不是只记录“通过”。
4. 先于末层汇合的 `W1·(SiLU(h)+h)` 与在末层后汇合的 `W1·SiLU(h)+h` 不等价。后者有额外的跨切面密文；不支持协同转换时必须保留大环且不创建转换密钥。被唯一一次写入完全覆盖的输出容器不等于一条密文旁路。

本契约的加密覆盖聚焦 Linear → SiLU → Linear 推理。独立的 Pass 1 自举实验不是本次混环推理证据，当前不可发射的 bootstrap、通用升环或多密文拆包不应推断为受支持。复核信任独立提供的、与版本/模型/模数匹配的估计器输出；计划指纹是内容绑定，不是对不可信估计器的密码学签名。观测值不证明全系统失败概率或转换密钥依赖安全。

区分两个历史版本：早期路径是 HEIR 生成模型 Go，`go-bind emit` 再改写；最初的 Pass 4 原型则由 C++ 输出程序描述，在 Go 中解释执行。两者都不是现在的专有直出发射器。新模型是单独的 `package planned` 文件，编译和运行时只 import 固定的公共函数及执行内核，不依赖旧版生成的模型 Go。对照脚本构建旧 `go-bind` 只是为了产生对照臂。
