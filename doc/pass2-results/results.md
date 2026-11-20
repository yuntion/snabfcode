# Pass 2 实验结果

服务器原始运行目录：`/home/fhecompiler/build/pass2/accept.7PNPxr6u/experiment`。逐槽比较对象是 **float64 的真实 SiLU 网络**；门限固定为 `2^-12`。

|模型|方案|选中 DAG|第二层输入|输入→输出|Galois 密钥|超限槽|L∞|MSE|在线均值 ms|
|---|---|---|---:|---|---:|---:|---:|---:|---:|
|control|heir|HEIR 原生|—|7→0|2|0|7.908e-05|3.467e-10|154.1|
|control|economy|silu_direct.d3.ps|2|7→1|2|0|9.137e-05|1.307e-09|130.8|
|control|precision|silu_direct.d7.ps|1|7→0|2|0|7.963e-05|7.872e-10|156.0|
|strided64|heir|HEIR 原生|—|7→0|3|0|4.256e-05|7.06e-11|171.4|
|strided64|economy|silu_direct.d3.ps|2|7→1|3|0|0.0001565|1.15e-09|133.9|
|strided64|precision|silu_direct.d7.ps|1|7→0|3|0|0.0001461|1.129e-09|162.4|
|reuse64|heir|HEIR 原生|—|7→0|5|0|4.301e-05|6.995e-11|171.4|
|reuse64|economy|silu_direct.d3.ps|2|7→1|3|0|0.0001618|1.056e-09|135.5|
|reuse64|precision|silu_direct.d7.ps|1|7→0|3|0|0.0001419|1.16e-09|160.8|
|residual|heir|HEIR 原生|—|6→0|5|1098|0.001357|6.993e-08|141.5|
|residual|economy|silu_direct.d3.ps|2|7→1|4|0|0.0001415|9.794e-10|114.3|
|residual|precision|silu_direct.d7.ps|1|7→0|3|0|0.0002032|1.154e-09|145.1|
|composite_coverage|heir|HEIR 原生|—|7→0|2|0|4.06e-05|1.324e-10|150.7|
|composite_coverage|free|silu_direct.d3.ps|5|10→4|2|0|0.0001082|1.165e-09|225.2|
|composite_coverage|forced_composite|silu_tanh_rational.d3.ps.d3.ps|2|10→1|2|0|9.672e-05|1.061e-09|236.6|
|composite_coverage|forced_composite_tight|silu_tanh_rational.d3.ps.d7.ps|1|10→0|2|0|0.0001415|2.102e-09|263.0|

## 按实验口径汇总

- 四个主模型（含 HEIR 对照）：768 次推理，37632 个逻辑输出，超限 1098 个；最大绝对误差 0.00135715，按输出数加权 MSE 8.36442e-09。
- 其中自由规划的 SN-ABF 方案：512 次推理，25088 个逻辑输出，超限 0 个；最大绝对误差 0.000203173，按输出数加权 MSE 1.10358e-09。
- 额外模板覆盖（含独立对照）：256 次推理，1024 个逻辑输出，超限 0 个；最大绝对误差 0.000141512，按输出数加权 MSE 1.11524e-09。

常规 precision 组只收紧公开的确定性逼近误差合约。额外 forced 组检验指定模板的密文实现，不是跨模板自由选型证据。
密钥字节为 Galois 密钥序列化之和，不是峰值内存；在线时间不含预处理、密钥生成、加密或解密。

`metrics.json` 和 `frontiers.json` 为可读索引。压缩包包含完整预算证据、所选 MLIR、Go 程序、原始输入和逐槽解密输出，以及复现命令与哈希。
证据包 SHA-256：`cd945b3f39baf60793e1e44dccbe0ebe8d9c9339817c7449bb4d0b6a5b8272e6`。
