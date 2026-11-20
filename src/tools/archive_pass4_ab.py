"""Publish a *completed* A/B run and the reproducible IR/Go/slot evidence."""
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path


def digest(f):
    h=hashlib.sha256()
    with f.open('rb') as data:
        for block in iter(lambda:data.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def archive(run,dest):
    summary=json.loads((run/'acceptance.json').read_text());paired=summary['paired']; boundary=summary['boundary']
    gates=summary['native_gates']
    if (set(paired)!={'control','strided64','reuse64','residual'} or set(boundary)!={'positive','security','bypass'} or
        gates['direct']['negative_cases']<1 or gates['mixed']['negative_cases']<1 or
        not gates['replay']['checked_crossing'] or not gates['replay']['checked_output_shell']):
        raise ValueError('incomplete experiment')
    dest.mkdir(parents=True,exist_ok=False)
    reports=['# C++ 直出发射与切面/转换密钥安全回归','','## 相同实际密文与评估密钥的单环对照','',
             '|网络|配对输入|核对槽数/侧|直出/旧版 L∞|两侧最大逐槽差|在线均值 ms（直出/旧版）|',
             '|---|---:|---:|---:|---:|---:|']
    trials=slots=0
    for net,m in paired.items():
        a,b=m['linf_direct_legacy'];x,y=m['online_ms_direct_legacy'];n=m['trials'];k=m['slots_per_arm']
        if n!=64 or not m['same_ciphertext_and_keys'] or max(a,b,m['paired_linf'])>2**-12:raise ValueError(net)
        reports.append(f"|{net}|{n}|{k:,}|{a:.4g} / {b:.4g}|{m['paired_linf']:.4g}|{x:.2f} / {y:.2f}|");trials+=2*n; slots+=2*k
    reports+=['','每个输入只加密一次，两臂用副本与同一评估器进行运算，轮换计时顺序。Galois 密钥集合/序列化尺寸与旧版 configure 独立核对。MSE、逐槽解密值、站点截断事件和时长见归档；误差门限为 2⁻¹²。','',
             '## 末层之前的三种决策','', '|策略|源图|活跃密文数|实测输出 N / L|转换密钥数|L∞|在线 ms|','|---|---|---:|---:|---:|---:|---:|']
    for name,m in boundary.items():
        if m['trials']!=64 or m['misses'] or m['linf']>2**-12:raise ValueError(name)
        count=sum(k['Role']=='down' for k in m['keys']); net='cross_residual' if name=='bypass' else 'reuse64'
        reports.append(f"|{name}|{net}|{len(m['cut']['live_values'])}|{m['actual_output_ring']} / {m['actual_level']}|{count}|{m['linf']:.4g}|{m['online_ms']:.2f}|")
        trials+=m['trials'];slots+=m['physical_encoded_slots_checked']
    reports+=['',f'共 {trials} 次模型密文推理，核对 {slots:,} 个编码槽，所有输出通过门限。直出逐边检查物理环维/层级/尺度。跨环输出由目标域解密，不用目标域密钥解旧版大环密文。', '',
        '`positive` 与 `security` 使用逐字节相同的估计器缓存；后者仅把转换密钥门槛提高到 512 bit。拒绝转换不是说原参数低于 128 bit；筛查针对指定经典模型，不是量子、循环密钥依赖或全系统失败概率的证明。', '',
        '跨切面残差是 `W1·SiLU(h) + h`；它和旧的 `W1·(SiLU(h) + h)` 是不同的源图。未实现的多值协同转换、split/repack、通用升环没有被计为完成。', '',
        f"本轮 A/B 链路负例 {len(summary['negatives'])} 项；独立单环/混环发射门禁各 {gates['direct']['negative_cases']}/{gates['mixed']['negative_cases']} 项，跨域回放门禁 {len(gates['replay']['denied'])} 项。计数按测试调用，各组有重叠，不相加作为独立攻击类型。历史对照仍用 go-bind；它不是新发射路径的依赖。", '',
        '在工程根目录执行 `bash src/tools/run_pass4_ab.sh`；命令创建新的独立运行目录。测试不复用历史解密结果。']
    (dest/'results.md').write_text('\n'.join(reports)+'\n');shutil.copyfile(run/'acceptance.json',dest/'acceptance.json');shutil.copyfile(run/'run.log',dest/'run.log')
    files=sorted(f for f in run.rglob('*') if f.is_file() and (f.suffix in {'.go','.json','.jsonl','.mlir','.log','.sha256'} or f.name.endswith('.go.txt')))
    with tarfile.open(dest/'evidence.tar.gz','w:gz',compresslevel=5) as tar:
        for f in files:tar.add(f,arcname=f.relative_to(run),recursive=False)
    (dest/'sha256.json').write_text(json.dumps({'archive':digest(dest/'evidence.tar.gz'),'members':{f.relative_to(run).as_posix():digest(f) for f in files}},indent=2)+'\n')


if __name__=='__main__':archive(*(Path(x).resolve() for x in sys.argv[1:]))
