"""Standalone negative checks; only fresh scratch outputs are modified."""
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import subprocess
import sys
import tempfile

root, run, d = map(Path, sys.argv[1:4])
name = sys.argv[4] if len(sys.argv)>4 else 'extra'
assert re.fullmatch(r'extra[-a-z0-9]*',name)
out = d/name; out.mkdir(mode=0o700)
opt = d/'native/ir/snabf-opt'
base = {'PATH':'/usr/bin:/bin', 'LC_ALL':'C', 'PYTHONDONTWRITEBYTECODE':'1',
        'LD_LIBRARY_PATH':':'.join(str(root/p) for p in ('extern/mlir24/usr/lib/llvm-24/lib', 'extern/mlir24/usr/lib/x86_64-linux-gnu', 'extern/highs-install/lib')),
        'SNABF_PASS4_SWITCHING_KEY_SECURITY_BITS':'128'}
cmd = [str(opt),'--allow-unregistered-dialect','--mlir-print-op-on-diagnostic=false']
def invoke(args, env, **kw):
    return subprocess.run(args, env=env, capture_output=True, text=True, timeout=90, **kw)
def success(result):
    assert result.returncode == 0, result.stderr
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

env = dict(base, SNABF_PASS4_ESTIMATES=str(run/'residual/cache.json'))
denied = []
for label in ('shell-fixed','shell-adaptive'):
    source = d/'replay'/(label+'.mlir'); left, anchor, right = source.read_text().partition('snabf.ring_search =')
    assert anchor
    right,n = re.subn(r'\bfully_overwritten = true\b','fully_overwritten = false',right,count=1); assert n == 1
    bad = out/(label+'-bad.mlir'); bad.write_text(left+anchor+right)
    for gate in ('--snabf-check-plan','--snabf-lower-ckks'):
        dest = out/(label+gate+'-out.mlir')
        success(invoke([*cmd,gate,str(source),'-o','/dev/null'],env))
        r = invoke([*cmd,gate,str(bad),'-o',str(dest)],env)
        assert r.returncode > 0 and 'error:' in r.stderr and 'stale_or_forged_ring_search' in r.stderr and not dest.exists(), (label,gate,r.stderr,dest.exists())
        (out/(label+gate+'.stderr')).write_text(r.stderr)
        denied.append(dict(case=label, gate=gate, changed='fully_overwritten: true -> false', rc=r.returncode, reason='stale_or_forged_ring_search', produced_output=False))
(out/'tamper.json').write_text(json.dumps(denied,indent=2)+'\n')
print('fully_overwritten: 4 authentic controls passed, 4 modified plans denied; no outputs',flush=True)

# Exercise the final publication failure with real sticky-directory permissions.
# Only disposable copies are exposed to an unprivileged process; neither the
# build directory nor the deployed tree changes ownership or access mode.
env = dict(base, SNABF_PASS4_ESTIMATES=str(run/'reuse64/cache.json'))
new = out/'mixed.lowered.mlir'; old = out/'single.lowered.mlir'
success(invoke([*cmd,'--snabf-lower-ckks',str(d/'replay/positive.mlir'),'-o',str(new)],env))
success(invoke([*cmd,'--snabf-check-plan','--snabf-lower-ckks=direct-go',str(run/'reuse64/input.mlir'),'-o',str(old)],env))
def bind(model, sidecar):
    return f'--snabf-bind-lattigo=ring-go={model} audit={sidecar}'
def pair(label, ir):
    folder = out/label; folder.mkdir()
    m,a = folder/'model.go',folder/'program.json'
    success(invoke([*cmd,bind(m,a),str(ir),'-o','/dev/null'],env)); record=json.loads(a.read_text())
    assert record['emit_sha256'] in m.read_text()
    return m,a
oldm,olda = pair('old',old); newm,newa = pair('new',new)
assert sha(oldm) != sha(newm) and sha(olda) != sha(newa)
user=pwd.getpwnam('nobody')
uid,gid=user.pw_uid,user.pw_gid
events=[]
with tempfile.TemporaryDirectory(prefix='snabf-private-rename-') as t:
    area=Path(t); os.chown(area,0,gid); area.chmod(0o750)
    def private_copy(p, name, mode):
        dest=area/name; shutil.copyfile(p,dest); os.chown(dest,0,gid); dest.chmod(mode); assert sha(dest)==sha(p); return dest
    worker=private_copy(opt,'snabf-opt',0o750)
    ir=private_copy(new,'lowered.mlir',0o640)
    cache=private_copy(run/'reuse64/cache.json','estimates.json',0o640)
    for soname in ('libMLIR.so.24.0','libLLVM.so.24.0'):
        private_copy(root/'extern/mlir24/usr/lib/llvm-24/lib'/soname,soname,0o640)
    settings=dict(base,SNABF_PASS4_ESTIMATES=str(cache),LD_LIBRARY_PATH=str(area))
    def emit(m,a):
        return invoke([str(worker),*cmd[1:],bind(m,a),str(ir),'-o','/dev/null'],settings,user=uid,group=gid,extra_groups=[],cwd=area,umask=0o077)
    for label in ('control','old-audit','no-audit'):
        folder=area/label; folder.mkdir(); os.chown(folder,0,gid); folder.chmod(0o1770)
        m,a=folder/'model.go',folder/'program.json'
        if label=='control':
            success(emit(m,a)); assert (sha(m),sha(a))==(sha(newm),sha(newa)); continue
        shutil.copyfile(oldm,m); os.chown(m,0,gid); m.chmod(0o640)
        old_inode=None
        if label=='old-audit':
            shutil.copyfile(olda,a); os.chown(a,uid,gid); a.chmod(0o600); old_inode=a.stat().st_ino
        r=emit(m,a); (out/(label+'.stderr')).write_text(r.stderr)
        assert r.returncode>0 and 'cannot_publish_output' in r.stderr, r.stderr
        assert sha(m)==sha(oldm) and not list(folder.glob('*.tmp')), list(folder.iterdir())
        if old_inode is not None:
            assert sha(a)==sha(olda) and a.stat().st_ino!=old_inode
        else:
            assert not a.exists()
        events.append(dict(case=label, reason='cannot_publish_output', rc=r.returncode,
            old_model_preserved=True, old_model_sha256=sha(m), audit_after_sha256=sha(a) if a.exists() else None,
            audit_restored_as_new_inode=(old_inode is not None and a.stat().st_ino!=old_inode), temp_files=0))
(out/'publish.json').write_text(json.dumps(events,indent=2)+'\n')
print('publication: clean control passed; final Go rename denied in both cases; previous pair/absent audit restored; 0 temporaries',flush=True)
