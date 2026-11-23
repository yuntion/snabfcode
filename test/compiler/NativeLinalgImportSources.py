"""C-1 structural tests. Never runs HEIR or a ciphertext backend."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from native_import.oracle import HASHES, check, original, read_receipt, sha, spec_for, verify


def changed(raw, old, new):
    check(old in raw, f'test did not match {old!r}')
    return raw.replace(old, new, 1)


class Suite:
    def __init__(self, exe, root):
        self.exe, self.root = str(exe.resolve()), root
        traps = root/'traps'; traps.mkdir(); self.hit = root/'HEIR-WAS-RUN'
        for tool in ('heir-opt', 'heir-translate'):
            path = traps/tool
            path.write_bytes(b'#!/bin/sh\necho invoked > "$SNABF_TRAP_FILE"\nexit 99\n'); path.chmod(0o755)
        self.env = dict(os.environ, PATH=str(traps), HEIR_OPT=str(traps/'heir-opt'), HEIR_TRANSLATE=str(traps/'heir-translate'),
                        SNABF_TRAP_FILE=str(self.hit))
        self.positive, self.negative, self.tampered = [], [], []

    def invoke(self, name, src, spec, *, reject=None, line=None, stdout=False):
        source, ports, out = [self.root/(name+ext) for ext in ('.mlir','.inputs.json','.json')]
        source.write_bytes(src); ports.write_bytes(spec)
        cmd = [self.exe, str(source), '--input-spec', str(ports)]
        if not stdout: cmd += ['-o', str(out)]
        run = subprocess.run(cmd, capture_output=True, env=self.env, timeout=20)
        diag = run.stderr.decode(errors='replace')
        if reject is not None:
            check(run.returncode == 1 and not run.stdout and not out.exists(), f'{name}: not cleanly rejected: {run.returncode} {diag}')
            reasons = (reject,) if isinstance(reject,str) else reject
            check(any(reason in diag for reason in reasons), f'{name}: unexpected rejection: {diag}')
            row = str(line) if line else r'[1-9]\d*'
            check(re.search(re.escape(str(source))+fr':{row}:[1-9]\d*:', diag), f'{name}: no physical location: {diag}')
            self.negative.append(name)
            return
        check(run.returncode == 0 and not run.stderr, f'{name}: {run.returncode} {diag}')
        check(stdout or not run.stdout, f'{name}: unexpected stdout')
        self.positive.append(name)
        return read_receipt(run.stdout if stdout else out.read_bytes())

    def tamper(self, name, src, spec, receipt, edit):
        forged = copy.deepcopy(receipt); edit(forged)
        try:
            verify(src, spec, forged)
        except ValueError as exc:
            check('not bound' in str(exc), f'{name}: wrong independent failure: {exc}')
            self.tampered.append(name)
        else:
            raise AssertionError(f'{name}: forged receipt accepted')


def source_negatives(t, c, cs, r, rs):
    def fail(name, old, new, reason, source=c, spec=cs, at=None):
        mutated = changed(source, old, new)
        line = None if at is None else next(i for i, text in enumerate(mutated.splitlines(), 1) if at in text)
        t.invoke(name, mutated, spec, reject=reason, line=line)
    for name, old, new, diagnostic in [
        ('nonzero-fill', b'%zero = arith.constant 0.0', b'%zero = arith.constant 1.0', 'SNABF_NATIVE_OUTS'),
        ('negative-zero-fill', b'%zero = arith.constant 0.0', b'%zero = arith.constant -0.0', 'SNABF_NATIVE_OUTS'),
        ('empty-accumulator', b'outs(%init', b'outs(%empty', 'SNABF_NATIVE_OUTS'),
        ('live-accumulator', b'outs(%init', b'outs(%x', 'SNABF_NATIVE_OUTS'),
        ('erf-wrong-x', b'%z = arith.mulf %in', b'%z = arith.mulf %old', 'reads outs'),
        ('erf-wrong-bypass', b'%s = arith.mulf %in', b'%s = arith.mulf %e', 'source-SSA'),
        ('erf-wrong-yield', b'linalg.yield %p', b'linalg.yield %d', 'source-SSA'),
        ('erf-ulp32', b'0.7071067811865475244', b'0x3F3504F4', 'source-SSA'),
        ('erf-wrong-half', b'0.5 : f32', b'0.25 : f32', 'source-SSA'),
        ('erf-tanh', b'math.erf %z', b'math.tanh %z', 'unsupported scalar operation'),
        ('erf-dead-recognized', b'%e = math.erf', b'%dead = math.erf %z : f32\n%e = math.erf', 'source-SSA'),
        ('erf-dead-unknown', b'%e = math.erf', b'%dead = math.sin %z : f32\n%e = math.erf', 'unsupported scalar operation'),
        ('erf-fastmath', b'math.erf %z :', b'math.erf %z fastmath<fast> :', 'SNABF_NATIVE_FLOAT'),
        ('fake-activation-tag', b'domain_lower =', b'snabf.activation = "gelu", domain_lower =', 'SNABF_NATIVE_ATTR'),
        ('non-identity-map', b'[#id, #id]', b'[affine_map<(i)->(0)>, #id]', 'identity maps'),
        ('wrong-domain', b'domain_lower = -1.0', b'domain_lower = 2.0', 'activation bounds'),
        ('missing-domain', b'domain_upper = 1.0 : f64', b'domain_upper = 1.0 : f32', 'activation bounds'),
        ('secret-arg-value', b'{secret.secret}', b'{secret.secret = true}', 'argument unit marker'),
        ('other-arg-attr', b'{secret.secret}', b'{secret.other}', 'argument unit marker'),
        ('mgmt-arg-attr', b'{secret.secret}', b'{mgmt.foo}', 'argument unit marker'),
        ('secret-result-attr', b'-> tensor<4xf32> {\n', b'-> (tensor<4xf32> {secret.secret}) {\n', 'result attributes'),
        ('mgmt-on-constant', b'0.0 : f32', b'{mgmt.foo} 0.0 : f32', 'unsupported attribute'),
        ('secret-on-yield', b'linalg.yield %p : f32', b'"linalg.yield"(%p) {secret.secret} : (f32) -> ()', 'unsupported attribute'),
        ('module-marker', b'module {', b'module attributes {secret.secret} {', 'unsupported attribute'),
        ('secret-op', b'return %l1', b'"secret.generic"() : () -> ()\nreturn %l1', 'secret'),
        ('mgmt-op', b'return %l1', b'"mgmt.modreduce"() : () -> ()\nreturn %l1', 'mgmt'),
        ('unknown-op', b'return %l1', b'"alien.foo"() : () -> ()\nreturn %l1', 'alien'),
        ('known-unadmitted-op', b'return %l1', b'%dead = math.sin %l1 : tensor<4xf32>\nreturn %l1', 'unsupported operation'),
        ('dynamic-loop', b'return %l1', b'scf.while : () -> () {\n%b = arith.constant false\nscf.condition(%b)\n} do {\nscf.yield\n}\nreturn %l1', 'unsupported control flow'),
        ('dynamic-branch', b'return %l1', b'%cond = arith.constant false\ncf.cond_br %cond, ^yes, ^no\n^yes:\nreturn %l1 : tensor<4xf32>\n^no:\nreturn %l1', 'unsupported control flow'),
        ('call', b'return %l1', b'%call = func.call @f88280ebd6b8f89e(%x) : (tensor<4xf32>)->tensor<4xf32>\nreturn %l1', 'unsupported control flow'),
        ('relu-in-erf-not-a-witness', b'%e = math.erf %z', b'%e = arith.maximumf %z, %zero', 'SNABF_NATIVE_RELU_SSA'),
        ('source-nul', b'return %l1', b'\0return %l1', 'embedded NUL'),
    ]:
        fail(name, old, new, diagnostic)
    for name, old, new in [
        ('silu-wrong-numerator', b'arith.divf %one, %den', b'arith.divf %in, %den'),
        ('silu-reverse-division', b'arith.divf %one, %den', b'arith.divf %den, %one'),
        ('silu-wrong-branch', b'arith.mulf %in, %sig', b'arith.mulf %exp, %sig'),
        ('silu-wrong-negation', b'arith.negf %in', b'arith.negf %one'),
        ('silu-wrong-one', b'1.0 : f32', b'0x3F800001 : f32'),
    ]:
        fail(name, old, new, 'source-SSA', source=r, spec=rs)
    fail('physical-not-debug-loc', b'%e = math.erf %z : f32', b'%e = math.sin %z : f32 loc("spoof.mlir":999:1)',
         'unsupported scalar operation', at=b'math.sin')


def spec_tests(t, c, cs):
    def bad(name, edit):
        d = json.loads(cs); edit(d); t.invoke(name, c, json.dumps(d).encode(), reject='SNABF_NATIVE_INPUT_SPEC')
    for name, field, value in [('wrong-function','func_symbol','not_the_entry'), ('wrong-index','arg_index',1),
                               ('arg-name','arg_name','x'), ('public-with-marker','classification','public'),
                               ('empty-source-id','source_id',''), ('inverted-input-domain','domain',[1,-1]),
                               ('nonfinite-input-domain','domain',[0,'Infinity']), ('null-domain','domain',None)]:
        bad(name, lambda d, f=field,v=value: d['inputs'][0].update({f:v}))
    bad('missing-class', lambda d: d['inputs'][0].pop('classification'))
    bad('duplicate-port', lambda d: d['inputs'].append(d['inputs'][0]))
    bad('schema-version', lambda d: d.update(schema='future'))
    t.invoke('no-input-spec', c, b'{}', reject='SNABF_NATIVE_INPUT_SPEC')
    duplicate = changed(cs, b'"classification":"secret"', b'"classification":"public","classific\\u0061tion":"secret"')
    t.invoke('ambiguous-duplicate-key', c, duplicate, reject='SNABF_NATIVE_INPUT_SPEC')


def quoted_ops(t, c, cs):
    v, w = b'tensor<4xf32>', b'tensor<4x4xf32>'
    orig = next(l for l in c.splitlines() if l.startswith(b'%l0 = '))
    m = (b'%l0 = "linalg.matvec"(%w0,%x,%init) <{operandSegmentSizes=array<i32:2,1>}> ({\n'
         b'^bb0(%a:f32,%v:f32,%acc:f32):\n%p = arith.mulf %a, %v : f32\n%s = arith.addf %acc, %p : f32\n'
         b'linalg.yield %s : f32\n}) : ('+w+b','+v+b','+v+b') -> '+v)
    t.invoke('quoted-matvec', changed(c,orig,m), cs)
    for name, old, new in [('quoted-ignore-acc',b'%acc, %p',b'%p, %p'), ('quoted-sub-product',b'arith.mulf',b'arith.subf'),
                           ('quoted-swap-w-x',b'%a, %v',b'%v, %a'), ('quoted-yield-acc',b'yield %s',b'yield %acc')]:
        t.invoke(name, changed(c,orig,changed(m,old,new)), cs, reject=('SNABF_NATIVE_MATVEC','expected add/mul op in the body'))
    cache = changed(m,b'}) :',b'}) {linalg.memoized_indexing_maps = [42 : i64]} :')
    t.invoke('fake-matvec-maps',changed(c,orig,cache),cs,reject='source-supplied')
    forged_loc = changed(m,b'%acc, %p',b'%p, %p')+b' loc("not-the-input.mlir":999:1)'
    raw = changed(c,orig,forged_loc); row = raw[:raw.index(b'"linalg.matvec"')].count(b'\n')+1
    t.invoke('upstream-physical-loc',raw,cs,reject=('SNABF_NATIVE_MATVEC','expected add/mul op in the body'),line=row)
    orig = next(l for l in c.splitlines() if l.startswith(b'%init = '))
    fill = (b'%init = "linalg.fill"(%zero,%empty) <{operandSegmentSizes=array<i32:1,1>}> ({\n'
            b'^bb0(%z:f32,%out:f32):\nlinalg.yield %z : f32\n}) : (f32,'+v+b') -> '+v)
    t.invoke('quoted-fill',changed(c,orig,fill),cs)
    t.invoke('quoted-fill-read-outs',changed(c,orig,changed(fill,b'yield %z',b'yield %out')),cs,reject='SNABF_NATIVE_OUTS')


def other_forms(t, c, cs):
    for name, raw in [('renamed-arg',c.replace(b'%x',b'%image')), ('no-marker',c.replace(b' {secret.secret}',b'')),
                      ('crlf',c.replace(b'\n',b'\r\n')), ('comment',b'// raw bytes count\n'+c),
                      ('f64',c.replace(b'f32',b'f64'))]:
        receipt = t.invoke(name, raw, cs, stdout=True); verify(raw, cs, receipt)
        check(receipt['source_sha256'] != sha(c), 'source was normalized')
    f64 = c.replace(b'f32',b'f64'); bad = changed(f64,b'0.7071067811865475244',b'0x3FE6A09E667F3BCC')
    t.invoke('erf-ulp64',bad,cs,reject='source-SSA')
    unknown = json.loads(cs); del unknown['inputs'][0]['domain']; spec = json.dumps(unknown).encode()
    receipt = t.invoke('unknown-domain',c,spec); verify(c,spec,receipt)
    check(receipt['inputs'][0]['domain'] is None, 'unknown domain fabricated')
    receipt = t.invoke('spec-raw-bytes',c,b' '+cs); verify(c,b' '+cs,receipt)
    check(receipt['input_spec_sha256']!=sha(cs), 'spec normalized')

    head = (b'module {func.func @f88280ebd6b8f89e(%x: tensor<4xf32> {secret.secret}) -> tensor<4xf32> {\n')
    v = b'tensor<4xf32>'
    program = (head+b'%w = arith.constant dense<[2.0, -3.0, 0.0, 1.0]> : '+v+b'\n'
               b'%n = arith.negf %x : '+v+b'\n%p = arith.mulf %n, %w : '+v+b'\n'
               b'%s = arith.subf %p, %x : '+v+b'\n%a = arith.addf %s, %x : '+v+b'\nreturn %a : '+v+b'\n}}')
    res = t.invoke('elementwise',program,cs)
    check([(n['kind'],n['inputs']) for n in res['nodes']] == [('constant',[]),('negate',['arg0']),
          ('mul_plain',['n1','n0']),('sub',['n2','arg0']),('add',['n3','arg0'])], 'wrong elementwise edges')
    t.invoke('ct-ct-elementwise-mul',changed(program,b'mulf %n, %w',b'mulf %n, %x'),cs,reject='SNABF_NATIVE_ELEMENTWISE')
    ident = head+b'return %x : '+v+b'\n}}'
    t.invoke('dynamic-tensor',ident.replace(b'4xf32',b'?xf32'),cs,reject='SNABF_NATIVE_SHAPE')
    t.invoke('ragged-period',ident.replace(b'4xf32',b'3xf32'),cs,reject='SNABF_NATIVE_SHAPE')
    t.invoke('malformed-trailing', c+b'garbage\n',cs,reject='error:')


def tamper_tests(t,c,cs,receipt):
    def wrong_x(r):
        lock = r['activation_locks'][0]; lock['x'] = 'arg0'
        next(n for n in r['nodes'] if n['id']==lock['site'])['inputs'][0] = lock['x']
    def wrong_inv(r):
        lit = next(v for v in r['activation_locks'][0]['literals'] if v['role']=='inv_sqrt2')
        lit['bits'] = '0x3f3504f4'; next(n for n in r['nodes'] if n['id']==lit['id'])['bits'] = [lit['bits']]
    def wrong_weight(r):
        next(n for n in r['nodes'] if len(n.get('shape',[]))==2)['bits'][0]='0x3e000001'
    def wrong_outs(r):
        m = next(n for n in r['nodes'] if n['kind']=='matvec'); m['inputs'][2] = m['outs'] = 'arg0'
    for name, edit in [('concordant-wrong-x',wrong_x),('concordant-ulp',wrong_inv),('weight-ulp',wrong_weight),
                       ('concordant-wrong-outs',wrong_outs),('result-edge',lambda r:r['outputs'][0].update(value='arg0')),
                       ('boolean-result-index',lambda r:r['outputs'][0].update(result_index=False)),
                       ('covering-digest',lambda r:r.update(source_sha256='0'*64))]:
        t.tamper(name,c,cs,receipt,edit)

    start = c.index(b'%gelu = linalg.generic'); end = c.index(b'} -> tensor<4xf32>\n',start)+len(b'} -> tensor<4xf32>\n')
    clone = c[start:end].replace(b'%gelu =',b'%second =').replace(b'ins(%l0',b'ins(%gelu')
    two = c[:end]+clone+c[end:].replace(b'%w1, %gelu',b'%w1, %second')
    out = t.invoke('two-sites',two,cs); verify(two,cs,out)
    t.tamper('swap-site-locks',two,cs,out,lambda r:r['activation_locks'].reverse())
    def swap_bodies(r):
        a,b = r['activation_locks']; a['body'],b['body'] = b['body'],a['body']
        a['region_result'],b['region_result'] = b['region_result'],a['region_result']
    t.tamper('swap-scoped-bodies',two,cs,out,swap_bodies)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('importer', nargs='?', type=Path)
    ap.add_argument('--c1-source', type=Path, help='actual archive member; exact SHA enforced')
    ap.add_argument('--keep', type=Path, help='new directory for the four canonical receipts')
    ap.add_argument('--replay', type=Path, help='offline independent replay; no importer subprocess')
    ap.add_argument('--strace', type=Path, help='optional Linux execve/execveat audit for all four original inputs')
    a = ap.parse_args()
    if a.replay:
        for case in HASHES:
            verify(original(case,a.c1_source),spec_for(case),read_receipt((a.replay/(case+'.json')).read_bytes()))
        print('OFFLINE: 4/4 complete receipts match frozen source/spec bytes'); return
    if not a.importer: ap.error('importer required unless --replay is used')
    if a.keep: a.keep.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as tmp:
        t = Suite(a.importer,Path(tmp)); receipts = {}
        for case, digest in HASHES.items():
            src = original(case,a.c1_source); spec = spec_for(case)
            receipt = t.invoke(case,src,spec); verify(src,spec,receipt); receipts[case] = receipt
            if a.keep: (a.keep/(case+'.json')).write_bytes((t.root/(case+'.json')).read_bytes())
            if a.strace:
                trace = (a.keep or t.root)/(case+'.execve')
                run = subprocess.run([str(a.strace.resolve()),'-f','-qq','-e','trace=execve,execveat','-o',str(trace),
                    t.exe,str(t.root/(case+'.mlir')),'--input-spec',str(t.root/(case+'.inputs.json'))],
                    env=t.env,capture_output=True,timeout=30)
                lines = [line for line in trace.read_text().splitlines() if re.search(r'\bexecve(?:at)?\(',line)]
                check(run.returncode==0 and len(lines)==1 and f'execve("{t.exe}"' in lines[0], f'{case}: unexpected exec tree: {lines} {run.stderr!r}')
                verify(src,spec,read_receipt(run.stdout))
            print(f'{case}: {digest}  nodes={len(receipt["nodes"])} locks={len(receipt["activation_locks"])}')
        c,cs = original('c1-single',a.c1_source),spec_for('c1-single')
        r,rs = original('reuse64'),spec_for('reuse64')
        source_negatives(t,c,cs,r,rs); spec_tests(t,c,cs); quoted_ops(t,c,cs)
        other_forms(t,c,cs); tamper_tests(t,c,cs,receipts['c1-single'])
        # The lexical site IDs and local expression are identical on these R3
        # sources. Swapping just the site receipt still crosses source identity.
        t.tamper('cross-source-same-site', r, rs, receipts['reuse64'],
                 lambda out: out.update(activation_locks=receipts['residual']['activation_locks']))
        check(not t.hit.exists(),'HEIR trap executed')
        print(f'{len(t.positive)} accepted (including 4 frozen), {len(t.negative)} source/spec refusals, '
              f'{len(t.tampered)} source-bound tamper refusals; HEIR trap hits=0')
        if a.strace: print('exec audit: 4/4 had one execve (importer), zero additional execve/execveat')


if __name__ == '__main__':
    main()
