"""Run B2.1 tamper tests on a fresh single-ring direct package, never edit its inputs."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/tools'))
from replay_noise_audit import Attributes, Q, independent, load, need, pow2


def run(opt, checked, program, audit, legacy):
    expected = independent(checked, program, audit)
    checked_text = checked.read_text(); data = load(program); report = load(audit)
    # The first fractional plaintext is a real encoder call, not a test attribute.
    row = next(r for r in data['encoding_receipts']['receipts'] if Q(r['log_argument_exact']).denominator != 1)
    start = checked_text.index('snabf.encoding_receipts = ') + len('snabf.encoding_receipts = ')
    reader = Attributes(checked_text, start); reader.value(); end = reader.pos
    block = checked_text[start:end]
    pat = r'\{branch = [^{}]*?node_id = "' + re.escape(row['node_id']) + r'"[^{}]*?\}'
    match = re.search(pat, block); need(match, 'receipt row not found')
    a,b = match.span()
    without = block[:a] + block[b+2:] if block[b:b+2] == ', ' else block[:a-2] + block[b:]
    cases = {'missing_plain_binding': checked_text[:start] + without + checked_text[end:],
             'unbound_sealed': checked_text[:start-len('snabf.encoding_receipts = ')] + checked_text[end+2:]}
    sink = data['instructions'][data['sink']]; log = math.nextafter(sink['physical_log_scale'], math.inf)
    terminal = re.compile(r'(\{ID = "' + re.escape(sink['ID']) + r'",[^{}]*?physical_log_scale = )[\w.+-]+( : f64)')
    pos = checked_text.index('snabf.pass3 = ')
    changed,n = terminal.subn(lambda m: m[1] + '0x' + struct.pack('>d',log).hex().upper() + m[2], checked_text[pos:], count=1)
    need(n == 1, 'terminal binding not found'); cases['terminal_log_one_ulp'] = checked_text[:pos] + changed
    def refused(fn):
        try: fn()
        except ValueError: return
        raise AssertionError('independent verifier accepted mutation')
    with tempfile.TemporaryDirectory(prefix='snabf-b2-negative-') as tmp:
        t = Path(tmp)
        def native(ir, flag):
            dest = t/'should-not-publish.mlir'
            z = subprocess.run([opt, '--allow-unregistered-dialect','--mlir-print-op-on-diagnostic=false','--verify-each',flag,ir,'-o',dest],
                               capture_output=True,text=True,cwd=t)
            need(z.returncode > 0 and not dest.exists() and 'error:' in z.stderr, 'native tamper accepted or crashed: '+ir.name)
        for label, text in cases.items():
            ir = t/(label+'.mlir'); ir.write_text(text)
            refused(lambda: independent(ir,program,audit))
            for flag in ('--snabf-check-plan','--snabf-lower-ckks=direct-go=true'): native(ir,flag)
        if legacy:
            for flag in ('--snabf-check-plan','--snabf-lower-ckks=direct-go=true'): native(legacy,flag)
        def reject_program(fn):
            obj = copy.deepcopy(data); fn(obj); path=t/'params.json'; path.write_text(json.dumps(obj))
            refused(lambda: independent(checked,path,audit))
        reject_program(lambda v:v['encoding_receipts']['receipts'].pop(row['index'] if row['index']==1 else 0))
        reject_program(lambda v:v['instructions'][row['index']]['values'].__setitem__(0, 3.125))
        reject_program(lambda v:v['epochs'][0]['q'].reverse())
        def reject_audit(fn):
            obj = copy.deepcopy(report); fn(obj); path=t/'audit.json'; path.write_text(json.dumps(obj))
            refused(lambda: independent(checked,program,path))
        scale = Q(expected['sink_scale']); e = scale.numerator.bit_length() - scale.denominator.bit_length()
        ulp = pow2(e - int(scale < pow2(e)) - 127)
        reject_audit(lambda v:v['sinks'][0].update(physical_scale_exact=str(scale+ulp)))
        reject_audit(lambda v:v.update(probabilistic_cert_issued=True))
        reject_audit(lambda v:v['missing_contract_terms'].pop())
    return dict(matched=True, native_denials=2*(len(cases)+bool(legacy)), independent_denials=len(cases)+6)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('snabf-opt','checked','program','audit'): parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--legacy-checked',type=Path)
    args=parser.parse_args()
    print(json.dumps(run(args.snabf_opt.resolve(),args.checked.resolve(),args.program.resolve(),args.audit.resolve(),
                         args.legacy_checked.resolve() if args.legacy_checked else None)))
