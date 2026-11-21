#!/usr/bin/env python3
"""Test-only exec tap: retain the exact stdin-file and argv handed to Pass 3."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == '__main__':
    real = Path(os.environ['SNABF_A2_REAL_OPT']).resolve(strict=True)
    argv = sys.argv[1:]
    options = [v.partition('=')[2] for v in argv if v.startswith('--snabf-pass3=')]
    if options:
        if len(options) != 1 or argv[-3:] != ['selected.mlir', '-o', 'proposal.mlir']:
            sys.exit('unexpected Pass 3 invocation in the frozen rejection probe')
        opts = dict(pair.split('=', 1) for pair in options[0].split())
        alpha, k = int(opts['gadget-alpha']), int(opts['max-k'])
        if alpha not in (1, 2) or k not in (0, 2):
            sys.exit('unexpected 17Q arm')
        out = Path(os.environ['SNABF_A2_CAPTURE']) / f'a{alpha}-k{k}'
        out.mkdir(exist_ok=False)  # no silent reuse or overwritten specializations
        for name in ('selected.mlir', 'profile.json'):
            shutil.copyfile(name, out/name)
        receipt = dict(argv=argv, native_sha256=sha(real), trace=os.getenv('SNABF_PASS3_TRACE_REJECTIONS'),
                       inputs={name: sha(out/name) for name in ('selected.mlir', 'profile.json')})
        (out/'call.json').write_text(json.dumps(receipt, sort_keys=True)+'\n')
    os.execv(real, [str(real), *argv])
