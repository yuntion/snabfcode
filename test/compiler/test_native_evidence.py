import json
from pathlib import Path
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/tools'))
import verify_n1_equivalence as n1


class EvidenceTests(unittest.TestCase):
    def test_changed_source(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'source.mlir'
            p.write_bytes(b'not the frozen input\r\n')
            with self.assertRaisesRegex(ValueError, 'original source bytes'):
                n1.source_digest('reuse64', p)

    @unittest.skipUnless(sys.platform=='linux' and shutil.which('strace'), 'Linux strace required')
    def test_grandchild_absolute_path(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            child = d/'heir-opt'
            child.symlink_to('/usr/bin/true')
            for leaf in ('/usr/bin/true', str(child)):
                # Absolute path in a grandchild: trapping names on PATH alone misses it.
                inner = 'import subprocess; subprocess.check_call(['+repr(leaf)+'])'
                outer = 'import subprocess,sys; subprocess.check_call([sys.executable,"-c",'+repr(inner)+'])'
                argv = [sys.executable,'-c',outer]
                if leaf == str(child):
                    with self.assertRaisesRegex(ValueError, 'forbidden HEIR'):
                        n1.audit_exec(argv, d)
                else:
                    p, r = n1.audit_exec(argv, d)
                    self.assertEqual(p.returncode, 0)
                    self.assertGreaterEqual(r['attempts'], 3)

    def test_slot_gate_is_strict(self):
        rows = [dict(x=[1],truth=[0.0]) for _ in range(64)]
        with tempfile.TemporaryDirectory() as t, patch.object(n1,'corpus',return_value=rows):
            p = Path(t)/'trials.jsonl'
            opts = dict(seal='c'*64, ring=8192, level=1, slots=2)
            obs = [dict(trial=i, emit_sha256=opts['seal'], source_sha256=n1.SOURCE['residual'],
                        ring=8192, level=1, decoded=[0.0,0.0], x=[1]) for i in range(64)]
            def replay():
                p.write_text(''.join(json.dumps(r)+'\n' for r in obs))
                return n1.replay_slots('residual',None,p,**opts)
            self.assertEqual(replay()['slots_checked'],128)
            obs[-1]['decoded'][-1] = 2**-12
            with self.assertRaisesRegex(ValueError,'slot 1'):
                replay()
            obs.pop()
            with self.assertRaises(ValueError):
                replay()


if __name__=='__main__': unittest.main()
