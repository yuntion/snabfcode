"""N1 isolated-package gate; the small real CKKS circuit is *not* a four-graph run."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src/tools'))
from native_package import digest, publish_native_package


class PackageTest(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(prefix='n1-pkg-test-')
        self.addCleanup(t.cleanup)
        self.root = Path(t.name)
        self.src = self.root/'emitted'
        self.out = self.root/'released'
        shutil.copytree(ROOT/'test/compiler/rfc004/testdata/n1_package', self.src)
        self.seal = 'a'*64
        self.plan = self.src/'program.json'
        self.plan.write_text(json.dumps({'emit_sha256': self.seal}), encoding='ascii')

    def test_whole_package(self):
        r = publish_native_package(self.src, self.out, self.seal)
        self.assertEqual(r['program_file_sha256'], digest(self.plan.read_bytes()))
        self.assertEqual((self.out/'planned/program.json').read_bytes(), self.plan.read_bytes())
        self.assertEqual(r['emit_sha256'], self.seal)
        self.assertIn('github.com/tuneinsight/lattigo/v6/schemes/ckks', r['dependencies'])
        self.assertFalse(any('heirckks' in x for x in r['dependencies']))
        with self.assertRaisesRegex(ValueError, 'OUTPUT_EXISTS'):
            publish_native_package(self.src, self.out, self.seal)

    def denied(self, why):
        with self.assertRaisesRegex(ValueError, why):
            publish_native_package(self.src, self.out, self.seal)
        self.assertFalse(self.out.exists())
        self.assertFalse(list(self.root.glob('.snabf-native-*')))

    def test_missing_file(self):
        (self.src/'native_runtime.go').unlink()
        self.denied('incomplete')

    def test_observer(self):
        import native_observations as obs
        import verify_n1_equivalence as proof
        source = self.root/'fixture.txt'
        source.write_bytes(b'fixture: two consumers of x*x')
        pins = {'test_shared_square':digest(source.read_bytes())}
        self.plan.write_text(json.dumps(dict(emit_sha256=self.seal,frontend_provenance=dict(
            mode='native-linalg',source_sha256=pins['test_shared_square']))))
        publish_native_package(self.src,self.out,self.seal)
        rows = [dict(x=[i/128], truth=[2*(i/128)**2]) for i in range(64)]
        def run(label, scale):
            return obs.observe('test_shared_square',source,self.out,self.root/label,
                               seal=self.seal,ring=1024,level=2,slots=512,scale=str(scale))
        # Only the test's tiny graph oracle is substituted. Key generation, encryption,
        # evaluation, decoding, ABI and exact-scale checks run in the real executable.
        with patch.dict(proof.SOURCE,pins), patch.object(proof,'corpus',return_value=rows), patch.object(obs,'corpus',return_value=rows):
            r = run('observed',2**70)
            self.assertEqual(r['slots_checked'],64*512)
            self.assertTrue((self.root/'observed/run_observations.json').is_file())
            self.assertEqual((self.out/'planned/program.json').read_bytes(),self.plan.read_bytes())
            with self.assertRaisesRegex(ValueError,'physical metadata'):
                run('wrong-scale',2**70+1)
            self.assertFalse((self.root/'wrong-scale').exists())
            source.write_bytes(b'different original source')
            with patch.dict(proof.SOURCE,{'test_shared_square':digest(source.read_bytes())}), self.assertRaisesRegex(ValueError,'provenance'):
                run('wrong-source',2**70)
            self.assertFalse((self.root/'wrong-source').exists())

    def test_mismatched_seal(self):
        self.plan.write_text('{"emit_sha256":"'+('b'*64)+'"}')
        self.denied('seal mismatch')

    def test_duplicate_seal(self):
        self.plan.write_text('{"emit_sha256":"wrong","emit_sha256":"'+self.seal+'"}')
        self.denied('duplicate')

    def test_go_seal_not_a_comment(self):
        p = self.src/'model.go'
        p.write_text(p.read_text().replace(self.seal, 'b'*64)+'\n// const PlanSHA256 = "'+self.seal+'"\n')
        self.denied('plan/Go seal mismatch')

    def test_static_helper(self):
        src = ROOT/'src/backend/ringswitch/epochs/heirckks/standard.go'
        (self.src/'native_runtime.go').write_bytes(src.read_bytes().replace(b'package heirckks', b'package planned'))
        self.denied('static HEIR helper')

    def test_importing_helper(self):
        with (self.src/'bad_test.go').open('x') as f:
            f.write('package planned\nimport _ "snabf.local/compiler/src/backend/ringswitch/epochs/heirckks"\n')
        self.denied('heirckks')

    def test_detect_plan_write(self):
        with (self.src/'write_test.go').open('x') as f:
            f.write('package planned\nimport ("os"; "testing")\nfunc TestOverwrite(t *testing.T) {\n'
                    'if e:=os.WriteFile("program.json", []byte("changed"), 0600); e != nil { t.Fatal(e) }\n}\n')
        self.denied('mutated release files')

    def test_aliasing_product(self):
        p = self.src/'native_runtime.go'
        p.write_text(p.read_text().replace('return e.RelinearizeNew(x)', 'return x, e.Relinearize(x, x)'))
        self.denied('go test')


if __name__ == '__main__':
    unittest.main()
