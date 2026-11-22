"""Fast failure contracts. Run on Linux with python3 test/compiler/test_phase2_entry.py."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).resolve().parents[2]/'src/tools'
sys.path.insert(0,str(TOOLS))
from security_cache import checked_attacks, digest, measure, request
from snabf_compile import main, scale_search_mode


class EntryTests(unittest.TestCase):
    def test_frontend_dispatch_precedes_tools_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'absent'/'model'
            common = ['compile','--mlir','missing.mlir','--input-bound','1','--out-dir',str(dest),'--security','192']
            for args, reason in (([], 'SNABF_UNSUPPORTED_SECURITY_TARGET'),
                                 (['--frontend','legacy-heir'], 'SNABF_UNSUPPORTED_SECURITY_TARGET'),
                                 (['--frontend','native-linalg'], 'SNABF_NATIVE_PROFILE')):
                err = io.StringIO()
                with patch.object(sys,'argv',common+args), contextlib.redirect_stderr(err), \
                     patch('subprocess.Popen',side_effect=AssertionError('unexpected subprocess')), \
                     patch('shutil.which',side_effect=AssertionError('unexpected discovery')), \
                     patch('tempfile.TemporaryDirectory',side_effect=AssertionError('unexpected staging')), \
                     self.assertRaises(SystemExit) as ex:
                    main()
                self.assertEqual(ex.exception.code,1)
                self.assertIn(reason,err.getvalue())
                self.assertFalse(dest.parent.exists())

    def test_explicit_narrow_does_not_hide_a_conflicting_legacy_request(self):
        self.assertEqual(scale_search_mode(None,False),'narrow')
        self.assertEqual(scale_search_mode('narrow',False),'narrow')
        self.assertEqual(scale_search_mode(None,True),'heir-anchor')
        self.assertEqual(scale_search_mode('heir-anchor',False),'heir-anchor')
        with self.assertRaisesRegex(ValueError,'conflicts'):
            scale_search_mode('narrow',True)

    def test_no_silent_downgrade(self):
        for opts,reason in ((['--security','192'],'SNABF_UNSUPPORTED_SECURITY_TARGET'),
                            (['--certify-failure-bits','40'],'SNABF_PROBABILISTIC_CERT_UNAVAILABLE')):
            with tempfile.TemporaryDirectory() as d:
                out=Path(d)/'result'; stderr=io.StringIO()
                with patch.object(sys,'argv',['compile','--mlir','absent','--input-bound','1','--out-dir',str(out),*opts]), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as ex:
                    main()
                self.assertNotEqual(ex.exception.code,0); self.assertIn(reason,stderr.getvalue()); self.assertFalse(out.exists())

    def test_concrete_cache_identity(self):
        a=request('source.QP',8,[97],[113],97*113)
        b=request('source.QP',8,[97],[101],97*101)
        self.assertEqual(int(a['modulus']).bit_length(),int(b['modulus']).bit_length())
        for other in [b, dict(a,role='target.QP'),dict(a,ring=16),dict(a,model='other'),dict(a,p=a['q'],q=a['p'])]:
            self.assertNotEqual(digest(a),digest(other))

    def test_no_finite_attack_is_not_admission(self):
        costs={'usvp':'inf','dual_hybrid':184.}
        self.assertEqual(checked_attacks(costs,{'role':'target.Q'}),costs)
        with self.assertRaises(ValueError): checked_attacks(costs,{'role':'source.Q'})
        with self.assertRaises(ValueError): checked_attacks(dict(costs,usvp=float('nan')),{'role':'target.Q'})

    @unittest.skipUnless(os.name=='posix','POSIX private estimator cache')
    def test_miss_and_corrupt_cache_fail_closed(self):
        a=request('source.QP',8,[97],[113],97*113)
        b=request('source.QP',8,[97],[101],97*101)
        with tempfile.TemporaryDirectory() as d, patch('shutil.which',return_value=None):
            p=Path(d); key=digest(a); f=p/key
            with self.assertRaisesRegex(ValueError,'SECURITY_ORACLE_UNAVAILABLE'): measure([a],p,None)
            costs={'usvp':150.,'dual_hybrid':145.}
            f.write_text(json.dumps(dict(request=a,attacks=costs,checksum=digest([a,costs])))); f.chmod(0o600)
            self.assertEqual(measure([a],p,None)[key],costs)
            with self.assertRaisesRegex(ValueError,'SECURITY_ORACLE_UNAVAILABLE'): measure([b],p,None)
            f.write_text(json.dumps(dict(request=a,attacks=dict(costs,usvp=999.),checksum=digest([a,costs]))))
            with self.assertRaisesRegex(ValueError,'INVALID_ESTIMATE'): measure([a],p,None)
            p.chmod(0o755)
            with self.assertRaisesRegex(ValueError,'UNTRUSTED_CACHE'): measure([a],p,None)


if __name__=='__main__': unittest.main()
