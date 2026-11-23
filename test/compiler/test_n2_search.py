"""No estimator or secret keys: exercise fail-closed routing, not security values."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src/tools/n2'))
import search as n2


def costs(requests, score):
    return {n2.digest(r):{'usvp':score(r)+2,'dual_hybrid':score(r)} for r in requests}


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.binding = dict(source='1'*64, input_spec='2'*64, ring=16384, encoded_slots=4096,
                            q_count=2, alpha=1, min_scale=36, max_scale=44, security_cap=352, scale_mode='wide')
        def primes(bits,count):
            result=[]; candidate=(1<<bits)+1
            while len(result)<count:
                if n2.ntt_prime(candidate,16384): result.append(candidate)
                candidate+=32768
            return result
        self.plan = dict(self.binding,schema='snabf.n2.native-proposal.v1',graph_sha256='3'*64,
                         q=primes(38,2),p=primes(42,1),scale=38,heir_scale_anchor=False,max_k=0,
                         wide_prime_bits=[24,59],margin_bits=2,k=[0],edges=1,arithmetic=2**-14,
                         approximation=0,tolerance=2**-12)

    def test_cap_only_from_ring_target_and_actual_probe(self):
        seen=[]
        def oracle(rs):
            seen.extend(rs)
            return costs(rs,lambda r:128 if int(r['modulus']) <= 1<<375 else 127.999)
        env=n2.search_envelope(16384,128,oracle)
        self.assertEqual(env['cap_bits'],352)
        self.assertTrue(all(r['ring']==16384 and r['q']==r['p']==[] for r in seen))
        self.assertIn(str(1<<352),[r['modulus'] for r in seen])

    def test_no_safe_or_invalid_envelope(self):
        for n,lam in [(16384,128),(8192,128),(16384,127),(16384,128.0)]:
            with self.subTest(n=n,lam=lam),self.assertRaises(ValueError):
                n2.search_envelope(n,lam,lambda rs:costs(rs,lambda r:127))
        with self.assertRaises(ValueError):
            n2.search_envelope(16384,128,lambda rs:costs(rs,lambda r:float('nan')))

    def test_exact_instances_and_reordered_basis(self):
        rs=[]
        def oracle(rows):
            rs.extend(rows); return costs(rows,lambda r:129)
        proof=n2.concrete_screen(self.plan,self.binding,oracle)
        self.assertTrue(proof['meets_128'])
        q=self.plan['q'][0]*self.plan['q'][1]
        self.assertEqual([int(r['modulus']) for r in rs],[q,q*self.plan['p'][0],1<<352])
        self.assertTrue(all(r['q']==self.plan['q'] and r['p']==self.plan['p'] for r in rs))
        old={n2.digest(r) for r in rs}; rs.clear()
        self.plan['q'].reverse()
        n2.concrete_screen(self.plan,self.binding,oracle)
        self.assertTrue(old.isdisjoint(n2.digest(r) for r in rs))

    def test_safe_envelope_does_not_grant_concrete_qp(self):
        screen=n2.concrete_screen(self.plan,self.binding,
            lambda rs:costs(rs,lambda r:127.99 if r['role']=='source.QP' else 140))
        self.assertFalse(screen['meets_128'])
        self.assertEqual(screen['minimum_bits'],127.99)

    def test_bad_bindings_and_composite_before_estimator(self):
        for changes in [dict(q=[32769,786433]),dict(q=[65537,65537]),dict(p=[65537]),
                        dict(security_cap=384),dict(ring=32768),dict(alpha=2),dict(heir_scale_anchor=True),
                        dict(arithmetic=2**-11),dict(k=[1]),dict(source='4'*64)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                n2.concrete_screen(dict(self.plan,**changes),self.binding,
                                   lambda _:self.fail('invalid basis reached estimator'))
        plan=copy.deepcopy(self.plan); plan['security_cap']=32
        with self.assertRaisesRegex(ValueError,'exceeds measured envelope'):
            n2.check_proposal(plan,dict(self.binding,security_cap=32))


if __name__=='__main__':
    unittest.main()
