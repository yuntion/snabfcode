"""RFC-004 review oracle: exact slot arithmetic, not a CKKS implementation."""
from fractions import Fraction as F
from math import prod
import unittest


# The first three source Q in the frozen R3 residual fixture (level order).
Q = (16580609, 540311683073, 540311912449)


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def rn(x, bits=128):
    x = F(x)
    need(x > 0, 'nonpositive scale')
    k = x.numerator.bit_length() - x.denominator.bit_length()
    k -= x < F(2)**k
    ulp = F(2)**(k-bits+1)
    return round(x/ulp)*ulp


def check_align(mode, level, after, tail, pmeta, peff, branch, source=8, target=12):
    need(mode in (64,128), 'mode')
    k = 1 if mode == 64 else 2
    need(k <= level < len(Q) and after == level-k, 'level')
    need(tuple(tail) == tuple(Q[level-i] for i in range(k)), 'ordered Q suffix')
    need(branch in ('f64', 'value') and pmeta == rn(pmeta), 'encoder receipt')
    need(peff == (rn(pmeta,53) if branch == 'f64' else pmeta), 'effective scale')
    meta = rn(source*pmeta)
    for q in tail:
        meta = rn(meta/q)
    need(meta == target and F(source*peff, prod(tail)) == target, 'physical target')


def join(ma, mb, ea, eb, a, b, sign):
    s = max(ma, mb)
    m = int(rn(F(s, min(ma,mb))))
    ga, gb = (m,1) if ma < mb else (1,m)
    actual = F(ga*ea*a + sign*gb*eb*b, s)
    bound = F(abs(ga*ea-s)*abs(a) + abs(gb*eb-s)*abs(b), s)
    return actual, bound


class Review(unittest.TestCase):
    def test_non_dyadic_fold_is_not_exact(self):
        requested = F(97, 3)
        metadata = rn(requested)
        effective = rn(metadata, 53)
        self.assertEqual(len({requested, metadata, effective}), 3)
        self.assertNotEqual(3*effective/97 - 1, 0)
        self.assertNotEqual(3*effective/97 - 1, 3*metadata/97 - 1)

    def test_fold_direction_and_side_sink(self):
        q, rho = Q[2], F(3,2)
        check_align(64,2,1,[q],q*rho,q*rho,'f64')
        wrong = rn(q/rho,53)
        with self.assertRaisesRegex(ValueError, 'physical target'):
            check_align(64,2,1,[q],wrong,wrong,'f64')
        # Both encodings give 10 at the join, but the side sink must stay 6.
        def sinks(w,p):
            meta, raw = F(8*p,q), F(8*3*w*p,q)
            return join(meta,12,meta,12,raw/meta,4,1)[0], raw/meta
        self.assertEqual(sinks(2,q*rho), (10,6))
        self.assertEqual(sinks(2*rho,q), (10,9))
        with self.assertRaisesRegex(ValueError, 'side sink'):
            need(sinks(2*rho,q) == (10,6), 'side sink')

    def test_left_right_join_and_margin(self):
        a,b = 2**30,2**30+1
        self.assertLess(F(b-a,a), F(1,10**8))
        for sign in (1,-1):
            for ma,mb,x,y in ((a,b,1,0),(b,a,0,1)):
                got,loss = join(ma,mb,ma,mb,x,y,sign)
                self.assertEqual(abs(got-(x+sign*y)), F(1,b))
                self.assertEqual(loss,F(1,b))
                with self.assertRaisesRegex(ValueError, 'residual margin'):
                    need(loss <= F(1,2**32), 'residual margin')
        self.assertEqual(join(8,22,8,22,1,0,1)[0], F(16,22))
        self.assertEqual(join(12,12,11,12,1,0,1)[1], F(1,12))

    def test_two_primes_and_receipt(self):
        tail = Q[2],Q[1]
        p = F(3,2)*prod(tail)
        check_align(128,2,0,tail,p,p,'value')
        for level,after,declared in [(2,1,tail),(1,0,tail),(2,0,tail[:1]),
                                      (2,0,tail[::-1]),(2,0,(2**39,tail[1]))]:
            with self.subTest(level=level,after=after,tail=declared):
                with self.assertRaises(ValueError):
                    check_align(128,level,after,declared,p,p,'value')
        with self.assertRaisesRegex(ValueError, 'effective scale'):
            check_align(128,2,0,tail,p,p,'f64')


if __name__ == '__main__':
    unittest.main(verbosity=2)
