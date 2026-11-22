"""Arithmetic and binding counterexamples for the public plaintext transfer."""
from fractions import Fraction as Q
import math
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src/tools'))
from b2i_noise_replay import DCEncoding, Diagonal, centered_linear, constant, transfers
from exact_scale_replay import encode, prime


def call(values, *, site='m', q=(193,257), p=(449,), n=32, slots=16, scale=1):
    ctx = dict(call_id='encode:'+site, site_id=site, operand_index=1, level=len(q)-1, slots=slots,
               key_id='k', ring_n=n, q=q, p=p, default_log_scale=39, dynamic_input=False, is_batched=True,
               arg_type='[]float64', arg_precision_bits=53, encoder_prec_bits=53)
    return encode(ctx, Q(scale), b''.join(struct.pack('>d', w) for w in values))


class PlainFamily(unittest.TestCase):
    def test_rounding_and_raw_limbs(self):
        for w, c, peak in [(0.5,1,None), (-0.5,-1,None), (2.5,3,None), (math.nextafter(1.5,0),1,None),
                           (193,193,193+10*193), (200,200,200+10*193), (-193,-193,None),
                           (-386,-386,193+10*193), (-0.25,0,257+10*257), (-0.,0,None)]:
            v = [w]*4
            self.assertEqual(constant(v, call(v), [193,257]), DCEncoding(c, c-Q(w), peak))

    def test_no_borrowed_bits_or_fft(self):
        v = [0.]*4; receipt = call(v)
        for change in [-0., math.nextafter(0.,1.)]:
            changed = v.copy(); changed[2] = change
            with self.assertRaisesRegex(ValueError, 'bound public encoding argument'):
                constant(changed, receipt, [193,257])
            self.assertEqual(constant(changed, call(changed), [193,257]), 'vector_ifft_not_proved')
        self.assertEqual(constant([100.], call([100.], q=(193,)), [193]), 'plaintext_centered_lift_unproved')
        self.assertEqual(constant([1.], call([1.], scale=2**52), [193,257]), 'unsupported_fixed_point_range')

    def test_word_limit(self):
        # A genuine NTT prime and a negative input rounding to zero: the fast
        # CRT writer supplies q, so an assumed canonical input would be wrong.
        n = 1024
        q = next(i for i in range(2**60-2*n+1, 2**59, -2*n) if prime(i))
        v = [-.25]*2
        self.assertEqual(constant(v, call(v, q=(q,), p=(12289,), n=n, slots=n//2), [q]), 'unproved_ntt_word_range')

    def test_diagonal_strict_lift_and_provenance(self):
        v = [97., -97.]*8
        result = constant(v, call(v), [193,257], mask=True)
        self.assertIsInstance(result, Diagonal)
        self.assertEqual((result.proof['period'], result.proof['coefficient_stride'], result.proof['centered_coeff_abs']), (2,8,'97'))
        self.assertEqual(constant(v, call(v, q=(193,)), [193], mask=True), 'plaintext_centered_lift_unproved')
        self.assertEqual(constant(v, call(v), [193,257]), 'vector_ifft_not_proved')
        with self.assertRaisesRegex(ValueError, 'bound basis, branch or physical scale'):
            constant(v, call(v), [257,193], mask=True)
        with self.assertRaisesRegex(ValueError, 'bound basis, branch or physical scale'):
            constant(v, dict(call(v), realized_eff_scale='2'), [193,257], mask=True)
        with self.assertRaisesRegex(ValueError, 'bound public encoding argument'):
            constant(v, dict(call(v), argument_sha256=None), [193,257], mask=True)
        long = [float(i) for i in range(128)]
        self.assertEqual(constant(long, call(long, q=(7681,), p=(12289,), n=256, slots=128), [7681], mask=True),
                         'vector_ifft_not_proved')

    def test_exact_products_and_unknown_parent(self):
        self.assertEqual(centered_linear(97, ['u','v'], [1,1], {'u':48,'v':1}), (49,False))
        self.assertEqual(centered_linear(97, ['u','u'], [1,-1], {'u':48}), (0,True))
        for sign in (-1,1):
            # Do not feed any proposed DC fact into replay. A uniform zero
            # multiplier manufactures a whole zero polynomial from a secret.
            ops = [('u','source',[],[]), ('z','mask',['u'],[0.,0.]), ('b','add_plain',['z'],[sign*96]*2),
                   ('wrap','mask',['b'],[5]*4), ('next','mask',['wrap'],[-1]*2), ('secret','mask',['u'],[5]*4),
                   ('unknown','mask',['b'],[5,4]), ('after','mask',['unknown'],[5]*4)]
            trace = SimpleNamespace(d={'q':[193]}, edges=[], calls=[])
            instructions, physical = [], []
            for name, op, parents, vals in ops:
                trace.edges.append(dict(node_id=name, op=op, inputs=parents, level=0))
                instructions.append(dict(ID=name, values=vals))
                if vals:
                    trace.calls.append(call(vals, site=name, q=(193,)))
                physical.append(dict(node_id=name, multiplier_a='1', multiplier_b='1', physical_scale_exact='1', ideal_abs_bound='96'))
            program = {'instructions':instructions, 'slots':16}
            rows = {r['site_id']:r for r in transfers(trace, program, physical, guarded_diagonals=True)['transfers']}
            r = rows['node/wrap/mul_plain']
            self.assertEqual((r['formula_ref'], r['dc_before_center'], r['dc_after_center'], r['dc_center_correction'], r['local_bias_units']),
                             ('b2i/exact-dc-recenter-v1', str(sign*480), str(sign*94), str(-sign*386), '386'))
            self.assertEqual(rows['node/next/mul_plain']['dc_after_center'], str(-sign*94))
            for node in ['secret','after']:
                unknown = rows[f'node/{node}/mul_plain']
                self.assertIsNone(unknown['dc_center_correction'])
                self.assertIsNone(unknown['local_bias_units'])
                self.assertEqual(unknown['blocker'], 'need_tighter_centered_decryption_coefficients')
            self.assertEqual(rows['node/unknown/mul_plain']['blocker'], 'need_tighter_centered_decryption_coefficients')
            with self.assertRaisesRegex(ValueError, 'bound encoding use'):
                transfers(trace, dict(program, slots=8), physical)


if __name__ == '__main__':
    unittest.main()
