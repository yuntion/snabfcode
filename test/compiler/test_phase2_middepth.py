import copy
import json
import unittest

import phase2_multip_middepth as mid


def graph():
    # Two parallel rescales, then a join with a dropped source, then a rescale.
    # Summing all rescale nodes would report three instead of two.
    rows = [('source',3,[]), ('rescale',2,[0]), ('rescale',2,[0]), ('add',2,[1,2]),
            ('drop',2,[0]), ('add',2,[3,4]), ('rescale',1,[5])]
    return dict(epochs=[{'q':[17,97,113,193]}],sink=6,
                instructions=[dict(op=op,Level=l,inputs=parents) for op,l,parents in rows])


class MiddepthTests(unittest.TestCase):
    def test_replay_one_rejected_candidate_not_the_entire_arm(self):
        # alpha=2, short last original Q block. Both product fields exceed 2^53.
        q, p = [7340033, 23068673, 36028797019488257], [469762049, 167772161]
        row = dict(schema=mid.REJECTION, stage='concrete_replay', status='rejected',
                   reason='aux_block_margin', n='1024', q_count='3', alpha='2', max_k='0', round='1',
                   scale_bits='32', tolerance='0.000244140625', security_cap_bits='350',
                   basis_source='concrete_replay', q=list(map(str,q)), p=list(map(str,p)),
                   witness=dict(block_first='2', block_end_exclusive='3', margin_bits='2',
                                p_product=str(p[0]*p[1]), required_product=str(4*q[2])))
        def check(*rows):
            lines = ['Pass 3 infeasible under requested scale, precision and QP bounds\n']
            lines += [json.dumps(r, sort_keys=True) + '\n' for r in rows]
            return mid.rejection_records(lines, n=1024, q_count=3, alpha=2, max_k=0)
        unknown = {k:v for k,v in row.items() if k not in ('basis_source','q','p','witness')}
        unknown.update(stage='solver', status='infeasible', reason='unknown', round='2')
        self.assertEqual(check(row, unknown), [row, unknown])
        self.assertEqual(check(), [])  # legacy log, no retroactive 17Q attribution
        with self.assertRaisesRegex(ValueError, 'wrong rejection basis source'):
            check(dict(unknown, basis_source='concrete_replay', q=row['q'], p=row['p']))

        # These squares pass the NTT congruence and the strict, independently
        # recomputed inequality. Only primality invalidates each claimed basis.
        for which, i, composite in [('q', 0, 12289**2), ('q', 2, 998244353**2), ('p', 0, 12289**2)]:
            forged = copy.deepcopy(row)
            forged[which][i] = str(composite)
            actual = int(forged['p'][0]) * int(forged['p'][1])
            required = 4 * int(forged['q'][2])
            forged['witness'].update(p_product=str(actual), required_product=str(required))
            with self.subTest(modulus=which, index=i):
                self.assertEqual(composite % (2*1024), 1)
                self.assertLess(composite, 1 << 60)
                self.assertLess(actual, required)
                with self.assertRaisesRegex(ValueError, 'composite rejection modulus'):
                    check(forged)

        for edit in (dict(stage='solver',status='infeasible'), dict(reason='unknown'), dict(max_k='2'),
                     dict(q=list(reversed(row['q']))), dict(basis_source='profile_input'),
                     dict(witness=dict(row['witness'], required_product=str(4*q[2]+1))),
                     dict(witness=dict(row['witness'], margin_bits='3',required_product=str(8*q[2]))),
                     dict(witness=dict(row['witness'], block_first='1')),
                     dict(witness=dict(row['witness'], block_first='0', block_end_exclusive='2',
                                       required_product=str(4*q[0]*q[1])))):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                check(dict(row, **edit))

    def test_deterministic_u64_primality_boundaries(self):
        for n in (2, 65537, (1 << 64) - 59):
            self.assertTrue(mid.prime_u64(n), n)
        for n in (0, 1, 341550071728321, (1 << 64) - 1, 1 << 64, (1 << 64) + 13):
            self.assertFalse(mid.prime_u64(n), n)

    def test_critical_path_uses_real_levels(self):
        out = mid.depth(graph())
        self.assertEqual((out['critical_rescales'],out['consumed_levels'],out['max_path_drop_limbs']), (2,2,1))
        g=graph(); g['sink']=4
        self.assertEqual((mid.depth(g)['critical_rescales'],mid.depth(g)['consumed_levels']), (0,1))

    def test_invalid_transitions_are_not_measurements(self):
        for i, change in ((3,dict(Level=1)),(4,dict(Level=3)),(6,dict(op='bootstrap')),
                          (2,dict(inputs=[6])),(6,dict(inputs=[5,4])),(0,dict(Level=1))):
            with self.subTest(i=i,change=change):
                g=copy.deepcopy(graph()); g['instructions'][i].update(change)
                with self.assertRaises(ValueError): mid.depth(g)

    def test_source_oracle_is_independent_of_candidate_params(self):
        first=mid.generate(4,11,13,2)
        self.assertEqual(first,mid.generate(4,11,13,2))
        self.assertNotEqual(first[2]['source_sha256'],mid.generate(4,12,13,2)[2]['source_sha256'])
        self.assertEqual((first[2]['width'],first[2]['linear_layers'],first[2]['silu']), (64,4,3))
        with self.assertRaises(ValueError): mid.generate(12,11,13,2)
        smaller=mid.generate(4,11,13,2,0.5)
        self.assertEqual(first[0],smaller[0]); self.assertNotEqual(first[1],smaller[1]); self.assertEqual(smaller[2]['input_bound'],0.5)


if __name__ == '__main__':
    unittest.main()
