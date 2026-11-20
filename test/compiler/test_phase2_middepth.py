import copy
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
