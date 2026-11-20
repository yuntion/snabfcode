from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src/tools'))
from calibrate_manifest import generate


class ManifestTest(unittest.TestCase):
    def program(self):
        nodes=[dict(op='baby',Level=2,kernel_ring=16,inputs=[parent],rotation=rotation,route_site=site)
               for parent,rotation,site in ((0,1,0),(0,2,0),(4,1,0),(0,1,1))]
        return dict(epochs=[dict(n=16,q=[97,193,257],p=[353,449])],source_ring=16,target_ring=16,instructions=nodes)

    def test_group_once_and_preserve_pattern_order(self):
        g=self.program(); cases=generate([g,g],2,12)['cases']
        self.assertEqual([c['shifts'] for c in cases],[(1,),(1,2)])
        self.assertEqual(len(cases[1]['patterns']),9)
        self.assertIn((2,1),cases[1]['patterns'])

    def test_does_not_silently_omit_sites(self):
        g=self.program()
        with self.assertRaisesRegex(ValueError,'exceed'): generate([g],2,11)
        g['target_ring']=8
        with self.assertRaisesRegex(ValueError,'single-ring'): generate([g],2,12)


if __name__=='__main__': unittest.main()
