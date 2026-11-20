"""Fast oracle/protocol tests; these are not CKKS evidence."""

import json
import math
from pathlib import Path
import unittest

from snabf_frontend.pass2_evidence import Oracle, TOLERANCE, TRIALS, eval_dag, f32
from snabf_frontend.pass2_models import COVERAGE, WORKLOADS, source_model


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'test/fixtures'


class TestSourceOracle(unittest.TestCase):
    def test_frozen_inputs(self):
        for work in (*WORKLOADS, COVERAGE):
            with self.subTest(work=work):
                o = Oracle(source_model(FIXTURES, work))
                cases = o.corpus()
                self.assertEqual(len(cases), TRIALS)
                self.assertEqual(len({tuple(c['x']) for c in cases}), TRIALS)
                self.assertTrue(all(f32(x) == x and -1 <= x <= 1 for c in cases for x in c['x']))
                self.assertEqual(cases, o.corpus())
        self.assertEqual(TRIALS, 64)
        self.assertEqual(TOLERANCE, 2 ** -12)

    def test_control_truth(self):
        o = Oracle(source_model(FIXTURES, 'control'))
        case = o.corpus()[0]
        self.assertEqual(case['x'], [-1] * 4)
        self.assertEqual(case['hidden'], [-.375, -.75, -.375, -.5])
        a = [v / (1 + math.exp(-v)) for v in case['hidden']]
        self.assertEqual(case['truth'][0], math.fsum([.25 * a[0], .5 * a[1], -.125 * a[3]]))

    def test_residual_truth_and_swapped_diagonals(self):
        raw = source_model(FIXTURES, 'residual')
        o = Oracle(raw)
        self.assertTrue(o.residual)
        for layer, expected in zip(o.w, [{0, 6, 8}, {0, 7, 9}]):
            self.assertEqual({(j-i) % o.width for i, row in enumerate(layer) for j, v in enumerate(row) if v}, expected)
        case = o.corpus()[-1]
        h = case['hidden']
        a = [x / (1 + math.exp(-x)) for x in h]
        skip = o.matvec(o.w[1], h)
        alone = o.matvec(o.w[1], a)
        self.assertGreater(max(map(abs, skip)), TOLERANCE)
        self.assertLess(max(abs(v - l - s) for v, l, s in zip(case['truth'], alone, skip)), 1e-14)

    def test_dag_replay_uses_output_and_edges(self):
        c = {'output': 2, 'nodes': [{'op': 'input', 'args': []}, {'op': 'mul', 'args': [0, 0]},
                                    {'op': 'linear', 'args': [0, 1], 'weights': [2, -1], 'constant': 1}]}
        self.assertEqual(eval_dag(c, .5), 1.75)
        c['output'] = 1
        self.assertEqual(eval_dag(c, .5), .25)
        c['nodes'][1]['args'] = [1, 0]
        with self.assertRaisesRegex(ValueError, 'non-topological'):
            eval_dag(c, .5)

    def test_longer_profile(self):
        base = json.loads((FIXTURES / 'reference_profile.json').read_text())
        extra = json.loads((ROOT / 'test/pass2/extended_profile.json').read_text())
        self.assertEqual({k: v for k, v in base.items() if k != 'Q'}, {k: v for k, v in extra.items() if k != 'Q'})
        self.assertEqual(extra['Q'][:len(base['Q'])], base['Q'])
        self.assertEqual(len(set(extra['Q'] + extra['P'])), len(extra['Q']) + len(extra['P']))
        for q in extra['Q'][len(base['Q']):]:
            self.assertEqual(q % (2 << base['logN']), 1)
            self.assertTrue(all(q % k for k in range(2, math.isqrt(q) + 1)))


if __name__ == '__main__':
    unittest.main()
