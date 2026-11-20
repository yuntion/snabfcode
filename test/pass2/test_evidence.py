"""Verdict semantics; native replay tests only ever mutate temporary copies."""

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from snabf_frontend.pass2_evidence import (Oracle, TOLERANCE, blocking_precision,
                                           f32, load, reduce_arm, save, suite_precision)


class TestGatePolicy(unittest.TestCase):
    def test_negative_comparator_is_not_a_pass(self):
        names = {'residual': ('heir', 'economy', 'precision'), 'control': ('heir',),
                 'strided64': ('heir',), 'reuse64': ('heir', 'rotation_cap'),
                 'composite_coverage': ('heir', 'free', 'forced_composite', 'forced_composite_tight')}
        reports = {w: {'arms': {a: {'precision_passed': True, 'hard_precision_gate': blocking_precision(w, a)}
                                for a in arms}} for w, arms in names.items()}
        self.assertTrue(suite_precision(reports)['all_arms_meet_tolerance'])
        reports['residual']['arms']['heir']['precision_passed'] = False
        self.assertEqual(suite_precision(reports), {'all_arms_meet_tolerance': False, 'regression_gates_passed': True,
                         'numerical_failures': ['residual/heir'], 'blocking_numerical_failures': []})
        for w, arms in names.items():
            for a in arms:
                if (w, a) == ('residual', 'heir'):
                    continue
                with self.subTest(work=w, arm=a):
                    s = reports[w]['arms'][a]
                    s['precision_passed'] = False
                    verdict = suite_precision(reports)
                    self.assertFalse(verdict['regression_gates_passed'])
                    self.assertEqual(verdict['blocking_numerical_failures'], [f'{w}/{a}'])
                    s['hard_precision_gate'] = False
                    with self.assertRaisesRegex(ValueError, 'invalid numerical gate'):
                        suite_precision(reports)
                    s.update(precision_passed=True, hard_precision_gate=True)


@unittest.skipUnless(os.getenv('SNABF_P2_TEST_RUN'), 'requires completed encrypted residual measurements')
class TestMeasuredReplay(unittest.TestCase):
    def setUp(self):
        self.src = Path(os.environ['SNABF_P2_TEST_RUN']) / 'residual'
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.d = Path(scratch.name) / 'residual'
        self.d.mkdir()
        for filename in ('profile.json', 'original-ckks.mlir', 'corpus.json'):
            shutil.copyfile(self.src / filename, self.d / filename)
        self.corpus = load(self.d / 'corpus.json')
        self.oracle = Oracle((self.src / 'input.mlir').read_text())
        self.assertEqual(self.corpus, self.oracle.corpus())

    def arm_copy(self, name):
        arm = self.d / name
        arm.mkdir()
        files = ['model.go', 'runtime.json', 'trials.jsonl', 'abi.json', 'accuracy.json']
        if name != 'heir':
            files.append('audit.json')
        for filename in files:
            shutil.copyfile(self.src / name / filename, arm / filename)
        return arm

    @staticmethod
    def rows(arm):
        return [json.loads(line) for line in (arm / 'trials.jsonl').read_text().splitlines()]

    @staticmethod
    def write_rows(arm, rows):
        (arm / 'trials.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def inject_misses(self, arm):
        # Keep the actual compiled program, audit, corpus and all other real
        # observations. These two deliberately invalid samples are never exported.
        rows = self.rows(arm)
        for i, j in ((0, 0), (-1, -1)):
            rows[i]['decrypted'][j] = f32(self.corpus[i]['truth'][j] + 4 * TOLERANCE)
        self.write_rows(arm, rows)
        errors = [abs(v - q) for row, case in zip(rows, self.corpus, strict=True)
                  for v, q in zip(row['decrypted'], case['truth'], strict=True)]
        over = sum(v > TOLERANCE for v in errors)
        self.assertGreaterEqual(over, 2)
        total = math.fsum(v*v for v in errors)
        check = load(arm / 'accuracy.json')
        check.update(accuracy_status='FAIL', precision_passed=False, slots_over_tolerance=over,
                     linf=max(errors), mse=total/len(errors), sum_squared_error=total,
                     first_failure={'trial': 0, 'slot': 0, 'abs_error': errors[0],
                                    'decrypted': rows[0]['decrypted'][0], 'truth': self.corpus[0]['truth'][0]})
        save(arm / 'accuracy.json', check)
        return over

    def test_same_misses_exempt_only_for_unmodified_stock_arm(self):
        stock, planned = self.arm_copy('heir'), self.arm_copy('economy')
        stock_n = self.inject_misses(stock)
        baseline, s = reduce_arm(stock, self.oracle, self.corpus)
        self.assertEqual((s['accuracy_status'], s['slots_over_tolerance'], s['logical_slots_checked']), ('FAIL', stock_n, 4096))
        planned_n = self.inject_misses(planned)
        with self.assertRaisesRegex(ValueError, f'{planned_n}/4096 slots exceed'):
            reduce_arm(planned, self.oracle, self.corpus, baseline)
        p = load(planned / 'metrics.json')
        self.assertEqual((p['accuracy_status'], p['slots_over_tolerance'], p['logical_slots_checked']), ('FAIL', planned_n, 4096))
        verdict = suite_precision({'residual': {'arms': {'heir': s, 'economy': p}}})
        self.assertEqual(verdict['blocking_numerical_failures'], ['residual/economy'])
        self.assertFalse(verdict['regression_gates_passed'])

    def test_negative_comparator_still_rejects_incomplete_or_invalid_evidence(self):
        arm = self.arm_copy('heir')
        for mutation, error in [('nonfinite', 'nonfinite output'), ('width', 'packing mismatch'),
                                ('level', 'actual input'), ('scale', 'actual output'), ('missing', 'expected 64 trials')]:
            with self.subTest(mutation=mutation):
                rows = self.rows(self.src / 'heir')
                if mutation == 'nonfinite':
                    rows[0]['decrypted'][0] = float('nan')
                elif mutation == 'width':
                    rows[0]['decrypted'].pop()
                elif mutation == 'level':
                    rows[0]['input_level'] = len(load(self.d / 'profile.json')['Q']) - 1
                elif mutation == 'scale':
                    rows[0]['output_log_scale'] += 1
                else:
                    rows.pop()
                self.write_rows(arm, rows)
                with self.assertRaisesRegex(ValueError, error):
                    reduce_arm(arm, self.oracle, self.corpus)
        self.write_rows(arm, self.rows(self.src / 'heir'))
        runtime = load(arm / 'runtime.json')
        runtime['Q'] = runtime['Q'][:-1]
        save(arm / 'runtime.json', runtime)
        with self.assertRaisesRegex(ValueError, 'runtime parameters'):
            reduce_arm(arm, self.oracle, self.corpus)


if __name__ == '__main__':
    unittest.main()
