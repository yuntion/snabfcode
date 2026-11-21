import copy
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path

import pass3_isolation as iso
import phase2_multip_middepth as mid
from phase2_17q_probe import frozen_inventory


def model():
    # QP row x+y <= 4; exactly at the column minimum. Second row -2*x+y <= -8.
    return dict(stage='solver', status='unknown', reason='unknown', round='1', security_cap_bits='4',
                isolation=dict(kind='individual_lp_rows_v1', highs_model_status='Unknown',
                    num_cols='2', num_rows='2', qp_cap_row='0', columns=[['1','2'],['3','inf']],
                    rows=[['-inf','4','0','1','1','1'], ['-inf','-8','0','-2','1','1']]))


class IsolationTest(unittest.TestCase):
    def test_frozen_candidate_inventory(self):
        cells = [(a,k,d) for a in (1,2) for k in (0,2) for d in range(36,47)]
        frozen_inventory(cells)
        for bad in (cells[:-1], cells+[cells[-1]], [cells[1],*cells[1:]], [*cells[:-1],(2,2,47)]):
            with self.assertRaisesRegex(ValueError,'44 candidates'): frozen_inventory(bad)

    def test_signed_unbounded_rows_only(self):
        r = model()
        x = iso.isolate(r)
        self.assertEqual(x['conflicts'], [dict(row=1,side='upper',extremum='-1',bound='-8')])
        # Exact contact and individually satisfiable rows: still Unknown, not feasible.
        r['isolation']['rows'][1][1] = '-1'
        self.assertFalse(iso.isolate(r)['excluded'])
        r['isolation']['rows'][1] = ['2','inf','0','-1']
        self.assertEqual(iso.isolate(r)['conflicts'], [dict(row=1,side='lower',extremum='-1',bound='2')])

    def test_exact_sum_beyond_binary64_accumulation(self):
        r = model(); t = r['isolation']
        t['columns'] = [['9007199254740992','9007199254740992'], ['1','1']]
        r['security_cap_bits'] = t['rows'][0][1] = '9007199254740992'
        self.assertEqual(iso.isolate(r)['conflicts'][0]['extremum'], '9007199254740993')

    def test_malformed_model_or_forged_status(self):
        for field, bad in [('num_cols','3'),('qp_cap_row','2'),('highs_model_status','Optimal'),
                           ('rows',[['-inf','4','0','1','0','1'],[]]),('columns',[['1','nan'],['2','3']])]:
            r = model(); r['isolation'][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError): iso.isolate(r)
        r = model(); r['stage'] = 'concrete_replay'
        with self.assertRaises(ValueError): iso.isolate(r)
        self.assertIsNone(iso.isolate({'reason':'unknown'}))

    def test_preflight_logarithm_and_deletion(self):
        r = dict(stage='model_bounds', status='base_width', reason='unknown', round='0', scale_bits='40', q_count='1',
                 isolation=dict(kind='base_width_v1', guard_bits='1', min_base='20', exclusive_ceiling='59',
                    builder_lower='60.75', edges=[dict(id='x',op='source',level='0',magnitude='1048576')]))
        x = iso.isolate(r); self.assertTrue(x['rows'][0]['excludes_alone'])
        self.assertEqual(x['rows'][0]['lower_without_this_edge'],20)
        # Python/libm rounds log2(nextafter(2^20,-inf)) to 20. Exact floor is 19.
        r['isolation']['edges'][0]['magnitude'] = repr(math.nextafter(2.**20, 0.))
        r['isolation']['exclusive_ceiling'] = '60'
        self.assertFalse(iso.isolate(r)['excluded'])
        for field, value in [('builder_lower','64'),('edges',[dict(id='x',op='rescale',inputs=['0'],level='0')])]:
            forged = copy.deepcopy(r); forged['isolation'][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): iso.isolate(forged)


def native(exe):
    child = subprocess.run([exe, '--isolation-fixtures'], text=True, capture_output=True, check=True)
    rows = mid.rejection_records(child.stdout.splitlines(), n=1024, q_count=1, alpha=1, max_k=0)
    iso.need(len(rows) == 2 and all(r['status']=='infeasible' and r['reason']=='unknown' for r in rows),
             'not the real native solver fixtures')
    mixed, bare = [r['constraint_isolation'] for r in rows]
    iso.need(rows[0]['round']=='2' and not mixed['excluded'] and rows[1]['round']=='1' and
             bare['conflicts'] == [dict(row=bare['qp_cap_row'], side='upper', extremum='60', bound='1')],
             'unexpected independently isolated row results')
    for r in rows:
        x = r['constraint_isolation']
        print(json.dumps(dict(cap=r['security_cap_bits'], round=r['round'], highs=x['solver_status'],
                              rows_checked=x['rows_checked'], conflicts=x['conflicts'])))


if __name__ == '__main__':
    if len(sys.argv)==3 and sys.argv[1]=='--native': native(Path(sys.argv[2]).resolve())
    else: unittest.main()
