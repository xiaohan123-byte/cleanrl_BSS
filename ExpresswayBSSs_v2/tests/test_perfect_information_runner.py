import json
from pathlib import Path
import tempfile
import unittest

from run_perfect_information import day_disposition
from src.perfect_information_reporting import aggregate


class PerfectRunnerTest(unittest.TestCase):
    def test_attempt_markers_prevent_implicit_resolve(self):
        with tempfile.TemporaryDirectory() as path:
            day = Path(path) / 'day'
            self.assertEqual(day_disposition(day, 'one'), 'pending')
            day.mkdir()
            for state in ['running', 'failed', 'interrupted']:
                (day / 'status.json').write_text(json.dumps(dict(state=state, run_id='one')))
                self.assertEqual(day_disposition(day, 'one'), 'blocked')
            (day / 'status.json').write_text(json.dumps(dict(state='finished', run_id='one')))
            (day / 'result.json').write_text(json.dumps(dict(run_id='one')))
            self.assertEqual(day_disposition(day, 'one'), 'reuse')
            with self.assertRaises(ValueError):
                day_disposition(day, 'two')

    def test_orphan_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as path:
            day = Path(path)
            (day / 'solver.log').write_text('old attempt')
            with self.assertRaises(ValueError):
                day_disposition(day, 'one')

    def test_missing_incumbent_keeps_valid_bound_without_fake_mean(self):
        records = [dict(day_id=d, has_verified_incumbent=True, incumbent_objective=10.,
                        best_bound=12., relative_gap=.2, solve_seconds=5., audit=dict(metrics={}))
                   for d in range(7)]
        records[0].update(has_verified_incumbent=False, incumbent_objective=None)
        result = aggregate(records, list(range(7)))
        self.assertIsNone(result['means']['incumbent_objective'])
        self.assertEqual(result['means']['best_bound'], 12.)
        self.assertEqual(result['solver_p95_seconds'], 5.)
        result = aggregate(records[1:], list(range(7)))
        self.assertIsNone(result['means']['best_bound'])
        self.assertFalse(result['complete_seven_days'])

    def test_unverified_incumbent_is_never_an_operating_result(self):
        records = [dict(day_id=d, has_verified_incumbent=False, incumbent_objective=123.,
                        best_bound=200., solve_seconds=1., relative_gap=None) for d in range(7)]
        result = aggregate(records, list(range(7)))
        self.assertIsNone(result['means']['incumbent_objective'])
        self.assertEqual(result['verified_incumbent_days'], 0)
        self.assertEqual(result['means']['best_bound'], 200.)


if __name__ == '__main__':
    unittest.main()
