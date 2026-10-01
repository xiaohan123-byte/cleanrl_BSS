"""Acceptance checks for frozen data, heterogeneous slots and complete settlement."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.domain import CandidateRequest, WaitingRequest, RollingState, MPCWindow
from src.experiment_control import ExperimentPaused, fingerprint
from src.experiment_metrics import trajectory_metrics
from src.forecast import build_forecast
from src.mpc_model import solve_mpc
from src.parameters import (BusinessParameters, StationParameters, ODPairParameters,
                            SolverParameters, slots_at, execution_period_limit)
from src.real_data_experiment import load_inputs, make_day, rng
from src.result_statistics import build_result_statistics
from src.rolling_runner import run_rolling_mpc
from src.scenario import SyntheticScenario
from src.terminal_features import FeatureSpec

ROOT = Path(__file__).resolve().parents[1]


def parameters():
    return BusinessParameters(num_periods=4, interval_hours=.25, horizon=2,
        path_update_interval=2, num_reservations=0, max_wait_hours=.5,
        reservation_entry_window_hours=1., min_swap_spacing_km=0., finish_pending_after_demand=True,
        station=StationParameters(num_stations=2, station_ids=[0, 1], positions_km=[30., 60.],
            num_slots=2, num_slots_by_station=[1, 2], initial_slot_soc=[[1.], [1., 1.]],
            charging_efficiency=1., slot_power_limits_kw=[[0.], [0., 0.]], station_power_limits_kw=[0., 0.]),
        od_pairs=[ODPairParameters(0, 0., 90., [0, 1])], vehicle_speed_kmh=60.,
        electricity_price=[[.4, .5, .6, .7]] * 2, swap_service_price=[[4., 1., 1., .1]] * 2,
        random_arrival_rate_per_hour=[0., 0.], solver=SolverParameters(time_limit_sec=5., mip_gap=0.))


class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.means, cls.hashes = load_inputs(ROOT / "data")

    def test_data_and_prices(self):
        p = self.base
        self.assertEqual(p.station.num_slots_by_station, [22,22,22,18,20,25,23,20,28,28,22])
        self.assertEqual(sum(map(len, p.station.initial_slot_soc)), 250)
        self.assertEqual((len(p.od_pairs), self.means.shape), (26, (7,11,24)))
        self.assertEqual(p.solver.mip_gap, .0001)
        self.assertEqual(p.electricity_price_at(0, 96), .41)
        self.assertAlmostEqual(p.swap_service_price_at(0, 96), .91)
        self.assertEqual(execution_period_limit(p), 186)

    def test_fixed_count_reproducibility_and_no_future_leakage(self):
        sampled = rng(300, 1).poisson(self.means[0])
        first, plans = make_day(self.base, self.means, sampled, 1)
        again, _ = make_day(self.base, self.means, sampled, 1)
        self.assertEqual(fingerprint(first.to_dict()), fingerprint(again.to_dict()))
        self.assertEqual((len(first.reservations), len(first.actual_random_requests), len(plans)), (200, 200, 200))
        altered = np.zeros((11,24), dtype=int); altered[0,0] = 1
        changed, _ = make_day(self.base, self.means, altered, 1)
        self.assertEqual(first.reservations, changed.reservations)
        self.assertEqual(first.params, changed.params)
        p = BusinessParameters.from_dict(first.params)
        self.assertEqual(build_forecast(p, first.observation_at(0), 0, 8),
                         build_forecast(p, changed.observation_at(0), 0, 8))
        self.assertTrue(all(r['arrival_time'] < 1 and r['station'] == 0 for r in changed.actual_random_requests))
        self.assertEqual(first.observation_at(0).random_history, [])
        with self.assertRaisesRegex(ValueError, 'zero-total'):
            make_day(self.base, self.means, np.zeros((11,24)), 1)

    def test_frozen_dataset_split_and_counts(self):
        directory = ROOT / 'data/final_real_data_v1'
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        days = manifest['test_day_ids_by_weekday']
        self.assertEqual([(d-1)%7+1 for d in days], list(range(1,8)))
        self.assertEqual(len(manifest['training_validation_pool']), 63)
        self.assertFalse(set(days) & set(manifest['training_validation_pool']))
        self.assertEqual(manifest['source_hashes'], self.hashes)
        for relative, digest in manifest['scenario_hashes'].items():
            scene = json.loads((directory / relative).read_text(encoding='utf-8'))
            self.assertEqual(fingerprint(scene), digest)
            self.assertEqual((scene['seed'], len(scene['reservations']), len(scene['actual_random_requests'])), (1,200,200))


class CleanupAndSlotsTests(unittest.TestCase):
    def test_execution_accepts_solver_tolerance_but_rejects_real_excess(self):
        p = parameters()
        state = RollingState(0, copy.deepcopy(p.station.initial_slot_soc))
        sol = solve_mpc(p, MPCWindow(0, 1, state, {}, []))
        sol.power[0][0][0] = 1e-8
        self.assertAlmostEqual(run_rolling_mpc(p, SyntheticScenario(p, [], []))['cleanup_periods'], 0)
        # Direct execution confirms the first-stage value at the solver bound.
        from src.execution import execute_step
        outcome = execute_step(p, RollingState(0, copy.deepcopy(p.station.initial_slot_soc)), sol, [])
        self.assertAlmostEqual(outcome.state.slot_soc[0][0], 1.)
        sol.power[0][0][0] = 2e-8
        with self.assertRaisesRegex(Exception, 'invalid slot charging power'):
            execute_step(p, RollingState(0, copy.deepcopy(p.station.initial_slot_soc)), sol, [])

    def test_real_slots_only_and_inventory_encoding(self):
        p = parameters()
        waiting = {}
        requests = []
        for i, count in enumerate((2,3)):
            for k in range(count):
                rid = f'{i}:{k}'
                waiting[rid] = WaitingRequest(rid, i, 'random', 0., .5, .2)
                requests.append(CandidateRequest(rid, i, 'random', .2, arrival_time=0., observed=True, deadline=.5))
        state = RollingState(0, copy.deepcopy(p.station.initial_slot_soc), waiting=waiting)
        window = MPCWindow(0, 1, state, {}, requests)
        sol = solve_mpc(p, window)
        self.assertEqual([len(row) for row in sol.power], [1, 2])
        self.assertEqual(len(sol.services), 3)
        self.assertTrue(all(s.slot < slots_at(p,s.station) for s in sol.services))
        spec = FeatureSpec(p, 'inventory_only')
        features = spec.encode_window(window)
        self.assertEqual([features[ix[0]] for ix in spec.inventory_indices], [1.,1.])
        p.station.initial_slot_soc[0].append(1.)
        with self.assertRaisesRegex(ValueError, 'shape'):
            p.validate()

    def test_midnight_service_and_cyclic_settlement(self):
        p = parameters()
        scene = SyntheticScenario(p, [], [dict(request_id='R', station=0, arrival_time=.99, return_soc=.2)])
        result = run_rolling_mpc(p, scene)
        self.assertEqual((result['demand_periods'],result['executed_periods'],result['cleanup_periods']), (4,5,1))
        event = next(e for e in result['ledger'] if e['type']=='random_service')
        self.assertEqual(event['time'], 1.)
        self.assertEqual(event['unit_price'], 4.)
        self.assertAlmostEqual(result['summary']['total_reward'], 320.)
        self.assertTrue(all(r['horizon']==2 for r in result['rounds']))
        self.assertEqual(result['rounds'][4]['forecast']['random_requests'], [])
        build_result_statistics(result)
        metrics = trajectory_metrics(result, scene.to_dict())
        self.assertEqual((metrics['actual_random_count'], metrics['random_services'], metrics['cleanup_periods']), (1,1,1))

    def test_exact_deadline_still_served_after_midnight(self):
        p = parameters()
        req = WaitingRequest('R', 0, 'random', .5, 1., .2)
        state = RollingState(4, copy.deepcopy(p.station.initial_slot_soc), waiting={'R':req})
        candidate = CandidateRequest('R', 0, 'random', .2, arrival_time=.5, observed=True, deadline=1.)
        result = solve_mpc(p, MPCWindow(4, 2, state, {}, [candidate]))
        self.assertEqual([r.period for r in result.services], [4])

    def test_unserved_request_expires_and_is_not_hidden(self):
        p = parameters(); p.station.initial_slot_soc = [[.2], [.2,.2]]
        scene = SyntheticScenario(p, [], [dict(request_id='R', station=0, arrival_time=.99, return_soc=.2)])
        result = run_rolling_mpc(p, scene)
        self.assertEqual(result['summary']['random_timeouts'], 1)
        self.assertEqual(result['cleanup_periods'], 2)
        self.assertEqual(result['final_state']['waiting'], {})
        build_result_statistics(result)

    def test_reservation_continues_until_completion(self):
        p = parameters(); p.num_reservations = 1
        scene = SyntheticScenario(p, [dict(user_key=[0,0], od_id=0, entry_time=.875, entry_soc=.2)], [])
        result = run_rolling_mpc(p, scene)
        self.assertGreater(result['cleanup_periods'], 0)
        self.assertEqual(result['summary']['completed_reservations'], 1)
        self.assertEqual(result['summary']['active_reservations'], 0)
        self.assertTrue(any(e['type']=='reservation_service' and e['time']>=1 for e in result['ledger']))
        build_result_statistics(result)
        metrics = trajectory_metrics(result, scene.to_dict())
        self.assertEqual(metrics['reservation_unfinished_at_L'], 1)
        self.assertEqual(metrics['reservation_unfinished_after_cleanup'], 0)

    def test_resume_at_demand_boundary_does_not_repeat_events(self):
        p = parameters()
        scene = SyntheticScenario(p, [], [dict(request_id='R', station=0, arrival_time=.99, return_soc=.2)])
        def pause(row):
            if row['period'] == 3:
                raise ExperimentPaused('test boundary pause')
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ExperimentPaused):
                run_rolling_mpc(p, scene, progress=pause, journal_dir=directory)
            result = run_rolling_mpc(p, scene, journal_dir=directory)
            repeated = run_rolling_mpc(p, scene, journal_dir=directory)
        self.assertEqual(result['ledger'], repeated['ledger'])
        self.assertEqual(result['executed_periods'], 5)
        self.assertEqual(result['summary']['random_services'], 1)
        build_result_statistics(result)


if __name__ == '__main__':
    unittest.main()
