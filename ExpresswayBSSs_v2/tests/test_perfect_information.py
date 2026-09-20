"""Small exact solves and independent physical replay for the offline model."""
import copy
import unittest

from src.domain import MPCSolution, ServiceDecision
from src.parameters import BusinessParameters, StationParameters, ODPairParameters, SolverParameters
from src.perfect_information import build_perfect_window, no_service_solution, replay_solution
from src.perfect_information_model import request_timing, solve_perfect, valid_upper_bound
from src.scenario import SyntheticScenario


def fixture(*, counts=(1, 2), reservations=True, random=(), power=60., entry=.01):
    station = StationParameters(num_stations=2, station_ids=[0, 1], positions_km=[20., 50.],
        num_slots=max(counts), num_slots_by_station=list(counts),
        initial_slot_soc=[[1.] * n for n in counts],
        slot_power_limits_kw=[[power] * n for n in counts], station_power_limits_kw=[power] * 2)
    p = BusinessParameters(num_periods=8, interval_hours=.25, horizon=4, path_update_interval=2,
        station=station, od_pairs=[ODPairParameters(0, 0., 80., [0, 1])],
        vehicle_speed_kmh=60., range_km=60., min_swap_spacing_km=0., max_wait_hours=.5,
        num_reservations=int(reservations), reservation_entry_soc_range=[0., 1.],
        terminal_experiment=True, finish_pending_after_demand=True,
        od_sampling_weights=[1.], reservation_hourly_weights=[1., 1.],
        random_hourly_means=[[0., 0.], [0., 0.]],
        electricity_price=[[.5] * 8 for _ in range(2)], swap_service_price=[[1.] * 8 for _ in range(2)],
        entry_soc_error=.05, entry_time_error_hours=.1, report_entry_soc_range=[0., 1.],
        solver=SolverParameters(threads=1, time_limit_sec=10., mip_gap=0., feasibility_tol=1e-9))
    p.validate()
    users = [dict(user_key=[0, 0], od_id=0, entry_time=entry, entry_soc=.55,
                  actual_entry_time=entry, actual_entry_soc=.6,
                  segment_time_multipliers=[1.] * (len(p.physical_nodes()) - 1))] if reservations else []
    scenario = SyntheticScenario(p, users, list(random), 1)
    plans = {'0:0': [0, 1]} if reservations else {}
    return p, scenario, plans


def unpack(result):
    values = copy.deepcopy(result['solution'])
    values['services'] = [ServiceDecision(**s) for s in values['services']]
    return MPCSolution(**values)


class PerfectInformationTest(unittest.TestCase):
    def solve_replay(self, p, scene, plans):
        before = scene.to_dict()
        window = build_perfect_window(p, scene, plans)
        seed = replay_solution(p, scene, plans, window, no_service_solution(p, window, plans))
        self.assertEqual(seed['status'], 'passed')
        result = solve_perfect(p, window, plans)
        self.assertEqual(result['status'], 'optimal')
        solution = unpack(result)
        replay = replay_solution(p, scene, plans, window, solution)
        self.assertEqual(replay['status'], 'passed')
        self.assertEqual(before, scene.to_dict())
        self.assertGreaterEqual(result['best_bound'] + 1e-6, solution.objective)
        return window, result, replay

    def test_truth_and_protected_route_with_heterogeneous_slots(self):
        p, scene, plans = fixture()
        window, result, replay = self.solve_replay(p, scene, plans)
        first = next(r for r in window.requests if r.arc == ('entry', 0))
        self.assertAlmostEqual(first.return_soc, .6 - 20 / 60)
        self.assertEqual([len(s) for s in result['solution']['power']], [1, 2])
        self.assertEqual(replay['metrics']['reservations_completed'], 1)
        self.assertEqual(replay['metrics']['path_adjustments'], 0)
        self.assertAlmostEqual(replay['metrics']['charging_cost'], 0.)

    def test_first_station_wait_changes_downstream_arrival(self):
        p, scene, plans = fixture()
        p.swap_service_price[0][3] = 3.
        _, result, _ = self.solve_replay(p, scene, plans)
        sol = result['solution']
        first = next(s for s in sol['services'] if s['station'] == 0)
        self.assertEqual(first['period'], 3)
        downstream = sol['request_outcomes']['oracle:0:0:0:1']
        self.assertAlmostEqual(downstream['arrival_time'], 1.25)

    def test_failed_first_station_never_activates_downstream(self):
        p, scene, plans = fixture(power=0.)
        p.station.initial_slot_soc[0] = [.2]
        _, result, replay = self.solve_replay(p, scene, plans)
        self.assertEqual(result['solution']['services'], [])
        self.assertEqual(replay['metrics']['reservation_failures'], 1)
        self.assertEqual(replay['metrics']['reservation_failure_cost'], 1000.)

    def test_preentry_route_change_has_no_actual_adjustment_cost(self):
        p, scene, plans = fixture(power=0.)
        p.range_km = 100.
        p.path_adjustment_penalty = 100000.
        p.station.initial_slot_soc[0] = [.2]
        _, result, replay = self.solve_replay(p, scene, plans)
        self.assertEqual(result['solution']['paths']['0:0'], [1])
        self.assertEqual(replay['metrics']['adjustment_cost'], 0.)
        self.assertEqual(replay['metrics']['reservations_completed'], 1)

    def test_true_segment_multipliers_are_used(self):
        p, scene, plans = fixture()
        p.travel_time_relative_error = .1
        scene.reservations[0]['segment_time_multipliers'] = [1.1, .9, 1.]
        window, result, replay = self.solve_replay(p, scene, plans)
        first = next(r for r in window.requests if r.arc == ('entry', 0))
        self.assertAlmostEqual(first.arrival_time, .01 + 20 / (60 / 1.1))
        self.assertEqual(replay['status'], 'passed')

    def test_empty_route_completes_without_a_failure(self):
        p, scene, plans = fixture(power=0.)
        p.range_km = 300.
        p.station.initial_slot_soc = [[.2], [.2, .2]]
        _, result, replay = self.solve_replay(p, scene, plans)
        self.assertEqual(result['solution']['paths']['0:0'], [])
        self.assertEqual(replay['metrics']['reservations_completed'], 1)
        self.assertEqual(replay['metrics']['reservation_failures'], 0)

    def test_midnight_prices_and_complete_cleanup(self):
        p, scene, plans = fixture(entry=1.9)
        _, result, replay = self.solve_replay(p, scene, plans)
        self.assertTrue(all(s['period'] >= p.num_periods for s in result['solution']['services']))
        self.assertEqual(replay['metrics']['reservations_completed'], 1)

    def test_no_demand_has_no_recharge_or_salvage(self):
        p, scene, plans = fixture(reservations=False)
        p.station.initial_slot_soc = [[.2], [.3, .4]]
        _, result, _ = self.solve_replay(p, scene, plans)
        self.assertAlmostEqual(result['incumbent_objective'], 0.)
        self.assertTrue(all(abs(v) < 1e-8 for station in result['solution']['power'] for slot in station for v in slot))

    def test_exact_deadline_and_no_artificial_tie_break(self):
        random = [dict(request_id='a', station=0, arrival_time=.25, return_soc=.8, deadline=.5),
                  dict(request_id='b', station=0, arrival_time=.25, return_soc=.1, deadline=.5)]
        p, scene, plans = fixture(reservations=False, random=random, power=0.)
        p.swap_service_price[0][2] = 2.
        _, result, replay = self.solve_replay(p, scene, plans)
        self.assertEqual([(s['request_id'], s['period']) for s in result['solution']['services']], [('b', 2)])
        self.assertEqual(replay['metrics']['random_timeouts'], 1)

    def test_fcfs_and_reservation_priority(self):
        random = [dict(request_id='earlier', station=0, arrival_time=.26, return_soc=.99, deadline=.5),
                  dict(request_id='later', station=0, arrival_time=.27, return_soc=.1, deadline=.5)]
        p, scene, plans = fixture(reservations=False, random=random, power=0.)
        _, result, _ = self.solve_replay(p, scene, plans)
        self.assertEqual(result['solution']['services'][0]['request_id'], 'earlier')
        p, scene, plans = fixture(random=random, power=0.)
        _, result, _ = self.solve_replay(p, scene, plans)
        self.assertTrue(all(s['request_id'].startswith('oracle:') for s in result['solution']['services']))

    def test_truncated_horizon_is_rejected(self):
        p, scene, plans = fixture()
        window = build_perfect_window(p, scene, plans)
        window.horizon = 2
        with self.assertRaisesRegex(ValueError, 'settle'):
            request_timing(p, window)

    def test_replay_rejects_infeasible_power_and_services(self):
        p, scene, plans = fixture()
        window, result, _ = self.solve_replay(p, scene, plans)
        solution = unpack(result)
        solution.power[0][0][0] = 61.
        with self.assertRaises(ValueError):
            replay_solution(p, scene, plans, window, solution)
        solution = unpack(result)
        solution.services[0].period = 0
        with self.assertRaises(ValueError):
            replay_solution(p, scene, plans, window, solution)

    def test_valid_bounds_without_incumbent_and_invalid_statuses(self):
        self.assertTrue(valid_upper_bound(42., 'time_limit'))
        self.assertTrue(valid_upper_bound(-42., 'optimal'))
        for bound, status in [(float('inf'), 'time_limit'), (1e30, 'time_limit'),
                              (42., 'numerical'), (42., 'infeasible'), (None, 'optimal')]:
            self.assertFalse(valid_upper_bound(bound, status))


if __name__ == '__main__':
    unittest.main()
