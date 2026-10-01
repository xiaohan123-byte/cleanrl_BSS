"""Restriction tests against the original physical execution, with exact small solves."""
import copy
import unittest

from test_perfect_information import fixture, unpack
from src.domain import ServiceDecision
from src.perfect_information import build_perfect_window, no_service_solution, replay_solution
from src.perfect_information_model import solve_perfect
from run_perfect_information_diagnostics import verify_fixed_decisions


class RestrictedDiagnosticsTest(unittest.TestCase):
    def test_fixed_routes_still_optimize_service_times_and_following_arrivals(self):
        p, scene, plans = fixture()
        p.swap_service_price[0][3] = 3.
        w = build_perfect_window(p, scene, plans)
        warm = no_service_solution(p, w, plans)
        result = solve_perfect(p, w, plans, restriction='routes', warm_start=warm,
                               formulation='strengthened_compact')
        sol = unpack(result)
        self.assertEqual(sol.paths, plans)
        self.assertEqual(next(d.period for d in sol.services if d.station == 0), 3)
        self.assertAlmostEqual(sol.request_outcomes['oracle:0:0:0:1']['arrival_time'], 1.25)
        self.assertEqual(replay_solution(p, scene, plans, w, sol)['status'], 'passed')
        self.assertEqual(result['bound_scope'], 'restricted_subproblem_only')

    def test_fixed_empty_service_set_does_not_add_services(self):
        p, scene, plans = fixture()
        w = build_perfect_window(p, scene, plans)
        warm = no_service_solution(p, w, plans)
        result = solve_perfect(p, w, plans, restriction='services', warm_start=warm)
        self.assertEqual(result['solution']['services'], [])
        self.assertAlmostEqual(result['incumbent_objective'], warm.objective)
        self.assertEqual(replay_solution(p, scene, plans, w, unpack(result))['status'], 'passed')

    def test_fixed_service_leaves_slots_and_charging_free(self):
        random = [dict(request_id='r', station=0, arrival_time=.25, deadline=.5, return_soc=.1)]
        p, scene, plans = fixture(counts=(2, 1), reservations=False, random=random)
        w = build_perfect_window(p, scene, plans)
        warm = no_service_solution(p, w, plans)
        warm.services = [ServiceDecision('r', 0, 1, 1)]
        # A deliberately wasteful, feasible recharge after the single service.
        energy = .1 * p.battery_capacity_kwh / p.station.charging_efficiency
        warm.power[0][1][1] = energy / p.interval_hours
        warm.soc[0][1][2:] = [.2] * (w.horizon - 1)
        warm.objective_terms.update(income=90., charging_cost=.5 * energy)
        warm.objective = 90. - .5 * energy
        self.assertEqual(replay_solution(p, scene, plans, w, warm)['status'], 'passed')
        result = solve_perfect(p, w, plans, restriction='services', warm_start=warm)
        sol = unpack(result)
        self.assertAlmostEqual(sol.objective, 90., places=6)
        self.assertTrue(verify_fixed_decisions(warm, sol, 'services')['service_plan_unchanged'])
        self.assertEqual(replay_solution(p, scene, plans, w, sol)['status'], 'passed')
        # Relabel the warm-start battery: fixed matrix must remain identical.
        other = copy.deepcopy(warm)
        other.services[0].slot = 0
        other.power[0].reverse()
        other.soc[0].reverse()
        second = solve_perfect(p, w, plans, restriction='services', warm_start=other, build_only=True)
        self.assertEqual(result['diagnostics']['fingerprints'], second['diagnostics']['fingerprints'])

    def test_restrictions_require_valid_warm_start(self):
        p, scene, plans = fixture()
        w = build_perfect_window(p, scene, plans)
        with self.assertRaisesRegex(ValueError, 'require a warm start'):
            solve_perfect(p, w, plans, restriction='routes')
        warm = no_service_solution(p, w, plans)
        warm.services = [ServiceDecision('unknown', 0, 0, 0)]
        with self.assertRaisesRegex(ValueError, 'invalid or duplicate'):
            solve_perfect(p, w, plans, restriction='services', warm_start=warm)


if __name__ == '__main__':
    unittest.main()
