"""Count-order equivalence and fixed-assignment continuous charging checks."""
from collections import Counter
import copy
import unittest

from test_perfect_information import fixture, unpack
from src.domain import ServiceDecision
from src.perfect_information import build_perfect_window, no_service_solution, replay_solution
from src.perfect_information_model import solve_perfect
from src.perfect_information_charging import solve_fixed_assignment_charging
from src.perfect_information_strengthening import canonicalize_slots, slot_groups


class CountOrderTest(unittest.TestCase):
    def test_same_small_exact_optima_and_only_count_rows_added(self):
        for p, scene, plans in [fixture(counts=(2, 3)), fixture(entry=1.9), fixture(power=0.),
                               fixture(random=[dict(request_id='r', station=0, arrival_time=.26,
                                                    deadline=.5, return_soc=.1)])]:
            w = build_perfect_window(p, scene, plans)
            base = solve_perfect(p, w, plans, formulation='strengthened_compact')
            ordered = solve_perfect(p, w, plans, formulation='strengthened_compact',
                                    slot_order='total_services', warm_start=unpack(base))
            self.assertEqual(ordered['status'], 'optimal')
            self.assertEqual(ordered['bound_scope'], 'original_full_problem')
            self.assertAlmostEqual(base['incumbent_objective'], ordered['incumbent_objective'], places=6)
            a, b = base['diagnostics'], ordered['diagnostics']
            expected = sum(len(g)-1 for i in range(p.station.num_stations) for g in slot_groups(p, w, i))
            self.assertEqual(b['model_variables'], a['model_variables'])
            self.assertEqual(b['model_binary_variables'], a['model_binary_variables'])
            self.assertEqual(b['model_constraints']-a['model_constraints'], expected)
            self.assertEqual(replay_solution(p, scene, plans, w, unpack(ordered))['status'], 'passed')
            # A globally optimal MILP assignment must have the same LP charging optimum.
            charge = solve_fixed_assignment_charging(p, w, unpack(base))
            self.assertEqual(charge['status'], 'optimal')
            self.assertEqual(charge['diagnostics']['model_binary_variables'], 0)
            self.assertAlmostEqual(charge['incumbent_objective'], base['incumbent_objective'], places=6)
            self.assertEqual(replay_solution(p, scene, plans, w, unpack(charge))['status'], 'passed')

    def test_count_order_allows_cumulative_counts_to_cross(self):
        random = [dict(request_id=str(n), station=0, arrival_time=n*.25,
                       deadline=n*.25, return_soc=.99) for n in range(3)]
        p, scene, plans = fixture(counts=(2, 1), reservations=False, random=random)
        w = build_perfect_window(p, scene, plans)
        warm = no_service_solution(p, w, plans)
        warm.services = [ServiceDecision('0', 0, 1, 0), ServiceDecision('1', 0, 0, 1),
                         ServiceDecision('2', 0, 0, 2)]
        for d in warm.services:
            warm.power[d.station][d.slot][d.period] = .01*p.battery_capacity_kwh / (p.station.charging_efficiency*p.interval_hours)
        warm.objective_terms.update(income=3., charging_cost=3./p.station.charging_efficiency*.5)
        warm.objective = warm.objective_terms['income']-warm.objective_terms['charging_cost']
        canonical, _ = canonicalize_slots(p, w, warm, order='total_services')
        self.assertEqual([d.slot for d in canonical.services], [1, 0, 0])
        self.assertEqual(replay_solution(p, scene, plans, w, canonical)['status'], 'passed')
        built = solve_perfect(p, w, plans, warm_start=warm, slot_order='total_services', build_only=True)
        self.assertLess(built['diagnostics']['initial_max_constraint_residual'], 1e-8)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            solve_perfect(p, w, plans, formulation='strengthened', slot_order='total_services')

    def test_charging_lp_removes_waste_preserves_assignments(self):
        random = [dict(request_id='r', station=0, arrival_time=.25, deadline=.5, return_soc=.1)]
        p, scene, plans = fixture(counts=(2, 1), reservations=False, random=random)
        w = build_perfect_window(p, scene, plans)
        warm = no_service_solution(p, w, plans)
        warm.services = [ServiceDecision('r', 0, 1, 1)]
        energy = .1*p.battery_capacity_kwh/p.station.charging_efficiency
        warm.power[0][1][1] = energy/p.interval_hours
        warm.soc[0][1][2:] = [.2]*(w.horizon-1)
        warm.objective_terms.update(income=90., charging_cost=.5*energy)
        warm.objective = 90.-.5*energy
        result = solve_fixed_assignment_charging(p, w, warm)
        sol = unpack(result)
        self.assertEqual(sol.services, warm.services)
        self.assertEqual(sol.paths, warm.paths)
        self.assertAlmostEqual(sol.objective, 90., places=6)
        self.assertEqual(replay_solution(p, scene, plans, w, sol)['status'], 'passed')
        bad = copy.deepcopy(warm)
        bad.services.append(copy.deepcopy(bad.services[0]))
        with self.assertRaisesRegex(ValueError, 'invalid fixed service'):
            solve_fixed_assignment_charging(p, w, bad)

    def test_slot_relabeling_does_not_mix_heterogeneous_slots(self):
        p, scene, plans = fixture(counts=(3, 1))
        p.station.slot_power_limits_kw[0] = [60., 30., 60.]
        w = build_perfect_window(p, scene, plans)
        initial = no_service_solution(p, w, plans)
        initial.services = [ServiceDecision('a', 0, 2, 1), ServiceDecision('b', 0, 2, 2),
                            ServiceDecision('c', 0, 1, 3)]
        canonical, permutations = canonicalize_slots(p, w, initial, order='total_services')
        self.assertEqual([d.slot for d in canonical.services], [0, 0, 1])
        self.assertTrue(all(x['old_slot'] != 1 and x['new_slot'] != 1 for x in permutations))


if __name__ == '__main__':
    unittest.main()
