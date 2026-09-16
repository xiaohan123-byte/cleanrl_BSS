"""Hand-solvable regression cases for the discrete MILP's business rules."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

from src.domain import CandidateRequest, MPCWindow, RollingState, UserNetwork, WaitingRequest
from src.mpc_model import MPCSolveError, solve_mpc


def toy_parameters(stations=1, slots=1, power=80.0, station_power=None, wait=0.25):
    return SimpleNamespace(
        num_periods=20, interval_hours=1 / 12, max_wait_hours=wait,
        battery_capacity_kwh=60.0, path_adjustment_penalty=1.0, reservation_failure_penalty=100.0,
        station=SimpleNamespace(num_stations=stations, num_slots=slots, charging_efficiency=0.9),
        electricity_price=[[0.5] * 20 for _ in range(stations)],
        swap_service_price=[[1.2] * 20 for _ in range(stations)],
        slot_power_limit=lambda i, b: power,
        station_power_limit=lambda i: power * slots if station_power is None else station_power,
        solver=SimpleNamespace(output_flag=0, threads=1, time_limit_sec=10.0, mip_gap=0.0,
                               feasibility_tol=1e-9),
    )


def request(rid, station=0, kind="random", rho=0.3, arrival=0.0, observed=True,
            key=None, arc=None, predecessors=(), travel=0.0, deadline=None):
    return CandidateRequest(rid, station, kind, rho, key, arc, predecessors,
                            arrival if not predecessors else None, travel, observed, deadline)


def window(requests, soc, horizon=1, ell=0, networks=None):
    waiting = {
        r.request_id: WaitingRequest(r.request_id, r.station, r.kind, r.arrival_time,
                                    r.deadline if r.deadline is not None else r.arrival_time + 0.25,
                                    r.return_soc, r.user_key)
        for r in requests if r.observed
    }
    return MPCWindow(ell, horizon, RollingState(ell, soc, waiting=waiting), networks or {}, requests)


class MPCModelTest(unittest.TestCase):
    def test_no_demand_has_no_terminal_inventory_reward(self):
        result = solve_mpc(toy_parameters(), window([], [[0.2]], horizon=4))
        self.assertEqual(result.status, "optimal")
        self.assertAlmostEqual(result.objective, 0.0)
        self.assertTrue(all(abs(p) < 1e-9 for p in result.power[0][0]))
        self.assertEqual(set(result.objective_terms),
                         {"income", "charging_cost", "adjustment_cost", "failure_cost"})

    def test_two_minute_arrival_needs_next_boundary_and_joint_charging(self):
        req = request("arriving", arrival=2 / 60, observed=False)
        result = solve_mpc(toy_parameters(), window([req], [[0.9]], horizon=2))
        self.assertEqual([(s.request_id, s.period) for s in result.services], [("arriving", 1)])
        self.assertAlmostEqual(result.power[0][0][0], 80.0)
        self.assertAlmostEqual(result.power[0][0][1], 0.0)
        self.assertAlmostEqual(result.soc[0][0][1], 1.0)
        self.assertAlmostEqual(result.soc[0][0][2], 0.3)
        self.assertAlmostEqual(result.objective_terms["income"], 50.4)
        self.assertAlmostEqual(result.objective, 50.4 - 80 / 12 * 0.5)

    def test_unobserved_request_cannot_use_first_stage_even_if_eta_is_now(self):
        req = request("forecast", arrival=0.0, observed=False)
        result = solve_mpc(toy_parameters(), window([req], [[1.0]], horizon=2))
        self.assertEqual([s.period for s in result.services], [1])

    def test_full_battery_service_can_wait_for_a_later_price(self):
        params = toy_parameters()
        params.swap_service_price[0][0] = 0.1
        result = solve_mpc(params, window([request("waiting")], [[1.0]], horizon=2))
        self.assertEqual([s.period for s in result.services], [1])
        self.assertAlmostEqual(result.power[0][0][0], 0.0)

    def test_each_slot_serves_at_most_once_at_a_boundary(self):
        requests = [request(f"r{i}", rho=0.1 * (i + 1)) for i in range(3)]
        result = solve_mpc(toy_parameters(slots=2), window(requests, [[1.0, 1.0]]))
        self.assertEqual(len(result.services), 2)
        self.assertEqual(len({s.slot for s in result.services}), 2)

    def test_reservation_priority_over_more_profitable_random_request(self):
        requests = [request("random", rho=0.0),
                    request("reservation", kind="reservation", rho=0.9, key=(0, 0))]
        result = solve_mpc(toy_parameters(), window(requests, [[1.0]]))
        self.assertEqual([s.request_id for s in result.services], ["reservation"])

    def test_strict_fcfs_but_equal_arrivals_have_no_artificial_tiebreak(self):
        earlier = request("earlier", rho=0.9, arrival=-0.01)
        later = request("later", rho=0.1)
        result = solve_mpc(toy_parameters(), window([earlier, later], [[1.0]]))
        self.assertEqual([s.request_id for s in result.services], ["earlier"])
        earlier.arrival_time = later.arrival_time
        result = solve_mpc(toy_parameters(), window([earlier, later], [[1.0]]))
        self.assertEqual([s.request_id for s in result.services], ["later"])

    def test_wait_at_first_station_propagates_to_next_arrival(self):
        first = request("first", kind="reservation", key=(0, 0))
        second = request("second", station=1, kind="reservation", key=(0, 0),
                         predecessors=("first",), travel=0.1, observed=False)
        result = solve_mpc(toy_parameters(stations=2), window([second, first], [[0.9], [1.0]], horizon=4))
        services = {s.request_id: s.period for s in result.services}
        self.assertEqual(services, {"first": 1, "second": 3})
        self.assertAlmostEqual(result.request_outcomes["second"]["arrival_time"], 1 / 12 + 0.1)

    def test_unserved_predecessor_blocks_downstream_and_only_one_failure(self):
        first = request("first", kind="reservation", key=(0, 0), deadline=0.1)
        second = request("second", station=1, kind="reservation", key=(0, 0),
                         predecessors=("first",), travel=0.05, observed=False)
        result = solve_mpc(toy_parameters(stations=2, power=0.0),
                           window([first, second], [[0.0], [1.0]], horizon=5))
        self.assertFalse(result.services)
        self.assertTrue(result.request_outcomes["first"]["failed"])
        self.assertFalse(result.request_outcomes["second"]["failed"])
        self.assertIsNone(result.request_outcomes["second"]["arrival_time"])
        self.assertAlmostEqual(result.objective_terms["failure_cost"], 100.0)

    def test_deadline_at_horizon_is_not_failed_but_just_before_is(self):
        req = request("reservation", kind="reservation", key=(0, 0), deadline=1 / 12)
        params = toy_parameters(power=0.0)
        result = solve_mpc(params, window([req], [[0.0]]))
        self.assertFalse(result.request_outcomes[req.request_id]["failed"])
        self.assertTrue(result.request_outcomes[req.request_id]["waiting_at_end"])
        req.deadline -= 5e-10
        result = solve_mpc(params, window([req], [[0.0]]))
        self.assertTrue(result.request_outcomes[req.request_id]["failed"])

    def test_receding_execution_matches_first_stage_soc_and_selected_slot(self):
        from src.execution import execute_step
        from src.parameters import get_default_parameters

        params = get_default_parameters()
        state = RollingState(0, [[0.9] * params.station.num_slots
                                 for _ in range(params.station.num_stations)])
        state.waiting["real"] = WaitingRequest("real", 0, "random", 0.0, 0.25, 0.3)
        executed = []
        for _ in range(4):
            candidates = [CandidateRequest(r.request_id, r.station, r.kind, r.return_soc,
                                           arrival_time=r.arrival_time, deadline=r.deadline, observed=True)
                          for r in state.waiting.values()]
            result = solve_mpc(params, MPCWindow(state.period, 4, state, {}, candidates))
            step = execute_step(params, state, result)
            selected = [s for s in result.services if s.period == state.period]
            actual = [e for e in step.events if e["type"] == "random_service"]
            self.assertEqual([(s.request_id, s.station, s.slot) for s in selected],
                             [(e["request_id"], e["station"], e["slot"]) for e in actual])
            for i in range(params.station.num_stations):
                for b in range(params.station.num_slots):
                    self.assertAlmostEqual(step.state.slot_soc[i][b], result.soc[i][b][1], places=7)
            executed.extend(actual)
            state = step.state
        self.assertEqual(len(executed), 1)
        self.assertNotIn("real", state.waiting)

    def test_deadline_at_service_boundary_is_inclusive(self):
        req = request("due", deadline=1 / 12)
        result = solve_mpc(toy_parameters(), window([req], [[0.9]], horizon=2))
        self.assertEqual([s.period for s in result.services], [1])

    def test_paths_change_only_when_network_permits_and_charge_one_adjustment(self):
        arcs = [("entry", 0), (0, "exit"), ("entry", 1), (1, "exit")]
        network = UserNetwork((0, 0), "entry", arcs, (0,), (0,), True)
        requests = [request("a", station=0, kind="reservation", key=(0, 0), arc=("entry", 0),
                            observed=False, arrival=1 / 12),
                    request("b", station=1, kind="reservation", key=(0, 0), arc=("entry", 1),
                            observed=False, arrival=1 / 12)]
        params = toy_parameters(stations=2, power=0.0)
        w = window(requests, [[0.0], [1.0]], horizon=2, networks={"0:0": network})
        result = solve_mpc(params, w)
        self.assertEqual(result.paths["0:0"], [1])
        self.assertAlmostEqual(result.objective_terms["adjustment_cost"], 1.0)
        network.can_update = False
        result = solve_mpc(params, w)
        self.assertEqual(result.paths["0:0"], [0])
        self.assertFalse(result.services)
        self.assertAlmostEqual(result.objective_terms["adjustment_cost"], 0.0)

    def test_station_limit_prevents_simultaneous_full_rate_charging(self):
        params = toy_parameters(slots=2, station_power=80.0)
        requests = [request("a", observed=False, arrival=1 / 12),
                    request("b", observed=False, arrival=1 / 12)]
        result = solve_mpc(params, window(requests, [[0.9, 0.9]], horizon=2))
        self.assertEqual(len(result.services), 1)
        self.assertLessEqual(sum(slot[0] for slot in result.power[0]), 80.0 + 1e-7)
        self.assertAlmostEqual(result.objective_terms["charging_cost"], 80 / 12 * 0.5)

    def test_dynamic_arrival_fcfs_depends_on_actual_predecessor_service(self):
        # The later request is more profitable. An early upstream service makes
        # the dependent request arrive first, and it must then get the one pack.
        params = toy_parameters(stations=2, power=0.0, wait=0.4)
        params.swap_service_price[0] = [20.0, 0.0] + [0.0] * 18
        first = request("upstream", kind="reservation", rho=0.0, key=(0, 0))
        dependent = request("dependent", station=1, kind="reservation", rho=0.9, key=(0, 0),
                            predecessors=("upstream",), travel=0.01, observed=False)
        other = request("other", station=1, kind="reservation", rho=0.0, key=(0, 1),
                        arrival=0.02, observed=False)
        result = solve_mpc(params, window([first, dependent, other], [[1.0], [1.0]], horizon=3))
        ids = {s.request_id for s in result.services}
        self.assertEqual(ids, {"upstream", "dependent"})
        self.assertAlmostEqual(result.request_outcomes["dependent"]["arrival_time"], 0.01)
        # Force upstream to delay using inventory instead. The independent
        # request now arrives first, even if request IDs/insertion order do not.
        params = toy_parameters(stations=2, wait=0.4)
        result = solve_mpc(params, window([first, dependent, other], [[0.9], [1.0]], horizon=3))
        served = {s.request_id: s.period for s in result.services}
        self.assertEqual(served["other"], 1)
        self.assertNotIn("dependent", served)

    def test_shared_predecessors_respect_exclusive_path_arcs(self):
        params = toy_parameters(stations=4, power=0.0, wait=0.4)
        arcs = [("entry", 0), ("entry", 1), (0, 2), (1, 2), (2, 3), (3, "exit")]
        network = UserNetwork((0, 0), "entry", arcs, (1, 2, 3), (1, 2, 3), True)
        requests = [
            request("first0", station=0, kind="reservation", key=(0, 0), arc=("entry", 0),
                    arrival=1 / 12, observed=False),
            request("first1", station=1, kind="reservation", key=(0, 0), arc=("entry", 1),
                    arrival=1 / 12, observed=False),
            request("via0", station=2, kind="reservation", key=(0, 0), arc=(0, 2),
                    predecessors=("first0",), travel=1 / 12, observed=False),
            request("via1", station=2, kind="reservation", key=(0, 0), arc=(1, 2),
                    predecessors=("first1",), travel=1 / 12, observed=False),
            request("merged", station=3, kind="reservation", key=(0, 0), arc=(2, 3),
                    predecessors=("via0", "via1"), travel=1 / 12, observed=False),
        ]
        result = solve_mpc(params, window(requests, [[0.0], [1.0], [1.0], [1.0]],
                                          horizon=4, networks={"0:0": network}))
        self.assertEqual(result.paths["0:0"], [1, 2, 3])
        self.assertEqual({s.request_id: s.period for s in result.services},
                         {"first1": 1, "via1": 2, "merged": 3})
        self.assertFalse(result.request_outcomes["first0"]["selected"])
        self.assertFalse(result.request_outcomes["via0"]["selected"])
        self.assertAlmostEqual(result.request_outcomes["merged"]["arrival_time"], 3 / 12)

    def test_endogenous_equal_arrivals_do_not_impose_fcfs_tiebreak(self):
        params = toy_parameters(stations=2, power=0.0, wait=0.4)
        params.swap_service_price[0] = [20.0] + [0.0] * 19
        first = request("upstream", kind="reservation", rho=0.0, key=(0, 0))
        dependent = request("dependent", station=1, kind="reservation", rho=0.9, key=(0, 0),
                            predecessors=("upstream",), travel=0.01, observed=False)
        other = request("other", station=1, kind="reservation", rho=0.0, key=(0, 1),
                        arrival=0.01, observed=False)
        result = solve_mpc(params, window([first, dependent, other], [[1.0], [1.0]], horizon=3))
        self.assertEqual({s.request_id for s in result.services}, {"upstream", "other"})
        self.assertAlmostEqual(result.request_outcomes["dependent"]["arrival_time"], 0.01)

    def test_zero_adjustment_price_does_not_introduce_a_hidden_path_cost(self):
        params = toy_parameters(stations=2, power=0.0)
        params.path_adjustment_penalty = 0.0
        network = UserNetwork((0, 0), "entry", [("entry", 0), (0, "exit"),
                                               ("entry", 1), (1, "exit")], (0,), (0,), True)
        requests = [request("a", station=0, kind="reservation", key=(0, 0), arc=("entry", 0),
                            observed=False, arrival=1 / 12),
                    request("b", station=1, kind="reservation", key=(0, 0), arc=("entry", 1),
                            observed=False, arrival=1 / 12)]
        result = solve_mpc(params, window(requests, [[0.0], [1.0]], horizon=2,
                                          networks={"0:0": network}))
        self.assertEqual(result.paths["0:0"], [1])
        self.assertEqual(result.objective_terms["adjustment_cost"], 0.0)
        self.assertAlmostEqual(result.objective, result.objective_terms["income"])

    def test_observed_deadline_one_ulp_before_now_forbids_service(self):
        now = 1 / 12
        req = request("expired", kind="reservation", key=(0, 0),
                      deadline=math.nextafter(now, -math.inf))
        result = solve_mpc(toy_parameters(), window([req], [[1.0]], horizon=1, ell=1))
        self.assertFalse(result.services)
        self.assertTrue(result.request_outcomes["expired"]["failed"])
        self.assertEqual(result.request_outcomes["expired"]["deadline"], req.deadline)

    def test_observed_arrival_one_ulp_after_now_cannot_use_current_stage(self):
        now = 1 / 12
        req = request("later", arrival=math.nextafter(now, math.inf), observed=True)
        result = solve_mpc(toy_parameters(), window([req], [[1.0]], horizon=1, ell=1))
        self.assertFalse(result.services)
        self.assertEqual(result.request_outcomes["later"]["arrival_time"], req.arrival_time)

    def test_observed_fcfs_keeps_timestamps_one_ulp_apart(self):
        now = 1 / 12
        early = request("early", arrival=math.nextafter(now, -math.inf), rho=0.9)
        later = request("later", arrival=now, rho=0.0)
        result = solve_mpc(toy_parameters(), window([early, later], [[1.0]], horizon=1, ell=1))
        self.assertEqual([s.request_id for s in result.services], ["early"])

    def test_dynamic_fcfs_rank_does_not_merge_external_time_one_ulp_earlier(self):
        params = toy_parameters(stations=2, power=0.0, wait=0.4)
        params.swap_service_price[0] = [20.0] + [0.0] * 19
        first = request("upstream", kind="reservation", rho=0.0, key=(0, 0))
        dependent = request("dependent", station=1, kind="reservation", rho=0.0, key=(0, 0),
                            predecessors=("upstream",), travel=0.01, observed=False)
        other = request("other", station=1, kind="reservation", rho=0.9, key=(0, 1),
                        arrival=math.nextafter(0.01, -math.inf), observed=False)
        result = solve_mpc(params, window([first, dependent, other], [[1.0], [1.0]], horizon=3))
        self.assertEqual({s.request_id for s in result.services}, {"upstream", "other"})
        self.assertLess(result.request_outcomes["other"]["arrival_time"],
                        result.request_outcomes["dependent"]["arrival_time"])

    def test_infeasible_frozen_path_raises_instead_of_replaying(self):
        network = UserNetwork((0, 0), "entry", [("entry", "exit")], (0,), (0,), False)
        with self.assertRaisesRegex(MPCSolveError, "without a feasible incumbent"):
            solve_mpc(toy_parameters(), window([], [[1.0]], networks={"0:0": network}))


if __name__ == "__main__":
    unittest.main()
