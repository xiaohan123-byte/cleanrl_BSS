"""Physical execution regressions independent of the MILP solver."""
import unittest

from src.accounting import summarize_ledger
from src.domain import MPCSolution, Reservation, RollingState, ServiceDecision, WaitingRequest
from src.execution import ExecutionError, advance_to_boundary, execute_step
from src.parameters import BusinessParameters, ODPairParameters, StationParameters


def parameters(periods=6):
    return BusinessParameters(
        num_periods=periods, horizon=3, num_reservations=0,
        station=StationParameters(
            num_stations=2, station_ids=[0, 1], positions_km=[5., 10.], num_slots=2,
            initial_slot_soc=[[1., 1.], [1., 1.]], charging_efficiency=1.,
            slot_power_limits_kw=[[60., 60.], [60., 60.]], station_power_limits_kw=[100., 100.]),
        od_pairs=[ODPairParameters(0, 0., 15., [0, 1])],
        vehicle_speed_kmh=60., min_swap_spacing_km=0.,
        random_arrival_rate_per_hour=[0., 0.],
    )


def solution(params, state, services=None, paths=None, power=None):
    return MPCSolution(
        status="optimal", objective=0., objective_terms={},
        paths=paths or {}, services=services or [],
        power=power or [[[0.] for _ in range(params.station.num_slots)] for _ in range(params.station.num_stations)],
        soc=[],
    )


def waiting_reservation(deadline=.25):
    request = WaitingRequest("A:0:0:0", 0, "reservation", 0., deadline, .2, (0, 0))
    user = Reservation((0, 0), 0., .8, 5., .2, True, day_ahead=[0, 1],
                       retained_plan=[0, 1], published_plan=[0, 1],
                       waiting_request_id=request.request_id)
    return request, user


class ExecutionTests(unittest.TestCase):
    def test_interval_arrival_waits_and_solver_selected_slot_is_used(self):
        p = parameters()
        state = RollingState(0, [[1., 1.], [1., 1.]])
        record = dict(request_id="R:1", station=0, arrival_time=2 / 60, return_soc=.2)
        first = execute_step(p, state, solution(p, state), [record])
        self.assertEqual(state.period, 0)
        self.assertFalse(state.waiting)
        self.assertIn("R:1", first.state.waiting)
        self.assertEqual(summarize_ledger(first.state.ledger)["random_services"], 0)
        action = ServiceDecision("R:1", 0, 1, 1)
        power = [[[0.], [60.]], [[0.], [0.]]]
        second = execute_step(p, first.state, solution(p, first.state, [action], power=power))
        self.assertEqual(second.state.slot_soc[0], [1., .25])
        self.assertEqual(summarize_ledger(second.state.ledger)["random_services"], 1)

    def test_full_battery_does_not_force_service(self):
        p = parameters()
        request = WaitingRequest("R", 0, "random", 0., .25, .2)
        state = RollingState(0, [[1., 1.], [1., 1.]], waiting={"R": request})
        result = execute_step(p, state, solution(p, state))
        self.assertIn("R", result.state.waiting)
        self.assertEqual(result.state.slot_soc, state.slot_soc)

    def test_wait_propagates_to_next_station(self):
        p = parameters()
        request, user = waiting_reservation()
        state = RollingState(0, [[1., 1.], [1., 1.]], {"0:0": user}, {request.request_id: request})
        delayed = execute_step(p, state, solution(p, state, paths={"0:0": [1]}))
        self.assertEqual(delayed.state.users["0:0"].position_km, 5.)
        served = execute_step(p, delayed.state, solution(p, delayed.state,
            [ServiceDecision(request.request_id, 0, 0, 1)], {"0:0": [1]}))
        next_request = served.state.waiting["A:0:0:1"]
        self.assertEqual(next_request.station, 1)
        self.assertAlmostEqual(next_request.arrival_time, 2 * p.interval_hours)
        self.assertEqual(served.state.users["0:0"].completed_stations, [0])
        self.assertEqual(summarize_ledger(served.state.ledger)["reservation_services"], 1)

    def test_boundary_deadline_kept_then_failure_once_cancels_following(self):
        p = parameters()
        request, user = waiting_reservation(p.interval_hours)
        state = RollingState(0, [[1., 1.], [1., 1.]], {"0:0": user}, {request.request_id: request})
        first = execute_step(p, state, solution(p, state, paths={"0:0": [1]}))
        self.assertEqual(first.state.waiting[request.request_id].deadline, p.interval_hours)
        second = execute_step(p, first.state, solution(p, first.state, paths={"0:0": [1]}))
        self.assertEqual(second.state.users["0:0"].status, "failed")
        self.assertEqual(second.state.users["0:0"].retained_plan, [])
        third = execute_step(p, second.state, solution(p, second.state))
        self.assertEqual(summarize_ledger(third.state.ledger)["reservation_failures"], 1)

    def test_boundary_deadline_can_be_served(self):
        p = parameters()
        request = WaitingRequest("R", 0, "random", 0., p.interval_hours, .2)
        state = RollingState(1, [[1., 1.], [1., 1.]], waiting={"R": request})
        result = execute_step(p, state, solution(p, state, [ServiceDecision("R", 0, 0, 1)]))
        self.assertFalse(result.state.waiting)

    def test_reservation_priority_blocks_random(self):
        p = parameters()
        request, user = waiting_reservation()
        random = WaitingRequest("R", 0, "random", 0., .25, .2)
        state = RollingState(0, [[1., 1.], [1., 1.]], {"0:0": user},
                             {request.request_id: request, "R": random})
        with self.assertRaisesRegex(ExecutionError, "higher-priority"):
            execute_step(p, state, solution(p, state, [ServiceDecision("R", 0, 0, 0)]))

    def test_close_but_distinct_fcfs_arrivals_are_ordered(self):
        p = parameters()
        waiting = {"early": WaitingRequest("early", 0, "random", 0., .25, .2),
                   "later": WaitingRequest("later", 0, "random", 5e-10, .25, .2)}
        state = RollingState(1, [[1., 1.], [1., 1.]], waiting=waiting)
        with self.assertRaisesRegex(ExecutionError, "higher-priority"):
            execute_step(p, state, solution(p, state, [ServiceDecision("later", 0, 0, 1)]))

    def test_near_boundary_actual_arrival_is_not_discarded(self):
        p = parameters()
        state = RollingState(0, [[1., 1.], [1., 1.]])
        record = dict(request_id="R", station=0, arrival_time=p.interval_hours - 5e-10, return_soc=.2)
        result = execute_step(p, state, solution(p, state), [record])
        self.assertEqual(result.state.waiting["R"].arrival_time, record["arrival_time"])

    def test_near_boundary_deadline_expires_strictly(self):
        p = parameters()
        request, user = waiting_reservation(p.interval_hours - 5e-10)
        state = RollingState(0, [[1., 1.], [1., 1.]], {"0:0": user}, {request.request_id: request})
        result = execute_step(p, state, solution(p, state))
        self.assertEqual(result.state.users["0:0"].status, "failed")
        self.assertEqual(summarize_ledger(result.state.ledger)["reservation_failures"], 1)

    def test_slightly_future_arrival_cannot_be_served(self):
        p = parameters()
        request = WaitingRequest("R", 0, "random", 5e-10, .25, .2)
        state = RollingState(0, [[1., 1.], [1., 1.]], waiting={"R": request})
        with self.assertRaisesRegex(ExecutionError, "observed waiting"):
            execute_step(p, state, solution(p, state, [ServiceDecision("R", 0, 0, 0)]))

    def test_duplicate_assignment_and_incomplete_battery_rejected(self):
        p = parameters()
        request = WaitingRequest("R", 0, "random", 0., .25, .2)
        state = RollingState(0, [[.9, 1.], [1., 1.]], waiting={"R": request})
        with self.assertRaisesRegex(ExecutionError, "full battery"):
            execute_step(p, state, solution(p, state, [ServiceDecision("R", 0, 0, 0)]))
        with self.assertRaisesRegex(ExecutionError, "assigned twice"):
            execute_step(p, state, solution(p, state, [ServiceDecision("R", 0, 1, 0)] * 2))

    def test_power_is_never_projected_or_used_to_overcharge(self):
        p = parameters()
        state = RollingState(0, [[1., 1.], [1., 1.]])
        with self.assertRaisesRegex(ExecutionError, "overcharge"):
            execute_step(p, state, solution(p, state, power=[[[1.], [0.]], [[0.], [0.]]]))
        with self.assertRaisesRegex(ExecutionError, "station power"):
            execute_step(p, state, solution(p, state, power=[[[60.], [60.]], [[0.], [0.]]]))

    def test_boundary_random_observation_is_admitted_once(self):
        p = parameters()
        state = RollingState(0, [[1., 1.], [1., 1.]])
        request = dict(request_id="R", station=0, arrival_time=0., return_soc=.2)
        first = advance_to_boundary(p, state, [request])
        second = advance_to_boundary(p, first.state, [request])
        self.assertEqual(len(second.state.ledger), 1)
        result = execute_step(p, second.state, solution(p, second.state,
            [ServiceDecision("R", 0, 1, 0)]), [request])
        self.assertEqual(summarize_ledger(result.state.ledger)["random_services"], 1)


if __name__ == "__main__":
    unittest.main()
