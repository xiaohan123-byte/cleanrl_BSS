from __future__ import annotations

import unittest

from src.domain import RequestKind, RollingState, SlotState, WaitingRequest
from src.event_engine import ContinuousEventEngine
from src.time_grid import TimeGrid


def _request(
    request_id: str,
    *,
    kind: RequestKind = RequestKind.RANDOM,
    arrival: float = 0.0,
    deadline: float = 0.9,
    return_soc: float = 0.2,
) -> WaitingRequest:
    return WaitingRequest(
        request_id=request_id,
        kind=kind,
        station=0,
        arrival_time=arrival,
        deadline=deadline,
        return_soc=return_soc,
    )


class ContinuousEventEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = TimeGrid(interval_hours=1.0, num_intervals=4)
        self.engine = ContinuousEventEngine(
            self.grid,
            battery_capacity_kwh=100.0,
            charging_efficiency=1.0,
            max_wait_hours=0.25,
        )

    @staticmethod
    def _state(soc: float = 1.0) -> RollingState:
        return RollingState(now=0.0, slots=[[SlotState(0, 0, soc)]])

    def test_ready_before_arrival_serves_immediately(self) -> None:
        result = self.engine.simulate_interval(
            self._state(), 0, [[0.0]], [_request("ready", arrival=0.2)]
        )
        self.assertEqual(len(result.services), 1)
        self.assertEqual(result.services[0].occurred_at, 0.2)
        self.assertEqual(result.services[0].slot, 0)
        self.assertAlmostEqual(result.state.slots[0][0].soc, 0.2)
        self.assertFalse(result.state.slots[0][0].ready)

    def test_near_full_soc_is_not_a_service_ready_battery(self) -> None:
        result = self.engine.simulate_interval(
            self._state(1.0 - 5e-10),
            0,
            [[0.0]],
            [_request("strict-full", arrival=0.0, deadline=0.25)],
        )
        self.assertEqual(result.services, [])
        self.assertEqual(
            [item.request.request_id for item in result.timeouts],
            ["strict-full"],
        )

    def test_charge_complete_at_deadline_precedes_timeout(self) -> None:
        result = self.engine.simulate_interval(
            self._state(0.9),
            0,
            [[20.0]],
            [_request("deadline", arrival=0.0, deadline=0.5)],
        )
        self.assertEqual([item.request.request_id for item in result.services], ["deadline"])
        self.assertEqual(result.timeouts, [])
        self.assertEqual(result.services[0].occurred_at, 0.5)

    def test_charge_after_deadline_times_out(self) -> None:
        result = self.engine.simulate_interval(
            self._state(0.9),
            0,
            [[10.0]],
            [_request("late", arrival=0.0, deadline=0.5)],
        )
        self.assertEqual(result.services, [])
        self.assertEqual([item.request.request_id for item in result.timeouts], ["late"])
        self.assertAlmostEqual(result.timeouts[0].wait_hours, 0.5)

    def test_reservation_priority_and_stable_fcfs(self) -> None:
        state = RollingState(
            now=0.0,
            slots=[[SlotState(0, 0, 1.0), SlotState(0, 1, 1.0)]],
        )
        result = self.engine.simulate_interval(
            state,
            0,
            [[0.0, 0.0]],
            [
                _request("random", arrival=0.1, deadline=2.0),
                _request("res-b", kind=RequestKind.RESERVATION, arrival=0.1, deadline=2.0),
                _request("res-a", kind=RequestKind.RESERVATION, arrival=0.1, deadline=2.0),
            ],
        )
        self.assertEqual(
            [item.request.request_id for item in result.services], ["res-a", "res-b"]
        )
        self.assertEqual(result.services[0].slot, 0)
        self.assertEqual(result.services[1].slot, 1)
        self.assertEqual(
            [item.request_id for item in result.state.queue_for(0, RequestKind.RANDOM)],
            ["random"],
        )

    def test_empty_lower_index_station_does_not_block_other_station(self) -> None:
        state = RollingState(
            now=0.0,
            slots=[[SlotState(0, 0, 1.0)], [SlotState(1, 0, 1.0)]],
        )
        station_one_request = WaitingRequest(
            request_id="station-one",
            kind=RequestKind.RANDOM,
            station=1,
            arrival_time=0.1,
            deadline=0.9,
            return_soc=0.3,
        )
        result = self.engine.simulate_interval(
            state, 0, [[0.0], [0.0]], [station_one_request]
        )
        self.assertEqual(len(result.services), 1)
        self.assertEqual(result.services[0].station, 1)
        self.assertEqual(result.services[0].slot, 0)

    def test_same_slot_recharges_before_second_service(self) -> None:
        result = self.engine.simulate_interval(
            self._state(),
            0,
            [[100.0]],
            [
                _request("first", arrival=0.0, deadline=0.9, return_soc=0.8),
                _request("second", arrival=0.3, deadline=0.9, return_soc=0.4),
            ],
        )
        self.assertEqual([item.request.request_id for item in result.services], ["first", "second"])
        self.assertAlmostEqual(result.services[1].occurred_at, 0.3)
        # A complete 0.0--0.2 charging episode is present between services.
        self.assertTrue(
            any(
                segment.start_time == 0.0 and segment.end_time == 0.2
                for segment in result.charging_segments
            )
        )

    def test_endpoint_arrival_and_completion_are_deferred_once(self) -> None:
        request = _request("edge", arrival=1.0, deadline=1.25)
        first = self.engine.simulate_interval(
            self._state(0.9), 0, [[10.0]], [request]
        )
        self.assertEqual(first.services, [])
        self.assertEqual(first.timeouts, [])
        self.assertFalse(first.state.slots[0][0].ready)
        self.assertEqual(first.state.slots[0][0].completion_due_at, 1.0)

        second = self.engine.simulate_interval(
            first.state, 1, [[0.0]], [request]
        )
        self.assertEqual([item.request.request_id for item in second.services], ["edge"])
        self.assertEqual(second.services[0].occurred_at, 1.0)
        self.assertEqual(len(second.ledger_entries), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
