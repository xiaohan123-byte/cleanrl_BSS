from __future__ import annotations

import unittest

from data_generation_test.parameter import get_default_parameters
from src.domain import CandidateRequest, RequestKind
from src.reference_rollout import (
    ReferenceRolloutError,
    build_accepted_reservation_rollouts,
    build_reservation_rollout,
    events_in_prediction_window,
    flatten_reservation_events,
    reservation_dependency_map,
)
from src.time_grid import TimeGrid


class ReferenceRolloutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = get_default_parameters()
        self.record = {
            "reservation_id": 17,
            "od_id": 1,
            "accepted": True,
            "day_ahead_entry_time": 0.5,
            "day_ahead_entry_soc": 0.8,
            # These are deliberately hidden truth fields.  The adapter must
            # not use them unless an explicit visible override is supplied.
            "actual_entry_time": 99.0,
            "actual_entry_soc": 0.01,
            "path_arcs": [["entry", 0], [0, 2], [2, 4], [4, "exit"]],
            # Old discrete fields must not drive the continuous adaptation.
            "swap_periods": [0, 0, 0],
            "return_socs": [0.0, 0.0, 0.0],
        }

    def test_continuous_eta_soc_ids_and_dependencies_follow_path(self) -> None:
        rollout = build_reservation_rollout(self.params, self.record)
        events = rollout.events

        self.assertEqual([event.station for event in events], [0, 2, 4])
        self.assertEqual(
            [event.event_id for event in events],
            [
                "reservation:1:17:0:0",
                "reservation:1:17:1:2",
                "reservation:1:17:2:4",
            ],
        )
        self.assertIsNone(events[0].upstream_request_id)
        self.assertEqual(events[1].upstream_request_id, events[0].event_id)
        self.assertEqual(events[2].upstream_request_id, events[1].event_id)
        self.assertAlmostEqual(events[0].arrival_time, 0.5 + 80.0 / 75.0)
        self.assertAlmostEqual(events[1].arrival_time, 0.5 + 280.0 / 75.0)
        self.assertAlmostEqual(events[2].arrival_time, 0.5 + 480.0 / 75.0)
        self.assertAlmostEqual(events[0].return_soc, 0.8 - 80.0 / 300.0)
        self.assertAlmostEqual(events[1].return_soc, 1.0 - 200.0 / 300.0)
        self.assertAlmostEqual(events[2].return_soc, 1.0 - 200.0 / 300.0)
        self.assertAlmostEqual(
            events[1].deadline - events[1].arrival_time,
            self.params.max_wait_hours,
        )
        self.assertEqual(
            reservation_dependency_map([rollout]),
            {
                "reservation:1:17:0:0": ["reservation:1:17:1:2"],
                "reservation:1:17:1:2": ["reservation:1:17:2:4"],
            },
        )

    def test_only_explicit_visible_entry_can_override_day_ahead_values(self) -> None:
        # No future actual_* value on the day-ahead record is read.
        planned = build_reservation_rollout(self.params, self.record)
        self.assertEqual(planned.entry.source, "day_ahead")
        self.assertAlmostEqual(planned.entry.entry_time, 0.5)
        self.assertAlmostEqual(planned.entry.entry_soc, 0.8)

        visible_entry = {
            "reservation_id": 17,
            "od_id": 1,
            "arrival_time": 0.75,
            "arrival_soc": 0.9,
        }
        observed = build_reservation_rollout(
            self.params, self.record, visible_entries=[visible_entry]
        )
        self.assertEqual(observed.entry.source, "visible_override")
        self.assertAlmostEqual(observed.events[0].arrival_time, 0.75 + 80.0 / 75.0)
        self.assertAlmostEqual(observed.events[0].return_soc, 0.9 - 80.0 / 300.0)

    def test_infeasible_visible_entry_is_not_silently_made_feasible(self) -> None:
        with self.assertRaisesRegex(ReferenceRolloutError, "arrival SOC"):
            build_reservation_rollout(
                self.params,
                self.record,
                visible_entries=[
                    {
                        "reservation_id": 17,
                        "arrival_time": 0.5,
                        "arrival_soc": 0.1,
                    }
                ],
            )

    def test_collection_skips_rejected_and_window_is_half_open(self) -> None:
        rejected = dict(self.record)
        rejected["reservation_id"] = 18
        rejected["accepted"] = False
        plan = {"reservations": [self.record, rejected]}
        rollouts = build_accepted_reservation_rollouts(self.params, plan)
        self.assertEqual(len(rollouts), 1)
        self.assertEqual(len(flatten_reservation_events(rollouts)), 3)

        before_end = CandidateRequest(
            request_id="before-end",
            kind=RequestKind.RESERVATION,
            station=0,
            arrival_time=3.99999995,
            deadline=4.1,
            return_soc=0.2,
            user_key=(1, 90),
        )
        exact_end = CandidateRequest(
            request_id="at-end",
            kind=RequestKind.RESERVATION,
            station=0,
            arrival_time=4.0,
            deadline=4.1,
            return_soc=0.2,
            user_key=(1, 91),
        )
        grid = TimeGrid(interval_hours=1.0, num_intervals=5)
        selected = events_in_prediction_window(
            [before_end, exact_end], grid, period_ell=0, horizon=4
        )
        self.assertEqual([event.request_id for event in selected], ["before-end"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
