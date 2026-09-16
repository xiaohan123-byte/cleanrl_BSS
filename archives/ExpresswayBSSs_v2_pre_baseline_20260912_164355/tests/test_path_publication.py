from __future__ import annotations

import unittest

from data_generation_test.candidate_network import generate_candidate_network
from data_generation_test.parameter import get_default_parameters
from src.path_state import (
    VIRTUAL_ORIGIN,
    build_remaining_network,
    publish_if_changed,
    remaining_path_after_executed,
    station_sequence,
)


class PathPublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = get_default_parameters()
        self.network = generate_candidate_network(self.params)

    def test_future_internal_change_is_not_published(self) -> None:
        state = {
            "od_id": 1,
            "user_id": 1,
            "phase": "future",
            "initial_published_path": [("entry", 2), (2, "exit")],
        }
        decision = publish_if_changed(state, [("entry", 3), (3, "exit")], 2.0)
        self.assertFalse(decision.changed)
        self.assertIsNone(decision.event_id)

    def test_virtual_origin_motion_without_station_change_is_free(self) -> None:
        state = {
            "od_id": 1,
            "user_id": 7,
            "phase": "enroute",
            "last_published_remaining_path": [("entry", 3), (3, 5), (5, "exit")],
        }
        decision = publish_if_changed(
            state,
            [(VIRTUAL_ORIGIN, 3), (3, 5), (5, "exit")],
            3.0,
        )
        self.assertFalse(decision.changed)
        self.assertEqual(decision.previous_station_sequence, (3, 5))
        self.assertEqual(decision.proposed_station_sequence, (3, 5))

    def test_enroute_station_change_has_one_stable_publication_id(self) -> None:
        state = {
            "od_id": 1,
            "user_id": 8,
            "phase": "enroute",
            "last_published_remaining_path": [("entry", 2), (2, 5), (5, "exit")],
        }
        decision = publish_if_changed(
            state,
            [(VIRTUAL_ORIGIN, 3), (3, 5), (5, "exit")],
            4.0,
        )
        self.assertTrue(decision.changed)
        self.assertEqual(decision.event_id, "publish:1:8:4.000000000")

    def test_waiting_station_remains_in_remaining_network(self) -> None:
        state = {
            "od_id": 1,
            "user_id": 9,
            "phase": "waiting",
            "position_km": 280.0,
            "vehicle_soc": 0.25,
            "last_actual_swap_km": 180.0,
            "waiting_station": 2,
            "last_published_remaining_path": [("entry", 2), (2, 5), (5, "exit")],
        }
        remaining = build_remaining_network(self.network, self.params, state, now=4.0)
        self.assertIn((VIRTUAL_ORIGIN, 2), remaining.arcs)
        self.assertEqual(remaining.waiting_station, 2)

    def test_executed_prefix_is_not_misclassified_as_a_new_path(self) -> None:
        path = [("entry", 2), (2, 4), (4, "exit")]
        tail = remaining_path_after_executed(path, [2])
        self.assertEqual(tail, [(2, 4), (4, "exit")])
        self.assertEqual(station_sequence(tail), (4,))
        publication = publish_if_changed(
            {
                "od_id": 1,
                "user_id": 7,
                "phase": "enroute",
                "last_published_remaining_path": tail,
            },
            [(VIRTUAL_ORIGIN, 4), (4, "exit")],
            4.0,
        )
        self.assertFalse(publication.changed)


if __name__ == "__main__":
    unittest.main()
