from __future__ import annotations

import unittest

from data_generation_test.candidate_network import generate_candidate_network
from data_generation_test.parameter import get_default_parameters
from src.event_path_search import build_path_options, enumerate_remaining_paths
from src.path_state import VIRTUAL_ORIGIN, build_remaining_network, station_sequence
from src.reference_rollout import ReservationEventRollout, VisibleEntry


class EventPathSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = get_default_parameters()
        self.network = generate_candidate_network(self.params)
        self.previous = ReservationEventRollout(
            reservation_id="0",
            od_id=1,
            user_key=(1, 0),
            entry=VisibleEntry(0.0, 1.0, "visible_override"),
            path_arcs=(("entry", 2), (2, 4), (4, "exit")),
            events=(),
        )

    def _remaining(self, *, position: float, soc: float):
        return build_remaining_network(
            self.network,
            self.params,
            {
                "od_id": 1,
                "user_id": 0,
                "phase": "enroute",
                "position_km": position,
                "vehicle_soc": soc,
                "last_actual_swap_km": 0.0,
                "last_published_remaining_path": [
                    ("entry", 2),
                    (2, 4),
                    (4, "exit"),
                ],
            },
            now=1.0,
        )

    def test_virtual_origin_spacing_is_measured_from_last_actual_swap(self) -> None:
        remaining = self._remaining(position=150.0, soc=0.8)
        # Station 1 is only 30 km ahead, but it is 180 km after the last
        # actual swap at the O-D entry and therefore satisfies D_min=100 km.
        self.assertIn((VIRTUAL_ORIGIN, 1), remaining.arcs)

    def test_short_station_is_restored_when_it_is_the_only_safe_path(self) -> None:
        remaining = self._remaining(position=0.956174625, soc=0.440601414)
        self.assertIn((VIRTUAL_ORIGIN, 0), remaining.arcs)
        self.assertTrue(enumerate_remaining_paths(remaining))

    def test_seed42_candidate_materialises_observed_eta_and_soc(self) -> None:
        remaining = self._remaining(position=75.0, soc=0.75)
        sequences = [station_sequence(path) for path in enumerate_remaining_paths(remaining)]
        self.assertIn((1, 3, 5), sequences)

        options = build_path_options(
            self.params,
            remaining,
            self.previous,
            now=1.0,
            position_km=75.0,
            vehicle_soc=0.75,
            request_status={},
        )
        option = next(item for item in options if item.station_sequence == (1, 3, 5))
        self.assertEqual(option.path_arcs[0], (VIRTUAL_ORIGIN, 1))
        self.assertEqual([event.station for event in option.rollout.events], [1, 3, 5])
        self.assertAlmostEqual(option.rollout.events[0].arrival_time, 2.4)
        self.assertAlmostEqual(option.rollout.events[0].return_soc, 0.4)
        self.assertAlmostEqual(option.rollout.events[1].arrival_time, 5.066666666666666)
        self.assertAlmostEqual(option.rollout.events[2].arrival_time, 7.733333333333333)


if __name__ == "__main__":
    unittest.main()
