from __future__ import annotations

import unittest

from src.event_engine import project_requested_power


class PowerProjectionTest(unittest.TestCase):
    def test_projection_clips_slots_then_scales_station_energy(self) -> None:
        projection = project_requested_power(
            [[80.0, -3.0], [40.0, 50.0]],
            slot_power_limit_kw=[[50.0, 50.0], [50.0, 30.0]],
            station_energy_limit_kwh=[20.0, 24.0],
            interval_hours=1.0,
            shape=[2, 2],
        )
        # Station 0: [50, 0] -> proportional energy cap 20.
        self.assertEqual(projection.power_kw[0], [20.0, 0.0])
        # Station 1: [40, 30] -> cap 24/70.
        self.assertAlmostEqual(sum(projection.power_kw[1]), 24.0)
        self.assertTrue(all(power >= 0.0 for row in projection.power_kw for power in row))
        self.assertLessEqual(projection.power_kw[1][0], 50.0)
        self.assertLessEqual(projection.power_kw[1][1], 30.0)
        self.assertEqual(projection.station_energy_kwh, [20.0, 24.0])

    def test_projection_is_deterministic(self) -> None:
        kwargs = dict(
            requested_power=[[9.0, 7.0]],
            slot_power_limit_kw=8.0,
            station_energy_limit_kwh=6.0,
            interval_hours=0.5,
            shape=[2],
        )
        self.assertEqual(project_requested_power(**kwargs), project_requested_power(**kwargs))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
