# -*- coding: utf-8 -*-
"""六站 synthetic/mock 输入的最小回归测试。"""

from __future__ import annotations

import unittest

from data_generation_test.parameter import get_default_parameters
from data_generation_test.rl_data import (
    MockRLProvider,
    SyntheticScenario,
    generate_mock_data,
    project_requested_power,
)


class SixStationMockSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = get_default_parameters()

    def test_default_input_is_six_station_seeded_synthetic(self) -> None:
        data = generate_mock_data(self.params, seed=42)
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(data["data_source"], "synthetic")
        self.assertEqual(data["signal_source"], "mock")
        self.assertEqual(self.params.station.station_ids, list(range(6)))
        self.assertEqual(
            self.params.station.positions_km,
            [80.0, 180.0, 280.0, 380.0, 480.0, 580.0],
        )
        self.assertEqual(self.params.num_periods, 12)
        self.assertEqual(len(data["stations"]), 6)
        self.assertEqual(len(data["reservations"]), 6)
        for table_name in (
            "electricity_price",
            "swap_service_price",
            "station_energy_limit_kwh",
        ):
            table = data["parameter_snapshot"][table_name]
            self.assertEqual(len(table), 6)
            self.assertTrue(all(len(row) == 12 for row in table))
        for station in range(6):
            self.assertGreater(
                sum(len(row) for row in data["predicted_random_requests"][station]),
                0,
            )
            self.assertGreater(
                sum(len(row) for row in data["actual_random_requests"][station]),
                0,
            )

    def test_seed_reproducibility_and_observation_visibility(self) -> None:
        data = generate_mock_data(self.params, seed=2026)
        self.assertEqual(data, generate_mock_data(self.params, seed=2026))
        scenario = SyntheticScenario.from_dict(data)
        view = scenario.observation_at(0.0)
        exposed = view.to_dict()
        self.assertNotIn("actual_random_requests", exposed)
        self.assertNotIn("vehicle_trajectories", exposed)
        self.assertTrue(
            all(item["arrival_time"] <= 0.0 for item in view.actual_random_history)
        )
        self.assertTrue(
            all(
                point["time"] <= 0.0
                for points in view.vehicle_snapshots.values()
                for point in points
            )
        )
        self.assertTrue(
            all(
                "actual_entry_time" not in item
                for item in view.reservations
            )
        )

    def test_mock_power_uses_energy_projection_and_rejects_ground_truth(self) -> None:
        raw = [
            [[100.0] for _ in range(self.params.station.num_slots)]
            for _ in range(self.params.station.num_stations)
        ]
        projected = project_requested_power(self.params, raw, start_period=0)
        for station in range(self.params.station.num_stations):
            energy = self.params.interval_hours * sum(
                projected[station][slot][0]
                for slot in range(self.params.station.num_slots)
            )
            self.assertAlmostEqual(
                energy, self.params.station_energy_limit_at(station, 0)
            )
            self.assertTrue(
                all(
                    value <= self.params.station.slot_power_limit_kw
                    for value in (projected[station][slot][0] for slot in range(5))
                )
            )

        scenario = SyntheticScenario.from_dict(generate_mock_data(self.params, 42))
        provider = MockRLProvider(self.params)
        signals = provider.get_signals(
            self.params,
            0,
            self.params.horizon,
            self.params.station.initial_slot_soc,
            observation=scenario.observation_at(0.0),
        )
        self.assertEqual(signals.signal_source, "mock")
        signals.validate(self.params)
        with self.assertRaises(TypeError):
            provider.get_signals(
                self.params,
                0,
                self.params.horizon,
                self.params.station.initial_slot_soc,
                observation=scenario,  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
