"""Checks for the agreed daily generator and its information boundary."""
import copy
import tempfile
import unittest
from pathlib import Path

from src.parameters import BusinessParameters
from src.scenario import SyntheticScenario, generate_synthetic_scenario, load_scenario, save_scenario

ROOT = Path(__file__).resolve().parents[1]


class TerminalScenarioTests(unittest.TestCase):
    def setUp(self):
        self.params = BusinessParameters.load_json(ROOT / "configs/terminal_experiment.json")

    def test_source_network_and_operating_scale(self):
        p = self.params
        self.assertEqual(p.station.positions_km, [14., 61., 255., 375., 542., 613.])
        self.assertEqual(len(p.od_pairs), 30)
        self.assertEqual(sum(p.od_direction(i) < 0 for i in range(30)), 15)
        self.assertEqual(len(p.physical_nodes()), 14)
        self.assertEqual((p.num_periods, p.horizon, p.path_update_interval), (288, 72, 3))
        self.assertEqual(p.station.num_slots, 21)
        self.assertTrue(all(v == 1. for row in p.station.initial_slot_soc for v in row))
        self.assertAlmostEqual(sum(map(sum, p.random_hourly_means)), 100.)
        self.assertAlmostEqual(sum(p.od_sampling_weights), 1.)
        for i, od in enumerate(p.od_pairs):
            self.assertGreater(p.distance_km(i, "entry", "exit"), 0.)
            self.assertEqual(p.distance_km(i, "entry", "exit"), abs(od.exit_km - od.entry_km))

    def test_price_mapping_and_hour_indexing(self):
        p = self.params
        for station, markup in enumerate([.89, .89, .79, .45, .45, .45]):
            for n in range(288):
                self.assertAlmostEqual(p.swap_service_price_at(station, n) - p.electricity_price_at(station, n), markup)
        self.assertEqual(p.electricity_price[0], p.electricity_price[1])
        self.assertEqual(p.electricity_price[2], p.electricity_price[5])
        self.assertEqual(p.electricity_price_at(0, 6 * 12 - 1), .34)
        self.assertEqual(p.electricity_price_at(0, 6 * 12), .67)
        self.assertEqual(p.electricity_price_at(2, 17 * 12), 1.20)

    def test_reproduction_serialization_and_noise_support(self):
        one = generate_synthetic_scenario(self.params, 101)
        self.assertEqual(one.to_dict(), generate_synthetic_scenario(self.params, 101).to_dict())
        self.assertEqual(len(one.reservations), 100)
        self.assertEqual(len(one.actual_random_requests), 99)
        for r in one.reservations:
            self.assertTrue(.5 < r["actual_entry_soc"] <= 1.)
            self.assertTrue(.5 <= r["entry_soc"] <= 1.)
            self.assertLessEqual(abs(r["actual_entry_time"] - r["entry_time"]), 1/6 + 1e-12)
            self.assertLessEqual(abs(r["actual_entry_soc"] - r["entry_soc"]), .05 + 1e-12)
            self.assertEqual(len(r["segment_time_multipliers"]), 13)
            self.assertTrue(all(.9 <= v <= 1.1 for v in r["segment_time_multipliers"]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            save_scenario(one, path)
            self.assertEqual(load_scenario(path).to_dict(), one.to_dict())

    def test_report_and_truth_reveal_are_separate(self):
        record = {"user_key": [1, 0], "od_id": 1, "entry_time": 1., "entry_soc": .7,
                  "actual_entry_time": 1.1, "actual_entry_soc": .68,
                  "segment_time_multipliers": [1.05] * 13}
        scenario = SyntheticScenario(self.params, [record], [])
        public = scenario.initial_reservations()[0]
        self.assertEqual(set(public), {"user_key", "od_id", "entry_time", "entry_soc"})
        self.assertNotIn("actual_entry_time", scenario.observation_at(1.05).reservations[0])
        observed = scenario.observation_at(1.1).reservations[0]
        self.assertEqual(observed["entry_time"], 1.)
        self.assertEqual(observed["actual_entry_time"], 1.1)
        self.assertEqual(observed["actual_entry_soc"], .68)
        self.assertNotIn("segment_time_multipliers", observed)
        record["actual_entry_time"] = .9
        scenario = SyntheticScenario(self.params, [record], [])
        self.assertIn("actual_entry_time", scenario.observation_at(.95).reservations[0])

    def test_changing_report_error_does_not_change_truth(self):
        one = generate_synthetic_scenario(self.params, 102)
        changed = copy.deepcopy(self.params)
        changed.entry_time_error_hours = .05
        changed.entry_soc_error = .01
        two = generate_synthetic_scenario(changed, 102)
        fields = ("od_id", "actual_entry_time", "actual_entry_soc", "segment_time_multipliers")
        keyed = lambda scenario: {tuple(r["user_key"]): {f:r[f] for f in fields} for r in scenario.reservations}
        self.assertEqual(keyed(one), keyed(two))
        self.assertEqual(one.actual_random_requests, two.actual_random_requests)

    def test_request_attributes_have_separate_random_sources(self):
        one = generate_synthetic_scenario(self.params, 103)
        changed = copy.deepcopy(self.params)
        changed.num_reservations = 2
        changed.random_return_soc_range = [.2, .25]
        two = generate_synthetic_scenario(changed, 103)
        project = lambda scenario: [(r["request_id"],r["station"],r["arrival_time"]) for r in scenario.actual_random_requests]
        self.assertEqual(project(one), project(two))
        self.assertNotEqual(one.actual_random_requests, two.actual_random_requests)
        self.assertTrue(all(.1 <= r["return_soc"] <= .3 for r in one.actual_random_requests))
        self.assertTrue(all(not 3 <= r["arrival_time"] < 7 for r in one.actual_random_requests))
        self.assertTrue(any(abs(r["arrival_time"] * 12 - round(r["arrival_time"] * 12)) > 1e-6 for r in one.actual_random_requests))

    def test_generator_does_not_reject_low_soc_or_rewrite_od(self):
        p = copy.deepcopy(self.params)
        # e2->e5 starts at km94 and first station is km255, which is unreachable
        # at this SOC. The user's independent input assumption must stay intact.
        target = next(i for i, od in enumerate(p.od_pairs) if od.entry_km == 94. and od.exit_km == 340.)
        p.od_sampling_weights = [float(i == target) for i in range(30)]
        p.reservation_entry_soc_range = [.5001, .501]
        scenario = generate_synthetic_scenario(p, 104)
        self.assertEqual(len(scenario.reservations), 100)
        self.assertEqual({r["od_id"] for r in scenario.reservations}, {p.od_pairs[target].od_id})
        self.assertTrue(all(.5001 < r["actual_entry_soc"] <= .501 for r in scenario.reservations))


if __name__ == "__main__":
    unittest.main()
