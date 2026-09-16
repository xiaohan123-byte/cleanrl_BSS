from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cost_stress_scenarios import (
    NUM_STATIONS,
    RESERVATION_FAILURE_PENALTY,
    build_cost_stress_results,
    run_reservation_timeout_scenario,
    run_waiting_then_service_scenario,
    write_cost_stress_artifacts,
)


class CostStressScenariosTest(unittest.TestCase):
    def test_waiting_request_is_served_after_slot_fills(self) -> None:
        result = run_waiting_then_service_scenario()

        self.assertEqual(result["input"]["num_stations"], NUM_STATIONS)
        self.assertEqual(result["observed"]["service_count"], 1)
        self.assertEqual(result["observed"]["timeout_count"], 0)
        self.assertAlmostEqual(
            result["observed"]["service_wait_hours"][0],
            0.17543859649122806,
            places=10,
        )
        self.assertGreater(result["observed"]["service_wait_hours"][0], 0.0)
        self.assertEqual(
            result["observed"]["financial"]["reservation_failure_cost"], 0.0
        )
        self.assertFalse(result["observed"]["waiting_cost_defined"])
        self.assertTrue(result["checks"]["all_passed"])

    def test_zero_power_request_times_out_and_costs_1000(self) -> None:
        result = run_reservation_timeout_scenario()

        self.assertEqual(result["input"]["num_stations"], NUM_STATIONS)
        self.assertEqual(result["observed"]["service_count"], 0)
        self.assertEqual(result["observed"]["timeout_count"], 1)
        self.assertEqual(result["observed"]["timeout_wait_hours"], [0.25])
        self.assertEqual(
            result["observed"]["financial"]["reservation_failure_cost"],
            RESERVATION_FAILURE_PENALTY,
        )
        self.assertEqual(
            result["observed"]["financial"]["reward_delta"],
            -RESERVATION_FAILURE_PENALTY,
        )
        self.assertTrue(result["checks"]["all_passed"])

    def test_artifacts_are_json_round_trippable_and_report_both_cases(self) -> None:
        results = build_cost_stress_results()
        with tempfile.TemporaryDirectory() as directory:
            paths = write_cost_stress_artifacts(results, directory)
            saved = json.loads(paths["json"].read_text(encoding="utf-8"))
            markdown = paths["markdown"].read_text(encoding="utf-8")

        self.assertTrue(saved["checks"]["all_scenarios_passed"])
        self.assertEqual(saved["num_stations"], NUM_STATIONS)
        self.assertEqual(len(saved["scenarios"]), 2)
        self.assertIn("0.175438596491", markdown)
        self.assertIn("1000.00", markdown)
        self.assertEqual(paths["json"].name, "cost_stress_test_results.json")
        self.assertEqual(paths["markdown"].name, "cost_stress_test_results.md")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
