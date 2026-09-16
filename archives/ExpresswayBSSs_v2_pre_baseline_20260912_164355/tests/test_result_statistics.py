from __future__ import annotations

import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.result_statistics import (
    DEFAULT_RESULT_PATH,
    StatisticsError,
    build_result_statistics,
    write_statistics_artifacts,
)


_OUTPUT_DIR = DEFAULT_RESULT_PATH.parent


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class ResultStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = _load(DEFAULT_RESULT_PATH)
        cls.mock = _load(_OUTPUT_DIR / "mock_rl_data.json")
        cls.plan = _load(_OUTPUT_DIR / "dayahead_plan.json")
        cls.statistics = build_result_statistics(
            cls.result,
            mock=cls.mock,
            plan=cls.plan,
            strict=True,
        )

    def test_default_result_has_six_station_and_twelve_period_rows(self) -> None:
        statistics = self.statistics
        self.assertEqual(len(statistics["by_station"]), 6)
        self.assertEqual(
            [row["station"] for row in statistics["by_station"]],
            list(range(6)),
        )
        self.assertEqual(len(statistics["by_period"]), 12)
        self.assertTrue(statistics["checks"]["all_evaluated_checks_passed"])
        self.assertTrue(statistics["checks"]["station_energy_limits_satisfied"])
        self.assertTrue(statistics["checks"]["station_inventory_energy_balanced"])
        path = statistics["path_optimization"]
        self.assertEqual(path["status"], "PAPER_GUROBI_OPTIMAL")
        self.assertEqual(path["method"], "gurobi_joint_paper_milp")
        self.assertTrue(path["global_milp_optimality_claimed"])
        self.assertGreater(path["publication_count"], 0)
        self.assertEqual(
            path["publication_count"],
            statistics["realized_summary"]["path_publications"],
        )
        self.assertEqual(
            path["actual_service_station_sequences"]["1:0"],
            [1, 3, 5],
        )

    def test_ledger_is_independently_reconciled_by_station_and_period(self) -> None:
        postings = self.result["ledger"]["postings"]
        components = (
            "income_reservation",
            "income_random",
            "charging_cost",
            "adjustment_cost",
            "reservation_failure_cost",
            "reward_delta",
        )
        for component in components:
            expected = sum(float(posting[component]) for posting in postings)
            self.assertAlmostEqual(
                self.statistics["realized_summary"][component],
                expected,
                places=8,
            )
            period_total = sum(
                float(row[component]) for row in self.statistics["by_period"]
            )
            self.assertAlmostEqual(period_total, expected, places=8)

        for station_row in self.statistics["by_station"]:
            station = station_row["station"]
            station_postings = [
                posting
                for posting in postings
                if posting["entry"].get("station") == station
            ]
            expected_energy = sum(
                float(posting["entry"]["energy_kwh"])
                for posting in station_postings
                if posting["entry"]["event_type"] == "charging"
            )
            self.assertAlmostEqual(
                station_row["charging_energy_kwh"], expected_energy, places=8
            )
            self.assertLessEqual(
                abs(station_row["inventory_balance_residual_kwh"]), 1e-6
            )

    def test_prediction_objectives_never_change_realized_statistics(self) -> None:
        modified = copy.deepcopy(self.result)
        for round_record in modified["rounds"]:
            round_record["prediction"]["objective_total"] = 1e12
            round_record["prediction"]["terminal_value"] = -1e12
        changed = build_result_statistics(
            modified,
            mock=self.mock,
            plan=self.plan,
            strict=True,
        )
        self.assertEqual(
            changed["realized_summary"], self.statistics["realized_summary"]
        )
        self.assertFalse(changed["diagnostics"]["prediction_objectives_aggregated"])

    def test_strict_mode_rejects_duplicate_realized_event_id(self) -> None:
        modified = copy.deepcopy(self.result)
        modified["ledger"]["postings"].append(
            copy.deepcopy(modified["ledger"]["postings"][0])
        )
        with self.assertRaisesRegex(StatisticsError, "unique_realized_event_ids"):
            build_result_statistics(
                modified,
                mock=self.mock,
                plan=self.plan,
                strict=True,
            )

    def test_strict_mode_rejects_nonphysical_final_slot_state(self) -> None:
        modified = copy.deepcopy(self.result)
        # Changing only readiness keeps the SOC energy balance unchanged, so
        # the explicit slot-state invariant must catch this corruption.
        cell = modified["final_state"]["slots"][0][0]
        self.assertLess(cell["soc"], 1.0)
        cell["ready"] = True
        with self.assertRaisesRegex(StatisticsError, "final_slot_state_physical"):
            build_result_statistics(
                modified,
                mock=self.mock,
                plan=self.plan,
                strict=True,
            )

    def test_strict_mode_rejects_reverse_charging_segment(self) -> None:
        modified = copy.deepcopy(self.result)
        posting = next(
            item
            for item in modified["ledger"]["postings"]
            if item["entry"]["event_type"] == "charging"
        )
        posting["entry"]["metadata"]["start_time"] = (
            posting["entry"]["metadata"]["end_time"] + 1.0
        )
        with self.assertRaisesRegex(StatisticsError, "charging_segments_physical"):
            build_result_statistics(
                modified,
                mock=self.mock,
                plan=self.plan,
                strict=True,
            )

    def test_strict_mode_rejects_negative_service_wait(self) -> None:
        modified = copy.deepcopy(self.result)
        posting = next(
            item
            for item in modified["ledger"]["postings"]
            if item["entry"]["event_type"] == "reservation_service"
        )
        posting["entry"]["occurred_at"] = posting["entry"]["arrival_time"] - 0.01
        posting["entry"]["metadata"]["wait_hours"] = -0.01
        with self.assertRaisesRegex(StatisticsError, "request_wait_times_physical"):
            build_result_statistics(
                modified,
                mock=self.mock,
                plan=self.plan,
                strict=True,
            )

    def test_strict_mode_recomputes_prices_instead_of_trusting_summaries(self) -> None:
        modified = copy.deepcopy(self.result)
        posting = next(
            item
            for item in modified["ledger"]["postings"]
            if item["entry"]["event_type"] == "charging"
        )
        period = posting["entry"]["interval"]
        posting["charging_cost"] += 10.0
        posting["reward_delta"] -= 10.0
        modified["ledger"]["summary"]["charging_cost"] += 10.0
        modified["ledger"]["summary"]["reward_delta"] -= 10.0
        modified["summary"]["total_charging_cost"] += 10.0
        modified["summary"]["total_reward_delta"] -= 10.0
        modified["summary"]["total_reward"] -= 10.0
        modified["rounds"][period]["ledger"]["charging_cost"] += 10.0
        modified["rounds"][period]["ledger"]["reward_delta"] -= 10.0
        with self.assertRaisesRegex(
            StatisticsError, "ledger_components_recomputed_from_parameters"
        ):
            build_result_statistics(
                modified,
                mock=self.mock,
                plan=self.plan,
                strict=True,
            )

    def test_strict_mode_rejects_conflicting_request_outcomes(self) -> None:
        modified = copy.deepcopy(self.result)
        random_services = [
            item
            for item in modified["ledger"]["postings"]
            if item["entry"]["event_type"] == "random_service"
        ]
        duplicate_request_id = random_services[0]["entry"]["metadata"][
            "request_event_id"
        ]
        random_services[1]["entry"]["metadata"][
            "request_event_id"
        ] = duplicate_request_id
        with self.assertRaisesRegex(
            StatisticsError, "request_outcome_ids_mutually_exclusive"
        ):
            build_result_statistics(
                modified,
                mock=self.mock,
                plan=self.plan,
                strict=True,
            )

    def test_sidecar_files_are_reloadable_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = write_statistics_artifacts(
                self.statistics,
                temporary_directory,
                stem="test_run",
            )
            self.assertEqual(_load(paths["json"]), self.statistics)
            with paths["station_csv"].open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                station_rows = list(csv.DictReader(handle))
            with paths["period_csv"].open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                period_rows = list(csv.DictReader(handle))
            self.assertEqual(len(station_rows), 6)
            self.assertEqual(len(period_rows), 12)
            markdown = paths["markdown"].read_text(encoding="utf-8")
            self.assertIn("# 六站 Mock MPC 运行统计", markdown)
            self.assertIn("滚动预测窗口互相重叠", markdown)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
