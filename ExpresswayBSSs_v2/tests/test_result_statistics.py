import json
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.domain import MPCSolution, ServiceDecision
from src.parameters import BusinessParameters
from src.result_statistics import StatisticsError, build_result_statistics, write_statistics_artifacts
from src.rolling_runner import run_rolling_mpc
from src.scenario import SyntheticScenario


def miniature_result():
    p = BusinessParameters(num_periods=3, horizon=2, num_reservations=0,
                           random_arrival_rate_per_hour=[0.] * 6)
    scenario = SyntheticScenario(p, [], [
        dict(request_id="R", station=0, arrival_time=0., return_soc=.2)])
    def scripted(params, window):
        services = [ServiceDecision("R", 0, 0, 0)] if window.ell == 0 else []
        power = [[[0.] * window.horizon for _ in range(5)] for _ in range(6)]
        if window.ell == 0:
            power[0][0][0] = 60.
        soc = [[[value] * (window.horizon + 1) for value in row] for row in window.state.slot_soc]
        if window.ell == 0:
            soc[0][0][1:] = [.2475] * window.horizon
        return MPCSolution("optimal", 1e9, {"income": 1e9}, {}, services, power, soc)
    with patch("src.rolling_runner.solve_mpc", side_effect=scripted):
        return run_rolling_mpc(p, scenario)


class ResultStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.result = miniature_result()

    def test_ledger_reconciliation_excludes_overlapping_forecast_income(self):
        stats = build_result_statistics(self.result)
        self.assertAlmostEqual(stats["summary"]["income"], 96.)
        self.assertAlmostEqual(stats["summary"]["charging_cost"], 1.75)
        self.assertAlmostEqual(stats["summary"]["total_reward"], 94.25)
        self.assertEqual(stats["summary"]["random_services"], 1)
        self.assertTrue(all(stats["checks"].values()))
        self.assertAlmostEqual(sum(row["income"] for row in stats["per_station"]), 96.)

    def test_duplicate_and_tampered_entries_rejected(self):
        duplicated = deepcopy(self.result)
        duplicated["ledger"].append(deepcopy(duplicated["ledger"][0]))
        with self.assertRaises(StatisticsError):
            build_result_statistics(duplicated)
        for field in ("energy_kwh", "unit_price", "end_soc"):
            modified = deepcopy(self.result)
            charging = next(row for row in modified["ledger"] if row["type"] == "charging" and row["power_kw"] > 0)
            charging[field] += .1
            with self.assertRaises(StatisticsError):
                build_result_statistics(modified)

    def test_bad_summary_or_inventory_cannot_pass(self):
        modified = deepcopy(self.result)
        modified["summary"]["total_reward"] += 1.
        with self.assertRaises(StatisticsError):
            build_result_statistics(modified)
        modified = deepcopy(self.result)
        modified["final_state"]["slot_soc"][0][0] = .5
        with self.assertRaises(StatisticsError):
            build_result_statistics(modified)

    def test_changed_solver_slot_power_or_carried_soc_is_rejected(self):
        modified = deepcopy(self.result)
        modified["rounds"][0]["solution"]["services"][0]["slot"] = 1
        with self.assertRaisesRegex(StatisticsError, "assignments"):
            build_result_statistics(modified)
        modified = deepcopy(self.result)
        modified["rounds"][0]["solution"]["power"][0][0][0] = 50.
        with self.assertRaisesRegex(StatisticsError, "solver power"):
            build_result_statistics(modified)
        modified = deepcopy(self.result)
        modified["rounds"][1]["state_before"]["slot_soc"][0][0] = .9
        with self.assertRaisesRegex(StatisticsError, "carried SOC"):
            build_result_statistics(modified)
        modified = deepcopy(self.result)
        modified["rounds"][0]["solution"]["soc"][0][0][1] = .3
        with self.assertRaisesRegex(StatisticsError, "predicted SOC"):
            build_result_statistics(modified)

    def test_station_limit_is_checked_independently(self):
        modified = deepcopy(self.result)
        modified["parameter_snapshot"]["station"]["station_power_limits_kw"][0] = 50.
        with self.assertRaisesRegex(StatisticsError, "station charging power"):
            build_result_statistics(modified)

    def test_all_four_standalone_artifacts_are_written(self):
        stats = build_result_statistics(self.result)
        with TemporaryDirectory() as directory:
            paths = write_statistics_artifacts(stats, directory, "review")
            self.assertEqual(set(paths), {"json", "markdown", "station_csv", "period_csv"})
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            loaded = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(loaded["summary"], stats["summary"])
            self.assertIn("Net realised reward", Path(paths["markdown"]).read_text(encoding="utf-8"))
            self.assertEqual(len(Path(paths["period_csv"]).read_text(encoding="utf-8-sig").splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
