"""Entrypoint fixtures use temporary outputs and a mocked optimizer only."""
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import experiment_worker
import run_experiments
from src.domain import MPCSolution
from src.experiment_control import atomic_json
from src.experiment_recovery import ExperimentAlreadyRunning
from src.parameters import BusinessParameters, ODPairParameters, StationParameters
from src.scenario import SyntheticScenario


class ExperimentEntrypointTests(unittest.TestCase):
    def test_live_controller_refuses_before_status_write_or_process_launch(self):
        for state in ("alive", "unknown"):
            with self.subTest(identity=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prior = {"state": "running", "pid": os.getpid() + 100000, "created": 123.,
                         "started_at": "2026-09-14T02:00:00+08:00", "fixture": True}
                target = root / "controller_status.json"
                atomic_json(target, prior)
                before = target.read_bytes()
                with patch("run_experiments.inspect_process_identity", return_value={"state": state}), \
                     patch("run_experiments.atomic_json") as writer, \
                     patch("run_experiments.subprocess.Popen") as launcher:
                    with self.assertRaises(ExperimentAlreadyRunning):
                        run_experiments.main(["--config", str(root / "unused_config.json"),
                            "--output-root", str(root), "--stage", "pilot",
                            "--deadline", "2099-01-01T00:00:00+00:00"])
                    writer.assert_not_called()
                    launcher.assert_not_called()
                self.assertEqual(target.read_bytes(), before)

    def test_live_worker_refuses_before_status_write_or_owned_execution(self):
        for state in ("alive", "unknown"):
            with self.subTest(identity=state), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                prior = {"state": "running", "pid": os.getpid() + 100000, "created": 123.,
                         "started_at": "2026-09-14T02:00:00+08:00", "fixture": True}
                target = root / "worker_status.json"
                atomic_json(target, prior)
                before = target.read_bytes()
                with patch("experiment_worker.inspect_process_identity", return_value={"state": state}), \
                     patch("experiment_worker.atomic_json") as writer, \
                     patch("experiment_worker._run_owned") as owned:
                    with self.assertRaises(ExperimentAlreadyRunning):
                        experiment_worker.main(["--scenario", str(root / "unused_scenario.json"),
                            "--output-dir", str(root), "--deadline", "2099-01-01T00:00:00+00:00"])
                    writer.assert_not_called()
                    owned.assert_not_called()
                self.assertEqual(target.read_bytes(), before)

    def _complete_fixture(self, history):
        params = BusinessParameters(num_periods=2, horizon=1, num_reservations=0,
            station=StationParameters(num_stations=1, station_ids=[0], positions_km=[5.],
                num_slots=1, initial_slot_soc=[[1.]], charging_efficiency=1.,
                slot_power_limits_kw=[[60.]], station_power_limits_kw=[60.]),
            od_pairs=[ODPairParameters(0, 0., 10., [0])], random_arrival_rate_per_hour=[0.])
        scenario = SyntheticScenario(params, [], [])
        def solver(p, window, **kwargs):
            return MPCSolution("optimal", 0., {}, {}, [], [[[0.] * window.horizon]], [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in history.items():
                atomic_json(root / "attempt_history" / name, value)
            with ExitStack() as stack:
                optimizer = stack.enter_context(patch("src.rolling_runner.solve_mpc", side_effect=solver))
                stack.enter_context(patch("experiment_worker.load_scenario", return_value=scenario))
                stack.enter_context(patch("experiment_worker._capture_worker_source",
                    return_value=(str(root / "fixture_source"), {})))
                stack.enter_context(patch("experiment_worker.build_result_statistics", return_value={"fixture": True}))
                statistics = stack.enter_context(patch("experiment_worker.write_statistics_artifacts"))
                stack.enter_context(patch("experiment_worker.perf_counter", side_effect=[100., 105., 105.]))
                stack.enter_context(patch("builtins.print"))
                code = experiment_worker.main(["--scenario", str(root / "fixture_scenario.json"),
                    "--output-dir", str(root), "--deadline", "2099-01-01T00:00:00+00:00"])
            self.assertEqual(code, 0)
            self.assertEqual(optimizer.call_count, 2)
            statistics.assert_called_once()
            status = json.loads((root / "worker_status.json").read_text())
            metrics = json.loads((root / "metrics.json").read_text())
            self.assertEqual(status["state"], "complete")
            self.assertTrue((root / "result.json.gz").exists())
            self.assertEqual(metrics["completed_periods"], 2)
            self.assertEqual(metrics["current_attempt_wall_seconds"], 5.)
            return metrics, status

    def test_completed_worker_deduplicates_same_historical_attempt(self):
        prior = {"state": "paused", "pid": 501, "started_at": "2026-09-14T02:00:00+08:00",
                 "wall_seconds": 60., "prior_attempt_seconds": 99999., "fixture": True}
        second = dict(prior, pid=502, started_at="2026-09-14T02:10:00+08:00", wall_seconds=20.)
        metrics, _ = self._complete_fixture({"attempt_001.json": prior,
            "attempt_duplicate.json": deepcopy(prior), "attempt_other.json": second})
        self.assertEqual(metrics["prior_attempt_wall_seconds"], 80.)
        self.assertEqual(metrics["known_runtime_seconds"], 85.)
        self.assertEqual(metrics["worker_wall_seconds"], 85.)
        self.assertFalse(metrics["runtime_timing_incomplete"])

    def test_completed_worker_keeps_unknown_prior_duration_separate(self):
        known = {"state": "paused", "pid": 501, "started_at": "2026-09-14T02:00:00+08:00",
                 "wall_seconds": 60., "fixture": True}
        unknown = {"state": "running", "pid": 502, "started_at": "2026-09-14T03:00:00+08:00",
                   "deadline": "2026-09-14T07:00:00+08:00", "fixture": True}
        metrics, status = self._complete_fixture({"known.json": known, "unknown.json": unknown})
        self.assertEqual(metrics["prior_attempt_wall_seconds"], 60.)
        self.assertEqual(metrics["known_runtime_seconds"], 65.)
        self.assertIsNone(metrics["worker_wall_seconds"])
        self.assertTrue(metrics["runtime_timing_incomplete"])
        self.assertTrue(status["runtime_timing_incomplete"])


    def test_conflicting_duplicate_duration_refuses_before_new_worker_status(self):
        prior = {"state": "paused", "pid": 501, "started_at": "2026-09-14T02:00:00+08:00",
                 "wall_seconds": 60., "fixture": True}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "attempt_history" / "one.json", prior)
            atomic_json(root / "attempt_history" / "two.json", dict(prior, wall_seconds=61.))
            with patch("experiment_worker.atomic_json") as writer, \
                 patch("experiment_worker.load_scenario") as loader:
                with self.assertRaisesRegex(ValueError, "conflicting durations"):
                    experiment_worker.main(["--scenario", str(root / "unused_scenario.json"),
                        "--output-dir", str(root), "--deadline", "2099-01-01T00:00:00+00:00"])
                writer.assert_not_called()
                loader.assert_not_called()

    def test_near_identical_duplicates_prefer_measured_over_estimated_runtime(self):
        measured = {"state": "paused", "pid": 501, "started_at": "2026-09-14T02:00:00+08:00",
                    "wall_seconds": 60., "timing_incomplete": False, "wall_seconds_estimated": False,
                    "fixture": True}
        estimated = dict(measured, wall_seconds=60.01, wall_seconds_estimated=True)
        metrics, _ = self._complete_fixture({"a_estimated.json": estimated, "b_measured.json": measured})
        self.assertEqual(metrics["prior_attempt_wall_seconds"], 60.)
        self.assertEqual(metrics["worker_wall_seconds"], 65.)
        self.assertFalse(metrics["runtime_timing_estimated"])

    def test_later_sealed_duplicate_resolves_old_missing_timing(self):
        raw = {"state": "running", "pid": 501, "started_at": "2026-09-14T02:00:00+08:00",
               "deadline": "2026-09-14T07:00:00+08:00", "fixture": True}
        sealed = dict(raw, state="paused", finished_at="2026-09-14T07:00:00+08:00",
                      wall_seconds=18000., timing_incomplete=False, wall_seconds_estimated=True,
                      timing_source="legacy_guard_observed_at_estimate")
        metrics, _ = self._complete_fixture({"attempt_001.json": raw, "attempt_sealed.json": sealed})
        self.assertEqual(metrics["prior_attempt_wall_seconds"], 18000.)
        self.assertEqual(metrics["worker_wall_seconds"], 18005.)
        self.assertFalse(metrics["runtime_timing_incomplete"])
        self.assertTrue(metrics["runtime_timing_estimated"])


if __name__ == "__main__":
    unittest.main()
