"""Independent orchestration audits; no optimization problem is solved."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import deadline_guard
import experiment_worker
from src.domain import MPCSolution, ServiceDecision
from src.experiment_control import ExperimentPaused
from src.parameters import BusinessParameters, ODPairParameters, StationParameters
from src.rolling_runner import run_rolling_mpc
from src.scenario import SyntheticScenario
from src.value_training import fit_value_model


def tiny_params():
    return BusinessParameters(num_periods=3, horizon=1, num_reservations=0,
        station=StationParameters(num_stations=1, station_ids=[0], positions_km=[5.],
            num_slots=1, initial_slot_soc=[[1.]], charging_efficiency=1.,
            slot_power_limits_kw=[[60.]], station_power_limits_kw=[60.]),
        od_pairs=[ODPairParameters(0, 0., 10., [0])], random_arrival_rate_per_hour=[0.])


class ExperimentIndependentAuditTests(unittest.TestCase):
    def test_deadline_guard_stops_descendants_but_keeps_its_own_process(self):
        parent = Mock(pid=100)
        child = Mock(pid=101)
        grandchild = Mock(pid=102)
        guard = Mock(pid=999)
        parent.create_time.return_value = 12.
        parent.children.return_value = [child, grandchild, guard]
        with patch("deadline_guard.psutil.Process", return_value=parent), \
             patch("os.getpid", return_value=999), \
             patch("deadline_guard.psutil.wait_procs", return_value=([parent, child], [grandchild])) as wait:
            self.assertTrue(deadline_guard.stop_tree(100, 12.))
        parent.children.assert_called_once_with(recursive=True)
        for process in (parent, child, grandchild):
            process.terminate.assert_called_once()
        guard.terminate.assert_not_called()
        grandchild.kill.assert_called_once()
        self.assertEqual({p.pid for p in wait.call_args.args[0]}, {100, 101, 102})

    def test_deadline_guard_does_not_kill_a_reused_pid(self):
        parent = Mock(pid=100)
        parent.create_time.return_value = 999.
        with patch("deadline_guard.psutil.Process", return_value=parent):
            self.assertFalse(deadline_guard.stop_tree(100, 12.))
        parent.children.assert_not_called()
        parent.terminate.assert_not_called()

    def test_worker_pause_and_artifact_failure_are_never_complete(self):
        params = tiny_params()
        scene = SyntheticScenario(params, [], [])
        for failure in ("paused", "artifact"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                argv = ["--scenario", str(output / "scenario.json"), "--output-dir", str(output),
                        "--deadline", "2099-01-01T00:00:00+00:00"]
                with patch("experiment_worker.load_scenario", return_value=scene), \
                     patch("experiment_worker.run_rolling_mpc") as runner, \
                     patch("experiment_worker.build_result_statistics") as statistics, \
                     patch("builtins.print"):
                    if failure == "paused":
                        runner.side_effect = ExperimentPaused("audit pause")
                    else:
                        runner.return_value = {"completed": True}
                        statistics.side_effect = ValueError("audit artifact failure")
                    code = experiment_worker.main(argv)
                status = json.loads((output / "worker_status.json").read_text())
                self.assertEqual(status["state"], "paused" if failure == "paused" else "failed")
                self.assertEqual(code, 75 if failure == "paused" else 1)
                self.assertFalse((output / "metrics.json").exists())

    def test_resume_preserves_nonzero_realized_service_reward_once(self):
        params = tiny_params()
        scene = SyntheticScenario(params, [], [
            {"request_id": "audit-random", "station": 0, "arrival_time": 0., "return_soc": .2}])
        calls = []
        def solve(p, window, **kwargs):
            calls.append(window.ell)
            services = [ServiceDecision("audit-random", 0, 0, 0)] if window.ell == 0 else []
            return MPCSolution("optimal", 0., {}, {}, services, [[[0.] * window.horizon]], [])
        def interrupt(record):
            raise ExperimentPaused("audit after committed service")
        with tempfile.TemporaryDirectory() as directory, patch("src.rolling_runner.solve_mpc", side_effect=solve):
            with self.assertRaises(ExperimentPaused):
                run_rolling_mpc(params, scene, journal_dir=directory, progress=interrupt)
            resumed = run_rolling_mpc(params, scene, journal_dir=directory)
            self.assertEqual(calls, [0, 1, 2])
            full = run_rolling_mpc(params, scene)
        self.assertGreater(resumed["summary"]["total_reward"], 0.)
        self.assertEqual(resumed["summary"]["total_reward"], full["summary"]["total_reward"])
        self.assertEqual(resumed["ledger"], full["ledger"])
        services = [e for e in resumed["ledger"] if e["type"] == "random_service"]
        self.assertEqual(len(services), 1)

    def test_changed_terminal_input_schema_is_refused_on_resume(self):
        params = tiny_params()
        scene = SyntheticScenario(params, [], [])
        class Spec:
            variant = "audit"
            names = ["one"]
            def encode_window(self, window):
                return [1.]
        spec = Spec()
        def solve(p, window, **kwargs):
            return MPCSolution("optimal", 0., {}, {}, [], [[[0.] * window.horizon]], [])
        def interrupt(record):
            raise ExperimentPaused("audit")
        with tempfile.TemporaryDirectory() as directory, patch("src.rolling_runner.solve_mpc", side_effect=solve):
            with self.assertRaises(ExperimentPaused):
                run_rolling_mpc(params, scene, feature_spec=spec, journal_dir=directory, progress=interrupt)
            spec.names = ["different_same_dimension"]
            with self.assertRaisesRegex(ValueError, "changed inputs or model"):
                run_rolling_mpc(params, scene, feature_spec=spec, journal_dir=directory)


    def test_completed_suite_cache_rejects_changed_scenario_truth(self):
        from src.experiment_control import atomic_json
        from src.experiment_suite import _Suite
        params = tiny_params()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            params.save_json(config)
            worker = Mock(return_value={"worker_status": "complete",
                "metrics": {"completed_periods": 3, "net_profit_yuan": 17.}})
            suite = _Suite(config, root, None, worker,
                [{"completed_periods": 3, "worker_wall_seconds": 60.}] * 3)
            suite._job("base", params, 41000, "test/zero")
            path = root / "formal_suite/scenarios/base/41000.json"
            payload = json.loads(path.read_text())
            payload["actual_random_requests"].append(
                {"request_id": "changed-truth", "station": 0, "arrival_time": .01, "return_soc": .2})
            atomic_json(path, payload)
            with self.assertRaisesRegex(ValueError, "inputs have changed"):
                suite._job("base", params, 41000, "test/zero")
            self.assertEqual(worker.call_count, 1)

    def test_fitted_value_binds_physical_configuration_but_allows_horizon_change(self):
        from src.terminal_features import FeatureSpec
        from src.terminal_value import terminal_configuration_fingerprint
        params = tiny_params()
        spec = FeatureSpec(params, variant="inventory_only")
        x = np.zeros((3, spec.dimension))
        model, _ = fit_value_model(x, [1000., 2000., 3000.], spec, kind="linear")
        self.assertEqual(model.configuration_fingerprint, terminal_configuration_fingerprint(params))
        changed = copy.deepcopy(params)
        changed.horizon = 2
        self.assertEqual(model.configuration_fingerprint, terminal_configuration_fingerprint(changed))
        changed.station.station_power_limits_kw = [30.]
        self.assertNotEqual(model.configuration_fingerprint, terminal_configuration_fingerprint(changed))

    def test_linear_fit_checks_deadline_after_blocking_regression(self):
        class Spec:
            names = ["x"]
            dimension = 1
            variant = "full"
        with patch("src.value_training.check_deadline", side_effect=[None, ExperimentPaused("audit expired")]):
            with self.assertRaises(ExperimentPaused):
                fit_value_model(np.array([[0.], [1.], [2.]]), [1000., 2000., 3000.], Spec(), kind="linear")


if __name__ == "__main__":
    unittest.main()
