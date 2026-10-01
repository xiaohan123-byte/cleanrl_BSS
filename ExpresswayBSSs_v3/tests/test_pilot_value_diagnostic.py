"""Cheap synthetic journals test pilot diagnostics without COPT or torch fitting."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import diagnose_pilot_value_fit as diagnostic
from src.accounting import COMPONENTS, event_components, summarize_ledger
from src.candidate_network import generate_candidate_network
from src.domain import RollingState
from src.experiment_control import atomic_json, fingerprint
from src.terminal_value import TerminalValueModel
from tests.test_terminal_value import params


def fixture(root):
    p = params(stations=1)
    p.num_periods = 288
    p.horizon = 72
    p.electricity_price = [[.5] * 288]
    p.swap_service_price = [[1.2] * 288]
    p.solver.time_limit_sec = 60.
    p.validate()
    for seed in diagnostic.SEEDS:
        directory = root / "pilot" / f"seed_{seed}"
        journal = directory / "journal"
        journal.mkdir(parents=True)
        identity = {"parameters": p.to_dict(), "terminal_model": None,
                    "route_mode": "joint", "charging_mode": "joint"}
        header = {"identity": identity, "fingerprint": fingerprint(identity)}
        atomic_json(journal / "initial.json", header)
        period_statistics = []
        all_events = []
        with (journal / "rounds.jsonl").open("w", encoding="utf-8") as stream:
            for n in range(288):
                events = []
                # Nonzero first and final rewards exercise MC suffix sums and the exact L boundary.
                if n in (0, 287):
                    event = {"event_id": f"adjust:{n}", "type": "path_adjustment",
                             "user_key": "fixture", "period": n, "time": n / 12}
                    event.update(event_components(p, event))
                    events.append(event)
                all_events.extend(events)
                summary = summarize_ledger(events)
                period_statistics.append(dict(summary, period=n))
                state = RollingState(n, [[1.]])
                before = state.to_dict()
                before.pop("ledger")
                before["ledger_event_count"] = len(all_events) - len(events)
                horizon = min(72, 288 - n)
                row = {"period": n, "horizon": horizon, "state_before": before,
                       "forecast": {"random_requests": [], "start_time": n / 12,
                                    "end_time": (n + horizon) / 12},
                       "events": events, "reward": summary["total_reward"]}
                stream.write(json.dumps(row) + "\n")
        summary = summarize_ledger(all_events)
        metrics = {"completed_periods": 288, "operating_duration_hours": 24.,
                   "terminal_settlement": "none", "net_profit_yuan": summary["total_reward"]}
        atomic_json(journal / "status.json", {"state": "complete", "completed_periods": 288,
                    "fingerprint": header["fingerprint"], "summary": summary})
        atomic_json(directory / "worker_status.json", {"state": "complete", "model": None,
                    "route_mode": "joint", "charging_mode": "joint", "metrics": metrics})
        atomic_json(directory / "metrics.json", metrics)
        atomic_json(directory / "statistics.json", {"summary": summary, "per_period": period_statistics,
                    "checks": dict.fromkeys(diagnostic.REQUIRED_CHECKS, True)})
        atomic_json(directory / "network.json", generate_candidate_network(p))
    return p


class PilotValueDiagnosticTests(unittest.TestCase):
    def test_incomplete_pilot_rejected_before_missing_metrics_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pilot = root / "pilot/seed_101"
            atomic_json(pilot / "worker_status.json", {"state": "running"})
            atomic_json(pilot / "journal/status.json", {"state": "running"})
            with self.assertRaisesRegex(ValueError, "partial"):
                diagnostic.prepare_dataset(root, deadline=None)
            self.assertFalse((root / "pilot_only_value_diagnostics/dataset").exists())

    def test_complete_data_uses_same_features_mc_and_immutable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            original = {path: diagnostic._sha(path) for path in (root / "pilot").rglob("*") if path.is_file()}
            with patch("diagnose_terminal_window.solve_mpc") as solve:
                manifest = diagnostic.prepare_dataset(root, deadline=None)
                solve.assert_not_called()
            data = root / "pilot_only_value_diagnostics/dataset"
            self.assertEqual(manifest["samples"], 864)
            self.assertEqual(manifest["complete_days"], 3)
            x = np.load(data / "features.npy")
            y = np.load(data / "returns_yuan.npy")
            self.assertEqual(x.shape, (864, manifest["feature_dimension"]))
            for day in range(3):
                self.assertAlmostEqual(y[day * 288], -2.)
                np.testing.assert_allclose(y[day * 288 + 1:(day + 1) * 288], -1.)
            self.assertTrue(all(diagnostic._sha(path) == digest for path, digest in original.items()))
            with self.assertRaises(FileExistsError):
                diagnostic.prepare_dataset(root, deadline=None)
            diagnostic._load_dataset(root)

    def test_financial_mismatch_and_incomplete_final_line_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "pilot/seed_103/metrics.json"
            metrics = diagnostic._json(path)
            metrics["net_profit_yuan"] += 1.
            atomic_json(path, metrics)
            with self.assertRaisesRegex(ValueError, "reconcile"):
                diagnostic.prepare_dataset(root, deadline=None)
            metrics["net_profit_yuan"] -= 1.
            atomic_json(path, metrics)
            rounds = root / "pilot/seed_103/journal/rounds.jsonl"
            rounds.write_bytes(rounds.read_bytes().rstrip(b"\n"))
            before = rounds.read_bytes()
            with self.assertRaisesRegex(ValueError, "partial last"):
                diagnostic.prepare_dataset(root, deadline=None)
            self.assertEqual(rounds.read_bytes(), before)

    def test_fit_protocol_is_fixed_and_model_wrapper_rejects_formal_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            diagnostic.prepare_dataset(root, deadline=None)
            calls = []
            def fake_fit(x, y, spec, **kwargs):
                calls.append(kwargs)
                kind = kwargs["kind"]
                extras = {} if kind == "linear" else {"hidden_weights": np.zeros((16, spec.dimension)),
                        "hidden_bias": np.zeros(16), "output_weights": np.zeros(16)}
                model = TerminalValueModel(kind, spec.names, "full", np.zeros(spec.dimension), **extras)
                return model, {"epochs": 200 if kind == "relu" else 1, "training_rmse_yuan": 1.}
            with patch("diagnose_pilot_value_fit.fit_value_model", side_effect=fake_fit), \
                 patch("diagnose_terminal_window.solve_mpc") as solve:
                results = diagnostic.fit_models(root, deadline=None)
                solve.assert_not_called()
            self.assertEqual([call["kind"] for call in calls], ["linear", "relu"])
            self.assertTrue(all(call["epochs"] == 200 and call["previous"] is None for call in calls))
            self.assertTrue(all(result["scope"] == "pilot_only" for result in results))
            path = root / "pilot_only_value_diagnostics/models/pilot_only_relu.json"
            envelope = diagnostic._json(path)
            self.assertFalse(envelope["is_formal_model"])
            self.assertEqual(TerminalValueModel.from_dict(envelope["model"]).hidden_units, 16)
            with self.assertRaises((TypeError, ValueError)):
                TerminalValueModel.load(path)
            with patch("diagnose_pilot_value_fit.run_comparison", return_value={"state": "complete"}) as compare:
                diagnostic.embed_models(root, deadline=None)
            args, kwargs = compare.call_args
            self.assertEqual(args[0].solver.time_limit_sec, 60.)
            self.assertEqual(args[1].ell, 72)
            self.assertEqual(set(kwargs["supplied_models"]), {"linear", "relu"})
            self.assertEqual(kwargs["metadata"]["scope"], "pilot_only")

    def test_rejects_formal_or_source_output_locations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target in (root / "pilot", root / "formal_suite", root.parent / "pilot_only_elsewhere"):
                with self.assertRaises(ValueError):
                    diagnostic._output_path(root, target)


if __name__ == "__main__":
    unittest.main()
