"""Small-array coverage fixtures; no solver, training, or real pilot reads."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_value_coverage import (TrainingRange, analyze_coverage, analyze_files, iter_query_records,
                                    load_training_range)
from src.experiment_control import atomic_json, fingerprint


def training(scope="pilot_only"):
    return TrainingRange(["a", "b", "c"], [0., 2., 1.], [1., 2., 3.], 2, {"scope": scope})


def range_record():
    return {"feature_names": ["a", "b", "c"], "minimum": [0., 2., 1.], "maximum": [1., 2., 3.],
            "samples": 2, "source": "current_training_batch_only", "scaling": "fixed_business_scales",
            "usage": "coverage_diagnostic_only"}


def diagnostic_fixture(root, *, boundary=False, completed=True):
    atomic_json(root / "feature_spec.json", {"names": ["c", "a", "b"], "dimension": 3})
    entries = [{"method": kind, "feasible_incumbent": kind != "failed"}
               for kind in ("zero", "linear", "relu", "failed")]
    atomic_json(root / "diagnostic_results.json", {
        "state": "complete_with_diagnostic_failures" if completed else "running",
        "period": 14 if boundary else 6, "horizon": 6, "methods": entries,
        "source": {"scope": "pilot_only", "diagnostic_parameter_snapshot": {"num_periods": 20}}})
    for kind, vector in (("zero", []), ("linear", [4., -.5, 2.2]), ("relu", [2., .5, 2.])):
        atomic_json(root / kind / "solution.json", {"terminal_features": [] if boundary else vector,
                    "terminal_value": 0.})
    return root / "diagnostic_results.json"


class ValueCoverageTests(unittest.TestCase):
    def test_reorder_ranges_constants_tolerance_and_boundary_denominator(self):
        rows = [{"feature_names": ["c", "a", "b"], "terminal_features": x} for x in
                ([2., .5, 2.], [4., -.5, 2.2], [3., 1., 2. + 5e-9])]
        rows.append({"skip_reason": "zero_value_operating_end_boundary"})
        report = analyze_coverage(training(), rows)
        self.assertEqual(report["query_states_with_features"], 3)
        self.assertAlmostEqual(report["outside_training_range_state_fraction"], 1 / 3)
        self.assertEqual(report["strict_zero_tolerance_outside_states"], 2)
        self.assertEqual(report["training_constant_changed_fields"], 1)
        self.assertEqual(report["skipped"]["zero_value_operating_end_boundary"], 1)
        self.assertEqual(report["reordered_query_schemas"], 1)
        by_name = {row["feature_name"]: row for row in report["per_feature"]}
        self.assertAlmostEqual(by_name["a"]["maximum_below_amount"], .5)
        self.assertAlmostEqual(by_name["b"]["maximum_above_amount"], .2)
        self.assertEqual(by_name["b"]["constant_changed_count"], 1)
        self.assertEqual(by_name["c"]["outside_count"], 1)
        self.assertFalse(report["is_independent_test_profit_result"])
        self.assertEqual(report["scope"], "pilot_only")

    def test_name_mismatch_duplicates_nonfinite_and_scales_rejected(self):
        for names, values in ((["a", "a", "c"], [0., 2., 1.]), (["a", "b", "d"], [0., 2., 1.]),
                              (["a", "b", "c"], [np.nan, 2., 1.])):
            with self.assertRaises(ValueError):
                analyze_coverage(training(), [{"feature_names": names, "terminal_features": values}])
        reference = training()
        reference.feature_spec = {"names": reference.feature_names, "count_scale": 100.}
        with self.assertRaisesRegex(ValueError, "scale differs"):
            analyze_coverage(reference, [{"feature_names": reference.feature_names, "terminal_features": [0., 2., 1.],
                                         "feature_spec": {"count_scale": 200.}}])

    def test_npz_fit_and_pilot_model_ranges_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "training.npz", features=np.array([[0., 2., 1.], [1., 2., 3.]]),
                     feature_names=np.array(["a", "b", "c"]), scope="pilot_only")
            atomic_json(root / "iteration_0_fit.json", {"diagnostics": {"training_feature_range": range_record()}})
            atomic_json(root / "pilot_only_relu.json", {"scope": "pilot_only", "model": {"feature_names": ["a", "b", "c"]},
                        "diagnostics": {"training_feature_range": range_record()}})
            ranges = [load_training_range(root / name) for name in
                      ("training.npz", "iteration_0_fit.json", "pilot_only_relu.json")]
            for result in ranges:
                np.testing.assert_equal(result.minimum, [0., 2., 1.])
                np.testing.assert_equal(result.maximum, [1., 2., 3.])
                self.assertEqual(result.samples, 2)
            self.assertEqual(ranges[-1].metadata["scope"], "pilot_only")
            bad = range_record();bad["source"] = "test_statistics"
            atomic_json(root / "bad.json", {"diagnostics": {"training_feature_range": bad}})
            with self.assertRaisesRegex(ValueError, "current training"):
                load_training_range(root / "bad.json")

    def test_pilot_prepare_matrix_hashes_and_completed_marker(self):
        from analyze_value_coverage import _sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "features.npy", np.array([[0., 2., 1.], [1., 2., 3.]]))
            atomic_json(root / "feature_spec.json", {"names": ["a", "b", "c"], "dimension": 3})
            manifest = {"scope": "pilot_only", "state": "complete", "samples": 2, "feature_dimension": 3,
                        "artifact_sha256": {name: _sha(root / name) for name in ("features.npy", "feature_spec.json")}}
            atomic_json(root / "manifest.json", manifest)
            result = load_training_range(root)
            self.assertEqual(result.samples, 2)
            np.save(root / "features.npy", np.ones((2, 3)))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_training_range(root)

    def test_completed_diagnostics_failure_zero_and_L_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = diagnostic_fixture(root / "middle")
            rows = list(iter_query_records(path))
            report = analyze_coverage(training(), rows)
            self.assertEqual(report["query_states_with_features"], 2)
            self.assertEqual(report["outside_training_range_state_fraction"], .5)
            self.assertEqual(report["skipped"]["no_terminal_model"], 1)
            self.assertEqual(report["skipped"]["no_feasible_incumbent_diagnostic"], 1)
            only = list(iter_query_records(path.parent / "linear/solution.json"))
            self.assertEqual(len(only), 1)
            boundary = diagnostic_fixture(root / "boundary", boundary=True)
            last = analyze_coverage(training(), iter_query_records(boundary))
            self.assertEqual(last["skipped"]["zero_value_operating_end_boundary"], 2)
            self.assertEqual(last["skipped"]["no_terminal_model"], 1)
            self.assertIsNone(last["outside_training_range_state_fraction"])
            incomplete = diagnostic_fixture(root / "incomplete", completed=False)
            with self.assertRaisesRegex(ValueError, "not completed"):
                list(iter_query_records(incomplete))

    def test_full_result_and_streaming_journal_skip_only_empty_zero_L(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"period": 0, "horizon": 1, "solution": {"terminal_features": [0., 2., 1.], "terminal_value": 1.}},
                    {"period": 1, "horizon": 1, "solution": {"terminal_features": [], "terminal_value": 0.}}]
            result = {"completed": True, "parameter_snapshot": {"num_periods": 2}, "feature_names": ["a", "b", "c"],
                      "method": {"terminal_kind": "relu"}, "rounds": rows}
            with gzip.open(root / "standalone.json.gz", "wt", encoding="utf-8") as stream:
                json.dump(result, stream)
            report = analyze_coverage(training(), iter_query_records(root / "standalone.json.gz"))
            self.assertEqual(report["query_states_with_features"], 1)
            self.assertEqual(report["skipped"]["zero_value_operating_end_boundary"], 1)
            identity = {"parameters": {"num_periods": 2}, "feature_names": ["a", "b", "c"], "terminal_model": {"kind": "relu"}}
            atomic_json(root / "journal/initial.json", {"identity": identity, "fingerprint": fingerprint(identity)})
            atomic_json(root / "journal/status.json", {"state": "complete", "completed_periods": 2, "fingerprint": fingerprint(identity)})
            atomic_json(root / "worker_status.json", {"state": "complete"})
            (root / "journal/rounds.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            # This intentionally invalid result would fail if the full gzip were read.
            (root / "result.json.gz").write_bytes(b"not loaded; completed journal is used")
            streamed = list(iter_query_records(root / "result.json.gz"))
            self.assertEqual(len(streamed), 2)
            rows[0]["solution"]["terminal_features"] = []
            (root / "journal/rounds.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            missing_report = analyze_coverage(training(), iter_query_records(root))
            self.assertEqual(missing_report["skipped"]["missing_terminal_features_without_verified_zero_L"], 1)
            self.assertIsNone(missing_report["outside_training_range_state_fraction"])
            rows[1]["solution"].pop("terminal_value")
            (root / "journal/rounds.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            no_proof = analyze_coverage(training(), iter_query_records(root))
            self.assertEqual(no_proof["skipped"]["zero_value_operating_end_boundary"], 0)
            self.assertEqual(no_proof["skipped"]["missing_terminal_features_without_verified_zero_L"], 2)
            self.assertIsNone(no_proof["outside_training_range_state_fraction"])

    def test_overlapping_query_inputs_cannot_double_count_one_selected_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "train_fit.json", {"diagnostics": {"training_feature_range": range_record()}})
            query = diagnostic_fixture(root / "query")
            with self.assertRaisesRegex(ValueError, "duplicate selected terminal"):
                analyze_files(root / "train_fit.json", [query, query.parent / "linear/solution.json"])

    def test_file_api_keeps_training_and_query_inputs_unchanged(self):
        from analyze_value_coverage import _sha
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root / "train_fit.json", {"scope": "pilot_only", "diagnostics": {"training_feature_range": range_record()}})
            query = diagnostic_fixture(root / "query")
            before = {path: _sha(path) for path in root.rglob("*") if path.is_file()}
            report = analyze_files(root / "train_fit.json", [query])
            self.assertEqual(report["query_states_with_features"], 2)
            self.assertTrue(all(_sha(path) == value for path, value in before.items()))


if __name__ == "__main__":
    unittest.main()
