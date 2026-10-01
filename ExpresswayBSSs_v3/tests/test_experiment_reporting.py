"""Report export checks use synthetic temporary records, not experimental data."""
import csv
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiment_control import atomic_json
from src.experiment_reporting import export_experiment_report, _process_identity_state
from src.experiment_suite import _Suite
from src.parameters import BusinessParameters
from src.terminal_value import TerminalValueModel, terminal_configuration_fingerprint


class ExperimentReportingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def read_csv(self,path):
        with Path(path).open(encoding="utf-8-sig",newline="") as stream:
            return list(csv.DictReader(stream))

    def test_partial_pilot_has_no_full_day_score_and_no_formal_samples(self):
        directory = self.root/"pilot/seed_101"
        atomic_json(directory/"worker_status.json",{"state":"paused","error":"deadline"})
        atomic_json(directory/"journal/status.json",{"state":"paused","completed_periods":12})
        atomic_json(directory/"input_manifest.json",{"parameters":{"num_periods":288}})
        # A stale metrics artifact must not make a paused worker count as done.
        atomic_json(directory/"metrics.json",{"completed_periods":288,"net_profit_yuan":1234.})
        paths = export_experiment_report(self.root)
        pilot = self.read_csv(paths["pilot_status_csv"])[0]
        self.assertEqual(pilot["state"],"paused")
        self.assertEqual(pilot["completed_periods"],"12")
        self.assertEqual(pilot["net_profit_yuan"],"")
        self.assertEqual(self.read_csv(paths["completed_groups_csv"]),[])
        self.assertEqual(self.read_csv(paths["paired_differences_csv"]),[])
        text = Path(paths["markdown"]).read_text(encoding="utf-8")
        self.assertIn("目前没有已完成的正式分组",text)
        self.assertNotIn("1234",text)

    def test_two_training_replicates_are_not_pooled_and_incomplete_groups_are_omitted(self):
        groups = {}
        for replicate in range(2):
            groups[f"base/relu_full_rep{replicate}"] = {
                "status":"complete","scenario_seeds":[41000,41001],
                "metadata":{"training_replicate":replicate,"kind":"relu","variant":"full"},
                "statistics":{"net_profit_yuan":{"n":2,"mean":100.+replicate,"std":2.,"ci95_low":80.,"ci95_high":120.}},
                "paired_difference_from_zero":{"net_profit_yuan":{"n":2,"mean":5.,"std":1.,"ci95_low":-2.,"ci95_high":12.}}}
        groups["partial"] = {"status":"paused","scenario_seeds":[999],
            "statistics":{"net_profit_yuan":{"n":1,"mean":999999.,"std":None}}}
        source = self.root/"formal_suite/report.json"
        atomic_json(source,{"status":"paused","groups":groups})
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        paths = export_experiment_report(self.root)
        rows = self.read_csv(paths["completed_groups_csv"])
        paired = self.read_csv(paths["paired_differences_csv"])
        self.assertEqual(len(rows),2)
        self.assertEqual({row["training_replicate"] for row in rows},{"0","1"})
        self.assertTrue(all(row["n"]=="2" for row in rows+paired))
        self.assertTrue(all(row["scenario_count"]=="2" for row in rows+paired))
        self.assertTrue(all(row["group"]!="partial" for row in rows))
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)
        self.assertNotIn("999999",Path(paths["markdown"]).read_text(encoding="utf-8"))

    def test_missing_completed_metrics_are_not_assumed_zero(self):
        directory = self.root/"pilot/seed_102"
        atomic_json(directory/"worker_status.json",{"state":"complete"})
        atomic_json(directory/"journal/status.json",{"completed_periods":288})
        paths = export_experiment_report(self.root)
        row = self.read_csv(paths["pilot_status_csv"])[0]
        self.assertEqual(row["state"],"complete_without_verified_metrics")
        self.assertEqual(row["net_profit_yuan"],"")
        self.assertEqual(row["verified_full_day"],"False")

    def test_duplicate_test_scenarios_are_rejected(self):
        atomic_json(self.root/"formal_suite/report.json",{"groups":{"bad":{"status":"complete","scenario_seeds":[1,1]}}})
        with self.assertRaisesRegex(ValueError,"duplicate test"):
            export_experiment_report(self.root)

    def deadline_fixture(self, *, own_guard=True):
        directory = self.root/"pilot/seed_101"
        status = {"state":"running","pid":777,"started_at":"1999-12-31T23:59:00+00:00",
                  "deadline":"2000-01-01T00:00:00+00:00"}
        process = {"worker_pid":777,"created":946684739.,"deadline":status["deadline"]}
        guard = {"state":"paused","reason":"user_wall_clock_deadline","worker_pid":777,
                 "deadline":status["deadline"],"observed_at":"2000-01-01T00:00:01+00:00",
                 "process_tree_terminated":True}
        atomic_json(directory/"worker_status.json",status)
        atomic_json(directory/"process.json",process)
        atomic_json(directory/"journal/status.json",{"completed_periods":80})
        atomic_json(directory/"input_manifest.json",{"parameters":{"num_periods":288}})
        if own_guard:
            atomic_json(directory/"deadline_pause.json",guard)
        return directory,status,process,guard

    def test_deadline_guard_and_exited_process_reconcile_stale_running(self):
        directory,status,process,guard = self.deadline_fixture()
        with patch("src.experiment_reporting._process_identity_state",return_value="exited"):
            paths = export_experiment_report(self.root)
        row = self.read_csv(paths["pilot_status_csv"])[0]
        self.assertEqual(row["state"],"paused")
        self.assertEqual(row["raw_state"],"running")
        self.assertEqual(row["state_source"],"deadline_guard_and_process_exit")
        self.assertEqual(row["net_profit_yuan"],"")
        self.assertEqual(json.loads((directory/"worker_status.json").read_text())["state"],"running")

    def test_guard_record_does_not_pause_an_alive_or_unverified_process(self):
        self.deadline_fixture()
        for evidence in ("alive","unknown"):
            with self.subTest(evidence=evidence), patch("src.experiment_reporting._process_identity_state",return_value=evidence):
                paths = export_experiment_report(self.root)
            row = self.read_csv(paths["pilot_status_csv"])[0]
            self.assertEqual(row["state"],"running")
            self.assertEqual(row["process_state"],evidence)

    def test_stale_guard_for_other_pid_or_deadline_is_not_used(self):
        directory,status,process,guard = self.deadline_fixture()
        for changed in ({"worker_pid":778},{"deadline":"2001-01-01T00:00:00+00:00"}):
            atomic_json(directory/"deadline_pause.json",{**guard,**changed})
            with patch("src.experiment_reporting._process_identity_state",return_value="exited"):
                paths = export_experiment_report(self.root)
            row = self.read_csv(paths["pilot_status_csv"])[0]
            self.assertEqual(row["state"],"interrupted_unclassified")
            self.assertEqual(row["deadline_evidence_path"],"")

    def test_controller_guard_can_identify_its_ended_worker_tree(self):
        directory,status,process,guard = self.deadline_fixture(own_guard=False)
        atomic_json(self.root/"controller_status.json",{"state":"running","pid":888,
            "started_at":"1999-12-31T23:58:00+00:00","deadline":status["deadline"]})
        atomic_json(self.root/"controller_deadline_pause.json",{**guard,"worker_pid":888})
        atomic_json(self.root/"formal_suite/report.json",{"status":"running","groups":{}})
        with patch("src.experiment_reporting._process_identity_state",return_value="exited"):
            paths = export_experiment_report(self.root)
        row = self.read_csv(paths["pilot_status_csv"])[0]
        self.assertEqual(row["state"],"paused")
        self.assertTrue(row["deadline_evidence_path"].endswith("controller_deadline_pause.json"))
        self.assertIn("正式套件状态：paused",Path(paths["markdown"]).read_text(encoding="utf-8"))

    def test_new_live_attempt_overrides_previous_paused_artifact(self):
        directory,status,process,guard = self.deadline_fixture()
        atomic_json(directory/"worker_status.json",{**status,"state":"paused"})
        atomic_json(directory/"process.json",{**process,"worker_pid":779,"created":946684820.})
        for evidence,source in (("alive","newer_live_process"),("unknown","newer_process_unverified")):
            with patch("src.experiment_reporting._process_identity_state",return_value=evidence):
                paths = export_experiment_report(self.root)
            row = self.read_csv(paths["pilot_status_csv"])[0]
            self.assertEqual(row["state"],"running")
            self.assertEqual(row["raw_state"],"paused")
            self.assertEqual(row["state_source"],source)

    def test_confirmed_failed_and_complete_states_are_preserved(self):
        directory,status,process,guard = self.deadline_fixture()
        for actual_state in ("failed","complete"):
            data = {**status,"state":actual_state,"metrics":{"completed_periods":288,"net_profit_yuan":25.}}
            atomic_json(directory/"worker_status.json",data)
            with patch("src.experiment_reporting._process_identity_state",return_value="exited"):
                paths = export_experiment_report(self.root)
            row = self.read_csv(paths["pilot_status_csv"])[0]
            self.assertEqual(row["state"],actual_state)
            self.assertEqual(row["net_profit_yuan"],"25.0" if actual_state=="complete" else "")

    def test_missing_target_period_count_prevents_full_day_profit(self):
        directory = self.root/"pilot/seed_103"
        atomic_json(directory/"worker_status.json",{"state":"complete","metrics":{"completed_periods":288,"net_profit_yuan":123.}})
        paths = export_experiment_report(self.root)
        row = self.read_csv(paths["pilot_status_csv"])[0]
        self.assertEqual(row["state"],"complete_without_verified_metrics")
        self.assertEqual(row["net_profit_yuan"],"")

    def test_process_identity_detects_pid_reuse_without_stopping_anything(self):
        self.assertEqual(_process_identity_state(os.getpid(),created=0.),"pid_reused")

    def test_simple_inventory_candidates_include_configuration_fingerprint(self):
        params = BusinessParameters(num_periods=2,horizon=1,num_reservations=0,random_arrival_rate_per_hour=[0.]*6)
        config = self.root/"config.json"
        params.save_json(config)
        observed = []
        def worker(job):
            if job["model"]:
                model = TerminalValueModel.load(job["model"])
                observed.append(model)
            return {"worker_status":"complete","metrics":{"completed_periods":2,"net_profit_yuan":1.}}
        suite = _Suite(config,self.root,None,worker,[{"completed_periods":2,"worker_wall_seconds":60.}]*3)
        zero = suite._test("base",params,"zero")
        suite._simple_inventory(zero)
        self.assertTrue(observed)
        self.assertTrue(all(model.configuration_fingerprint==terminal_configuration_fingerprint(params) for model in observed))
        stored = list((self.root/"formal_suite/models/base").glob("simple_inventory_*.json"))
        models = [TerminalValueModel.load(path) for path in stored if path.name!="simple_inventory_selection.json"]
        self.assertEqual(len(models),5)
        self.assertTrue(all(model.configuration_fingerprint==terminal_configuration_fingerprint(params) for model in models))


if __name__ == "__main__":
    unittest.main()
