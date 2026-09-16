"""Lightweight recovery checks; process fixtures never touch experiment outputs."""
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import psutil

from src.experiment_recovery import (ExperimentAlreadyRunning, OutputDirectoryLock,
    attempt_key, inspect_process_identity, seal_dead_attempt, _same_path)

START = "2026-09-14T02:00:00+08:00"
STOP = "2026-09-14T07:00:00+08:00"
CREATED = datetime.fromisoformat(START).timestamp()


def records():
    status = {"state": "running", "pid": 101, "started_at": START, "deadline": STOP}
    process = {"worker_pid": 101, "created": CREATED, "deadline": STOP}
    guard = {"target_pid": 101, "target_created": CREATED, "deadline": STOP,
             "stop_finished_at": STOP, "process_tree_terminated": True}
    return status, process, guard


class ExperimentRecoveryTests(unittest.TestCase):
    def test_lock_blocks_a_second_process_and_exit_releases_without_deletion(self):
        code = ("import os,sys; from src.experiment_recovery import OutputDirectoryLock; "
                "lock=OutputDirectoryLock(sys.argv[1]).acquire(); "
                "print('ready',flush=True); sys.stdin.readline(); os._exit(0)")
        with tempfile.TemporaryDirectory() as directory:
            child = subprocess.Popen([sys.executable, "-c", code, directory],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "ready")
                with self.assertRaises(ExperimentAlreadyRunning):
                    OutputDirectoryLock(directory).acquire()
            finally:
                child.communicate("\n", timeout=5)
            self.assertEqual(child.returncode, 0)
            with OutputDirectoryLock(directory):
                self.assertTrue((Path(directory) / ".controller.lock").exists())
            self.assertTrue((Path(directory) / ".controller.lock").exists())

    def test_real_current_process_identity_and_pid_reuse(self):
        current = psutil.Process(os.getpid())
        self.assertEqual(inspect_process_identity(current.pid, current.create_time())["state"], "alive")
        self.assertEqual(inspect_process_identity(current.pid, current.create_time() - 100)["state"], "unrelated")

    def test_legacy_identity_checks_script_operand_and_output(self):
        with tempfile.TemporaryDirectory() as directory:
            process = MagicMock()
            process.create_time.return_value = CREATED
            process.cwd.return_value = directory
            process.is_running.return_value = True
            process.status.return_value = psutil.STATUS_RUNNING
            with patch("src.experiment_recovery.psutil.Process", return_value=process):
                process.cmdline.return_value = [sys.executable, "-u", "run_experiments.py", "--output-root", directory]
                options = dict(script_name="run_experiments.py", output_dir=directory, output_flag="--output-root")
                self.assertEqual(inspect_process_identity(101, **options)["state"], "alive")
                process.cmdline.return_value = [sys.executable, "unrelated.py", "run_experiments.py", "--output-root", directory]
                self.assertEqual(inspect_process_identity(101, **options)["state"], "unrelated")
                process.cmdline.return_value = [sys.executable, "run_experiments.py"]
                self.assertEqual(inspect_process_identity(101, **options)["state"], "unknown")
                process.cmdline.side_effect = psutil.AccessDenied(101)
                self.assertEqual(inspect_process_identity(101, **options)["state"], "unknown")

    def test_existing_extended_windows_path_has_same_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory).resolve()
            self.assertTrue(_same_path(str(expected), expected))
            if os.name == "nt":
                raw = str(expected)
                extended = raw if raw.startswith("\\\\?\\") else "\\\\?\\" + raw
                self.assertTrue(_same_path(extended, expected))

    def test_direct_guard_seals_five_hours_without_counting_resume_delay(self):
        status, process, guard = records()
        sealed = seal_dead_attempt(status, process, guard_records=[guard], liveness={"state": "dead"})
        self.assertEqual(sealed["state"], "paused")
        self.assertEqual(sealed["wall_seconds"], 18000.)
        self.assertEqual(datetime.fromisoformat(sealed["finished_at"]), datetime.fromisoformat(STOP))
        self.assertEqual(sealed["timing_source"], "guard_pid_create_time")
        self.assertEqual(status["state"], "running")
        self.assertNotIn("finished_at", status)

    def test_parent_guard_requires_matching_child_creation_time(self):
        status, process, guard = records()
        guard.update(target_pid=500, target_created=CREATED - 60,
                     terminated_processes=[{"pid": 101, "created": CREATED}])
        sealed = seal_dead_attempt(status, process, guard_records=[guard], liveness={"state": "dead"})
        self.assertEqual(sealed["wall_seconds"], 18000.)
        self.assertEqual(sealed["timing_source"], "parent_guard_process_tree")
        guard["terminated_processes"][0]["created"] += 10
        rejected = seal_dead_attempt(status, process, guard_records=[guard], liveness={"state": "dead"})
        self.assertIsNone(rejected["wall_seconds"])
        self.assertTrue(rejected["timing_incomplete"])

    def test_legacy_guard_is_explicitly_estimated_and_checks_attempt_deadline(self):
        status, process, guard = records()
        guard.pop("target_created")
        guard["worker_pid"] = guard.pop("target_pid")
        guard["observed_at"] = guard.pop("stop_finished_at")
        sealed = seal_dead_attempt(status, process, guard_records=[guard], liveness={"state": "dead"})
        self.assertEqual(sealed["wall_seconds"], 18000.)
        self.assertEqual(sealed["timing_source"], "legacy_guard_observed_at_estimate")
        self.assertTrue(sealed["wall_seconds_estimated"])
        guard["deadline"] = "2026-09-13T07:00:00+08:00"
        rejected = seal_dead_attempt(status, process, guard_records=[guard], liveness={"state": "dead"})
        self.assertIsNone(rejected["wall_seconds"])

    def test_missing_end_evidence_does_not_use_current_time_or_deadline(self):
        status, process, _ = records()
        sealed = seal_dead_attempt(status, process, liveness={"state": "dead"})
        self.assertIsNone(sealed["wall_seconds"])
        self.assertTrue(sealed["timing_incomplete"])
        self.assertNotIn("finished_at", sealed)
        self.assertEqual(sealed["state"], "failed")

    def test_live_or_unknown_attempt_cannot_be_sealed(self):
        status, process, guard = records()
        for state in ("alive", "unknown"):
            with self.subTest(state=state), self.assertRaises(ExperimentAlreadyRunning):
                seal_dead_attempt(status, process, guard_records=[guard], liveness={"state": state})

    def test_idempotent_attempt_key_survives_sealing_and_later_missing_process_file(self):
        status, process, guard = records()
        key = attempt_key(status, process)
        sealed = seal_dead_attempt(status, process, guard_records=[guard], liveness={"state": "dead"})
        self.assertEqual(attempt_key(sealed), key)
        self.assertEqual(attempt_key(sealed, process), key)
        retried = deepcopy(status)
        retried["started_at"] = "2026-09-14T15:00:00+08:00"
        self.assertNotEqual(attempt_key(retried, process), key)

    def test_existing_paused_runtime_and_earliest_confirmed_stop_are_preserved(self):
        status, process, guard = records()
        status.update(state="paused", finished_at=STOP, wall_seconds=17999.25)
        sealed = seal_dead_attempt(status, process, guard_records=[guard], liveness={"state": "dead"})
        self.assertEqual(sealed["wall_seconds"], 17999.25)
        status.pop("wall_seconds")
        later = deepcopy(guard)
        later["stop_finished_at"] = "2026-09-14T07:00:02+08:00"
        sealed = seal_dead_attempt(status, process, guard_records=[later], liveness={"state": "dead"})
        self.assertEqual(sealed["wall_seconds"], 18000.)
        self.assertEqual(sealed["timing_source"], "status_finished_at")

    def test_stale_or_failed_guard_does_not_supply_a_stop(self):
        status, process, guard = records()
        for changes in ({"target_created": CREATED - 100}, {"process_tree_terminated": False},
                        {"stop_finished_at": "2026-09-14T01:00:00+08:00"}):
            with self.subTest(changes=changes):
                bad = dict(guard, **changes)
                sealed = seal_dead_attempt(status, process, guard_records=[bad], liveness={"state": "dead"})
                self.assertIsNone(sealed["wall_seconds"])


if __name__ == "__main__":
    unittest.main()
