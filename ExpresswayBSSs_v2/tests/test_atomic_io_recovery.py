"""I/O retry must never duplicate execution records or hide model failures."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.atomic_io import retry_atomic_writer
from src.experiment_control import RunJournal, atomic_json
from run_zero_terminal import reviewed_io_failure


def sharing_error():
    error = PermissionError("[WinError 5] access denied")
    error.winerror = 5
    return error


class AtomicRetryTests(unittest.TestCase):
    def test_transient_error_retries_identical_metadata(self):
        writer = Mock(side_effect=[sharing_error(), None])
        sleep = Mock()
        value = {"state": "running", "completed_periods": 14}
        retry_atomic_writer(writer, sleep=sleep)("status.json", value)
        self.assertEqual(writer.call_count, 2)
        self.assertEqual(writer.call_args_list[0], writer.call_args_list[1])
        sleep.assert_called_once_with(.05)

    def test_persistent_error_is_bounded_and_raised(self):
        writer = Mock(side_effect=sharing_error())
        sleep = Mock()
        with self.assertRaises(PermissionError):
            retry_atomic_writer(writer, attempts=3, sleep=sleep)("status.json", {})
        self.assertEqual(writer.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_other_failures_are_not_retried(self):
        for error in (ValueError("invalid JSON"), OSError("disk full"), PermissionError("not Windows sharing")):
            writer = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                retry_atomic_writer(writer, sleep=Mock())("status.json", {})
            self.assertEqual(writer.call_count, 1)

    def test_journal_append_is_not_repeated_when_status_replace_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = RunJournal(temporary, {"scenario": "unchanged"})
            journal.initialize({"period": 0}, {})
            replace = os.replace
            count = 0
            def transient(source, target):
                nonlocal count
                if Path(target).name == "status.json":
                    count += 1
                    if count == 1:
                        raise sharing_error()
                return replace(source, target)
            writer = retry_atomic_writer(atomic_json, sleep=Mock())
            with patch("src.experiment_control.atomic_json", writer), patch("os.replace", side_effect=transient):
                journal.append({"period": 0, "reward": 17})
            _, _, rounds = journal.recover()
            self.assertEqual(rounds, [{"period": 0, "reward": 17}])
            self.assertEqual(json.loads(journal.status_path.read_text())['completed_periods'], 1)

    def test_only_reviewed_atomic_write_failure_is_retryable(self):
        record = {"error_type": "PermissionError", "error": "[WinError 5] access denied",
                  "traceback": 'File "experiment_control.py"\n os.replace(pending, path)'}
        self.assertTrue(reviewed_io_failure(record))
        self.assertFalse(reviewed_io_failure({**record, "error_type": "MPCSolveError"}))
        self.assertFalse(reviewed_io_failure({**record, "traceback": 'File "mpc_model.py"'}))


if __name__ == '__main__':
    unittest.main()
