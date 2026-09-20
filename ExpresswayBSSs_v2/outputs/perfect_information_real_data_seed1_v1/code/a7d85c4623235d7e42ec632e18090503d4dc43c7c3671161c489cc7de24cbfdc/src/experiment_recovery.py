"""Exclusive experiment ownership and evidence-based recovery of stopped attempts.

These helpers never terminate processes or delete files. Timing is recovered
only from an attempt's own end marker or a matching termination record.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path

import psutil


class ExperimentAlreadyRunning(RuntimeError):
    """Another process owns the output or its identity cannot be checked safely."""


class OutputDirectoryLock:
    """Nonblocking OS lock; process exit releases it and leaves the file intact."""

    def __init__(self, directory, name=".controller.lock"):
        self.directory = Path(directory).resolve()
        self.path = self.directory / name
        self._stream = None

    def acquire(self):
        if self._stream is not None:
            return self
        self.directory.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        try:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise ExperimentAlreadyRunning("experiment output is already locked: " + str(self.directory)) from error
            raise
        self._stream = stream
        return self

    def release(self):
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *unused):
        self.release()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _time(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result.astimezone(timezone.utc) if result.tzinfo is not None else None
    except ValueError:
        return None


def _same_path(value, expected, cwd=None):
    try:
        path = Path(str(value).strip('"'))
        if not path.is_absolute() and cwd is not None:
            path = Path(cwd) / path
        expected_path = Path(expected)
        try:
            return os.path.samefile(path, expected_path)
        except OSError:
            def normalized(candidate):
                value = str(candidate.resolve())
                if os.name == "nt":
                    extended = chr(92) * 2 + "?" + chr(92)
                    if value.startswith(extended + "UNC" + chr(92)):
                        value = chr(92) * 2 + value[len(extended) + 4:]
                    elif value.startswith(extended):
                        value = value[len(extended):]
                return os.path.normcase(value)
            return normalized(path) == normalized(expected_path)
    except (OSError, ValueError):
        return False


def _script_and_arguments(command):
    """Locate the Python script operand, without mistaking a later data argument."""
    index = 1
    while index < len(command):
        part = command[index]
        if part == "-c":
            return None, []
        if part == "-m":
            return (command[index + 1].split(".")[-1] + ".py", command[index + 2:]) if index + 1 < len(command) else (None, [])
        if part in {"-W", "-X"}:
            index += 2
            continue
        if part.startswith("-"):
            index += 1
            continue
        return Path(part.strip('"')).name, command[index + 1:]
    return None, []


def _command_matches(command, script_name, output_dir, output_flag, cwd):
    script, arguments = _script_and_arguments(command)
    if script is None or os.path.normcase(script) != os.path.normcase(Path(script_name).name):
        return False
    values = []
    for index, part in enumerate(arguments):
        if part == output_flag and index + 1 < len(arguments):
            values.append(arguments[index + 1])
        elif part.startswith(output_flag + "="):
            values.append(part[len(output_flag) + 1:])
    # An implicit script default cannot establish legacy directory ownership.
    if not values:
        return None
    matches = [_same_path(value, output_dir, cwd) for value in values]
    return None if any(matches) and not all(matches) else all(matches)


def inspect_process_identity(pid, created=None, *, script_name=None, output_dir=None,
                             output_flag="--output-dir"):
    """Return state alive/dead/unrelated/unknown; unknown must block new writers.

    PID plus creation time is the preferred identity. Legacy records need the
    actual Python script operand and its exact output-directory argument.
    """
    result = {"pid": pid, "state": "unknown", "created": None, "identity_source": None}
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return dict(result, reason="missing_or_invalid_pid")
    try:
        process = psutil.Process(pid)
        with process.oneshot():
            actual_created = process.create_time()
            result["created"] = actual_created
            if _number(created):
                if abs(actual_created - created) > .01:
                    return dict(result, state="unrelated", reason="pid_reused", identity_source="pid_create_time")
                source = "pid_create_time"
            else:
                if not script_name or output_dir is None:
                    return dict(result, reason="creation_time_and_legacy_command_identity_unavailable")
                match = _command_matches(process.cmdline(), script_name, output_dir, output_flag, process.cwd())
                if match is None:
                    return dict(result, reason="legacy_output_directory_not_explicit_or_ambiguous")
                if not match:
                    return dict(result, state="unrelated", reason="legacy_command_differs",
                                identity_source="legacy_command_output_directory")
                source = "legacy_command_output_directory"
            if not process.is_running() or process.status() in {psutil.STATUS_ZOMBIE, getattr(psutil, "STATUS_DEAD", "dead")}:
                return dict(result, state="dead", reason="process_exited", identity_source=source)
            return dict(result, state="alive", reason="matching_process_running", identity_source=source)
    except psutil.NoSuchProcess:
        return dict(result, state="dead", reason="pid_absent")
    except psutil.AccessDenied:
        return dict(result, reason="process_access_denied")


def _record_pid(record):
    for key in ("worker_pid", "target_pid", "controller_pid", "pid"):
        if isinstance(record.get(key), int) and not isinstance(record[key], bool):
            return record[key]
    return None


def _attempt_created(status, process_record):
    for key in ("created", "process_created", "target_created"):
        if _number(status.get(key)):
            return float(status[key])
    if _record_pid(process_record) == status.get("pid"):
        for key in ("created", "process_created", "target_created"):
            if _number(process_record.get(key)):
                return float(process_record[key])
    return None


def attempt_key(status, process_record=None):
    """Stable archive basename, unchanged by sealing; prevents duplicate history."""
    process_record = process_record or {}
    existing = status.get("attempt_id")
    if isinstance(existing, str) and existing.startswith("attempt_") and len(existing) == 32 and all(c in "0123456789abcdef" for c in existing[8:]):
        return existing
    identity = {"pid": status.get("pid"), "created": _attempt_created(status, process_record),
                "started_at": status.get("started_at"), "deadline": status.get("deadline")}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "attempt_" + hashlib.sha256(encoded).hexdigest()[:24]


def _guard_end(guard, status, process_record, started):
    pid = status.get("pid")
    created = _attempt_created(status, process_record)
    expected_deadline = _time(status.get("deadline"))
    guard_deadline = _time(guard.get("deadline"))
    if expected_deadline is not None and guard_deadline != expected_deadline:
        return None
    if guard.get("process_tree_terminated") is not True:
        return None
    end = _time(guard.get("stop_finished_at")) or _time(guard.get("observed_at"))
    if end is None or end < started or (expected_deadline is not None and end < expected_deadline):
        return None
    if created is not None and end.timestamp() < created:
        return None
    if _record_pid(guard) == pid:
        guard_created = next((float(guard[key]) for key in ("target_created", "created", "process_created")
                              if _number(guard.get(key))), None)
        if guard_created is not None:
            if created is None or abs(guard_created - created) > .01:
                return None
            return end, "guard_pid_create_time", True
        # Legacy records have no creation time: require the original deadline
        # and a matching process envelope, with stop time after this start.
        if (expected_deadline is not None and guard_deadline == expected_deadline
                and _record_pid(process_record) == pid
                and _time(process_record.get("deadline")) == expected_deadline
                and (_attempt_created(status, {}) is None or _attempt_created(status, {}) == _attempt_created({"pid": pid}, process_record))):
            return end, "legacy_guard_observed_at_estimate", True
        return None
    for member in guard.get("terminated_processes", []):
        if not isinstance(member, dict):
            continue
        member_created = member.get("created")
        if member.get("pid") == pid and created is not None and _number(member_created):
            if abs(member_created - created) <= .01:
                return end, "parent_guard_process_tree", True
    return None


def seal_dead_attempt(status, process_record=None, *, guard_records=(), liveness=None):
    """Return a sealed copy, never write; never substitute resume time for exit.

    A missing end marker leaves timing incomplete. Its duration must not silently
    become zero in aggregate timing metrics. Guard-based times are explicitly
    marked as estimates because confirmation can follow the actual process exit.
    """
    process_record = process_record or {}
    result = deepcopy(status)
    result["attempt_id"] = result.get("attempt_id") or attempt_key(status, process_record)
    if status.get("state") not in {"running", "paused"}:
        return result
    if liveness is None:
        liveness = inspect_process_identity(status.get("pid"), _attempt_created(status, process_record))
    if liveness.get("state") in {"alive", "unknown"}:
        raise ExperimentAlreadyRunning("refusing to seal a live or unverified attempt: " + str(status.get("pid")))
    if liveness.get("state") not in {"dead", "unrelated"}:
        raise ValueError("invalid process-liveness state")
    if status.get("state") == "running":
        result.update(state="failed", recovery_reason="process_exited_without_end_evidence")
    started = _time(status.get("started_at"))
    start_source = "worker_started_at"
    if started is None:
        created = _attempt_created(status, process_record)
        if created is not None:
            started = datetime.fromtimestamp(created, timezone.utc)
            start_source = "process_create_time"
    if _number(status.get("wall_seconds")) and status["wall_seconds"] >= 0:
        result.setdefault("timing_source", "recorded_attempt_wall_seconds")
        result.setdefault("wall_seconds_estimated", False)
        result["timing_incomplete"] = False
        return result
    result["wall_seconds"] = None
    result["timing_incomplete"] = True
    result["timing_source"] = "no_matching_end_evidence"
    if started is None:
        return result
    candidates = []
    own_end = _time(status.get("finished_at"))
    if own_end is not None and own_end >= started:
        candidates.append((own_end, "status_finished_at", False))
    if _record_pid(process_record) == status.get("pid"):
        own_created = _attempt_created(status, {})
        recorded_created = None
        for key in ("created", "process_created", "target_created"):
            if _number(process_record.get(key)):
                recorded_created = float(process_record[key])
                break
        compatible = own_created is None or (recorded_created is not None and abs(own_created-recorded_created) <= .01)
        end = _time(process_record.get("finished_at")) or _time(process_record.get("exited_at"))
        if compatible and end is not None and end >= started:
            candidates.append((end, "matching_process_end_record", True))
    for guard in guard_records:
        if not isinstance(guard, dict):
            continue
        evidence = _guard_end(guard, status, process_record, started)
        if evidence is not None:
            candidates.append(evidence)
    if not candidates:
        return result
    # The first confirmed end bounds this attempt; later controller confirmation
    # must not extend it or include the time spent paused before a restart.
    end, source, estimated = min(candidates, key=lambda candidate: candidate[0])
    result.update(finished_at=end.isoformat(), wall_seconds=(end-started).total_seconds(),
                  timing_source=source, timing_start_source=start_source,
                  wall_seconds_estimated=estimated or start_source == "process_create_time",
                  timing_incomplete=False)
    if status.get("state") == "running":
        result["state"] = "paused" if source in {"guard_pid_create_time", "parent_guard_process_tree",
                                               "legacy_guard_observed_at_estimate"} else "failed"
        result["recovery_reason"] = "confirmed_deadline_stop" if result["state"] == "paused" else "process_exit_without_final_status"
    return result
