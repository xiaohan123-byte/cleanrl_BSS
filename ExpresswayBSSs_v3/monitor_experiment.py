"""Sample only a named experiment's small status files and worker resources.

The monitor appends observations to monitor_samples.jsonl. It never reads round
journals or console logs, changes experiment status, or controls worker processes.
"""
from __future__ import annotations

import argparse
from datetime import timedelta, timezone
import json
import os
from pathlib import Path
import time

import psutil

from src.experiment_control import DEFAULT_DEADLINE, parse_deadline, seconds_remaining, utc_now

BEIJING = timezone(timedelta(hours=8))
MAX_STATUS_BYTES = 256 * 1024
SKIP_DIRECTORIES = {"journal", "diagnostics", "attempt_history", "__pycache__"}


def _read_small(path):
    try:
        if path.stat().st_size > MAX_STATUS_BYTES:
            return {}, "status_file_exceeds_size_limit"
        with path.open("rb") as stream:
            content = stream.read(MAX_STATUS_BYTES + 1)
        if len(content) > MAX_STATUS_BYTES:
            return {}, "status_file_exceeds_size_limit"
        value = json.loads(content)
        return (value, None) if isinstance(value, dict) else ({}, "status_is_not_object")
    except FileNotFoundError:
        return {}, None
    except (OSError, ValueError) as error:
        return {}, type(error).__name__


def _worker_directories(root):
    # Prune before descent: round journals and solver diagnostics can be large.
    for base in (root / "pilot", root / "formal_suite" / "jobs"):
        if not base.is_dir():
            continue
        for current, directories, filenames in os.walk(base):
            directories[:] = sorted(name for name in directories if name not in SKIP_DIRECTORIES)
            if "worker_status.json" in filenames or "process.json" in filenames:
                yield Path(current)
                directories[:] = []


def collect_sample(root, previous_cpu):
    now = utc_now()
    memory = psutil.virtual_memory()
    controller, controller_error = _read_small(root / "controller_status.json")
    sample = {
        "schema_version": 1,
        "time_utc": now.isoformat(),
        "time_beijing": now.astimezone(BEIJING).isoformat(),
        "physical_memory_available_bytes": memory.available,
        "physical_memory_total_bytes": memory.total,
        "physical_memory_used_percent": memory.percent,
        "controller_state": controller.get("state"),
        "workers": [],
    }
    if controller_error:
        sample["controller_read_error"] = controller_error
    next_cpu = {}
    for directory in _worker_directories(root):
        status, status_error = _read_small(directory / "worker_status.json")
        journal, journal_error = _read_small(directory / "journal" / "status.json")
        process_record, process_error = _read_small(directory / "process.json")
        item = {
            "job": directory.relative_to(root).as_posix(),
            "worker_state": status.get("state"),
            "journal_state": journal.get("state"),
            "completed_periods": journal.get("completed_periods"),
            "pid": process_record.get("worker_pid", status.get("pid")),
            "process_observation": "identity_unavailable",
            "rss_bytes": None,
            "cpu_seconds_total": None,
            "cpu_percent_since_previous_sample": None,
        }
        for name, error in (("worker_status", status_error), ("journal_status", journal_error),
                            ("process", process_error)):
            if error:
                item.setdefault("read_errors", {})[name] = error
        pid, created = process_record.get("worker_pid"), process_record.get("created")
        if isinstance(pid, int) and isinstance(created, (int, float)):
            try:
                process = psutil.Process(pid)
                with process.oneshot():
                    actual_created = process.create_time()
                    if abs(actual_created - created) > .01:
                        item["process_observation"] = "pid_reused"
                    elif status.get("pid") not in (None, pid):
                        item["process_observation"] = "status_pid_mismatch"
                    else:
                        cpu = process.cpu_times()
                        cpu_total = cpu.user + cpu.system
                        observed = time.monotonic()
                        identity = (pid, actual_created)
                        prior = previous_cpu.get(identity)
                        item.update(process_observation="observed", process_status=process.status(),
                                    rss_bytes=process.memory_info().rss, cpu_seconds_total=cpu_total)
                        if prior is not None and observed > prior[0]:
                            item["cpu_percent_since_previous_sample"] = round(
                                100. * max(0., cpu_total - prior[1]) / (observed - prior[0]), 3)
                        next_cpu[identity] = (observed, cpu_total)
            except psutil.NoSuchProcess:
                item["process_observation"] = "exited"
            except psutil.AccessDenied:
                item["process_observation"] = "access_denied"
        sample["workers"].append(item)
    sample["observed_worker_count"] = sum(w["process_observation"] == "observed" for w in sample["workers"])
    sample["observed_worker_rss_bytes"] = sum(w["rss_bytes"] or 0 for w in sample["workers"])
    sample["cpu_percent_convention"] = "100 percent equals one CPU core; first observation is null"
    return sample, next_cpu


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--deadline", default=DEFAULT_DEADLINE)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    if not root.is_dir():
        parser.error("--output-root must name an existing experiment directory")
    deadline = parse_deadline(args.deadline)
    destination = root / "monitor_samples.jsonl"
    previous_cpu = {}
    samples = 0
    print(json.dumps({"monitor_pid": os.getpid(), "output": str(destination),
                      "deadline": args.deadline, "sample_interval_seconds": 60}), flush=True)
    while seconds_remaining(deadline) > 0:
        started = time.monotonic()
        sample, previous_cpu = collect_sample(root, previous_cpu)
        if seconds_remaining(deadline) <= 0:
            break
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
        samples += 1
        if args.once:
            print(json.dumps({"samples_written": samples,
                              "observed_worker_count": sample["observed_worker_count"],
                              "physical_memory_available_bytes": sample["physical_memory_available_bytes"]}),
                  flush=True)
            return 0
        delay = min(max(0., 60. - (time.monotonic() - started)), max(0., seconds_remaining(deadline)))
        if delay > 0:
            time.sleep(delay)
    print(json.dumps({"monitor_state": "finished", "reason": "deadline", "samples_written": samples}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
