"""Run only the frozen 35 seed-1 real-data zero-terminal jobs, sequentially.

--prepare-only validates and freezes the 70-day dataset without starting COPT.
Completed jobs are reused only with exactly matching source and input hashes.
No RL training, repeated seeds, automatic reruns, or paper updates are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import psutil

from src.experiment_control import atomic_json, fingerprint, utc_now
from src.experiment_recovery import OutputDirectoryLock
from src.real_data_experiment import prepare_dataset, immutable_json, HORIZONS
from src.atomic_io import retry_atomic_writer

atomic_json = retry_atomic_writer(atomic_json)

ROOT = Path(__file__).resolve().parent


def reviewed_io_failure(status):
    trace = status.get("traceback", "")
    return (status.get("error_type") == "PermissionError"
            and any(f"WinError {code}" in status.get("error", "") for code in (5, 32, 33))
            and "experiment_control.py" in trace and "os.replace(pending, path)" in trace)


def recovery_snapshot(output):
    contents = {"io_retry_worker.py": (ROOT / "io_retry_worker.py").read_bytes(),
                "atomic_io.py": (ROOT / "src/atomic_io.py").read_bytes(),
                "controller.py": Path(__file__).read_bytes()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    directory = output / "recovery_code" / fingerprint(hashes)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        target = directory / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError("immutable recovery wrapper changed")
        if not target.exists():
            target.write_bytes(data)
    immutable_json(directory / "manifest.json", hashes)
    return directory


def source_snapshot(output):
    files = sorted(list((ROOT / "src").glob("*.py")) +
                   [ROOT / "experiment_worker.py", ROOT / "run_zero_terminal.py", ROOT / "deadline_guard.py"] +
                   [ROOT / "paper_v2/sections" / name for name in
                    ("03_model.tex", "04_terminal_value_learning.tex", "05_numerical_experiments.tex")])
    contents = {str(p.relative_to(ROOT)): p.read_bytes() for p in files}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    directory = output / "code_snapshots" / fingerprint(hashes)
    for name, data in contents.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError("immutable code snapshot changed")
        if not path.exists():
            path.write_bytes(data)
    immutable_json(directory / "manifest.json", hashes)
    return directory, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/final_real_data_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/zero_terminal_real_data_seed1_v1")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-frozen", action="store_true",
                        help="Use the existing immutable plan, dataset and original model source")
    parser.add_argument("--resume-reviewed-model-fix", action="store_true",
                        help="Resume after reviewing the current source as a mathematically invariant numerical fix")
    parser.add_argument("--retry-io-failure", action="store_true",
                        help="Resume an inspected Windows atomic-status-write failure from its journal")
    args = parser.parse_args(argv)
    output, dataset_dir = args.output_dir.resolve(), args.dataset_dir.resolve()
    with OutputDirectoryLock(output):
        if (args.retry_io_failure or args.resume_reviewed_model_fix) and not args.resume_frozen:
            raise ValueError("reviewed recovery requires --resume-frozen")
        if args.retry_io_failure and args.resume_reviewed_model_fix:
            raise ValueError("use one reviewed recovery mode at a time")
        if args.resume_frozen and not args.resume_reviewed_model_fix:
            saved_plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
            dataset_dir = Path(saved_plan["dataset_dir"])
            manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
            if fingerprint(manifest) != saved_plan["dataset_manifest_hash"]:
                raise ValueError("frozen dataset manifest changed")
            hashes = saved_plan["source_hashes"]
            snapshot = output / "code_snapshots" / fingerprint(hashes)
            for name, digest in hashes.items():
                if hashlib.sha256((snapshot / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"frozen source changed: {name}")
            prior_controller = json.loads((output / "status.json").read_text(encoding="utf-8"))
            immutable_json(output / "controller_history" / f"attempt_{prior_controller['created']}.json", prior_controller)
            recovery = recovery_snapshot(output)
        else:
            if args.resume_frozen:
                saved_plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
                dataset_dir = Path(saved_plan["dataset_dir"])
                manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
                if fingerprint(manifest) != saved_plan["dataset_manifest_hash"]:
                    raise ValueError("frozen dataset manifest changed")
                prior_controller = json.loads((output / "status.json").read_text(encoding="utf-8"))
                immutable_json(output / "controller_history" / f"attempt_{prior_controller['created']}.json", prior_controller)
                # Keep the bounded JSON-write retry wrapper on workers in this mode too.
                recovery = recovery_snapshot(output)
            else:
                recovery = None
            manifest = prepare_dataset(args.data_dir, dataset_dir)
        if args.prepare_only:
            print(json.dumps({"state": "prepared", "dataset": str(dataset_dir),
                "test_days": manifest["test_day_ids_by_weekday"], "jobs": 35}, ensure_ascii=False))
            return 0
        status_recovering = {}
        if args.resume_reviewed_model_fix:
            snapshot, hashes = source_snapshot(output)
            status_recovering = {"reviewed_model_fix": "execution power bound checks accept solver feasibility residuals plus physically negligible relative noise, clamped to bounds"}
            plan = {**saved_plan, "source_hashes": hashes,
                    "reviewed_model_fix": "execution power bound checks accept solver feasibility residuals plus physically negligible relative noise, clamped to bounds",
                    "earlier_completed_groups": "unchanged; h8 and later use the reviewed fix"}
        elif not args.resume_frozen:
            snapshot, hashes = source_snapshot(output)
        days = manifest["test_day_ids_by_weekday"]
        if not args.resume_reviewed_model_fix:
            plan = {"dataset_manifest_hash": fingerprint(manifest), "source_hashes": hashes,
                "dataset_dir": str(dataset_dir), "horizons": list(HORIZONS), "test_days": days,
                "seed": 1, "beta": 0, "worker_concurrency": 1, "solver_threads": 16,
                "time_limit_seconds": 120, "relative_gap_target": .0001,
                "deadline": None, "post_experiment_analysis": "deferred until user requests it",
                "jobs": [{"job_id": f"h{h}_day{d:02d}", "horizon": h, "day_id": d,
                          "weekday": (d-1)%7+1} for h in HORIZONS for d in days]}
        if args.resume_reviewed_model_fix:
            atomic_json(output / "plan_reviewed_fix.json", plan)
        else:
            immutable_json(output / "plan.json", plan)
        environment = {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "cpu_logical": psutil.cpu_count(),
            "cpu_physical": psutil.cpu_count(logical=False), "ram_bytes": psutil.virtual_memory().total,
            "versions": {p: importlib.metadata.version(p) for p in ("coptpy", "numpy", "scipy", "psutil")}}
        atomic_json(output / "execution_environment.json", environment)
        created = psutil.Process().create_time()
        status = {"state": "running", "pid": os.getpid(), "created": created,
                  "started_at": utc_now().isoformat(), "total_jobs": 35, "completed_jobs": [],
                  "current_job": None, "source_snapshot": str(snapshot)}
        status.update(status_recovering)
        if recovery:
            status.update(recovery_code=str(recovery), recovery_kind="bounded JSON-write retry only",
                          reviewed_io_failure_retry=args.retry_io_failure)
        atomic_json(output / "status.json", status)
        controller_history = output / "controller_history"
        if args.resume_frozen and controller_history.exists():
            preserved = [json.loads(path.read_text(encoding="utf-8"))
                         for path in sorted(controller_history.glob("*.json"))]
            if not preserved:
                raise ValueError("missing preserved controller status")
            completed = list(dict.fromkeys(
                item for record in preserved for item in record.get("completed_jobs", [])))
        else:
            completed = []
        status["completed_jobs"] = completed
        atomic_json(output / "status.json", status)
        try:
            for job in plan["jobs"]:
                directory = output / "runs" / job["job_id"]
                directory.mkdir(parents=True, exist_ok=True)
                scenario = dataset_dir / "scenarios" / f"day_{job['day_id']:02d}.json"
                value = json.loads(scenario.read_text(encoding="utf-8"))
                expected = manifest["scenario_hashes"][f"scenarios/day_{job['day_id']:02d}.json"]
                if fingerprint(value) != expected:
                    raise ValueError("frozen scenario changed")
                identity = {**job, "source_hash": fingerprint(hashes), "scenario_hash": expected}
                marker = directory / "worker_status.json"
                previous = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None
                if previous and previous["state"] == "complete":
                    if not all((directory / f).exists() for f in ("result.json.gz", "metrics.json", "statistics.json")):
                        raise ValueError("completed job missing artifacts")
                    if job["job_id"] not in status["completed_jobs"]:
                        status["completed_jobs"].append(job["job_id"])
                        atomic_json(output / "status.json", status)
                    continue
                if args.resume_reviewed_model_fix:
                    # The original immutable job identity keeps its frozen source
                    # hash; the reviewed fix is recorded in a separate artifact.
                    atomic_json(directory / "job_reviewed_fix.json", identity)
                else:
                    immutable_json(directory / "job.json", identity)
                if previous:
                    # Failed attempts require explicit diagnosis; never silently repeat them.
                    if previous["state"] == "failed" and not (args.retry_io_failure and reviewed_io_failure(previous)) \
                            and not (args.resume_reviewed_model_fix
                                     and previous.get("error", "").startswith("invalid slot charging power")):
                        raise RuntimeError(f"previous failed job needs review: {job['job_id']}")
                    immutable_json(directory / "attempt_history" / f"attempt_{previous['created']}.json", previous)
                command = [sys.executable, "-u", str(snapshot / "experiment_worker.py"),
                           "--scenario", str(scenario), "--output-dir", str(directory),
                           "--horizon", str(job["horizon"]), "--no-deadline", "--source-snapshot", str(snapshot)]
                if recovery:
                    command = [sys.executable, "-u", str(recovery / "io_retry_worker.py"),
                               "--frozen-source", str(snapshot), *command[3:]]
                    io_record = directory / "io_recovery.json"
                    if io_record.exists():
                        io_record = directory / f"io_recovery_{fingerprint(hashes)[:12]}.json"
                    atomic_json(io_record, {
                        "recovery_code": str(recovery), "original_source": str(snapshot),
                        "original_source_hash": fingerprint(hashes), "scenario_hash": expected,
                        "change": "bounded retries of atomic JSON writes; solver/model/data unchanged",
                        "created_at": utc_now().isoformat()})
                status.update(current_job=job["job_id"], worker_pid=None, updated_at=utc_now().isoformat())
                atomic_json(output / "status.json", status)
                env = os.environ.copy()
                env.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
                with (directory / "console.log").open("a", encoding="utf-8") as log:
                    worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    status.update(worker_pid=worker.pid, worker_created=psutil.Process(worker.pid).create_time())
                    atomic_json(output / "status.json", status)
                    print(json.dumps({"state": "started", **job, "worker_pid": worker.pid}), flush=True)
                    code = worker.wait()
                outcome = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
                if code != 0 or outcome.get("state") != "complete":
                    raise RuntimeError(f"{job['job_id']} stopped (exit {code}): {outcome.get('error', 'no completion marker')}")
                status["completed_jobs"].append(job["job_id"])
                atomic_json(output / "status.json", status)
                print(json.dumps({"state": "complete", "job": job["job_id"],
                                  "completed": len(status["completed_jobs"]), "total": 35}), flush=True)
            status.update(state="complete", current_job=None, worker_pid=None, finished_at=utc_now().isoformat())
            atomic_json(output / "status.json", status)
            return 0
        except Exception as exc:
            status.update(state="failed", error=str(exc), traceback=traceback.format_exc(),
                          finished_at=utc_now().isoformat())
            atomic_json(output / "status.json", status)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
