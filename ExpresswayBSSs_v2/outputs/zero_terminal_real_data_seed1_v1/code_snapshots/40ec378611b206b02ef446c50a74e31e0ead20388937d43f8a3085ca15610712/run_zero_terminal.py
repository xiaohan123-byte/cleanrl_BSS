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

ROOT = Path(__file__).resolve().parent


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
    args = parser.parse_args(argv)
    output, dataset_dir = args.output_dir.resolve(), args.dataset_dir.resolve()
    with OutputDirectoryLock(output):
        manifest = prepare_dataset(args.data_dir, dataset_dir)
        if args.prepare_only:
            print(json.dumps({"state": "prepared", "dataset": str(dataset_dir),
                "test_days": manifest["test_day_ids_by_weekday"], "jobs": 35}, ensure_ascii=False))
            return 0
        snapshot, hashes = source_snapshot(output)
        days = manifest["test_day_ids_by_weekday"]
        plan = {"dataset_manifest_hash": fingerprint(manifest), "source_hashes": hashes,
                "dataset_dir": str(dataset_dir), "horizons": list(HORIZONS), "test_days": days,
                "seed": 1, "beta": 0, "worker_concurrency": 1, "solver_threads": 16,
                "time_limit_seconds": 120, "relative_gap_target": .0001,
                "deadline": None, "post_experiment_analysis": "deferred until user requests it",
                "jobs": [{"job_id": f"h{h}_day{d:02d}", "horizon": h, "day_id": d,
                          "weekday": (d-1)%7+1} for h in HORIZONS for d in days]}
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
                immutable_json(directory / "job.json", identity)
                marker = directory / "worker_status.json"
                if marker.exists():
                    previous = json.loads(marker.read_text(encoding="utf-8"))
                    if previous["state"] == "complete":
                        if not all((directory / f).exists() for f in ("result.json.gz", "metrics.json", "statistics.json")):
                            raise ValueError("completed job missing artifacts")
                        status["completed_jobs"].append(job["job_id"])
                        atomic_json(output / "status.json", status)
                        continue
                    # Failed attempts require explicit diagnosis; never silently repeat them.
                    if previous["state"] == "failed":
                        raise RuntimeError(f"previous failed job needs review: {job['job_id']}")
                    immutable_json(directory / "attempt_history" / f"attempt_{previous['created']}.json", previous)
                command = [sys.executable, "-u", str(snapshot / "experiment_worker.py"),
                           "--scenario", str(scenario), "--output-dir", str(directory),
                           "--horizon", str(job["horizon"]), "--no-deadline", "--source-snapshot", str(snapshot)]
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
