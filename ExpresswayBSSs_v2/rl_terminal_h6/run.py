"""Serial 56-train / 28-validation / 7-test controller with frozen source."""
from __future__ import annotations
import bootstrap
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from protocol import (atomic, read, now, immutable, digest, file_hash, source_hashes,
                      build_plan, scenario_for)
from src.experiment_recovery import OutputDirectoryLock


def prepare(args):
    if args.failure_penalty is None:
        raise ValueError("preparation requires explicit --failure-penalty 100 or 200")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes(bootstrap.ROOT)
    code_hash = digest(hashes)
    snapshot = args.output/"code_snapshots"/code_hash
    for relative, checksum in hashes.items():
        target = snapshot/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_hash(target) != checksum:
            raise ValueError("source snapshot collision")
        if not target.exists():
            shutil.copy2(bootstrap.ROOT/relative, target)
    immutable(args.output/"source_manifest.json", hashes)
    manifest = read(args.dataset/"manifest.json")
    for name, checksum in manifest["source_hashes"].items():
        if file_hash(args.dataset/"sources"/name) != checksum:
            raise ValueError(f"raw source changed: {name}")
    for day in range(1, 71):
        scenario_for(args.dataset, day, failure_penalty=args.failure_penalty)
    plan = build_plan(args.dataset, code_hash, failure_penalty=args.failure_penalty)
    immutable(args.output/"plan.json", plan)
    immutable(args.output/"dataset_manifest.json", manifest)
    immutable(args.output/"prepared.json", {"source": str(snapshot.resolve()), "code_hash": code_hash,
              "dataset": str(args.dataset.resolve()), "baseline": str(args.baseline.resolve())})
    print(str(snapshot/"run.py"), flush=True)


def run(args):
    plan = read(args.output/"plan.json")
    if args.failure_penalty is not None and args.failure_penalty != plan["business_overrides"]["reservation_failure_penalty"]:
        raise ValueError("requested failure penalty differs from the frozen plan")
    prepared = read(args.output/"prepared.json")
    if bootstrap.ROOT != Path(prepared["source"]):
        raise ValueError("formal execution must use the prepared frozen run.py")
    if digest(source_hashes(bootstrap.ROOT)) != plan["code_hash"]:
        raise ValueError("frozen code changed")
    if digest(read(args.dataset/"manifest.json")) != plan["dataset_manifest_hash"]:
        raise ValueError("frozen dataset manifest changed")
    # The other window's baseline owns its files and CPU until completion.
    if read(args.baseline/"status.json").get("state") != "complete":
        raise ValueError("zero-terminal controller must be complete before this serial experiment")
    if (args.output/"status.json").exists():
        status = read(args.output/"status.json")
        if status["state"] == "complete":
            return
        if not args.resume:
            raise ValueError("existing formal run requires --resume; committed rounds are reused")
    completed = []
    def status(state, **fields):
        atomic(args.output/"status.json", {"state": state, "time": now(), "pid": os.getpid(),
            "plan_hash": digest(plan), "completed_jobs": completed.copy(), **fields})
    status("running", current_job=None)
    def child(kind, directory, iteration, day=None, model=None):
        relative = directory.relative_to(args.output).as_posix()
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(bootstrap.ROOT/"worker.py"), kind,
                   "--dataset", str(args.dataset), "--output", str(args.output),
                   "--plan", str(args.output/"plan.json"), "--job", str(directory),
                   "--iteration", str(iteration)]
        if day is not None:
            command += ["--day", str(day)]
        if model is not None:
            command += ["--model", str(model)]
        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "1"})
        with (directory/"stdout.log").open("ab") as stdout, (directory/"stderr.log").open("ab") as stderr:
            process = subprocess.Popen(command, cwd=str(bootstrap.ROOT), env=env, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            status("running", current_job=relative, worker_pid=process.pid)
            if not (args.output/"launch_receipt.json").exists():
                atomic(args.output/"launch_receipt.json", {"time": now(), "controller_pid": os.getpid(),
                    "first_worker_pid": process.pid, "first_job": relative, "plan_hash": digest(plan)})
            returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"job failed ({returncode}): {relative}; see stderr.log and status.json")
        result = read(directory/"status.json")
        if result["state"] != "complete":
            raise RuntimeError(f"child did not commit complete results: {relative}")
        for name, checksum in result["artifacts"].items():
            if file_hash(directory/name) != checksum:
                raise ValueError(f"child artifact checksum mismatch: {relative}/{name}")
        completed.append(relative)
        status("running", current_job=None)
    try:
        previous = None
        scores = []
        for iteration, batch in enumerate(plan["split"]["training_batches"], 1):
            root = args.output/f"iteration_{iteration}"
            for day in batch:
                child("rollout", root/"train"/f"day_{day:02d}", iteration, day, previous)
            child("fit", root/"fit", iteration, model=previous)
            model = root/"fit"/"model.json"
            profits = []
            for day in plan["split"]["validation"]:
                directory = root/"validation"/f"day_{day:02d}"
                child("rollout", directory, iteration, day, model)
                profits.append(read(directory/"metrics.json")["net_profit_yuan"])
            scores.append({"iteration": iteration, "mean_actual_profit_yuan": sum(profits)/7,
                           "validation_days": plan["split"]["validation"], "model_hash": file_hash(model)})
            atomic(root/"validation_selection_score.json", scores[-1])
            previous = model
        best = max(scores, key=lambda item: (item["mean_actual_profit_yuan"], -item["iteration"]))
        selected = args.output/f"iteration_{best['iteration']}"/"fit"/"model.json"
        immutable(args.output/"selection.json", {"rule": plan["selection"], "candidates": scores, "selected": best})
        immutable(args.output/"selected_model.json", read(selected))
        for day in plan["split"]["test"]:
            child("rollout", args.output/"test"/f"day_{day:02d}", best["iteration"], day, args.output/"selected_model.json")
        # No final test aggregation or manuscript write: requested only on a later user inquiry.
        status("complete", current_job=None, selected_iteration=best["iteration"],
               report_pending_user_request=True)
    except BaseException as error:
        status("failed", error=repr(error), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "baseline"):
        parser.add_argument("--"+name, required=True, type=lambda p: Path(p).resolve())
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--failure-penalty", type=float, choices=[100., 200.])
    args = parser.parse_args()
    with OutputDirectoryLock(args.output):
        if args.prepare_only:
            prepare(args)
        else:
            run(args)


if __name__ == "__main__":
    main()
