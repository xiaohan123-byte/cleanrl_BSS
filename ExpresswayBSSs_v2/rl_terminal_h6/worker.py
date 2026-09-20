"""One isolated rollout or joint neural fitting job; no parallel day jobs."""
from __future__ import annotations
import bootstrap
import argparse
import os
import traceback
import numpy as np
from pathlib import Path
from protocol import (atomic, read, now, immutable, digest, file_hash, scenario_for,
                      load_result, save_result)
from learned_value import LearnedValue
from state_encoding import StateSpec
from src.experiment_recovery import OutputDirectoryLock
from src.rolling_runner import run_rolling_mpc
from src.result_statistics import build_result_statistics
from src.experiment_metrics import trajectory_metrics


def rollout(args, identity):
    plan = read(args.plan)
    penalty = plan["business_overrides"]["reservation_failure_penalty"]
    params, scenario, network = scenario_for(args.dataset, args.day, failure_penalty=penalty)
    atomic(args.job/"scenario_identity.json", {
        "source_scenario_hash": read(Path(args.dataset)/"manifest.json")["scenario_hashes"][f"scenarios/day_{args.day:02d}.json"],
        "effective_scenario_hash": digest(scenario.to_dict()), "business_overrides": plan["business_overrides"]})
    policy = LearnedValue.load(args.model) if args.model else None
    spec = StateSpec(params)
    if policy:
        policy.validate_params(params)
    def progress(row):
        print(f"period={row['period']} status={row['solution']['status']} gap={row['solution']['mip_gap']}", flush=True)
        atomic(args.job/"progress.json", {"time": now(), "completed_periods": row["period"]+1,
               "status": row["solution"]["status"], "actual_gap": row["solution"]["mip_gap"]})
    result = run_rolling_mpc(params, scenario, network, progress=progress,
                            terminal_model=policy, feature_spec=spec, journal_dir=args.job/"journal")
    result["experiment_identity"] = identity
    statistics = build_result_statistics(result)
    metrics = trajectory_metrics(result, scenario.to_dict())
    metrics["reservation_failure_penalty_yuan"] = params.reservation_failure_penalty
    demand_rounds = result["rounds"][:params.num_periods]
    for label, values in {
        "solver_demand_seconds": [r["solution"]["solve_seconds"] for r in demand_rounds],
        "model_wall_demand_seconds": [r["model_wall_seconds"] for r in demand_rounds],
    }.items():
        metrics[label+"_mean"] = float(np.mean(values))
        metrics[label+"_p95"] = float(np.percentile(values, 95))
    metrics["solver_statistics_scope"] = "solver_demand_*: first 96 periods; solver_seconds_*: all executed periods including drain"
    if metrics["reservation_count"] != 200 or metrics["actual_random_count"] != 200:
        raise ValueError("fixed daily demand count is not 200+200")
    if result["dayahead_plan"] != read(Path(args.dataset)/f"dayahead/day_{args.day:02d}.json"):
        raise ValueError("realized initial plans differ from frozen day-ahead plans")
    # Checks both MC settlement and the final absorbing state even in validation/test.
    from training import monte_carlo_states
    states, returns = monte_carlo_states(result)
    atomic(args.job/"training_targets.json", {"periods": [s["period"] for s in states],
           "returns_yuan": returns.tolist(), "state_input_hash": digest(states)})
    save_result(args.job, result)
    atomic(args.job/"statistics.json", statistics)
    atomic(args.job/"metrics.json", metrics)
    return ["result.json.gz", "statistics.json", "metrics.json", "training_targets.json", "scenario_identity.json"]


def fit_job(args, identity):
    from training import fit, monte_carlo_states
    plan = read(args.plan)
    batch = plan["split"]["training_batches"][args.iteration-1]
    params, _, _ = scenario_for(args.dataset, batch[0],
                               failure_penalty=plan["business_overrides"]["reservation_failure_penalty"])
    states, targets, provenance = [], [], []
    for day in batch:
        directory = args.output/f"iteration_{args.iteration}"/"train"/f"day_{day:02d}"
        result = load_result(directory)
        rid = result["experiment_identity"]
        if rid["policy_hash"] != identity["policy_hash"] or rid["plan_hash"] != identity["plan_hash"]:
            raise ValueError("training rollout used a different policy/plan")
        if result["parameter_snapshot"]["reservation_failure_penalty"] != params.reservation_failure_penalty:
            raise ValueError("training trajectory uses a different failure penalty")
        s, g = monte_carlo_states(result)
        states.extend(s)
        targets.extend(g.tolist())
        provenance.append({"day": day, "result_hash": file_hash(directory/"result.json.gz"), "samples": len(s)})
    previous = LearnedValue.load(args.model) if args.model else None
    def progress(record):
        print(record, flush=True)
        atomic(args.job/"progress.json", {"time": now(), **record})
    model, diagnostics = fit(states, targets, StateSpec(params), plan["contract"], previous=previous,
                             iteration=args.iteration-1, device="cuda", progress=progress)
    model.record["training_provenance"] = {"plan_hash": digest(plan), "days": batch,
        "trajectories": provenance, "samples_hash": digest(states), "targets_hash": digest(targets)}
    model.save(args.job/"model.json")
    atomic(args.job/"fit_diagnostics.json", diagnostics)
    atomic(args.job/"provenance.json", model.record["training_provenance"])
    return ["model.json", "fit_diagnostics.json", "provenance.json"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["rollout", "fit"])
    for name in ("dataset", "output", "job", "plan"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--day", type=int)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    args.job.mkdir(parents=True, exist_ok=True)
    identity = {"kind": args.kind, "day": args.day, "iteration": args.iteration,
                "plan_hash": digest(read(args.plan)),
                "policy_hash": file_hash(args.model) if args.model else "zero"}
    with OutputDirectoryLock(args.job, ".worker.lock"):
        immutable(args.job/"identity.json", identity)
        status_path = args.job/"status.json"
        if status_path.exists() and read(status_path)["state"] == "complete":
            status = read(status_path)
            for name, checksum in status["artifacts"].items():
                if file_hash(args.job/name) != checksum:
                    raise ValueError("complete artifact hash mismatch")
            return
        attempt = now().replace(":", "-")
        atomic(status_path, {"state": "running", "pid": os.getpid(), "started": now(), "identity": identity})
        try:
            artifacts = rollout(args, identity) if args.kind == "rollout" else fit_job(args, identity)
            status = {"state": "complete", "finished": now(), "identity": identity,
                      "artifacts": {name: file_hash(args.job/name) for name in artifacts}}
            atomic(status_path, status)
            atomic(args.job/"attempts"/(attempt+".json"), status)
        except BaseException as error:
            status = {"state": "failed", "finished": now(), "identity": identity,
                      "error": repr(error), "traceback": traceback.format_exc()}
            atomic(status_path, status)
            atomic(args.job/"attempts"/(attempt+".json"), status)
            raise


if __name__ == "__main__":
    main()
