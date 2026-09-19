"""One scenario/method worker; controlled by run_experiments.py."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

from src.candidate_network import generate_candidate_network, save_candidate_network
from src.experiment_control import (DEFAULT_DEADLINE, ExperimentPaused, atomic_json,
                                    check_deadline, fingerprint, utc_now)
from src.experiment_metrics import trajectory_metrics
from src.experiment_recovery import (OutputDirectoryLock, ExperimentAlreadyRunning,
    inspect_process_identity)
from src.parameters import BusinessParameters
from src.result_statistics import build_result_statistics, write_statistics_artifacts
from src.rolling_runner import run_rolling_mpc
from src.scenario import load_scenario


def _capture_worker_source(directory):
    """Archive the worker's current files separately from its launcher version."""
    import hashlib
    source_root=Path(__file__).resolve().parent
    paths=[source_root/"experiment_worker.py",source_root/"deadline_guard.py"]
    paths+=sorted((source_root/"src").glob("*.py"))
    contents={str(path.relative_to(source_root)):path.read_bytes() for path in paths}
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}
    snapshot=directory/"source_snapshots"/fingerprint(hashes)
    for name,data in contents.items():
        target=snapshot/name
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.read_bytes()!=data:
                raise ValueError("immutable worker source snapshot mismatch")
        else:
            temporary=target.with_suffix(target.suffix+".tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
    atomic_json(snapshot/"manifest.json",hashes)
    return str(snapshot),hashes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--deadline",default=DEFAULT_DEADLINE)
    parser.add_argument("--no-deadline",dest="deadline",action="store_const",const=None,
                        help="Run without the legacy global experiment deadline")
    parser.add_argument("--horizon",type=int)
    parser.add_argument("--time-limit",type=float)
    parser.add_argument("--model",type=Path)
    parser.add_argument("--feature-variant",choices=["full","inventory_only","simple_inventory"],default="full")
    parser.add_argument("--capture-features",action="store_true")
    parser.add_argument("--source-snapshot",default=None)
    parser.add_argument("--route-mode",choices=["joint","dayahead"],default="joint")
    parser.add_argument("--charging-mode",choices=["joint","baseline"],default="joint")
    args=parser.parse_args(argv)
    directory=args.output_dir.resolve()
    directory.mkdir(parents=True,exist_ok=True)
    with OutputDirectoryLock(directory,name=".worker.lock"):
        existing_path=directory/"worker_status.json"
        if existing_path.exists():
            existing=json.loads(existing_path.read_text(encoding="utf-8"))
            process_path=directory/"process.json"
            process_record=json.loads(process_path.read_text(encoding="utf-8")) if process_path.exists() else {}
            created=existing.get("created")
            if created is None and process_record.get("worker_pid")==existing.get("pid"):
                created=process_record.get("created")
            if existing.get("pid") and existing["pid"]!=__import__("os").getpid():
                identity=inspect_process_identity(existing["pid"],created,
                    script_name="experiment_worker.py",output_dir=directory,output_flag="--output-dir")
                if identity["state"] in {"alive","unknown"}:
                    raise ExperimentAlreadyRunning("another live or unverified worker owns this output directory")
        return _run_owned(args,directory)


def _run_owned(args,directory):
    started=perf_counter()
    prior_attempt_seconds=0.
    prior_timing_incomplete=False
    prior_timing_estimated=False
    grouped_attempts={}
    for prior_path in sorted((directory/"attempt_history").glob("*.json")):
        prior=json.loads(prior_path.read_text(encoding="utf-8"))
        identity=(prior.get("pid"),prior.get("started_at")) if prior.get("pid") and prior.get("started_at") else str(prior_path)
        duration=prior.get("wall_seconds")
        if duration is None and prior.get("started_at") and prior.get("finished_at"):
            from datetime import datetime
            duration=(datetime.fromisoformat(prior["finished_at"])-
                      datetime.fromisoformat(prior["started_at"])).total_seconds()
        if duration is not None:
            import math
            if not math.isfinite(float(duration)) or float(duration)<0:
                raise ValueError("invalid prior attempt duration")
            duration=float(duration)
        grouped_attempts.setdefault(identity,[]).append((prior,duration))
    for records in grouped_attempts.values():
        known=[record for record in records if record[1] is not None]
        if not known:
            prior_timing_incomplete=True
            continue
        if max(record[1] for record in known)-min(record[1] for record in known)>.05:
            raise ValueError("conflicting durations for the same prior attempt")
        # A later sealed record supersedes an older copy without end evidence.
        prior,duration=min(known,key=lambda item:(bool(item[0].get("timing_incomplete",False)),
                                                 bool(item[0].get("wall_seconds_estimated",False))))
        prior_attempt_seconds+=duration
        prior_timing_incomplete=prior_timing_incomplete or bool(prior.get("timing_incomplete",False))
        prior_timing_estimated=prior_timing_estimated or bool(prior.get("wall_seconds_estimated",False))
    status={"state":"running","started_at":utc_now().isoformat(),"scenario":str(args.scenario.resolve()),
            "route_mode":args.route_mode,"charging_mode":args.charging_mode,
            "model":str(args.model.resolve()) if args.model else None,
            "deadline":args.deadline,"pid":__import__("os").getpid(),
            "created":__import__("psutil").Process().create_time(),
            "source_snapshot":args.source_snapshot,"prior_attempt_seconds":prior_attempt_seconds,
            "runtime_timing_incomplete":prior_timing_incomplete,
            "runtime_timing_estimated":prior_timing_estimated}
    atomic_json(directory/"worker_status.json",status)
    try:
        check_deadline(args.deadline)
        actual_snapshot,actual_hashes=_capture_worker_source(directory)
        status.update(source_snapshot=actual_snapshot,
                      launcher_source_snapshot=args.source_snapshot)
        atomic_json(directory/"worker_status.json",status)
        scenario=load_scenario(args.scenario)
        params=BusinessParameters.from_dict(scenario.params)
        if args.horizon is not None:
            params.horizon=args.horizon
        if args.time_limit is not None:
            params.solver.time_limit_sec=args.time_limit
        params.validate()
        atomic_json(directory/"input_manifest.json",{
            "scenario_hash":fingerprint(scenario.to_dict()),"parameters":params.to_dict(),
            "python":sys.version,"platform":platform.platform(),
            "source_snapshot":actual_snapshot,"source_hashes":actual_hashes,
            "launcher_source_snapshot":args.source_snapshot,
            "deadline":args.deadline,"arguments":vars(args) | {
                "scenario":str(args.scenario),"output_dir":str(args.output_dir),
                "model":str(args.model) if args.model else None}})
        network=generate_candidate_network(params)
        save_candidate_network(network,directory/"network.json")
        model=None
        spec=None
        if args.model or args.capture_features:
            from src.terminal_features import FeatureSpec
            from src.terminal_value import TerminalValueModel
            if args.model:
                model=TerminalValueModel.load(args.model)
            variant=model.variant if model is not None else args.feature_variant
            spec=FeatureSpec(params,variant=variant)
            atomic_json(directory/"feature_manifest.json",{
                "variant":spec.variant,"dimension":spec.dimension,"names":list(spec.names)})
        def progress(record):
            print(json.dumps({"period":record["period"]+1,"status":record["solution"]["status"],
                              "solve_seconds":record["solution"]["solve_seconds"],
                              "model_wall_seconds":record["model_wall_seconds"],
                              "gap":record["solution"]["mip_gap"],
                              "actual_reward":record["reward"]}),flush=True)
        result=run_rolling_mpc(params,scenario,network,progress=progress,
            terminal_model=model,feature_spec=spec,route_mode=args.route_mode,
            charging_mode=args.charging_mode,deadline=args.deadline,journal_dir=directory/"journal")
        check_deadline(args.deadline)
        with gzip.open(directory/"result.json.gz","wt",encoding="utf-8",compresslevel=3) as stream:
            json.dump(result,stream,ensure_ascii=False,separators=(",",":"),allow_nan=False)
        statistics=build_result_statistics(result)
        write_statistics_artifacts(statistics,directory,"statistics")
        metrics=trajectory_metrics(result,scenario.to_dict())
        metrics["current_attempt_wall_seconds"]=perf_counter()-started
        metrics["prior_attempt_wall_seconds"]=prior_attempt_seconds
        metrics["known_runtime_seconds"]=prior_attempt_seconds+metrics["current_attempt_wall_seconds"]
        metrics["runtime_timing_incomplete"]=prior_timing_incomplete
        metrics["runtime_timing_estimated"]=prior_timing_estimated
        metrics["worker_wall_seconds"]=None if prior_timing_incomplete else metrics["known_runtime_seconds"]
        atomic_json(directory/"metrics.json",metrics)
        status.update(state="complete",finished_at=utc_now().isoformat(),
                      wall_seconds=perf_counter()-started,metrics=metrics)
        atomic_json(directory/"worker_status.json",status)
        print(json.dumps(status,ensure_ascii=False),flush=True)
        return 0
    except Exception as exc:
        status.update(state="paused" if isinstance(exc,ExperimentPaused) else "failed",
                      finished_at=utc_now().isoformat(),wall_seconds=perf_counter()-started,
                      error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        atomic_json(directory/"worker_status.json",status)
        print(json.dumps(status,ensure_ascii=False),flush=True)
        return 75 if isinstance(exc,ExperimentPaused) else 1


if __name__=="__main__":
    raise SystemExit(main())
