"""Run the authorized experiment sequence under a strict wall-clock deadline."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from src.experiment_control import (DEFAULT_DEADLINE, ExperimentPaused, atomic_json,
    check_deadline, fingerprint, seconds_remaining, utc_now)
from src.experiment_recovery import (OutputDirectoryLock, ExperimentAlreadyRunning,
    inspect_process_identity, seal_dead_attempt, attempt_key)
from src.parameters import BusinessParameters
from src.scenario import generate_synthetic_scenario, save_scenario

ROOT=Path(__file__).resolve().parent


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/terminal_experiment.json")
    parser.add_argument("--output-root",type=Path,default=ROOT/"outputs/experiment_20260914")
    parser.add_argument("--deadline",default=DEFAULT_DEADLINE)
    parser.add_argument("--stage",choices=["pilot","formal","paper","all"],default="all")
    parser.add_argument("--pilot-workers",type=int,default=3)
    parser.add_argument("--retry-failed",action="store_true")
    args=parser.parse_args(argv)
    output_root=args.output_root.resolve()
    output_root.mkdir(parents=True,exist_ok=True)
    with OutputDirectoryLock(output_root):
        return _run_owned(args,output_root)


def _optional_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _run_owned(args,output_root):
    previous_controller=_optional_json(output_root/"controller_status.json")
    if previous_controller.get("pid") and previous_controller["pid"]!=os.getpid():
        identity=inspect_process_identity(previous_controller["pid"],previous_controller.get("created"),
            script_name="run_experiments.py",output_dir=output_root,output_flag="--output-root")
        if identity["state"] in {"alive","unknown"}:
            raise ExperimentAlreadyRunning("another live or unverified controller owns this output directory")
    controller={"state":"running","started_at":utc_now().isoformat(),"deadline":args.deadline,
                "stage":args.stage,"pid":os.getpid(),"created":psutil.Process().create_time(),
                "pilot_workers":args.pilot_workers}
    atomic_json(output_root/"controller_status.json",controller)

    def run_job(job):
        check_deadline(args.deadline)
        directory=Path(job["output_dir"]).resolve()
        directory.mkdir(parents=True,exist_ok=True)
        canonical={k:str(v) if isinstance(v,Path) else v for k,v in job.items()}
        canonical["scenario_hash"]=fingerprint(json.loads(Path(job["scenario"]).read_text(encoding="utf-8")))
        if job.get("model"):
            canonical["model_hash"]=fingerprint(json.loads(Path(job["model"]).read_text(encoding="utf-8")))
        identity=fingerprint(canonical)
        job_path=directory/"job.json"
        if job_path.exists():
            saved=json.loads(job_path.read_text(encoding="utf-8"))
            if saved["identity"]!=identity:
                raise ValueError(f"job inputs changed: {job['job_id']}")
        else:
            atomic_json(job_path,{"identity":identity,"job":canonical})
        status_path=directory/"worker_status.json"
        if status_path.exists():
            prior=json.loads(status_path.read_text(encoding="utf-8"))
            if prior["state"]=="complete":
                if not (directory/"metrics.json").exists() or not (directory/"result.json.gz").exists():
                    raise ValueError("completed worker is missing artifacts")
                return prior
            process_record=_optional_json(directory/"process.json")
            prior_created=prior.get("created")
            if prior_created is None and process_record.get("worker_pid")==prior.get("pid"):
                prior_created=process_record.get("created")
            identity=inspect_process_identity(prior.get("pid"),prior_created,
                script_name="experiment_worker.py",output_dir=directory,output_flag="--output-dir")
            if identity["state"] in {"alive","unknown"}:
                raise ExperimentAlreadyRunning("another live or unverified worker owns "+str(directory))
            guards=[_optional_json(directory/"deadline_pause.json"),
                    _optional_json(output_root/"controller_deadline_pause.json")]
            prior=seal_dead_attempt(prior,process_record,guard_records=guards,liveness=identity)
            if prior_created is not None:
                prior.setdefault("created",prior_created)
            atomic_json(status_path,prior)
            if prior["state"]=="failed" and not args.retry_failed:
                return prior
            history=directory/"attempt_history"
            history.mkdir(exist_ok=True)
            atomic_json(history/(attempt_key(prior,process_record)+".json"),prior)
        command=[sys.executable,str(ROOT/"experiment_worker.py"),"--scenario",str(Path(job["scenario"]).resolve()),
                 "--output-dir",str(directory),"--deadline",args.deadline,
                 "--route-mode",job.get("route_mode","joint"),"--charging-mode",job.get("charging_mode","joint"),
                 "--feature-variant",job.get("feature_variant","full")]
        command.extend(["--source-snapshot",str(source_bundle)])
        if job.get("model"):
            command.extend(["--model",str(Path(job["model"]).resolve())])
        if job.get("capture_features",False):
            command.append("--capture-features")
        if job.get("horizon") is not None:
            command.extend(["--horizon",str(job["horizon"])])
        flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
        with (directory/"console.log").open("a",encoding="utf-8") as console:
            environment=os.environ.copy()
            environment.update(PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                               OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
            worker=subprocess.Popen(command,cwd=ROOT,stdout=console,stderr=subprocess.STDOUT,
                                    env=environment,creationflags=flags)
            created=psutil.Process(worker.pid).create_time()
            guard=subprocess.Popen([sys.executable,str(ROOT/"deadline_guard.py"),"--pid",str(worker.pid),
                "--created",str(created),"--deadline",args.deadline,"--record",str(directory/"deadline_pause.json")],
                cwd=ROOT,stdout=console,stderr=subprocess.STDOUT,creationflags=flags)
            atomic_json(directory/"process.json",{"worker_pid":worker.pid,"guard_pid":guard.pid,
                        "created":created,"deadline":args.deadline,"command":command})
            print(json.dumps({"job":job["job_id"],"state":"started","pid":worker.pid}),flush=True)
            while worker.poll() is None:
                if seconds_remaining(args.deadline)<=0:
                    from deadline_guard import stop_tree
                    stop_tree(worker.pid,created)
                    break
                time.sleep(.25)
            worker.wait()
            try: guard.wait(timeout=3)
            except subprocess.TimeoutExpired: pass
        if status_path.exists():
            status=json.loads(status_path.read_text(encoding="utf-8"))
        else:
            status={"state":"failed","error":"worker exited without a status artifact"}
        if status["state"]=="running":
            status.update(state="paused" if seconds_remaining(args.deadline)<=0 else "failed",
                          error="worker interrupted before completing the current solve",
                          finished_at=utc_now().isoformat(),returncode=worker.returncode)
            atomic_json(status_path,status)
        print(json.dumps({"job":job["job_id"],"state":status["state"],
                          "error":status.get("error"),"metrics":status.get("metrics")},ensure_ascii=False),flush=True)
        return status

    try:
        check_deadline(args.deadline)
        # This independent guard also covers regression/model construction in
        # the controller, so a blocked library call cannot overrun the deadline.
        flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
        created=psutil.Process(os.getpid()).create_time()
        with (output_root/"controller_guard.log").open("a",encoding="utf-8") as guard_log:
            subprocess.Popen([sys.executable,str(ROOT/"deadline_guard.py"),"--pid",str(os.getpid()),
                "--created",str(created),"--deadline",args.deadline,
                "--record",str(output_root/"controller_deadline_pause.json")],
                cwd=ROOT,stdout=guard_log,stderr=subprocess.STDOUT,creationflags=flags)
        config=json.loads(args.config.read_text(encoding="utf-8-sig"))
        source_hashes={}
        source_bytes={}
        for path in sorted(list((ROOT/"src").glob("*.py"))+list(ROOT.glob("*.py"))):
            relative=str(path.relative_to(ROOT))
            source_bytes[relative]=path.read_bytes()
            source_hashes[relative]=__import__("hashlib").sha256(source_bytes[relative]).hexdigest()
        source_bundle=output_root/"code_snapshots"/fingerprint(source_hashes)
        source_bundle.mkdir(parents=True,exist_ok=True)
        for relative,contents in source_bytes.items():
            target=source_bundle/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=contents:
                raise ValueError("immutable source snapshot mismatch")
            if not target.exists():
                target.write_bytes(contents)
        atomic_json(source_bundle/"manifest.json",source_hashes)
        environment_path=output_root/"execution_environment.json"
        if environment_path.exists():
            previous=json.loads(environment_path.read_text(encoding="utf-8"))
            history=output_root/"execution_history"
            history.mkdir(exist_ok=True)
            atomic_json(history/f"version_{len(list(history.glob('*.json')))+1:03d}.json",previous)
        import importlib.metadata
        atomic_json(environment_path,{
            "python":sys.version,"executable":sys.executable,"cpu_logical":psutil.cpu_count(),
            "cpu_physical":psutil.cpu_count(logical=False),"memory_bytes":psutil.virtual_memory().total,
            "platform":__import__("platform").platform(),"source_hashes":source_hashes,
            "source_snapshot":str(source_bundle),
            "package_versions":{name:importlib.metadata.version(name) for name in ["numpy","scipy","coptpy","torch"]},
            "solver_threads_per_worker":config["solver"]["threads"],
            "pilot_process_concurrency":args.pilot_workers,"deadline":args.deadline})
        pilot_jobs=[]
        for seed in [101,102,103]:
            scenario_path=output_root/"scenarios"/f"pilot_seed_{seed}.json"
            params=BusinessParameters.from_dict(config)
            if not scenario_path.exists():
                save_scenario(generate_synthetic_scenario(params,seed),scenario_path)
            envelope=json.loads(scenario_path.read_text(encoding="utf-8"))
            if envelope["seed"]!=seed or fingerprint(envelope["params"])!=fingerprint(params.to_dict()):
                raise ValueError("saved pilot inputs differ from the requested configuration or seed")
            pilot_jobs.append({"job_id":f"pilot_seed_{seed}","scenario":scenario_path,
                "output_dir":output_root/"pilot"/f"seed_{seed}","capture_features":False,
                "route_mode":"joint","charging_mode":"joint"})
        pilot_status=[]
        if args.stage in {"all","pilot"}:
            with ThreadPoolExecutor(max_workers=max(1,min(3,args.pilot_workers))) as pool:
                futures={pool.submit(run_job,job):job for job in pilot_jobs}
                for future in as_completed(futures):
                    status=future.result()
                    pilot_status.append({"job_id":futures[future]["job_id"],**status})
            atomic_json(output_root/"pilot_status.json",pilot_status)
        else:
            for job in pilot_jobs:
                status_file=Path(job["output_dir"])/"worker_status.json"
                if not status_file.exists():
                    raise ValueError("formal experiments require all three completed pilot days")
                existing=json.loads(status_file.read_text(encoding="utf-8"))
                if existing["state"]!="complete":
                    raise ValueError("formal experiments require all three completed pilot days")
                # Cached reuse still passes the same scenario/model/job identity
                # and complete-artifact checks as a newly requested pilot.
                pilot_status.append(run_job(job))
        if len(pilot_status)!=3 or any(s["state"]!="complete" for s in pilot_status):
            controller.update(state="paused" if any(s["state"]=="paused" for s in pilot_status) else "pilot_failed",
                              reason="formal experiments not started because pilot acceptance is incomplete")
            atomic_json(output_root/"controller_status.json",controller)
            return 75 if controller["state"]=="paused" else 1
        if args.stage!="pilot":
            from src.experiment_suite import run_formal_and_paper
            suite_result=run_formal_and_paper(args.config,output_root,args.deadline,run_job,
                [s["metrics"] for s in pilot_status],
                stage="core" if args.stage=="formal" else args.stage)
            controller["suite_result"]=suite_result
            if suite_result["status"]!="complete":
                controller.update(state=suite_result["status"],finished_at=utc_now().isoformat())
                atomic_json(output_root/"controller_status.json",controller)
                return 75 if controller["state"]=="paused" else 1
        controller.update(state="complete",finished_at=utc_now().isoformat())
        atomic_json(output_root/"controller_status.json",controller)
        return 0
    except Exception as exc:
        controller.update(state="paused" if isinstance(exc,ExperimentPaused) else "failed",
            finished_at=utc_now().isoformat(),error_type=type(exc).__name__,error=str(exc),
            traceback=__import__("traceback").format_exc())
        atomic_json(output_root/"controller_status.json",controller)
        print(json.dumps(controller,ensure_ascii=False),flush=True)
        return 75 if isinstance(exc,ExperimentPaused) else 1
    finally:
        try:
            from src.experiment_reporting import export_experiment_report
            export_experiment_report(output_root)
            from src.experiment_plots import export_experiment_plots
            export_experiment_plots(output_root)
        except Exception as report_error:
            atomic_json(output_root/"report_export_error.json",
                        {"error":str(report_error),"updated_at":utc_now().isoformat()})


if __name__=="__main__":
    raise SystemExit(main())
