"""Independent hard-deadline guard for exactly one experiment process tree."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import psutil
from src.experiment_control import atomic_json, parse_deadline, seconds_remaining, utc_now


def stop_tree(pid: int, created: float) -> bool:
    try:
        parent=psutil.Process(pid)
        if abs(parent.create_time()-created) > .01:
            return False
        processes=[p for p in parent.children(recursive=True)+[parent] if p.pid != __import__("os").getpid()]
    except psutil.NoSuchProcess:
        return False
    for process in reversed(processes):
        try: process.terminate()
        except psutil.NoSuchProcess: pass
    _,alive=psutil.wait_procs(processes,timeout=2)
    for process in alive:
        try: process.kill()
        except psutil.NoSuchProcess: pass
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid",type=int,required=True)
    p.add_argument("--created",type=float,required=True)
    p.add_argument("--deadline",required=True)
    p.add_argument("--record",type=Path,required=True)
    args=p.parse_args(argv)
    deadline=parse_deadline(args.deadline)
    while True:
        try:
            proc=psutil.Process(args.pid)
            if abs(proc.create_time()-args.created)>.01 or not proc.is_running():
                return 0
        except psutil.NoSuchProcess:
            return 0
        remaining=seconds_remaining(deadline)
        if remaining<=0:
            stop_started=utc_now().isoformat()
            observed_tree=[]
            try:
                parent=psutil.Process(args.pid)
                if abs(parent.create_time()-args.created)<=.01:
                    for target in parent.children(recursive=True)+[parent]:
                        if target.pid==__import__("os").getpid():
                            continue
                        try:
                            observed_tree.append({"pid":target.pid,"created":target.create_time()})
                        except psutil.NoSuchProcess:
                            pass
            except psutil.NoSuchProcess:
                pass
            killed=stop_tree(args.pid,args.created)
            stopped_at=utc_now().isoformat()
            atomic_json(args.record,{"state":"paused","reason":"user_wall_clock_deadline",
                "deadline":args.deadline,"observed_at":stopped_at,
                "stop_started_at":stop_started,"stop_finished_at":stopped_at,
                "worker_pid":args.pid,"target_pid":args.pid,"target_created":args.created,
                "terminated_processes":observed_tree if killed else [],
                "process_tree_terminated":killed,
                "recovery":"completed rounds remain in journal; interrupted solve has no MC label"})
            return 0
        time.sleep(min(.25,remaining))


if __name__=="__main__":
    raise SystemExit(main())
