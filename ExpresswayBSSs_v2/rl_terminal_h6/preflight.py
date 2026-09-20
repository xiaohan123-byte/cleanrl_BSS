"""Acceptance checks and ONE diagnostic real-size training-day MPC window."""
import bootstrap
import argparse
import copy
import platform
import sys
import time
import unittest
from pathlib import Path
import torch
from protocol import atomic, read, digest, file_hash, source_hashes, scenario_for, now
from test_rl import network_model
from state_encoding import StateSpec
from src.domain import initial_state
from src.dayahead_plan import generate_dayahead_plan
from src.execution import advance_to_boundary
from src.forecast import build_forecast
from src.request_builder import build_window
from src.mpc_model import solve_mpc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--skip-window",action="store_true")
    parser.add_argument("--failure-penalty",required=True,type=float,choices=[100.,200.])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    modules=["test_rl", "tests.test_real_data_experiment", "tests.test_terminal_execution",
             "tests.test_mpc_model", "tests.test_path_publication", "tests.test_accounting",
             "tests.test_result_statistics", "tests.test_rolling_runner", "tests.test_terminal_consistency_audit",
             "tests.test_atomic_io_recovery"]
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
    with (args.output/"tests.log").open("w",encoding="utf-8") as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    report={"time":now(),"failure_penalty":args.failure_penalty,"tests_run":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),
            "source_hash":digest(source_hashes(bootstrap.ROOT)),"python":platform.python_version(),
            "torch":torch.__version__,"cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0),
            "tests_log_hash":file_hash(args.output/"tests.log")}
    atomic(args.output/"preflight.json",report)
    if not result.wasSuccessful():
        raise RuntimeError("acceptance tests failed; inspect tests.log")
    print(f"Passed {result.testsRun} checks",flush=True)
    if not args.skip_window:
        started=time.perf_counter()
        params,scene,network=scenario_for(args.dataset,43,failure_penalty=args.failure_penalty)
        plans=generate_dayahead_plan(params,network,scene.initial_reservations())
        state=initial_state(params,scene.initial_reservations(),plans)
        state=advance_to_boundary(params,state,[],scenario=scene).state
        spec=StateSpec(params)
        # Untrained diagnostic model. It is never reused for fitting, validation or testing.
        model=network_model(params,[spec.raw_state(state)])
        forecast=build_forecast(params,scene.observation_at(0.),0,6)
        window=build_window(params,state,network,forecast,horizon=6)
        solution=solve_mpc(params,window,terminal_model=model,feature_spec=spec,
                           diagnostic_dir=args.output/"window_diagnostics")
        atomic(args.output/"diagnostic_window.json",solution.to_dict())
        report["real_size_window"]={"day":43,"period":0,"H":6,"purpose":"interface diagnostic only; untrained discarded network",
            "status":solution.status,"gap":solution.mip_gap,"seconds":time.perf_counter()-started,
            "diagnostics":solution.diagnostics,"parameter_count":model.parameter_count,"dimension":spec.dimension}
        atomic(args.output/"preflight.json",report)
        print(report["real_size_window"],flush=True)


if __name__=="__main__":
    main()
