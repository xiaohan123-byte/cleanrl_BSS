"""Untrained, single-window terminal MILP diagnostics; never operating results.

Default invocation prepares metadata only. Pass --execute after arranging enough
free memory. All methods use one restored completed journal round and the same
10-second diagnostic solver limit. Pilot inputs and journal files are read only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil

from src.candidate_network import validate_candidate_network
from src.domain import RollingState
from src.experiment_control import (DEFAULT_DEADLINE, ExperimentPaused, atomic_json,
    check_deadline, fingerprint, utc_now)
from src.forecast import Forecast
from src.mpc_model import solve_mpc
from src.parameters import BusinessParameters
from src.request_builder import build_window
from src.terminal_features import FeatureSpec
from src.terminal_value import TerminalValueModel, terminal_configuration_fingerprint

ROOT = Path(__file__).resolve().parent
DISCLAIMER = ("Untrained deterministic coefficient diagnostic only. This is not a trained policy, "
              "not a complete trajectory or MC label, and not a comparison of operating profit. "
              "The 10-second limit does not determine the final trained-model compute budget.")


def _read_completed_round(path, period):
    with Path(path).open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                break  # A concurrently appended incomplete record is never read or repaired.
            row = json.loads(raw)
            if row["period"] == period:
                return row, hashlib.sha256(raw).hexdigest()
            if row["period"] > period:
                break
    raise ValueError(f"completed journal round {period} is unavailable")


def prepare_window(journal_dir, period, network_path=None, solve_seconds=10.0):
    """Restore exactly one completed pre-decision state without its irrelevant ledger."""
    journal_dir = Path(journal_dir).resolve()
    header_path = journal_dir / "initial.json"
    raw_header = header_path.read_bytes()
    header = json.loads(raw_header)
    record, row_digest = _read_completed_round(journal_dir / "rounds.jsonl", period)
    original_parameters = deepcopy(header["identity"]["parameters"])
    params = BusinessParameters.from_dict(original_parameters)
    params.solver.time_limit_sec = float(solve_seconds)
    params.solver.threads = 1
    # Keep the recorded H and every physical/price/demand value unchanged.
    state_record = deepcopy(record["state_before"])
    ledger_count = state_record.pop("ledger_event_count", None)
    state_record["ledger"] = []
    state = RollingState.from_dict(state_record)
    if state.period != period:
        raise ValueError("restored state's period differs from selected completed round")
    forecast = Forecast(**deepcopy(record["forecast"]))
    network_path = Path(network_path).resolve() if network_path else journal_dir.parent / "network.json"
    raw_network = network_path.read_bytes()
    network = json.loads(raw_network)
    validate_candidate_network(network, params)
    window = build_window(params, state, network, forecast, horizon=record["horizon"])
    metadata = {
        "purpose": DISCLAIMER, "trained_model": False, "complete_operating_return": False,
        "journal_dir": str(journal_dir), "network_path": str(network_path), "period": period,
        "original_journal_fingerprint": header["fingerprint"],
        "header_sha256": hashlib.sha256(raw_header).hexdigest(), "round_sha256": row_digest,
        "network_sha256": hashlib.sha256(raw_network).hexdigest(),
        "original_parameter_snapshot": original_parameters,
        "diagnostic_parameter_snapshot": params.to_dict(),
        "original_ledger_event_count": ledger_count, "ledger_restored_for_diagnostic": False,
        "forecast_snapshot": deepcopy(record["forecast"]), "network_snapshot": deepcopy(network),
        "state_fingerprint": fingerprint(state.to_dict()), "window_fingerprint": fingerprint(asdict(window)),
        "horizon": window.horizon, "station_count": params.station.num_stations,
        "request_count": len(window.requests), "user_network_count": len(window.networks),
        "source_round_solver_status": record.get("solution", {}).get("status"),
        "diagnostic_solver_seconds": params.solver.time_limit_sec,
        "source_code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in ("src/mpc_model.py", "src/terminal_features.py",
                                            "src/terminal_value.py", "src/forecast.py", "src/request_builder.py")},
    }
    return params, window, metadata


def make_untrained_model(spec, kind):
    """Fixed modest diagnostic coefficients; no data fitting or seed selection."""
    if kind == "zero":
        return None
    if kind not in {"linear", "relu"}:
        raise ValueError("diagnostic method must be zero, linear or relu")
    d = spec.dimension
    linear = np.zeros(d)
    inventory = []
    request_count_indices = []
    for index, name in enumerate(spec.names):
        if name.startswith("battery:") and name.endswith(":mean_soc"):
            linear[index] = 0.025  # 25 yuan per unit station-average SOC after output scaling.
            inventory.append(index)
        elif name.startswith("chain:") and ":time:" in name and name.endswith(":count"):
            linear[index] = 0.030  # 0.30 yuan per pending reservation.
            request_count_indices.append(index)
        elif name.startswith("chain:") and name.endswith(":next_return_soc"):
            linear[index] = -0.005
        elif name.startswith("chain:") and name.endswith(":remaining_hours"):
            linear[index] = 0.005
    kwargs = dict(kind=kind, feature_names=list(spec.names), variant=spec.variant,
                  linear_weights=linear, bias=0.0, output_scale=1000.0,
                  configuration_fingerprint=terminal_configuration_fingerprint(spec.params))
    if kind == "relu":
        hidden = np.zeros((16, d))
        hidden_bias = np.zeros(16)
        output = np.zeros(16)
        for unit in range(16):
            hidden[unit, inventory[unit % len(inventory)]] = 1.0
            hidden[unit, inventory[(unit + 1) % len(inventory)]] += -0.2
            if request_count_indices:
                hidden[unit, request_count_indices[unit % len(request_count_indices)]] = 0.2
            hidden_bias[unit] = -(0.25 + 0.04 * (unit % 8))
            output[unit] = 0.003 if unit % 2 == 0 else -0.003
        kwargs.update(hidden_weights=hidden, hidden_bias=hidden_bias, output_weights=output)
    return TerminalValueModel(**kwargs)


def run_comparison(params, window, output_dir, *, deadline=DEFAULT_DEADLINE,
                   methods=("zero", "linear", "relu"), metadata=None, supplied_models=None,
                   purpose=DISCLAIMER):
    """Run isolated methods sequentially, preserving every success and failure."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "diagnostic_results.json"
    if result_path.exists():
        raise FileExistsError("use a new output directory; existing diagnostic results are immutable")
    spec = FeatureSpec(params, "full")
    state_features = spec.encode_window(window)
    initial_window_digest = fingerprint(asdict(window))
    result = {"purpose": purpose, "trained_model": supplied_models is not None, "complete_operating_return": False,
              "state": "running", "started_at": utc_now().isoformat(), "deadline": deadline,
              "period": window.ell, "horizon": window.horizon,
              "feature_dimension": spec.dimension, "chain_templates": len(spec.groups),
              "available_memory_bytes_at_start": psutil.virtual_memory().available,
              "methods": [], "source": metadata or {}}
    atomic_json(output_dir / "feature_spec.json", spec.to_dict())
    atomic_json(output_dir / "state_features.json", {"purpose": purpose,
                "period": window.ell, "features": state_features.tolist()})
    atomic_json(result_path, result)
    for method in methods:
        try:
            check_deadline(deadline, reserve_seconds=params.solver.time_limit_sec)
        except ExperimentPaused:
            result.update(state="paused", finished_at=utc_now().isoformat())
            atomic_json(result_path, result)
            raise
        # Each real diagnostic method rebuilds its own request/network objects
        # from a deep-cloned physical state and the same recorded forecast.
        method_params = BusinessParameters.from_dict(params.to_dict())
        if metadata and "forecast_snapshot" in metadata and "network_snapshot" in metadata:
            method_window = build_window(method_params, window.state.clone(),
                deepcopy(metadata["network_snapshot"]), Forecast(**deepcopy(metadata["forecast_snapshot"])),
                horizon=window.horizon)
        else:
            # This path supports only manually constructed small test fixtures.
            method_window = deepcopy(window)
        method_digest = fingerprint(asdict(method_window))
        if method_digest != initial_window_digest:
            raise ValueError("independent rebuilt method window differs from the recorded source window")
        model = (make_untrained_model(spec, method) if supplied_models is None
                 else None if method == "zero" else supplied_models[method])
        if model is not None and (model.kind != method or model.feature_names != spec.names):
            raise ValueError("supplied diagnostic model has incompatible kind or features")
        directory = output_dir / method
        directory.mkdir(exist_ok=True)
        if model is not None:
            if supplied_models is None:
                model.save(directory / "untrained_model.json")
            else:
                atomic_json(directory / "pilot_only_model.json", {"scope": "pilot_only",
                    "purpose": purpose, "generating_policy": "zero_terminal", "model": model.to_dict()})
        started = time.perf_counter()
        rss_before = psutil.Process().memory_info().rss
        entry = {"method": method, "trained_model": model is not None and supplied_models is not None, "solver_limit_seconds": params.solver.time_limit_sec,
                 "horizon": window.horizon, "feature_dimension": spec.dimension,
                 "model_parameter_count": 0 if model is None else int(model.parameter_count)}
        try:
            solution = solve_mpc(method_params, method_window, terminal_model=model,
                                 feature_spec=spec if model is not None else None,
                                 diagnostic_dir=directory / "solver_diagnostics")
            check_deadline(deadline)
            entry.update(feasible_incumbent=True, status=solution.status,
                mip_gap=solution.mip_gap, build_seconds=solution.build_seconds,
                solve_seconds=solution.solve_seconds, wall_seconds=solution.wall_seconds,
                measured_call_seconds=time.perf_counter() - started,
                terminal_value_yuan=solution.terminal_value,
                diagnostic_window_objective_yuan=solution.objective,
                diagnostics=solution.diagnostics)
            if model is not None and solution.terminal_features:
                numerical_value = model.predict(solution.terminal_features)
                entry["independent_value_yuan"] = numerical_value
                entry["network_mip_absolute_error"] = abs(numerical_value - solution.terminal_value)
            atomic_json(directory / "solution.json", solution.to_dict())
        except ExperimentPaused:
            result.update(state="paused", finished_at=utc_now().isoformat())
            atomic_json(result_path, result)
            raise
        except Exception as error:
            entry.update(feasible_incumbent=False,
                         status=getattr(error, "diagnostics", {}).get("status", "diagnostic_failed"),
                         error_type=type(error).__name__, error=str(error),
                         measured_call_seconds=time.perf_counter() - started,
                         diagnostics=getattr(error, "diagnostics", {}))
            atomic_json(directory / "error.json", dict(entry, traceback=traceback.format_exc()))
        memory = psutil.Process().memory_info()
        entry["process_rss_before_bytes"] = rss_before
        entry["process_rss_after_bytes"] = memory.rss
        entry["process_peak_working_set_bytes"] = getattr(memory, "peak_wset", memory.rss)
        entry["independently_rebuilt_window_fingerprint"] = method_digest
        entry["input_window_unchanged"] = (fingerprint(asdict(window)) == initial_window_digest
                                            and fingerprint(asdict(method_window)) == method_digest)
        if not entry["input_window_unchanged"]:
            raise RuntimeError("a diagnostic method mutated its shared input window")
        result["methods"].append(entry)
        atomic_json(result_path, result)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
    result.update(state="complete" if all(x["feasible_incumbent"] for x in result["methods"])
                  else "complete_with_diagnostic_failures", finished_at=utc_now().isoformat())
    atomic_json(result_path, result)
    return result


def _start_guard(output_dir, deadline):
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    created = psutil.Process(os.getpid()).create_time()
    command = [sys.executable, str(ROOT / "deadline_guard.py"), "--pid", str(os.getpid()),
               "--created", str(created), "--deadline", deadline,
               "--record", str(output_dir / "deadline_pause.json")]
    with (output_dir / "deadline_guard.log").open("a", encoding="utf-8") as log:
        guard = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
    atomic_json(output_dir / "process.json", {"pid": os.getpid(), "created": created,
                "guard_pid": guard.pid, "deadline": deadline, "command": command})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path,
        default=ROOT / "outputs/experiment_20260914_od26/pilot/seed_101/journal")
    parser.add_argument("--period", type=int, default=72)
    parser.add_argument("--network", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--solve-seconds", type=float, default=10.0)
    parser.add_argument("--deadline", default=DEFAULT_DEADLINE)
    parser.add_argument("--methods", nargs="+", choices=("zero", "linear", "relu"),
                        default=["zero", "linear", "relu"])
    parser.add_argument("--execute", action="store_true", help="actually solve; omit to prepare metadata only")
    args = parser.parse_args(argv)
    check_deadline(args.deadline, reserve_seconds=args.solve_seconds if args.execute else 0)
    if not 0 < args.solve_seconds <= 60:
        raise ValueError("diagnostic solve seconds must be in (0, 60]")
    output_dir = (args.output_dir or ROOT / f"outputs/experiment_20260914_od26/terminal_diagnostics/period_{args.period:03d}").resolve()
    if output_dir.is_relative_to(args.journal.resolve().parent):
        raise ValueError("diagnostic artifacts must be outside the read-only source pilot directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "diagnostic_results.json").exists():
        raise FileExistsError("existing diagnostic results are immutable; select a new output directory")
    if args.execute:
        _start_guard(output_dir, args.deadline)
    params, window, metadata = prepare_window(args.journal, args.period, args.network, args.solve_seconds)
    metadata.update(execute_requested=args.execute, prepared_at=utc_now().isoformat(), deadline=args.deadline)
    atomic_json(output_dir / "input_manifest.json", metadata)
    if not args.execute:
        spec = FeatureSpec(params, "full")
        atomic_json(output_dir / "feature_spec.json", spec.to_dict())
        print(json.dumps({"state": "prepared_only", "executed_solver": False,
                          "feature_dimension": spec.dimension, "chain_templates": len(spec.groups),
                          "request_count": len(window.requests), "output_dir": str(output_dir)}, ensure_ascii=False))
        return 0
    result = run_comparison(params, window, output_dir, deadline=args.deadline,
                            methods=tuple(args.methods), metadata=metadata)
    # Verify immutable source pieces. Concurrent appends elsewhere in the journal are allowed.
    _, round_digest = _read_completed_round(args.journal / "rounds.jsonl", args.period)
    unchanged = (round_digest == metadata["round_sha256"]
                 and hashlib.sha256((args.journal / "initial.json").read_bytes()).hexdigest() == metadata["header_sha256"]
                 and hashlib.sha256(Path(metadata["network_path"]).read_bytes()).hexdigest() == metadata["network_sha256"])
    atomic_json(output_dir / "source_integrity.json", {"source_files_unchanged": unchanged,
                "checked_at": utc_now().isoformat(), "source_journal": str(args.journal.resolve())})
    if not unchanged:
        raise RuntimeError("source diagnostic inputs changed during evaluation")
    return 0 if result["state"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentPaused as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(75)
