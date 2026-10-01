"""Pilot-only frozen-zero-policy value-fit and single-window MILP diagnostics.

Separate stages: prepare (three complete, reconciled days), fit (ridge / fixed
200-epoch ReLU), embed (fixed journal windows, normal 60-second solver limit).
There is no validation/test split, policy-improvement round, model selection,
formal training budget, or full learned-policy operating-profit result here.
Pilot model files deliberately wrap the actual weights, so the formal model
loader cannot accidentally consume them. All source pilot files are read-only.
The CLI starts an independent hard-deadline guard before any stage begins.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import psutil

from diagnose_terminal_window import ROOT, prepare_window, run_comparison, _start_guard
from src.accounting import COMPONENTS, event_components
from src.candidate_network import validate_candidate_network
from src.domain import RollingState
from src.experiment_control import (DEFAULT_DEADLINE, ExperimentPaused, atomic_json,
    check_deadline, fingerprint, utc_now)
from src.forecast import Forecast
from src.parameters import BusinessParameters
from src.request_builder import build_window
from src.terminal_features import FeatureSpec
from src.terminal_value import TerminalValueModel, terminal_configuration_fingerprint
from src.value_training import fit_value_model, monte_carlo_samples

SEEDS = (101, 102, 103)
PERIODS = 288
PURPOSE = ("Pilot-only feasibility diagnostic using three complete zero-terminal-policy days. "
           "MC labels evaluate the frozen zero-terminal policy. Training error is in-sample, "
           "not generalization or policy-improvement evidence. No formal validation, model "
           "selection, testing, seed allocation, or training-budget decision is performed. "
           "Single-window embeddings use the normal 60-second limit, not full learned-policy runs.")
REQUIRED_CHECKS = ("ledger_unique", "financial_components_reconciled",
                   "prices_and_energy_reconciled", "inventory_reconciled",
                   "station_power_limits_satisfied")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _memory():
    memory = psutil.Process().memory_info()
    return {"process_rss_bytes": memory.rss,
            "process_peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss),
            "available_system_memory_bytes": psutil.virtual_memory().available,
            "peak_scope": "cumulative peak for this stage process; includes imports"}


def _close(actual, expected, description):
    if not np.isfinite(float(actual)) or not np.isfinite(float(expected)) or not np.isclose(
            actual, expected, rtol=1e-8, atol=1e-6):
        raise ValueError(f"{description}: {actual} does not reconcile with {expected}")


def _output_path(experiment_root, output_root=None):
    root = Path(experiment_root).resolve()
    output = (Path(output_root) if output_root else root / "pilot_only_value_diagnostics").resolve()
    if not output.is_relative_to(root):
        raise ValueError("pilot diagnostics must remain in their experiment root")
    relative = output.relative_to(root)
    if not relative.parts or not relative.parts[0].startswith("pilot_only_"):
        raise ValueError("use a separate pilot_only_* directory; never pilot or formal experiment paths")
    return output


def verify_complete_pilots(experiment_root):
    """Require all three independently reconciled full days before any data use."""
    root = Path(experiment_root).resolve()
    pilots = []
    common_parameters = None
    for seed in SEEDS:
        path = root / "pilot" / f"seed_{seed}"
        worker = _json(path / "worker_status.json")
        status = _json(path / "journal/status.json")
        if worker.get("state") != "complete" or status.get("state") != "complete":
            raise ValueError(f"seed {seed}: partial or failed pilot cannot supply training labels")
        metrics = _json(path / "metrics.json")
        statistics = _json(path / "statistics.json")
        header = _json(path / "journal/initial.json")
        identity = header["identity"]
        if status.get("completed_periods") != PERIODS or metrics.get("completed_periods") != PERIODS:
            raise ValueError(f"seed {seed}: requires exactly 288 completed periods")
        if (worker.get("model") is not None or identity.get("terminal_model") is not None
                or worker.get("route_mode") != "joint" or worker.get("charging_mode") != "joint"
                or identity.get("route_mode") != "joint" or identity.get("charging_mode") != "joint"):
            raise ValueError(f"seed {seed}: source must be the joint zero-terminal policy")
        if header["fingerprint"] != fingerprint(identity) or status.get("fingerprint") != header["fingerprint"]:
            raise ValueError(f"seed {seed}: journal identity mismatch")
        if any(statistics.get("checks", {}).get(key) is not True for key in REQUIRED_CHECKS):
            raise ValueError(f"seed {seed}: independent trajectory reconciliation checks did not all pass")
        params = BusinessParameters.from_dict(identity["parameters"])
        if params.num_periods != PERIODS:
            raise ValueError(f"seed {seed}: not a full 288-period day")
        _close(params.interval_hours, 1 / 12, "five-minute interval")
        _close(metrics["operating_duration_hours"], 24., "operating duration")
        if metrics.get("terminal_settlement") != "none":
            raise ValueError("pilot must not include a terminal settlement")
        if common_parameters is None:
            common_parameters = params.to_dict()
        elif params.to_dict() != common_parameters:
            raise ValueError("all pilot days must use identical physical, forecasting and solver parameters")
        total = statistics["summary"]["total_reward"]
        _close(metrics["net_profit_yuan"], total, "metrics versus independent statistics")
        _close(status["summary"]["total_reward"], total, "journal status versus independent statistics")
        if "metrics" in worker:
            _close(worker["metrics"]["net_profit_yuan"], total, "worker metrics")
        period_stats = statistics["per_period"]
        if len(period_stats) != PERIODS or [r["period"] for r in period_stats] != list(range(PERIODS)):
            raise ValueError("independent statistics must cover all periods exactly once")
        _close(sum(r["total_reward"] for r in period_stats), total, "per-period statistics sum")
        network = _json(path / "network.json")
        validate_candidate_network(network, params)
        small_files = ("worker_status.json", "metrics.json", "statistics.json", "network.json",
                       "journal/initial.json", "journal/status.json")
        pilots.append({"seed": seed, "path": path, "params": params, "network": network,
            "statistics": statistics, "total_reward": float(total),
            "source": {"seed": seed, "pilot_dir": str(path), "journal_identity": header["fingerprint"],
                "source_snapshot": worker.get("source_snapshot"),
                "source_scenario": worker.get("scenario"), "net_profit_yuan": float(total),
                "small_file_sha256": {name: _sha(path / name) for name in small_files}}})
    return pilots


def prepare_dataset(experiment_root, output_root=None, *, deadline=DEFAULT_DEADLINE):
    check_deadline(deadline)
    output = _output_path(experiment_root, output_root)
    pilots = verify_complete_pilots(experiment_root)  # Do not create outputs for partial pilots.
    directory = output / "dataset"
    if directory.exists():
        raise FileExistsError("dataset directory is immutable; use a new pilot_only_* output root")
    directory.mkdir(parents=True)
    spec = FeatureSpec(pilots[0]["params"], "full")
    features = np.empty((len(SEEDS) * PERIODS, spec.dimension), dtype=np.float64)
    returns = np.empty(len(features), dtype=np.float64)
    rewards = np.empty(len(features), dtype=np.float64)
    rows = np.empty((len(features), 2), dtype=np.int64)
    started = time.perf_counter()
    for day, pilot in enumerate(pilots):
        params, network = pilot["params"], pilot["network"]
        start = day * PERIODS
        seen_events = set()
        components = dict.fromkeys(COMPONENTS, 0.)
        round_digest = hashlib.sha256()
        count = 0
        with (pilot["path"] / "journal/rounds.jsonl").open("rb") as stream:
            for raw in stream:  # Exactly one sequential pass; no full-result/journal materialization.
                check_deadline(deadline)
                if not raw.endswith(b"\n"):
                    raise ValueError("complete journal has a partial last record; source is left unchanged")
                round_digest.update(raw)
                row = json.loads(raw)
                if count >= PERIODS or row["period"] != count:
                    raise ValueError("journal periods are duplicated, missing, out of order or beyond L")
                state_record = deepcopy(row["state_before"])
                state_record.pop("ledger_event_count", None)
                state_record["ledger"] = []
                state = RollingState.from_dict(state_record)
                if state.period != count:
                    raise ValueError("state_before period differs from its journal row")
                if row["horizon"] != min(params.horizon, PERIODS - count):
                    raise ValueError("source row does not use the recorded clipped fixed H")
                window = build_window(params, state, network, Forecast(**deepcopy(row["forecast"])),
                                      horizon=row["horizon"])
                features[start + count] = spec.encode_window(window)
                reward = float(row["reward"])
                event_reward = 0.
                for event in row["events"]:
                    if event["event_id"] in seen_events or event["period"] != count:
                        raise ValueError("duplicate realised event or mismatched event period")
                    seen_events.add(event["event_id"])
                    event_values = event_components(params, event)
                    event_reward += event_values["reward_delta"]
                    for key in COMPONENTS:
                        components[key] += event_values[key]
                _close(reward, event_reward, "journal reward versus recomputed physical-event accounting")
                _close(reward, pilot["statistics"]["per_period"][count]["total_reward"], "per-period reward")
                rewards[start + count] = reward
                rows[start + count] = (pilot["seed"], count)
                count += 1
        if count != PERIODS:
            raise ValueError("complete pilot marker disagrees with incomplete journal")
        for key in COMPONENTS:
            _close(components[key], pilot["statistics"]["summary"][key], f"full-day {key}")
        # The production MC helper keeps the identical discount and end-of-day semantics.
        day_rows = [{"state_features": features[start + n], "reward": rewards[start + n]}
                    for n in range(PERIODS)]
        day_x, day_returns = monte_carlo_samples({"parameter_snapshot": params.to_dict(),
            "rounds": day_rows, "completed": True, "summary": {"total_reward": pilot["total_reward"]}})
        if not np.array_equal(day_x, features[start:start + PERIODS]):
            raise RuntimeError("MC helper changed fixed features")
        returns[start:start + PERIODS] = day_returns
        pilot["source"]["journal_sha256"] = round_digest.hexdigest()
        pilot["source"]["verified_rows"] = count
        # Metadata must not change during our read-only extraction.
        for name, expected in pilot["source"]["small_file_sha256"].items():
            if _sha(pilot["path"] / name) != expected:
                raise ValueError("completed pilot metadata changed during feature extraction")
    for name, values in (("features", features), ("returns_yuan", returns), ("rewards_yuan", rewards), ("rows", rows)):
        np.save(directory / f"{name}.npy", values, allow_pickle=False)
    atomic_json(directory / "feature_spec.json", spec.to_dict())
    atomic_json(directory / "parameter_snapshot.json", spec.params.to_dict())
    files = ["features.npy", "returns_yuan.npy", "rewards_yuan.npy", "rows.npy",
             "feature_spec.json", "parameter_snapshot.json"]
    manifest = {"schema_version": 1, "scope": "pilot_only", "purpose": PURPOSE,
        "state": "complete", "generating_policy": "zero_terminal", "discount_factor": 1.,
        "terminal_boundary_value": 0., "uncompleted_booking_settlement_at_L": False,
        "complete_days": len(SEEDS), "seeds": list(SEEDS), "samples": len(features),
        "feature_dimension": spec.dimension, "dtype": "float64",
        "feature_semantics": "state_before; same FeatureSpec/build_window and recorded forecast; no future truth",
        "configuration_fingerprint": terminal_configuration_fingerprint(spec.params),
        "parameters_fingerprint": fingerprint(spec.params.to_dict()),
        "sources": [pilot["source"] for pilot in pilots],
        "artifact_sha256": {name: _sha(directory / name) for name in files},
        "source_code_sha256": {name: _sha(ROOT / name) for name in (
            "diagnose_pilot_value_fit.py", "src/terminal_features.py", "src/value_training.py",
            "src/request_builder.py", "src/forecast.py", "src/terminal_value.py")},
        "created_at": utc_now().isoformat(), "wall_seconds": time.perf_counter() - started,
        "memory": _memory()}
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def _load_dataset(experiment_root, output_root=None):
    output = _output_path(experiment_root, output_root)
    directory = output / "dataset"
    manifest = _json(directory / "manifest.json")
    if (manifest.get("state") != "complete" or manifest.get("scope") != "pilot_only"
            or manifest.get("samples") != len(SEEDS) * PERIODS or manifest.get("seeds") != list(SEEDS)):
        raise ValueError("requires complete, explicitly pilot-only 864-row data")
    pilots = verify_complete_pilots(experiment_root)
    for pilot, source in zip(pilots, manifest["sources"]):
        if (pilot["source"]["small_file_sha256"] != source["small_file_sha256"]
                or pilot["source"]["journal_identity"] != source["journal_identity"]):
            raise ValueError("pilot metadata differs from the prepared dataset")
    for name, digest in manifest["artifact_sha256"].items():
        if _sha(directory / name) != digest:
            raise ValueError(f"prepared artifact changed: {name}")
    for name in ("src/terminal_features.py", "src/request_builder.py", "src/forecast.py", "src/terminal_value.py"):
        if _sha(ROOT / name) != manifest["source_code_sha256"][name]:
            raise ValueError("feature/model implementation changed since dataset preparation")
    params = BusinessParameters.from_dict(_json(directory / "parameter_snapshot.json"))
    spec = FeatureSpec(params, "full")
    if spec.to_dict() != _json(directory / "feature_spec.json"):
        raise ValueError("current FeatureSpec differs from the saved fixed feature specification")
    if terminal_configuration_fingerprint(params) != manifest["configuration_fingerprint"]:
        raise ValueError("dataset configuration fingerprint mismatch")
    return output, manifest, spec


def fit_models(experiment_root, output_root=None, *, kinds=("linear", "relu"), deadline=DEFAULT_DEADLINE):
    check_deadline(deadline)
    output, manifest, spec = _load_dataset(experiment_root, output_root)
    directory = output / "models"
    directory.mkdir(parents=True, exist_ok=True)
    for kind in kinds:
        if kind not in {"linear", "relu"}:
            raise ValueError("fit kind must be linear or relu")
        if (directory / f"pilot_only_{kind}.json").exists():
            raise FileExistsError("pilot model is immutable; use a new pilot_only_* output root")
    x = np.load(output / "dataset/features.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(output / "dataset/returns_yuan.npy", mmap_mode="r", allow_pickle=False)
    try:
        results = []
        for kind in kinds:
            check_deadline(deadline)
            before = _memory()
            model, diagnostics = fit_value_model(x, y, spec, kind=kind, previous=None, seed=20260914,
                                                  epochs=200, deadline=deadline)
            diagnostics["checkpoint_rule"] = "fixed final fit only; no validation, early stopping or model selection"
            diagnostics["error_scope"] = "in-sample error on the three zero-policy pilot days only"
            result = {"schema_version": 1, "scope": "pilot_only", "purpose": PURPOSE,
                "generating_policy": "zero_terminal", "is_formal_model": False,
                "policy_improvement_performed": False, "validation_or_test_performed": False,
                "dataset_manifest_sha256": _sha(output / "dataset/manifest.json"),
                "configuration_fingerprint": manifest["configuration_fingerprint"],
                "diagnostics": diagnostics, "memory_before": before, "memory_after": _memory(),
                "saved_at": utc_now().isoformat(), "model": model.to_dict()}
            # A wrapper deliberately prevents TerminalValueModel.load() from accepting this as formal weights.
            atomic_json(directory / f"pilot_only_{kind}.json", result)
            results.append({key: value for key, value in result.items() if key != "model"})
        return results
    finally:
        x._mmap.close()
        y._mmap.close()


def embed_models(experiment_root, output_root=None, *, seed=101, periods=(72,),
                 kinds=("linear", "relu"), deadline=DEFAULT_DEADLINE):
    check_deadline(deadline, reserve_seconds=60.)
    output, manifest, spec = _load_dataset(experiment_root, output_root)
    if seed not in SEEDS:
        raise ValueError("embedding source must be one of the fixed pilot seeds")
    models = {}
    for kind in kinds:
        if kind == "zero":
            continue
        if kind not in {"linear", "relu"}:
            raise ValueError("embedding kind must be zero, linear or relu")
        envelope = _json(output / "models" / f"pilot_only_{kind}.json")
        if (envelope.get("scope") != "pilot_only" or envelope.get("generating_policy") != "zero_terminal"
                or envelope.get("is_formal_model") is not False
                or envelope.get("dataset_manifest_sha256") != _sha(output / "dataset/manifest.json")):
            raise ValueError("model must be the pilot-only fit of this exact dataset")
        models[kind] = TerminalValueModel.from_dict(envelope["model"])
        if models[kind].feature_names != spec.names:
            raise ValueError("fitted model features differ from the prepared dataset")
    results = []
    for period in periods:
        check_deadline(deadline, reserve_seconds=60.)
        journal = Path(experiment_root) / "pilot" / f"seed_{seed}" / "journal"
        params, window, metadata = prepare_window(journal, int(period), solve_seconds=60.)
        if terminal_configuration_fingerprint(params) != manifest["configuration_fingerprint"]:
            raise ValueError("embedding source differs from the fitted physical configuration")
        metadata.update(purpose=PURPOSE, trained_model=True, scope="pilot_only",
                        generating_policy="zero_terminal", solver_limit_purpose="normal 60-second limit")
        directory = output / "embeddings" / f"seed_{seed}_period_{period:03d}"
        if directory.exists():
            raise FileExistsError("embedding directory is immutable; use a new output root or window")
        directory.mkdir(parents=True)
        atomic_json(directory / "input_manifest.json", metadata)
        result = run_comparison(params, window, directory, deadline=deadline, methods=tuple(kinds),
                                metadata=metadata, supplied_models=models, purpose=PURPOSE)
        results.append(result)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "embed"))
    parser.add_argument("--experiment-root", type=Path, default=ROOT / "outputs/experiment_20260914_od26")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--deadline", default=DEFAULT_DEADLINE)
    parser.add_argument("--kinds", nargs="+", choices=("zero", "linear", "relu"), default=["linear", "relu"])
    parser.add_argument("--seed", type=int, choices=SEEDS, default=101)
    parser.add_argument("--periods", nargs="+", type=int, default=[72])
    args = parser.parse_args(argv)
    check_deadline(args.deadline, reserve_seconds=60. if args.stage == "embed" else 0.)
    output = _output_path(args.experiment_root, args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    _start_guard(output, args.deadline)
    status = {"stage": args.stage, "scope": "pilot_only", "purpose": PURPOSE,
              "state": "running", "started_at": utc_now().isoformat(), "deadline": args.deadline}
    status_path = output / f"{args.stage}_status.json"
    atomic_json(status_path, status)
    try:
        if args.stage == "prepare":
            result = prepare_dataset(args.experiment_root, output, deadline=args.deadline)
        elif args.stage == "fit":
            result = fit_models(args.experiment_root, output, kinds=tuple(args.kinds), deadline=args.deadline)
        else:
            result = embed_models(args.experiment_root, output, seed=args.seed, periods=tuple(args.periods),
                                  kinds=tuple(args.kinds), deadline=args.deadline)
        status.update(state="complete", finished_at=utc_now().isoformat(), memory=_memory())
        atomic_json(status_path, status)
        print(json.dumps({"stage": args.stage, "state": "complete", "scope": "pilot_only",
                          "output_root": str(output)}, ensure_ascii=False), flush=True)
        return 0
    except BaseException as error:
        status.update(state="paused" if isinstance(error, ExperimentPaused) else "failed",
            finished_at=utc_now().isoformat(), error=str(error), error_type=type(error).__name__,
            traceback=traceback.format_exc(), memory=_memory())
        atomic_json(status_path, status)
        if isinstance(error, ExperimentPaused):
            return 75
        raise


if __name__ == "__main__":
    raise SystemExit(main())
