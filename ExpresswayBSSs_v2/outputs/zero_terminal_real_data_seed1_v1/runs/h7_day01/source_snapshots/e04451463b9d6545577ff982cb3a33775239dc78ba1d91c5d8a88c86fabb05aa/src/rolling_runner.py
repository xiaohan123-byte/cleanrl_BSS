"""Observation -> forecast -> joint MILP -> first-interval execution."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from time import perf_counter
from typing import Any

from src.accounting import summarize_ledger, append_events
from src.candidate_network import generate_candidate_network
from src.domain import initial_state, RollingState
from src.dayahead_plan import generate_dayahead_plan
from src.execution import advance_to_boundary, execute_step
from src.forecast import build_forecast
from src.mpc_model import solve_mpc
from src.request_builder import build_window
from src.parameters import BusinessParameters, execution_period_limit, prediction_horizon
from src.experiment_control import (RunJournal, ExperimentPaused, check_deadline,
                                    fingerprint, atomic_json)


def _snapshot(state) -> dict:
    result = state.to_dict()
    result["ledger_event_count"] = len(result.pop("ledger"))
    return result


def run_rolling_mpc(params: Any, scenario: Any, network: dict | None = None,
                    progress=None, *, terminal_model=None, feature_spec=None,
                    route_mode="joint", charging_mode="joint",
                    deadline=None, journal_dir=None) -> dict:
    """Run a full operating day, saving every executed interval when requested.

    A failed solve or deadline never returns a complete trajectory. Journals
    permit resuming unchanged runs from the last completed five-minute interval.
    The optional feature spec records actual decision-state features for MC
    training, including when the sampling policy has zero terminal value.
    """
    params.validate()
    scenario.validate()
    physical_snapshot = lambda values: {key: value for key, value in values.items()
                                        if key not in {"horizon", "solver"}}
    if physical_snapshot(BusinessParameters.from_dict(scenario.params).to_dict()) != physical_snapshot(params.to_dict()):
        raise ValueError("scenario parameter snapshot differs from run physical parameters")
    check_deadline(deadline)
    if network is None:
        network = generate_candidate_network(params)
    reservations = scenario.initial_reservations()
    plans = generate_dayahead_plan(params, network, reservations)
    state = initial_state(params, reservations, plans)
    initial = state.to_dict()
    rounds = []
    journal = None
    if journal_dir is not None:
        identity = {"parameters": params.to_dict(), "scenario_hash": fingerprint(scenario.to_dict()),
                    "terminal_model": terminal_model.to_dict() if terminal_model is not None else None,
                    "feature_names": list(feature_spec.names) if feature_spec is not None else None,
                    "route_mode": route_mode, "charging_mode": charging_mode}
        journal = RunJournal(journal_dir, identity)
        journal.initialize(initial, plans)
        initial, plans, rounds = journal.recover()
        if rounds:
            restored = deepcopy(rounds[-1]["state_after"])
            expected_count = restored.pop("ledger_event_count")
            restored["ledger"] = deepcopy(initial.get("ledger", []))
            # Round events contain physical events before accounting fields were
            # added. Recompute their validated financial components on recovery.
            from math import isclose
            for previous_round in rounds:
                recovered_reward = append_events(params, restored["ledger"], previous_round["events"])
                if not isclose(recovered_reward, previous_round["reward"], rel_tol=1e-8, abs_tol=1e-6):
                    raise ValueError("journal reward does not reconcile with recovered actual events")
            if len(restored["ledger"]) != expected_count:
                raise ValueError("journal event count differs from recovered state")
            state = RollingState.from_dict(restored)
        journal.status("running", completed_periods=len(rounds))
    experimental = bool(getattr(params, "terminal_experiment", False))
    drain = getattr(params, "finish_pending_after_demand", False)
    pending = lambda: bool(state.waiting) or any(u.status == "active" for u in state.users.values())
    for ell in range(len(rounds), execution_period_limit(params)):
        if ell >= params.num_periods and not pending():
            break
        try:
            # Leave the full allowed solver interval before the hard supervisor
            # deadline. The external supervisor also covers model construction.
            check_deadline(deadline, reserve_seconds=float(params.solver.time_limit_sec))
            if journal is not None and (journal.directory / "pause_requested.json").exists():
                raise ExperimentPaused("Paused at an execution boundary by a saved pause request.")
            round_start = perf_counter()
            now = ell * params.interval_hours
            observation = scenario.observation_at(now)
            boundary_arrivals = [item for item in observation.random_history
                                 if item["arrival_time"] == now]
            execution_options = {"scenario": scenario} if experimental else {}
            admission = advance_to_boundary(params, state, boundary_arrivals, **execution_options)
            state = admission.state
            before = _snapshot(state)
            horizon = prediction_horizon(params, ell, params.horizon)
            forecast = build_forecast(params, observation, ell, horizon)
            window = build_window(params, state, network, forecast, horizon=horizon)
            state_features = None
            if feature_spec is not None:
                state_features = feature_spec.encode_window(window)
                state_features = state_features.tolist() if hasattr(state_features, "tolist") else list(state_features)
            solve_options = {}
            if terminal_model is not None:
                solve_options.update(terminal_model=terminal_model, feature_spec=feature_spec)
            if route_mode != "joint":
                solve_options["route_mode"] = route_mode
            if charging_mode != "joint":
                solve_options["charging_mode"] = charging_mode
            if journal is not None:
                solve_options["diagnostic_dir"] = journal.directory / "diagnostics" / f"period_{ell:03d}"
                atomic_json(journal.directory / "solve_context.json",
                            {"period": ell, "state": before, "forecast": asdict(forecast),
                             "horizon": horizon, "completed_periods": len(rounds)})
            solve_start = perf_counter()
            solution = solve_mpc(params, window, **solve_options)
            model_wall_seconds = perf_counter() - solve_start
            check_deadline(deadline)
            execution = execute_step(params, state, solution, scenario.arrivals_between(
                now, (ell + 1) * params.interval_hours), **execution_options)
            state = execution.state
            record = {
                "period": ell, "time": now, "horizon": horizon,
                "path_update_allowed": ell % params.path_update_interval == 0,
                "state_before": before, "forecast": asdict(forecast),
                "solution": solution.to_dict(),
                "events": deepcopy(admission.events + execution.events),
                "reward": admission.reward + execution.reward,
                "state_after": _snapshot(state),
                "model_wall_seconds": model_wall_seconds,
                "round_wall_seconds": perf_counter() - round_start,
            }
            if state_features is not None:
                record["state_features"] = state_features
            if journal is not None:
                journal.append(record)
            rounds.append(record)
            if progress is not None:
                progress(record)
        except Exception as exc:
            if journal is not None:
                journal.status("paused" if isinstance(exc, ExperimentPaused) else "failed",
                               completed_periods=len(rounds), failing_period=ell,
                               error_type=type(exc).__name__, error=str(exc))
            raise
    if drain and pending():
        if journal is not None:
            journal.status("failed", completed_periods=len(rounds), error="cleanup safety bound exceeded")
        raise RuntimeError("cleanup safety bound exceeded with pending users or requests")
    summary = summarize_ledger(state.ledger)
    summary["completed_reservations"] = sum(user.status == "completed" for user in state.users.values())
    summary["active_reservations"] = sum(user.status == "active" for user in state.users.values())
    summary["pending_requests"] = len(state.waiting)
    summary["solve_seconds"] = sum(row["solution"]["solve_seconds"] for row in rounds)
    summary["model_wall_seconds"] = sum(row["model_wall_seconds"] for row in rounds)
    result = {
        "schema_version": 4,
        "run_mode": "discrete_mpc_terminal" if terminal_model is not None else "discrete_mpc_no_terminal",
        "completed": True, "solver_backend": "copt",
        "demand_periods": params.num_periods, "executed_periods": len(rounds),
        "cleanup_periods": max(0, len(rounds) - params.num_periods),
        "method": {"route_mode": route_mode, "charging_mode": charging_mode,
                   "terminal_kind": terminal_model.kind if terminal_model is not None else "zero",
                   "feature_variant": feature_spec.variant if feature_spec is not None else None},
        "parameter_snapshot": params.to_dict(), "initial_state": initial,
        "dayahead_plan": plans, "rounds": rounds, "ledger": deepcopy(state.ledger),
        "summary": summary, "final_state": state.to_dict(),
    }
    if journal is not None:
        journal.status("complete", completed_periods=len(rounds), summary=summary)
    return result
