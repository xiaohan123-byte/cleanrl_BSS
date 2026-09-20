"""Metrics derived from complete, independently reconciled actual trajectories."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
from .parameters import validate_complete_result


def trajectory_metrics(result: dict, scenario: dict) -> dict:
    validate_complete_result(result)
    ledger = result["ledger"]
    summary = result["summary"]
    reservations = len(scenario["reservations"])
    duration = result["parameter_snapshot"]["num_periods"] * result["parameter_snapshot"]["interval_hours"]
    random_count = sum(0 <= r["arrival_time"] < duration for r in scenario["actual_random_requests"])
    end_of_demand = result["rounds"][result["parameter_snapshot"]["num_periods"] - 1]["state_after"]
    unfinished_at_L = sum(user["status"] == "active" for user in end_of_demand["users"].values())
    if result["parameter_snapshot"].get("finish_pending_after_demand", False):
        if summary["completed_reservations"] + summary["reservation_failures"] != reservations:
            raise ValueError("final reservation outcomes do not cover all input users")
        if summary["random_services"] + summary["random_timeouts"] != random_count:
            raise ValueError("final random outcomes do not cover all input requests")
    services = [e for e in ledger if e["type"] in {"reservation_service", "random_service"}]
    waits = [(e["time"] - e["arrival_time"]) * 60 for e in services]
    solve_times = [r["solution"]["solve_seconds"] for r in result["rounds"]]
    wall_times = [r.get("model_wall_seconds", r["solution"]["solve_seconds"]) for r in result["rounds"]]
    gaps = [r["solution"]["mip_gap"] for r in result["rounds"]]
    counts = defaultdict(int)
    for row in result["rounds"]:
        counts[row["solution"]["status"]] += 1
    return {
        "net_profit_yuan": summary["total_reward"], "service_income_yuan": summary["income"],
        "charging_cost_yuan": summary["charging_cost"], "adjustment_cost_yuan": summary["adjustment_cost"],
        "reservation_failure_cost_yuan": summary["reservation_failure_cost"],
        "reservation_count": reservations, "actual_random_count": random_count,
        "reservation_failures": summary["reservation_failures"],
        "reservation_failure_rate": summary["reservation_failures"] / reservations if reservations else None,
        "reservation_completed": summary["completed_reservations"],
        "reservation_unfinished_at_L": unfinished_at_L,
        "reservation_unfinished_rate_at_L": unfinished_at_L / reservations if reservations else None,
        "reservation_unfinished_after_cleanup": summary["active_reservations"],
        "random_services": summary["random_services"],
        "random_service_rate": summary["random_services"] / random_count if random_count else None,
        "mean_wait_minutes_served": float(np.mean(waits)) if waits else None,
        "path_adjustments": summary["path_adjustments"],
        "solver_seconds_mean": float(np.mean(solve_times)),
        "solver_seconds_p95": float(np.percentile(solve_times,95)),
        "model_wall_seconds_mean": float(np.mean(wall_times)),
        "model_wall_seconds_p95": float(np.percentile(wall_times,95)),
        "model_wall_seconds_total": float(np.sum(wall_times)),
        "mip_gap_mean": float(np.mean(gaps)), "mip_gap_max": float(np.max(gaps)),
        "mip_gap_target_met_fraction": float(np.mean(np.array(gaps) <= result["parameter_snapshot"]["solver"]["mip_gap"] + 1e-10)),
        "solver_status_counts": dict(counts), "completed_periods": len(result["rounds"]),
        "demand_periods": result["parameter_snapshot"]["num_periods"],
        "cleanup_periods": result.get("cleanup_periods", 0),
        "cleanup_hours": result.get("cleanup_periods", 0) * result["parameter_snapshot"]["interval_hours"],
        "time_limit_fraction": counts["time_limit"] / len(result["rounds"]),
        "final_inventory_kwh": result["parameter_snapshot"]["battery_capacity_kwh"] * sum(map(sum, result["final_state"]["slot_soc"])),
        "random_timeouts": sum(e["type"] == "random_timeout" for e in ledger),
        "terminal_settlement": "none", "operating_duration_hours": duration,
    }


def mean_std(values) -> dict:
    data = np.asarray([x for x in values if x is not None],dtype=float)
    if not len(data):
        return {"n":0,"mean":None,"std":None,"ci95_low":None,"ci95_high":None}
    if not np.isfinite(data).all():
        raise ValueError("nonfinite experiment statistic")
    average = float(data.mean())
    if len(data) == 1:
        return {"n":1,"mean":average,"std":None,"ci95_low":None,"ci95_high":None}
    from scipy.stats import t
    sd = float(data.std(ddof=1))
    half = float(t.ppf(.975,len(data)-1)*sd/math.sqrt(len(data)))
    return {"n":len(data),"mean":average,"std":sd,"ci95_low":average-half,"ci95_high":average+half}
