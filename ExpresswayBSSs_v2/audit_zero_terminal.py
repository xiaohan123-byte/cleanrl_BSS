"""Independent audit of completed zero-terminal trajectories (read-only).

Rebuilds statistics and metrics from each raw result.json.gz, checks demand
counts, accounting identities, and the 30-minute waiting limit, then prints
per-horizon seven-day means. Does not modify any artifact.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.result_statistics import build_result_statistics
from src.experiment_metrics import trajectory_metrics

BASE = Path(__file__).resolve().parent / "outputs/zero_terminal_real_data_seed1_v1"
DATA = Path(__file__).resolve().parent / "data/final_real_data_v1"


def main():
    problems = []
    rows = []
    for job_dir in sorted(BASE.glob("runs/h*_day*")):
        ws_path = job_dir / "worker_status.json"
        if not ws_path.exists():
            continue
        ws = json.loads(ws_path.read_text(encoding="utf-8"))
        if ws.get("state") != "complete":
            continue
        job = job_dir.name

        def check(cond, msg):
            if not cond:
                problems.append(f"{job}: {msg}")

        try:
            with gzip.open(job_dir / "result.json.gz", "rt", encoding="utf-8") as f:
                result = json.load(f)
            s = result["summary"]
            check(s["completed_reservations"] + s["reservation_failures"] == 200,
                  f"reservation users {s['completed_reservations']}+{s['reservation_failures']} != 200")
            check(s["random_services"] + s["random_timeouts"] == 200,
                  f"random {s['random_services']}+{s['random_timeouts']} != 200")
            check(s["active_reservations"] == 0 and s["pending_requests"] == 0,
                  f"unfinished at end: active={s['active_reservations']} pending={s['pending_requests']}")
            check(abs(s["income"] - (s["income_reservation"] + s["income_random"])) < 1e-6,
                  "income mismatch")
            check(abs(s["total_reward"] - (s["income"] - s["charging_cost"] - s["adjustment_cost"]
                  - s["reservation_failure_cost"])) < 1e-6, "net profit mismatch")
            check(abs(s["reservation_failure_cost"] - 1000 * s["reservation_failures"]) < 1e-6,
                  "failure penalty != 1000 per failed user")
            check(abs(s["adjustment_cost"] - 10 * s["path_adjustments"]) < 1e-6,
                  "adjustment cost != 10 per adjustment")
            max_wait = max((ev.get("waiting_hours", 0.0) for ev in result["ledger"]
                            if ev.get("type") in ("reservation_service", "random_service")), default=0.0)
            check(max_wait <= 0.5 + 1e-6, f"max waiting {max_wait:.4f}h exceeds 30min")
            stats = build_result_statistics(result)
            saved_stats = json.loads((job_dir / "statistics.json").read_text(encoding="utf-8"))
            check(abs(stats["summary"]["total_reward"] - saved_stats["summary"]["total_reward"]) < 1e-6,
                  "statistics rebuild mismatch")
            day_id = int(job.split("day")[1])
            scenario = json.loads((DATA / "scenarios" / f"day_{day_id:02d}.json").read_text(encoding="utf-8"))
            metrics = trajectory_metrics(result, scenario)
            saved_metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
            for key in ("net_profit_yuan", "reservation_failure_rate", "random_service_rate",
                        "mean_wait_minutes_served", "path_adjustments"):
                check(abs(metrics[key] - saved_metrics[key]) < 1e-6, f"metrics rebuild mismatch: {key}")
            check(metrics["actual_random_count"] == 200 and metrics["reservation_count"] == 200,
                  "scenario demand counts != 200")
            rows.append({"job": job, "horizon": result["parameter_snapshot"]["horizon"],
                         "day_id": day_id, "net": s["total_reward"],
                         "res_fail": metrics["reservation_failure_rate"],
                         "rand_serv": metrics["random_service_rate"],
                         "wait_min": metrics["mean_wait_minutes_served"],
                         "adjust": s["path_adjustments"],
                         "solve_mean": metrics["solver_seconds_mean"],
                         "solve_p95": metrics["solver_seconds_p95"],
                         "tl_frac": metrics["time_limit_fraction"],
                         "gap_met": metrics["mip_gap_target_met_fraction"],
                         "cleanup": metrics["cleanup_periods"],
                         "final_kwh": metrics["final_inventory_kwh"],
                         "exec": result["executed_periods"]})
        except Exception as exc:
            problems.append(f"{job}: audit exception {type(exc).__name__}: {exc}")

    print(f"audited {len(rows)} complete jobs; problems: {len(problems)}")
    for p in problems:
        print("PROBLEM:", p)

    by_h = defaultdict(list)
    for r in rows:
        by_h[r["horizon"]].append(r)
    print("\nhorizon summary (per-day arithmetic means over completed days):")
    header = ("H", "days", "net_profit", "res_fail", "rand_serv", "wait_min", "adjust",
              "solve_mean", "solve_p95", "tl_frac", "gap_met", "cleanup", "final_kwh")
    print("{:<2} {:<5} {:>12} {:>9} {:>10} {:>9} {:>7} {:>11} {:>10} {:>8} {:>8} {:>8} {:>11}".format(*header))
    for h in sorted(by_h):
        g = by_h[h]
        n = len(g)
        m = lambda k: sum(r[k] for r in g) / n
        print(f"{h:<2} {n:<5} {m('net'):12.2f} {m('res_fail'):9.4f} {m('rand_serv'):10.4f} "
              f"{m('wait_min'):9.2f} {m('adjust'):7.1f} {m('solve_mean'):11.3f} {m('solve_p95'):10.3f} "
              f"{m('tl_frac'):8.4f} {m('gap_met'):8.4f} {m('cleanup'):8.2f} {m('final_kwh'):11.1f}")

    print("\nper-day detail:")
    for r in sorted(rows, key=lambda r: (r["horizon"], r["day_id"])):
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
