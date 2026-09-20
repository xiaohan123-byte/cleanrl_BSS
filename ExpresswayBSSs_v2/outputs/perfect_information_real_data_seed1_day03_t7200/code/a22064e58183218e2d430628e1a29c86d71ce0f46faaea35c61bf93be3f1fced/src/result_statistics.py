"""Independent ledger reconciliation and export for baseline run results."""
from __future__ import annotations

import argparse
import csv
import json
from math import isclose
from pathlib import Path

from src.accounting import COMPONENTS, LedgerError, append_events, summarize_ledger
from src.parameters import BusinessParameters, slots_at, validate_complete_result


class StatisticsError(ValueError):
    """A saved run cannot be reconciled with its physical ledger."""


def _close(actual, expected, label):
    if not isclose(float(actual), float(expected), abs_tol=1e-6, rel_tol=1e-8):
        raise StatisticsError(f"{label} does not reconcile: {actual} versus {expected}")


def build_result_statistics(result: dict) -> dict:
    """Recompute finances, energy, inventory and counts without running a solver."""
    if result.get("schema_version") != 4 or result.get("run_mode") not in {"discrete_mpc_no_terminal", "discrete_mpc_terminal"}:
        raise StatisticsError("expected complete discrete MPC result schema version 4")
    try:
        params = BusinessParameters.from_dict(result["parameter_snapshot"])
        validate_complete_result(result)
        ledger: list[dict] = []
        append_events(params, ledger, result["ledger"])
    except (KeyError, TypeError, ValueError, LedgerError) as exc:
        raise StatisticsError(str(exc)) from exc
    summary = summarize_ledger(ledger)
    for key, expected in summary.items():
        if key in result.get("summary", {}):
            _close(result["summary"][key], expected, f"summary.{key}")
    by_period = [[] for _ in result["rounds"]]
    by_station = [[] for _ in range(params.station.num_stations)]
    for entry in ledger:
        by_period[entry["period"]].append(entry)
        if entry.get("station") is not None:
            station = entry["station"]
            if not isinstance(station, int) or not 0 <= station < params.station.num_stations:
                raise StatisticsError("ledger contains an invalid station")
            by_station[station].append(entry)
    inventory = [list(row) for row in result["initial_state"]["slot_soc"]]
    round_ids = [event["event_id"] for row in result["rounds"] for event in row["events"]]
    if round_ids != [entry["event_id"] for entry in ledger]:
        raise StatisticsError("round events do not cover the realised ledger exactly once")
    if result["final_state"].get("ledger") != result["ledger"]:
        raise StatisticsError("final-state ledger differs from the top-level ledger")
    for n, events in enumerate(by_period):
        row = result["rounds"][n]
        for i, soc_row in enumerate(inventory):
            for b, soc in enumerate(soc_row):
                _close(row["state_before"]["slot_soc"][i][b], soc, f"round {n} carried SOC")
        selected_services = [(action["request_id"], action["station"], action["slot"])
                             for action in row["solution"]["services"] if action["period"] == n]
        actual_services = [(event["request_id"], event["station"], event["slot"])
                           for event in events if event["type"] in {"reservation_service", "random_service"}]
        if sorted(selected_services) != sorted(actual_services):
            raise StatisticsError("actual services differ from the solver's first-stage assignments")
        served_slots = set()
        charged_slots = set()
        station_power = [0.] * params.station.num_stations
        for event in events:
            if event["type"] not in {"reservation_service", "random_service"}:
                continue
            i, b = event["station"], event["slot"]
            if (i, b) in served_slots:
                raise StatisticsError("one battery serves twice at the same boundary")
            served_slots.add((i, b))
            _close(inventory[i][b], 1., "service battery fullness")
            inventory[i][b] = event["return_soc"]
            if not event["arrival_time"] <= n * params.interval_hours <= event["deadline"]:
                raise StatisticsError("service occurred outside the actual waiting window")
        for event in events:
            if event["type"] != "charging":
                continue
            i, b = event["station"], event["slot"]
            if (i, b) in charged_slots:
                raise StatisticsError("duplicate slot charging interval")
            charged_slots.add((i, b))
            _close(event["power_kw"], row["solution"]["power"][i][b][0], "executed solver power")
            _close(event["start_soc"], inventory[i][b], "post-swap charging SOC")
            inventory[i][b] = event["end_soc"]
            station_power[i] += event["power_kw"]
        if len(charged_slots) != sum(slots_at(params, i) for i in params.station.station_ids):
            raise StatisticsError("each slot needs one charging record per interval, including zero power")
        if any(power > params.station_power_limit(i) + 1e-7 for i, power in enumerate(station_power)):
            raise StatisticsError("aggregate station charging power exceeds its limit")
        row = result["rounds"][n]
        if row["period"] != n:
            raise StatisticsError("rounds are not in chronological order")
        _close(row["reward"], summarize_ledger(events)["reward_delta"], f"round {n} reward")
        for i, soc_row in enumerate(inventory):
            for b, soc in enumerate(soc_row):
                _close(row["state_after"]["slot_soc"][i][b], soc, f"round {n} final slot SOC")
                _close(row["solution"]["soc"][i][b][1], soc, f"round {n} first-stage predicted SOC")
    for i, row in enumerate(inventory):
        for b, soc in enumerate(row):
            _close(result["final_state"]["slot_soc"][i][b], soc, "final inventory")
    users = result["final_state"]["users"]
    summary["completed_reservations"] = sum(user["status"] == "completed" for user in users.values())
    summary["active_reservations"] = sum(user["status"] == "active" for user in users.values())
    summary["pending_requests"] = len(result["final_state"]["waiting"])
    summary["solve_seconds"] = sum(row["solution"]["solve_seconds"] for row in result["rounds"])
    return {
        "schema_version": 1,
        "source_schema_version": 4,
        "run_mode": result["run_mode"],
        "summary": summary,
        "per_station": [{"station": i, **summarize_ledger(events)} for i, events in enumerate(by_station)],
        "per_period": [{"period": n, "time_hours": n * params.interval_hours,
                        **summarize_ledger(events)} for n, events in enumerate(by_period)],
        "checks": {"ledger_unique": True, "financial_components_reconciled": True,
                   "prices_and_energy_reconciled": True, "inventory_reconciled": True,
                   "station_power_limits_satisfied": True},
    }


def write_statistics_artifacts(statistics: dict, output_dir: str | Path,
                               prefix: str = "mpc_statistics") -> dict[str, str]:
    """Write JSON, Markdown, station CSV and period CSV using UTF-8."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {kind: directory / f"{prefix}{suffix}" for kind, suffix in {
        "json": ".json", "markdown": ".md", "station_csv": "_stations.csv",
        "period_csv": "_periods.csv",
    }.items()}
    paths["json"].write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, rows in (("station_csv", statistics["per_station"]), ("period_csv", statistics["per_period"])):
        with paths[key].open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    summary = statistics["summary"]
    lines = [
        "# Discrete MPC baseline statistics", "",
        "All amounts below are recomputed from realised events. Overlapping forecast objectives are excluded.", "",
        "| Metric | Value |", "|---|---:|",
    ]
    labels = (
        ("Service income", "income"), ("Charging cost", "charging_cost"),
        ("Published path adjustment cost", "adjustment_cost"),
        ("Reservation failure cost", "reservation_failure_cost"), ("Net realised reward", "total_reward"),
        ("Reservation services", "reservation_services"), ("Random services", "random_services"),
        ("Reservation failures", "reservation_failures"), ("Random timeouts", "random_timeouts"),
        ("Grid energy (kWh)", "grid_energy_kwh"), ("Pending requests at end", "pending_requests"),
    )
    lines.extend(f"| {label} | {summary[key]:.6g} |" for label, key in labels)
    lines.extend(["", "Ledger, energy, price, inventory and station-power reconciliation passed.", ""])
    paths["markdown"].write_text("\n".join(lines), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path, nargs="?", default=Path("outputs/mpc_run_result.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = json.loads(args.result.read_text(encoding="utf-8"))
    stats = build_result_statistics(result)
    paths = write_statistics_artifacts(stats, args.output_dir or args.result.parent,
                                       args.result.stem + "_statistics")
    print(json.dumps({"summary": stats["summary"], "artifacts": paths}, ensure_ascii=False))


if __name__ == "__main__":
    main()
