"""Deterministic post-processing for continuous six-station MPC results.

The realised ledger is the only source of operational income, cost, service,
timeout, and charging statistics.  Rolling prediction objectives overlap in
time and are therefore deliberately excluded from realised aggregates.

The module can be used from Python or as a standalone command::

    python -m src.result_statistics --strict

It writes a machine-readable JSON file, two flat CSV tables, and a concise
Markdown report next to the source result by default.  No RL model or solver is
loaded and the MPC simulation is not rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


STATISTICS_SCHEMA_VERSION = 1
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_PATH = _REPO_ROOT / "data_generation_test" / "output" / "mpc_run_result.json"
DEFAULT_MOCK_PATH = _REPO_ROOT / "data_generation_test" / "output" / "mock_rl_data.json"
DEFAULT_PLAN_PATH = _REPO_ROOT / "data_generation_test" / "output" / "dayahead_plan.json"

_COMPONENTS = (
    "income_reservation",
    "income_random",
    "charging_cost",
    "adjustment_cost",
    "reservation_failure_cost",
    "reward_delta",
)
_SERVICE_TYPES = {"reservation_service", "random_service"}
_TIMEOUT_TYPES = {"reservation_timeout", "random_timeout"}
_PREDICTION_ONLY_TYPES = {
    "pending_at_horizon",
    "served_in_horizon",
    "failed_in_horizon",
    "terminal_soc_value",
    "outside_delivery_value",
    "outside_swap_value",
}
_ABS_TOL = 1e-7


class StatisticsError(RuntimeError):
    """Raised when a result cannot be safely aggregated."""


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StatisticsError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise StatisticsError(f"{name} must be finite")
    return number


def _safe_rate(numerator: float, denominator: float) -> Optional[float]:
    return None if abs(denominator) <= _ABS_TOL else numerator / denominator


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    """Return a deterministic linearly interpolated sample percentile."""

    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(value) for value in values]
    return {
        "count": len(clean),
        "mean": fmean(clean) if clean else None,
        "p50": _percentile(clean, 0.50),
        "p95": _percentile(clean, 0.95),
        "max": max(clean) if clean else None,
    }


def _coefficient_of_variation(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    mean = fmean(values)
    if abs(mean) <= _ABS_TOL:
        return None
    variance = fmean([(value - mean) ** 2 for value in values])
    return math.sqrt(variance) / mean


def _gini_nonnegative(values: Sequence[float]) -> Optional[float]:
    clean = sorted(float(value) for value in values)
    if not clean or any(value < 0 for value in clean):
        return None
    total = sum(clean)
    if total <= _ABS_TOL:
        return 0.0
    n = len(clean)
    weighted = sum((index + 1) * value for index, value in enumerate(clean))
    return (2.0 * weighted) / (n * total) - (n + 1.0) / n


def _zero_financials() -> Dict[str, float]:
    return {component: 0.0 for component in _COMPONENTS}


def _add_financials(target: MutableMapping[str, Any], posting: Mapping[str, Any]) -> None:
    for component in _COMPONENTS:
        target[component] = float(target.get(component, 0.0)) + _finite_float(
            posting.get(component, 0.0), component
        )


def _matrix_counts(
    table: Any, num_stations: int, num_periods: int
) -> Optional[List[List[int]]]:
    if not isinstance(table, Sequence) or isinstance(table, (str, bytes)):
        return None
    if len(table) != num_stations:
        return None
    counts: List[List[int]] = []
    for station_row in table:
        if not isinstance(station_row, Sequence) or isinstance(station_row, (str, bytes)):
            return None
        if len(station_row) < num_periods:
            return None
        row: List[int] = []
        for period_items in station_row[:num_periods]:
            if not isinstance(period_items, Sequence) or isinstance(period_items, (str, bytes)):
                return None
            row.append(len(period_items))
        counts.append(row)
    return counts


def _inventory_metrics(
    row: Any, battery_capacity_kwh: Optional[float]
) -> Dict[str, Any]:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        return {
            "slot_count": 0,
            "ready_slots": 0,
            "boundary_pending_full_slots": 0,
            "mean_soc": None,
            "min_soc": None,
            "max_soc": None,
            "stored_energy_kwh": None,
        }
    soc_values: List[float] = []
    ready_slots = 0
    boundary_pending = 0
    for cell in row:
        if not isinstance(cell, Mapping):
            continue
        soc = _finite_float(cell.get("soc"), "slot.soc")
        soc_values.append(soc)
        ready = bool(cell.get("ready", False))
        ready_slots += int(ready)
        boundary_pending += int(
            not ready
            and soc >= 1.0 - _ABS_TOL
            and cell.get("completion_due_at") is not None
        )
    stored_energy = None
    if battery_capacity_kwh is not None:
        stored_energy = sum(soc_values) * battery_capacity_kwh
    return {
        "slot_count": len(soc_values),
        "ready_slots": ready_slots,
        "boundary_pending_full_slots": boundary_pending,
        "mean_soc": fmean(soc_values) if soc_values else None,
        "min_soc": min(soc_values) if soc_values else None,
        "max_soc": max(soc_values) if soc_values else None,
        "stored_energy_kwh": stored_energy,
    }


def _waiting_counts(final_state: Mapping[str, Any], num_stations: int) -> List[Dict[str, int]]:
    result = [
        {"reservation": 0, "random": 0, "total": 0}
        for _ in range(num_stations)
    ]
    raw_queues = final_state.get("waiting_queues", {})
    if not isinstance(raw_queues, Mapping):
        return result
    for station in range(num_stations):
        queues = raw_queues.get(str(station), raw_queues.get(station, {}))
        if not isinstance(queues, Mapping):
            continue
        reservation = queues.get("reservation", [])
        random = queues.get("random", [])
        reservation_count = len(reservation) if isinstance(reservation, Sequence) else 0
        random_count = len(random) if isinstance(random, Sequence) else 0
        result[station] = {
            "reservation": reservation_count,
            "random": random_count,
            "total": reservation_count + random_count,
        }
    return result


def _waiting_request_ids(state: Mapping[str, Any]) -> set[str]:
    request_ids: set[str] = set()
    raw_queues = state.get("waiting_queues", {})
    if not isinstance(raw_queues, Mapping):
        return request_ids
    for queues in raw_queues.values():
        if not isinstance(queues, Mapping):
            continue
        for requests in queues.values():
            if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
                continue
            for request in requests:
                if not isinstance(request, Mapping):
                    continue
                request_event_id = request.get("event_id", request.get("request_id"))
                if request_event_id is not None:
                    request_ids.add(str(request_event_id))
    return request_ids


def _slot_matrix_is_physical(
    state: Mapping[str, Any],
    *,
    num_stations: int,
    expected_num_slots: Optional[int],
    boundary_time: Optional[float],
) -> bool:
    rows = state.get("slots", [])
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes))
        or len(rows) != num_stations
    ):
        return False
    inferred_num_slots = expected_num_slots
    if inferred_num_slots is None:
        if not rows or not isinstance(rows[0], Sequence):
            return False
        inferred_num_slots = len(rows[0])
    if inferred_num_slots <= 0:
        return False
    for station, row in enumerate(rows):
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != inferred_num_slots
        ):
            return False
        for slot, cell in enumerate(row):
            if not isinstance(cell, Mapping):
                return False
            try:
                soc = _finite_float(cell.get("soc"), "slot.soc")
            except StatisticsError:
                return False
            if cell.get("station") != station or cell.get("slot") != slot:
                return False
            if not -_ABS_TOL <= soc <= 1.0 + _ABS_TOL:
                return False
            ready = cell.get("ready")
            if not isinstance(ready, bool):
                return False
            completion_due_at = cell.get("completion_due_at")
            if ready:
                if soc < 1.0 - _ABS_TOL or completion_due_at is not None:
                    return False
            elif completion_due_at is not None:
                try:
                    due_at = _finite_float(completion_due_at, "completion_due_at")
                except StatisticsError:
                    return False
                if soc < 1.0 - _ABS_TOL or boundary_time is None:
                    return False
                if not math.isclose(
                    due_at, boundary_time, rel_tol=0.0, abs_tol=_ABS_TOL
                ):
                    return False
            elif soc >= 1.0 - _ABS_TOL:
                # A full but unavailable slot needs explicit right-boundary
                # completion provenance; otherwise inventory is underreported.
                return False
    return True


def _matrix_price(
    table: Any, station: int, period: int, name: str
) -> float:
    if (
        not isinstance(table, Sequence)
        or isinstance(table, (str, bytes))
        or station < 0
        or station >= len(table)
    ):
        raise StatisticsError(f"{name} must define every station")
    row = table[station]
    if (
        not isinstance(row, Sequence)
        or isinstance(row, (str, bytes))
        or period < 0
        or period >= len(row)
    ):
        raise StatisticsError(f"{name}[{station}] must define every period")
    return _finite_float(row[period], f"{name}[{station}][{period}]")


def _explicit_amount_or_default(entry: Mapping[str, Any], default: float) -> float:
    metadata = entry.get("metadata", {})
    if isinstance(metadata, Mapping) and "amount" in metadata:
        return _finite_float(metadata["amount"], "metadata.amount")
    amount = _finite_float(entry.get("amount", 0.0), "entry.amount")
    return amount if abs(amount) > _ABS_TOL else default


def _empty_station_row(station: int) -> Dict[str, Any]:
    return {
        "station": station,
        "position_km": None,
        "planned_reservation_visits": 0,
        "predicted_random_requests": None,
        "actual_random_arrivals": None,
        "reservation_services": 0,
        "random_services": 0,
        "total_services": 0,
        "reservation_timeouts": 0,
        "random_timeouts": 0,
        "charging_segment_count": 0,
        "charging_energy_kwh": 0.0,
        "slot_charging_hours": 0.0,
        "delivered_energy_kwh": 0.0,
        **_zero_financials(),
        "service_income": 0.0,
        "operating_cost": 0.0,
        "average_wait_hours": None,
        "p95_wait_hours": None,
        "max_wait_hours": None,
        "on_time_service_rate": None,
        "random_arrival_service_rate": None,
        "average_energy_price": None,
        "total_energy_limit_kwh": None,
        "energy_limit_utilization": None,
        "peak_period_energy_limit_utilization": None,
        "final_slot_count": 0,
        "final_ready_slots": 0,
        "final_boundary_pending_full_slots": 0,
        "final_mean_soc": None,
        "final_min_soc": None,
        "final_max_soc": None,
        "final_stored_energy_kwh": None,
        "final_waiting_reservation": 0,
        "final_waiting_random": 0,
        "final_waiting_total": 0,
        "inventory_balance_residual_kwh": None,
        "inventory_balance_passed": None,
    }


def _empty_period_row(period: int) -> Dict[str, Any]:
    return {
        "period": period,
        "time_start": None,
        "time_end": None,
        "status": None,
        "replay_matches": None,
        "solve_time_sec": None,
        "predicted_random_requests": None,
        "actual_random_arrivals": None,
        "reservation_services": 0,
        "random_services": 0,
        "total_services": 0,
        "reservation_timeouts": 0,
        "random_timeouts": 0,
        "charging_segment_count": 0,
        "charging_energy_kwh": 0.0,
        "slot_charging_hours": 0.0,
        "delivered_energy_kwh": 0.0,
        **_zero_financials(),
        "service_income": 0.0,
        "operating_cost": 0.0,
        "average_wait_hours": None,
        "max_wait_hours": None,
        "energy_limit_kwh": None,
        "energy_limit_utilization": None,
        "ending_waiting_count": None,
        "ending_ready_slots": None,
        "ending_boundary_pending_full_slots": None,
        "ending_mean_soc": None,
    }


def _financial_close(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        math.isclose(
            _finite_float(left.get(component, 0.0), component),
            _finite_float(right.get(component, 0.0), component),
            rel_tol=0.0,
            abs_tol=_ABS_TOL,
        )
        for component in _COMPONENTS
    )


def build_result_statistics(
    result: Mapping[str, Any],
    *,
    mock: Optional[Mapping[str, Any]] = None,
    plan: Optional[Mapping[str, Any]] = None,
    strict: bool = False,
    source_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Build deterministic realised statistics from a schema-3 run result.

    ``mock`` and ``plan`` are optional in descriptive mode.  They add
    count-level forecast comparison, independent price reconstruction,
    energy-limit/inventory checks, and plan-execution coverage without
    changing the ledger-based financial totals.  Strict mode is fail-closed
    and therefore requires both sidecars.
    """

    num_stations = int(result.get("num_stations", 0))
    num_periods = int(result.get("num_periods", 0))
    if num_stations <= 0 or num_periods <= 0:
        raise StatisticsError("result must define positive num_stations and num_periods")

    ledger = result.get("ledger", {})
    postings = ledger.get("postings", []) if isinstance(ledger, Mapping) else []
    if not isinstance(postings, Sequence) or isinstance(postings, (str, bytes)):
        raise StatisticsError("result.ledger.postings must be a list")

    rounds = result.get("rounds", [])
    rounds_by_period: Dict[int, Mapping[str, Any]] = {}
    round_time_windows: Dict[int, tuple[float, float]] = {}
    if isinstance(rounds, Sequence) and not isinstance(rounds, (str, bytes)):
        for round_record in rounds:
            if not isinstance(round_record, Mapping) or not isinstance(
                round_record.get("period"), int
            ):
                continue
            period = int(round_record["period"])
            rounds_by_period[period] = round_record
            time_window = round_record.get("time_window", [])
            if isinstance(time_window, Sequence) and len(time_window) == 2:
                round_time_windows[period] = (
                    _finite_float(time_window[0], "time_start"),
                    _finite_float(time_window[1], "time_end"),
                )

    checks: Dict[str, Optional[bool]] = {
        "result_schema_supported": result.get("schema_version") == 3,
        "six_station_run": num_stations == 6,
        "synthetic_mock_sources": (
            result.get("data_source") == "synthetic"
            and result.get("signal_source") == "mock"
        ),
        "mock_context_available": isinstance(mock, Mapping),
        "plan_context_available": isinstance(plan, Mapping),
    }
    station_rows = [_empty_station_row(station) for station in range(num_stations)]
    period_rows = [_empty_period_row(period) for period in range(num_periods)]
    station_waits: List[List[float]] = [[] for _ in range(num_stations)]
    station_on_time: List[List[bool]] = [[] for _ in range(num_stations)]
    period_waits: List[List[float]] = [[] for _ in range(num_periods)]
    service_waits: List[float] = []
    reservation_service_waits: List[float] = []
    random_service_waits: List[float] = []
    timeout_waits: List[float] = []
    all_resolved_waits: List[float] = []
    overall_financial = _zero_financials()
    event_type_counts: Counter[str] = Counter()
    event_ids: List[str] = []
    invalid_station_or_period = False
    realised_only = True
    no_prediction_entries = True
    posting_identities_hold = True
    supported_event_types = True
    event_times_match_intervals = True
    charging_segments_physical = True
    request_wait_times_physical = True
    service_request_ids: List[str] = []
    timeout_request_ids: List[str] = []
    reservation_service_request_ids: set[str] = set()
    reservation_timeout_request_ids: set[str] = set()
    random_service_request_ids: set[str] = set()
    random_timeout_request_ids: set[str] = set()
    network_level = {
        "path_publications": 0,
        "unassigned_posting_count": 0,
        **_zero_financials(),
    }

    for posting_index, posting in enumerate(postings):
        if not isinstance(posting, Mapping):
            raise StatisticsError(f"ledger posting {posting_index} must be an object")
        entry = posting.get("entry", {})
        if not isinstance(entry, Mapping):
            raise StatisticsError(f"ledger posting {posting_index}.entry must be an object")
        event_id = str(entry.get("event_id", ""))
        event_type = str(entry.get("event_type", ""))
        event_ids.append(event_id)
        event_type_counts[event_type] += 1
        realised_only = realised_only and bool(entry.get("realized", False))
        no_prediction_entries = no_prediction_entries and event_type not in _PREDICTION_ONLY_TYPES
        supported_event_types = supported_event_types and event_type in {
            "charging",
            "reservation_service",
            "random_service",
            "reservation_timeout",
            "random_timeout",
            "path_published",
            "request_cancelled",
        }
        _add_financials(overall_financial, posting)

        expected_reward = (
            _finite_float(posting.get("income_reservation", 0.0), "income_reservation")
            + _finite_float(posting.get("income_random", 0.0), "income_random")
            - _finite_float(posting.get("charging_cost", 0.0), "charging_cost")
            - _finite_float(posting.get("adjustment_cost", 0.0), "adjustment_cost")
            - _finite_float(
                posting.get("reservation_failure_cost", 0.0),
                "reservation_failure_cost",
            )
        )
        posting_identities_hold = posting_identities_hold and math.isclose(
            expected_reward,
            _finite_float(posting.get("reward_delta", 0.0), "reward_delta"),
            rel_tol=0.0,
            abs_tol=_ABS_TOL,
        )

        raw_station = entry.get("station")
        station = raw_station if isinstance(raw_station, int) else None
        raw_interval = entry.get("interval")
        period = raw_interval if isinstance(raw_interval, int) else None
        has_valid_station = station is not None and 0 <= station < num_stations
        has_valid_period = period is not None and 0 <= period < num_periods
        station_required = event_type in _SERVICE_TYPES | _TIMEOUT_TYPES | {"charging"}
        if station_required and not has_valid_station:
            invalid_station_or_period = True
        if not has_valid_period:
            invalid_station_or_period = True

        if has_valid_station:
            station_row = station_rows[station]
            _add_financials(station_row, posting)
        else:
            station_row = None
            network_level["unassigned_posting_count"] += 1
            _add_financials(network_level, posting)
        if has_valid_period:
            period_row = period_rows[period]
            _add_financials(period_row, posting)
        else:
            period_row = None

        metadata = entry.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        occurred_at = _finite_float(entry.get("occurred_at"), "occurred_at")
        window = round_time_windows.get(period) if has_valid_period else None
        if event_type != "charging":
            event_times_match_intervals = event_times_match_intervals and (
                window is not None
                and window[0] - _ABS_TOL <= occurred_at < window[1]
            )
        if event_type == "charging":
            energy = _finite_float(entry.get("energy_kwh", 0.0), "charging energy_kwh")
            start_time = _finite_float(metadata.get("start_time", entry.get("occurred_at", 0.0)), "start_time")
            end_time = _finite_float(metadata.get("end_time", entry.get("occurred_at", 0.0)), "end_time")
            power_kw = _finite_float(metadata.get("power_kw", 0.0), "power_kw")
            duration = end_time - start_time
            expected_energy = power_kw * duration
            charging_segments_physical = charging_segments_physical and (
                energy >= -_ABS_TOL
                and power_kw >= -_ABS_TOL
                and duration >= -_ABS_TOL
                and math.isclose(
                    occurred_at, end_time, rel_tol=0.0, abs_tol=_ABS_TOL
                )
                and math.isclose(
                    energy,
                    expected_energy,
                    rel_tol=1e-9,
                    abs_tol=_ABS_TOL,
                )
                and window is not None
                and window[0] - _ABS_TOL <= start_time
                and end_time <= window[1] + _ABS_TOL
            )
            event_times_match_intervals = (
                event_times_match_intervals
                and window is not None
                and window[0] - _ABS_TOL <= start_time
                and end_time <= window[1] + _ABS_TOL
            )
            duration_for_statistics = max(0.0, duration)
            if station_row is not None:
                station_row["charging_segment_count"] += 1
                station_row["charging_energy_kwh"] += energy
                station_row["slot_charging_hours"] += duration_for_statistics
            if period_row is not None:
                period_row["charging_segment_count"] += 1
                period_row["charging_energy_kwh"] += energy
                period_row["slot_charging_hours"] += duration_for_statistics
        elif event_type in _SERVICE_TYPES:
            kind_prefix = "reservation" if event_type == "reservation_service" else "random"
            capacity = metadata.get("battery_capacity_kwh")
            return_soc = metadata.get("return_soc")
            delivered_energy = 0.0
            if capacity is not None and return_soc is not None:
                capacity_value = _finite_float(capacity, "battery_capacity_kwh")
                return_soc_value = _finite_float(return_soc, "return_soc")
                request_wait_times_physical = request_wait_times_physical and (
                    capacity_value > 0.0
                    and -_ABS_TOL <= return_soc_value <= 1.0 + _ABS_TOL
                )
                delivered_energy = capacity_value * (1.0 - return_soc_value)
            else:
                request_wait_times_physical = False
            arrival_value = entry.get("arrival_time")
            deadline_value = entry.get("deadline")
            if arrival_value is None or deadline_value is None:
                arrival_time = occurred_at
                deadline = occurred_at
                request_wait_times_physical = False
            else:
                arrival_time = _finite_float(arrival_value, "arrival_time")
                deadline = _finite_float(deadline_value, "deadline")
            wait_value = metadata.get("wait_hours")
            observed_wait = occurred_at - arrival_time
            wait = (
                observed_wait
                if wait_value is None
                else _finite_float(wait_value, "wait_hours")
            )
            request_wait_times_physical = request_wait_times_physical and (
                arrival_time <= occurred_at + _ABS_TOL
                and occurred_at <= deadline + _ABS_TOL
                and deadline >= arrival_time - _ABS_TOL
                and wait >= -_ABS_TOL
                and math.isclose(
                    wait, observed_wait, rel_tol=0.0, abs_tol=_ABS_TOL
                )
            )
            request_event_id = str(
                metadata.get("request_event_id", entry.get("request_id", ""))
            )
            service_request_ids.append(request_event_id)
            if kind_prefix == "reservation":
                reservation_service_request_ids.add(request_event_id)
            else:
                random_service_request_ids.add(request_event_id)
            on_time = occurred_at <= deadline + _ABS_TOL
            service_waits.append(wait)
            all_resolved_waits.append(wait)
            if kind_prefix == "reservation":
                reservation_service_waits.append(wait)
            else:
                random_service_waits.append(wait)
            if station_row is not None:
                station_row[f"{kind_prefix}_services"] += 1
                station_row["delivered_energy_kwh"] += delivered_energy
                station_waits[station].append(wait)
                station_on_time[station].append(on_time)
            if period_row is not None:
                period_row[f"{kind_prefix}_services"] += 1
                period_row["delivered_energy_kwh"] += delivered_energy
                period_waits[period].append(wait)
        elif event_type in _TIMEOUT_TYPES:
            kind_prefix = "reservation" if event_type == "reservation_timeout" else "random"
            arrival_value = entry.get("arrival_time")
            deadline_value = entry.get("deadline")
            if arrival_value is None or deadline_value is None:
                arrival_time = occurred_at
                deadline = occurred_at
                request_wait_times_physical = False
            else:
                arrival_time = _finite_float(arrival_value, "arrival_time")
                deadline = _finite_float(deadline_value, "deadline")
            wait_value = metadata.get("wait_hours")
            observed_wait = occurred_at - arrival_time
            wait = (
                observed_wait
                if wait_value is None
                else _finite_float(wait_value, "wait_hours")
            )
            request_wait_times_physical = request_wait_times_physical and (
                deadline >= arrival_time - _ABS_TOL
                and wait >= -_ABS_TOL
                and math.isclose(
                    occurred_at, deadline, rel_tol=0.0, abs_tol=_ABS_TOL
                )
                and math.isclose(
                    wait, observed_wait, rel_tol=0.0, abs_tol=_ABS_TOL
                )
            )
            request_event_id = str(
                metadata.get("request_event_id", entry.get("request_id", ""))
            )
            timeout_request_ids.append(request_event_id)
            if kind_prefix == "reservation":
                reservation_timeout_request_ids.add(request_event_id)
            else:
                random_timeout_request_ids.add(request_event_id)
            timeout_waits.append(wait)
            all_resolved_waits.append(wait)
            if station_row is not None:
                station_row[f"{kind_prefix}_timeouts"] += 1
                station_waits[station].append(wait)
            if period_row is not None:
                period_row[f"{kind_prefix}_timeouts"] += 1
                period_waits[period].append(wait)
        elif event_type == "path_published":
            network_level["path_publications"] += 1

    checks["unique_realized_event_ids"] = (
        all(event_ids) and len(event_ids) == len(set(event_ids))
    )
    checks["realized_entries_only"] = realised_only
    checks["no_prediction_values_in_ledger"] = no_prediction_entries
    checks["ledger_event_types_supported"] = supported_event_types
    checks["posting_reward_identities_hold"] = posting_identities_hold
    checks["event_station_and_period_indices_valid"] = not invalid_station_or_period
    checks["event_times_match_declared_intervals"] = event_times_match_intervals
    checks["charging_segments_physical"] = charging_segments_physical
    checks["request_wait_times_physical"] = request_wait_times_physical

    ledger_summary = ledger.get("summary", {}) if isinstance(ledger, Mapping) else {}
    checks["ledger_summary_reconciled"] = (
        isinstance(ledger_summary, Mapping)
        and _financial_close(overall_financial, ledger_summary)
    )
    top_summary = result.get("summary", {})
    top_summary_financial = {
        component: (
            top_summary.get("total_reward", 0.0)
            if component == "reward_delta"
            else top_summary.get(f"total_{component}", 0.0)
        )
        for component in _COMPONENTS
    } if isinstance(top_summary, Mapping) else {}
    checks["top_level_summary_reconciled"] = (
        isinstance(top_summary, Mapping)
        and _financial_close(overall_financial, top_summary_financial)
    )
    checks["top_level_service_counts_reconciled"] = (
        isinstance(top_summary, Mapping)
        and top_summary.get("total_actual_reservation_services")
        == event_type_counts["reservation_service"]
        and top_summary.get("total_actual_random_services")
        == event_type_counts["random_service"]
        and top_summary.get("total_actual_reservation_timeouts")
        == event_type_counts["reservation_timeout"]
    )

    round_financial = _zero_financials()
    replay_matches: List[bool] = []
    solve_times: List[float] = []
    status_counts: Counter[str] = Counter()
    for period in range(num_periods):
        round_record = rounds_by_period.get(period)
        if round_record is None:
            continue
        round_ledger = round_record.get("ledger", {})
        if isinstance(round_ledger, Mapping):
            _add_financials(round_financial, round_ledger)
        period_row = period_rows[period]
        time_window = round_record.get("time_window", [])
        if isinstance(time_window, Sequence) and len(time_window) == 2:
            period_row["time_start"] = _finite_float(time_window[0], "time_start")
            period_row["time_end"] = _finite_float(time_window[1], "time_end")
        period_row["status"] = str(round_record.get("status", ""))
        status_counts[period_row["status"]] += 1
        replay = round_record.get("replay", {})
        replay_value = bool(replay.get("matches", False)) if isinstance(replay, Mapping) else False
        period_row["replay_matches"] = replay_value
        replay_matches.append(replay_value)
        solve_time = _finite_float(round_record.get("solve_time_sec", 0.0), "solve_time_sec")
        period_row["solve_time_sec"] = solve_time
        solve_times.append(solve_time)
        actual = round_record.get("actual", {})
        if isinstance(actual, Mapping):
            waiting_ids = actual.get("waiting_ids", [])
            period_row["ending_waiting_count"] = (
                len(waiting_ids) if isinstance(waiting_ids, Sequence) else None
            )
        state_end = round_record.get("state_end", {})
        if isinstance(state_end, Mapping):
            slot_rows = state_end.get("slots", [])
            if isinstance(slot_rows, Sequence):
                all_slots = [cell for row in slot_rows for cell in row] if slot_rows else []
                inventory = _inventory_metrics(all_slots, None)
                period_row["ending_ready_slots"] = inventory["ready_slots"]
                period_row["ending_boundary_pending_full_slots"] = inventory[
                    "boundary_pending_full_slots"
                ]
                period_row["ending_mean_soc"] = inventory["mean_soc"]
    checks["round_ledger_reconciled"] = (
        len(rounds_by_period) == num_periods
        and _financial_close(overall_financial, round_financial)
    )
    checks["all_first_interval_replays_matched"] = (
        len(replay_matches) == num_periods and all(replay_matches)
    )

    parameter_snapshot: Mapping[str, Any] = {}
    if isinstance(mock, Mapping) and isinstance(mock.get("parameter_snapshot"), Mapping):
        parameter_snapshot = mock["parameter_snapshot"]
    station_parameters = parameter_snapshot.get("station", {}) if parameter_snapshot else {}
    positions = station_parameters.get("positions_km", []) if isinstance(station_parameters, Mapping) else []
    battery_capacity = parameter_snapshot.get("battery_capacity_kwh") if parameter_snapshot else None
    battery_capacity_kwh = (
        _finite_float(battery_capacity, "battery_capacity_kwh")
        if battery_capacity is not None
        else None
    )
    charging_efficiency_value = station_parameters.get("charging_efficiency") if isinstance(station_parameters, Mapping) else None
    charging_efficiency = (
        _finite_float(charging_efficiency_value, "charging_efficiency")
        if charging_efficiency_value is not None
        else None
    )
    energy_limits = parameter_snapshot.get("station_energy_limit_kwh") if parameter_snapshot else None
    valid_energy_limits = (
        isinstance(energy_limits, Sequence)
        and len(energy_limits) == num_stations
        and all(isinstance(row, Sequence) and len(row) >= num_periods for row in energy_limits)
    )
    initial_slot_soc = station_parameters.get("initial_slot_soc") if isinstance(station_parameters, Mapping) else None
    valid_initial_soc = (
        isinstance(initial_slot_soc, Sequence)
        and len(initial_slot_soc) == num_stations
    )
    expected_num_slots = (
        int(station_parameters.get("num_slots"))
        if isinstance(station_parameters, Mapping)
        and station_parameters.get("num_slots") is not None
        else None
    )
    checks["mock_context_consistent"] = (
        isinstance(mock, Mapping)
        and mock.get("schema_version") == 2
        and mock.get("data_source") == "synthetic"
        and mock.get("signal_source") == "mock"
        and mock.get("seed") == result.get("seed")
        and bool(parameter_snapshot)
    )
    checks["plan_context_consistent"] = (
        isinstance(plan, Mapping)
        and plan.get("schema_version") == 2
        and plan.get("engine") == "continuous_event_v2"
        and plan.get("seed") == result.get("seed")
        and plan.get("data_source") == "synthetic"
        and plan.get("signal_source") == "mock"
    )
    checks["mock_parameter_shapes_valid"] = (
        isinstance(station_parameters, Mapping)
        and station_parameters.get("num_stations") == num_stations
        and isinstance(expected_num_slots, int)
        and expected_num_slots > 0
        and valid_energy_limits
        and valid_initial_soc
        and battery_capacity_kwh is not None
        and battery_capacity_kwh > 0.0
        and charging_efficiency is not None
        and 0.0 < charging_efficiency <= 1.0
    )

    # Recompute every ledger component from the immutable event payload and
    # the parameter snapshot.  Reconciliation against copied summaries alone
    # cannot detect a consistently corrupted posting.
    financials_recomputed = bool(checks["mock_parameter_shapes_valid"])
    energy_prices = parameter_snapshot.get("electricity_price") if parameter_snapshot else None
    service_prices = parameter_snapshot.get("swap_service_price") if parameter_snapshot else None
    failure_penalty = parameter_snapshot.get("reservation_failure_penalty") if parameter_snapshot else None
    adjustment_penalty = parameter_snapshot.get("path_adjustment_penalty") if parameter_snapshot else None
    if financials_recomputed:
        try:
            failure_penalty_value = _finite_float(
                failure_penalty, "reservation_failure_penalty"
            )
            adjustment_penalty_value = _finite_float(
                adjustment_penalty, "path_adjustment_penalty"
            )
            for posting in postings:
                entry = posting["entry"]
                event_type = str(entry.get("event_type", ""))
                station = entry.get("station")
                period = entry.get("interval")
                expected = _zero_financials()
                metadata = entry.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    metadata = {}
                if event_type == "charging":
                    price = _matrix_price(
                        energy_prices, int(station), int(period), "electricity_price"
                    )
                    expected["charging_cost"] = _finite_float(
                        entry.get("energy_kwh", 0.0), "energy_kwh"
                    ) * price
                elif event_type in _SERVICE_TYPES:
                    price = _matrix_price(
                        service_prices, int(station), int(period), "swap_service_price"
                    )
                    if "amount" in metadata or abs(
                        _finite_float(entry.get("amount", 0.0), "entry.amount")
                    ) > _ABS_TOL:
                        income = _explicit_amount_or_default(entry, price)
                    else:
                        service_energy = metadata.get("service_energy_kwh")
                        if service_energy is None and metadata.get("return_soc") is not None:
                            capacity = metadata.get(
                                "battery_capacity_kwh", battery_capacity_kwh
                            )
                            service_energy = _finite_float(
                                capacity, "battery_capacity_kwh"
                            ) * (
                                1.0
                                - _finite_float(metadata["return_soc"], "return_soc")
                            )
                        income = (
                            price
                            if service_energy is None
                            else price
                            * _finite_float(service_energy, "service_energy_kwh")
                        )
                    target = (
                        "income_reservation"
                        if event_type == "reservation_service"
                        else "income_random"
                    )
                    expected[target] = income
                elif event_type == "reservation_timeout":
                    expected["reservation_failure_cost"] = _explicit_amount_or_default(
                        entry, failure_penalty_value
                    )
                elif event_type == "path_published":
                    expected["adjustment_cost"] = _explicit_amount_or_default(
                        entry, adjustment_penalty_value
                    )
                expected["reward_delta"] = (
                    expected["income_reservation"]
                    + expected["income_random"]
                    - expected["charging_cost"]
                    - expected["adjustment_cost"]
                    - expected["reservation_failure_cost"]
                )
                financials_recomputed = financials_recomputed and _financial_close(
                    posting, expected
                )
        except (StatisticsError, TypeError, ValueError, IndexError):
            financials_recomputed = False
    checks["ledger_components_recomputed_from_parameters"] = financials_recomputed

    predicted_counts = _matrix_counts(
        mock.get("predicted_random_requests") if isinstance(mock, Mapping) else None,
        num_stations,
        num_periods,
    )
    actual_counts = _matrix_counts(
        mock.get("actual_random_requests") if isinstance(mock, Mapping) else None,
        num_stations,
        num_periods,
    )
    count_errors: List[float] = []
    if predicted_counts is not None and actual_counts is not None:
        for station in range(num_stations):
            station_rows[station]["predicted_random_requests"] = sum(predicted_counts[station])
            station_rows[station]["actual_random_arrivals"] = sum(actual_counts[station])
            for period in range(num_periods):
                period_rows[period]["predicted_random_requests"] = sum(
                    predicted_counts[i][period] for i in range(num_stations)
                )
                period_rows[period]["actual_random_arrivals"] = sum(
                    actual_counts[i][period] for i in range(num_stations)
                )
                count_errors.append(actual_counts[station][period] - predicted_counts[station][period])

    final_state = result.get("final_state", {})
    final_slots = final_state.get("slots", []) if isinstance(final_state, Mapping) else []
    final_now = (
        _finite_float(final_state.get("now"), "final_state.now")
        if isinstance(final_state, Mapping) and final_state.get("now") is not None
        else None
    )
    expected_final_time = (
        round_time_windows[num_periods - 1][1]
        if num_periods - 1 in round_time_windows
        else None
    )
    checks["final_time_matches_last_round"] = (
        final_now is not None
        and expected_final_time is not None
        and math.isclose(
            final_now, expected_final_time, rel_tol=0.0, abs_tol=_ABS_TOL
        )
    )
    checks["final_slot_state_physical"] = (
        isinstance(final_state, Mapping)
        and _slot_matrix_is_physical(
            final_state,
            num_stations=num_stations,
            expected_num_slots=expected_num_slots,
            boundary_time=final_now,
        )
    )
    round_slot_states_physical = len(rounds_by_period) == num_periods
    for period in range(num_periods):
        round_record = rounds_by_period.get(period, {})
        state_end = round_record.get("state_end", {}) if isinstance(round_record, Mapping) else {}
        boundary_time = round_time_windows.get(period, (None, None))[1]
        round_slot_states_physical = round_slot_states_physical and (
            isinstance(state_end, Mapping)
            and _slot_matrix_is_physical(
                state_end,
                num_stations=num_stations,
                expected_num_slots=expected_num_slots,
                boundary_time=boundary_time,
            )
        )
    checks["round_end_slot_states_physical"] = round_slot_states_physical
    final_waiting = _waiting_counts(
        final_state if isinstance(final_state, Mapping) else {}, num_stations
    )
    inventory_balance_results: List[bool] = []
    energy_limit_results: List[bool] = []
    period_station_energy: List[List[float]] = [
        [0.0 for _ in range(num_periods)] for _ in range(num_stations)
    ]
    for posting in postings:
        entry = posting.get("entry", {}) if isinstance(posting, Mapping) else {}
        if not isinstance(entry, Mapping) or entry.get("event_type") != "charging":
            continue
        station = entry.get("station")
        period = entry.get("interval")
        if isinstance(station, int) and isinstance(period, int) and 0 <= station < num_stations and 0 <= period < num_periods:
            period_station_energy[station][period] += _finite_float(
                entry.get("energy_kwh", 0.0), "energy_kwh"
            )

    for station, station_row in enumerate(station_rows):
        if isinstance(positions, Sequence) and station < len(positions):
            station_row["position_km"] = _finite_float(positions[station], "position_km")
        station_row["total_services"] = (
            station_row["reservation_services"] + station_row["random_services"]
        )
        station_row["service_income"] = (
            station_row["income_reservation"] + station_row["income_random"]
        )
        station_row["operating_cost"] = (
            station_row["charging_cost"]
            + station_row["adjustment_cost"]
            + station_row["reservation_failure_cost"]
        )
        wait_stats = _distribution(station_waits[station])
        station_row["average_wait_hours"] = wait_stats["mean"]
        station_row["p95_wait_hours"] = wait_stats["p95"]
        station_row["max_wait_hours"] = wait_stats["max"]
        station_row["on_time_service_rate"] = _safe_rate(
            sum(station_on_time[station]), len(station_on_time[station])
        )
        if station_row["actual_random_arrivals"] is not None:
            station_row["random_arrival_service_rate"] = _safe_rate(
                station_row["random_services"], station_row["actual_random_arrivals"]
            )
        station_row["average_energy_price"] = _safe_rate(
            station_row["charging_cost"], station_row["charging_energy_kwh"]
        )
        if valid_energy_limits:
            limits = [
                _finite_float(energy_limits[station][period], "station_energy_limit_kwh")
                for period in range(num_periods)
            ]
            ratios = [
                _safe_rate(period_station_energy[station][period], limits[period]) or 0.0
                for period in range(num_periods)
            ]
            station_row["total_energy_limit_kwh"] = sum(limits)
            station_row["energy_limit_utilization"] = _safe_rate(
                station_row["charging_energy_kwh"], sum(limits)
            )
            station_row["peak_period_energy_limit_utilization"] = max(ratios)
            energy_limit_results.extend(
                period_station_energy[station][period] <= limits[period] + _ABS_TOL
                for period in range(num_periods)
            )
        slot_row = final_slots[station] if isinstance(final_slots, Sequence) and station < len(final_slots) else []
        inventory = _inventory_metrics(slot_row, battery_capacity_kwh)
        station_row["final_slot_count"] = inventory["slot_count"]
        station_row["final_ready_slots"] = inventory["ready_slots"]
        station_row["final_boundary_pending_full_slots"] = inventory[
            "boundary_pending_full_slots"
        ]
        station_row["final_mean_soc"] = inventory["mean_soc"]
        station_row["final_min_soc"] = inventory["min_soc"]
        station_row["final_max_soc"] = inventory["max_soc"]
        station_row["final_stored_energy_kwh"] = inventory["stored_energy_kwh"]
        station_row["final_waiting_reservation"] = final_waiting[station]["reservation"]
        station_row["final_waiting_random"] = final_waiting[station]["random"]
        station_row["final_waiting_total"] = final_waiting[station]["total"]
        if (
            battery_capacity_kwh is not None
            and charging_efficiency is not None
            and valid_initial_soc
            and isinstance(initial_slot_soc[station], Sequence)
            and inventory["stored_energy_kwh"] is not None
        ):
            initial_energy = sum(
                _finite_float(value, "initial_slot_soc")
                for value in initial_slot_soc[station]
            ) * battery_capacity_kwh
            expected_change = (
                charging_efficiency * station_row["charging_energy_kwh"]
                - station_row["delivered_energy_kwh"]
            )
            residual = (
                station_row["final_stored_energy_kwh"]
                - initial_energy
                - expected_change
            )
            station_row["inventory_balance_residual_kwh"] = residual
            station_row["inventory_balance_passed"] = abs(residual) <= 1e-6
            inventory_balance_results.append(station_row["inventory_balance_passed"])

    checks["station_and_network_financials_reconciled"] = all(
        math.isclose(
            sum(float(row[component]) for row in station_rows)
            + float(network_level[component]),
            overall_financial[component],
            rel_tol=0.0,
            abs_tol=_ABS_TOL,
        )
        for component in _COMPONENTS
    )

    for period, period_row in enumerate(period_rows):
        period_row["total_services"] = (
            period_row["reservation_services"] + period_row["random_services"]
        )
        period_row["service_income"] = (
            period_row["income_reservation"] + period_row["income_random"]
        )
        period_row["operating_cost"] = (
            period_row["charging_cost"]
            + period_row["adjustment_cost"]
            + period_row["reservation_failure_cost"]
        )
        wait_stats = _distribution(period_waits[period])
        period_row["average_wait_hours"] = wait_stats["mean"]
        period_row["max_wait_hours"] = wait_stats["max"]
        if valid_energy_limits:
            limit = sum(
                _finite_float(energy_limits[station][period], "station_energy_limit_kwh")
                for station in range(num_stations)
            )
            period_row["energy_limit_kwh"] = limit
            period_row["energy_limit_utilization"] = _safe_rate(
                period_row["charging_energy_kwh"], limit
            )

    checks["station_energy_limits_satisfied"] = (
        all(energy_limit_results) if energy_limit_results else None
    )
    checks["station_inventory_energy_balanced"] = (
        all(inventory_balance_results) if inventory_balance_results else None
    )
    checks["station_rows_complete"] = len(station_rows) == num_stations
    checks["period_rows_complete"] = len(period_rows) == num_periods

    plan_execution: Dict[str, Any] = {
        "accepted_reservation_users": None,
        "rejected_reservation_users": None,
        "planned_reservation_events": None,
        "served_planned_events": None,
        "timed_out_planned_events": None,
        "unrealized_planned_events": None,
        "planned_event_realization_rate": None,
        "completed_reservation_users": len(top_summary.get("completed_reservations", [])) if isinstance(top_summary, Mapping) else None,
        "failed_reservation_users": len(top_summary.get("failed_reservations", [])) if isinstance(top_summary, Mapping) else None,
        "accepted_user_completion_rate": None,
        "service_time_deviation_from_dayahead_hours": _distribution([]),
    }
    if isinstance(plan, Mapping):
        reservations = plan.get("reservations", [])
        if isinstance(reservations, Sequence):
            accepted_records = [
                record for record in reservations
                if isinstance(record, Mapping) and bool(record.get("accepted", False))
            ]
            rejected_records = [
                record for record in reservations
                if isinstance(record, Mapping) and not bool(record.get("accepted", False))
            ]
            planned_events: Dict[str, Dict[str, Any]] = {}
            for record in accepted_records:
                event_ids_raw = record.get("event_ids", [])
                stations_raw = record.get("swap_stations", [])
                times_raw = record.get("swap_times", [])
                if not isinstance(event_ids_raw, Sequence):
                    continue
                for index, raw_event_id in enumerate(event_ids_raw):
                    event_id = str(raw_event_id)
                    event_station = stations_raw[index] if isinstance(stations_raw, Sequence) and index < len(stations_raw) else None
                    event_time = times_raw[index] if isinstance(times_raw, Sequence) and index < len(times_raw) else None
                    planned_events[event_id] = {
                        "station": event_station,
                        "time": event_time,
                    }
                    if isinstance(event_station, int) and 0 <= event_station < num_stations:
                        station_rows[event_station]["planned_reservation_visits"] += 1
            served_ids: set[str] = set()
            timeout_ids: set[str] = set()
            timing_deviations: List[float] = []
            for posting in postings:
                entry = posting.get("entry", {}) if isinstance(posting, Mapping) else {}
                if not isinstance(entry, Mapping):
                    continue
                event_type = entry.get("event_type")
                metadata = entry.get("metadata", {})
                request_event_id = (
                    str(metadata.get("request_event_id"))
                    if isinstance(metadata, Mapping) and metadata.get("request_event_id") is not None
                    else str(entry.get("request_id", ""))
                )
                if event_type == "reservation_service":
                    served_ids.add(request_event_id)
                    planned = planned_events.get(request_event_id)
                    if planned is not None and planned.get("time") is not None:
                        timing_deviations.append(
                            _finite_float(entry.get("occurred_at"), "occurred_at")
                            - _finite_float(planned["time"], "planned swap time")
                        )
                elif event_type == "reservation_timeout":
                    timeout_ids.add(request_event_id)
            planned_ids = set(planned_events)
            realized_ids = (served_ids | timeout_ids) & planned_ids
            accepted_user_keys = {
                (int(record["od_id"]), int(record["reservation_id"]))
                for record in accepted_records
                if record.get("od_id") is not None
                and record.get("reservation_id") is not None
            }

            def belongs_to_accepted_route(request_id: str) -> bool:
                parts = request_id.split(":")
                if len(parts) != 5 or parts[0] != "reservation":
                    return False
                try:
                    key = (int(parts[1]), int(parts[2]))
                    int(parts[3])
                    int(parts[4])
                except ValueError:
                    return False
                return key in accepted_user_keys

            reservation_ledger_ids = (
                reservation_service_request_ids
                | reservation_timeout_request_ids
            )
            checks["reservation_ledger_ids_match_plan"] = (
                all(
                    request_id in planned_ids
                    or belongs_to_accepted_route(request_id)
                    for request_id in reservation_ledger_ids
                )
            )
            completed_users = plan_execution["completed_reservation_users"] or 0
            plan_execution.update(
                {
                    "accepted_reservation_users": len(accepted_records),
                    "rejected_reservation_users": len(rejected_records),
                    "planned_reservation_events": len(planned_ids),
                    "served_planned_events": len(served_ids & planned_ids),
                    "timed_out_planned_events": len(timeout_ids & planned_ids),
                    "unrealized_planned_events": len(planned_ids - realized_ids),
                    "planned_event_realization_rate": _safe_rate(
                        len(realized_ids), len(planned_ids)
                    ),
                    "accepted_user_completion_rate": _safe_rate(
                        completed_users, len(accepted_records)
                    ),
                    "service_time_deviation_from_dayahead_hours": _distribution(
                        timing_deviations
                    ),
                }
            )
            baseline_visits = plan.get("baseline_station_visits")
            if isinstance(baseline_visits, Sequence) and len(baseline_visits) == num_stations:
                baseline_total = sum(
                    sum(_finite_float(value, "baseline_station_visits") for value in row)
                    for row in baseline_visits
                    if isinstance(row, Sequence)
                )
                checks["dayahead_visit_count_reconciled"] = math.isclose(
                    baseline_total, len(planned_ids), rel_tol=0.0, abs_tol=_ABS_TOL
                )

    actual_random_arrivals = (
        sum(sum(row) for row in actual_counts) if actual_counts is not None else None
    )
    predicted_random_requests = (
        sum(sum(row) for row in predicted_counts) if predicted_counts is not None else None
    )
    final_random_waiting = sum(item["random"] for item in final_waiting)
    final_waiting_ids = _waiting_request_ids(
        final_state if isinstance(final_state, Mapping) else {}
    )
    random_services = event_type_counts["random_service"]
    random_timeouts = event_type_counts["random_timeout"]
    checks["request_outcome_ids_mutually_exclusive"] = (
        all(service_request_ids)
        and all(timeout_request_ids)
        and len(service_request_ids) == len(set(service_request_ids))
        and len(timeout_request_ids) == len(set(timeout_request_ids))
        and len(final_waiting_ids) == sum(item["total"] for item in final_waiting)
        and not (set(service_request_ids) & set(timeout_request_ids))
        and not (set(service_request_ids) & final_waiting_ids)
        and not (set(timeout_request_ids) & final_waiting_ids)
    )
    if actual_random_arrivals is not None:
        checks["actual_random_arrivals_accounted"] = (
            random_services + random_timeouts + final_random_waiting
            == actual_random_arrivals
        )
        expected_random_request_ids: set[str] = set()
        actual_random_table = mock.get("actual_random_requests", []) if isinstance(mock, Mapping) else []
        if isinstance(actual_random_table, Sequence):
            for station, station_periods in enumerate(actual_random_table):
                if not isinstance(station_periods, Sequence):
                    continue
                for requests in station_periods[:num_periods]:
                    if not isinstance(requests, Sequence):
                        continue
                    for request in requests:
                        if isinstance(request, Mapping) and request.get("request_id") is not None:
                            expected_random_request_ids.add(
                                f"actual:{station}:{request['request_id']}"
                            )
        realized_or_waiting_random_ids = (
            random_service_request_ids
            | random_timeout_request_ids
            | {
                request_id
                for request_id in final_waiting_ids
                if request_id.startswith("actual:")
            }
        )
        checks["actual_random_request_ids_accounted"] = (
            realized_or_waiting_random_ids == expected_random_request_ids
        )

    total_services = event_type_counts["reservation_service"] + random_services
    total_service_income = (
        overall_financial["income_reservation"] + overall_financial["income_random"]
    )
    total_operating_cost = (
        overall_financial["charging_cost"]
        + overall_financial["adjustment_cost"]
        + overall_financial["reservation_failure_cost"]
    )
    total_charging_energy = sum(row["charging_energy_kwh"] for row in station_rows)
    total_delivered_energy = sum(row["delivered_energy_kwh"] for row in station_rows)
    final_inventory_rows = [
        _inventory_metrics(
            final_slots[station] if isinstance(final_slots, Sequence) and station < len(final_slots) else [],
            battery_capacity_kwh,
        )
        for station in range(num_stations)
    ]
    final_soc_values = [
        cell.get("soc")
        for row in (final_slots if isinstance(final_slots, Sequence) else [])
        if isinstance(row, Sequence)
        for cell in row
        if isinstance(cell, Mapping) and cell.get("soc") is not None
    ]
    realized_summary = {
        **overall_financial,
        "service_income": total_service_income,
        "operating_cost": total_operating_cost,
        "total_reward": overall_financial["reward_delta"],
        "reservation_services": event_type_counts["reservation_service"],
        "random_services": random_services,
        "total_services": total_services,
        "reservation_timeouts": event_type_counts["reservation_timeout"],
        "random_timeouts": random_timeouts,
        "path_publications": event_type_counts["path_published"],
        "charging_segment_count": event_type_counts["charging"],
        "charging_energy_kwh": total_charging_energy,
        "delivered_energy_kwh": total_delivered_energy,
        "average_realized_energy_price": _safe_rate(
            overall_financial["charging_cost"], total_charging_energy
        ),
        "average_service_income_per_delivered_kwh": _safe_rate(
            total_service_income, total_delivered_energy
        ),
        "reward_per_service": _safe_rate(
            overall_financial["reward_delta"], total_services
        ),
        "net_reward_margin": _safe_rate(
            overall_financial["reward_delta"], total_service_income
        ),
        "reservation_event_service_rate": _safe_rate(
            event_type_counts["reservation_service"],
            event_type_counts["reservation_service"] + event_type_counts["reservation_timeout"],
        ),
        "random_resolved_service_rate": _safe_rate(
            random_services, random_services + random_timeouts
        ),
        "random_arrival_service_rate": _safe_rate(
            random_services, actual_random_arrivals or 0
        ) if actual_random_arrivals is not None else None,
        "reservation_service_share": _safe_rate(
            event_type_counts["reservation_service"], total_services
        ),
        "service_wait_hours": _distribution(service_waits),
        "reservation_service_wait_hours": _distribution(reservation_service_waits),
        "random_service_wait_hours": _distribution(random_service_waits),
        "timeout_wait_hours": _distribution(timeout_waits),
        "all_resolved_request_wait_hours": _distribution(all_resolved_waits),
        "final_ready_slots": sum(row["ready_slots"] for row in final_inventory_rows),
        "final_boundary_pending_full_slots": sum(
            row["boundary_pending_full_slots"] for row in final_inventory_rows
        ),
        "final_mean_soc": fmean(_finite_float(value, "final soc") for value in final_soc_values) if final_soc_values else None,
        "final_waiting_requests": sum(item["total"] for item in final_waiting),
        "positive_reward_periods": sum(row["reward_delta"] > _ABS_TOL for row in period_rows),
        "negative_reward_periods": sum(row["reward_delta"] < -_ABS_TOL for row in period_rows),
    }

    demand_comparison = {
        "comparison_unit": "station-period request count",
        "predicted_random_requests": predicted_random_requests,
        "actual_random_arrivals": actual_random_arrivals,
        "actual_minus_predicted_total": (
            actual_random_arrivals - predicted_random_requests
            if actual_random_arrivals is not None and predicted_random_requests is not None
            else None
        ),
        "count_error_mean_actual_minus_predicted": fmean(count_errors) if count_errors else None,
        "count_error_mae": fmean(abs(value) for value in count_errors) if count_errors else None,
        "count_error_rmse": math.sqrt(fmean(value * value for value in count_errors)) if count_errors else None,
        "note": "Predicted and actual mock request streams are independently generated; these are aggregate count differences, not request-level accuracy.",
    }
    service_counts = [float(row["total_services"]) for row in station_rows]
    station_balance = {
        "service_count_mean": fmean(service_counts),
        "service_count_coefficient_of_variation": _coefficient_of_variation(service_counts),
        "service_count_gini": _gini_nonnegative(service_counts),
        "busiest_station": max(range(num_stations), key=lambda station: station_rows[station]["total_services"]),
        "highest_local_reward_station": max(range(num_stations), key=lambda station: station_rows[station]["reward_delta"]),
        "lowest_local_reward_station": min(range(num_stations), key=lambda station: station_rows[station]["reward_delta"]),
        "note": "These values describe one seeded mock run and are not evidence of policy fairness or superiority.",
    }
    diagnostics = {
        "ledger_posting_count": len(postings),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "round_status_counts": dict(sorted(status_counts.items())),
        "replay_match_rate": _safe_rate(sum(replay_matches), len(replay_matches)),
        "solve_time_sec": _distribution(solve_times),
        "peak_waiting_requests": max(
            (row["ending_waiting_count"] or 0 for row in period_rows), default=0
        ),
        "max_station_period_energy_limit_utilization": max(
            (
                row["peak_period_energy_limit_utilization"]
                for row in station_rows
                if row["peak_period_energy_limit_utilization"] is not None
            ),
            default=None,
        ),
        "max_abs_inventory_balance_residual_kwh": max(
            (
                abs(row["inventory_balance_residual_kwh"])
                for row in station_rows
                if row["inventory_balance_residual_kwh"] is not None
            ),
            default=None,
        ),
        "prediction_objectives_aggregated": False,
    }

    failed_checks = sorted(name for name, passed in checks.items() if passed is False)
    checks["all_evaluated_checks_passed"] = not failed_checks
    if strict and failed_checks:
        raise StatisticsError(
            "strict statistics validation failed: " + ", ".join(failed_checks)
        )

    source = {
        "result_schema_version": result.get("schema_version"),
        "run_mode": result.get("run_mode"),
        "seed": result.get("seed"),
        "data_source": result.get("data_source"),
        "signal_source": result.get("signal_source"),
        "num_stations": num_stations,
        "num_periods": num_periods,
    }
    if source_sha256 is not None:
        source["result_sha256"] = source_sha256
    path_publication_details: List[Dict[str, Any]] = []
    if isinstance(rounds, Sequence) and not isinstance(rounds, (str, bytes)):
        for round_record in rounds:
            if not isinstance(round_record, Mapping):
                continue
            period = round_record.get("period")
            decisions = round_record.get("path_decisions", [])
            if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes)):
                continue
            for decision in decisions:
                if not isinstance(decision, Mapping) or not decision.get(
                    "publication_event_id"
                ):
                    continue
                path_publication_details.append(
                    {
                        "period": period,
                        "user_key": list(decision.get("user_key", [])),
                        "previous_station_sequence": list(
                            decision.get("previous_station_sequence", [])
                        ),
                        "proposed_station_sequence": list(
                            decision.get("proposed_station_sequence", [])
                        ),
                        "publication_event_id": decision["publication_event_id"],
                    }
                )
    actual_route_events: Dict[str, List[tuple[float, int]]] = {}
    for posting in postings:
        if not isinstance(posting, Mapping):
            continue
        entry = posting.get("entry", {})
        if not isinstance(entry, Mapping) or entry.get("event_type") != "reservation_service":
            continue
        user_key = entry.get("user_key")
        station = entry.get("station")
        if not isinstance(user_key, Sequence) or len(user_key) != 2 or not isinstance(station, int):
            continue
        key_text = f"{int(user_key[0])}:{int(user_key[1])}"
        actual_route_events.setdefault(key_text, []).append(
            (_finite_float(entry.get("occurred_at"), "occurred_at"), station)
        )
    path_optimization = {
        "status": result.get("path_search", {}).get("status")
        if isinstance(result.get("path_search"), Mapping)
        else None,
        "method": result.get("path_search", {}).get("method")
        if isinstance(result.get("path_search"), Mapping)
        else None,
        "global_milp_optimality_claimed": bool(
            result.get("path_search", {}).get("global_milp_optimality_claimed", False)
        )
        if isinstance(result.get("path_search"), Mapping)
        else False,
        "publication_count": len(path_publication_details),
        "realized_adjustment_cost": overall_financial["adjustment_cost"],
        "publications": path_publication_details,
        "actual_service_station_sequences": {
            key: [station for _, station in sorted(events)]
            for key, events in sorted(actual_route_events.items())
        },
    }
    return {
        "statistics_schema_version": STATISTICS_SCHEMA_VERSION,
        "generator": "src.result_statistics",
        "source": source,
        "checks": checks,
        "realized_summary": realized_summary,
        "plan_execution": plan_execution,
        "random_demand_count_comparison": demand_comparison,
        "network_level_statistics": network_level,
        "station_balance_diagnostics": station_balance,
        "path_optimization": path_optimization,
        "by_station": station_rows,
        "by_period": period_rows,
        "diagnostics": diagnostics,
        "definitions": {
            "financial_source": "realized ledger postings only",
            "charging_energy_kwh": "sum of charging ledger segment energy; segment count is not a charging-session count",
            "delivered_energy_kwh": "battery_capacity_kwh * (1 - return_soc) for realized services",
            "ready_slots": "count of final slot states with ready=true; SOC=1 at the right boundary is not sufficient",
            "reservation_service_rate": "realized reservation service events divided by service plus timeout events",
            "random_arrival_service_rate": "realized random services divided by all simulated actual random arrivals, including unresolved end-of-day requests",
        },
    }


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def statistics_markdown(statistics: Mapping[str, Any]) -> str:
    """Render a concise, deterministic Chinese report."""

    source = statistics["source"]
    summary = statistics["realized_summary"]
    plan = statistics["plan_execution"]
    demand = statistics["random_demand_count_comparison"]
    checks = statistics["checks"]
    path = statistics.get("path_optimization", {})
    lines = [
        "# 六站 Mock MPC 运行统计",
        "",
        (
            f"场景：`{source['run_mode']}`；seed={source['seed']}；"
            f"{source['num_stations']} 个站点，{source['num_periods']} 个时段。"
        ),
        "",
        "## 实际运营汇总",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 预约服务次数 | {_format_number(summary['reservation_services'])} |",
        f"| 随机服务次数 | {_format_number(summary['random_services'])} |",
        f"| 预约/随机超时次数 | {_format_number(summary['reservation_timeouts'])} / {_format_number(summary['random_timeouts'])} |",
        f"| 实际充电量（kWh） | {_format_number(summary['charging_energy_kwh'])} |",
        f"| 实际交付电量（kWh） | {_format_number(summary['delivered_energy_kwh'])} |",
        f"| 服务收入 | {_format_number(summary['service_income'])} |",
        f"| 充电成本 | {_format_number(summary['charging_cost'])} |",
        f"| 路径调整/预约失败成本 | {_format_number(summary['adjustment_cost'])} / {_format_number(summary['reservation_failure_cost'])} |",
        f"| 实际净收益 | {_format_number(summary['total_reward'])} |",
        f"| 单次服务净收益 | {_format_number(summary['reward_per_service'])} |",
        f"| 服务平均/P95等待（h） | {_format_number(summary['service_wait_hours']['mean'])} / {_format_number(summary['service_wait_hours']['p95'])} |",
        f"| 日终可立即服务槽位 | {_format_number(summary['final_ready_slots'])} |",
        "",
    ]
    if isinstance(path, Mapping):
        lines.extend(
            [
                "## 路径优化与实际执行",
                "",
                (
                    f"状态：`{path.get('status')}`；方法：`{path.get('method')}`；"
                    f"实际发布 {path.get('publication_count', 0)} 次，"
                    f"调整成本 {_format_number(path.get('realized_adjustment_cost'))}。"
                ),
                "",
                "| 时段 | 用户 | 原剩余站序 | 新剩余站序 | 发布事件 |",
                "|---:|---|---|---|---|",
            ]
        )
        publications = path.get("publications", [])
        if isinstance(publications, Sequence):
            for publication in publications:
                if not isinstance(publication, Mapping):
                    continue
                user_key = publication.get("user_key", [])
                user_text = ":".join(str(item) for item in user_key)
                old_text = "-".join(
                    str(item)
                    for item in publication.get("previous_station_sequence", [])
                ) or "直达出口"
                new_text = "-".join(
                    str(item)
                    for item in publication.get("proposed_station_sequence", [])
                ) or "直达出口"
                lines.append(
                    f"| {publication.get('period')} | {user_text} | {old_text} | "
                    f"{new_text} | `{publication.get('publication_event_id')}` |"
                )
        lines.append("")
    lines.extend(
        [
            "## 分站统计",
            "",
            "| 站点 | 预约 | 随机 | 超时 | 充电量/kWh | 交付量/kWh | 服务收入 | 充电成本 | 本地净收益 | 日终平均SOC | 可用槽 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in statistics["by_station"]:
        lines.append(
            "| {station} | {reservation_services} | {random_services} | {timeouts} | "
            "{charging} | {delivered} | {income} | {cost} | {reward} | {soc} | {ready} |".format(
                station=row["station"],
                reservation_services=row["reservation_services"],
                random_services=row["random_services"],
                timeouts=row["reservation_timeouts"] + row["random_timeouts"],
                charging=_format_number(row["charging_energy_kwh"]),
                delivered=_format_number(row["delivered_energy_kwh"]),
                income=_format_number(row["service_income"]),
                cost=_format_number(row["charging_cost"]),
                reward=_format_number(row["reward_delta"]),
                soc=_format_number(row["final_mean_soc"]),
                ready=row["final_ready_slots"],
            )
        )
    lines.extend(
        [
            "",
            "## 分时段统计",
            "",
            "| 时段 | 预约 | 随机 | 充电量/kWh | 服务收入 | 充电成本 | 实际净收益 | 期末等待 | 期末可用槽 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in statistics["by_period"]:
        lines.append(
            "| {period} | {reservation_services} | {random_services} | {charging} | "
            "{income} | {cost} | {reward} | {waiting} | {ready} |".format(
                period=row["period"],
                reservation_services=row["reservation_services"],
                random_services=row["random_services"],
                charging=_format_number(row["charging_energy_kwh"]),
                income=_format_number(row["service_income"]),
                cost=_format_number(row["charging_cost"]),
                reward=_format_number(row["reward_delta"]),
                waiting=_format_number(row["ending_waiting_count"]),
                ready=_format_number(row["ending_ready_slots"]),
            )
        )
    lines.extend(
        [
            "",
            "## 计划、需求与一致性诊断",
            "",
            f"- 已接纳预约用户：{_format_number(plan['accepted_reservation_users'])}；完成用户：{_format_number(plan['completed_reservation_users'])}。",
            f"- 日前预约事件：{_format_number(plan['planned_reservation_events'])}；已服务：{_format_number(plan['served_planned_events'])}；超时：{_format_number(plan['timed_out_planned_events'])}；未实现：{_format_number(plan['unrealized_planned_events'])}。",
            f"- 随机预测/实际请求：{_format_number(demand['predicted_random_requests'])} / {_format_number(demand['actual_random_arrivals'])}；站点—时段计数 MAE：{_format_number(demand['count_error_mae'])}。",
            f"- 全部已执行检查通过：{'是' if checks['all_evaluated_checks_passed'] else '否'}。",
            "",
            (
                "说明：财务与服务统计仅累计真实账本事件。滚动预测窗口互相重叠，因此未累计 "
                "`prediction.objective_total`。当前路径与请求结果由完整站级事件模式扩展的 "
                "Gurobi MILP 联合优化；逐轮状态为 OPTIMAL 时，全局性限于已生成的完整候选网络"
                "和固定 Mock 信号。RL 尚未训练，因此这些结果不能解释为 RL 策略性能。"
                if path.get("global_milp_optimality_claimed")
                else
                "说明：财务与服务统计仅累计真实账本事件。滚动预测窗口互相重叠，因此未累计 "
                "`prediction.objective_total`。当前求解未声明全局最优，且 RL 尚未训练。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_statistics_artifacts(
    statistics: Mapping[str, Any],
    output_dir: Path | str,
    *,
    stem: str = "mpc_run",
) -> Dict[str, Path]:
    """Write JSON, station CSV, period CSV, and Markdown sidecars."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / f"{stem}_statistics.json",
        "station_csv": destination / f"{stem}_station_statistics.csv",
        "period_csv": destination / f"{stem}_period_statistics.csv",
        "markdown": destination / f"{stem}_statistics.md",
    }
    with paths["json"].open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(
            statistics,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")
    for key, rows_key in (("station_csv", "by_station"), ("period_csv", "by_period")):
        rows = list(statistics[rows_key])
        if not rows:
            continue
        with paths[key].open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    with paths["markdown"].open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(statistics_markdown(statistics))
    return paths


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise StatisticsError(f"{path} must contain a JSON object")
    return payload


def _default_stem(result_path: Path) -> str:
    stem = result_path.stem
    return stem[:-7] if stem.endswith("_result") else stem


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Generate deterministic realised statistics for a six-station MPC result."
    )
    parser.add_argument("--result", default=str(DEFAULT_RESULT_PATH), help="schema-3 result JSON")
    parser.add_argument("--mock-data", default=None, help="optional schema-2 mock JSON")
    parser.add_argument("--plan", default=None, help="optional schema-2 day-ahead plan JSON")
    parser.add_argument("--output-dir", default=None, help="statistics output directory")
    parser.add_argument("--stem", default=None, help="output filename stem")
    parser.add_argument("--strict", action="store_true", help="fail if consistency checks do not pass")
    args = parser.parse_args(argv)

    result_path = Path(args.result).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else result_path.parent
    mock_path = Path(args.mock_data).resolve() if args.mock_data else result_path.parent / DEFAULT_MOCK_PATH.name
    plan_path = Path(args.plan).resolve() if args.plan else result_path.parent / DEFAULT_PLAN_PATH.name
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes.decode("utf-8"))
    mock = _load_json(mock_path) if mock_path.exists() else None
    plan = _load_json(plan_path) if plan_path.exists() else None
    statistics = build_result_statistics(
        result,
        mock=mock,
        plan=plan,
        strict=args.strict,
        source_sha256=hashlib.sha256(result_bytes).hexdigest(),
    )
    paths = write_statistics_artifacts(
        statistics,
        output_dir,
        stem=args.stem or _default_stem(result_path),
    )
    print(
        "统计已生成："
        + "；".join(f"{name}={path}" for name, path in paths.items())
    )
    return statistics


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "DEFAULT_RESULT_PATH",
    "STATISTICS_SCHEMA_VERSION",
    "StatisticsError",
    "build_result_statistics",
    "statistics_markdown",
    "write_statistics_artifacts",
]
