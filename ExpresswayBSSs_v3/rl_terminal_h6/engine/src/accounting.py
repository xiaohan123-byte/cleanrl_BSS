"""Accounting of realised events in the discrete rolling baseline.

Financial entries are recomputed from physical quantities and parameters.
Prediction objectives and request outcomes are never accepted as ledger events.
"""
from __future__ import annotations
from .parameters import slots_at, price_at, execution_period_limit

from math import isclose, isfinite
from typing import Any, Iterable, Mapping


class LedgerError(ValueError):
    """A realised event is inconsistent with the accounting contract."""


class DuplicateLedgerEventError(LedgerError):
    """A unique physical event was submitted more than once."""


class UnsupportedLedgerEventError(LedgerError):
    """An unsupported or prediction-only quantity was submitted."""


COMPONENTS = (
    "income_reservation", "income_random", "charging_cost", "adjustment_cost",
    "reservation_failure_cost", "reward_delta",
)
EVENT_TYPES = {
    "reservation_service", "random_service", "charging", "path_adjustment",
    "reservation_failure", "random_timeout", "reservation_arrival",
    "random_arrival", "reservation_entry", "reservation_exit", "path_publication",
}


def _number(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"{name} must be numeric") from exc
    if not isfinite(result):
        raise LedgerError(f"{name} must be finite")
    return result


def _check_value(event: Mapping[str, Any], key: str, expected: float) -> None:
    if key in event and not isclose(_number(event[key], key), expected, abs_tol=1e-7, rel_tol=1e-8):
        raise LedgerError(f"{event['event_id']}: {key} does not match physical accounting")


def event_components(params: Any, event: Mapping[str, Any]) -> dict[str, float]:
    """Validate one actual event and derive its signed financial contribution."""
    if not event.get("event_id"):
        raise LedgerError("event_id is required")
    kind = event.get("type")
    if kind not in EVENT_TYPES or event.get("realized", True) is not True:
        raise UnsupportedLedgerEventError(f"not a realised baseline event: {kind}")
    n = event.get("period")
    if not isinstance(n, int) or not 0 <= n < execution_period_limit(params):
        raise LedgerError("event period is outside the operating horizon")
    time = _number(event.get("time"), "time")
    start, end = n * params.interval_hours, (n + 1) * params.interval_hours
    if time < start - 1e-8 or time > end + 1e-8:
        raise LedgerError("event occurrence is outside its execution interval")
    if kind in {"reservation_service", "random_service", "path_adjustment"} and not isclose(time, start, abs_tol=1e-8):
        raise LedgerError("service and path adjustment must occur at a decision boundary")
    out = dict.fromkeys(COMPONENTS, 0.0)
    if kind in {"reservation_service", "random_service", "charging"}:
        i, b = event.get("station"), event.get("slot")
        if not isinstance(i, int) or not 0 <= i < params.station.num_stations:
            raise LedgerError("invalid station")
        if not isinstance(b, int) or not 0 <= b < slots_at(params, i):
            raise LedgerError("invalid slot")
        if kind == "charging":
            power = _number(event.get("power_kw"), "power_kw")
            if power < -1e-8 or power > params.slot_power_limit(i, b) + 1e-7:
                raise LedgerError("charging power exceeds slot limit")
            energy = params.interval_hours * power
            price = price_at(params, "electricity_price", i, n)
            before = _number(event.get("start_soc"), "start_soc")
            after = _number(event.get("end_soc"), "end_soc")
            if not -1e-7 <= before <= 1 + 1e-7 or not -1e-7 <= after <= 1 + 1e-7:
                raise LedgerError("charging SOC is outside [0, 1]")
            expected_after = before + energy * params.station.charging_efficiency / params.battery_capacity_kwh
            if not isclose(after, expected_after, abs_tol=1e-7, rel_tol=1e-8):
                raise LedgerError("charging SOC does not reconcile with grid energy")
            out["charging_cost"] = energy * price
        else:
            if not event.get("request_id"):
                raise LedgerError("service requires an actual request_id")
            rho = _number(event.get("return_soc"), "return_soc")
            if not 0 <= rho < 1:
                raise LedgerError("return SOC is outside [0, 1)")
            energy = params.battery_capacity_kwh * (1 - rho)
            price = price_at(params, "swap_service_price", i, n)
            out["income_reservation" if kind == "reservation_service" else "income_random"] = energy * price
        _check_value(event, "energy_kwh", energy)
        _check_value(event, "unit_price", price)
    elif kind == "path_adjustment":
        if not event.get("user_key"):
            raise LedgerError("path adjustment requires a user_key")
        out["adjustment_cost"] = params.path_adjustment_penalty
    elif kind == "reservation_failure":
        if not event.get("user_key") or not event.get("request_id"):
            raise LedgerError("reservation failure requires a user and request")
        if time >= end:
            raise LedgerError("deadline at the next boundary must remain pending")
        out["reservation_failure_cost"] = params.reservation_failure_penalty
    out["reward_delta"] = (
        out["income_reservation"] + out["income_random"] - out["charging_cost"]
        - out["adjustment_cost"] - out["reservation_failure_cost"]
    )
    for key, expected in out.items():
        _check_value(event, key, expected)
    return out


def append_events(params: Any, ledger: list[dict[str, Any]], events: Iterable[Mapping[str, Any]]) -> float:
    """Append validated events atomically; duplicate actual effects are errors."""
    event_ids = {entry["event_id"] for entry in ledger}
    serviced = {entry["request_id"] for entry in ledger if entry["type"] in {"reservation_service", "random_service"}}
    failed = {str(entry["user_key"]) for entry in ledger if entry["type"] == "reservation_failure"}
    terminal_types = {"reservation_service", "random_service", "reservation_failure", "random_timeout"}
    outcomes = {entry["request_id"] for entry in ledger if entry["type"] in terminal_types}
    additions = []
    for item in events:
        event = dict(item)
        event_id = event.get("event_id")
        if event_id in event_ids:
            raise DuplicateLedgerEventError(f"duplicate event: {event_id}")
        event_ids.add(event_id)
        if event.get("type") in terminal_types:
            request_id = event.get("request_id")
            if request_id in outcomes:
                raise DuplicateLedgerEventError(f"request already has a terminal outcome: {request_id}")
            outcomes.add(request_id)
        if event.get("type") == "reservation_service" and str(event.get("user_key")) in failed:
            raise LedgerError("a failed reservation cannot receive a later service")
        if event.get("type") in {"reservation_service", "random_service"}:
            request_id = event.get("request_id")
            if request_id in serviced:
                raise DuplicateLedgerEventError(f"request already served: {request_id}")
            serviced.add(request_id)
        if event.get("type") == "reservation_failure":
            key = str(event.get("user_key"))
            if key in failed:
                raise DuplicateLedgerEventError(f"reservation already failed: {key}")
            failed.add(key)
        event.update(event_components(params, event))
        event["realized"] = True
        additions.append(event)
    ledger.extend(additions)
    return sum(item["reward_delta"] for item in additions)


def summarize_ledger(ledger: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    entries = list(ledger)
    summary: dict[str, Any] = {key: sum(float(item.get(key, 0)) for item in entries) for key in COMPONENTS}
    summary.update(
        total_reward=summary["reward_delta"],
        income=summary["income_reservation"] + summary["income_random"],
        reservation_services=sum(item["type"] == "reservation_service" for item in entries),
        random_services=sum(item["type"] == "random_service" for item in entries),
        reservation_failures=sum(item["type"] == "reservation_failure" for item in entries),
        random_timeouts=sum(item["type"] == "random_timeout" for item in entries),
        path_adjustments=sum(item["type"] == "path_adjustment" for item in entries),
        grid_energy_kwh=sum(float(item.get("energy_kwh", 0)) for item in entries if item["type"] == "charging"),
        ledger_event_count=len(entries),
    )
    return summary
