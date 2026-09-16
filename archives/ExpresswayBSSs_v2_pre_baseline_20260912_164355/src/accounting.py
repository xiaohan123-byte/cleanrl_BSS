"""Realised-event accounting for the continuous execution kernel.

Forecast terminal values and MPC pending labels have no occurrence time, so
they are intentionally absent from this module.  ``RealizedLedger`` accepts
only idempotent, physically realised events emitted by ``ContinuousEventEngine``
or a path-publication adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Union

from src.domain import LedgerEntry, LedgerEventType
from src.time_grid import TimeGrid, TimeGridError


class LedgerError(RuntimeError):
    """Base class for realised-ledger validation failures."""


class DuplicateLedgerEventError(LedgerError):
    """Raised when a physically unique event is submitted twice."""


class UnsupportedLedgerEventError(TypeError, LedgerError):
    """Raised for a prediction-only or otherwise non-realised quantity."""


PriceSource = Union[
    float,
    Sequence[float],
    Sequence[Sequence[float]],
    Mapping[Any, Any],
    Callable[..., float],
]


@dataclass(frozen=True)
class LedgerPosting:
    """The deterministic financial effect of one accepted realised event."""

    entry: LedgerEntry
    income_reservation: float = 0.0
    income_random: float = 0.0
    charging_cost: float = 0.0
    adjustment_cost: float = 0.0
    reservation_failure_cost: float = 0.0
    reward_delta: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry": self.entry.to_dict(),
            "income_reservation": self.income_reservation,
            "income_random": self.income_random,
            "charging_cost": self.charging_cost,
            "adjustment_cost": self.adjustment_cost,
            "reservation_failure_cost": self.reservation_failure_cost,
            "reward_delta": self.reward_delta,
        }


def _price_at(
    source: PriceSource, interval: int, name: str, station: int | None = None
) -> float:
    """Read an interval price or a station-by-interval price.

    Supported forms include a scalar, ``[period]``, ``[station][period]``, a
    mapping with tuple ``(station, period)`` keys, and callables accepting
    either ``(period)`` or ``(station, period)``.
    """

    if callable(source):
        if station is None:
            value = source(interval)
        else:
            try:
                value = source(station, interval)
            except TypeError:
                value = source(interval)
    elif isinstance(source, Mapping):
        if station is not None and (station, interval) in source:
            value = source[(station, interval)]
        elif station is not None and f"{station}:{interval}" in source:
            value = source[f"{station}:{interval}"]
        else:
            station_row = None
            if station is not None:
                station_row = source.get(station, source.get(str(station)))
            if isinstance(station_row, Mapping) or (
                isinstance(station_row, Sequence)
                and not isinstance(station_row, (str, bytes))
            ):
                return _price_at(station_row, interval, name)
            if interval in source:
                value = source[interval]
            elif str(interval) in source:
                value = source[str(interval)]
            else:
                raise LedgerError(f"{name} does not define interval {interval}")
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        if (
            station is not None
            and len(source) > 0
            and isinstance(source[0], Sequence)
            and not isinstance(source[0], (str, bytes))
        ):
            if station < 0 or station >= len(source):
                raise LedgerError(f"{name} does not define station {station}")
            return _price_at(source[station], interval, name)
        if interval < 0 or interval >= len(source):
            raise LedgerError(f"{name} does not define interval {interval}")
        value = source[interval]
    else:
        value = source
    numeric = float(value)
    if not isfinite(numeric):
        raise LedgerError(f"{name} value must be finite")
    return numeric


class RealizedLedger:
    """Idempotent accounting of actual services, charging, and failures.

    All components use the sign convention used by the reward equation:

    ``income_reservation + income_random - charging_cost - adjustment_cost
    - reservation_failure_cost``.
    """

    _PREDICTION_ONLY_TYPES = {
        "pending_at_horizon",
        "served_in_horizon",
        "failed_in_horizon",
        "terminal_soc_value",
        "outside_delivery_value",
        "outside_swap_value",
    }

    def __init__(
        self,
        time_grid: TimeGrid,
        *,
        energy_price: PriceSource = 0.0,
        reservation_service_price: PriceSource = 0.0,
        random_service_price: PriceSource = 0.0,
        reservation_failure_penalty: PriceSource = 0.0,
        path_adjustment_cost: PriceSource = 0.0,
        battery_capacity_kwh: float | None = None,
    ) -> None:
        self.time_grid = time_grid
        self.energy_price = energy_price
        self.reservation_service_price = reservation_service_price
        self.random_service_price = random_service_price
        self.reservation_failure_penalty = reservation_failure_penalty
        self.path_adjustment_cost = path_adjustment_cost
        if battery_capacity_kwh is not None and (
            not isfinite(battery_capacity_kwh) or battery_capacity_kwh <= 0
        ):
            raise LedgerError("battery_capacity_kwh must be positive when specified")
        self.battery_capacity_kwh = (
            None if battery_capacity_kwh is None else float(battery_capacity_kwh)
        )
        self._postings: List[LedgerPosting] = []
        self._event_ids: set[str] = set()

    @property
    def event_ids(self) -> set[str]:
        return set(self._event_ids)

    @property
    def postings(self) -> List[LedgerPosting]:
        return list(self._postings)

    @property
    def entries(self) -> List[LedgerEntry]:
        return [posting.entry for posting in self._postings]

    def submit(self, entry: LedgerEntry | Mapping[str, Any]) -> LedgerPosting:
        """Validate, account, and retain one realised entry exactly once."""

        normalized = self._coerce_entry(entry)
        if not normalized.realized:
            raise UnsupportedLedgerEventError("prediction-only entry cannot be realised")
        if normalized.event_id in self._event_ids:
            raise DuplicateLedgerEventError(
                f"ledger event was already submitted: {normalized.event_id}"
            )
        posting = self._make_posting(normalized)
        self._event_ids.add(normalized.event_id)
        self._postings.append(posting)
        return posting

    # Alias for streaming callers.
    record = submit

    def submit_many(
        self, entries: Iterable[LedgerEntry | Mapping[str, Any]]
    ) -> List[LedgerPosting]:
        return [self.submit(entry) for entry in entries]

    def reward_for_interval(self, interval: int) -> float:
        """Return only realised reward components for one interval."""

        return sum(
            posting.reward_delta
            for posting in self._postings
            if posting.entry.interval == interval
        )

    def components_for_interval(self, interval: int) -> Dict[str, float]:
        fields = (
            "income_reservation",
            "income_random",
            "charging_cost",
            "adjustment_cost",
            "reservation_failure_cost",
            "reward_delta",
        )
        result = {field: 0.0 for field in fields}
        for posting in self._postings:
            if posting.entry.interval != interval:
                continue
            for field in fields:
                result[field] += getattr(posting, field)
        return result

    def summary(self) -> Dict[str, float]:
        fields = (
            "income_reservation",
            "income_random",
            "charging_cost",
            "adjustment_cost",
            "reservation_failure_cost",
            "reward_delta",
        )
        result = {field: 0.0 for field in fields}
        for posting in self._postings:
            for field in fields:
                result[field] += getattr(posting, field)
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_ids": sorted(self._event_ids),
            "postings": [posting.to_dict() for posting in self._postings],
            "summary": self.summary(),
        }

    def _coerce_entry(self, entry: LedgerEntry | Mapping[str, Any]) -> LedgerEntry:
        if isinstance(entry, LedgerEntry):
            return entry
        if not isinstance(entry, Mapping):
            raise LedgerError(f"unsupported ledger entry type: {type(entry)!r}")
        event_type = str(entry.get("event_type", ""))
        if event_type in self._PREDICTION_ONLY_TYPES:
            raise UnsupportedLedgerEventError(
                f"{event_type} has no realised occurrence time and cannot enter ledger"
            )
        try:
            return LedgerEntry.from_dict(entry)
        except ValueError as exc:
            if event_type in self._PREDICTION_ONLY_TYPES:
                raise UnsupportedLedgerEventError(event_type) from exc
            raise LedgerError(str(exc)) from exc

    def _make_posting(self, entry: LedgerEntry) -> LedgerPosting:
        event_type = entry.event_type
        income_reservation = 0.0
        income_random = 0.0
        charging_cost = 0.0
        adjustment_cost = 0.0
        reservation_failure_cost = 0.0

        if event_type is LedgerEventType.CHARGING:
            charging_cost = entry.energy_kwh * _price_at(
                self.energy_price, entry.interval, "energy_price", entry.station
            )
        elif event_type is LedgerEventType.RESERVATION_SERVICE:
            self._assert_service_interval(entry)
            income_reservation = self._service_income(
                entry, self.reservation_service_price, "reservation_service_price"
            )
        elif event_type is LedgerEventType.RANDOM_SERVICE:
            self._assert_service_interval(entry)
            income_random = self._service_income(
                entry, self.random_service_price, "random_service_price"
            )
        elif event_type is LedgerEventType.PATH_PUBLISHED:
            adjustment_cost = self._explicit_or_price(
                entry, self.path_adjustment_cost, "path_adjustment_cost"
            )
        elif event_type is LedgerEventType.RESERVATION_TIMEOUT:
            self._assert_service_interval(entry)
            reservation_failure_cost = self._explicit_or_price(
                entry,
                self.reservation_failure_penalty,
                "reservation_failure_penalty",
            )
        elif event_type is LedgerEventType.RANDOM_TIMEOUT:
            self._assert_service_interval(entry)
            # Record the realised timeout but intentionally assign no random
            # loss cost; no hidden lost-demand or waiting penalty is added.
        elif event_type is LedgerEventType.REQUEST_CANCELLED:
            # Cancellation makes downstream events inactive; it is not another
            # failure charge and therefore leaves all components at zero.
            pass
        else:  # pragma: no cover - Enum construction guards this branch.
            raise UnsupportedLedgerEventError(str(event_type))

        reward = (
            income_reservation
            + income_random
            - charging_cost
            - adjustment_cost
            - reservation_failure_cost
        )
        return LedgerPosting(
            entry=entry,
            income_reservation=income_reservation,
            income_random=income_random,
            charging_cost=charging_cost,
            adjustment_cost=adjustment_cost,
            reservation_failure_cost=reservation_failure_cost,
            reward_delta=reward,
        )

    def _assert_service_interval(self, entry: LedgerEntry) -> None:
        try:
            actual_interval = self.time_grid.interval_of(entry.occurred_at)
        except TimeGridError as exc:
            raise LedgerError(
                f"realised service/timeout time {entry.occurred_at} is outside time grid"
            ) from exc
        if actual_interval != entry.interval:
            raise LedgerError(
                f"entry {entry.event_id} declares interval {entry.interval}, "
                f"but its occurrence belongs to {actual_interval}"
            )

    @staticmethod
    def _explicit_or_price(entry: LedgerEntry, source: PriceSource, name: str) -> float:
        if "amount" in entry.metadata:
            value = entry.metadata["amount"]
        elif entry.amount != 0.0:
            value = entry.amount
        else:
            value = _price_at(source, entry.interval, name, entry.station)
        numeric = float(value)
        if not isfinite(numeric):
            raise LedgerError(f"{name} must be finite")
        return numeric

    def _service_income(self, entry: LedgerEntry, source: PriceSource, name: str) -> float:
        """Compute swap income from actually supplied energy when available.

        Executor entries carry ``return_soc`` and ``battery_capacity_kwh`` so
        the normal path is ``price[station, interval] * E_B * (1-return_soc)``.
        An explicit ``amount`` remains an already-calculated total;
        entries without return-SOC metadata retain flat pricing.
        """

        if "amount" in entry.metadata:
            value = entry.metadata["amount"]
        elif entry.amount != 0.0:
            value = entry.amount
        else:
            price = _price_at(source, entry.interval, name, entry.station)
            energy = entry.metadata.get("service_energy_kwh")
            if energy is None and "return_soc" in entry.metadata:
                return_soc = float(entry.metadata["return_soc"])
                if not 0.0 <= return_soc <= 1.0:
                    raise LedgerError("service return_soc must lie in [0, 1]")
                capacity = entry.metadata.get(
                    "battery_capacity_kwh", self.battery_capacity_kwh
                )
                if capacity is not None:
                    capacity = float(capacity)
                    if not isfinite(capacity) or capacity <= 0:
                        raise LedgerError("service battery_capacity_kwh must be positive")
                    energy = capacity * (1.0 - return_soc)
            value = price if energy is None else price * float(energy)
        numeric = float(value)
        if not isfinite(numeric):
            raise LedgerError(f"{name} must be finite")
        return numeric


__all__ = [
    "DuplicateLedgerEventError",
    "LedgerError",
    "LedgerPosting",
    "RealizedLedger",
    "UnsupportedLedgerEventError",
]
