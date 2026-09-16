"""Deterministic continuous-time battery-swap event executor.

``ContinuousEventEngine`` is intentionally a pure-Python physical kernel.  It
does not know about a MILP incumbent or an RL policy: callers provide requested
slot powers and realised/predicted arrivals, and it advances the actual state
through charging completions, arrivals, services, and deadlines in one fixed
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from src.domain import (
    CandidateRequest,
    DomainError,
    LedgerEntry,
    LedgerEventType,
    PhysicalRequestStatus,
    RequestKind,
    RollingState,
    SlotState,
    WaitingRequest,
)
from src.time_grid import TimeGrid


_TIME_EPSILON = 1e-12
_POWER_EPSILON = 1e-12


class EventEngineError(RuntimeError):
    """Raised when a caller supplies an inconsistent event/execution input."""


@dataclass(frozen=True)
class PowerProjection:
    """Projected per-slot powers and the requested station energy totals."""

    power_kw: List[List[float]]
    station_energy_kwh: List[float]


@dataclass(frozen=True)
class ChargingSegment:
    station: int
    slot: int
    start_time: float
    end_time: float
    power_kw: float
    energy_kwh: float
    soc_start: float
    soc_end: float
    interval: int
    event_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station": self.station,
            "slot": self.slot,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "power_kw": self.power_kw,
            "energy_kwh": self.energy_kwh,
            "soc_start": self.soc_start,
            "soc_end": self.soc_end,
            "interval": self.interval,
            "event_id": self.event_id,
        }


@dataclass(frozen=True)
class ServiceEvent:
    event_id: str
    request: WaitingRequest
    station: int
    slot: int
    occurred_at: float
    wait_hours: float
    interval: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "request_id": self.request.request_id,
            "request_event_id": self.request.event_id,
            "kind": self.request.kind.value,
            "station": self.station,
            "slot": self.slot,
            "occurred_at": self.occurred_at,
            "wait_hours": self.wait_hours,
            "interval": self.interval,
            "return_soc": self.request.return_soc,
        }


@dataclass(frozen=True)
class TimeoutEvent:
    event_id: str
    request: WaitingRequest
    occurred_at: float
    wait_hours: float
    interval: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "request_id": self.request.request_id,
            "request_event_id": self.request.event_id,
            "kind": self.request.kind.value,
            "station": self.request.station,
            "occurred_at": self.occurred_at,
            "wait_hours": self.wait_hours,
            "interval": self.interval,
        }


@dataclass
class ExecutionResult:
    """Output of one interval or a horizon replay.

    ``ledger_entries`` is deliberately empty for a predictive replay.  The
    service/timeout records can still be inspected by reference/MPC code, but
    only entries from a realised execution may be supplied to
    ``RealizedLedger``.
    """

    state: RollingState
    start_time: float
    end_time: float
    charging_segments: List[ChargingSegment] = field(default_factory=list)
    services: List[ServiceEvent] = field(default_factory=list)
    timeouts: List[TimeoutEvent] = field(default_factory=list)
    ledger_entries: List[LedgerEntry] = field(default_factory=list)
    actual_power_kw: List[List[float]] = field(default_factory=list)
    horizon_pending_ids: List[str] = field(default_factory=list)

    @property
    def service_events(self) -> List[ServiceEvent]:
        return self.services

    @property
    def timeout_events(self) -> List[TimeoutEvent]:
        return self.timeouts

    @property
    def ledger(self) -> List[LedgerEntry]:
        return self.ledger_entries

    def extend(self, other: "ExecutionResult") -> None:
        self.charging_segments.extend(other.charging_segments)
        self.services.extend(other.services)
        self.timeouts.extend(other.timeouts)
        self.ledger_entries.extend(other.ledger_entries)
        self.state = other.state
        self.end_time = other.end_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "state": self.state.to_dict(),
            "charging_segments": [item.to_dict() for item in self.charging_segments],
            "services": [item.to_dict() for item in self.services],
            "timeouts": [item.to_dict() for item in self.timeouts],
            "ledger_entries": [item.to_dict() for item in self.ledger_entries],
            "actual_power_kw": self.actual_power_kw,
            "horizon_pending_ids": list(self.horizon_pending_ids),
        }


def _matrix_value(value: Any, station: int, slot: int, default: float) -> float:
    """Read a scalar, station vector, or station/slot matrix limit."""

    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        station_value = value[station]
    except (IndexError, KeyError, TypeError):
        return default
    if isinstance(station_value, (int, float)):
        return float(station_value)
    try:
        return float(station_value[slot])
    except (IndexError, KeyError, TypeError):
        return default


def _station_value(value: Any, station: int, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value[station])
    except (IndexError, KeyError, TypeError):
        return default


def project_requested_power(
    requested_power: Sequence[Sequence[float]] | Mapping[int, Sequence[float]],
    *,
    slot_power_limit_kw: Any = None,
    station_energy_limit_kwh: Any = None,
    interval_hours: float,
    shape: Sequence[int] | None = None,
) -> PowerProjection:
    """Clip slot powers then proportionally enforce each station energy cap.

    The projection is deterministic and only constrains the requested power
    over a complete interval.  The event engine may subsequently deliver less
    energy because a slot fills early, which remains physically feasible.
    """

    if not isfinite(interval_hours) or interval_hours <= 0:
        raise EventEngineError("interval_hours must be a positive finite number")
    if isinstance(requested_power, Mapping):
        if shape is None:
            station_count = (max((int(key) for key in requested_power), default=-1) + 1)
            shape = [len(requested_power.get(index, [])) for index in range(station_count)]
        rows = [list(requested_power.get(index, [])) for index in range(len(shape))]
    else:
        rows = [list(row) for row in requested_power]
        if shape is None:
            shape = [len(row) for row in rows]
    if len(rows) != len(shape):
        raise EventEngineError("requested power station count does not match state shape")

    projected: List[List[float]] = []
    energy: List[float] = []
    for station, expected_slots in enumerate(shape):
        row = rows[station]
        if len(row) != expected_slots:
            raise EventEngineError(
                f"requested power row {station} has {len(row)} slots; expected {expected_slots}"
            )
        clipped: List[float] = []
        for slot, raw_power in enumerate(row):
            try:
                raw = float(raw_power)
            except (TypeError, ValueError) as exc:
                raise EventEngineError("requested power must be numeric") from exc
            if not isfinite(raw):
                raise EventEngineError("requested power must be finite")
            upper = _matrix_value(slot_power_limit_kw, station, slot, float("inf"))
            if not isfinite(upper) and upper != float("inf"):
                raise EventEngineError("slot power limit must be finite or omitted")
            if upper < 0:
                raise EventEngineError("slot power limit must be non-negative")
            clipped.append(min(max(0.0, raw), upper))
        requested_energy = sum(clipped) * interval_hours
        energy_limit = _station_value(station_energy_limit_kwh, station, float("inf"))
        if not isfinite(energy_limit) and energy_limit != float("inf"):
            raise EventEngineError("station energy limit must be finite or omitted")
        if energy_limit < 0:
            raise EventEngineError("station energy limit must be non-negative")
        if requested_energy > energy_limit + _POWER_EPSILON and requested_energy > 0:
            scale = energy_limit / requested_energy
            clipped = [power * scale for power in clipped]
            requested_energy = sum(clipped) * interval_hours
        projected.append(clipped)
        energy.append(requested_energy)
    return PowerProjection(projected, energy)


# Short alias useful to callers that do not need the projection diagnostics.
project_power = project_requested_power


class ContinuousEventEngine:
    """Advance ``RollingState`` through continuous charging and swap events.

    Same-time ordering is fixed and deliberately public:

    ``charging-complete -> arrivals -> reservation-priority allocation -> timeout``.

    This permits a battery that becomes full exactly at a request deadline to
    serve that request, while an event at a rolling right endpoint is deferred
    to the next interval under the grid's half-open convention.
    """

    def __init__(
        self,
        time_grid: TimeGrid,
        battery_capacity_kwh: float,
        charging_efficiency: float = 1.0,
        max_wait_hours: float | None = None,
        *,
        slot_power_limit_kw: Any = None,
        station_energy_limit_kwh: Any = None,
    ) -> None:
        if not isfinite(battery_capacity_kwh) or battery_capacity_kwh <= 0:
            raise EventEngineError("battery_capacity_kwh must be positive")
        if not isfinite(charging_efficiency) or not 0 < charging_efficiency <= 1:
            raise EventEngineError("charging_efficiency must lie in (0, 1]")
        if max_wait_hours is not None and (
            not isfinite(max_wait_hours) or max_wait_hours < 0
        ):
            raise EventEngineError("max_wait_hours must be non-negative when specified")
        self.time_grid = time_grid
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.charging_efficiency = float(charging_efficiency)
        self.max_wait_hours = None if max_wait_hours is None else float(max_wait_hours)
        self.slot_power_limit_kw = slot_power_limit_kw
        self.station_energy_limit_kwh = station_energy_limit_kwh

    def project_power(
        self, requested_power: Any, state: RollingState, interval_index: int | None = None
    ) -> PowerProjection:
        """Apply the configured per-slot and per-station request projection."""

        energy_limit = self.station_energy_limit_kwh
        # Parameters commonly carry [station][interval] energy limits.  Select
        # the requested interval while retaining scalar/vector compatibility.
        if interval_index is not None and isinstance(energy_limit, (list, tuple)):
            selected: List[Any] = []
            for station_value in energy_limit:
                if isinstance(station_value, (list, tuple)):
                    if interval_index >= len(station_value):
                        raise EventEngineError("station energy limit lacks requested interval")
                    selected.append(station_value[interval_index])
                else:
                    selected.append(station_value)
            energy_limit = selected
        return project_requested_power(
            requested_power,
            slot_power_limit_kw=self.slot_power_limit_kw,
            station_energy_limit_kwh=energy_limit,
            interval_hours=self.time_grid.interval_hours,
            shape=[len(row) for row in state.slots],
        )

    def simulate_interval(
        self,
        state: RollingState,
        interval_index: int,
        requested_power: Any,
        arrivals: Iterable[WaitingRequest | CandidateRequest | Mapping[str, Any]] = (),
        *,
        realized: bool = True,
        in_place: bool = False,
        stop_before_end: bool = True,
    ) -> ExecutionResult:
        """Simulate one half-open interval using a constant requested power.

        The input state is not mutated unless ``in_place=True``.  With the
        default ``stop_before_end=True``, events at the right endpoint remain
        for the next call; a slot that fills exactly there stores
        ``completion_due_at`` rather than becoming ready prematurely.
        """

        working = state if in_place else state.clone()
        start, end = self.time_grid.interval(interval_index)
        if abs(working.now - start) > self.time_grid.boundary_tolerance:
            raise EventEngineError(
                f"state.now={working.now} does not equal start of interval {interval_index} ({start})"
            )
        working.now = start
        projection = self.project_power(requested_power, working, interval_index)
        result = ExecutionResult(
            state=working,
            start_time=start,
            end_time=start,
            actual_power_kw=[list(row) for row in projection.power_kw],
        )
        interval_arrivals = self._coerce_arrivals(arrivals, start, end)
        arrivals_by_time: Dict[float, List[WaitingRequest]] = {}
        for request in interval_arrivals:
            arrivals_by_time.setdefault(request.arrival_time, []).append(request)
        for values in arrivals_by_time.values():
            values.sort(key=lambda request: (request.kind.value, request.queue_sort_key))

        # Every interval begins by completing any charge episode intentionally
        # carried at its predecessor's right endpoint.
        self._process_point(
            working,
            start,
            projection.power_kw,
            interval_index,
            arrivals_by_time.pop(start, []),
            result,
            realized,
        )
        current = start

        while current < end:
            next_time = end
            for arrival_time in arrivals_by_time:
                if arrival_time > current and arrival_time < next_time:
                    next_time = arrival_time
            for deadline in self._future_deadlines(working, current):
                if deadline > current and deadline < next_time:
                    next_time = deadline
            for full_time in self._future_completion_times(working, projection.power_kw, current):
                # This time is derived from the charging equation, so it has
                # boundary-calculation provenance.  It may be canonicalised to
                # the interval end for round-off only; external arrivals and
                # deadlines are never globally snapped this way.
                full_time = self.time_grid.normalize_for_window(
                    full_time, end, proven_boundary=True
                )
                if full_time > current and full_time < next_time:
                    next_time = full_time

            if next_time <= current:
                # The only legal same-time events are processed below.  If an
                # external input created an indistinguishable future event,
                # advance it as a deterministic point event rather than loop.
                candidates = [
                    value
                    for value in list(arrivals_by_time)
                    if abs(value - current) <= _TIME_EPSILON
                ]
                for value in candidates:
                    arrivals_by_time.setdefault(current, []).extend(arrivals_by_time.pop(value))
                self._process_point(
                    working,
                    current,
                    projection.power_kw,
                    interval_index,
                    arrivals_by_time.pop(current, []),
                    result,
                    realized,
                )
                # A strictly progressing candidate is required below.  An
                # equality here means an input violated event monotonicity and
                # has already been processed once at this time.
                break

            self._integrate(
                working,
                current,
                next_time,
                projection.power_kw,
                interval_index,
                result,
                realized,
            )
            current = next_time
            working.now = current

            if current == end and stop_before_end:
                self._defer_terminal_completions(working, end)
                break

            self._process_point(
                working,
                current,
                projection.power_kw,
                interval_index,
                arrivals_by_time.pop(current, []),
                result,
                realized,
            )
            if current == end:
                break

        working.now = end
        result.end_time = end
        return result

    def simulate_horizon(
        self,
        state: RollingState,
        start_interval: int,
        horizon: int,
        requested_power: Any,
        arrivals: Iterable[WaitingRequest | CandidateRequest | Mapping[str, Any]] | Mapping[Any, Any] = (),
        *,
        stop_before_end: bool = True,
        realized: bool = False,
        in_place: bool = False,
    ) -> ExecutionResult:
        """Replay a sequence of intervals and derive, but do not realise, pending IDs."""

        if not isinstance(horizon, int) or horizon <= 0:
            raise EventEngineError("horizon must be a positive integer")
        working = state if in_place else state.clone()
        start, end = self.time_grid.prediction_bounds(start_interval, horizon)
        if abs(working.now - start) > self.time_grid.boundary_tolerance:
            raise EventEngineError("state.now must equal start_interval start")
        all_arrivals = self._flatten_arrivals(arrivals)
        aggregate = ExecutionResult(state=working, start_time=start, end_time=start)
        for offset in range(horizon):
            interval_index = start_interval + offset
            step_power = self._power_for_horizon_step(requested_power, offset, working)
            step_arrivals = [
                request
                for request in all_arrivals
                if self.time_grid.contains_execution_time(request.arrival_time, interval_index)
            ]
            step = self.simulate_interval(
                working,
                interval_index,
                step_power,
                step_arrivals,
                realized=realized,
                in_place=True,
                stop_before_end=True if offset < horizon - 1 else stop_before_end,
            )
            aggregate.extend(step)

        # ``all_arrivals`` is normalised to active ``WaitingRequest`` instances;
        # arrival candidates at the right endpoint remain forecast-pending and
        # are not inserted into the physical queue until the next round.
        terminal_ids: set[str] = set()
        for request in all_arrivals:
            if request.arrival_time >= end:
                terminal_ids.add(request.event_id or request.request_id)
        terminal_ids.update(
            request.event_id or request.request_id
            for request in working.all_waiting_requests()
        )
        aggregate.horizon_pending_ids = sorted(terminal_ids)
        aggregate.state = working
        aggregate.end_time = end
        return aggregate

    # ------------------------------------------------------------------
    # State/arrival adapters
    # ------------------------------------------------------------------
    def _coerce_request(
        self, value: WaitingRequest | CandidateRequest | Mapping[str, Any]
    ) -> WaitingRequest:
        if isinstance(value, WaitingRequest):
            return value
        if isinstance(value, CandidateRequest):
            if not value.active:
                raise EventEngineError("inactive CandidateRequest must be filtered before simulation")
            return value.to_waiting_request()
        if not isinstance(value, Mapping):
            raise EventEngineError(f"unsupported arrival type: {type(value)!r}")
        raw = dict(value)
        if not bool(raw.get("active", True)):
            raise EventEngineError("inactive arrival dictionary must be filtered before simulation")
        if "deadline" not in raw:
            if self.max_wait_hours is None:
                raise EventEngineError("arrival lacks deadline and engine has no max_wait_hours")
            raw["deadline"] = float(raw["arrival_time"]) + self.max_wait_hours
        return WaitingRequest.from_dict(raw)

    def _flatten_arrivals(
        self,
        arrivals: Iterable[WaitingRequest | CandidateRequest | Mapping[str, Any]] | Mapping[Any, Any],
    ) -> List[WaitingRequest]:
        flattened: List[WaitingRequest] = []
        if isinstance(arrivals, Mapping):
            # Support a mapping ``interval -> arrivals`` as well as a single
            # request dictionary.  The latter always has request_id.
            if "request_id" in arrivals:
                arrivals = [arrivals]
            else:
                nested: List[Any] = []
                for values in arrivals.values():
                    if isinstance(values, Mapping) and "request_id" in values:
                        nested.append(values)
                    else:
                        nested.extend(values)
                arrivals = nested
        for item in arrivals:
            if isinstance(item, CandidateRequest) and not item.active:
                continue
            if isinstance(item, Mapping) and not bool(item.get("active", True)):
                continue
            flattened.append(self._coerce_request(item))
        seen: set[str] = set()
        for request in flattened:
            identifier = request.event_id or request.request_id
            if identifier in seen:
                raise EventEngineError(f"duplicate arrival event_id: {identifier}")
            seen.add(identifier)
        return flattened

    def _coerce_arrivals(
        self,
        arrivals: Iterable[WaitingRequest | CandidateRequest | Mapping[str, Any]],
        start: float,
        end: float,
    ) -> List[WaitingRequest]:
        output: List[WaitingRequest] = []
        for request in self._flatten_arrivals(arrivals):
            # Never globally snap a near-right-end event.  Strictly-before is
            # current; exact end is left to the next rolling call.
            if start <= request.arrival_time < end:
                output.append(request)
            elif request.arrival_time < start:
                raise EventEngineError(
                    f"arrival {request.event_id} precedes interval start; put carried requests in state queues"
                )
        return output

    def _power_for_horizon_step(self, power: Any, offset: int, state: RollingState) -> List[List[float]]:
        """Accept the project convention [station][slot][h] or a step list."""

        if isinstance(power, Mapping):
            if offset in power:
                return [list(row) for row in power[offset]]
            if str(offset) in power:
                return [list(row) for row in power[str(offset)]]
        if not isinstance(power, Sequence) or isinstance(power, (str, bytes)):
            raise EventEngineError("horizon requested_power must be a sequence or mapping")
        # Common synthetic/MPC shape: [station][slot][h].
        if len(power) == len(state.slots) and all(
            isinstance(row, Sequence) and not isinstance(row, (str, bytes))
            for row in power
        ):
            try:
                return [
                    [float(power[station][slot][offset]) for slot in range(len(state.slots[station]))]
                    for station in range(len(state.slots))
                ]
            except (IndexError, TypeError):
                pass
        # Alternative explicit [h][station][slot] shape.
        if offset < len(power):
            candidate = power[offset]
            if isinstance(candidate, Sequence) and len(candidate) == len(state.slots):
                return [list(row) for row in candidate]
        raise EventEngineError("cannot infer requested_power horizon shape")

    # ------------------------------------------------------------------
    # Event progression
    # ------------------------------------------------------------------
    def _future_deadlines(self, state: RollingState, current: float) -> List[float]:
        return sorted(
            {
                request.deadline
                for request in state.all_waiting_requests()
                if request.deadline > current + _TIME_EPSILON
            }
        )

    def _future_completion_times(
        self, state: RollingState, power: List[List[float]], current: float
    ) -> List[float]:
        times: List[float] = []
        for station, row in enumerate(state.slots):
            for slot, cell in enumerate(row):
                if cell.ready or cell.soc == 1.0:
                    continue
                requested = power[station][slot]
                if requested <= _POWER_EPSILON:
                    continue
                remaining_energy = (1.0 - cell.soc) * self.battery_capacity_kwh
                # Canonicalise the *derived event time* (never an external
                # arrival/deadline) so mathematically equal point events share
                # one timestamp.  ``_integrate`` separately canonicalises the
                # corresponding state to exact SOC 1.
                due = round(
                    current
                    + remaining_energy / (self.charging_efficiency * requested),
                    12,
                )
                if due > current + _TIME_EPSILON:
                    times.append(due)
        return times

    def _process_point(
        self,
        state: RollingState,
        now: float,
        power: List[List[float]],
        interval_index: int,
        arrivals: Iterable[WaitingRequest],
        result: ExecutionResult,
        realized: bool,
    ) -> None:
        self._mark_completed_slots(state, now)
        # Historical deadlines are defensive clean-up only.  Normal interval
        # simulation never leaves one behind, and deadline==now remains
        # eligible for the ordering below.
        self._timeout_requests_before(state, now, interval_index, result, realized)
        for request in arrivals:
            if self._is_request_cancelled(state, request):
                self._set_status(state, request, PhysicalRequestStatus.CANCELLED)
                continue
            if state.find_waiting(request.event_id or request.request_id) is not None:
                raise EventEngineError(f"arrival already waiting: {request.event_id}")
            state.add_waiting(request)
        self._serve_ready_slots(state, now, interval_index, result, realized)
        self._timeout_requests_at(state, now, interval_index, result, realized)

    def _mark_completed_slots(self, state: RollingState, now: float) -> None:
        for row in state.slots:
            for slot in row:
                due = slot.completion_due_at
                should_complete = (
                    due is not None and due <= now + _TIME_EPSILON
                ) or (not slot.ready and slot.soc == 1.0)
                if should_complete:
                    slot.soc = 1.0
                    slot.ready = True
                    slot.completion_due_at = None
                    slot.last_update_time = now

    def _timeout_requests_before(
        self,
        state: RollingState,
        now: float,
        interval_index: int,
        result: ExecutionResult,
        realized: bool,
    ) -> None:
        overdue = [
            request
            for request in state.all_waiting_requests()
            if request.deadline < now - _TIME_EPSILON
        ]
        for request in overdue:
            self._record_timeout(state, request, request.deadline, interval_index, result, realized)

    def _timeout_requests_at(
        self,
        state: RollingState,
        now: float,
        interval_index: int,
        result: ExecutionResult,
        realized: bool,
    ) -> None:
        due = [
            request
            for request in state.all_waiting_requests()
            if abs(request.deadline - now) <= _TIME_EPSILON
        ]
        for request in due:
            self._record_timeout(state, request, now, interval_index, result, realized)

    def _serve_ready_slots(
        self,
        state: RollingState,
        now: float,
        interval_index: int,
        result: ExecutionResult,
        realized: bool,
    ) -> None:
        while True:
            ready = self._smallest_ready_slot(state)
            if ready is None:
                return
            request = self._next_waiting_request(state, ready.station)
            if request is None:
                return
            popped = state.pop_waiting(request.event_id or request.request_id)
            if popped is None:
                raise EventEngineError("queue changed while assigning a ready slot")
            ready.soc = popped.return_soc
            ready.ready = popped.return_soc == 1.0
            ready.completion_due_at = None
            ready.last_update_time = now
            self._set_status(state, popped, PhysicalRequestStatus.SERVED)
            service_id = f"service:{popped.event_id}"
            record = ServiceEvent(
                event_id=service_id,
                request=popped,
                station=ready.station,
                slot=ready.slot,
                occurred_at=now,
                wait_hours=max(0.0, now - popped.arrival_time),
                interval=interval_index,
            )
            result.services.append(record)
            if realized:
                event_type = (
                    LedgerEventType.RESERVATION_SERVICE
                    if popped.kind is RequestKind.RESERVATION
                    else LedgerEventType.RANDOM_SERVICE
                )
                self._append_ledger_entry(
                    state,
                    result,
                    LedgerEntry(
                        event_id=service_id,
                        event_type=event_type,
                        occurred_at=now,
                        interval=interval_index,
                        station=ready.station,
                        slot=ready.slot,
                        request_id=popped.request_id,
                        user_key=popped.user_key,
                        arrival_time=popped.arrival_time,
                        deadline=popped.deadline,
                        metadata={
                            "request_event_id": popped.event_id,
                            "return_soc": popped.return_soc,
                            "battery_capacity_kwh": self.battery_capacity_kwh,
                            "wait_hours": record.wait_hours,
                        },
                    )
                )

    def _smallest_ready_slot(self, state: RollingState) -> SlotState | None:
        stations_with_waiting = {
            station
            for station in range(state.num_stations)
            if self._next_waiting_request(state, station) is not None
        }
        candidates = [
            cell
            for row in state.slots
            for cell in row
            if (
                cell.station in stations_with_waiting
                and cell.ready
                and cell.soc == 1.0
            )
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda cell: (cell.station, cell.slot))

    def _next_waiting_request(self, state: RollingState, station: int) -> WaitingRequest | None:
        reservations = state.queue_for(station, RequestKind.RESERVATION)
        if reservations:
            return reservations[0]
        randoms = state.queue_for(station, RequestKind.RANDOM)
        return randoms[0] if randoms else None

    def _record_timeout(
        self,
        state: RollingState,
        request: WaitingRequest,
        occurred_at: float,
        interval_index: int,
        result: ExecutionResult,
        realized: bool,
    ) -> None:
        popped = state.pop_waiting(request.event_id or request.request_id)
        if popped is None:
            return
        self._set_status(state, popped, PhysicalRequestStatus.TIMED_OUT)
        timeout_id = f"timeout:{popped.event_id}"
        wait_hours = max(0.0, occurred_at - popped.arrival_time)
        record = TimeoutEvent(
            event_id=timeout_id,
            request=popped,
            occurred_at=occurred_at,
            wait_hours=wait_hours,
            interval=interval_index,
        )
        result.timeouts.append(record)
        if realized:
            event_type = (
                LedgerEventType.RESERVATION_TIMEOUT
                if popped.kind is RequestKind.RESERVATION
                else LedgerEventType.RANDOM_TIMEOUT
            )
            self._append_ledger_entry(
                state,
                result,
                LedgerEntry(
                    event_id=timeout_id,
                    event_type=event_type,
                    occurred_at=occurred_at,
                    interval=interval_index,
                    station=popped.station,
                    request_id=popped.request_id,
                    user_key=popped.user_key,
                    arrival_time=popped.arrival_time,
                    deadline=popped.deadline,
                    metadata={
                        "request_event_id": popped.event_id,
                        "wait_hours": wait_hours,
                    },
                )
            )
        if popped.kind is RequestKind.RESERVATION:
            self.cancel_downstream_requests(state, popped.event_id or popped.request_id)

    def _set_status(
        self, state: RollingState, request: WaitingRequest, status: PhysicalRequestStatus
    ) -> None:
        state.request_status[request.event_id or request.request_id] = status
        # ``request_id`` is a convenient compatibility alias.  Event IDs remain
        # authoritative and should be used where duplicate IDs are possible.
        state.request_status.setdefault(request.request_id, status)

    def _is_request_cancelled(self, state: RollingState, request: WaitingRequest) -> bool:
        if request.upstream_request_id is None:
            return False
        status = state.request_status.get(request.upstream_request_id)
        return status in (PhysicalRequestStatus.TIMED_OUT, PhysicalRequestStatus.CANCELLED)

    def cancel_downstream_requests(self, state: RollingState, upstream_event_id: str) -> List[str]:
        """Deactivate dependent reservation events after an upstream timeout.

        Cancellation is a physical-state update, not an additional failure
        charge.  It is idempotent and recursively removes both waiting and
        future/enroute request references.
        """

        cancelled: List[str] = []
        pending = list(state.reservation_dependencies.get(upstream_event_id, []))
        # Dependencies may be keyed by request ID rather than event ID;
        # accepting both preserves the one-way cancellation semantics.
        if ":" in upstream_event_id:
            pending.extend(state.reservation_dependencies.get(upstream_event_id.rsplit(":", 1)[-1], []))
        seen: set[str] = set()
        while pending:
            child = pending.pop(0)
            if child in seen:
                continue
            seen.add(child)
            request = state.pop_waiting(child)
            if request is not None:
                self._set_status(state, request, PhysicalRequestStatus.CANCELLED)
            else:
                state.request_status[child] = PhysicalRequestStatus.CANCELLED
            cancelled.append(child)
            pending.extend(state.reservation_dependencies.get(child, []))
            for collection in (state.future, state.enroute):
                for reservation in collection.values():
                    if reservation.waiting_request_id == child:
                        reservation.waiting_request_id = None
        return cancelled

    def _integrate(
        self,
        state: RollingState,
        start: float,
        end: float,
        power: List[List[float]],
        interval_index: int,
        result: ExecutionResult,
        realized: bool,
    ) -> None:
        duration = end - start
        if duration <= 0.0:
            return
        for station, row in enumerate(state.slots):
            for slot_index, cell in enumerate(row):
                requested = power[station][slot_index]
                if cell.ready or cell.soc == 1.0 or requested <= _POWER_EPSILON:
                    cell.last_update_time = end
                    continue
                soc_start = cell.soc
                energy = requested * duration
                soc_increase = self.charging_efficiency * energy / self.battery_capacity_kwh
                completion_time = round(
                    start
                    + (1.0 - soc_start)
                    * self.battery_capacity_kwh
                    / (self.charging_efficiency * requested),
                    12,
                )
                # This comparison is event provenance, not a relaxed SOC
                # threshold: ``end`` is the exact completion candidate emitted
                # by ``_future_completion_times``.  Canonicalising its state to
                # 1 preserves the continuous integral despite floating-point
                # addition/subtraction round-off.
                if end >= completion_time:
                    cell.soc = 1.0
                else:
                    cell.soc = min(1.0, cell.soc + soc_increase)
                cell.last_update_time = end
                if energy > _POWER_EPSILON:
                    event_id = self._charging_event_id(
                        interval_index, station, slot_index, start, end, len(result.charging_segments)
                    )
                    segment = ChargingSegment(
                        station=station,
                        slot=slot_index,
                        start_time=start,
                        end_time=end,
                        power_kw=requested,
                        energy_kwh=energy,
                        soc_start=soc_start,
                        soc_end=cell.soc,
                        interval=interval_index,
                        event_id=event_id,
                    )
                    result.charging_segments.append(segment)
                    if realized:
                        self._append_ledger_entry(
                            state,
                            result,
                            LedgerEntry(
                                event_id=event_id,
                                event_type=LedgerEventType.CHARGING,
                                occurred_at=end,
                                interval=interval_index,
                                energy_kwh=energy,
                                station=station,
                                slot=slot_index,
                                metadata={
                                    "start_time": start,
                                    "end_time": end,
                                    "power_kw": requested,
                                    "soc_start": soc_start,
                                    "soc_end": cell.soc,
                                },
                            )
                        )

    def _defer_terminal_completions(self, state: RollingState, terminal: float) -> None:
        for row in state.slots:
            for slot in row:
                if not slot.ready and slot.soc == 1.0:
                    slot.soc = 1.0
                    slot.ready = False
                    slot.completion_due_at = terminal
                    slot.last_update_time = terminal

    @staticmethod
    def _append_ledger_entry(
        state: RollingState, result: ExecutionResult, entry: LedgerEntry
    ) -> None:
        """Attach one realised event and retain its all-day id in state."""

        if entry.event_id in state.accounted_event_ids:
            raise EventEngineError(f"realised event already accounted: {entry.event_id}")
        state.accounted_event_ids.add(entry.event_id)
        result.ledger_entries.append(entry)

    @staticmethod
    def _charging_event_id(
        interval: int, station: int, slot: int, start: float, end: float, ordinal: int
    ) -> str:
        return (
            f"charge:{interval}:{station}:{slot}:"
            f"{start:.12g}:{end:.12g}:{ordinal}"
        )


__all__ = [
    "ChargingSegment",
    "ContinuousEventEngine",
    "EventEngineError",
    "ExecutionResult",
    "PowerProjection",
    "ServiceEvent",
    "TimeoutEvent",
    "project_power",
    "project_requested_power",
]
