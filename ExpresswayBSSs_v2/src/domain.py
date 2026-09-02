"""Domain objects shared by the continuous event executor and its callers.

The module is deliberately free of Gurobi, Gym, and RL dependencies.  It is
the serialisable state boundary between a rolling optimiser and the realised
physical executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


UserKey = Tuple[int, int]
SOC_EPSILON = 1e-9


class DomainError(ValueError):
    """Raised when a domain object violates a physical-state invariant."""


class RequestKind(str, Enum):
    RESERVATION = "reservation"
    RANDOM = "random"


class PhysicalRequestStatus(str, Enum):
    """Statuses that may occur in the realised rolling state."""

    WAITING = "waiting"
    SERVED = "served"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class PredictedOutcome(str, Enum):
    """Ephemeral result labels; never valid realised ledger events."""

    SERVED_IN_HORIZON = "served_in_horizon"
    FAILED_IN_HORIZON = "failed_in_horizon"
    PENDING_AT_HORIZON = "pending_at_horizon"


class LedgerEventType(str, Enum):
    CHARGING = "charging"
    RESERVATION_SERVICE = "reservation_service"
    RANDOM_SERVICE = "random_service"
    RESERVATION_TIMEOUT = "reservation_timeout"
    RANDOM_TIMEOUT = "random_timeout"
    PATH_PUBLISHED = "path_published"
    REQUEST_CANCELLED = "request_cancelled"


def _finite(value: float, name: str) -> float:
    numeric = float(value)
    if not isfinite(numeric):
        raise DomainError(f"{name} must be finite")
    return numeric


def _user_key(value: UserKey | List[int] | Tuple[int, int] | None) -> UserKey | None:
    if value is None:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise DomainError("user_key must be a pair (od_id, user_id)")
    return (int(value[0]), int(value[1]))


def _json_value(value: Any) -> Any:
    """Convert nested enum/tuple values to JSON-compatible primitives."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class WaitingRequest:
    """An arrived request waiting for a fully charged battery.

    Request arrival and deadline are immutable identity facts.  The mutable
    status belongs to :class:`RollingState`, avoiding accidental deadline
    resets when a request crosses an MPC boundary.
    """

    request_id: str
    kind: RequestKind | str
    station: int
    arrival_time: float
    deadline: float
    return_soc: float
    user_key: UserKey | None = None
    source_arc: Any = None
    path_order: int = 0
    event_id: str | None = None
    upstream_request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise DomainError("request_id must be non-empty")
        try:
            kind = self.kind if isinstance(self.kind, RequestKind) else RequestKind(self.kind)
        except ValueError as exc:
            raise DomainError(f"unknown request kind: {self.kind!r}") from exc
        if int(self.station) < 0:
            raise DomainError("station must be non-negative")
        arrival = _finite(self.arrival_time, "arrival_time")
        deadline = _finite(self.deadline, "deadline")
        if deadline < arrival:
            raise DomainError("deadline must not precede arrival_time")
        return_soc = _finite(self.return_soc, "return_soc")
        if not 0.0 <= return_soc <= 1.0:
            raise DomainError("return_soc must lie in [0, 1]")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "station", int(self.station))
        object.__setattr__(self, "arrival_time", arrival)
        object.__setattr__(self, "deadline", deadline)
        object.__setattr__(self, "return_soc", return_soc)
        object.__setattr__(self, "user_key", _user_key(self.user_key))
        object.__setattr__(self, "path_order", int(self.path_order))
        if self.event_id is None:
            prefix = "reservation" if kind is RequestKind.RESERVATION else "random"
            derived = f"{prefix}:{self.station}:{self.request_id}"
            object.__setattr__(self, "event_id", derived)

    @property
    def queue_sort_key(self) -> Tuple[float, str, int]:
        """Stable FCFS ordering within one request class."""

        return (self.arrival_time, self.request_id, self.path_order)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "kind": self.kind.value,
            "station": self.station,
            "arrival_time": self.arrival_time,
            "deadline": self.deadline,
            "return_soc": self.return_soc,
            "user_key": list(self.user_key) if self.user_key is not None else None,
            "source_arc": _json_value(self.source_arc),
            "path_order": self.path_order,
            "event_id": self.event_id,
            "upstream_request_id": self.upstream_request_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WaitingRequest":
        return cls(
            request_id=str(data["request_id"]),
            kind=data["kind"],
            station=int(data["station"]),
            arrival_time=float(data["arrival_time"]),
            deadline=float(data["deadline"]),
            return_soc=float(data.get("return_soc", 0.0)),
            user_key=data.get("user_key"),
            source_arc=data.get("source_arc"),
            path_order=int(data.get("path_order", 0)),
            event_id=data.get("event_id"),
            upstream_request_id=data.get("upstream_request_id"),
        )


@dataclass(frozen=True)
class CandidateRequest:
    """A deterministic model/forecast candidate that can become a request.

    It mirrors ``WaitingRequest`` rather than containing mutable queue state,
    allowing both dataclasses and JSON dictionaries to use the same adapter.
    Reservation event IDs are stable across windows through
    :meth:`reservation_event_id`.
    """

    request_id: str
    kind: RequestKind | str
    station: int
    arrival_time: float
    deadline: float
    return_soc: float
    user_key: UserKey | None = None
    source_arc: Any = None
    path_order: int = 0
    event_id: str | None = None
    upstream_request_id: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        normalized_kind = (
            self.kind if isinstance(self.kind, RequestKind) else RequestKind(self.kind)
        )
        normalized_user_key = _user_key(self.user_key)
        event_id = self.event_id
        if (
            event_id is None
            and normalized_kind is RequestKind.RESERVATION
            and normalized_user_key is not None
        ):
            event_id = self.reservation_event_id(
                normalized_user_key, self.path_order, self.station
            )
        waiting = WaitingRequest(
            request_id=self.request_id,
            kind=normalized_kind,
            station=self.station,
            arrival_time=self.arrival_time,
            deadline=self.deadline,
            return_soc=self.return_soc,
            user_key=normalized_user_key,
            source_arc=self.source_arc,
            path_order=self.path_order,
            event_id=event_id,
            upstream_request_id=self.upstream_request_id,
        )
        object.__setattr__(self, "kind", waiting.kind)
        object.__setattr__(self, "station", waiting.station)
        object.__setattr__(self, "arrival_time", waiting.arrival_time)
        object.__setattr__(self, "deadline", waiting.deadline)
        object.__setattr__(self, "return_soc", waiting.return_soc)
        object.__setattr__(self, "user_key", waiting.user_key)
        object.__setattr__(self, "path_order", waiting.path_order)
        object.__setattr__(self, "event_id", waiting.event_id)

    @staticmethod
    def reservation_event_id(
        user_key: UserKey, path_order: int, station: int
    ) -> str:
        """Return the stable ``(p, k, j, i)`` reservation event identifier."""

        od_id, user_id = _user_key(user_key)  # type: ignore[misc]
        return f"reservation:{od_id}:{user_id}:{int(path_order)}:{int(station)}"

    def to_waiting_request(self) -> WaitingRequest:
        return WaitingRequest(
            request_id=self.request_id,
            kind=self.kind,
            station=self.station,
            arrival_time=self.arrival_time,
            deadline=self.deadline,
            return_soc=self.return_soc,
            user_key=self.user_key,
            source_arc=self.source_arc,
            path_order=self.path_order,
            event_id=self.event_id,
            upstream_request_id=self.upstream_request_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        result = self.to_waiting_request().to_dict()
        result["active"] = self.active
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CandidateRequest":
        return cls(
            request_id=str(data["request_id"]),
            kind=data["kind"],
            station=int(data["station"]),
            arrival_time=float(data["arrival_time"]),
            deadline=float(data["deadline"]),
            return_soc=float(data.get("return_soc", 0.0)),
            user_key=data.get("user_key"),
            source_arc=data.get("source_arc"),
            path_order=int(data.get("path_order", 0)),
            event_id=data.get("event_id"),
            upstream_request_id=data.get("upstream_request_id"),
            active=bool(data.get("active", True)),
        )


@dataclass
class SlotState:
    """Physical state of one inventory slot (not a permanently named battery)."""

    station: int
    slot: int
    soc: float
    ready: bool | None = None
    completion_due_at: float | None = None
    last_update_time: float = 0.0

    def __post_init__(self) -> None:
        self.station = int(self.station)
        self.slot = int(self.slot)
        if self.station < 0 or self.slot < 0:
            raise DomainError("station and slot must be non-negative")
        self.soc = _finite(self.soc, "soc")
        if not -SOC_EPSILON <= self.soc <= 1.0 + SOC_EPSILON:
            raise DomainError("soc must lie in [0, 1]")
        self.soc = min(1.0, max(0.0, self.soc))
        self.last_update_time = _finite(self.last_update_time, "last_update_time")
        if self.completion_due_at is not None:
            self.completion_due_at = _finite(self.completion_due_at, "completion_due_at")
        if self.ready is None:
            # ``SOC_EPSILON`` validates harmless serialization round-off only;
            # it is not an operational full-battery threshold.  main.tex
            # requires service readiness exactly at the normalized target 1.
            self.ready = self.soc == 1.0 and self.completion_due_at is None
        else:
            self.ready = bool(self.ready)
        if self.ready and self.soc != 1.0:
            raise DomainError("a ready slot must contain a battery at exact target SOC 1")

    def copy(self) -> "SlotState":
        return SlotState.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "station": self.station,
            "slot": self.slot,
            "soc": self.soc,
            "ready": bool(self.ready),
            "completion_due_at": self.completion_due_at,
            "last_update_time": self.last_update_time,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SlotState":
        return cls(
            station=int(data["station"]),
            slot=int(data["slot"]),
            soc=float(data["soc"]),
            ready=data.get("ready"),
            completion_due_at=data.get("completion_due_at"),
            last_update_time=float(data.get("last_update_time", 0.0)),
        )


@dataclass
class EnrouteReservation:
    """Known real-time state for a reservation that has not finished."""

    user_key: UserKey
    current_position: float = 0.0
    vehicle_soc: float = 0.0
    executed_station_prefix: List[int] = field(default_factory=list)
    dayahead_initial_path: List[Any] = field(default_factory=list)
    last_published_remaining_path: List[Any] = field(default_factory=list)
    last_actual_swap_station: int | None = None
    known_eta: Dict[str, float] = field(default_factory=dict)
    waiting_request_id: str | None = None

    def __post_init__(self) -> None:
        self.user_key = _user_key(self.user_key)  # type: ignore[assignment]
        if self.user_key is None:
            raise DomainError("EnrouteReservation requires a user_key")
        self.current_position = _finite(self.current_position, "current_position")
        self.vehicle_soc = _finite(self.vehicle_soc, "vehicle_soc")
        if not 0.0 <= self.vehicle_soc <= 1.0:
            raise DomainError("vehicle_soc must lie in [0, 1]")
        self.executed_station_prefix = [int(item) for item in self.executed_station_prefix]
        self.known_eta = {str(key): _finite(value, "known_eta") for key, value in self.known_eta.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_key": list(self.user_key),
            "current_position": self.current_position,
            "vehicle_soc": self.vehicle_soc,
            "executed_station_prefix": list(self.executed_station_prefix),
            "dayahead_initial_path": _json_value(self.dayahead_initial_path),
            "last_published_remaining_path": _json_value(self.last_published_remaining_path),
            "last_actual_swap_station": self.last_actual_swap_station,
            "known_eta": dict(self.known_eta),
            "waiting_request_id": self.waiting_request_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnrouteReservation":
        return cls(
            user_key=data["user_key"],
            current_position=float(data.get("current_position", 0.0)),
            vehicle_soc=float(data.get("vehicle_soc", 0.0)),
            executed_station_prefix=list(data.get("executed_station_prefix", [])),
            dayahead_initial_path=list(data.get("dayahead_initial_path", [])),
            last_published_remaining_path=list(data.get("last_published_remaining_path", [])),
            last_actual_swap_station=data.get("last_actual_swap_station"),
            known_eta=dict(data.get("known_eta", {})),
            waiting_request_id=data.get("waiting_request_id"),
        )


def _empty_station_queues() -> Dict[RequestKind, List[WaitingRequest]]:
    return {RequestKind.RESERVATION: [], RequestKind.RANDOM: []}


@dataclass
class RollingState:
    """Serializable realised state carried from one rolling interval to another."""

    now: float
    slots: List[List[SlotState]]
    future: Dict[str, EnrouteReservation] = field(default_factory=dict)
    enroute: Dict[str, EnrouteReservation] = field(default_factory=dict)
    waiting_queues: Dict[int, Dict[RequestKind, List[WaitingRequest]]] = field(
        default_factory=dict
    )
    accounted_event_ids: set[str] = field(default_factory=set)
    latest_random_history: List[Dict[str, Any]] = field(default_factory=list)
    request_status: Dict[str, PhysicalRequestStatus] = field(default_factory=dict)
    reservation_dependencies: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.now = _finite(self.now, "now")
        converted_slots: List[List[SlotState]] = []
        for station, row in enumerate(self.slots):
            converted_row: List[SlotState] = []
            for slot, value in enumerate(row):
                cell = value if isinstance(value, SlotState) else SlotState.from_dict(value)
                if cell.station != station or cell.slot != slot:
                    raise DomainError("slot matrix indices must match SlotState.station/slot")
                converted_row.append(cell)
            converted_slots.append(converted_row)
        self.slots = converted_slots
        self.future = {
            str(key): value if isinstance(value, EnrouteReservation) else EnrouteReservation.from_dict(value)
            for key, value in self.future.items()
        }
        self.enroute = {
            str(key): value if isinstance(value, EnrouteReservation) else EnrouteReservation.from_dict(value)
            for key, value in self.enroute.items()
        }
        converted_queues: Dict[int, Dict[RequestKind, List[WaitingRequest]]] = {}
        for raw_station, raw_queues in self.waiting_queues.items():
            station = int(raw_station)
            queues = _empty_station_queues()
            for raw_kind, requests in raw_queues.items():
                kind = raw_kind if isinstance(raw_kind, RequestKind) else RequestKind(raw_kind)
                queues[kind] = [
                    item if isinstance(item, WaitingRequest) else WaitingRequest.from_dict(item)
                    for item in requests
                ]
                if any(item.station != station for item in queues[kind]):
                    raise DomainError("queue station does not match request.station")
                queues[kind].sort(key=lambda item: item.queue_sort_key)
            converted_queues[station] = queues
        self.waiting_queues = converted_queues
        self.accounted_event_ids = {str(item) for item in self.accounted_event_ids}
        self.request_status = {
            str(key): value if isinstance(value, PhysicalRequestStatus) else PhysicalRequestStatus(value)
            for key, value in self.request_status.items()
        }
        self.reservation_dependencies = {
            str(key): [str(child) for child in children]
            for key, children in self.reservation_dependencies.items()
        }

    @property
    def slot_states(self) -> List[List[SlotState]]:
        """Alias used by callers that prefer an explicit state name."""

        return self.slots

    @property
    def num_stations(self) -> int:
        return len(self.slots)

    def queue_for(self, station: int, kind: RequestKind | str) -> List[WaitingRequest]:
        station = int(station)
        normalized_kind = kind if isinstance(kind, RequestKind) else RequestKind(kind)
        queues = self.waiting_queues.setdefault(station, _empty_station_queues())
        return queues.setdefault(normalized_kind, [])

    def add_waiting(self, request: WaitingRequest) -> None:
        queue = self.queue_for(request.station, request.kind)
        if any(item.event_id == request.event_id for item in self.all_waiting_requests()):
            raise DomainError(f"request event already waiting: {request.event_id}")
        queue.append(request)
        queue.sort(key=lambda item: item.queue_sort_key)
        self.request_status[request.event_id or request.request_id] = PhysicalRequestStatus.WAITING

    def all_waiting_requests(self) -> List[WaitingRequest]:
        requests: List[WaitingRequest] = []
        for station in sorted(self.waiting_queues):
            for kind in (RequestKind.RESERVATION, RequestKind.RANDOM):
                requests.extend(self.waiting_queues[station].get(kind, []))
        return requests

    def pop_waiting(self, event_id: str) -> WaitingRequest | None:
        for station in sorted(self.waiting_queues):
            for kind in (RequestKind.RESERVATION, RequestKind.RANDOM):
                queue = self.waiting_queues[station].get(kind, [])
                for index, request in enumerate(queue):
                    if request.event_id == event_id:
                        return queue.pop(index)
        return None

    def find_waiting(self, event_id: str) -> WaitingRequest | None:
        for request in self.all_waiting_requests():
            if request.event_id == event_id:
                return request
        return None

    def clone(self) -> "RollingState":
        return RollingState.from_dict(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "now": self.now,
            "slots": [[cell.to_dict() for cell in row] for row in self.slots],
            "future": {key: value.to_dict() for key, value in sorted(self.future.items())},
            "enroute": {key: value.to_dict() for key, value in sorted(self.enroute.items())},
            "waiting_queues": {
                str(station): {
                    kind.value: [request.to_dict() for request in queues.get(kind, [])]
                    for kind in (RequestKind.RESERVATION, RequestKind.RANDOM)
                }
                for station, queues in sorted(self.waiting_queues.items())
            },
            "accounted_event_ids": sorted(self.accounted_event_ids),
            "latest_random_history": _json_value(self.latest_random_history),
            "request_status": {
                key: value.value for key, value in sorted(self.request_status.items())
            },
            "reservation_dependencies": {
                key: list(value) for key, value in sorted(self.reservation_dependencies.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RollingState":
        raw_queues = data.get("waiting_queues", {})
        queues: Dict[int, Dict[RequestKind, List[WaitingRequest]]] = {}
        for raw_station, raw_kinds in raw_queues.items():
            queues[int(raw_station)] = {
                RequestKind(kind): [WaitingRequest.from_dict(item) for item in requests]
                for kind, requests in raw_kinds.items()
            }
        return cls(
            now=float(data["now"]),
            slots=[
                [SlotState.from_dict(cell) for cell in row]
                for row in data.get("slots", data.get("slot_states", []))
            ],
            future={
                str(key): EnrouteReservation.from_dict(value)
                for key, value in data.get("future", {}).items()
            },
            enroute={
                str(key): EnrouteReservation.from_dict(value)
                for key, value in data.get("enroute", {}).items()
            },
            waiting_queues=queues,
            accounted_event_ids=set(data.get("accounted_event_ids", [])),
            latest_random_history=list(data.get("latest_random_history", [])),
            request_status={
                str(key): PhysicalRequestStatus(value)
                for key, value in data.get("request_status", {}).items()
            },
            reservation_dependencies={
                str(key): list(value)
                for key, value in data.get("reservation_dependencies", {}).items()
            },
        )


@dataclass(frozen=True)
class LedgerEntry:
    """One unique realised event submitted to :class:`RealizedLedger`."""

    event_id: str
    event_type: LedgerEventType | str
    occurred_at: float
    interval: int
    amount: float = 0.0
    energy_kwh: float = 0.0
    station: int | None = None
    slot: int | None = None
    request_id: str | None = None
    user_key: UserKey | None = None
    arrival_time: float | None = None
    deadline: float | None = None
    realized: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise DomainError("ledger event_id must be non-empty")
        try:
            event_type = (
                self.event_type
                if isinstance(self.event_type, LedgerEventType)
                else LedgerEventType(self.event_type)
            )
        except ValueError as exc:
            raise DomainError(f"unsupported ledger event type: {self.event_type!r}") from exc
        if int(self.interval) < 0:
            raise DomainError("ledger interval must be non-negative")
        occurred_at = _finite(self.occurred_at, "occurred_at")
        energy = _finite(self.energy_kwh, "energy_kwh")
        if energy < -SOC_EPSILON:
            raise DomainError("energy_kwh must be non-negative")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "interval", int(self.interval))
        object.__setattr__(self, "occurred_at", occurred_at)
        object.__setattr__(self, "amount", _finite(self.amount, "amount"))
        object.__setattr__(self, "energy_kwh", max(0.0, energy))
        object.__setattr__(self, "user_key", _user_key(self.user_key))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at,
            "interval": self.interval,
            "amount": self.amount,
            "energy_kwh": self.energy_kwh,
            "station": self.station,
            "slot": self.slot,
            "request_id": self.request_id,
            "user_key": list(self.user_key) if self.user_key is not None else None,
            "arrival_time": self.arrival_time,
            "deadline": self.deadline,
            "realized": self.realized,
            "metadata": _json_value(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LedgerEntry":
        return cls(
            event_id=str(data["event_id"]),
            event_type=data["event_type"],
            occurred_at=float(data["occurred_at"]),
            interval=int(data["interval"]),
            amount=float(data.get("amount", 0.0)),
            energy_kwh=float(data.get("energy_kwh", 0.0)),
            station=data.get("station"),
            slot=data.get("slot"),
            request_id=data.get("request_id"),
            user_key=data.get("user_key"),
            arrival_time=data.get("arrival_time"),
            deadline=data.get("deadline"),
            realized=bool(data.get("realized", True)),
            metadata=dict(data.get("metadata", {})),
        )


__all__ = [
    "CandidateRequest",
    "DomainError",
    "EnrouteReservation",
    "LedgerEntry",
    "LedgerEventType",
    "PhysicalRequestStatus",
    "PredictedOutcome",
    "RequestKind",
    "RollingState",
    "SOC_EPSILON",
    "SlotState",
    "UserKey",
    "WaitingRequest",
]
