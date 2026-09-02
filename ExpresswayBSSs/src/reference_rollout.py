"""Reference-path adapter for the continuous reservation event flow.

The day-ahead planner persists a *path* for every accepted reservation.  The
continuous executor, on the other hand, consumes individual
``CandidateRequest`` objects.  This module is the deliberately small boundary
between those two representations:

* it derives continuous station arrival times from ``path_arcs`` (never from
  the old integer ``swap_periods`` field);
* it derives the returned-battery SOC from the real entry SOC for the first
  station and a full departing battery for every later station;
* it gives every reservation visit a stable event id and a one-way upstream
  dependency; and
* it reads no scenario or oracle object.  In particular, ``actual_entry_*``
  fields attached to a day-ahead record are intentionally ignored unless the
  caller explicitly supplies the corresponding record as a visible override.

The adapter is useful before an optimisation call as well as for realised
execution.  It does not choose a new route: an infeasible visible entry update
raises :class:`ReferenceRolloutError`, leaving path re-selection to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from data_generation_test.parameter import (
    ENTRY_NODE,
    EXIT_NODE,
    BusinessParameters,
    NodeId,
)
from src.domain import CandidateRequest, RequestKind, SOC_EPSILON, UserKey
from src.time_grid import TimeGrid


Arc = Tuple[NodeId, NodeId]


class ReferenceRolloutError(ValueError):
    """Raised when a day-ahead path cannot be adapted to a physical event path."""


@dataclass(frozen=True)
class VisibleEntry:
    """The entry state used to materialise one reference reservation path.

    ``source`` is intentionally limited to ``"day_ahead"`` and
    ``"visible_override"``.  It makes downstream logs auditable without
    retaining the raw visible-entry payload (which could otherwise accidentally
    grow to include unrelated external fields).
    """

    entry_time: float
    entry_soc: float
    source: str


@dataclass(frozen=True)
class ReservationEventRollout:
    """Continuous events derived from one accepted day-ahead reservation."""

    reservation_id: str
    od_id: int
    user_key: UserKey
    entry: VisibleEntry
    path_arcs: Tuple[Arc, ...]
    events: Tuple[CandidateRequest, ...]

    @property
    def dependency_map(self) -> Dict[str, List[str]]:
        """Return ``upstream_event_id -> [downstream_event_id]`` for this path."""

        return reservation_dependency_map([self])


def _mapping_value(value: Mapping[str, Any], name: str, default: Any = None) -> Any:
    return value[name] if name in value else default


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceRolloutError(f"{label} must be a finite number") from exc
    if not isfinite(number):
        raise ReferenceRolloutError(f"{label} must be a finite number")
    return number


def _normalise_node(value: Any) -> NodeId:
    if value in (ENTRY_NODE, EXIT_NODE):
        return value
    if isinstance(value, bool):
        raise ReferenceRolloutError("a path node cannot be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ReferenceRolloutError(f"unknown path node {value!r}") from exc
    raise ReferenceRolloutError(f"unsupported path node {value!r}")


def _normalise_arc(value: Any, index: int) -> Arc:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ReferenceRolloutError(f"path_arcs[{index}] must be a two-node sequence")
    return (_normalise_node(value[0]), _normalise_node(value[1]))


def _od_index(params: BusinessParameters, od_id: int) -> int:
    for index, od in enumerate(params.od_pairs):
        if int(od.od_id) == int(od_id):
            return index
    raise ReferenceRolloutError(f"unknown od_id={od_id}")


def _reservation_id(record: Mapping[str, Any]) -> str:
    for name in ("reservation_id", "request_id", "user_id"):
        if name in record and record[name] is not None:
            return str(record[name])
    raise ReferenceRolloutError("reservation record needs reservation_id, request_id, or user_id")


def _user_key(record: Mapping[str, Any], od_id: int, reservation_id: str) -> UserKey:
    raw = record.get("user_key")
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        try:
            key = (int(raw[0]), int(raw[1]))
        except (TypeError, ValueError) as exc:
            raise ReferenceRolloutError("user_key must contain integer od_id and user_id") from exc
        if key[0] != od_id:
            raise ReferenceRolloutError(
                f"user_key O-D {key[0]} does not match reservation od_id={od_id}"
            )
        return key
    try:
        user_id = int(reservation_id)
    except (TypeError, ValueError) as exc:
        raise ReferenceRolloutError(
            "a non-numeric reservation_id requires an explicit user_key"
        ) from exc
    return (od_id, user_id)


def _record_identifiers(record: Mapping[str, Any]) -> set[Tuple[str, str]]:
    """Return identifiers suitable for matching a caller-provided visible entry."""

    identifiers: set[Tuple[str, str]] = set()
    for name in ("reservation_id", "request_id"):
        if name in record and record[name] is not None:
            identifiers.add((name, str(record[name])))
    raw_key = record.get("user_key")
    if isinstance(raw_key, (tuple, list)) and len(raw_key) == 2:
        identifiers.add(("user_key", f"{raw_key[0]}:{raw_key[1]}"))
    return identifiers


def _visible_entry_records(
    visible_entries: Mapping[Any, Any] | Iterable[Mapping[str, Any]] | None,
) -> List[Mapping[str, Any]]:
    """Normalise one override, a keyed mapping, or an iterable of overrides."""

    if visible_entries is None:
        return []
    if isinstance(visible_entries, Mapping):
        # A regular observation entry is itself a mapping with identifiers or
        # entry-state fields.  A keyed mapping has entry records as its values.
        field_names = {
            "reservation_id",
            "request_id",
            "user_key",
            "arrival_time",
            "entry_time",
            "actual_entry_time",
        }
        if any(name in visible_entries for name in field_names):
            return [visible_entries]
        values = list(visible_entries.values())
        if not all(isinstance(value, Mapping) for value in values):
            raise ReferenceRolloutError(
                "a keyed visible_entries mapping must map identifiers to entry mappings"
            )
        return [value for value in values if isinstance(value, Mapping)]
    records = list(visible_entries)
    if not all(isinstance(value, Mapping) for value in records):
        raise ReferenceRolloutError("visible_entries must contain mappings")
    return records


def _find_visible_override(
    record: Mapping[str, Any],
    visible_entries: Mapping[Any, Any] | Iterable[Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    identifiers = _record_identifiers(record)
    if not identifiers:
        return None
    matches: List[Mapping[str, Any]] = []
    for candidate in _visible_entry_records(visible_entries):
        if identifiers.intersection(_record_identifiers(candidate)):
            matches.append(candidate)
    if len(matches) > 1:
        raise ReferenceRolloutError(
            f"multiple visible entry overrides match reservation {_reservation_id(record)!r}"
        )
    return matches[0] if matches else None


def _first_present(record: Mapping[str, Any], names: Sequence[str], label: str) -> float:
    for name in names:
        if name in record and record[name] is not None:
            return _finite(record[name], label)
    raise ReferenceRolloutError(f"{label} is missing")


def resolve_visible_entry(
    reservation_record: Mapping[str, Any],
    visible_entries: Mapping[Any, Any] | Iterable[Mapping[str, Any]] | None = None,
) -> VisibleEntry:
    """Resolve the entry state without consulting hidden scenario truth.

    A supplied visible override may use either the observation schema
    (``arrival_time`` / ``arrival_soc``) or explicit entry fields.  The
    day-ahead record itself is deliberately read only through
    ``day_ahead_entry_*`` (with ``entry_*`` as a legacy fallback); its
    ``actual_entry_*`` fields are never read.
    """

    override = _find_visible_override(reservation_record, visible_entries)
    if override is not None:
        entry_time = _first_present(
            override,
            ("arrival_time", "entry_time", "actual_entry_time"),
            "visible entry time",
        )
        entry_soc = _first_present(
            override,
            ("arrival_soc", "entry_soc", "actual_entry_soc", "return_soc"),
            "visible entry SOC",
        )
        source = "visible_override"
    else:
        entry_time = _first_present(
            reservation_record,
            ("day_ahead_entry_time", "entry_time"),
            "day-ahead entry time",
        )
        entry_soc = _first_present(
            reservation_record,
            ("day_ahead_entry_soc", "entry_soc"),
            "day-ahead entry SOC",
        )
        source = "day_ahead"
    if entry_time < 0.0:
        raise ReferenceRolloutError("entry time must be non-negative")
    if not 0.0 <= entry_soc <= 1.0:
        raise ReferenceRolloutError("entry SOC must lie in [0, 1]")
    return VisibleEntry(entry_time=entry_time, entry_soc=entry_soc, source=source)


def _path_arcs(
    params: BusinessParameters, od_index: int, record: Mapping[str, Any]
) -> Tuple[Arc, ...]:
    raw_arcs = record.get("path_arcs")
    if not isinstance(raw_arcs, Sequence) or isinstance(raw_arcs, (str, bytes)):
        raise ReferenceRolloutError("accepted reservation requires path_arcs")
    arcs = tuple(_normalise_arc(value, index) for index, value in enumerate(raw_arcs))
    if not arcs:
        raise ReferenceRolloutError("accepted reservation path_arcs cannot be empty")
    if arcs[0][0] != ENTRY_NODE or arcs[-1][1] != EXIT_NODE:
        raise ReferenceRolloutError("path_arcs must start at entry and end at exit")
    nodes = set(params.od_nodes(od_index))
    current: NodeId = ENTRY_NODE
    for index, (source, target) in enumerate(arcs):
        if source != current:
            raise ReferenceRolloutError(
                f"path_arcs is disconnected at index {index}: expected source {current!r}, got {source!r}"
            )
        if source not in nodes or target not in nodes:
            raise ReferenceRolloutError(f"path arc {(source, target)!r} is outside the O-D nodes")
        if target == ENTRY_NODE or source == EXIT_NODE:
            raise ReferenceRolloutError(f"path arc {(source, target)!r} reverses an O-D endpoint")
        try:
            distance = params.distance_km(od_index, source, target)
        except (TypeError, ValueError) as exc:
            raise ReferenceRolloutError(f"invalid path arc {(source, target)!r}") from exc
        if distance <= 0.0:
            raise ReferenceRolloutError(f"path arc {(source, target)!r} is not downstream")
        current = target
    return arcs


def _normalise_soc(value: float, *, epsilon: float, label: str) -> float:
    if -epsilon <= value <= 1.0 + epsilon:
        return min(1.0, max(0.0, value))
    raise ReferenceRolloutError(f"{label}={value:.12g} lies outside [0, 1]")


def build_reservation_rollout(
    params: BusinessParameters,
    reservation_record: Mapping[str, Any],
    *,
    visible_entries: Mapping[Any, Any] | Iterable[Mapping[str, Any]] | None = None,
) -> ReservationEventRollout:
    """Materialise the continuous reference events for one accepted reservation.

    The function derives every station arrival from the physical path and
    ``params.vehicle_speed_kmh``.  It intentionally ignores persisted
    ``swap_periods`` and ``return_socs`` because they are old discrete-time
    planning artefacts.  The caller may pass a currently visible entry record
    to override only the entry time/SOC; no future scenario value is requested
    or retained here.
    """

    if reservation_record.get("accepted") is False:
        raise ReferenceRolloutError("cannot build a rollout for a rejected reservation")
    if "od_id" not in reservation_record:
        raise ReferenceRolloutError("reservation record needs od_id")
    od_id = int(reservation_record["od_id"])
    od_index = _od_index(params, od_id)
    reservation_id = _reservation_id(reservation_record)
    user_key = _user_key(reservation_record, od_id, reservation_id)
    entry = resolve_visible_entry(reservation_record, visible_entries)
    arcs = _path_arcs(params, od_index, reservation_record)
    # SOC round-off is a domain tolerance, not the time-grid tolerance used by
    # boundary ownership.  Keeping the two separate prevents a time setting
    # from silently changing physical feasibility.
    epsilon = SOC_EPSILON

    time_at_node = entry.entry_time
    departure_soc = entry.entry_soc
    events: List[CandidateRequest] = []
    previous_event_id: str | None = None

    for source, target in arcs:
        travel_time = params.travel_time_hours(od_index, source, target)
        travel_soc = params.soc_consumption(od_index, source, target)
        time_at_node += travel_time
        arrival_soc = _normalise_soc(
            departure_soc - travel_soc,
            epsilon=epsilon,
            label=f"reservation {reservation_id} arrival SOC at node {target!r}",
        )
        if target == EXIT_NODE:
            if arrival_soc < params.min_exit_soc - epsilon:
                raise ReferenceRolloutError(
                    f"reservation {reservation_id} reaches exit below min_exit_soc"
                )
            continue
        if not isinstance(target, int):  # endpoint validity was checked above.
            raise ReferenceRolloutError(f"unexpected non-station target {target!r}")
        path_order = len(events)
        event_id = CandidateRequest.reservation_event_id(user_key, path_order, target)
        events.append(
            CandidateRequest(
                # Using the event id as request id keeps all consumers on one
                # stable identifier; it also makes retry/cancellation matching
                # independent of Python list order.
                request_id=event_id,
                kind=RequestKind.RESERVATION,
                station=target,
                arrival_time=time_at_node,
                deadline=time_at_node + params.max_wait_hours,
                return_soc=arrival_soc,
                user_key=user_key,
                source_arc=(source, target),
                path_order=path_order,
                event_id=event_id,
                upstream_request_id=previous_event_id,
                active=True,
            )
        )
        previous_event_id = event_id
        # A successful swap supplies a full battery before the next road leg.
        departure_soc = 1.0

    return ReservationEventRollout(
        reservation_id=reservation_id,
        od_id=od_id,
        user_key=user_key,
        entry=entry,
        path_arcs=arcs,
        events=tuple(events),
    )


def _records_from_plan_or_iterable(
    accepted_records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    if isinstance(accepted_records, Mapping):
        if "reservations" in accepted_records:
            records = accepted_records["reservations"]
        else:
            records = [accepted_records]
    else:
        records = list(accepted_records)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ReferenceRolloutError("accepted_records must be reservation records or a plan")
    if not all(isinstance(record, Mapping) for record in records):
        raise ReferenceRolloutError("accepted_records must contain mappings")
    return list(records)


def build_accepted_reservation_rollouts(
    params: BusinessParameters,
    accepted_records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
    *,
    visible_entries: Mapping[Any, Any] | Iterable[Mapping[str, Any]] | None = None,
) -> List[ReservationEventRollout]:
    """Build rollouts for all accepted records in a day-ahead plan or iterable.

    Rejected records are omitted.  Input order is retained so a caller can
    preserve its day-ahead submission audit; use
    :func:`flatten_reservation_events` for a deterministic event-time order.
    """

    rollouts: List[ReservationEventRollout] = []
    for record in _records_from_plan_or_iterable(accepted_records):
        if record.get("accepted") is False:
            continue
        rollouts.append(
            build_reservation_rollout(
                params,
                record,
                visible_entries=visible_entries,
            )
        )
    return rollouts


def flatten_reservation_events(
    rollouts: Iterable[ReservationEventRollout],
) -> List[CandidateRequest]:
    """Flatten rollouts into a stable arrival/event-id order for MPC input."""

    events = [event for rollout in rollouts for event in rollout.events]
    return sorted(events, key=lambda event: (event.arrival_time, event.event_id or ""))


def reservation_dependency_map(
    rollouts: Iterable[ReservationEventRollout],
) -> Dict[str, List[str]]:
    """Return the recursive timeout-cancellation map expected by ``RollingState``."""

    dependencies: Dict[str, List[str]] = {}
    for rollout in rollouts:
        for event in rollout.events:
            if event.upstream_request_id is None or event.event_id is None:
                continue
            children = dependencies.setdefault(event.upstream_request_id, [])
            if event.event_id not in children:
                children.append(event.event_id)
    for children in dependencies.values():
        children.sort()
    return dict(sorted(dependencies.items()))


def events_in_prediction_window(
    events: Iterable[CandidateRequest],
    time_grid: TimeGrid,
    period_ell: int,
    horizon: int,
) -> List[CandidateRequest]:
    """Select arrivals in ``[t_ell, t_(ell+H))`` without a near-end guard band.

    Events exactly at the prediction right endpoint are intentionally excluded
    and will be returned in the next rolling window.  Ordinary values just
    before it remain in the current window.
    """

    start, end = time_grid.prediction_bounds(period_ell, horizon)
    return [event for event in events if start <= event.arrival_time < end]


__all__ = [
    "Arc",
    "ReferenceRolloutError",
    "ReservationEventRollout",
    "VisibleEntry",
    "build_accepted_reservation_rollouts",
    "build_reservation_rollout",
    "events_in_prediction_window",
    "flatten_reservation_events",
    "reservation_dependency_map",
    "resolve_visible_entry",
]
