"""Deterministic remaining-path candidates for the Mock event controller.

The module contains no solver and does not inspect scenario future truth.  It
builds paths from the online remaining network, then materialises their future
reservation events from the currently observed vehicle position/SOC.  The
caller is responsible for scoring candidates with the shared event engine and
for committing exactly one winning path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from data_generation_test.parameter import EXIT_NODE, BusinessParameters, NodeId
from src.domain import CandidateRequest, PhysicalRequestStatus, RequestKind
from src.path_state import Arc, RemainingNetwork, VIRTUAL_ORIGIN, station_sequence
from src.reference_rollout import ReservationEventRollout, VisibleEntry


_EPS = 1e-9
UserKey = Tuple[int, int]


class EventPathSearchError(ValueError):
    """Raised when a remaining-path candidate is physically malformed."""


@dataclass(frozen=True)
class EventPathOption:
    """One connected virtual-origin-to-exit remaining path."""

    option_id: str
    user_key: UserKey
    path_arcs: Tuple[Arc, ...]
    station_sequence: Tuple[int, ...]
    rollout: ReservationEventRollout


def _node_key(node: NodeId) -> Tuple[int, str]:
    if node == VIRTUAL_ORIGIN:
        return (-1, str(node))
    if isinstance(node, int):
        return (0, f"{node:09d}")
    return (1, str(node))


def enumerate_remaining_paths(
    remaining: RemainingNetwork,
    *,
    max_paths: int | None = None,
) -> List[Tuple[Arc, ...]]:
    """Enumerate every connected ``virtual_origin -> exit`` path stably.

    The online graph is a downstream DAG.  A defensive cycle check is still
    included so malformed external candidate data fails closed.
    """

    adjacency: Dict[NodeId, List[NodeId]] = {}
    for source, target in remaining.arcs:
        adjacency.setdefault(source, []).append(target)
    for source in adjacency:
        adjacency[source] = sorted(set(adjacency[source]), key=_node_key)

    paths: List[Tuple[Arc, ...]] = []

    def visit(node: NodeId, arcs: List[Arc], seen: set[NodeId]) -> None:
        if node == EXIT_NODE:
            paths.append(tuple(arcs))
            return
        for target in adjacency.get(node, []):
            if target in seen:
                raise EventPathSearchError("remaining candidate network contains a cycle")
            arcs.append((node, target))
            visit(target, arcs, {*seen, target})
            arcs.pop()

    visit(VIRTUAL_ORIGIN, [], {VIRTUAL_ORIGIN})
    if not paths:
        raise EventPathSearchError(
            f"reservation {remaining.user_key} has no virtual-origin-to-exit path"
        )

    # Keep the currently published station sequence first.  This makes ties
    # deterministic and prevents a gratuitous publication when objectives are
    # numerically equal.
    reference = remaining.reference_station_sequence
    paths.sort(
        key=lambda path: (
            station_sequence(path) != reference,
            len(station_sequence(path)),
            station_sequence(path),
            tuple((_node_key(a), _node_key(b)) for a, b in path),
        )
    )
    if max_paths is not None:
        if max_paths <= 0:
            raise EventPathSearchError("max_paths must be positive when supplied")
        paths = paths[:max_paths]
    return paths


def _od_index(params: BusinessParameters, od_id: int) -> int:
    for index, od in enumerate(params.od_pairs):
        if int(od.od_id) == int(od_id):
            return index
    raise EventPathSearchError(f"unknown od_id={od_id}")


def _status_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, PhysicalRequestStatus):
        return value.value
    return str(getattr(value, "value", value)).lower()


def build_remaining_rollout(
    params: BusinessParameters,
    previous: ReservationEventRollout,
    path_arcs: Iterable[Sequence[NodeId]],
    *,
    now: float,
    position_km: float,
    vehicle_soc: float,
    request_status: Mapping[str, Any],
) -> ReservationEventRollout:
    """Materialise a candidate tail while retaining realised service history.

    The first road leg starts at the observed virtual origin.  Every later leg
    starts with a full battery after a successful swap.  Only already-served
    events are retained from the previous rollout; a timed-out/cancelled user
    is terminal and therefore cannot be re-routed.
    """

    arcs = tuple((arc[0], arc[1]) for arc in path_arcs)
    if not arcs or arcs[0][0] != VIRTUAL_ORIGIN or arcs[-1][1] != EXIT_NODE:
        raise EventPathSearchError(
            "remaining path must start at virtual_origin and end at exit"
        )
    for index in range(1, len(arcs)):
        if arcs[index - 1][1] != arcs[index][0]:
            raise EventPathSearchError(f"remaining path is disconnected at arc {index}")

    now = float(now)
    position_km = float(position_km)
    vehicle_soc = float(vehicle_soc)
    if not 0.0 <= vehicle_soc <= 1.0:
        raise EventPathSearchError("vehicle_soc must lie in [0, 1]")

    historical: List[CandidateRequest] = []
    for event in previous.events:
        identifier = event.event_id or event.request_id
        status = _status_name(request_status.get(identifier))
        if status in {
            PhysicalRequestStatus.TIMED_OUT.value,
            PhysicalRequestStatus.CANCELLED.value,
        }:
            raise EventPathSearchError(
                f"terminal reservation {previous.user_key} cannot be re-routed"
            )
        if status == PhysicalRequestStatus.SERVED.value:
            historical.append(event)

    od_index = _od_index(params, previous.od_id)
    od = params.od_pairs[od_index]
    if position_km < od.entry_km - _EPS or position_km > od.exit_km + _EPS:
        raise EventPathSearchError("observed position lies outside the reservation O-D")

    future: List[CandidateRequest] = []
    time_at_node = now
    departure_soc = vehicle_soc
    previous_event_id = (
        historical[-1].event_id or historical[-1].request_id
        if historical
        else None
    )
    order_offset = len(historical)

    for index, (source, target) in enumerate(arcs):
        if source == VIRTUAL_ORIGIN:
            target_position = params.node_position_km(od_index, target)
            distance_km = float(target_position) - position_km
        else:
            distance_km = params.distance_km(od_index, source, target)
        if distance_km < -_EPS:
            raise EventPathSearchError(f"remaining arc {(source, target)!r} moves upstream")
        distance_km = max(0.0, distance_km)
        time_at_node += distance_km / params.vehicle_speed_kmh
        arrival_soc = departure_soc - distance_km / params.range_km
        if arrival_soc < -_EPS or arrival_soc > 1.0 + _EPS:
            raise EventPathSearchError(
                f"candidate reaches {target!r} with SOC {arrival_soc:.12g}"
            )
        arrival_soc = min(1.0, max(0.0, arrival_soc))

        if target == EXIT_NODE:
            if arrival_soc < params.min_exit_soc - _EPS:
                raise EventPathSearchError("candidate reaches exit below min_exit_soc")
            continue
        if not isinstance(target, int):
            raise EventPathSearchError(f"unexpected remaining target {target!r}")

        path_order = order_offset + len(future)
        event_id = CandidateRequest.reservation_event_id(
            previous.user_key, path_order, target
        )
        future.append(
            CandidateRequest(
                request_id=event_id,
                kind=RequestKind.RESERVATION,
                station=target,
                arrival_time=time_at_node,
                deadline=time_at_node + params.max_wait_hours,
                return_soc=arrival_soc,
                user_key=previous.user_key,
                source_arc=(source, target),
                path_order=path_order,
                event_id=event_id,
                upstream_request_id=previous_event_id,
            )
        )
        previous_event_id = event_id
        departure_soc = 1.0

    return ReservationEventRollout(
        reservation_id=previous.reservation_id,
        od_id=previous.od_id,
        user_key=previous.user_key,
        entry=VisibleEntry(now, vehicle_soc, "visible_override"),
        path_arcs=arcs,
        events=tuple([*historical, *future]),
    )


def build_path_options(
    params: BusinessParameters,
    remaining: RemainingNetwork,
    previous: ReservationEventRollout,
    *,
    now: float,
    position_km: float,
    vehicle_soc: float,
    request_status: Mapping[str, Any],
    max_paths: int = 32,
) -> List[EventPathOption]:
    """Enumerate and materialise all valid options up to ``max_paths``."""

    options: List[EventPathOption] = []
    for path in enumerate_remaining_paths(remaining, max_paths=max_paths):
        try:
            rollout = build_remaining_rollout(
                params,
                previous,
                path,
                now=now,
                position_km=position_km,
                vehicle_soc=vehicle_soc,
                request_status=request_status,
            )
        except EventPathSearchError:
            # The graph is a SOC-bin superset.  Exact current-SOC validation
            # belongs here and may legitimately reject individual paths.
            continue
        sequence = station_sequence(path)
        option_id = "direct_exit" if not sequence else "stations:" + "-".join(map(str, sequence))
        options.append(
            EventPathOption(
                option_id=option_id,
                user_key=previous.user_key,
                path_arcs=path,
                station_sequence=sequence,
                rollout=rollout,
            )
        )
    if not options:
        raise EventPathSearchError(
            f"reservation {previous.user_key} has no exact-SOC feasible path option"
        )
    return options


__all__ = [
    "EventPathOption",
    "EventPathSearchError",
    "build_path_options",
    "build_remaining_rollout",
    "enumerate_remaining_paths",
]
