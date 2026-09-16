"""Remaining-path and publication helpers for rolling MPC.

The candidate network is generated from an O-D entry.  During rolling
execution a reservation may already be on the road, so the online problem
needs a virtual origin without changing the semantic sequence of remaining
stations.  This module keeps that transformation and the publication rule in
one small, deterministic place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:
    from data_generation_test.parameter import ENTRY_NODE, EXIT_NODE, BusinessParameters, NodeId
except ImportError:  # pragma: no cover - direct script compatibility
    from parameter import ENTRY_NODE, EXIT_NODE, BusinessParameters, NodeId  # type: ignore


Arc = Tuple[NodeId, NodeId]
UserKey = Tuple[int, int]
VIRTUAL_ORIGIN = "virtual_origin"


def _value(record: Union[Mapping[str, Any], Any], name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _as_arc(arc: Sequence[NodeId]) -> Arc:
    if len(arc) != 2:
        raise ValueError(f"path arc must have two nodes, got {arc!r}")
    return (arc[0], arc[1])


def station_sequence(path_arcs: Iterable[Sequence[NodeId]]) -> Tuple[int, ...]:
    """Return the ordered physical station visits of a path."""

    return tuple(int(to) for _, to in map(_as_arc, path_arcs) if isinstance(to, int))


def path_nodes(path_arcs: Iterable[Sequence[NodeId]]) -> List[NodeId]:
    """Materialise a connected arc list as nodes, validating continuity."""

    arcs = [_as_arc(arc) for arc in path_arcs]
    if not arcs:
        return []
    nodes: List[NodeId] = [arcs[0][0]]
    for source, target in arcs:
        if nodes[-1] != source:
            raise ValueError(f"path is not connected at arc {(source, target)!r}")
        nodes.append(target)
    return nodes


def remaining_path_after_executed(
    path_arcs: Iterable[Sequence[NodeId]],
    executed_stations: Iterable[int],
) -> List[Arc]:
    """Drop the executed station prefix while retaining a connected tail.

    The returned first arc intentionally starts at the last executed station,
    rather than a moving virtual origin.  ``station_sequence`` consequently
    compares only future physical visits, so passing a virtual origin later
    does not turn ordinary driving progress into a path publication.
    """

    arcs = [_as_arc(arc) for arc in path_arcs]
    executed = [int(station) for station in executed_stations]
    if not executed:
        return arcs
    cursor = 0
    cut = 0
    for index, (_, target) in enumerate(arcs):
        if cursor < len(executed) and target == executed[cursor]:
            cursor += 1
            cut = index + 1
    if cursor != len(executed):
        raise ValueError(
            f"executed stations {executed!r} are not a prefix of path {station_sequence(arcs)!r}"
        )
    return arcs[cut:]


@dataclass(frozen=True)
class RemainingNetwork:
    """Online candidate graph from a real or virtual origin."""

    user_key: UserKey
    od_id: int
    virtual_origin_km: float
    virtual_origin_soc: float
    arcs: Tuple[Arc, ...]
    reference_station_sequence: Tuple[int, ...]
    waiting_station: Optional[int] = None

    @property
    def remaining_stations(self) -> Tuple[int, ...]:
        return tuple(sorted({node for arc in self.arcs for node in arc if isinstance(node, int)}))


@dataclass(frozen=True)
class PathPublication:
    """A single real publication decision at a rolling boundary."""

    user_key: UserKey
    occurred_at: float
    changed: bool
    previous_station_sequence: Tuple[int, ...]
    proposed_station_sequence: Tuple[int, ...]
    event_id: Optional[str]


def _od_index(params: BusinessParameters, od_id: int) -> int:
    for index, od in enumerate(params.od_pairs):
        if od.od_id == od_id:
            return index
    raise ValueError(f"unknown od_id={od_id}")


def _node_position(params: BusinessParameters, od_index: int, node: NodeId) -> float:
    return float(params.node_position_km(od_index, node))


def _remaining_reference(path: Sequence[Sequence[NodeId]], position_km: float, params: BusinessParameters, od_index: int) -> Tuple[int, ...]:
    visits = station_sequence(path)
    return tuple(
        station
        for station in visits
        if _node_position(params, od_index, station) > position_km + params.time_epsilon
    )


def build_remaining_network(
    network: Mapping[str, Any],
    params: BusinessParameters,
    reservation_state: Union[Mapping[str, Any], Any],
    now: float,
) -> RemainingNetwork:
    """Build a downstream-only online network.

    ``reservation_state`` accepts either a mapping or an object.  Its required
    fields are ``od_id`` and ``user_id``; absent real-time position/SOC fall
    back to the O-D entry and the effective entry SOC.  The function never
    moves a user upstream and preserves a waiting station even if its physical
    position equals the virtual origin.
    """

    od_id = int(_value(reservation_state, "od_id"))
    user_id = int(_value(reservation_state, "user_id", _value(reservation_state, "reservation_id")))
    od_index = _od_index(params, od_id)
    od = params.od_pairs[od_index]
    position = float(_value(reservation_state, "position_km", od.entry_km))
    soc = float(
        _value(
            reservation_state,
            "vehicle_soc",
            _value(reservation_state, "effective_entry_soc", _value(reservation_state, "actual_entry_soc", 1.0)),
        )
    )
    if not od.entry_km - params.time_epsilon <= position < od.exit_km + params.time_epsilon:
        raise ValueError(f"reservation {(od_id, user_id)} position {position} is outside its O-D")
    if not 0.0 <= soc <= 1.0:
        raise ValueError(f"reservation {(od_id, user_id)} vehicle SOC must be in [0,1]")

    raw_reference = _value(
        reservation_state,
        "last_published_remaining_path",
        _value(reservation_state, "initial_published_path", _value(reservation_state, "baseline_path_arcs", ())),
    )
    reference_arcs = [_as_arc(arc) for arc in raw_reference]
    reference = _remaining_reference(reference_arcs, position, params, od_index)
    waiting_station_raw = _value(reservation_state, "waiting_station", None)
    waiting_station = int(waiting_station_raw) if waiting_station_raw is not None else None

    # The offline graph uses entry SOC filtering.  Its downstream subgraph is
    # still valid after a real swap, because all later legs depart with a full
    # battery.  The first virtual leg is rebuilt below using the observed SOC.
    # Recover downstream topology from the union of offline SOC bins.  Using
    # only the full-entry-SOC bin is incorrect: its pruning can remove a short
    # station (for example the only station reachable by a low-SOC vehicle)
    # even though that station's downstream arcs remain physically valid.
    try:
        bins = network["od_networks"][od_index]["bins"]
        offline_arc_set = {
            _as_arc(arc)
            for bin_info in bins
            for arc in bin_info["candidate_arcs"]
        }
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"candidate network lacks O-D {od_id} SOC-bin arcs") from exc
    # Preserve the paper's direct-exit rule after taking the union.
    for station in od.station_indices:
        if (station, EXIT_NODE) in offline_arc_set:
            offline_arc_set = {
                arc
                for arc in offline_arc_set
                if arc[0] != station or arc[1] == EXIT_NODE
            }
    offline_arcs = sorted(offline_arc_set, key=lambda arc: (str(arc[0]), str(arc[1])))
    positions: Dict[NodeId, float] = {
        node: _node_position(params, od_index, node)
        for node in params.od_nodes(od_index)
    }

    tail_arcs: List[Arc] = []
    for source, target in offline_arcs:
        if source == ENTRY_NODE:
            continue
        if positions[source] + params.time_epsilon < position:
            continue
        if positions[target] + params.time_epsilon < position:
            continue
        tail_arcs.append((source, target))

    origin_arcs: List[Arc] = []
    reachable_origin_arcs: List[Arc] = []
    last_swap_km = float(
        _value(reservation_state, "last_actual_swap_km", od.entry_km)
    )
    for target in [*od.station_indices, EXIT_NODE]:
        target_pos = positions[target]
        distance = target_pos - position
        if distance < -params.time_epsilon:
            continue
        required_soc = params.min_exit_soc if target == EXIT_NODE else 0.0
        if soc - distance / params.range_km < required_soc - params.time_epsilon:
            continue
        reachable_origin_arcs.append((VIRTUAL_ORIGIN, target))
        # The minimum swap spacing is measured from the last *actual swap*,
        # not from the moving virtual origin.  Measuring it from ``position``
        # would incorrectly require another full D_min after every boundary.
        if target != EXIT_NODE and target_pos - last_swap_km < params.min_swap_spacing_km - params.time_epsilon:
            continue
        origin_arcs.append((VIRTUAL_ORIGIN, target))

    def reaches_exit(candidates: Sequence[Arc]) -> bool:
        adjacency: Dict[NodeId, List[NodeId]] = {}
        for source, target in [*candidates, *tail_arcs]:
            adjacency.setdefault(source, []).append(target)
        frontier: List[NodeId] = [VIRTUAL_ORIGIN]
        seen: set[NodeId] = set()
        while frontier:
            node = frontier.pop()
            if node == EXIT_NODE:
                return True
            if node in seen:
                continue
            seen.add(node)
            frontier.extend(adjacency.get(node, []))
        return False

    # ``min_swap_spacing_km`` is a pruning preference in main.tex, not a hard
    # feasibility constraint.  Retain the spaced virtual arcs when they admit
    # a complete path; otherwise restore physically reachable short arcs so a
    # low-SOC vehicle does not lose its only safe station.
    if not reaches_exit(origin_arcs) and reaches_exit(reachable_origin_arcs):
        origin_arcs = reachable_origin_arcs

    if waiting_station is not None:
        if waiting_station not in od.station_indices:
            raise ValueError(f"waiting station {waiting_station} is not on O-D {od_id}")
        # A waiting request is physically at the station.  Its future path
        # cannot be cancelled merely because an online graph rebuild happens.
        origin_arcs = [(VIRTUAL_ORIGIN, waiting_station)]
        tail_arcs = [arc for arc in tail_arcs if arc[0] == waiting_station or positions[arc[0]] > positions[waiting_station]]

    arcs = tuple(sorted(set(origin_arcs + tail_arcs), key=lambda arc: (str(arc[0]), str(arc[1]))))
    if not origin_arcs:
        raise ValueError(f"reservation {(od_id, user_id)} has no reachable downstream arc at t={now}")
    return RemainingNetwork(
        user_key=(od_id, user_id),
        od_id=od_id,
        virtual_origin_km=position,
        virtual_origin_soc=soc,
        arcs=arcs,
        reference_station_sequence=reference,
        waiting_station=waiting_station,
    )


def publish_if_changed(
    reservation_state: Union[Mapping[str, Any], Any],
    proposed_path: Iterable[Sequence[NodeId]],
    now: float,
) -> PathPublication:
    """Return the only publication decision allowed at a rolling boundary.

    Future (not-yet-entered) users may be internally re-planned but never
    publish.  En-route users publish exactly when the remaining physical
    station sequence changes; a moved virtual origin alone has no cost.
    """

    od_id = int(_value(reservation_state, "od_id"))
    user_id = int(_value(reservation_state, "user_id", _value(reservation_state, "reservation_id")))
    phase = str(_value(reservation_state, "phase", _value(reservation_state, "status", "future"))).lower()
    old = _value(
        reservation_state,
        "last_published_remaining_path",
        _value(reservation_state, "initial_published_path", _value(reservation_state, "baseline_path_arcs", ())),
    )
    old_sequence = station_sequence(old)
    new_sequence = station_sequence(proposed_path)
    enroute = phase in {"enroute", "waiting", "at_station"}
    changed = enroute and old_sequence != new_sequence
    event_id = f"publish:{od_id}:{user_id}:{now:.9f}" if changed else None
    return PathPublication(
        user_key=(od_id, user_id),
        occurred_at=float(now),
        changed=changed,
        previous_station_sequence=old_sequence,
        proposed_station_sequence=new_sequence,
        event_id=event_id,
    )


__all__ = [
    "Arc",
    "UserKey",
    "VIRTUAL_ORIGIN",
    "RemainingNetwork",
    "PathPublication",
    "station_sequence",
    "path_nodes",
    "remaining_path_after_executed",
    "build_remaining_network",
    "publish_if_changed",
]
