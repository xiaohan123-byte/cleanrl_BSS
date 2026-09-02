"""Gurobi implementation of the path/event MPC in :mod:`paper.main.tex`.

The paper describes a continuous-event MILP.  With the requested charging
power fixed by the (currently synthetic) RL provider, every station has a
finite deterministic response for each feasible vector of active reservation
arrivals.  This module projects the Appendix B/C queue, charging and slot
variables into *station event patterns* and keeps the cross-station variables
explicit:

* ``y`` and flow conservation implement ``eq:flow``;
* ``x`` and ``d`` implement ``eq:station_visit_indicator`` and
  ``eq:path_adjustment_indicator``;
* ``a/s/f/omega`` implement ``eq:reservation_alive_chain`` and
  ``eq:request_outcome_conservation``;
* station-pattern binaries are an extended formulation of Appendix B/C.  A
  selected pattern is produced by the same deterministic continuous event
  kernel used for first-stage execution, so it already satisfies queue
  priority, exact charging completion, smallest-ready-slot assignment and
  half-open boundary semantics;
* ``A_boundary`` and ``w_pending`` implement Appendix D.

This is a genuine MILP, not coordinate path enumeration.  All users and all
stations are coupled in one Gurobi model.  Projecting deterministic local
event recursions into columns is mathematically equivalent to retaining their
binary state-machine variables when the complete local activation space is
generated.  The public result records whether that space was complete.

RL training is intentionally outside this module.  ``RLSignals`` values are
read-only parameters exactly as required by the paper.
"""

from __future__ import annotations

import copy
import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:  # pragma: no cover - exercised on solver-free deployments
    gp = None  # type: ignore[assignment]
    GRB = None  # type: ignore[assignment]

from data_generation_test.candidate_network import get_feasible_arcs
from data_generation_test.parameter import ENTRY_NODE, EXIT_NODE, BusinessParameters, NodeId
from src.domain import (
    CandidateRequest,
    PhysicalRequestStatus,
    PredictedOutcome,
    RequestKind,
    RollingState,
    WaitingRequest,
)
from src.event_engine import ContinuousEventEngine, ExecutionResult
from src.event_path_search import build_remaining_rollout
from src.path_state import VIRTUAL_ORIGIN, build_remaining_network, station_sequence
from src.reference_rollout import (
    ReservationEventRollout,
    VisibleEntry,
    build_reservation_rollout,
)
from src.time_grid import TimeGrid


UserKey = Tuple[int, int]
Arc = Tuple[NodeId, NodeId]
_EPS = 1e-9


class PaperMPCError(RuntimeError):
    """Base error for the paper MILP branch."""


class PaperMPCSolverUnavailable(PaperMPCError):
    """Raised when the local Gurobi installation/license cannot create a model."""


class PaperMPCNoSolution(PaperMPCError):
    """Raised when Gurobi returns no feasible incumbent."""


def _key_text(key: UserKey) -> str:
    return f"{key[0]}:{key[1]}"


def _safe(value: Any) -> str:
    return str(value).replace(" ", "_").replace(":", "_").replace("-", "m")


def _od_index(params: BusinessParameters, od_id: int) -> int:
    for index, od in enumerate(params.od_pairs):
        if int(od.od_id) == int(od_id):
            return index
    raise PaperMPCError(f"unknown od_id={od_id}")


def _enumerate_paths(arcs: Sequence[Arc], origin: NodeId) -> List[Tuple[Arc, ...]]:
    adjacency: Dict[NodeId, List[NodeId]] = {}
    for source, target in arcs:
        adjacency.setdefault(source, []).append(target)
    for source in adjacency:
        adjacency[source] = sorted(
            set(adjacency[source]),
            key=lambda item: (0 if isinstance(item, int) else 1, str(item)),
        )
    paths: List[Tuple[Arc, ...]] = []

    def visit(node: NodeId, prefix: List[Arc], seen: set[NodeId]) -> None:
        if node == EXIT_NODE:
            paths.append(tuple(prefix))
            return
        for target in adjacency.get(node, []):
            if target in seen:
                raise PaperMPCError("candidate network contains a cycle")
            prefix.append((node, target))
            visit(target, prefix, {*seen, target})
            prefix.pop()

    visit(origin, [], {origin})
    if not paths:
        raise PaperMPCError(f"no {origin!r}-to-exit path in candidate network")
    paths.sort(
        key=lambda path: (
            len(station_sequence(path)),
            station_sequence(path),
            tuple((str(a), str(b)) for a, b in path),
        )
    )
    return paths


@dataclass(frozen=True)
class PaperUserNetwork:
    user_key: UserKey
    od_id: int
    phase: str
    origin: NodeId
    arcs: Tuple[Arc, ...]
    reference_station_sequence: Tuple[int, ...]
    position_km: float
    initial_soc: float
    effective_entry_time: float
    plan_record: Mapping[str, Any] = field(repr=False, compare=False)
    previous_rollout: Optional[ReservationEventRollout] = field(
        default=None, repr=False, compare=False
    )
    waiting_request_id: Optional[str] = None

    @property
    def nodes(self) -> Tuple[NodeId, ...]:
        values = {self.origin, EXIT_NODE}
        for source, target in self.arcs:
            values.add(source)
            values.add(target)
        return tuple(
            sorted(values, key=lambda item: (0 if item == self.origin else 1, str(item)))
        )


@dataclass(frozen=True)
class PaperReservationEvent:
    event_id: str
    user_key: UserKey
    station: int
    source: Optional[NodeId]
    arc: Optional[Arc]
    arrival_time: float
    deadline: float
    return_soc: float
    in_prediction_domain: bool
    carried: bool = False
    request: Optional[CandidateRequest] = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class StationPattern:
    station: int
    pattern_id: str
    active_event_ids: Tuple[str, ...]
    served_ids: Tuple[str, ...]
    failed_ids: Tuple[str, ...]
    pending_ids: Tuple[str, ...]
    service_times: Tuple[Tuple[str, float], ...]
    service_slots: Tuple[Tuple[str, int], ...]
    terminal_soc: Tuple[float, ...]
    income_reservation: float
    income_random: float
    charging_cost: float


@dataclass
class PaperMPCSolution:
    status: str
    is_optimal: bool
    has_incumbent: bool
    solve_time_sec: float
    pattern_generation_time_sec: float
    objective_total: float
    income_reservation: float
    income_random: float
    charging_cost: float
    adjustment_cost: float
    reservation_failure_cost: float
    terminal_value: float
    terminal_value_weight: float
    selected_paths: Dict[UserKey, Tuple[Arc, ...]]
    selected_rollouts: Dict[UserKey, ReservationEventRollout]
    selected_enroute_rollouts: Dict[UserKey, ReservationEventRollout]
    path_adjusted: Dict[UserKey, bool]
    path_decisions: List[Dict[str, Any]]
    request_outcomes: Dict[str, str]
    model_statistics: Dict[str, Any]
    selected_station_patterns: Dict[int, StationPattern] = field(repr=False)


def _entry_index(entries: Iterable[Mapping[str, Any]]) -> Dict[UserKey, Mapping[str, Any]]:
    result: Dict[UserKey, Mapping[str, Any]] = {}
    for entry in entries:
        key = (int(entry["od_id"]), int(entry["reservation_id"]))
        result[key] = entry
    return result


def _build_user_networks(
    params: BusinessParameters,
    network: Mapping[str, Any],
    state: RollingState,
    plan_records: Mapping[UserKey, Mapping[str, Any]],
    visible_entries: Iterable[Mapping[str, Any]],
    current_rollouts: Mapping[UserKey, ReservationEventRollout],
    terminal_users: Iterable[UserKey],
    now: float,
) -> Dict[UserKey, PaperUserNetwork]:
    visible = _entry_index(visible_entries)
    terminal = set(terminal_users)
    users: Dict[UserKey, PaperUserNetwork] = {}
    for key in sorted(plan_records):
        if key in terminal:
            continue
        record = plan_records[key]
        od_index = _od_index(params, key[0])
        od = params.od_pairs[od_index]
        text = _key_text(key)
        if text in state.enroute:
            live = state.enroute[text]
            remaining = build_remaining_network(
                dict(network),
                params,
                {
                    "od_id": key[0],
                    "user_id": key[1],
                    "phase": "waiting" if live.waiting_request_id else "enroute",
                    "position_km": live.current_position,
                    "vehicle_soc": live.vehicle_soc,
                    "last_actual_swap_km": (
                        params.station.positions_km[live.last_actual_swap_station]
                        if live.last_actual_swap_station is not None
                        else od.entry_km
                    ),
                    "waiting_station": (
                        state.find_waiting(live.waiting_request_id).station
                        if live.waiting_request_id is not None
                        and state.find_waiting(live.waiting_request_id) is not None
                        else None
                    ),
                    "last_published_remaining_path": live.last_published_remaining_path,
                },
                now,
            )
            users[key] = PaperUserNetwork(
                user_key=key,
                od_id=key[0],
                phase="waiting" if live.waiting_request_id else "enroute",
                origin=VIRTUAL_ORIGIN,
                arcs=tuple(remaining.arcs),
                reference_station_sequence=(
                    station_sequence(live.last_published_remaining_path)
                    if live.waiting_request_id is not None
                    else remaining.reference_station_sequence
                ),
                position_km=float(live.current_position),
                initial_soc=float(live.vehicle_soc),
                effective_entry_time=float(now),
                plan_record=record,
                previous_rollout=current_rollouts.get(key),
                waiting_request_id=live.waiting_request_id,
            )
            continue

        # A late but not-yet-observed user remains future.  Its effective ETA
        # starts no earlier than the current boundary, matching main.tex.
        entry = visible.get(key)
        entry_time = max(
            float(now),
            float((entry or {}).get("arrival_time", record["day_ahead_entry_time"])),
        )
        entry_soc = float((entry or {}).get("arrival_soc", record["day_ahead_entry_soc"]))
        arcs = tuple(get_feasible_arcs(dict(network), od_index, entry_soc))
        users[key] = PaperUserNetwork(
            user_key=key,
            od_id=key[0],
            phase="future",
            origin=ENTRY_NODE,
            arcs=arcs,
            reference_station_sequence=station_sequence(record["path_arcs"]),
            position_km=float(od.entry_km),
            initial_soc=entry_soc,
            effective_entry_time=entry_time,
            plan_record=record,
        )
    return users


def _event_id(key: UserKey, source: NodeId, station: int) -> str:
    return f"paper_res:{key[0]}:{key[1]}:{_safe(source)}:{station}"


def _build_reservation_events(
    params: BusinessParameters,
    users: Mapping[UserKey, PaperUserNetwork],
    state: RollingState,
    t_start: float,
    t_end: float,
) -> Tuple[Dict[str, PaperReservationEvent], Dict[str, PaperReservationEvent]]:
    events: Dict[str, PaperReservationEvent] = {}
    carried: Dict[str, PaperReservationEvent] = {}
    for user in users.values():
        od_index = _od_index(params, user.od_id)
        for source, target in user.arcs:
            if not isinstance(target, int):
                continue
            # The physical request already in the queue is immutable.  Do not
            # create a second candidate for virtual_origin -> waiting station.
            if user.waiting_request_id is not None and source == user.origin:
                continue
            target_position = float(params.node_position_km(od_index, target))
            arrival = user.effective_entry_time + (
                target_position - user.position_km
            ) / params.vehicle_speed_kmh
            if source == user.origin:
                return_soc = user.initial_soc - (
                    target_position - user.position_km
                ) / params.range_km
            else:
                return_soc = 1.0 - params.distance_km(od_index, source, target) / params.range_km
            if return_soc < -_EPS or return_soc > 1.0 + _EPS:
                raise PaperMPCError(
                    f"candidate arc {(source, target)!r} gives invalid return SOC "
                    f"{return_soc:.12g} for user {user.user_key}"
                )
            return_soc = min(1.0, max(0.0, return_soc))
            identifier = _event_id(user.user_key, source, target)
            in_domain = t_start <= arrival < t_end
            request = None
            if in_domain:
                request = CandidateRequest(
                    # Keep the physical reservation namespace for the fixed
                    # FIFO tie-break.  ``event_id`` remains arc-unique for the
                    # MILP, while the prefix/user ordering now matches the
                    # committed rollout and carried queue.
                    request_id=(
                        f"reservation:{user.user_key[0]}:{user.user_key[1]}:"
                        f"candidate:{_safe(source)}:{target}"
                    ),
                    kind=RequestKind.RESERVATION,
                    station=target,
                    arrival_time=arrival,
                    deadline=arrival + params.max_wait_hours,
                    return_soc=return_soc,
                    user_key=user.user_key,
                    source_arc=(source, target),
                    path_order=0,
                    event_id=identifier,
                )
            events[identifier] = PaperReservationEvent(
                event_id=identifier,
                user_key=user.user_key,
                station=target,
                source=source,
                arc=(source, target),
                arrival_time=arrival,
                deadline=arrival + params.max_wait_hours,
                return_soc=return_soc,
                in_prediction_domain=in_domain,
                request=request,
            )

    for waiting in state.all_waiting_requests():
        if waiting.kind is not RequestKind.RESERVATION or waiting.user_key is None:
            continue
        if waiting.user_key not in users:
            raise PaperMPCError(
                f"carried reservation {waiting.event_id or waiting.request_id!r} "
                f"has no active paper user {waiting.user_key}"
            )
        identifier = waiting.event_id or waiting.request_id
        carried[identifier] = PaperReservationEvent(
            event_id=identifier,
            user_key=waiting.user_key,
            station=waiting.station,
            source=None,
            arc=None,
            arrival_time=waiting.arrival_time,
            deadline=waiting.deadline,
            return_soc=waiting.return_soc,
            in_prediction_domain=True,
            carried=True,
        )
    return events, carried


def _local_state(state: RollingState, station: int) -> RollingState:
    local = state.clone()
    local.waiting_queues = {
        station: copy.deepcopy(state.waiting_queues.get(station, {}))
    }
    waiting_ids = {
        request.event_id or request.request_id
        for request in local.all_waiting_requests()
    }
    local.request_status = {
        identifier: status
        for identifier, status in state.request_status.items()
        if identifier in waiting_ids
    }
    local.reservation_dependencies = {}
    local.accounted_event_ids = set()
    return local


def _pattern_from_execution(
    params: BusinessParameters,
    signals: Any,
    station: int,
    pattern_id: str,
    active_event_ids: Iterable[str],
    execution: ExecutionResult,
) -> StationPattern:
    served = {
        service.request.event_id or service.request.request_id: service
        for service in execution.services
        if service.station == station
    }
    failed = {
        timeout.request.event_id or timeout.request.request_id: timeout
        for timeout in execution.timeouts
        if timeout.request.station == station
    }
    pending = {
        request.event_id or request.request_id
        for request in execution.state.all_waiting_requests()
        if request.station == station
    }
    income_reservation = 0.0
    income_random = 0.0
    for service in served.values():
        energy = params.battery_capacity_kwh * (1.0 - service.request.return_soc)
        income = params.swap_service_price[station][service.interval] * energy
        if service.request.kind is RequestKind.RESERVATION:
            income_reservation += income
        else:
            income_random += income
    charging_cost = sum(
        params.electricity_price[station][segment.interval] * segment.energy_kwh
        for segment in execution.charging_segments
        if segment.station == station
    )
    return StationPattern(
        station=station,
        pattern_id=pattern_id,
        active_event_ids=tuple(sorted(active_event_ids)),
        served_ids=tuple(sorted(served)),
        failed_ids=tuple(sorted(failed)),
        pending_ids=tuple(sorted(pending)),
        service_times=tuple(sorted((identifier, event.occurred_at) for identifier, event in served.items())),
        service_slots=tuple(sorted((identifier, event.slot) for identifier, event in served.items())),
        terminal_soc=tuple(cell.soc for cell in execution.state.slots[station]),
        income_reservation=income_reservation,
        income_random=income_random,
        charging_cost=charging_cost,
    )


def _build_station_patterns(
    params: BusinessParameters,
    state: RollingState,
    events: Mapping[str, PaperReservationEvent],
    forecast_random: Sequence[CandidateRequest],
    signals: Any,
    engine: ContinuousEventEngine,
    period: int,
    horizon: int,
    *,
    max_patterns_per_station: int,
) -> Dict[int, List[StationPattern]]:
    patterns: Dict[int, List[StationPattern]] = {}
    random_by_station: Dict[int, List[CandidateRequest]] = {
        station: [] for station in range(params.station.num_stations)
    }
    prediction_start, prediction_end = engine.time_grid.prediction_bounds(period, horizon)
    for request in forecast_random:
        if not request.active:
            raise PaperMPCError(
                f"inactive random candidate {request.event_id or request.request_id!r} "
                "must be filtered before the paper MILP"
            )
        if not prediction_start <= request.arrival_time < prediction_end:
            raise PaperMPCError(
                f"random candidate {request.event_id or request.request_id!r} at "
                f"{request.arrival_time} is outside the prediction domain"
            )
        random_by_station[request.station].append(request)

    for station in range(params.station.num_stations):
        station_events = [
            event
            for event in events.values()
            if event.station == station and event.in_prediction_domain
        ]
        by_user: Dict[UserKey, List[PaperReservationEvent]] = {}
        for event in station_events:
            by_user.setdefault(event.user_key, []).append(event)
        choice_sets: List[List[Optional[PaperReservationEvent]]] = []
        for key in sorted(by_user):
            options = sorted(by_user[key], key=lambda item: item.event_id)
            choice_sets.append([None, *options])
        combination_count = math.prod(len(items) for items in choice_sets) if choice_sets else 1
        if combination_count > max_patterns_per_station:
            raise PaperMPCError(
                f"station {station} requires {combination_count} exact event patterns, "
                f"exceeding configured limit {max_patterns_per_station}; refusing to truncate"
            )
        station_patterns: List[StationPattern] = []
        iterator = itertools.product(*choice_sets) if choice_sets else [tuple()]
        for index, choices in enumerate(iterator):
            selected_events = [item for item in choices if item is not None]
            arrivals = [
                event.request
                for event in selected_events
                if event.request is not None
            ]
            arrivals.extend(random_by_station[station])
            execution = engine.simulate_horizon(
                _local_state(state, station),
                period,
                horizon,
                signals.requested_power,
                arrivals,
                realized=False,
                in_place=False,
                stop_before_end=True,
            )
            station_patterns.append(
                _pattern_from_execution(
                    params,
                    signals,
                    station,
                    f"s{station}_p{index}",
                    (event.event_id for event in selected_events),
                    execution,
                )
            )
        patterns[station] = station_patterns
    return patterns


def _validate_selected_patterns(
    *,
    params: BusinessParameters,
    state: RollingState,
    events: Mapping[str, PaperReservationEvent],
    forecast_random: Sequence[CandidateRequest],
    signals: Any,
    engine: ContinuousEventEngine,
    period: int,
    horizon: int,
    selected: Mapping[int, StationPattern],
) -> float:
    """Replay the selected columns together and fail closed on any mismatch."""

    started = time.perf_counter()
    active_ids = {
        identifier
        for pattern in selected.values()
        for identifier in pattern.active_event_ids
    }
    arrivals = [
        event.request
        for identifier, event in events.items()
        if identifier in active_ids and event.request is not None
    ]
    arrivals.extend(forecast_random)
    execution = engine.simulate_horizon(
        state.clone(),
        period,
        horizon,
        signals.requested_power,
        arrivals,
        realized=False,
        in_place=False,
        stop_before_end=True,
    )
    for station, expected in selected.items():
        replayed = _pattern_from_execution(
            params,
            signals,
            station,
            f"validation_s{station}",
            expected.active_event_ids,
            execution,
        )
        exact_fields = (
            "active_event_ids",
            "served_ids",
            "failed_ids",
            "pending_ids",
            "service_slots",
        )
        for field_name in exact_fields:
            if getattr(replayed, field_name) != getattr(expected, field_name):
                raise PaperMPCNoSolution(
                    f"selected station pattern {station} failed joint replay on {field_name}: "
                    f"model={getattr(expected, field_name)!r}, "
                    f"replay={getattr(replayed, field_name)!r}"
                )
        expected_times = dict(expected.service_times)
        replayed_times = dict(replayed.service_times)
        if expected_times.keys() != replayed_times.keys() or any(
            not math.isclose(expected_times[key], replayed_times[key], abs_tol=1e-10)
            for key in expected_times
        ):
            raise PaperMPCNoSolution(
                f"selected station pattern {station} failed joint service-time replay"
            )
        for label, expected_values, replayed_values in (
            ("terminal SOC", expected.terminal_soc, replayed.terminal_soc),
            ("objective components", (
                expected.income_reservation,
                expected.income_random,
                expected.charging_cost,
            ), (
                replayed.income_reservation,
                replayed.income_random,
                replayed.charging_cost,
            )),
        ):
            if len(expected_values) != len(replayed_values) or any(
                not math.isclose(left, right, abs_tol=1e-8)
                for left, right in zip(expected_values, replayed_values)
            ):
                raise PaperMPCNoSolution(
                    f"selected station pattern {station} failed joint replay on {label}"
                )
    return time.perf_counter() - started


def _gurobi_status_name(status: int) -> str:
    if GRB is None:
        return "SOLVER_UNAVAILABLE"
    names = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    }
    return names.get(int(status), f"STATUS_{status}")


def _selected_path(
    user: PaperUserNetwork,
    y: Mapping[Tuple[UserKey, Arc], Any],
) -> Tuple[Arc, ...]:
    outgoing: Dict[NodeId, List[NodeId]] = {}
    for arc in user.arcs:
        if y[(user.user_key, arc)].X > 0.5:
            outgoing.setdefault(arc[0], []).append(arc[1])
    node = user.origin
    path: List[Arc] = []
    seen: set[NodeId] = {node}
    while node != EXIT_NODE:
        targets = outgoing.get(node, [])
        if len(targets) != 1:
            raise PaperMPCNoSolution(
                f"incumbent path for {user.user_key} has {len(targets)} selected "
                f"outgoing arcs at {node!r}"
            )
        target = targets[0]
        if target in seen:
            raise PaperMPCNoSolution(f"incumbent path for {user.user_key} contains a cycle")
        path.append((node, target))
        seen.add(target)
        node = target
    return tuple(path)


def _materialize_rollout(
    params: BusinessParameters,
    user: PaperUserNetwork,
    path: Tuple[Arc, ...],
    state: RollingState,
    now: float,
) -> ReservationEventRollout:
    if user.phase == "waiting":
        if user.previous_rollout is None:
            raise PaperMPCError(f"waiting user {user.user_key} has no committed rollout")
        waiting = state.find_waiting(user.waiting_request_id or "")
        if waiting is None:
            raise PaperMPCError(
                f"waiting user {user.user_key} has no carried request "
                f"{user.waiting_request_id!r}"
            )
        if not path or path[0] != (VIRTUAL_ORIGIN, waiting.station):
            raise PaperMPCError(
                f"waiting user {user.user_key} path does not retain station {waiting.station}"
            )

        historical: List[CandidateRequest] = []
        for event in user.previous_rollout.events:
            identifier = event.event_id or event.request_id
            if state.request_status.get(identifier) is PhysicalRequestStatus.SERVED:
                historical.append(event)
        carried_event = CandidateRequest(
            request_id=waiting.request_id,
            kind=waiting.kind,
            station=waiting.station,
            arrival_time=waiting.arrival_time,
            deadline=waiting.deadline,
            return_soc=waiting.return_soc,
            user_key=waiting.user_key,
            source_arc=waiting.source_arc,
            path_order=waiting.path_order,
            event_id=waiting.event_id,
            upstream_request_id=waiting.upstream_request_id,
            active=True,
        )
        future: List[CandidateRequest] = []
        previous_event_id = carried_event.event_id or carried_event.request_id
        od_index = _od_index(params, user.od_id)
        for source, target in path[1:]:
            if target == EXIT_NODE:
                continue
            if not isinstance(source, int) or not isinstance(target, int):
                raise PaperMPCError(
                    f"unexpected waiting-tail arc {(source, target)!r} for {user.user_key}"
                )
            target_position = float(params.node_position_km(od_index, target))
            arrival = now + (target_position - user.position_km) / params.vehicle_speed_kmh
            return_soc = 1.0 - params.distance_km(od_index, source, target) / params.range_km
            path_order = len(historical) + 1 + len(future)
            event_id = CandidateRequest.reservation_event_id(
                user.user_key, path_order, target
            )
            future.append(
                CandidateRequest(
                    request_id=event_id,
                    kind=RequestKind.RESERVATION,
                    station=target,
                    arrival_time=arrival,
                    deadline=arrival + params.max_wait_hours,
                    return_soc=min(1.0, max(0.0, return_soc)),
                    user_key=user.user_key,
                    source_arc=(source, target),
                    path_order=path_order,
                    event_id=event_id,
                    upstream_request_id=previous_event_id,
                )
            )
            previous_event_id = event_id
        return ReservationEventRollout(
            reservation_id=user.previous_rollout.reservation_id,
            od_id=user.previous_rollout.od_id,
            user_key=user.user_key,
            entry=VisibleEntry(now, user.initial_soc, "visible_override"),
            path_arcs=path,
            events=tuple([*historical, carried_event, *future]),
        )
    if user.phase == "enroute":
        if user.previous_rollout is None:
            raise PaperMPCError(f"en-route user {user.user_key} has no committed rollout")
        return build_remaining_rollout(
            params,
            user.previous_rollout,
            path,
            now=now,
            position_km=user.position_km,
            vehicle_soc=user.initial_soc,
            request_status=state.request_status,
        )

    record = dict(user.plan_record)
    record["path_arcs"] = [[source, target] for source, target in path]
    # Use the effective entry snapshot compiled into the MILP.  In particular,
    # a late unobserved future user cannot be projected into the past.
    override = {
        "reservation_id": record.get("reservation_id", user.user_key[1]),
        "user_key": list(user.user_key),
        "arrival_time": user.effective_entry_time,
        "arrival_soc": user.initial_soc,
    }
    return build_reservation_rollout(params, record, visible_entries=[override])


def solve_paper_mpc(
    *,
    params: BusinessParameters,
    network: Mapping[str, Any],
    state: RollingState,
    plan_records: Mapping[UserKey, Mapping[str, Any]],
    visible_entries: Iterable[Mapping[str, Any]],
    current_rollouts: Mapping[UserKey, ReservationEventRollout],
    terminal_users: Iterable[UserKey],
    forecast_random: Sequence[CandidateRequest],
    signals: Any,
    engine: ContinuousEventEngine,
    grid: TimeGrid,
    period: int,
    horizon: int,
    max_patterns_per_station: int = 100_000,
    output_flag: int = 0,
) -> PaperMPCSolution:
    """Solve one joint six-station MPC using the equations in ``main.tex``.

    The Appendix B/C deterministic station recursion is represented by its
    complete local graph (one binary column per feasible activation vector).
    No path combination is preselected or greedily scored: all user arc flows,
    alive chains and station columns are chosen jointly by Gurobi.
    """

    if gp is None or GRB is None:
        raise PaperMPCSolverUnavailable(
            "gurobipy is not installed; the paper MILP has no fallback heuristic"
        )
    if params.station.num_stations != 6:
        raise PaperMPCError("the executable paper scenario requires exactly six stations")
    if horizon <= 0:
        raise PaperMPCError("horizon must be positive")
    signals.validate(params)
    t_start, t_end = grid.prediction_bounds(period, horizon)
    if abs(state.now - t_start) > params.time_epsilon:
        raise PaperMPCError(
            f"rolling state time {state.now} does not match MPC boundary {t_start}"
        )

    entries = list(visible_entries)
    terminal = set(terminal_users)
    users = _build_user_networks(
        params,
        network,
        state,
        plan_records,
        entries,
        current_rollouts,
        terminal,
        t_start,
    )
    events, carried = _build_reservation_events(
        params, users, state, t_start, t_end
    )

    pattern_started = time.perf_counter()
    station_patterns = _build_station_patterns(
        params,
        state,
        events,
        forecast_random,
        signals,
        engine,
        period,
        horizon,
        max_patterns_per_station=max_patterns_per_station,
    )
    pattern_generation_time = time.perf_counter() - pattern_started

    try:
        model = gp.Model(f"paper_event_mpc_{period}")
        model.Params.OutputFlag = int(output_flag)
        model.Params.Threads = int(params.solver.threads)
        model.Params.TimeLimit = float(params.solver.time_limit_sec)
        model.Params.MIPGap = float(params.solver.mip_gap)
        model.Params.FeasibilityTol = float(params.solver.feasibility_tol)
        model.Params.Seed = int(params.seed)
    except gp.GurobiError as exc:
        raise PaperMPCSolverUnavailable(
            "Gurobi could not create the paper MILP in the current OS user "
            f"environment: {exc}"
        ) from exc

    equation_counts: Dict[str, int] = {}

    def add_constraint(expression: Any, *, equation: str, name: str) -> Any:
        equation_counts[equation] = equation_counts.get(equation, 0) + 1
        return model.addConstr(expression, name=name)

    # ------------------------------------------------------------------
    # Path flows, station visits and one-time adjustment indicators.
    # ------------------------------------------------------------------
    y: Dict[Tuple[UserKey, Arc], Any] = {}
    x: Dict[Tuple[UserKey, int], Any] = {}
    d: Dict[UserKey, Any] = {}
    for key, user in users.items():
        key_name = _safe(_key_text(key))
        for index, arc in enumerate(user.arcs):
            y[(key, arc)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"y[{key_name},{index},{_safe(arc[0])},{_safe(arc[1])}]",
            )
        for station in range(params.station.num_stations):
            x[(key, station)] = model.addVar(
                vtype=GRB.BINARY, name=f"x[{key_name},{station}]"
            )
        d[key] = model.addVar(vtype=GRB.BINARY, name=f"d[{key_name}]")

    for key, user in users.items():
        key_name = _safe(_key_text(key))
        for node in user.nodes:
            outgoing = gp.quicksum(
                y[(key, arc)] for arc in user.arcs if arc[0] == node
            )
            incoming = gp.quicksum(
                y[(key, arc)] for arc in user.arcs if arc[1] == node
            )
            rhs = 1 if node == user.origin else (-1 if node == EXIT_NODE else 0)
            add_constraint(
                outgoing - incoming == rhs,
                equation="eq:flow",
                name=f"flow[{key_name},{_safe(node)}]",
            )
        for station in range(params.station.num_stations):
            incoming = gp.quicksum(
                y[(key, arc)] for arc in user.arcs if arc[1] == station
            )
            add_constraint(
                x[(key, station)] == incoming,
                equation="eq:station_visit_indicator",
                name=f"visit[{key_name},{station}]",
            )
            reference = int(station in user.reference_station_sequence)
            add_constraint(
                d[key] >= x[(key, station)] - reference,
                equation="eq:path_adjustment_indicator",
                name=f"adjust_pos[{key_name},{station}]",
            )
            add_constraint(
                d[key] >= reference - x[(key, station)],
                equation="eq:path_adjustment_indicator",
                name=f"adjust_neg[{key_name},{station}]",
            )

    # ------------------------------------------------------------------
    # Reservation activation/outcome variables and alive chains.
    # ------------------------------------------------------------------
    in_domain_events = {
        identifier: event
        for identifier, event in events.items()
        if event.in_prediction_domain
    }
    all_service_events = {**in_domain_events, **carried}
    a: Dict[str, Any] = {}
    s: Dict[str, Any] = {}
    f: Dict[str, Any] = {}
    omega: Dict[str, Any] = {}
    for identifier in sorted(all_service_events):
        name = _safe(identifier)
        a[identifier] = model.addVar(vtype=GRB.BINARY, name=f"a[{name}]")
        s[identifier] = model.addVar(vtype=GRB.BINARY, name=f"sA[{name}]")
        f[identifier] = model.addVar(vtype=GRB.BINARY, name=f"fA[{name}]")
        omega[identifier] = model.addVar(vtype=GRB.BINARY, name=f"omegaA[{name}]")

    for identifier, event in sorted(in_domain_events.items()):
        assert event.arc is not None and event.source is not None
        delta = y[(event.user_key, event.arc)]
        name = _safe(identifier)
        if event.source == users[event.user_key].origin:
            add_constraint(
                a[identifier] == delta,
                equation="eq:reservation_alive_chain",
                name=f"alive_first[{name}]",
            )
        else:
            upstream_service = gp.quicksum(
                s[upstream_id]
                for upstream_id, upstream in all_service_events.items()
                if upstream.user_key == event.user_key
                and upstream.station == event.source
            )
            add_constraint(
                a[identifier] <= delta,
                equation="eq:reservation_alive_chain",
                name=f"alive_arc[{name}]",
            )
            add_constraint(
                a[identifier] <= upstream_service,
                equation="eq:reservation_alive_chain",
                name=f"alive_upstream[{name}]",
            )
            add_constraint(
                a[identifier] >= delta + upstream_service - 1,
                equation="eq:reservation_alive_chain",
                name=f"alive_and[{name}]",
            )
    for identifier in carried:
        add_constraint(
            a[identifier] == 1,
            equation="eq:event_activation_and_soc",
            name=f"carried_active[{_safe(identifier)}]",
        )

    def crosses_boundary(deadline: float) -> int:
        normalized = grid.normalize_for_window(deadline, t_end)
        return int(normalized >= t_end)

    for identifier, event in sorted(all_service_events.items()):
        name = _safe(identifier)
        h_terminal = crosses_boundary(event.deadline)
        add_constraint(
            s[identifier] + f[identifier] + omega[identifier] == a[identifier],
            equation="eq:request_outcome_conservation",
            name=f"reservation_conservation[{name}]",
        )
        add_constraint(
            omega[identifier] <= h_terminal,
            equation="eq:reservation_outcome_boundary",
            name=f"reservation_pending_boundary[{name}]",
        )
        add_constraint(
            f[identifier] <= 1 - h_terminal,
            equation="eq:reservation_outcome_boundary",
            name=f"reservation_failure_boundary[{name}]",
        )

    # ------------------------------------------------------------------
    # Exact deterministic station-event patterns (Appendix B/C projection).
    # ------------------------------------------------------------------
    lam: Dict[Tuple[int, int], Any] = {}
    for station, patterns in station_patterns.items():
        for index, pattern in enumerate(patterns):
            lam[(station, index)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"station_pattern[{station},{index}]",
            )
        add_constraint(
            gp.quicksum(lam[(station, index)] for index in range(len(patterns))) == 1,
            equation="app:B-C:station_pattern_convexity",
            name=f"one_station_pattern[{station}]",
        )

    for identifier, event in sorted(in_domain_events.items()):
        patterns = station_patterns[event.station]
        add_constraint(
            a[identifier]
            == gp.quicksum(
                lam[(event.station, index)]
                for index, pattern in enumerate(patterns)
                if identifier in pattern.active_event_ids
            ),
            equation="eq:event_activation_and_soc",
            name=f"pattern_active[{_safe(identifier)}]",
        )
    for identifier, event in sorted(all_service_events.items()):
        patterns = station_patterns[event.station]
        name = _safe(identifier)
        add_constraint(
            s[identifier]
            == gp.quicksum(
                lam[(event.station, index)]
                for index, pattern in enumerate(patterns)
                if identifier in pattern.served_ids
            ),
            equation="eq:queue_outcome_link",
            name=f"pattern_served[{name}]",
        )
        add_constraint(
            f[identifier]
            == gp.quicksum(
                lam[(event.station, index)]
                for index, pattern in enumerate(patterns)
                if identifier in pattern.failed_ids
            ),
            equation="eq:queue_outcome_link",
            name=f"pattern_failed[{name}]",
        )
        add_constraint(
            omega[identifier]
            == gp.quicksum(
                lam[(event.station, index)]
                for index, pattern in enumerate(patterns)
                if identifier in pattern.pending_ids
            ),
            equation="eq:queue_outcome_link",
            name=f"pattern_pending[{name}]",
        )

    # Random requests have fixed existence; their loss is 1-z-omegaR.
    random_requests: Dict[str, WaitingRequest | CandidateRequest] = {}
    for request in forecast_random:
        identifier = request.event_id or request.request_id
        if identifier in random_requests:
            raise PaperMPCError(f"duplicate random event id {identifier!r}")
        random_requests[identifier] = request
    for request in state.all_waiting_requests():
        if request.kind is not RequestKind.RANDOM:
            continue
        identifier = request.event_id or request.request_id
        if identifier in random_requests:
            raise PaperMPCError(f"duplicate carried random event id {identifier!r}")
        random_requests[identifier] = request
    z: Dict[str, Any] = {}
    omega_random: Dict[str, Any] = {}
    for identifier, request in sorted(random_requests.items()):
        name = _safe(identifier)
        z[identifier] = model.addVar(vtype=GRB.BINARY, name=f"zR[{name}]")
        omega_random[identifier] = model.addVar(
            vtype=GRB.BINARY, name=f"omegaR[{name}]"
        )
        patterns = station_patterns[request.station]
        add_constraint(
            z[identifier]
            == gp.quicksum(
                lam[(request.station, index)]
                for index, pattern in enumerate(patterns)
                if identifier in pattern.served_ids
            ),
            equation="eq:queue_outcome_link",
            name=f"random_served[{name}]",
        )
        add_constraint(
            omega_random[identifier]
            == gp.quicksum(
                lam[(request.station, index)]
                for index, pattern in enumerate(patterns)
                if identifier in pattern.pending_ids
            ),
            equation="eq:queue_outcome_link",
            name=f"random_pending[{name}]",
        )
        h_terminal = crosses_boundary(request.deadline)
        add_constraint(
            z[identifier] + omega_random[identifier] <= 1,
            equation="eq:random_outcome_boundary",
            name=f"random_conservation[{name}]",
        )
        add_constraint(
            omega_random[identifier] <= h_terminal,
            equation="eq:random_outcome_boundary",
            name=f"random_pending_boundary[{name}]",
        )
        add_constraint(
            z[identifier] + omega_random[identifier] >= h_terminal,
            equation="eq:random_outcome_boundary",
            name=f"random_alive_boundary[{name}]",
        )

    # ------------------------------------------------------------------
    # Appendix D boundary-alive and pending-delivery variables.
    # ------------------------------------------------------------------
    alive_boundary: Dict[UserKey, Any] = {}
    for key in users:
        name = _safe(_key_text(key))
        alive_boundary[key] = model.addVar(
            vtype=GRB.BINARY, name=f"A_boundary[{name}]"
        )
        user_failures = [
            f[identifier]
            for identifier, event in all_service_events.items()
            if event.user_key == key
        ]
        if not user_failures:
            add_constraint(
                alive_boundary[key] == 1,
                equation="eq:boundary_alive",
                name=f"boundary_alive_empty[{name}]",
            )
        else:
            add_constraint(
                alive_boundary[key] >= 1 - gp.quicksum(user_failures),
                equation="eq:boundary_alive",
                name=f"boundary_alive_lb[{name}]",
            )
            for index, failure in enumerate(user_failures):
                add_constraint(
                    alive_boundary[key] <= 1 - failure,
                    equation="eq:boundary_alive",
                    name=f"boundary_alive_ub[{name},{index}]",
                )

    delivery_events: Dict[str, PaperReservationEvent] = {**events, **carried}
    pending_delivery: Dict[str, Any] = {}
    for identifier, event in sorted(delivery_events.items()):
        name = _safe(identifier)
        pending_delivery[identifier] = model.addVar(
            vtype=GRB.BINARY, name=f"w_pending[{name}]"
        )
        delta = 1 if event.carried else y[(event.user_key, event.arc)]  # type: ignore[index]
        service_in_domain = s[identifier] if identifier in s else 0
        add_constraint(
            pending_delivery[identifier] <= delta,
            equation="eq:pending_delivery_alive",
            name=f"pending_path[{name}]",
        )
        add_constraint(
            pending_delivery[identifier] <= alive_boundary[event.user_key],
            equation="eq:pending_delivery_alive",
            name=f"pending_alive[{name}]",
        )
        add_constraint(
            pending_delivery[identifier] <= 1 - service_in_domain,
            equation="eq:pending_delivery_alive",
            name=f"pending_unserved[{name}]",
        )
        add_constraint(
            pending_delivery[identifier]
            >= delta + alive_boundary[event.user_key] - service_in_domain - 1,
            equation="eq:pending_delivery_alive",
            name=f"pending_and[{name}]",
        )

    terminal_soc: Dict[Tuple[int, int], Any] = {}
    for station, patterns in station_patterns.items():
        for slot in range(params.station.num_slots):
            terminal_soc[(station, slot)] = model.addVar(
                lb=0.0,
                ub=1.0,
                vtype=GRB.CONTINUOUS,
                name=f"S_terminal[{station},{slot}]",
            )
            add_constraint(
                terminal_soc[(station, slot)]
                == gp.quicksum(
                    pattern.terminal_soc[slot] * lam[(station, index)]
                    for index, pattern in enumerate(patterns)
                ),
                equation="eq:event_charging_transition",
                name=f"terminal_soc_link[{station},{slot}]",
            )

    # ------------------------------------------------------------------
    # Objective (income - energy - adjustment - failure + beta * critic).
    # ------------------------------------------------------------------
    income_reservation_expr = gp.quicksum(
        pattern.income_reservation * lam[(station, index)]
        for station, patterns in station_patterns.items()
        for index, pattern in enumerate(patterns)
    )
    income_random_expr = gp.quicksum(
        pattern.income_random * lam[(station, index)]
        for station, patterns in station_patterns.items()
        for index, pattern in enumerate(patterns)
    )
    charging_cost_expr = gp.quicksum(
        pattern.charging_cost * lam[(station, index)]
        for station, patterns in station_patterns.items()
        for index, pattern in enumerate(patterns)
    )
    adjustment_cost_expr = params.path_adjustment_penalty * gp.quicksum(d.values())
    failure_cost_expr = params.reservation_failure_penalty * gp.quicksum(f.values())
    terminal_soc_expr = gp.quicksum(
        signals.terminal_soc_value[station][slot] * variable
        for (station, slot), variable in terminal_soc.items()
    )
    outside_delivery_expr = gp.quicksum(
        signals.outside_swap_value(event.station, event.return_soc)
        * pending_delivery[identifier]
        for identifier, event in delivery_events.items()
    )
    terminal_value_expr = terminal_soc_expr + outside_delivery_expr
    objective_expr = (
        income_reservation_expr
        + income_random_expr
        - charging_cost_expr
        - adjustment_cost_expr
        - failure_cost_expr
        + params.terminal_value_weight * terminal_value_expr
    )
    model.setObjective(objective_expr, GRB.MAXIMIZE)

    try:
        model.optimize()
    except gp.GurobiError as exc:
        raise PaperMPCSolverUnavailable(f"Gurobi failed while optimizing the paper MILP: {exc}") from exc
    status_name = _gurobi_status_name(model.Status)
    if model.SolCount <= 0:
        raise PaperMPCNoSolution(
            f"paper MILP returned {status_name} with no feasible incumbent"
        )

    selected_paths: Dict[UserKey, Tuple[Arc, ...]] = {}
    selected_rollouts: Dict[UserKey, ReservationEventRollout] = {}
    selected_enroute_rollouts: Dict[UserKey, ReservationEventRollout] = {}
    path_adjusted: Dict[UserKey, bool] = {}
    path_decisions: List[Dict[str, Any]] = []
    for key, user in users.items():
        path = _selected_path(user, y)
        rollout = _materialize_rollout(params, user, path, state, t_start)
        proposed_sequence = station_sequence(path)
        adjusted = bool(d[key].X > 0.5)
        selected_paths[key] = path
        selected_rollouts[key] = rollout
        path_adjusted[key] = adjusted
        if user.phase in {"enroute", "waiting"}:
            selected_enroute_rollouts[key] = rollout
        path_decisions.append(
            {
                "user_key": list(key),
                "status": (
                    "MILP_WAITING_TAIL_SELECTED"
                    if user.phase == "waiting"
                    else "MILP_PATH_SELECTED"
                ),
                "search_method": "gurobi_joint_paper_milp",
                "phase": user.phase,
                "previous_station_sequence": list(user.reference_station_sequence),
                "proposed_station_sequence": list(proposed_sequence),
                "proposed_path": [[source, target] for source, target in path],
                "candidate_count": len(_enumerate_paths(user.arcs, user.origin)),
                "adjusted": adjusted,
                "will_publish": adjusted and user.phase in {"enroute", "waiting"},
                "published": False,
            }
        )

    selected_station_patterns: Dict[int, StationPattern] = {}
    for station, patterns in station_patterns.items():
        selected = [
            pattern
            for index, pattern in enumerate(patterns)
            if lam[(station, index)].X > 0.5
        ]
        if len(selected) != 1:
            raise PaperMPCNoSolution(
                f"station {station} incumbent selected {len(selected)} event patterns"
            )
        selected_station_patterns[station] = selected[0]

    selected_pattern_replay_time = _validate_selected_patterns(
        params=params,
        state=state,
        events=events,
        forecast_random=forecast_random,
        signals=signals,
        engine=engine,
        period=period,
        horizon=horizon,
        selected=selected_station_patterns,
    )

    request_outcomes: Dict[str, str] = {}
    for identifier in sorted(all_service_events):
        if a[identifier].X <= 0.5:
            request_outcomes[identifier] = "inactive"
        elif s[identifier].X > 0.5:
            request_outcomes[identifier] = PredictedOutcome.SERVED_IN_HORIZON.value
        elif f[identifier].X > 0.5:
            request_outcomes[identifier] = PredictedOutcome.FAILED_IN_HORIZON.value
        else:
            request_outcomes[identifier] = PredictedOutcome.PENDING_AT_HORIZON.value
    for identifier in sorted(set(events) - set(in_domain_events)):
        request_outcomes[identifier] = (
            PredictedOutcome.PENDING_AT_HORIZON.value
            if pending_delivery[identifier].X > 0.5
            else "inactive"
        )
    for identifier in sorted(random_requests):
        if z[identifier].X > 0.5:
            request_outcomes[identifier] = PredictedOutcome.SERVED_IN_HORIZON.value
        elif omega_random[identifier].X > 0.5:
            request_outcomes[identifier] = PredictedOutcome.PENDING_AT_HORIZON.value
        else:
            request_outcomes[identifier] = "lost_in_horizon"

    def value(expression: Any) -> float:
        return float(expression.getValue())

    is_optimal = model.Status == GRB.OPTIMAL
    mip_gap = float(model.MIPGap) if model.SolCount else None
    best_bound = float(model.ObjBound) if model.SolCount else None
    pattern_count_by_station = {
        str(station): len(patterns) for station, patterns in station_patterns.items()
    }
    model_statistics = {
        "model_kind": "paper_continuous_event_station_pattern_milp",
        "formulation": "complete_station_pattern_extended_formulation",
        "solver_backend": "gurobi",
        "solver_version": ".".join(str(item) for item in gp.gurobi.version()),
        "solver_status": status_name,
        "is_optimal": is_optimal,
        "has_incumbent": True,
        "mip_gap": mip_gap,
        "best_bound": best_bound,
        "gurobi_runtime_sec": float(model.Runtime),
        "pattern_generation_time_sec": pattern_generation_time,
        "total_optimization_time_sec": pattern_generation_time + float(model.Runtime),
        "selected_pattern_replay_time_sec": selected_pattern_replay_time,
        "selected_pattern_replay_validated": True,
        "variable_count": int(model.NumVars),
        "constraint_count": int(model.NumConstrs),
        "binary_variable_count": int(model.NumBinVars),
        "candidate_event_count": len(in_domain_events),
        "carried_reservation_count": len(carried),
        "random_request_count": len(random_requests),
        "path_user_count": len(users),
        "station_pattern_count": sum(pattern_count_by_station.values()),
        "station_pattern_count_by_station": pattern_count_by_station,
        "station_pattern_limit": max_patterns_per_station,
        "station_pattern_space_complete": True,
        "path_candidate_space_complete": True,
        "global_milp_optimality_claimed": is_optimal,
        "fixed_mock_power": True,
        "rl_training_implemented": False,
        "paper_equation_constraint_counts": dict(sorted(equation_counts.items())),
        "paper_equation_scope": {
            "explicit": [
                "eq:flow",
                "eq:station_visit_indicator",
                "eq:path_adjustment_indicator",
                "eq:event_activation_and_soc",
                "eq:reservation_alive_chain",
                "eq:request_outcome_conservation",
                "eq:random_outcome_boundary",
                "eq:boundary_alive",
                "eq:pending_delivery_alive",
                "eq:objective",
            ],
            "projected_by_complete_station_patterns": [
                "Appendix B queue/priority/work-conservation",
                "Appendix C continuous charging/completion/slot assignment",
            ],
        },
        "selected_y": {
            _key_text(key): [[source, target] for source, target in path]
            for key, path in selected_paths.items()
        },
        "selected_d": {_key_text(key): int(variable.X > 0.5) for key, variable in d.items()},
        "selected_station_pattern": {
            str(station): pattern.pattern_id
            for station, pattern in selected_station_patterns.items()
        },
        "t_start": t_start,
        "t_end": t_end,
    }

    return PaperMPCSolution(
        status=status_name,
        is_optimal=is_optimal,
        has_incumbent=True,
        solve_time_sec=float(model.Runtime),
        pattern_generation_time_sec=pattern_generation_time,
        objective_total=float(model.ObjVal),
        income_reservation=value(income_reservation_expr),
        income_random=value(income_random_expr),
        charging_cost=value(charging_cost_expr),
        adjustment_cost=value(adjustment_cost_expr),
        reservation_failure_cost=value(failure_cost_expr),
        terminal_value=value(terminal_value_expr),
        terminal_value_weight=float(params.terminal_value_weight),
        selected_paths=selected_paths,
        selected_rollouts=selected_rollouts,
        selected_enroute_rollouts=selected_enroute_rollouts,
        path_adjusted=path_adjusted,
        path_decisions=path_decisions,
        request_outcomes=request_outcomes,
        model_statistics=model_statistics,
        selected_station_patterns=selected_station_patterns,
    )


__all__ = [
    "PaperMPCError",
    "PaperMPCSolverUnavailable",
    "PaperMPCNoSolution",
    "PaperUserNetwork",
    "PaperReservationEvent",
    "StationPattern",
    "PaperMPCSolution",
    "solve_paper_mpc",
]
