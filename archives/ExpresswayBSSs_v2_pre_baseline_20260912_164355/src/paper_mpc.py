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
from data_generation_test.rl_data import MockRLProvider, RLProvider, RLSignals
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


# ======================================================================
# 事件 MPC 接口与 replay 控制器
# 说明：本区段是连续事件 MPC 的输入/输出类型、replay-first 求值与
# MPCController 门面。
# ======================================================================

class MPCError(RuntimeError):
    """MPC 求解相关异常基类。"""


class MPCInputError(ValueError):
    """MPC 窗口输入不合法。"""




class MPCNoSolutionError(MPCError):
    """求解结束但无可行 incumbent（如时限内未找到解）。"""



@dataclass(frozen=True)
class MPCEventRequest:
    """MPC 侧的连续候选请求快照。

    该类与 ``src.domain.CandidateRequest`` / ``WaitingRequest`` 字段兼容，
    但不替代领域层的真实状态。它只描述当前一次预测中已枚举的候选，
    ``deadline`` 在构造后不可改写，且时刻均以运营日起点后的小时表示。
    ``kind`` 接受 ``reservation`` / ``random`` 或对应 enum 的 ``.value``。
    """

    request_id: str
    kind: Any
    station: int
    arrival_time: float
    deadline: float
    return_soc: float
    user_key: Optional[UserKey] = None
    source_arc: Optional[Arc] = None
    path_order: int = 0
    event_id: Optional[str] = None
    upstream_request_id: Optional[str] = None
    active: bool = True


@dataclass
class EventMPCWindowInput:
    """连续事件 MPC 的显式输入类型。

    全部调用方统一使用该类型构造预测窗口。
    """

    params: BusinessParameters
    rolling_state: RollingState
    period_ell: int
    rl_signals: RLSignals
    event_requests: List[Any] = field(default_factory=list)
    event_engine: Optional[Any] = None
    time_grid: Optional[Any] = None
    horizon: Optional[int] = None
    reference_context: Optional[Any] = None
    # Fixed prediction-only cost supplied by the deterministic path search.
    # It never enters the realised ledger; the runner records a real
    # PATH_PUBLISHED event only after committing the winning route.
    planned_adjustment_cost: float = 0.0


@dataclass(frozen=True)
class MPCReplayReport:
    """首区间重放比对结果；``matches`` 为真才允许进入真实执行。"""

    matches: bool
    expected_services: Tuple[Tuple[Any, ...], ...]
    replayed_services: Tuple[Tuple[Any, ...], ...]
    expected_slots: Tuple[Tuple[Any, ...], ...]
    replayed_slots: Tuple[Tuple[Any, ...], ...]
    message: str = ""


@dataclass
class FirstStageExecution:
    """首阶段（时段 ell）执行包：唯一被实际执行的部分。"""

    period: int  # = ell
    power_kw: List[List[float]]  # P_act[i][b]：首区间实际充电功率
    ready: List[List[int]]  # g*[i][b] 服务就绪指示
    available_full: List[int]  # F*[i] 可用满电电池数
    assignments: List[Dict[str, Any]]  # 事件-槽匹配（按 站, 槽 升序）
    # 连续事件分支的可重放证据。旧调用方构造该类时无需提供这些字段。
    charging_segments: List[Dict[str, Any]] = field(default_factory=list)
    state_after: Optional[Any] = None
    execution_result: Optional[Any] = None


@dataclass
class EventMPCModelBundle:
    """连续事件 MPC 的构模快照。

    请求功率由 Mock 信号固定，物理可行性与同刻事件顺序由
    ``ContinuousEventEngine`` 定义。因此这里保存的是经规范化后的候选、
    时间契约和稳定排序。
    ``model`` 保留为 ``None``，使调用方可以明确区分尚未将事件位置
    MILP 化的 replay-first 实现与旧 Gurobi bundle。
    """

    window: EventMPCWindowInput
    time_grid: Any
    horizon: int
    requests: List[MPCEventRequest]
    requested_power: List[List[List[float]]]
    event_engine: Any
    model: Optional[Any] = None
    candidate_event_count: int = 0
    variable_count: int = 0
    constraint_count: int = 0


@dataclass
class EventMPCResult:
    """连续事件 MPC 的可重放结果。

    目前 RL 输出和外部参数均为 Mock；功率轨迹是输入参数，因而本结果以
    事件执行器对候选请求的确定性预测为可行 incumbent。它不把
    ``PENDING_AT_HORIZON`` 写回真实状态，也不将终端近似写入 ledger。
    """

    period_ell: int
    horizon: int
    status: str
    is_optimal: bool
    solve_time_sec: float
    objective_total: float
    income_reservation: float
    income_random: float
    charging_cost: float
    adjustment_cost: float
    reservation_failure_cost: float
    terminal_value: float
    terminal_value_weight: float
    request_outcomes: Dict[str, str]
    events: List[Dict[str, Any]]
    first_stage: FirstStageExecution
    terminal_state: Any
    pending_request_ids: List[str]
    replay_initial_state: Any = field(repr=False)
    replay_requests: List[MPCEventRequest] = field(repr=False)
    replay_requested_power: List[List[float]] = field(repr=False)
    time_grid: Any = field(repr=False)
    event_engine: Any = field(repr=False)
    model_statistics: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# 连续事件接口的无状态工具
# ----------------------------------------------------------------------
_MISSING = object()


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    """同时支持 dataclass / 普通对象 / JSON dict 的只读字段访问。"""
    if isinstance(value, dict):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)
    if default is _MISSING:
        raise MPCInputError(f"连续事件输入缺少必填字段 {name!r}")
    return default


def _finite_number(value: Any, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MPCInputError(f"{label} 必须是有限浮点数，当前为 {value!r}") from exc
    if not math.isfinite(out):
        raise MPCInputError(f"{label} 必须是有限浮点数，当前为 {value!r}")
    return out


def _interval_hours(params: Any) -> float:
    """读取外层区间时长（小时）。"""
    return _finite_number(
        getattr(params, "interval_hours", None), "params.interval_hours"
    )


def _time_epsilon(params: Any) -> float:
    """仅用于消除浮点运算误差，绝不定义预测域的 guard 带。"""
    epsilon = _finite_number(getattr(params, "time_epsilon", 1e-9), "time_epsilon")
    if epsilon < 0.0:
        raise MPCInputError("time_epsilon 不能为负")
    return epsilon


@dataclass(frozen=True)
class _FallbackTimeGrid:
    """event_core 尚未导入时使用的最小时间契约实现。

    正常路径会使用 ``src.time_grid.TimeGrid``；保留该实现能让本模块的
    输入校验和边界契约独立可测，且其半开区间规则与正式实现一致。
    """

    interval_hours: float
    time_epsilon: float

    def start(self, interval: int) -> float:
        return interval * self.interval_hours

    def end(self, interval: int) -> float:
        return (interval + 1) * self.interval_hours

    def interval(self, interval: int) -> Tuple[float, float]:
        return (self.start(interval), self.end(interval))

    def prediction_bounds(self, ell: int, horizon: int) -> Tuple[float, float]:
        return (self.start(ell), self.end(ell + horizon - 1))

    def normalize_for_window(self, t: float, t_end: float) -> float:
        # 仅规范数值计算所得的“精确右端”。例如 t_end-5e-8 仍是当前
        # 窗口事件，不能用求解器 guard 带将其偷移到下一轮。
        if abs(t - t_end) <= self.time_epsilon:
            return t_end
        return t

    def contains_execution_time(self, t: float, interval: int) -> bool:
        start, end = self.interval(interval)
        return start <= t < end

    def current_event_upper_bound(self, t_end: float) -> float:
        return t_end


def _time_grid(params: Any, supplied: Any = None) -> Any:
    if supplied is not None:
        return supplied
    duration = _interval_hours(params)
    epsilon = _time_epsilon(params)
    try:
        from src.time_grid import TimeGrid  # type: ignore

        return TimeGrid(duration)
    except (ImportError, TypeError):
        # 其余逻辑仍按相同时间契约执行，避免为了导入顺序而回到旧的
        # floor/整时段语义。
        return _FallbackTimeGrid(duration, epsilon)


def _grid_prediction_bounds(grid: Any, ell: int, horizon: int) -> Tuple[float, float]:
    bounds = _field(grid, "prediction_bounds")(ell, horizon)
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise MPCInputError("TimeGrid.prediction_bounds 必须返回 (t_start, t_end)")
    start = _finite_number(bounds[0], "prediction t_start")
    end = _finite_number(bounds[1], "prediction t_end")
    if not start < end:
        raise MPCInputError(f"预测窗口必须满足 t_start<t_end，当前为 ({start}, {end})")
    return start, end


def _grid_normalize(grid: Any, t: float, t_end: float) -> float:
    normalizer = _field(grid, "normalize_for_window", None)
    if normalizer is None:
        return t
    value = normalizer(t, t_end)
    # 有些早期实现返回 (normalized_time, snapped)，适配其第一项。
    if isinstance(value, (tuple, list)):
        if not value:
            raise MPCInputError("TimeGrid.normalize_for_window 返回空值")
        value = value[0]
    return _finite_number(value, "normalized event time")


def _kind_name(kind: Any) -> str:
    raw = getattr(kind, "value", kind)
    name = str(raw).strip().lower()
    if name in {"reservation", "reserved", "res", "预约"}:
        return "reservation"
    if name in {"random", "rand", "随机"}:
        return "random"
    raise MPCInputError(f"不支持的请求类别 {kind!r}；仅允许 reservation/random")


def _station_energy_limit(params: Any, station: int, period: int) -> float:
    """读取逐站逐区间能量上限，兼容旧参数对象的标量方法。"""
    value = getattr(params, "station_energy_limit_kwh", None)
    if callable(value):
        try:
            value = value(station, period)
        except TypeError:
            value = value()
    if value is not None:
        if isinstance(value, (int, float)):
            return _finite_number(value, "station_energy_limit_kwh")
        try:
            row = value[station]
            item = row[period]
        except (IndexError, KeyError, TypeError) as exc:
            raise MPCInputError(
                f"station_energy_limit_kwh 缺少站 {station}、区间 {period}"
            ) from exc
        return _finite_number(item, f"station_energy_limit_kwh[{station}][{period}]")
    raise MPCInputError("参数缺少 station_energy_limit_kwh")


def _record(value: Any) -> Dict[str, Any]:
    """将执行器事件转换为稳定、JSON 友好的浅记录。"""
    if isinstance(value, dict):
        return dict(value)
    names = (
        "event_id", "request_id", "kind", "station", "slot", "time",
        "occurred_at", "service_time", "arrival_time", "deadline",
        "return_soc", "start_time", "end_time", "power_kw", "duration",
        "interval", "amount", "energy_kwh", "wait_hours",
    )
    out: Dict[str, Any] = {}
    for name in names:
        if hasattr(value, name):
            item = getattr(value, name)
            if hasattr(item, "value"):
                item = item.value
            if isinstance(item, tuple):
                item = list(item)
            out[name] = item
    # ServiceEvent / TimeoutEvent 将请求嵌在 ``request`` 字段，展开为与
    # JSON ``to_dict`` 一致的扁平证据，避免结果提取丢失 request_id。
    request = getattr(value, "request", None)
    if request is not None:
        for name in (
            "request_id", "kind", "station", "arrival_time", "deadline",
            "return_soc", "event_id", "user_key", "source_arc", "path_order",
        ):
            if name not in out and hasattr(request, name):
                item = getattr(request, name)
                if hasattr(item, "value"):
                    item = item.value
                if isinstance(item, tuple):
                    item = list(item)
                out[name] = item
        if "event_id" in out and "request_event_id" not in out:
            # 外层 event_id 是 service:/timeout:，嵌套 id 才是请求标识。
            outer_id = out["event_id"]
            request_id = getattr(request, "event_id", None)
            if request_id is not None:
                out["request_event_id"] = request_id
            out["event_id"] = outer_id
    return out


# ----------------------------------------------------------------------
# 内部结构
# ----------------------------------------------------------------------
class MPCController:
    """连续事件 MPC 控制器（replay-first 求值与首区间重放）。

    参数
    ----
    params : BusinessParameters
        业务参数对象（data_generation_test.parameter）。
    rl_provider : RLProvider, optional
        RL 信号提供者；缺省 MockRLProvider(params)。
    """

    def __init__(
        self,
        params: BusinessParameters,
        rl_provider: Optional[RLProvider] = None,
    ) -> None:
        params.validate()
        self.params = params
        self.rl_provider = (
            rl_provider if rl_provider is not None else MockRLProvider(params)
        )

    def build_model(self, window: EventMPCWindowInput) -> EventMPCModelBundle:
        """兼容别名：构造连续事件 MPC 的构模快照。"""
        return self.build_event_model(window)

    def solve_step(self, window: EventMPCWindowInput) -> EventMPCResult:
        """求解本轮连续事件 MPC（replay-first），返回可重放结果。"""
        return self.solve_event_step(window)

    @staticmethod
    def _event_request_from(value: Any) -> MPCEventRequest:
        """把领域层 CandidateRequest / WaitingRequest 转为只读 MPC 快照。"""
        if isinstance(value, MPCEventRequest):
            return value
        raw_key = _field(value, "user_key", None)
        if raw_key is not None:
            try:
                raw_key = (int(raw_key[0]), int(raw_key[1]))
            except (TypeError, IndexError, ValueError) as exc:
                raise MPCInputError(f"请求 user_key={raw_key!r} 非法") from exc
        raw_arc = _field(value, "source_arc", None)
        if raw_arc is not None:
            try:
                raw_arc = (raw_arc[0], raw_arc[1])
            except (TypeError, IndexError) as exc:
                raise MPCInputError(f"请求 source_arc={raw_arc!r} 非法") from exc
        return MPCEventRequest(
            request_id=str(_field(value, "request_id")),
            kind=_field(value, "kind"),
            station=int(_field(value, "station")),
            arrival_time=_finite_number(_field(value, "arrival_time"), "arrival_time"),
            deadline=_finite_number(_field(value, "deadline"), "deadline"),
            return_soc=_finite_number(_field(value, "return_soc"), "return_soc"),
            user_key=raw_key,
            source_arc=raw_arc,
            path_order=int(_field(value, "path_order", 0)),
            event_id=_field(value, "event_id", None),
            upstream_request_id=_field(value, "upstream_request_id", None),
            active=bool(_field(value, "active", True)),
        )

    @staticmethod
    def _slot_matrices(
        state: Any, n_sta: int, n_slot: int
    ) -> Tuple[List[List[float]], List[List[int]], Tuple[Tuple[Any, ...], ...]]:
        """提取领域状态的逐槽 SOC/ready，用于首区间快照与重放校验。"""
        soc = [[0.0 for _ in range(n_slot)] for _ in range(n_sta)]
        ready = [[0 for _ in range(n_slot)] for _ in range(n_sta)]
        seen: set[Tuple[int, int]] = set()
        slots = _field(state, "slots", _field(state, "slot_states", None))
        if slots is None:
            old_soc = _field(state, "soc_obs", None)
            if old_soc is None:
                return soc, ready, tuple()
            if len(old_soc) != n_sta or any(len(row) != n_slot for row in old_soc):
                raise MPCInputError("事件状态的 slots / soc_obs 形状与站点配置不一致")
            for i in range(n_sta):
                for b in range(n_slot):
                    value = _finite_number(old_soc[i][b], f"soc_obs[{i}][{b}]")
                    soc[i][b] = value
                    ready[i][b] = int(value == 1.0)
                    seen.add((i, b))
        else:
            # 支持扁平 [SlotState]、二维 [站][槽] 和 {station: [SlotState]}。
            if isinstance(slots, dict):
                iterable: Iterable[Any] = (
                    item for row in slots.values() for item in row
                )
            elif slots and isinstance(slots[0], (list, tuple)):
                iterable = (item for row in slots for item in row)
            else:
                iterable = iter(slots)
            for fallback, item in enumerate(iterable):
                i = int(_field(item, "station", fallback // n_slot))
                b = int(_field(item, "slot", fallback % n_slot))
                if not (0 <= i < n_sta and 0 <= b < n_slot):
                    raise MPCInputError(f"SlotState({i}, {b}) 超出站点/槽位范围")
                if (i, b) in seen:
                    raise MPCInputError(f"事件状态重复包含 SlotState({i}, {b})")
                value = _finite_number(_field(item, "soc"), f"slot[{i},{b}].soc")
                if not 0.0 <= value <= 1.0:
                    raise MPCInputError(f"slot[{i},{b}].soc={value} 超出 [0,1]")
                soc[i][b] = value
                ready[i][b] = int(bool(_field(item, "ready", value == 1.0)))
                seen.add((i, b))
        signature: List[Tuple[Any, ...]] = []
        for i in range(n_sta):
            for b in range(n_slot):
                signature.append((i, b, round(soc[i][b], 12), ready[i][b]))
        return soc, ready, tuple(signature)

    def _validate_event_window(
        self, window: EventMPCWindowInput
    ) -> Tuple[Any, int, List[MPCEventRequest], List[List[List[float]]], Any]:
        """验证连续时序、Mock 信号和站级能量边界，并返回规范化快照。

        这里刻意不读取外部真值或上一轮的求解结果；所有输入必须来自
        ObservationView / Mock reference 的当前快照。
        """
        p = window.params
        try:
            p.validate()
        except (AttributeError, TypeError, ValueError) as exc:
            raise MPCInputError(f"业务参数不合法: {exc}") from exc
        st = p.station
        n_sta, n_slot = st.num_stations, st.num_slots
        ell = int(window.period_ell)
        if not 0 <= ell < p.num_periods:
            raise MPCInputError(f"period_ell={ell} 超出运营日范围")
        grid = _time_grid(p, window.time_grid)
        horizon = int(window.horizon if window.horizon is not None else p.horizon)
        if horizon <= 0:
            raise MPCInputError("连续事件 horizon 必须为正整数")
        if ell + horizon > p.num_periods:
            raise MPCInputError(
                f"预测窗口 [{ell}, {ell + horizon}) 超出 num_periods={p.num_periods}；"
                "请在日末缩短 horizon，而不是用最后一段价格/能量限额延展"
            )
        t_start, t_end = _grid_prediction_bounds(grid, ell, horizon)
        state_now = _field(window.rolling_state, "now", None)
        if state_now is not None:
            now = _finite_number(state_now, "state.now")
            if abs(now - t_start) > _time_epsilon(p):
                raise MPCInputError(
                    f"state.now={now} 与 period_ell={ell} 的左端 {t_start} 不一致"
                )
        self._slot_matrices(window.rolling_state, n_sta, n_slot)

        sig = window.rl_signals
        if sig is None:
            raise MPCInputError("连续事件 MPC 必须接收 Mock RLSignals")
        signal_source = _field(sig, "signal_source", None)
        if signal_source is not None and str(signal_source).lower() != "mock":
            raise MPCInputError(
                f"本轮仅允许 Mock 信号，signal_source={signal_source!r}"
            )
        if int(_field(sig, "start_period")) != ell:
            raise MPCInputError("RLSignals.start_period 与 period_ell 不一致")
        if int(_field(sig, "horizon")) != horizon:
            raise MPCInputError("RLSignals.horizon 与连续事件 horizon 不一致")
        raw_power = _field(sig, "requested_power")
        if len(raw_power) != n_sta or any(len(raw_power[i]) != n_slot for i in range(n_sta)):
            raise MPCInputError("requested_power 形状必须为 [station][slot][horizon]")
        slot_cap = _finite_number(st.slot_power_limit_kw, "slot_power_limit_kw")
        requested_power: List[List[List[float]]] = []
        duration = _interval_hours(p)
        for i in range(n_sta):
            station_rows: List[List[float]] = []
            for b in range(n_slot):
                row = raw_power[i][b]
                if len(row) != horizon:
                    raise MPCInputError(
                        f"requested_power[{i}][{b}] 长度必须为 horizon={horizon}"
                    )
                power_row: List[float] = []
                for h, item in enumerate(row):
                    value = _finite_number(item, f"requested_power[{i}][{b}][{h}]")
                    if not 0.0 <= value <= slot_cap:
                        raise MPCInputError(
                            f"requested_power[{i}][{b}][{h}]={value} 超出 [0,{slot_cap}]"
                        )
                    power_row.append(value)
                station_rows.append(power_row)
            requested_power.append(station_rows)
            for h in range(horizon):
                energy = duration * sum(requested_power[i][b][h] for b in range(n_slot))
                limit = _station_energy_limit(p, i, ell + h)
                if limit < 0.0:
                    raise MPCInputError(f"station_energy_limit_kwh[{i}][{ell+h}] 不能为负")
                if energy > limit + 1e-9:
                    raise MPCInputError(
                        f"站 {i} 区间 {ell+h} 请求能量 {energy} kWh 超过上限 {limit} kWh"
                    )

        seen_ids: set[str] = set()
        requests: List[MPCEventRequest] = []
        for raw in window.event_requests:
            req = self._event_request_from(raw)
            if not req.request_id:
                raise MPCInputError("候选请求 request_id 不能为空")
            if req.request_id in seen_ids:
                raise MPCInputError(f"候选请求 id {req.request_id!r} 重复")
            seen_ids.add(req.request_id)
            _kind_name(req.kind)
            if not 0 <= req.station < n_sta:
                raise MPCInputError(f"请求 {req.request_id} 的 station={req.station} 非法")
            if not 0.0 <= req.return_soc <= 1.0:
                raise MPCInputError(
                    f"请求 {req.request_id} 的 return_soc={req.return_soc} 超出 [0,1]"
                )
            arrival = _grid_normalize(grid, req.arrival_time, t_end)
            deadline = _grid_normalize(grid, req.deadline, t_end)
            if deadline < arrival:
                raise MPCInputError(
                    f"请求 {req.request_id} 的 deadline={deadline} 早于 arrival={arrival}"
                )
            requests.append(
                MPCEventRequest(
                    request_id=req.request_id,
                    kind=req.kind,
                    station=req.station,
                    arrival_time=arrival,
                    deadline=deadline,
                    return_soc=req.return_soc,
                    user_key=req.user_key,
                    source_arc=req.source_arc,
                    path_order=req.path_order,
                    event_id=req.event_id,
                    upstream_request_id=req.upstream_request_id,
                    active=req.active,
                )
            )
        # 预约优先、类内 FCFS 与 stable id：该顺序同样传给执行器，禁止由
        # dict 插入顺序或求解器变量编号隐式决定。
        requests.sort(
            key=lambda r: (
                r.arrival_time,
                0 if _kind_name(r.kind) == "reservation" else 1,
                r.request_id,
                r.station,
                r.path_order,
            )
        )
        return grid, horizon, requests, requested_power, window.event_engine

    def _event_engine(self, window_engine: Any, grid: Any) -> Any:
        """获取共享连续事件内核；不在 MPC 内复制物理规则。"""
        if window_engine is not None:
            return window_engine
        try:
            from src.event_engine import ContinuousEventEngine  # type: ignore
        except ImportError as exc:
            raise MPCInputError(
                "连续事件 MPC 需要 src.event_engine.ContinuousEventEngine"
            ) from exc
        p = self.params
        try:
            return ContinuousEventEngine(
                time_grid=grid,
                battery_capacity_kwh=p.battery_capacity_kwh,
                charging_efficiency=p.station.charging_efficiency,
                max_wait_hours=getattr(p, "max_wait_hours", None),
                slot_power_limit_kw=p.station.slot_power_limit_kw,
                station_energy_limit_kwh=getattr(p, "station_energy_limit_kwh", None),
            )
        except TypeError:
            # 兼容开发期的严格构造器：max_wait 已由每个 request.deadline
            # 固化时无需重复传入。
            return ContinuousEventEngine(
                time_grid=grid,
                battery_capacity_kwh=p.battery_capacity_kwh,
                charging_efficiency=p.station.charging_efficiency,
                slot_power_limit_kw=p.station.slot_power_limit_kw,
                station_energy_limit_kwh=getattr(p, "station_energy_limit_kwh", None),
            )

    @staticmethod
    def _contains_time(grid: Any, time_value: float, period: int) -> bool:
        contains = _field(grid, "contains_execution_time", None)
        if contains is not None:
            return bool(contains(time_value, period))
        start, end = _field(grid, "interval")(period)
        return start <= time_value < end

    @staticmethod
    def _run_event_interval(
        engine: Any,
        state: Any,
        period: int,
        requested_power: List[List[float]],
        arrivals: List[MPCEventRequest],
    ) -> Any:
        """调用共享内核，兼容 keyword 与早期 positional 形态。"""
        try:
            return engine.simulate_interval(
                state,
                interval_index=period,
                requested_power=requested_power,
                arrivals=arrivals,
                realized=False,
            )
        except TypeError as keyword_error:
            try:
                return engine.simulate_interval(
                    state, period, requested_power, arrivals, realized=False
                )
            except TypeError:
                raise keyword_error

    @staticmethod
    def _engine_requests(requests: Sequence[MPCEventRequest]) -> List[Any]:
        """把 MPC 快照转换为领域层候选，避免执行器依赖历史 dataclass。"""
        try:
            from src.domain import CandidateRequest, RequestKind  # type: ignore

            return [
                CandidateRequest(
                    request_id=req.request_id,
                    kind=RequestKind(_kind_name(req.kind)),
                    station=req.station,
                    arrival_time=req.arrival_time,
                    deadline=req.deadline,
                    return_soc=req.return_soc,
                    user_key=req.user_key,
                    source_arc=req.source_arc,
                    path_order=req.path_order,
                    event_id=req.event_id,
                    upstream_request_id=req.upstream_request_id,
                    active=req.active,
                )
                for req in requests
            ]
        except ImportError:
            # 仅在 event_core 尚未安装完成的开发阶段允许 duck typing；正式
            # 执行仍必须经过共享领域类型。
            return list(requests)

    @staticmethod
    def _result_items(result: Any, name: str) -> List[Any]:
        value = _field(result, name, [])
        return list(value or [])

    @staticmethod
    def _event_time(record: Dict[str, Any], fallback: Optional[float] = None) -> Optional[float]:
        for key in ("service_time", "occurred_at", "time", "start_time"):
            if key in record and record[key] is not None:
                return _finite_number(record[key], key)
        return fallback

    @staticmethod
    def _service_signature(records: Sequence[Dict[str, Any]]) -> Tuple[Tuple[Any, ...], ...]:
        out: List[Tuple[Any, ...]] = []
        for record in records:
            time_value = MPCController._event_time(record)
            out.append(
                (
                    record.get("request_id"),
                    record.get("station"),
                    record.get("slot"),
                    None if time_value is None else round(time_value, 12),
                )
            )
        return tuple(sorted(out, key=repr))

    def build_event_model(
        self, window: EventMPCWindowInput
    ) -> EventMPCModelBundle:
        """构造连续事件 MPC 的固定候选与时间契约快照。

        该阶段不再生成 ``P/g/F/S_pre`` 等整时段变量：Mock 请求功率是
        参数，连续状态由统一事件内核产生。调用 ``solve_event_step`` 后
        首区间快照可被同一内核逐字段重放。
        """
        event_window = window
        grid, horizon, requests, power, supplied_engine = self._validate_event_window(
            event_window
        )
        engine = self._event_engine(supplied_engine, grid)
        return EventMPCModelBundle(
            window=event_window,
            time_grid=grid,
            horizon=horizon,
            requests=requests,
            requested_power=power,
            event_engine=engine,
            candidate_event_count=len(requests),
            variable_count=0,
            constraint_count=0,
        )

    def solve_event_step(
        self, window: EventMPCWindowInput
    ) -> EventMPCResult:
        """以共享执行器评价一个给定候选并产生可重放 incumbent。

        单次调用仍不创建事件位置变量：Mock ``P_hat`` 和该调用的请求集合
        都是固定参数。公开 runner 在本函数外枚举剩余路径并逐候选调用本
        函数，赢家因此可标为 ``EVENT_PATH_ENUM_REPLAY``；这与联合路径—
        事件位置 MILP 的全局最优性是两个不同层级。
        """
        bundle = self.build_event_model(window)
        event_window = bundle.window
        p = event_window.params
        st = p.station
        ell = event_window.period_ell
        t_start, t_end = _grid_prediction_bounds(bundle.time_grid, ell, bundle.horizon)
        state = copy.deepcopy(event_window.rolling_state)
        initial_state = copy.deepcopy(state)
        first_execution: Optional[Any] = None
        all_services: List[Dict[str, Any]] = []
        all_timeouts: List[Dict[str, Any]] = []
        all_segments: List[Dict[str, Any]] = []
        all_events: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        for h in range(bundle.horizon):
            q = ell + h
            arrivals = [
                req for req in bundle.requests
                if req.active
                and self._contains_time(bundle.time_grid, req.arrival_time, q)
            ]
            power_q = [
                [bundle.requested_power[i][b][h] for b in range(st.num_slots)]
                for i in range(st.num_stations)
            ]
            execution = self._run_event_interval(
                bundle.event_engine, state, q, power_q, self._engine_requests(arrivals)
            )
            state = _field(execution, "state", state)
            services = [_record(item) for item in self._result_items(execution, "services")]
            timeouts = [_record(item) for item in self._result_items(execution, "timeouts")]
            segments = [
                _record(item) for item in self._result_items(execution, "charging_segments")
            ]
            all_services.extend(services)
            all_timeouts.extend(timeouts)
            all_segments.extend(segments)
            all_events.extend({"event_type": "SERVICE", **item} for item in services)
            all_events.extend({"event_type": "TIMEOUT", **item} for item in timeouts)
            if h == 0:
                first_execution = execution
        solve_time = time.perf_counter() - t0
        if first_execution is None:  # defensive; horizon has already been validated > 0.
            raise MPCNoSolutionError("连续事件 MPC 未生成首区间执行结果")

        request_by_id = {req.request_id: req for req in bundle.requests}
        served_ids = {
            str(item["request_id"])
            for item in all_services
            if item.get("request_id") is not None
        }
        timeout_ids = {
            str(item["request_id"])
            for item in all_timeouts
            if item.get("request_id") is not None
        }
        outcomes: Dict[str, str] = {}
        for req in bundle.requests:
            if not req.active:
                outcomes[req.request_id] = "NOT_EFFECTIVE"
            elif req.request_id in served_ids:
                outcomes[req.request_id] = "SERVED_IN_HORIZON"
            elif req.request_id in timeout_ids:
                outcomes[req.request_id] = "FAILED_IN_HORIZON"
            else:
                # 包含 t_end / guard 内事件：它们严格属于下一轮，不能被
                # 改写为本轮服务或失败。
                outcomes[req.request_id] = "PENDING_AT_HORIZON"
        pending_ids = sorted(
            request_id for request_id, outcome in outcomes.items()
            if outcome == "PENDING_AT_HORIZON"
        )
        all_events.extend(
            {
                "event_type": "PENDING_AT_HORIZON",
                "request_id": request_id,
                "outcome": outcomes[request_id],
            }
            for request_id in pending_ids
        )

        # ---- 目标分项：均由事件时间/分段积分取得，不用整时段平均快照。 ----
        income_reservation = 0.0
        income_random = 0.0
        for service in all_services:
            req = request_by_id.get(str(service.get("request_id")))
            if req is None:
                continue
            service_time = self._event_time(service, req.arrival_time)
            if service_time is None:
                continue
            interval_of = _field(bundle.time_grid, "interval_of", None)
            if interval_of is None:
                period = int(service_time / _interval_hours(p))
            else:
                period = int(interval_of(service_time))
            price_row = p.swap_service_price[req.station]
            if not 0 <= period < len(price_row):
                raise MPCError(f"服务 {req.request_id} 的价格区间 {period} 非法")
            income = p.battery_capacity_kwh * price_row[period] * (1.0 - req.return_soc)
            if _kind_name(req.kind) == "reservation":
                income_reservation += income
            else:
                income_random += income
        charging_cost = 0.0
        for segment in all_segments:
            station = segment.get("station")
            power_kw = segment.get("power_kw")
            if station is None or power_kw is None:
                continue
            begin = self._event_time(segment)
            end = segment.get("end_time")
            duration = segment.get("duration")
            if duration is None and begin is not None and end is not None:
                duration = _finite_number(end, "segment.end_time") - begin
            if duration is None:
                continue
            duration = _finite_number(duration, "segment.duration")
            if duration <= 0.0:
                continue
            period = int(_field(bundle.time_grid, "interval_of")(begin)) if begin is not None else ell
            charging_cost += (
                p.electricity_price[int(station)][period]
                * _finite_number(power_kw, "segment.power_kw")
                * duration
            )
        reservation_failure_cost = p.reservation_failure_penalty * sum(
            1
            for request_id in timeout_ids
            if request_id in request_by_id
            and _kind_name(request_by_id[request_id].kind) == "reservation"
        )
        terminal_soc, _, _ = self._slot_matrices(state, st.num_stations, st.num_slots)
        terminal_value = 0.0
        terminal_values = _field(event_window.rl_signals, "terminal_soc_value")
        for i in range(st.num_stations):
            for b in range(st.num_slots):
                terminal_value += _finite_number(
                    terminal_values[i][b], f"terminal_soc_value[{i}][{b}]"
                ) * terminal_soc[i][b]
        outside_value = _field(event_window.rl_signals, "outside_swap_value", None)
        if outside_value is not None:
            for request_id in pending_ids:
                req = request_by_id[request_id]
                terminal_value += _finite_number(
                    outside_value(req.station, req.return_soc),
                    f"outside_swap_value({req.station}, {req.return_soc})",
                )
        beta = _finite_number(p.terminal_value_weight, "terminal_value_weight")
        adjustment_cost = _finite_number(
            event_window.planned_adjustment_cost,
            "planned_adjustment_cost",
        )
        if adjustment_cost < 0.0:
            raise MPCInputError("planned_adjustment_cost cannot be negative")
        objective_total = (
            income_reservation + income_random - charging_cost
            - adjustment_cost - reservation_failure_cost + beta * terminal_value
        )

        first_services = [_record(item) for item in self._result_items(first_execution, "services")]
        first_segments = [
            _record(item) for item in self._result_items(first_execution, "charging_segments")
        ]
        _, first_ready, _ = self._slot_matrices(
            _field(first_execution, "state", state), st.num_stations, st.num_slots
        )
        first_power = [
            [bundle.requested_power[i][b][0] for b in range(st.num_slots)]
            for i in range(st.num_stations)
        ]
        first_stage = FirstStageExecution(
            period=ell,
            power_kw=first_power,
            ready=first_ready,
            available_full=[sum(row) for row in first_ready],
            assignments=sorted(
                first_services,
                key=lambda item: (item.get("station", -1), item.get("slot", -1), str(item.get("request_id", ""))),
            ),
            charging_segments=first_segments,
            state_after=copy.deepcopy(_field(first_execution, "state", state)),
            execution_result=first_execution,
        )
        context = event_window.reference_context
        path_search_enabled = bool(
            _field(context, "path_search_enabled", False)
            if context is not None
            else False
        )
        status = "EVENT_PATH_ENUM_REPLAY" if path_search_enabled else "EVENT_REPLAY"
        stats = {
            "model_kind": (
                "event_path_enumeration_mock"
                if path_search_enabled
                else "event_replay_mock"
            ),
            "candidate_event_count": bundle.candidate_event_count,
            "variable_count": bundle.variable_count,
            "constraint_count": bundle.constraint_count,
            "runtime_sec": solve_time,
            "mip_gap": 0.0,
            "best_bound": objective_total,
            "t_start": t_start,
            "t_end": t_end,
        }
        return EventMPCResult(
            period_ell=ell,
            horizon=bundle.horizon,
            status=status,
            # 这是给定 Mock 功率/候选的物理可行 incumbent，而非事件位置
            # MILP 的全局最优性声明。
            is_optimal=False,
            solve_time_sec=solve_time,
            objective_total=objective_total,
            income_reservation=income_reservation,
            income_random=income_random,
            charging_cost=charging_cost,
            adjustment_cost=adjustment_cost,
            reservation_failure_cost=reservation_failure_cost,
            terminal_value=terminal_value,
            terminal_value_weight=beta,
            request_outcomes=outcomes,
            events=all_events,
            first_stage=first_stage,
            terminal_state=copy.deepcopy(state),
            pending_request_ids=pending_ids,
            replay_initial_state=initial_state,
            replay_requests=list(bundle.requests),
            replay_requested_power=copy.deepcopy(first_power),
            time_grid=bundle.time_grid,
            event_engine=bundle.event_engine,
            model_statistics=stats,
        )

    def replay_first_interval(self, result: EventMPCResult) -> MPCReplayReport:
        """用同一连续事件内核重放首区间，并返回稳定字段的逐项比较。"""
        state = copy.deepcopy(result.replay_initial_state)
        period = result.period_ell
        arrivals = [
            req for req in result.replay_requests
            if req.active
            and self._contains_time(result.time_grid, req.arrival_time, period)
        ]
        try:
            replay = self._run_event_interval(
                result.event_engine,
                state,
                period,
                copy.deepcopy(result.replay_requested_power),
                self._engine_requests(arrivals),
            )
            replay_services = [
                _record(item) for item in self._result_items(replay, "services")
            ]
            expected_services = self._service_signature(result.first_stage.assignments)
            actual_services = self._service_signature(replay_services)
            expected_state = result.first_stage.state_after
            _, _, expected_slots = self._slot_matrices(
                expected_state, self.params.station.num_stations, self.params.station.num_slots
            )
            _, _, actual_slots = self._slot_matrices(
                _field(replay, "state", state),
                self.params.station.num_stations, self.params.station.num_slots,
            )
            matches = expected_services == actual_services and expected_slots == actual_slots
            message = "" if matches else "首区间事件服务或逐槽状态与重放不一致"
            return MPCReplayReport(
                matches=matches,
                expected_services=expected_services,
                replayed_services=actual_services,
                expected_slots=expected_slots,
                replayed_slots=actual_slots,
                message=message,
            )
        except Exception as exc:  # 报告错误而非把重放失败伪装为匹配。
            return MPCReplayReport(
                matches=False,
                expected_services=self._service_signature(result.first_stage.assignments),
                replayed_services=tuple(),
                expected_slots=tuple(),
                replayed_slots=tuple(),
                message=f"首区间重放异常: {exc}",
            )

    # ------------------------------------------------------------------


__all__ = [
    "PaperMPCError",
    "PaperMPCSolverUnavailable",
    "PaperMPCNoSolution",
    "PaperUserNetwork",
    "PaperReservationEvent",
    "StationPattern",
    "PaperMPCSolution",
    "solve_paper_mpc",
    "MPCError",
    "MPCInputError",
    "MPCNoSolutionError",
    "MPCEventRequest",
    "EventMPCWindowInput",
    "MPCReplayReport",
    "FirstStageExecution",
    "EventMPCModelBundle",
    "EventMPCResult",
    "MPCController",
]
