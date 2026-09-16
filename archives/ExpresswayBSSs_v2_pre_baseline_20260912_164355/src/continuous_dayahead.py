"""Continuous-event day-ahead baseline planning.

This module intentionally owns the new schema rather than retrofitting the
old integer-period inventory routine.  It uses :class:`ContinuousEventEngine`
for every trial and for the final inventory trajectory, so the day-ahead
acceptance check has exactly the same charging, queue, deadline, and boundary
semantics as the rolling executor.

Only deterministic synthetic inputs and :class:`MockRLProvider` are accepted
in this revision.  No trained RL policy, actual random future, or optimiser
output is used to construct the reference trajectory.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from data_generation_test.candidate_network import get_feasible_arcs
from data_generation_test.parameter import (
    ENTRY_NODE,
    EXIT_NODE,
    BusinessParameters,
    NodeId,
)
from data_generation_test.rl_data import MockRLProvider, RLProvider
from src.domain import CandidateRequest, PhysicalRequestStatus, RequestKind, RollingState, SlotState
from src.event_engine import ContinuousEventEngine
from src.reference_rollout import (
    build_accepted_reservation_rollouts,
    flatten_reservation_events,
    reservation_dependency_map,
)
from src.time_grid import TimeGrid


SCHEMA_VERSION = 2
_EPS = 1e-9


class DayAheadPlanningError(ValueError):
    """Raised when a reference path or its continuous inventory replay is invalid."""


def _od_index_of(params: BusinessParameters, od_id: int) -> int:
    for index, od in enumerate(params.od_pairs):
        if int(od.od_id) == int(od_id):
            return index
    raise DayAheadPlanningError(f"unknown od_id={od_id}")


def _state_from_soc(initial_slot_soc: Sequence[Sequence[float]]) -> RollingState:
    return RollingState(
        now=0.0,
        slots=[
            [
                SlotState(station=station, slot=slot, soc=float(soc), last_update_time=0.0)
                for slot, soc in enumerate(row)
            ]
            for station, row in enumerate(initial_slot_soc)
        ],
    )


def _slot_soc(state: RollingState) -> List[List[float]]:
    return [[cell.soc for cell in row] for row in state.slots]


def _event_period(grid: TimeGrid, event: CandidateRequest) -> int:
    return grid.interval_of(event.arrival_time)


def _horizon_length(params: BusinessParameters, reservations: Iterable[Mapping[str, Any]]) -> int:
    """Return a finite replay length including all station deadlines.

    The extra two intervals are intentional: an event whose deadline falls at
    an interval boundary is processed at the next interval start under the
    half-open contract, rather than being silently discarded at a finite-grid
    endpoint.
    """

    furthest = params.num_periods * params.interval_hours
    for record in reservations:
        od_index = _od_index_of(params, int(record["od_id"]))
        entry_time = float(record["day_ahead_entry_time"])
        for station in params.od_pairs[od_index].station_indices:
            furthest = max(
                furthest,
                entry_time
                + params.travel_time_from_entry_hours(od_index, station)
                + params.max_wait_hours,
            )
    return max(params.num_periods, int(furthest / params.interval_hours) + 2)


def _random_candidates(
    params: BusinessParameters,
    forecast: Sequence[Sequence[Sequence[Mapping[str, Any]]]],
) -> List[CandidateRequest]:
    candidates: List[CandidateRequest] = []
    for station, periods in enumerate(forecast):
        for requests in periods:
            for raw in requests:
                request_id = str(raw["request_id"])
                arrival = float(raw["arrival_time"])
                candidates.append(
                    CandidateRequest(
                        request_id=request_id,
                        kind=RequestKind.RANDOM,
                        station=station,
                        arrival_time=arrival,
                        deadline=float(raw.get("deadline", arrival + params.max_wait_hours)),
                        return_soc=float(raw.get("return_soc", raw["arrival_soc"])),
                        event_id=f"dayahead-random:{station}:{request_id}",
                    )
                )
    return sorted(candidates, key=lambda event: (event.arrival_time, event.event_id or ""))


def _simulation_engine(params: BusinessParameters, grid: TimeGrid) -> ContinuousEventEngine:
    # Reference paths may reach a station after the priced operating day.  Such
    # out-of-day intervals retain their physical service/deadline semantics but
    # have no fabricated charging allowance or price signal.
    total_intervals = grid.num_intervals or params.num_periods
    energy_limits = [
        [
            params.station_energy_limit_kwh[station][period]
            if period < params.num_periods
            else 0.0
            for period in range(total_intervals)
        ]
        for station in range(params.station.num_stations)
    ]
    return ContinuousEventEngine(
        grid,
        battery_capacity_kwh=params.battery_capacity_kwh,
        charging_efficiency=params.station.charging_efficiency,
        max_wait_hours=params.max_wait_hours,
        slot_power_limit_kw=params.station.slot_power_limit_kw,
        station_energy_limit_kwh=energy_limits,
    )


def _status_name(value: Any) -> str:
    return value.value if isinstance(value, PhysicalRequestStatus) else str(value)


def _reservations_are_served_on_arrival(
    replay: Mapping[str, Any],
    reservation_events: Sequence[CandidateRequest],
) -> bool:
    """Return whether every accepted day-ahead event has a full battery on arrival.

    The day-ahead admission contract is stricter than online execution: an
    accepted reservation may not borrow ``max_wait_hours`` from its later
    physical queue.  The shared engine is still the authority for the actual
    event order, so we compare its service timestamps with each immutable
    candidate arrival rather than maintaining a separate inventory rule.
    """

    service_times: Dict[str, float] = {}
    for per_station in replay["service_log"]:
        for per_interval in per_station:
            for service in per_interval:
                if service.get("kind") == RequestKind.RESERVATION.value:
                    event_id = service.get("event_id")
                    if event_id is not None:
                        service_times[str(event_id)] = float(service["service_time"])
    for event in reservation_events:
        event_id = event.event_id or event.request_id
        service_time = service_times.get(event_id)
        if service_time is None or abs(service_time - event.arrival_time) > _EPS:
            return False
    return True


def simulate_dayahead_inventory(
    params: BusinessParameters,
    initial_slot_soc: Sequence[Sequence[float]],
    forecast: Sequence[Sequence[Sequence[Mapping[str, Any]]]],
    reservation_events: Sequence[CandidateRequest],
    rl_provider: Optional[RLProvider],
    num_sim_periods: int,
) -> Dict[str, Any]:
    """Replay a deterministic day-ahead trajectory with the shared event engine.

    This is public mostly for validation and tests.  It deliberately returns
    serialisable diagnostics, not a mutable planning state.
    """

    if num_sim_periods <= 0:
        raise DayAheadPlanningError("num_sim_periods must be positive")
    grid = TimeGrid(params.interval_hours, num_intervals=num_sim_periods)
    engine = _simulation_engine(params, grid)
    state = _state_from_soc(initial_slot_soc)
    random_events = _random_candidates(params, forecast)
    all_events = sorted(
        [*reservation_events, *random_events],
        key=lambda event: (event.arrival_time, event.event_id or ""),
    )
    state.reservation_dependencies.update(
        {
            upstream: list(children)
            for upstream, children in reservation_dependency_map_from_events(reservation_events).items()
        }
    )

    station_count = params.station.num_stations
    slot_soc_end: List[List[List[float]]] = [
        [[] for _ in range(num_sim_periods)] for _ in range(station_count)
    ]
    full_after_reservation = [[0] * num_sim_periods for _ in range(station_count)]
    full_after_random = [[0] * num_sim_periods for _ in range(station_count)]
    service_log: List[List[List[Dict[str, Any]]]] = [
        [[] for _ in range(num_sim_periods)] for _ in range(station_count)
    ]
    timeout_log: List[List[List[Dict[str, Any]]]] = [
        [[] for _ in range(num_sim_periods)] for _ in range(station_count)
    ]
    charging_log: List[List[List[Dict[str, Any]]]] = [
        [[] for _ in range(num_sim_periods)] for _ in range(station_count)
    ]

    provider = rl_provider if rl_provider is not None else MockRLProvider(params)
    for period in range(num_sim_periods):
        if period < params.num_periods:
            signals = provider.get_signals(
                params,
                period_ell=period,
                horizon=1,
                soc_obs=_slot_soc(state),
            )
            power = [
                [signals.requested_power[station][slot][0] for slot in range(params.station.num_slots)]
                for station in range(station_count)
            ]
        else:
            power = [[0.0] * params.station.num_slots for _ in range(station_count)]

        start, end = grid.interval(period)
        arrivals = [event for event in all_events if start <= event.arrival_time < end]
        result = engine.simulate_interval(
            state,
            period,
            power,
            arrivals,
            realized=False,
            in_place=True,
        )
        state = result.state

        for service in result.services:
            request = service.request
            service_log[service.station][period].append(
                {
                    "kind": request.kind.value,
                    "request_id": request.request_id,
                    "event_id": request.event_id,
                    "user_key": list(request.user_key) if request.user_key is not None else None,
                    "slot": service.slot,
                    "return_soc": request.return_soc,
                    "service_time": service.occurred_at,
                    "arrival_time": request.arrival_time,
                    "deadline": request.deadline,
                }
            )
        for timeout in result.timeouts:
            request = timeout.request
            timeout_log[request.station][period].append(
                {
                    "kind": request.kind.value,
                    "request_id": request.request_id,
                    "event_id": request.event_id,
                    "user_key": list(request.user_key) if request.user_key is not None else None,
                    "occurred_at": timeout.occurred_at,
                    "arrival_time": request.arrival_time,
                    "deadline": request.deadline,
                }
            )
        for segment in result.charging_segments:
            charging_log[segment.station][period].append(segment.to_dict())
        for station, row in enumerate(state.slots):
            ready_count = sum(
                1 for cell in row if cell.ready and cell.soc >= 1.0 - _EPS
            )
            # These diagnostics record the state at the interval's right
            # limit.  Continuous service may recharge again after a
            # reservation, so a single interior snapshot would be misleading;
            # the full event log above is authoritative.
            full_after_reservation[station][period] = ready_count
            full_after_random[station][period] = ready_count
            slot_soc_end[station][period] = [cell.soc for cell in row]

    event_status = {
        str(key): _status_name(value)
        for key, value in sorted(state.request_status.items())
    }
    reservation_ids = [event.event_id or event.request_id for event in reservation_events]
    unmet = sum(
        event_status.get(event_id) != PhysicalRequestStatus.SERVED.value
        for event_id in reservation_ids
    )
    return {
        "num_sim_periods": num_sim_periods,
        "slot_soc_end": slot_soc_end,
        "full_after_reservation": full_after_reservation,
        "full_after_random": full_after_random,
        "service_log": service_log,
        "timeout_log": timeout_log,
        "charging_log": charging_log,
        "event_status": event_status,
        "unmet_reservation_events": unmet,
        "final_state": state.to_dict(),
    }


def reservation_dependency_map_from_events(
    events: Sequence[CandidateRequest],
) -> Dict[str, List[str]]:
    """Build the event-engine dependency map without requiring plan records."""

    result: Dict[str, List[str]] = defaultdict(list)
    for event in events:
        if event.upstream_request_id is not None and event.event_id is not None:
            result[event.upstream_request_id].append(event.event_id)
    return {key: sorted(set(value)) for key, value in sorted(result.items())}


def _candidate_event(
    params: BusinessParameters,
    reservation: Mapping[str, Any],
    od_index: int,
    source: NodeId,
    station: int,
    path_order: int,
    previous_event_id: str | None,
) -> CandidateRequest:
    od_id = int(reservation["od_id"])
    reservation_id = int(reservation["reservation_id"])
    entry_soc = float(reservation["day_ahead_entry_soc"])
    entry_time = float(reservation["day_ahead_entry_time"])
    return_soc = (
        entry_soc if source == ENTRY_NODE else 1.0
    ) - params.soc_consumption(od_index, source, station)
    if return_soc < -_EPS or return_soc > 1.0 + _EPS:
        raise DayAheadPlanningError(
            f"reservation {reservation_id} reaches station {station} with invalid SOC {return_soc}"
        )
    event_id = CandidateRequest.reservation_event_id(
        (od_id, reservation_id), path_order, station
    )
    arrival = entry_time + params.travel_time_from_entry_hours(od_index, station)
    return CandidateRequest(
        request_id=event_id,
        kind=RequestKind.RESERVATION,
        station=station,
        arrival_time=arrival,
        deadline=arrival + params.max_wait_hours,
        return_soc=min(1.0, max(0.0, return_soc)),
        user_key=(od_id, reservation_id),
        source_arc=(source, station),
        path_order=path_order,
        event_id=event_id,
        upstream_request_id=previous_event_id,
    )


def _path_record(
    reservation: Mapping[str, Any],
    path_arcs: Sequence[Tuple[NodeId, NodeId]],
    events: Sequence[CandidateRequest],
    grid: TimeGrid,
) -> Dict[str, Any]:
    nodes = [path_arcs[0][0]] + [target for _, target in path_arcs] if path_arcs else []
    return {
        "reservation_id": int(reservation["reservation_id"]),
        "request_id": str(reservation.get("request_id", f"reservation_{reservation['reservation_id']}")),
        "od_id": int(reservation["od_id"]),
        "user_key": [int(reservation["od_id"]), int(reservation["reservation_id"])],
        "day_ahead_entry_time": float(reservation["day_ahead_entry_time"]),
        "day_ahead_entry_soc": float(reservation["day_ahead_entry_soc"]),
        "accepted": True,
        "reject_reason": None,
        "path_nodes": nodes,
        "path_arcs": [[source, target] for source, target in path_arcs],
        "swap_stations": [event.station for event in events],
        "swap_times": [event.arrival_time for event in events],
        "swap_periods": [_event_period(grid, event) for event in events],
        "return_socs": [event.return_soc for event in events],
        "event_ids": [event.event_id for event in events],
    }


def _rejected_record(reservation: Mapping[str, Any], reason: str) -> Dict[str, Any]:
    return {
        "reservation_id": int(reservation["reservation_id"]),
        "request_id": str(reservation.get("request_id", f"reservation_{reservation['reservation_id']}")),
        "od_id": int(reservation["od_id"]),
        "user_key": [int(reservation["od_id"]), int(reservation["reservation_id"])],
        "day_ahead_entry_time": float(reservation["day_ahead_entry_time"]),
        "day_ahead_entry_soc": float(reservation["day_ahead_entry_soc"]),
        "accepted": False,
        "reject_reason": reason,
        "path_nodes": [],
        "path_arcs": [],
        "swap_stations": [],
        "swap_times": [],
        "swap_periods": [],
        "return_socs": [],
        "event_ids": [],
    }


def _try_plan_reservation_continuous(
    params: BusinessParameters,
    candidate_network: Mapping[str, Any],
    initial_slot_soc: Sequence[Sequence[float]],
    forecast: Sequence[Sequence[Sequence[Mapping[str, Any]]]],
    reservation: Mapping[str, Any],
    accepted_events: Sequence[CandidateRequest],
    rl_provider: RLProvider,
    num_sim_periods: int,
) -> Tuple[Dict[str, Any], List[CandidateRequest]]:
    """Atomically choose a strict-feasible reference path for one reservation."""

    reservation_id = int(reservation["reservation_id"])
    od_id = int(reservation["od_id"])
    od_index = _od_index_of(params, od_id)
    entry_soc = float(reservation["day_ahead_entry_soc"])
    grid = TimeGrid(params.interval_hours, num_intervals=num_sim_periods)
    try:
        feasible = set(get_feasible_arcs(dict(candidate_network), od_index, entry_soc))
    except ValueError as exc:
        return _rejected_record(reservation, f"no feasible individual path: {exc}"), []

    positions = {
        node: params.node_position_km(od_index, node)
        for node in params.od_nodes(od_index)
    }
    current: NodeId = ENTRY_NODE
    path: List[Tuple[NodeId, NodeId]] = []
    tentative: List[CandidateRequest] = []
    while True:
        if (current, EXIT_NODE) in feasible:
            path.append((current, EXIT_NODE))
            return _path_record(reservation, path, tentative, grid), tentative

        candidates: List[Tuple[int, CandidateRequest, int]] = []
        for _, target in sorted(
            (arc for arc in feasible if arc[0] == current and arc[1] != EXIT_NODE),
            key=lambda arc: (positions[arc[1]], int(arc[1])),
        ):
            if not isinstance(target, int):
                continue
            try:
                event = _candidate_event(
                    params,
                    reservation,
                    od_index,
                    current,
                    target,
                    len(tentative),
                    tentative[-1].event_id if tentative else None,
                )
            except DayAheadPlanningError:
                continue
            replay = simulate_dayahead_inventory(
                params,
                initial_slot_soc,
                forecast,
                [*accepted_events, *tentative, event],
                rl_provider,
                num_sim_periods,
            )
            if (
                replay["unmet_reservation_events"]
                or not _reservations_are_served_on_arrival(
                    replay, [*accepted_events, *tentative, event]
                )
            ):
                continue
            period = _event_period(grid, event)
            margin = replay["full_after_random"][target][period]
            candidates.append((target, event, margin))
        if not candidates:
            return (
                _rejected_record(
                    reservation,
                    f"no strict inventory-feasible station downstream of {current!r}",
                ),
                [],
            )
        # Higher realised right-limit inventory, then the furthest downstream
        # station, then stable station ID.  The trajectory replay is the
        # feasibility authority; this margin is only a deterministic tie-break.
        candidates.sort(key=lambda item: (-item[2], -positions[item[0]], item[0]))
        target, event, _ = candidates[0]
        path.append((current, target))
        tentative.append(event)
        current = target


def generate_continuous_dayahead_plan(
    params: BusinessParameters,
    candidate_network: Mapping[str, Any],
    mock_data: Mapping[str, Any],
    rl_provider: Optional[RLProvider] = None,
) -> Dict[str, Any]:
    """Generate schema-2 six-station reference paths with continuous replay."""

    params.validate()
    if mock_data.get("data_source") != "synthetic" or mock_data.get("signal_source") != "mock":
        raise DayAheadPlanningError("this revision accepts only synthetic/mock input data")
    provider = rl_provider if rl_provider is not None else MockRLProvider(params)
    reservations = sorted(
        list(mock_data["reservations"]), key=lambda item: int(item["reservation_id"])
    )
    forecast = mock_data["day_ahead_random_forecast"]
    initial_slot_soc = mock_data["initial_slot_soc"]
    num_sim_periods = _horizon_length(params, reservations)

    accepted_events: List[CandidateRequest] = []
    records: List[Dict[str, Any]] = []
    for reservation in reservations:
        record, new_events = _try_plan_reservation_continuous(
            params,
            candidate_network,
            initial_slot_soc,
            forecast,
            reservation,
            accepted_events,
            provider,
            num_sim_periods,
        )
        records.append(record)
        accepted_events.extend(new_events)

    final_replay = simulate_dayahead_inventory(
        params,
        initial_slot_soc,
        forecast,
        accepted_events,
        provider,
        num_sim_periods,
    )
    if final_replay["unmet_reservation_events"] or not _reservations_are_served_on_arrival(
        final_replay, accepted_events
    ):
        raise DayAheadPlanningError(
            "accepted day-ahead reservation was not served exactly at arrival"
        )

    visits = [[0] * num_sim_periods for _ in range(params.station.num_stations)]
    for event in accepted_events:
        visits[event.station][_event_period(TimeGrid(params.interval_hours, num_sim_periods), event)] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "src.continuous_dayahead",
        "engine": "continuous_event_v2",
        "seed": mock_data.get("seed"),
        "data_source": "synthetic",
        "signal_source": "mock",
        "num_periods": params.num_periods,
        "num_sim_periods": num_sim_periods,
        "reservations": records,
        "baseline_station_visits": visits,
        "inventory_trajectory": {
            key: final_replay[key]
            for key in (
                "num_sim_periods",
                "slot_soc_end",
                "full_after_reservation",
                "full_after_random",
                "service_log",
                "timeout_log",
                "charging_log",
                "event_status",
            )
        },
    }


def _validate_path_record(
    params: BusinessParameters,
    candidate_network: Mapping[str, Any],
    record: Mapping[str, Any],
    grid: TimeGrid,
) -> None:
    if not record.get("accepted"):
        for field in ("path_nodes", "path_arcs", "swap_stations", "event_ids"):
            if record.get(field):
                raise DayAheadPlanningError(
                    f"rejected reservation {record.get('reservation_id')} has residual {field}"
                )
        return
    od_index = _od_index_of(params, int(record["od_id"]))
    arcs = [tuple(arc) for arc in record["path_arcs"]]
    if not arcs or arcs[0][0] != ENTRY_NODE or arcs[-1][1] != EXIT_NODE:
        raise DayAheadPlanningError("accepted path must be a connected entry-to-exit path")
    if any(left[1] != right[0] for left, right in zip(arcs, arcs[1:])):
        raise DayAheadPlanningError("accepted path arcs are disconnected")
    feasible = set(get_feasible_arcs(dict(candidate_network), od_index, float(record["day_ahead_entry_soc"])))
    if not set(arcs).issubset(feasible):
        raise DayAheadPlanningError("accepted path contains an infeasible candidate arc")
    expected_nodes = [arcs[0][0]] + [arc[1] for arc in arcs]
    if list(record["path_nodes"]) != expected_nodes:
        raise DayAheadPlanningError("path_nodes does not match path_arcs")

    rollouts = build_accepted_reservation_rollouts(params, [record])
    if len(rollouts) != 1:
        raise DayAheadPlanningError("accepted reservation did not materialise one rollout")
    events = list(rollouts[0].events)
    expected = {
        "swap_stations": [event.station for event in events],
        "swap_times": [event.arrival_time for event in events],
        "swap_periods": [_event_period(grid, event) for event in events],
        "return_socs": [event.return_soc for event in events],
        "event_ids": [event.event_id for event in events],
    }
    for field, values in expected.items():
        actual = list(record.get(field, []))
        if len(actual) != len(values):
            raise DayAheadPlanningError(f"{field} has an invalid length")
        for actual_value, expected_value in zip(actual, values):
            if isinstance(expected_value, float):
                if abs(float(actual_value) - expected_value) > _EPS:
                    raise DayAheadPlanningError(f"{field} disagrees with continuous reference rollout")
            elif actual_value != expected_value:
                raise DayAheadPlanningError(f"{field} disagrees with continuous reference rollout")


def validate_continuous_dayahead_plan(
    plan: Mapping[str, Any],
    params: BusinessParameters,
    candidate_network: Mapping[str, Any],
    mock_data: Mapping[str, Any],
    rl_provider: Optional[RLProvider] = None,
) -> None:
    """Validate schema-2 plan fields and replay its full continuous trajectory."""

    params.validate()
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise DayAheadPlanningError(
            f"unsupported day-ahead schema {plan.get('schema_version')}; regenerate inputs"
        )
    if plan.get("data_source") != "synthetic" or plan.get("signal_source") != "mock":
        raise DayAheadPlanningError("plan provenance must be synthetic/mock")
    num_sim_periods = int(plan["num_sim_periods"])
    grid = TimeGrid(params.interval_hours, num_intervals=num_sim_periods)
    records = list(plan["reservations"])
    for record in records:
        _validate_path_record(params, candidate_network, record, grid)
    events = flatten_reservation_events(
        build_accepted_reservation_rollouts(params, records)
    )
    provider = rl_provider if rl_provider is not None else MockRLProvider(params)
    replay = simulate_dayahead_inventory(
        params,
        mock_data["initial_slot_soc"],
        mock_data["day_ahead_random_forecast"],
        events,
        provider,
        num_sim_periods,
    )
    if replay["unmet_reservation_events"] or not _reservations_are_served_on_arrival(
        replay, events
    ):
        raise DayAheadPlanningError(
            "accepted reservation was not served exactly at arrival in validation replay"
        )
    trajectory = plan["inventory_trajectory"]
    for field in (
        "num_sim_periods",
        "slot_soc_end",
        "full_after_reservation",
        "full_after_random",
        "service_log",
        "timeout_log",
        "charging_log",
        "event_status",
    ):
        if trajectory.get(field) != replay[field]:
            raise DayAheadPlanningError(f"inventory trajectory mismatch for {field}")
    visits = [[0] * num_sim_periods for _ in range(params.station.num_stations)]
    for event in events:
        visits[event.station][_event_period(grid, event)] += 1
    if plan.get("baseline_station_visits") != visits:
        raise DayAheadPlanningError("baseline_station_visits does not match accepted events")


__all__ = [
    "DayAheadPlanningError",
    "SCHEMA_VERSION",
    "generate_continuous_dayahead_plan",
    "reservation_dependency_map_from_events",
    "simulate_dayahead_inventory",
    "validate_continuous_dayahead_plan",
]
