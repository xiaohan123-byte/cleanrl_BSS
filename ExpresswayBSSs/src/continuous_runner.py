"""Paper-aligned event-driven rolling execution for the six-station scenario.

Each boundary solves one joint Gurobi MILP for all remaining user paths and
all station event patterns, verifies the selected first interval with the
continuous event kernel, and then advances physical state using actual arrivals
only.  Predicted requests and terminal values never enter the realised ledger.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from data_generation_test.parameter import BusinessParameters
from data_generation_test.rl_data import (
    MockRLProvider,
    ObservationView,
    RLProvider,
    SyntheticScenario,
)
from src.accounting import RealizedLedger
from src.domain import (
    CandidateRequest,
    EnrouteReservation,
    LedgerEntry,
    LedgerEventType,
    PhysicalRequestStatus,
    RequestKind,
    RollingState,
    SlotState,
)
from src.event_engine import ContinuousEventEngine
from src.event_path_search import EventPathSearchError, build_path_options
from src.mpc_model import EventMPCResult, EventMPCWindowInput, MPCController
from src.paper_mpc import (
    PaperMPCError,
    PaperMPCNoSolution,
    PaperMPCSolverUnavailable,
    solve_paper_mpc,
)
from src.path_state import (
    build_remaining_network,
    publish_if_changed,
    remaining_path_after_executed,
    station_sequence,
)
from src.reference_rollout import (
    ReferenceRolloutError,
    ReservationEventRollout,
    build_accepted_reservation_rollouts,
    build_reservation_rollout,
    events_in_prediction_window,
    flatten_reservation_events,
    reservation_dependency_map,
)
from src.time_grid import TimeGrid


SCHEMA_VERSION = 3
_EPS = 1e-9
UserKey = Tuple[int, int]


class ContinuousRunError(RuntimeError):
    """Raised when a synthetic event run violates the public execution contract."""


def _user_key(record: Mapping[str, Any]) -> UserKey:
    return (int(record["od_id"]), int(record["reservation_id"]))


def _key_text(key: UserKey) -> str:
    return f"{key[0]}:{key[1]}"


def _slot_soc(state: RollingState) -> List[List[float]]:
    return [[cell.soc for cell in row] for row in state.slots]


def _initial_state(params: BusinessParameters) -> RollingState:
    return RollingState(
        now=0.0,
        slots=[
            [
                SlotState(station=station, slot=slot, soc=soc, last_update_time=0.0)
                for slot, soc in enumerate(row)
            ]
            for station, row in enumerate(params.station.initial_slot_soc)
        ],
    )


def _engine(params: BusinessParameters, grid: TimeGrid) -> ContinuousEventEngine:
    return ContinuousEventEngine(
        grid,
        battery_capacity_kwh=params.battery_capacity_kwh,
        charging_efficiency=params.station.charging_efficiency,
        max_wait_hours=params.max_wait_hours,
        slot_power_limit_kw=params.station.slot_power_limit_kw,
        station_energy_limit_kwh=params.station_energy_limit_kwh,
    )


def _request_from_random(raw: Mapping[str, Any], station: int, *, source: str) -> CandidateRequest:
    arrival = float(raw["arrival_time"])
    request_id = f"{source}:{station}:{raw['request_id']}"
    return CandidateRequest(
        request_id=request_id,
        kind=RequestKind.RANDOM,
        station=station,
        arrival_time=arrival,
        deadline=float(raw.get("deadline", arrival)),
        return_soc=float(raw.get("return_soc", raw["arrival_soc"])),
        event_id=request_id,
    )


def _predicted_random_events(
    view: ObservationView,
    grid: TimeGrid,
    period: int,
    horizon: int,
) -> List[CandidateRequest]:
    events: List[CandidateRequest] = []
    for station, periods in enumerate(view.predicted_random_requests):
        for item_period in range(period, min(period + horizon, len(periods))):
            for raw in periods[item_period]:
                events.append(_request_from_random(raw, station, source="predicted"))
    return events_in_prediction_window(events, grid, period, horizon)


def _actual_random_events(
    payload: Mapping[str, Any],
    grid: TimeGrid,
    period: int,
) -> List[CandidateRequest]:
    start, end = grid.interval(period)
    events: List[CandidateRequest] = []
    for station, periods in enumerate(payload["actual_random_requests"]):
        if period >= len(periods):
            continue
        for raw in periods[period]:
            event = _request_from_random(raw, station, source="actual")
            if start <= event.arrival_time < end:
                events.append(event)
    return sorted(events, key=lambda event: (event.arrival_time, event.event_id or ""))


def _entry_index(entries: Iterable[Mapping[str, Any]]) -> Dict[UserKey, Mapping[str, Any]]:
    result: Dict[UserKey, Mapping[str, Any]] = {}
    for entry in entries:
        key = (int(entry["od_id"]), int(entry["reservation_id"]))
        if key in result:
            raise ContinuousRunError(f"duplicate visible reservation entry {key}")
        result[key] = entry
    return result


def _latest_visible_snapshot(
    view: ObservationView, key: UserKey
) -> Mapping[str, Any] | None:
    """Return the latest currently visible trajectory point for one user key."""

    candidates = [
        snapshot
        for snapshot in view.vehicle_snapshots.get(str(key[1]), [])
        if int(snapshot.get("od_id", key[0])) == key[0]
        and int(snapshot.get("reservation_id", key[1])) == key[1]
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda snapshot: float(snapshot.get("time", float("-inf"))))


def _refresh_enroute_snapshot(
    params: BusinessParameters,
    reservation: EnrouteReservation,
    snapshot: Mapping[str, Any] | None,
) -> None:
    """Apply only already-observed vehicle state, never scenario future truth."""

    if snapshot is None or reservation.waiting_request_id is not None:
        return
    if "position_km" in snapshot:
        reservation.current_position = float(snapshot["position_km"])
    if reservation.last_actual_swap_station is not None:
        swap_position = params.station.positions_km[reservation.last_actual_swap_station]
        distance_after_swap = max(0.0, reservation.current_position - swap_position)
        reservation.vehicle_soc = min(
            1.0, max(0.0, 1.0 - distance_after_swap / params.range_km)
        )
    elif "vehicle_soc" in snapshot:
        reservation.vehicle_soc = float(snapshot["vehicle_soc"])
    if "eta_to_exit_hours" in snapshot:
        reservation.known_eta["exit"] = float(snapshot["eta_to_exit_hours"])


def _update_user_phases(
    params: BusinessParameters,
    state: RollingState,
    plan_records: Mapping[UserKey, Mapping[str, Any]],
    view: ObservationView,
    terminal_users: Iterable[UserKey] = (),
) -> List[UserKey]:
    """Move newly visible users from ``future`` to ``enroute`` without truth leaks."""

    revealed = _entry_index(view.revealed_reservation_entries)
    terminal = set(terminal_users)
    newly_enroute: List[UserKey] = []
    for key, record in plan_records.items():
        text = _key_text(key)
        if key in terminal:
            state.future.pop(text, None)
            state.enroute.pop(text, None)
            continue
        snapshot = _latest_visible_snapshot(view, key)
        if key not in revealed:
            if text not in state.future and text not in state.enroute:
                state.future[text] = EnrouteReservation(
                    user_key=key,
                    current_position=0.0,
                    vehicle_soc=float(record["day_ahead_entry_soc"]),
                    dayahead_initial_path=list(record["path_arcs"]),
                    last_published_remaining_path=list(record["path_arcs"]),
                )
            continue
        if text in state.enroute:
            _refresh_enroute_snapshot(params, state.enroute[text], snapshot)
            continue
        entry = revealed[key]
        state.future.pop(text, None)
        state.enroute[text] = EnrouteReservation(
            user_key=key,
            current_position=float((snapshot or {}).get("position_km", 0.0)),
            vehicle_soc=float((snapshot or {}).get("vehicle_soc", entry["arrival_soc"])),
            dayahead_initial_path=list(record["path_arcs"]),
            last_published_remaining_path=list(record["path_arcs"]),
        )
        _refresh_enroute_snapshot(params, state.enroute[text], snapshot)
        newly_enroute.append(key)
    return newly_enroute


def _build_rollouts(
    params: BusinessParameters,
    plan: Mapping[str, Any],
    visible_entries: Iterable[Mapping[str, Any]],
) -> List[ReservationEventRollout]:
    try:
        return build_accepted_reservation_rollouts(
            params, plan, visible_entries=list(visible_entries)
        )
    except ReferenceRolloutError as exc:
        raise ContinuousRunError(f"unable to materialise reference reservation event: {exc}") from exc


def _new_actual_rollouts(
    params: BusinessParameters,
    plan_records: Mapping[UserKey, Mapping[str, Any]],
    visible_entries: Iterable[Mapping[str, Any]],
    cache: MutableMapping[UserKey, ReservationEventRollout],
) -> List[UserKey]:
    """Materialise actual paths only after their entry information is visible."""

    created: List[UserKey] = []
    for key, entry in _entry_index(visible_entries).items():
        if key in cache or key not in plan_records:
            continue
        try:
            cache[key] = build_reservation_rollout(
                params, plan_records[key], visible_entries=[entry]
            )
        except ReferenceRolloutError as exc:
            # This is an actual, observable route infeasibility, not a hidden
            # prediction result.  It is kept explicit rather than inventing a
            # route or mutating the immutable entry facts.
            raise ContinuousRunError(
                f"visible reservation {key} is infeasible on its day-ahead path: {exc}"
            ) from exc
        created.append(key)
    return created


def _sync_waiting_reservations(state: RollingState) -> None:
    """Mirror the physical reservation queue into the en-route route state."""

    for reservation in state.enroute.values():
        reservation.waiting_request_id = None
    for request in state.all_waiting_requests():
        if request.kind is not RequestKind.RESERVATION or request.user_key is None:
            continue
        reservation = state.enroute.get(_key_text(request.user_key))
        if reservation is None:
            continue
        if reservation.waiting_request_id is not None:
            raise ContinuousRunError(
                f"reservation {request.user_key} has more than one physical waiting request"
            )
        reservation.waiting_request_id = request.event_id


def _forecast_rollouts(
    params: BusinessParameters,
    plan: Mapping[str, Any],
    visible_entries: Iterable[Mapping[str, Any]],
    actual_rollouts: Mapping[UserKey, ReservationEventRollout],
    terminal_users: Iterable[UserKey],
) -> List[ReservationEventRollout]:
    """Use committed online paths for visible users and day-ahead paths for future users."""

    terminal = set(terminal_users)
    indexed = {
        rollout.user_key: rollout
        for rollout in _build_rollouts(params, plan, visible_entries)
        if rollout.user_key not in terminal
    }
    for key, rollout in actual_rollouts.items():
        if key not in terminal:
            indexed[key] = rollout
    return [indexed[key] for key in sorted(indexed)]


def _prediction_state(
    state: RollingState,
    rollouts: Iterable[ReservationEventRollout],
) -> RollingState:
    """Create an isolated forecast state with an exact, non-stale dependency graph."""

    predicted = state.clone()
    predicted.reservation_dependencies = reservation_dependency_map(rollouts)
    return predicted


def _path_state_record(
    params: BusinessParameters,
    state: RollingState,
    key: UserKey,
) -> Dict[str, Any]:
    reservation = state.enroute[_key_text(key)]
    waiting = (
        state.find_waiting(reservation.waiting_request_id)
        if reservation.waiting_request_id is not None
        else None
    )
    od = next(od for od in params.od_pairs if int(od.od_id) == key[0])
    last_swap_km = (
        params.station.positions_km[reservation.last_actual_swap_station]
        if reservation.last_actual_swap_station is not None
        else od.entry_km
    )
    return {
        "od_id": key[0],
        "user_id": key[1],
        "phase": "waiting" if waiting is not None else "enroute",
        "position_km": reservation.current_position,
        "vehicle_soc": reservation.vehicle_soc,
        "last_actual_swap_km": last_swap_km,
        "waiting_station": waiting.station if waiting is not None else None,
        "last_published_remaining_path": reservation.last_published_remaining_path,
    }


def _active_prediction_events(
    state: RollingState,
    events: Iterable[CandidateRequest],
) -> List[CandidateRequest]:
    """Do not inject an already realised or carried event into a prediction."""

    output: List[CandidateRequest] = []
    for event in events:
        identifier = event.event_id or event.request_id
        status = state.request_status.get(identifier)
        if status in (
            PhysicalRequestStatus.SERVED,
            PhysicalRequestStatus.TIMED_OUT,
            PhysicalRequestStatus.CANCELLED,
        ):
            continue
        if state.find_waiting(identifier) is not None:
            continue
        output.append(event)
    return output


def _solve_forecast(
    *,
    params: BusinessParameters,
    plan: Mapping[str, Any],
    visible_entries: Iterable[Mapping[str, Any]],
    actual_rollouts: Mapping[UserKey, ReservationEventRollout],
    terminal_users: Iterable[UserKey],
    state: RollingState,
    controller: MPCController,
    engine: ContinuousEventEngine,
    grid: TimeGrid,
    period: int,
    horizon: int,
    signals: Any,
    forecast_random: Sequence[CandidateRequest],
    planned_adjustment_cost: float,
    observation_time: float,
) -> Tuple[EventMPCResult, List[ReservationEventRollout]]:
    """Run one isolated fixed-power forecast for a selected route book."""

    rollouts = _forecast_rollouts(
        params,
        plan,
        visible_entries,
        actual_rollouts,
        terminal_users,
    )
    predicted_state = _prediction_state(state, rollouts)
    forecast_reservations = _active_prediction_events(
        state,
        events_in_prediction_window(
            flatten_reservation_events(rollouts), grid, period, horizon
        ),
    )
    window = EventMPCWindowInput(
        params=params,
        rolling_state=predicted_state,
        period_ell=period,
        rl_signals=signals,
        event_requests=[*forecast_reservations, *forecast_random],
        event_engine=engine,
        time_grid=grid,
        horizon=horizon,
        reference_context={
            "signal_source": "mock",
            "observation_time": observation_time,
            "path_search_enabled": True,
            "path_search_backend": "gurobi_joint_paper_milp",
        },
        planned_adjustment_cost=planned_adjustment_cost,
    )
    result = controller.solve_step(window)
    if not isinstance(result, EventMPCResult):
        raise ContinuousRunError("rolling controller did not select the event MPC branch")
    return result, rollouts


def _optimise_enroute_paths(
    *,
    params: BusinessParameters,
    network: Mapping[str, Any],
    plan: Mapping[str, Any],
    visible_entries: Iterable[Mapping[str, Any]],
    state: RollingState,
    actual_rollouts: Mapping[UserKey, ReservationEventRollout],
    terminal_users: Iterable[UserKey],
    controller: MPCController,
    engine: ContinuousEventEngine,
    grid: TimeGrid,
    period: int,
    horizon: int,
    signals: Any,
    forecast_random: Sequence[CandidateRequest],
    now: float,
    max_paths_per_user: int = 16,
) -> Tuple[Dict[UserKey, ReservationEventRollout], List[Dict[str, Any]], float, int]:
    """Coordinate-enumerate all current en-route users' remaining paths.

    Each candidate is scored by the same fixed-Mock-power continuous event
    forecast.  This is a deterministic candidate-set search, not a claim of
    global event-position MILP optimality.
    """

    selected = dict(actual_rollouts)
    decisions: List[Dict[str, Any]] = []
    planned_adjustment_cost = 0.0
    evaluations = 0

    for key in sorted(selected):
        reservation = state.enroute.get(_key_text(key))
        if reservation is None or key in set(terminal_users):
            continue
        previous_sequence = station_sequence(
            reservation.last_published_remaining_path
        )
        if reservation.waiting_request_id is not None:
            decisions.append(
                {
                    "user_key": list(key),
                    "status": "WAITING_PATH_FROZEN",
                    "previous_station_sequence": list(previous_sequence),
                    "proposed_station_sequence": list(previous_sequence),
                    "candidate_count": 0,
                    "evaluated_count": 0,
                    "adjusted": False,
                    "published": False,
                }
            )
            continue

        try:
            remaining = build_remaining_network(
                network,
                params,
                _path_state_record(params, state, key),
                now,
            )
            options = build_path_options(
                params,
                remaining,
                selected[key],
                now=now,
                position_km=reservation.current_position,
                vehicle_soc=reservation.vehicle_soc,
                request_status=state.request_status,
                max_paths=max_paths_per_user,
            )
        except (EventPathSearchError, ValueError) as exc:
            decisions.append(
                {
                    "user_key": list(key),
                    "status": "REFERENCE_PATH_RETAINED",
                    "reason": str(exc),
                    "previous_station_sequence": list(previous_sequence),
                    "proposed_station_sequence": list(previous_sequence),
                    "candidate_count": 0,
                    "evaluated_count": 0,
                    "adjusted": False,
                    "published": False,
                }
            )
            continue

        winner = None
        winner_prediction: EventMPCResult | None = None
        winner_changed = False
        candidate_scores: List[Dict[str, Any]] = []
        for option in options:
            candidate_book = dict(selected)
            candidate_book[key] = option.rollout
            changed = option.station_sequence != previous_sequence
            candidate_cost = planned_adjustment_cost + (
                params.path_adjustment_penalty if changed else 0.0
            )
            prediction, _ = _solve_forecast(
                params=params,
                plan=plan,
                visible_entries=visible_entries,
                actual_rollouts=candidate_book,
                terminal_users=terminal_users,
                state=state,
                controller=controller,
                engine=engine,
                grid=grid,
                period=period,
                horizon=horizon,
                signals=signals,
                forecast_random=forecast_random,
                planned_adjustment_cost=candidate_cost,
                observation_time=now,
            )
            evaluations += 1
            candidate_scores.append(
                {
                    "option_id": option.option_id,
                    "station_sequence": list(option.station_sequence),
                    "objective_total": prediction.objective_total,
                    "adjustment_cost": candidate_cost,
                }
            )
            if winner is None:
                choose = True
            else:
                assert winner_prediction is not None
                delta = prediction.objective_total - winner_prediction.objective_total
                choose = delta > 1e-9 or (
                    abs(delta) <= 1e-9
                    and (
                        int(changed),
                        option.option_id,
                    )
                    < (
                        int(winner_changed),
                        winner.option_id,
                    )
                )
            if choose:
                winner = option
                winner_prediction = prediction
                winner_changed = changed

        if winner is None or winner_prediction is None:  # defensive
            raise ContinuousRunError(f"path search produced no option for {key}")
        selected[key] = winner.rollout
        if winner_changed:
            planned_adjustment_cost += params.path_adjustment_penalty
        decisions.append(
            {
                "user_key": list(key),
                "status": "PATH_SELECTED",
                "search_method": "coordinate_enumeration",
                "previous_station_sequence": list(previous_sequence),
                "proposed_station_sequence": list(winner.station_sequence),
                "proposed_path": [[source, target] for source, target in winner.path_arcs],
                "candidate_count": len(options),
                "evaluated_count": len(options),
                "adjusted": winner_changed,
                "published": winner_changed,
                "winning_objective_total": winner_prediction.objective_total,
                "candidate_scores": candidate_scores,
            }
        )

    return selected, decisions, planned_adjustment_cost, evaluations


def _actual_reservation_events(
    rollouts: Mapping[UserKey, ReservationEventRollout],
    state: RollingState,
    grid: TimeGrid,
    period: int,
) -> List[CandidateRequest]:
    start, end = grid.interval(period)
    all_events = flatten_reservation_events(rollouts.values())
    return [
        event
        for event in _active_prediction_events(state, all_events)
        if start <= event.arrival_time < end
    ]


def _service_records(result: Any, kind: RequestKind) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for service in result.services:
        if service.request.kind is not kind:
            continue
        data = service.to_dict()
        data["user_key"] = (
            list(service.request.user_key) if service.request.user_key is not None else None
        )
        output.append(data)
    return output


def _timeout_records(result: Any, kind: RequestKind) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for timeout in result.timeouts:
        if timeout.request.kind is not kind:
            continue
        data = timeout.to_dict()
        data["user_key"] = (
            list(timeout.request.user_key) if timeout.request.user_key is not None else None
        )
        output.append(data)
    return output


def _record_path_publication(
    state: RollingState,
    key: UserKey,
    proposed_path: Sequence[Sequence[Any]],
    occurred_at: float,
    period: int,
) -> LedgerEntry | None:
    """Publish one winning remaining path when its physical station sequence changes."""

    reservation = state.enroute.get(_key_text(key))
    if reservation is None:
        return None
    publication = publish_if_changed(
        {
            "od_id": key[0],
            "user_id": key[1],
            "phase": "enroute",
            "last_published_remaining_path": reservation.last_published_remaining_path,
        },
        proposed_path,
        occurred_at,
    )
    if not publication.changed or publication.event_id is None:
        return None
    reservation.last_published_remaining_path = [list(arc) for arc in proposed_path]
    return LedgerEntry(
        event_id=publication.event_id,
        event_type=LedgerEventType.PATH_PUBLISHED,
        occurred_at=occurred_at,
        interval=period,
        user_key=key,
        # Let RealizedLedger resolve the configured path-adjustment price.
        # A literal ``amount=1`` would silently override a non-unit penalty.
        metadata={"path_changed": True},
    )


def _update_vehicle_progress(
    params: BusinessParameters,
    state: RollingState,
    execution: Any,
) -> None:
    for service in execution.services:
        key = service.request.user_key
        if key is None:
            continue
        reservation = state.enroute.get(_key_text(key))
        if reservation is None:
            continue
        if service.station not in reservation.executed_station_prefix:
            reservation.executed_station_prefix.append(service.station)
            reservation.last_published_remaining_path = [
                list(arc)
                for arc in remaining_path_after_executed(
                    reservation.last_published_remaining_path,
                    [service.station],
                )
            ]
        reservation.last_actual_swap_station = service.station
        reservation.current_position = params.station.positions_km[service.station]
        reservation.vehicle_soc = 1.0
        reservation.waiting_request_id = None
    for timeout in execution.timeouts:
        key = timeout.request.user_key
        if key is None:
            continue
        reservation = state.enroute.get(_key_text(key))
        if reservation is not None:
            reservation.waiting_request_id = None


def _reservation_outcome_sets(
    state: RollingState,
    rollouts: Mapping[UserKey, ReservationEventRollout],
) -> Tuple[List[UserKey], List[UserKey]]:
    completed: List[UserKey] = []
    failed: List[UserKey] = []
    for key, rollout in sorted(rollouts.items()):
        statuses = [
            state.request_status.get(event.event_id or event.request_id)
            for event in rollout.events
        ]
        if any(status is PhysicalRequestStatus.TIMED_OUT for status in statuses):
            failed.append(key)
        elif statuses and all(status is PhysicalRequestStatus.SERVED for status in statuses):
            completed.append(key)
    return completed, failed


def run_continuous_rolling_mpc(
    params: BusinessParameters,
    network: Mapping[str, Any],
    mock: Mapping[str, Any],
    plan: Mapping[str, Any],
    rl_provider: Optional[RLProvider] = None,
) -> Dict[str, Any]:
    """Execute all intervals with the paper Gurobi MILP and Mock RL signals."""

    params.validate()
    if params.station.num_stations != 6:
        raise ContinuousRunError("the executable revision is configured for six swap stations")
    if mock.get("schema_version") != 2 or mock.get("data_source") != "synthetic":
        raise ContinuousRunError("mock input must be regenerated as schema-2 synthetic data")
    if plan.get("schema_version") != 2 or plan.get("engine") != "continuous_event_v2":
        raise ContinuousRunError("day-ahead plan must be regenerated as continuous schema 2")

    scenario = SyntheticScenario.from_dict(mock)
    provider = rl_provider if rl_provider is not None else MockRLProvider(params)
    grid = TimeGrid(params.interval_hours, num_intervals=params.num_periods)
    engine = _engine(params, grid)
    state = _initial_state(params)
    controller = MPCController(params, dict(network), rl_provider=provider, dayahead_plan=dict(plan))
    ledger = RealizedLedger(
        grid,
        energy_price=params.electricity_price,
        reservation_service_price=params.swap_service_price,
        random_service_price=params.swap_service_price,
        reservation_failure_penalty=params.reservation_failure_penalty,
        path_adjustment_cost=params.path_adjustment_penalty,
        battery_capacity_kwh=params.battery_capacity_kwh,
    )
    plan_records: Dict[UserKey, Mapping[str, Any]] = {
        _user_key(record): record
        for record in plan["reservations"]
        if record.get("accepted")
    }
    actual_rollouts: Dict[UserKey, ReservationEventRollout] = {}
    terminal_users: set[UserKey] = set()
    rounds: List[Dict[str, Any]] = []

    for period in range(params.num_periods):
        start, end = grid.interval(period)
        if abs(state.now - start) > grid.boundary_tolerance:
            raise ContinuousRunError(
                f"state time {state.now} does not match rolling boundary {start}"
            )
        state_before = state.to_dict()
        view = scenario.observation_at(start)
        newly_enroute = _update_user_phases(
            params, state, plan_records, view, terminal_users
        )
        _new_actual_rollouts(params, plan_records, view.revealed_reservation_entries, actual_rollouts)
        _sync_waiting_reservations(state)

        # The forecast context receives only the isolated observation.  The
        # scenario payload remains owned by the execution side below.
        horizon = min(params.horizon, params.num_periods - period)
        forecast_random = _active_prediction_events(
            state, _predicted_random_events(view, grid, period, horizon)
        )
        signals = provider.get_signals(
            params,
            period_ell=period,
            horizon=horizon,
            soc_obs=_slot_soc(state),
            observation=view,
            rolling_state=None,
        )
        search_started = time.perf_counter()
        try:
            paper_solution = solve_paper_mpc(
                params=params,
                network=network,
                state=state,
                plan_records=plan_records,
                visible_entries=view.revealed_reservation_entries,
                current_rollouts=actual_rollouts,
                terminal_users=terminal_users,
                forecast_random=forecast_random,
                signals=signals,
                engine=engine,
                grid=grid,
                period=period,
                horizon=horizon,
                output_flag=int(getattr(params.solver, "output_flag", 0)),
            )
        except (PaperMPCSolverUnavailable, PaperMPCNoSolution, PaperMPCError) as exc:
            raise ContinuousRunError(
                f"paper Gurobi MILP failed at period {period}: {exc}"
            ) from exc
        path_search_time = time.perf_counter() - search_started

        # Future-user paths are internal prediction decisions only.  They are
        # deliberately supplied to the isolated forecast below but never
        # copied into the physical execution book.  En-route/waiting paths are
        # the only routes eligible for immediate publication and execution.
        path_decisions = paper_solution.path_decisions
        planned_adjustment_cost = paper_solution.adjustment_cost
        actual_rollouts = dict(paper_solution.selected_enroute_rollouts)
        prediction, _ = _solve_forecast(
            params=params,
            plan=plan,
            visible_entries=view.revealed_reservation_entries,
            actual_rollouts=paper_solution.selected_rollouts,
            terminal_users=terminal_users,
            state=state,
            controller=controller,
            engine=engine,
            grid=grid,
            period=period,
            horizon=horizon,
            signals=signals,
            forecast_random=forecast_random,
            planned_adjustment_cost=planned_adjustment_cost,
            observation_time=start,
        )
        replay_forecast_components = {
            "objective_total": prediction.objective_total,
            "income_reservation": prediction.income_reservation,
            "income_random": prediction.income_random,
            "charging_cost": prediction.charging_cost,
            "adjustment_cost": prediction.adjustment_cost,
            "reservation_failure_cost": prediction.reservation_failure_cost,
            "terminal_value": prediction.terminal_value,
        }
        paper_components = {
            "objective_total": paper_solution.objective_total,
            "income_reservation": paper_solution.income_reservation,
            "income_random": paper_solution.income_random,
            "charging_cost": paper_solution.charging_cost,
            "adjustment_cost": paper_solution.adjustment_cost,
            "reservation_failure_cost": paper_solution.reservation_failure_cost,
            "terminal_value": paper_solution.terminal_value,
        }
        component_deltas = {
            name: replay_forecast_components[name] - paper_components[name]
            for name in paper_components
        }
        replay_statistics = dict(prediction.model_statistics)
        prediction.status = f"PAPER_GUROBI_{paper_solution.status}"
        prediction.is_optimal = paper_solution.is_optimal
        prediction.objective_total = paper_solution.objective_total
        prediction.income_reservation = paper_solution.income_reservation
        prediction.income_random = paper_solution.income_random
        prediction.charging_cost = paper_solution.charging_cost
        prediction.adjustment_cost = paper_solution.adjustment_cost
        prediction.reservation_failure_cost = paper_solution.reservation_failure_cost
        prediction.terminal_value = paper_solution.terminal_value
        prediction.model_statistics = dict(paper_solution.model_statistics)
        prediction.model_statistics["fixed_route_replay_audit"] = {
            "model_kind": replay_statistics.get("model_kind"),
            "components": replay_forecast_components,
            "milp_minus_replay_sign_convention": "delta = replay - milp",
            "component_deltas": component_deltas,
            "first_interval_authority": "continuous_event_replay",
            "optimization_authority": "gurobi_milp",
        }
        replay = controller.replay_first_interval(prediction)
        if not replay.matches:
            raise ContinuousRunError(f"first-interval prediction replay failed: {replay.message}")
        prediction.model_statistics["path_search"] = {
            "method": "gurobi_joint_paper_milp",
            "formulation": "complete_station_pattern_extended_formulation",
            "solver_backend": "gurobi",
            "fixed_mock_power": True,
            "search_complete": True,
            "global_milp_optimality_claimed": paper_solution.is_optimal,
            "solver_status": paper_solution.status,
            "runtime_sec": path_search_time,
        }

        publication_entries: List[LedgerEntry] = []
        for decision in path_decisions:
            if not decision.get("adjusted") or not decision.get("will_publish"):
                continue
            key = (int(decision["user_key"][0]), int(decision["user_key"][1]))
            entry = _record_path_publication(
                state,
                key,
                decision["proposed_path"],
                start,
                period,
            )
            if entry is not None:
                publication_entries.append(entry)
                decision["publication_event_id"] = entry.event_id
                decision["published"] = True

        # Only the simulated actual stream reaches the physical state.  Future
        # actual arrivals are neither present in ``view`` nor fed to MPC.
        state.reservation_dependencies = reservation_dependency_map(
            rollout
            for key, rollout in actual_rollouts.items()
            if key not in terminal_users
        )
        actual_arrivals = [
            *_actual_reservation_events(actual_rollouts, state, grid, period),
            *_actual_random_events(mock, grid, period),
        ]
        actual_arrivals.sort(key=lambda event: (event.arrival_time, event.event_id or ""))
        execution = engine.simulate_interval(
            state,
            period,
            prediction.first_stage.power_kw,
            actual_arrivals,
            realized=True,
            in_place=True,
        )
        state = execution.state
        _update_vehicle_progress(params, state, execution)
        _sync_waiting_reservations(state)
        postings = ledger.submit_many([*publication_entries, *execution.ledger_entries])
        state.accounted_event_ids.update(posting.entry.event_id for posting in postings)

        completed_now, failed_now = _reservation_outcome_sets(state, actual_rollouts)
        terminal_users.update(completed_now)
        terminal_users.update(failed_now)
        for key in terminal_users:
            state.future.pop(_key_text(key), None)
            state.enroute.pop(_key_text(key), None)

        round_components = ledger.components_for_interval(period)
        reservation_services = _service_records(execution, RequestKind.RESERVATION)
        random_services = _service_records(execution, RequestKind.RANDOM)
        reservation_timeouts = _timeout_records(execution, RequestKind.RESERVATION)
        random_timeouts = _timeout_records(execution, RequestKind.RANDOM)
        rounds.append(
            {
                "period": period,
                "time_window": [start, end],
                "status": prediction.status,
                "model_kind": prediction.model_statistics.get("model_kind"),
                "prediction": {
                    "objective_total": prediction.objective_total,
                    "income_reservation": prediction.income_reservation,
                    "income_random": prediction.income_random,
                    "charging_cost": prediction.charging_cost,
                    "adjustment_cost": prediction.adjustment_cost,
                    "reservation_failure_cost": prediction.reservation_failure_cost,
                    "terminal_value": prediction.terminal_value,
                    "pending_request_ids": list(prediction.pending_request_ids),
                    "request_outcomes": dict(prediction.request_outcomes),
                    "event_count": len(prediction.events),
                },
                "model_statistics": dict(prediction.model_statistics),
                "replay": {
                    "matches": replay.matches,
                    "message": replay.message,
                    "expected_services": [list(item) for item in replay.expected_services],
                    "replayed_services": [list(item) for item in replay.replayed_services],
                },
                "actual": {
                    "power_kw": prediction.first_stage.power_kw,
                    "charging_segments": [segment.to_dict() for segment in execution.charging_segments],
                    "reservation_services": reservation_services,
                    "random_services": random_services,
                    "reservation_timeouts": reservation_timeouts,
                    "random_timeouts": random_timeouts,
                    "waiting_ids": [request.event_id for request in state.all_waiting_requests()],
                    "ledger_event_ids": [posting.entry.event_id for posting in postings],
                    "reward": round_components["reward_delta"],
                },
                "ledger": round_components,
                "path_search": dict(prediction.model_statistics["path_search"]),
                "path_decisions": path_decisions,
                "state_start": state_before,
                "state_end": state.to_dict(),
                "newly_enroute": [list(key) for key in newly_enroute],
                "solve_time_sec": prediction.solve_time_sec + path_search_time,
            }
        )

    completed, failed = _reservation_outcome_sets(state, actual_rollouts)
    summary = ledger.summary()
    reservation_service_count = sum(
        len(round_record["actual"]["reservation_services"]) for round_record in rounds
    )
    random_service_count = sum(
        len(round_record["actual"]["random_services"]) for round_record in rounds
    )
    reservation_timeout_count = sum(
        len(round_record["actual"]["reservation_timeouts"]) for round_record in rounds
    )
    path_publication_count = sum(
        1
        for round_record in rounds
        for decision in round_record["path_decisions"]
        if decision.get("publication_event_id")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "src.continuous_runner",
        "run_mode": "paper_gurobi_continuous_event_mpc",
        "seed": mock.get("seed"),
        "data_source": "synthetic",
        "signal_source": "mock",
        "num_stations": params.station.num_stations,
        "num_periods": params.num_periods,
        "horizon": params.horizon,
        "service_policy": {
            "time_domain": "half_open_continuous",
            "same_time_order": ["charging_complete", "arrival", "reservation_priority", "timeout"],
            "reservation_priority": True,
            "random_waiting": True,
            "power_source": "mock_fixed_parameter",
            "path_source": "gurobi_joint_path_flow",
            "path_search_optimality": "global_when_solver_status_optimal",
            "event_formulation": "complete_station_pattern_extended_milp",
            "station_energy_limit": "per_station_per_interval_kwh",
        },
        "rounds": rounds,
        "summary": {
            **{f"total_{key}": value for key, value in summary.items()},
            "total_reward": summary["reward_delta"],
            "total_actual_reservation_services": reservation_service_count,
            "total_actual_random_services": random_service_count,
            "total_actual_reservation_timeouts": reservation_timeout_count,
            "total_path_publications": path_publication_count,
            "accepted_reservations": [list(key) for key in sorted(plan_records)],
            "observed_enroute_reservations": [list(key) for key in sorted(actual_rollouts)],
            "completed_reservations": [list(key) for key in completed],
            "failed_reservations": [list(key) for key in failed],
            "final_waiting_request_ids": [request.event_id for request in state.all_waiting_requests()],
        },
        "path_search": {
            "status": rounds[-1]["status"] if rounds else "NOT_RUN",
            "method": "gurobi_joint_paper_milp",
            "solver_backend": "gurobi",
            "fixed_mock_power": True,
            "global_milp_optimality_claimed": bool(rounds) and all(
                round_record["path_search"].get("global_milp_optimality_claimed", False)
                for round_record in rounds
            ),
        },
        "ledger": ledger.to_dict(),
        "final_state": state.to_dict(),
    }


__all__ = ["ContinuousRunError", "SCHEMA_VERSION", "run_continuous_rolling_mpc"]
