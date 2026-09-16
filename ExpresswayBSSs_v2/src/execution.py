"""Execute one five-minute interval using only actual observed requests.

The optimiser chooses the batteries and boundary service times. This module
validates those actions, swaps first, and applies the specified constant power.
Arrivals between boundaries only enter the next round's waiting set.
"""
from __future__ import annotations

from math import isclose, isfinite, ulp
from typing import Any

from src.accounting import append_events
from src.domain import ExecutionResult, MPCSolution, RollingState, WaitingRequest, user_key_text
from src.path_state import apply_path_decisions
from src.forecast import user_direction, road_segments, segment_at, predict_travel_time


class ExecutionError(ValueError):
    """An action is not physically executable from the observed state."""


def _event(kind: str, event_id: str, n: int, time: float, **details) -> dict:
    return {"event_id": event_id, "type": kind, "period": n, "time": time,
            "realized": True, **details}


def _computed_grid_time(value, params):
    """Canonicalise only arithmetic roundoff of a computed grid boundary."""
    boundary = round(value / params.interval_hours) * params.interval_hours
    return boundary if abs(value - boundary) <= 4 * max(ulp(value), ulp(boundary)) else value


def _record_random(params, state, records, events, period, lower, upper, include_upper=False):
    eps = params.time_epsilon
    seen = set(state.seen_random_ids)
    for record in sorted(records, key=lambda item: (item["arrival_time"], item["request_id"])):
        arrival = float(record["arrival_time"])
        if not isfinite(arrival) or arrival < lower:
            raise ExecutionError("random arrival is outside the execution interval")
        if arrival > upper or (not include_upper and arrival >= upper):
            raise ExecutionError("future random truth cannot be admitted early")
        rid = str(record["request_id"])
        if rid in seen:
            continue
        station = int(record["station"])
        if not 0 <= station < params.station.num_stations:
            raise ExecutionError("random arrival has an invalid station")
        req = WaitingRequest(
            request_id=rid, station=station, kind="random", arrival_time=arrival,
            deadline=float(record.get("deadline", arrival + params.max_wait_hours)),
            return_soc=float(record["return_soc"]),
        )
        state.waiting[rid] = req
        state.seen_random_ids.append(rid)
        seen.add(rid)
        events.append(_event("random_arrival", f"arrival:{rid}", period, arrival,
                             request_id=rid, station=station, return_soc=req.return_soc,
                             deadline=req.deadline))


def _remaining_stations(params, user):
    route = user.published_plan if user.published_plan is not None else user.retained_plan
    direction = user_direction(params, user)
    return [i for i in route if i not in user.completed_stations
            and direction * (params.station.positions_km[i] - user.position_km) >= -params.time_epsilon]


def _scenario_records(scenario):
    return {} if scenario is None else {
        user_key_text(record["user_key"]): record for record in scenario.reservations}


def _move_user(params, state, user, start, end, events, period, truth=None):
    if user.status != "active" or user.waiting_request_id is not None:
        return
    eps = params.time_epsilon
    key = user_key_text(user.user_key)
    od = next(od for od in params.od_pairs if od.od_id == user.user_key[0])
    direction = user_direction(params, user)
    experimental = getattr(params, "terminal_experiment", False)
    if experimental and truth is None:
        raise ExecutionError("terminal experiment execution requires private scenario truth")
    actual_entry = float(truth.get("actual_entry_time", user.entry_time)) if truth is not None else user.entry_time
    clock = start
    if not user.entered:
        if actual_entry > end:
            return
        if actual_entry < start:
            raise ExecutionError("unobserved past reservation entry")
        clock = max(start, actual_entry)
        user.entered = True
        user.actual_entry_time = actual_entry
        user.soc = float(truth.get("actual_entry_soc", user.entry_soc)) if truth is not None else user.entry_soc
        user.position_km = od.entry_km
        user.published_plan = list(user.retained_plan)
        user.observed_segment_index = None
        user.observed_speed_kmh = None
        events.append(_event("reservation_entry", f"entry:{key}", period, actual_entry,
                             user_key=key, path=list(user.published_plan), actual_entry_soc=user.soc))
        if experimental and not user.retained_plan:
            # An admitted empty service plan has no remaining swapping task.
            # Retire it before the next optimizer can add a new station.
            user.status = "completed"
            user.next_arrival_time = None
            user.observation_time = actual_entry
            return
    route = _remaining_stations(params, user)
    target_station = route[0] if route else None
    target = params.station.positions_km[target_station] if target_station is not None else od.exit_km
    origin = user.position_km
    distance = direction * (target - origin)
    if distance < -eps:
        raise ExecutionError("selected route points upstream")
    if user.soc - distance / params.range_km < -1e-7:
        raise ExecutionError("vehicle cannot reach its selected next destination")
    multipliers = truth.get("segment_time_multipliers") if truth is not None else None
    segments = road_segments(params, origin, target) if experimental else [(None, origin, target)]
    actual_segments = []
    for index, a, b in segments:
        multiplier = float(multipliers[index]) if multipliers is not None else 1.
        if not isfinite(multiplier) or multiplier <= 0:
            raise ExecutionError("invalid physical-segment travel-time multiplier")
        speed = params.vehicle_speed_kmh / multiplier
        actual_segments.append((index, a, b, speed))
    travel_time = sum(abs(b - a) / speed for _, a, b, speed in actual_segments)
    eta = _computed_grid_time(clock + travel_time, params)
    user.observation_time = end
    if eta > end:
        remaining = max(0., end - clock)
        position = origin
        for _, a, b, speed in actual_segments:
            duration = abs(b - a) / speed
            if remaining >= duration:
                position = b
                remaining -= duration
            else:
                position = a + direction * remaining * speed
                break
        travelled = abs(position - origin)
        user.position_km = round(position, 12)
        user.soc = round(user.soc - travelled / params.range_km, 12)
        # Infer speed only when two moving observations identify one physical
        # segment. Crossing a node invalidates the previous segment estimate.
        first_segment = segment_at(params, origin, direction)
        current_segment = segment_at(params, user.position_km, direction)
        if end > clock and travelled > 0 and current_segment == first_segment:
            user.observed_segment_index = current_segment
            user.observed_speed_kmh = travelled / (end - clock)
        elif current_segment != user.observed_segment_index:
            user.observed_segment_index = None
            user.observed_speed_kmh = None
        # The visible state contains a forecast, never the private full-route ETA.
        user.next_arrival_time = end + predict_travel_time(params, user, user.position_km, target)
        return
    user.position_km = target
    user.soc = round(user.soc - distance / params.range_km, 12)
    user.observed_segment_index = None
    user.observed_speed_kmh = None
    if target_station is None:
        if user.soc < params.min_exit_soc - 1e-7:
            raise ExecutionError("vehicle reached its exit below the required SOC")
        user.status = "completed"
        user.next_arrival_time = None
        events.append(_event("reservation_exit", f"exit:{key}", period, eta,
                             user_key=key, exit_soc=user.soc))
        return
    rid = f"A:{key}:{len(user.completed_stations)}"
    if rid in state.waiting:
        raise ExecutionError("reservation request identity was reused")
    req = WaitingRequest(rid, target_station, "reservation", eta,
                         _computed_grid_time(eta + params.max_wait_hours, params), user.soc, user.user_key)
    state.waiting[rid] = req
    user.waiting_request_id = rid
    user.next_arrival_time = None
    events.append(_event("reservation_arrival", f"arrival:{rid}", period, eta,
                         request_id=rid, station=target_station, user_key=key,
                         return_soc=req.return_soc, deadline=req.deadline))


def advance_to_boundary(params: Any, state: RollingState, random_arrivals: list[dict] | None = None, scenario=None) -> ExecutionResult:
    """Admit observations exactly at t_n before constructing the optimisation.

    A runner may call this repeatedly: observed random IDs are retained. No
    service, charging, or future movement happens in this function.
    """
    result = state.clone()
    now = result.period * params.interval_hours
    events: list[dict] = []
    for record in random_arrivals or []:
        if float(record["arrival_time"]) != now:
            raise ExecutionError("boundary admission only accepts current observations")
    _record_random(params, result, random_arrivals or [], events, result.period, now, now, True)
    truth = _scenario_records(scenario)
    for key, user in result.users.items():
        _move_user(params, result, user, now, now, events, result.period, truth.get(key))
    reward = append_events(params, result.ledger, events)
    result.validate()
    return ExecutionResult(result, events, reward)


def _validate_services(params, state, services, now):
    eps = params.time_epsilon
    chosen_ids = set()
    chosen_slots = set()
    eligible = {rid: req for rid, req in state.waiting.items()
                if req.arrival_time <= now and req.deadline >= now}
    for action in services:
        rid, i, b = action.request_id, action.station, action.slot
        if rid not in eligible:
            raise ExecutionError(f"only an observed waiting request can be served: {rid}")
        if rid in chosen_ids or (i, b) in chosen_slots:
            raise ExecutionError("a request or battery is assigned twice at one boundary")
        if not 0 <= i < params.station.num_stations or not 0 <= b < params.station.num_slots:
            raise ExecutionError("invalid service station or slot")
        if eligible[rid].station != i:
            raise ExecutionError("request cannot be served at another station")
        if not isclose(state.slot_soc[i][b], 1., abs_tol=1e-7, rel_tol=0.):
            raise ExecutionError("a service requires a full battery before the swap")
        chosen_ids.add(rid)
        chosen_slots.add((i, b))
    for rid in chosen_ids:
        request = eligible[rid]
        for qid, prior in eligible.items():
            if qid == rid or prior.station != request.station:
                continue
            higher = (prior.kind == "reservation" and request.kind == "random") or (
                prior.kind == request.kind and prior.arrival_time < request.arrival_time)
            if higher and qid not in chosen_ids:
                raise ExecutionError("a higher-priority waiting request must also be served")


def _validate_power(params, state, solution):
    if len(solution.power) != params.station.num_stations:
        raise ExecutionError("power must define every station")
    powers = []
    for i in range(params.station.num_stations):
        if len(solution.power[i]) != params.station.num_slots:
            raise ExecutionError("power must define every charging slot")
        row = []
        for b in range(params.station.num_slots):
            if not solution.power[i][b]:
                raise ExecutionError("first-period power is missing")
            power = float(solution.power[i][b][0])
            if not isfinite(power) or power < 0 or power > params.slot_power_limit(i, b) + 1e-7:
                raise ExecutionError("invalid slot charging power")
            row.append(power)
        if sum(row) > params.station_power_limit(i) + 1e-7:
            raise ExecutionError("station power limit exceeded")
        powers.append(row)
    return powers


def execute_step(params: Any, state: RollingState, solution: MPCSolution,
                 random_arrivals: list[dict] | None = None, scenario=None) -> ExecutionResult:
    """Apply the first control interval on a clone and return actual events."""
    if not 0 <= state.period < params.num_periods:
        raise ExecutionError("cannot execute outside the operating horizon")
    state.validate()
    result = state.clone()
    n = result.period
    now, end = n * params.interval_hours, (n + 1) * params.interval_hours
    events = list(apply_path_decisions(params, result, solution.paths))
    current_services = [item for item in solution.services if item.period == n]
    if any(item.period < n for item in solution.services):
        raise ExecutionError("solution contains a stale service decision")
    _validate_services(params, result, current_services, now)
    powers = _validate_power(params, result, solution)
    for action in current_services:
        request = result.waiting.pop(action.request_id)
        result.slot_soc[action.station][action.slot] = request.return_soc
        details = dict(request_id=request.request_id, station=action.station,
                       slot=action.slot, return_soc=request.return_soc,
                       arrival_time=request.arrival_time, deadline=request.deadline,
                       waiting_hours=max(0., now - request.arrival_time),
                       energy_kwh=params.battery_capacity_kwh * (1. - request.return_soc),
                       unit_price=params.swap_service_price[action.station][n])
        if request.kind == "reservation":
            key = user_key_text(request.user_key)
            user = result.users[key]
            user.waiting_request_id = None
            user.soc = 1.
            user.position_km = params.station.positions_km[action.station]
            user.last_swap_position_km = user.position_km
            user.completed_stations.append(action.station)
            user.retained_plan = [i for i in user.retained_plan if i != action.station]
            if user.published_plan is not None:
                user.published_plan = [i for i in user.published_plan if i != action.station]
            user.observed_segment_index = None
            user.observed_speed_kmh = None
            if getattr(params, "terminal_experiment", False) and not user.retained_plan:
                user.status = "completed"
                user.next_arrival_time = None
            details["user_key"] = key
        events.append(_event(f"{request.kind}_service", f"service:{request.request_id}", n, now, **details))
    for i, row in enumerate(powers):
        for b, power in enumerate(row):
            before = result.slot_soc[i][b]
            after = round(before + params.station.charging_efficiency * params.interval_hours
                          * power / params.battery_capacity_kwh, 12)
            if not -1e-7 <= after <= 1 + 1e-7:
                raise ExecutionError("specified constant power would overcharge a battery")
            # Canonicalise numerical feasibility residuals only; never alter power.
            if after < 0:
                after = 0.
            elif after > 1:
                after = 1.
            result.slot_soc[i][b] = after
            events.append(_event("charging", f"charging:{n}:{i}:{b}", n, now,
                                 station=i, slot=b, power_kw=power,
                                 energy_kwh=power * params.interval_hours,
                                 unit_price=params.electricity_price[i][n],
                                 start_soc=before, end_soc=after))
    _record_random(params, result, random_arrivals or [], events, n, now, end)
    truth = _scenario_records(scenario)
    for key, user in result.users.items():
        _move_user(params, result, user, now, end, events, n, truth.get(key))
    for rid, request in list(result.waiting.items()):
        # Equality at the next boundary is still eligible there. A deadline
        # equal to t_n expires only after that boundary's service opportunity.
        if request.deadline < end:
            del result.waiting[rid]
            details = dict(request_id=rid, station=request.station,
                           arrival_time=request.arrival_time, deadline=request.deadline)
            if request.kind == "reservation":
                key = user_key_text(request.user_key)
                user = result.users[key]
                if user.status != "active":
                    raise ExecutionError("reservation cannot fail more than once")
                user.status = "failed"
                user.waiting_request_id = None
                user.next_arrival_time = None
                user.retained_plan = []
                user.published_plan = [] if user.published_plan is not None else None
                details["user_key"] = key
                kind = "reservation_failure"
            else:
                kind = "random_timeout"
            events.append(_event(kind, f"timeout:{rid}", n, max(now, request.deadline), **details))
    reward = append_events(params, result.ledger, events)
    result.period += 1
    result.validate()
    return ExecutionResult(result, events, reward)
