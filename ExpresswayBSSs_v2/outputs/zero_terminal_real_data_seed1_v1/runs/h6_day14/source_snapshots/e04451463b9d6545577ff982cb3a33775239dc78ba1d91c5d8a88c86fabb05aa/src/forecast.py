"""Forecasts use reports and observed motion, never future scenario truth."""
from __future__ import annotations
from .parameters import execution_period_limit, prediction_horizon

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import ceil
import numpy as np
from .parameters import BusinessParameters
from .scenario import ObservationView


@dataclass(frozen=True)
class Forecast:
    random_requests: list[dict]
    start_time: float
    end_time: float


def user_direction(params, user):
    od = next((od for od in getattr(params, "od_pairs", []) if od.od_id == user.user_key[0]), None)
    if od is None:
        return 1
    if hasattr(od, "entry_km") and hasattr(od, "exit_km"):
        return 1 if od.exit_km > od.entry_km else -1
    stations = od.station_indices
    return 1 if len(stations) < 2 or params.station.positions_km[stations[-1]] > params.station.positions_km[stations[0]] else -1


def physical_nodes(params):
    if hasattr(params, "physical_nodes"):
        return list(params.physical_nodes())
    nodes = set(params.station.positions_km)
    for od in params.od_pairs:
        if hasattr(od, "entry_km"):
            nodes.add(od.entry_km)
            nodes.add(od.exit_km)
    return sorted(nodes)


def segment_at(params, position, direction=1):
    nodes = physical_nodes(params)
    index = bisect_right(nodes, position) - 1 if direction > 0 else bisect_left(nodes, position) - 1
    return index if 0 <= index < len(nodes) - 1 else None


def road_segments(params, start, target):
    """Return fixed physical segments intersected by a directed road interval."""
    if target == start:
        return []
    direction = 1 if target > start else -1
    nodes = physical_nodes(params)
    middle = sorted((x for x in nodes if min(start, target) < x < max(start, target)), reverse=direction < 0)
    points = [start, *middle, target]
    return [(bisect_right(nodes, (a + b) / 2) - 1, a, b) for a, b in zip(points, points[1:])]


def predicted_entry_time(params, user, now):
    """Use the original report, then the remaining supported interval midpoint."""
    if user.entered:
        return now
    report = user.entry_time
    if not getattr(params, "terminal_experiment", False):
        return max(now, report)
    if report > now:
        return report
    error = params.entry_time_error_hours
    upper = min(report + error, params.num_periods * params.interval_hours)
    lower = max(now, report - error, 0.)
    if upper <= lower:
        raise ValueError(f"unobserved reservation entry lies outside its report support: {user.user_key}")
    return (lower + upper) / 2.


def _prediction_speed(params, user, index):
    if (getattr(params, "terminal_experiment", False) and user.entered
            and user.observed_segment_index == index
            and user.observed_speed_kmh is not None):
        return user.observed_speed_kmh
    return params.vehicle_speed_kmh


def predict_travel_time(params, user, start_position, target_position):
    if not getattr(params, "terminal_experiment", False):
        return abs(target_position - start_position) / params.vehicle_speed_kmh
    return sum(abs(b - a) / _prediction_speed(params, user, index)
               for index, a, b in road_segments(params, start_position, target_position))


def forecast_motion(params, user, target_station, end_time, departure_station=None,
                    departure_time=None, now=None):
    """Predict a vehicle until arrival; keep it at the requested station thereafter.

    Explicit departure_station/time represents an upstream predicted swap, so the
    vehicle starts there with a full battery. Otherwise use visible current state.
    """
    now = (user.observation_time or 0.) if now is None else now
    if departure_station is None:
        position = user.position_km
        soc = user.soc if user.entered else user.entry_soc
        start = now if user.entered else predicted_entry_time(params, user, now)
        anchor = user.last_swap_position_km
    else:
        if departure_time is None:
            raise ValueError("a predicted upstream service needs its departure time")
        position = params.station.positions_km[departure_station]
        soc, start, anchor = 1., departure_time, position
    target = params.station.positions_km[target_station]
    arrival = start + predict_travel_time(params, user, position, target)
    budget = max(0., end_time - start)
    result_position = position
    if getattr(params, "terminal_experiment", False):
        segments = road_segments(params, position, target)
    else:
        segments = [(None, position, target)]
    for index, a, b in segments:
        speed = _prediction_speed(params, user, index) if index is not None else params.vehicle_speed_kmh
        length = abs(b - a)
        elapsed = length / speed
        if budget >= elapsed:
            result_position = b
            budget -= elapsed
        else:
            direction = 1 if b >= a else -1
            result_position = a + direction * budget * speed
            break
    result_soc = soc - abs(result_position - position) / params.range_km
    return dict(position_km=result_position, soc=result_soc,
                entered=bool(user.entered or end_time >= start),
                last_swap_position_km=anchor, arrival_time=arrival)


def deterministic_random_requests(params, start_time, end_time):
    start, end = start_time, end_time
    requests = []
    duration = params.num_periods * params.interval_hours
    for station in range(params.station.num_stations):
        cumulative, index = 0., 0
        for hour in range(ceil(duration)):
            right = min(hour + 1., duration)
            rate = params.random_rate_at(station, float(hour))
            upper = cumulative + rate * (right - hour)
            while rate > 0 and index + .5 <= upper + 1e-12:
                arrival = hour + (index + .5 - cumulative) / rate
                if start < arrival < end and arrival < duration:
                    requests.append(dict(request_id=f"forecast:{station}:{index}", station=station,
                                         arrival_time=arrival, return_soc=params.random_soc_prediction))
                index += 1
            cumulative = upper
    return sorted(requests, key=lambda record: (record["arrival_time"], record["request_id"]))


def build_forecast(params: BusinessParameters, observation: ObservationView, ell: int, horizon: int) -> Forecast:
    if ell < 0 or horizon <= 0 or ell >= execution_period_limit(params):
        raise ValueError("forecast indices must lie within the operating grid")
    start = ell * params.interval_hours
    end = (ell + prediction_horizon(params, ell, horizon)) * params.interval_hours
    if abs(observation.now - start) > params.time_epsilon:
        raise ValueError("forecast observation time does not match the current round")
    if getattr(params, "terminal_experiment", False):
        return Forecast(deterministic_random_requests(params, start, end), start, end)
    if getattr(params, "finish_pending_after_demand", False):
        # Legacy stochastic forecasts also stop admitting new demand at day end.
        end = min(end, params.num_periods * params.interval_hours)
        if start >= end:
            return Forecast([], start, (ell + horizon) * params.interval_hours)
    # Keep the original baseline's independent per-round forecast stream.
    rng = np.random.default_rng(np.random.SeedSequence([params.seed, 303, ell]))
    requests = []
    for station, rate in enumerate(params.random_arrival_rate_per_hour):
        count = int(rng.poisson(rate * (end - start)))
        for index, arrival in enumerate(sorted(float(value) for value in rng.uniform(start, end, count))):
            requests.append({"request_id": f"forecast:{ell}:{station}:{index}", "station": station,
                             "arrival_time": arrival, "return_soc": float(rng.uniform(.05, .4))})
    requests.sort(key=lambda record: (record["arrival_time"], record["request_id"]))
    return Forecast(requests, start, end)
