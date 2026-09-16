"""Assemble arc-dependent requests without fixing downstream arrival times."""
from __future__ import annotations
from math import ulp
from src.domain import CandidateRequest, DomainError, MPCWindow, user_key_text
from src.path_state import build_remaining_network
from src.forecast import user_direction, predict_travel_time, predicted_entry_time

def _computed_arrival(params, value):
    """Normalize only the ETA arithmetic generated here, never observed times."""
    boundary = round(value / params.interval_hours) * params.interval_hours
    return boundary if abs(value - boundary) <= 4 * max(ulp(value), ulp(boundary)) else value


def _arc_id(key, source, station):
    return f'candidate:{key}:{source}:{station}'

def build_window(params, state, network, forecast, horizon=None):
    state.validate()
    ell = state.period
    horizon = min(params.horizon if horizon is None else horizon, params.num_periods - ell)
    if horizon <= 0:
        raise DomainError('empty prediction window')
    requests = []
    for observed in state.waiting.values():
        requests.append(CandidateRequest(
            observed.request_id, observed.station, observed.kind, observed.return_soc,
            user_key=observed.user_key, arrival_time=observed.arrival_time,
            observed=True, deadline=observed.deadline))
    networks = {}
    now = ell * params.interval_hours
    for key, user in state.users.items():
        if user.status != 'active':
            continue
        user_network = build_remaining_network(params, state, user, network)
        networks[key] = user_network
        origin = user_network.origin
        waiting = state.waiting.get(user.waiting_request_id)
        arrivals = {}
        for source, target in user_network.arcs:
            if isinstance(target, int):
                arrivals.setdefault(target, []).append(_arc_id(key, source, target))
        for source, target in user_network.arcs:
            if not isinstance(target, int):
                continue
            if source == origin:
                source_position = user.position_km
                departure_soc = 1.0 if waiting is not None else user.soc
                if not user.entered:
                    departure_soc = user.entry_soc
                predecessors = (waiting.request_id,) if waiting is not None else ()
            else:
                source_position = params.station.positions_km[source]
                departure_soc = 1.0
                predecessors = tuple(arrivals.get(source, []))
                if not predecessors:
                    raise DomainError('station arc without incoming predecessor request')
            distance = user_direction(params, user) * (params.station.positions_km[target] - source_position)
            travel = predict_travel_time(params, user, source_position, params.station.positions_km[target])
            rho = departure_soc - distance / params.range_km
            if rho < 0 and rho >= -1e-9:
                rho = 0.0
            start = now if user.entered else predicted_entry_time(params, user, now)
            arrival = None if predecessors else _computed_arrival(params, start + travel)
            requests.append(CandidateRequest(
                _arc_id(key, source, target), target, 'reservation', rho,
                user_key=user.user_key, arc=(source, target), predecessors=predecessors,
                arrival_time=arrival, travel_time=travel,
                deadline=None if arrival is None else _computed_arrival(params, arrival + params.max_wait_hours)))
    seen = {request.request_id for request in requests}
    observed_ids = set(state.seen_random_ids) | set(state.waiting)
    end = (ell + horizon) * params.interval_hours
    for record in forecast.random_requests:
        rid = record['request_id']
        if rid in seen or rid in observed_ids:
            continue
        arrival = float(record['arrival_time'])
        if arrival < now or arrival >= end or (getattr(params, 'terminal_experiment', False) and arrival == now):
            continue
        requests.append(CandidateRequest(
            rid, int(record['station']), 'random', float(record['return_soc']),
            arrival_time=arrival, observed=False,
            deadline=float(record.get('deadline', arrival + params.max_wait_hours))))
        seen.add(rid)
    if len({r.request_id for r in requests}) != len(requests):
        raise DomainError('duplicate request ID within prediction window')
    return MPCWindow(ell, horizon, state.clone(), networks, requests)
