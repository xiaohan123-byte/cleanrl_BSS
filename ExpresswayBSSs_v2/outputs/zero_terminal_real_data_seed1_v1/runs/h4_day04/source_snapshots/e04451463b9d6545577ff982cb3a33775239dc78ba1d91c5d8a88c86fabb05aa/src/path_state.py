"""Remaining candidate networks and publication of boundary path decisions."""
from __future__ import annotations
from src.candidate_network import build_user_arcs
from src.domain import DomainError, UserNetwork, user_key_text
from src.forecast import user_direction, predict_travel_time, predicted_entry_time

VIRTUAL_ORIGIN = 'origin'

def remaining_stations(params, state, user, plan):
    excluded = set(user.completed_stations)
    if user.waiting_request_id is not None:
        excluded.add(state.waiting[user.waiting_request_id].station)
    direction = user_direction(params, user)
    return tuple(i for i in (plan or []) if i not in excluded
                 and direction * (params.station.positions_km[i] - user.position_km) >= -1e-9)

def build_remaining_network(params, state, user, network):
    od_index = next(i for i, od in enumerate(params.od_pairs) if od.od_id == user.user_key[0])
    od = params.od_pairs[od_index]
    current = remaining_stations(params, state, user,
                                 user.published_plan if user.entered else user.retained_plan)
    reference = remaining_stations(params, state, user,
                                   user.published_plan if user.entered else user.day_ahead)
    waiting = state.waiting.get(user.waiting_request_id)
    if not user.entered:
        origin, position, soc = 'entry', od.entry_km, user.entry_soc
        if getattr(params, 'terminal_experiment', False):
            soc = max(.5, soc - params.entry_soc_error)
        last_swap = od.entry_km
    elif waiting is not None:
        origin = waiting.station
        position = params.station.positions_km[origin]
        soc, last_swap = 1., position
    else:
        origin, position, soc = VIRTUAL_ORIGIN, user.position_km, user.soc
        last_swap = user.last_swap_position_km
    arcs = build_user_arcs(network, od_index, origin, position, soc,
                           last_swap_position_km=last_swap,
                           protected_stations=current)
    if not arcs:
        raise DomainError(f'no reachable remaining path for {user_key_text(user.user_key)}')
    return UserNetwork(user.user_key, origin, arcs, reference, current,
                       state.period % params.path_update_interval == 0)

def apply_path_decisions(params, state, paths):
    """Update only plan fields; return chargeable changes actually published."""
    events = []
    now = state.period * params.interval_hours
    for key, stations in paths.items():
        if key not in state.users:
            raise DomainError(f'unknown user in path decision: {key}')
        user = state.users[key]
        if user.status != 'active':
            raise DomainError('cannot revise inactive user path')
        selected = list(stations)
        od = next(od for od in params.od_pairs if od.od_id == user.user_key[0])
        if len(set(selected)) != len(selected) or any(i not in od.station_indices for i in selected):
            raise DomainError('invalid station sequence')
        direction = user_direction(params, user)
        if selected != sorted(selected, key=lambda i: direction * params.station.positions_km[i]):
            raise DomainError('path must move downstream')
        if tuple(selected) != remaining_stations(params, state, user, selected):
            raise DomainError('path includes a completed, passed, or currently waiting station')
        old = remaining_stations(params, state, user,
                                 user.published_plan if user.entered else user.retained_plan)
        if state.period % params.path_update_interval and tuple(selected) != old:
            raise DomainError('path change outside the publication clock')
        user.retained_plan = selected.copy()
        if user.entered:
            if user.published_plan is not None and tuple(selected) != old:
                events.append(dict(
                    event_id=f'adjust:{state.period}:{key}', type='path_adjustment',
                    period=state.period, time=now, user_key=list(user.user_key),
                    old_path=list(old), new_path=selected.copy(), realized=True))
            user.published_plan = selected.copy()
        if user.waiting_request_id is None and selected:
            travel = predict_travel_time(params, user, user.position_km, params.station.positions_km[selected[0]])
            start = now if user.entered else predicted_entry_time(params, user, now)
            user.next_arrival_time = start + travel
        else:
            user.next_arrival_time = None
        if (getattr(params, 'terminal_experiment', False) and user.entered
                and user.waiting_request_id is None and not selected):
            user.status = 'completed'
    return events
