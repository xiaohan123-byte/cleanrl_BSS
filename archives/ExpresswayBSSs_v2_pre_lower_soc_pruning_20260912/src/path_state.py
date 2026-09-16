"""Remaining candidate networks and publication of boundary path decisions."""
from __future__ import annotations
from src.candidate_network import enumerate_paths, get_candidate_arcs, get_feasible_arcs
from src.domain import DomainError, UserNetwork, user_key_text

VIRTUAL_ORIGIN = 'origin'

def remaining_stations(params, state, user, plan):
    excluded = set(user.completed_stations)
    if user.waiting_request_id is not None:
        excluded.add(state.waiting[user.waiting_request_id].station)
    return tuple(i for i in (plan or []) if i not in excluded
                 and params.station.positions_km[i] >= user.position_km - 1e-9)

def build_remaining_network(params, state, user, network):
    od_index = next(i for i, od in enumerate(params.od_pairs) if od.od_id == user.user_key[0])
    od = params.od_pairs[od_index]
    current = remaining_stations(params, state, user,
                                 user.published_plan if user.entered else user.retained_plan)
    reference = remaining_stations(params, state, user,
                                   user.published_plan if user.entered else user.day_ahead)
    if not user.entered:
        origin = 'entry'
        arcs = get_feasible_arcs(network, od_index, user.entry_soc)
    else:
        inherited = get_candidate_arcs(network, od_index, user.entry_soc)
        waiting = state.waiting.get(user.waiting_request_id)
        origin = waiting.station if waiting is not None else VIRTUAL_ORIGIN
        position = user.position_km
        positions = {i: params.station.positions_km[i] for i in od.station_indices}
        positions['exit'] = od.exit_km
        arcs = [(a, b) for a, b in inherited if isinstance(a, int)
                and (a == origin or positions[a] > position + 1e-9)
                and positions[b] > position + 1e-9]
        if waiting is None:
            nodes = sorted({v for arc in inherited for v in arc if isinstance(v, int)})
            for station in nodes:
                distance = positions[station] - position
                if distance > 1e-9 and distance / params.range_km <= user.soc + 1e-10:
                    arcs.append((origin, station))
            if user.soc - (od.exit_km - position) / params.range_km >= params.min_exit_soc - 1e-10:
                arcs.append((origin, 'exit'))
            # Spacing uses the last actual swap/entry, never the moving virtual origin.
            short = sorted((arc for arc in arcs if arc[0] == origin and isinstance(arc[1], int)
                            and positions[arc[1]] - user.last_swap_position_km < params.min_swap_spacing_km),
                           key=lambda arc: (positions[arc[1]] - user.last_swap_position_km, arc[1]))
            for arc in short:
                alternative = [edge for edge in arcs if edge != arc]
                if enumerate_paths(alternative, origin=origin):
                    arcs = alternative
    paths = enumerate_paths(arcs, origin=origin)
    if not paths:
        raise DomainError(f'no reachable remaining path for {user_key_text(user.user_key)}')
    valid = {edge for path in paths for edge in path}
    arcs = sorted(valid, key=lambda edge: (str(edge[0]), str(edge[1])))
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
        if selected != sorted(selected, key=lambda i: params.station.positions_km[i]):
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
            distance = params.station.positions_km[selected[0]] - user.position_km
            user.next_arrival_time = max(now, user.entry_time) + distance / params.vehicle_speed_kmh
        else:
            user.next_arrival_time = None
    return events
