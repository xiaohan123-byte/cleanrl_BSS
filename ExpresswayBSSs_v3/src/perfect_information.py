"""Full-day clairvoyant inputs and execution replay, independent of MPC runs."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import ulp

from .accounting import summarize_ledger
from .candidate_network import build_user_arcs, generate_candidate_network
from .domain import (CandidateRequest, MPCSolution, MPCWindow, ServiceDecision,
                     UserNetwork, initial_state, user_key_text)
from .execution import advance_to_boundary, execute_step
from .forecast import road_segments
from .parameters import execution_period_limit


def grid_time(value, delta):
    boundary = round(value / delta) * delta
    return boundary if abs(value - boundary) <= 4 * max(ulp(value), ulp(boundary)) else value


def build_perfect_window(params, scenario, plans):
    """Use private truth only here; never mutate frozen scenario/report fields."""
    network = generate_candidate_network(params)
    state = initial_state(params, scenario.initial_reservations(), plans)
    networks, requests = {}, []
    delta = params.interval_hours
    for record in scenario.reservations:
        key = user_key_text(record['user_key'])
        od_index = params.od_index(record['od_id'])
        od = params.od_pairs[od_index]
        entry = record['actual_entry_time']
        actual_soc = record['actual_entry_soc']
        if entry <= 0:
            raise ValueError('fixed free initialization requires entry strictly after time zero')
        arcs = build_user_arcs(network, od_index, 'entry', od.entry_km, actual_soc,
                               protected_stations=plans[key])
        if not arcs:
            raise ValueError(f'no physically feasible route for {key}')
        initial_arcs = list(zip(['entry', *plans[key]], [*plans[key], 'exit']))
        if not set(initial_arcs) <= set(map(tuple, arcs)):
            raise ValueError(f'original day-ahead path is not protected: {key}')
        networks[key] = UserNetwork(tuple(record['user_key']), 'entry', arcs,
                                    tuple(plans[key]), tuple(plans[key]), True)
        incoming = defaultdict(list)
        for source, target in arcs:
            if isinstance(target, int):
                incoming[target].append(f'oracle:{key}:{source}:{target}')
        for source, target in arcs:
            if not isinstance(target, int):
                continue
            start = od.entry_km if source == 'entry' else params.station.positions_km[source]
            finish = params.station.positions_km[target]
            travel = sum(abs(b - a) / (params.vehicle_speed_kmh / record['segment_time_multipliers'][idx])
                         for idx, a, b in road_segments(params, start, finish))
            predecessors = () if source == 'entry' else tuple(incoming[source])
            arrival = grid_time(entry + travel, delta) if not predecessors else None
            rho = (actual_soc if source == 'entry' else 1.) - abs(finish - start) / params.range_km
            requests.append(CandidateRequest(
                f'oracle:{key}:{source}:{target}', target, 'reservation', max(0., rho),
                tuple(record['user_key']), (source, target), predecessors, arrival, travel,
                deadline=None if arrival is None else grid_time(arrival + params.max_wait_hours, delta)))
    for record in scenario.actual_random_requests:
        arrival = float(record['arrival_time'])
        if not 0 <= arrival < params.num_periods * delta:
            raise ValueError('new random demand must lie inside the demand day')
        requests.append(CandidateRequest(
            record['request_id'], int(record['station']), 'random', float(record['return_soc']),
            arrival_time=arrival, observed=arrival == 0.,
            deadline=float(record.get('deadline', arrival + params.max_wait_hours))))
    return MPCWindow(0, execution_period_limit(params), state, networks, requests)


def no_service_solution(params, window, plans):
    """A deterministic feasible starting schedule, including natural expiry."""
    failures = sum(bool(route) for route in plans.values())
    h = window.horizon
    return MPCSolution(
        'initial', -failures * params.reservation_failure_penalty,
        dict(income=0., charging_cost=0., adjustment_cost=0.,
             failure_cost=failures * params.reservation_failure_penalty),
        plans, [], [[[0.] * h for _ in row] for row in window.state.slot_soc],
        [[[soc] * (h + 1) for soc in row] for row in window.state.slot_soc])


def replay_solution(params, scenario, plans, window, solution):
    """Replay fixed decisions through the shared physics and accounting engine.

    Oracle arc IDs are mapped to real sequential request IDs only after actual
    arrival is observed. Every chosen service and its upstream station is checked.
    No optimization, online trajectory, or policy fallback is involved.
    """
    state = initial_state(params, scenario.initial_reservations(), plans)
    requests = {r.request_id: r for r in window.requests}
    services = defaultdict(list)
    for service in solution.services:
        if service.period < 0 or service.period >= window.horizon:
            raise ValueError('service outside full-day horizon')
        services[service.period].append(service)
    if set(solution.paths) != set(window.networks):
        raise ValueError('solution must define every reservation path')
    for key, route in solution.paths.items():
        arcs = list(zip(['entry', *route], [*route, 'exit']))
        if not set(arcs) <= set(map(tuple, window.networks[key].arcs)):
            raise ValueError(f'selected route outside protected truth network: {key}')
    ledger, state_checks = [], []
    max_soc_error, max_arrival_error, max_return_soc_error = 0., 0., 0.
    # The shared engine clones state/ledger. Keep each step's ledger short and
    # perform an independent global event/request uniqueness check afterwards.
    for n in range(window.horizon):
        now, end = n * params.interval_hours, (n + 1) * params.interval_hours
        boundary = [r for r in scenario.actual_random_requests if r['arrival_time'] == now]
        state = advance_to_boundary(params, state, boundary, scenario).state
        ledger.extend(state.ledger)
        state.ledger = []
        actual_services = []
        for item in services[n]:
            req = requests[item.request_id]
            rid = item.request_id
            if req.kind == 'reservation':
                user = state.users[user_key_text(req.user_key)]
                rid = user.waiting_request_id
                previous = user.completed_stations[-1] if user.completed_stations else 'entry'
                if req.arc != (previous, item.station):
                    raise ValueError('service does not follow selected path prefix')
            if rid not in state.waiting:
                raise ValueError(f'planned request has not arrived: {item.request_id}, period={n}')
            actual = state.waiting[rid]
            outcome = solution.request_outcomes.get(item.request_id, {})
            if outcome.get('arrival_time') is not None:
                error = abs(actual.arrival_time - outcome['arrival_time'])
                max_arrival_error = max(max_arrival_error, error)
                if error > 1e-8:
                    raise ValueError('true arrival differs from optimized arrival')
            error = abs(actual.return_soc - req.return_soc)
            max_return_soc_error = max(max_return_soc_error, error)
            if error > 1e-8:
                raise ValueError('true return SOC differs from optimized SOC')
            actual_services.append(ServiceDecision(rid, item.station, item.slot, n))
        step = MPCSolution('replay', 0., {}, solution.paths if n == 0 else {}, actual_services,
                           [[[row[n]] for row in station] for station in solution.power], [])
        result = execute_step(params, state, step, scenario.arrivals_between(now, end), scenario)
        state = result.state
        ledger.extend(state.ledger)
        state.ledger = []
        for i, station in enumerate(state.slot_soc):
            for b, soc in enumerate(station):
                max_soc_error = max(max_soc_error, abs(soc - solution.soc[i][b][n + 1]))
        if max_soc_error > 1e-6:
            raise ValueError(f'SOC replay discrepancy: {max_soc_error}')
        state_checks.append(dict(period=n, slot_soc=state.slot_soc,
                                 waiting_count=len(state.waiting),
                                 active_users=sum(u.status == 'active' for u in state.users.values())))
    if state.waiting or any(u.status == 'active' for u in state.users.values()):
        raise ValueError('unsettled demand at full-day horizon')
    if len({event['event_id'] for event in ledger}) != len(ledger):
        raise ValueError('duplicate global ledger event')
    terminals = [e['request_id'] for e in ledger if e['type'] in
                 {'reservation_service', 'random_service', 'reservation_failure', 'random_timeout'}]
    if len(terminals) != len(set(terminals)):
        raise ValueError('request settled more than once')
    financial = summarize_ledger(ledger)
    outcomes = Counter(u.status for u in state.users.values())
    if outcomes['completed'] + outcomes['failed'] != len(scenario.reservations):
        raise ValueError('reservation conservation failed')
    if financial['random_services'] + financial['random_timeouts'] != len(scenario.actual_random_requests):
        raise ValueError('random request conservation failed')
    if outcomes['failed'] != financial['reservation_failures'] or financial['path_adjustments']:
        raise ValueError('failure or fixed-route accounting mismatch')
    tolerance = max(1e-4, 1e-8 * abs(solution.objective))
    error = financial['total_reward'] - solution.objective
    if abs(error) > tolerance:
        raise ValueError(f'objective replay discrepancy: {error}, tolerance={tolerance}')
    for name, actual in [('income', financial['income']), ('charging_cost', financial['charging_cost']),
                         ('failure_cost', financial['reservation_failure_cost']),
                         ('adjustment_cost', financial['adjustment_cost'])]:
        if abs(actual - solution.objective_terms[name]) > tolerance:
            raise ValueError(f'objective component does not reconcile: {name}')
    waits = [e['waiting_hours'] * 60 for e in ledger
             if e['type'] in {'reservation_service', 'random_service'}]
    metrics = dict(financial, reservations_completed=outcomes['completed'],
                   reservation_failure_rate=outcomes['failed'] / len(scenario.reservations)
                   if scenario.reservations else None,
                   random_service_rate=financial['random_services'] / len(scenario.actual_random_requests)
                   if scenario.actual_random_requests else None,
                   mean_wait_minutes=sum(waits) / len(waits) if waits else None)
    return dict(status='passed', metrics=metrics, objective_residual=error,
                objective_tolerance=tolerance, max_soc_residual=max_soc_error,
                max_arrival_residual_hours=max_arrival_error, max_return_soc_residual=max_return_soc_error,
                ledger=ledger, state_checks=state_checks, final_state=state.to_dict())
