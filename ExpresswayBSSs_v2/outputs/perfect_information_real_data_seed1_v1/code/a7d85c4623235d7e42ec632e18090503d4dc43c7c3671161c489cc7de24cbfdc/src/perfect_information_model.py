"""Independent full-day MILP; service/SOC/FCFS formulation follows mpc_model.

There is one protected route per user, no forecast, no repeated replanning
penalty and no terminal reward. This module never imports or changes solve_mpc.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import isfinite
from time import perf_counter

from .domain import MPCSolution, ServiceDecision, user_key_text
from .parameters import price_at, slots_at
from .perfect_information import grid_time


def request_timing(params, window):
    requests = {r.request_id: r for r in window.requests}
    if len(requests) != len(window.requests):
        raise ValueError('duplicate candidate request ID')
    ordered, possible, cases, visiting = [], {}, {}, set()
    delta, end = params.interval_hours, window.horizon * params.interval_hours

    def deadline(req, arrival):
        if not req.predecessors and req.deadline is not None:
            return req.deadline
        due = arrival + params.max_wait_hours
        return grid_time(due, delta) if req.predecessors else due

    def visit(rid):
        if rid in possible:
            return
        if rid in visiting:
            raise ValueError('cyclic service dependency')
        visiting.add(rid)
        req = requests[rid]
        for q in req.predecessors:
            visit(q)
        values = [(grid_time(n * delta + req.travel_time, delta), q, n)
                  for q in req.predecessors for n in possible[q]] if req.predecessors else [
                      (req.arrival_time, None, None)]
        if any(deadline(req, t) >= end for t, _, _ in values):
            raise ValueError('full-day horizon cannot settle every possible request')
        cases[rid] = values
        possible[rid] = [n for n in range(window.horizon)
                         if any(t <= n * delta <= deadline(req, t) for t, _, _ in values)]
        ordered.append(rid)
        visiting.remove(rid)

    for rid in requests:
        visit(rid)
    return requests, ordered, possible, cases, deadline


def valid_upper_bound(value, status):
    """COPT's finite infinity sentinel and invalid statuses are not bounds."""
    return (value is not None and isfinite(value) and abs(value) < 1e29
            and status in {'optimal', 'time_limit', 'interrupted', 'node_limit', 'iteration_limit'})


def solve_perfect(params, window, plans, *, log_path=None, build_only=False):
    import coptpy as cp
    from coptpy import COPT

    started = perf_counter()
    if window.ell != 0 or window.state.period != 0:
        raise ValueError('the perfect-information model starts at time zero')
    requests, ordered, possible, cases, deadline = request_timing(params, window)
    h, delta = window.horizon, params.interval_hours
    counts = [slots_at(params, i) for i in range(params.station.num_stations)]
    config = cp.EnvrConfig()
    config.set('nobanner', '1')
    environment = cp.Envr(config)
    model = None
    try:
        model = environment.createModel('perfect_information_full_day')
        solver = params.solver
        for name, value in [('Logging', solver.output_flag), ('Threads', solver.threads),
                            ('TimeLimit', solver.time_limit_sec), ('RelGap', solver.mip_gap),
                            ('AbsGap', 0.), ('FeasTol', solver.feasibility_tol),
                            ('IntTol', min(1e-8, solver.feasibility_tol))]:
            model.setParam(getattr(COPT.Param, name), value)
        if log_path is not None:
            model.setLogFile(str(log_path))
        variables, start_values = [], []

        def var(*, seed=0., **kwargs):
            value = model.addVar(**kwargs)
            if value.index != len(start_values):
                raise ValueError('unexpected solver variable indexing')
            variables.append(value)
            start_values.append(float(seed))
            return value

        y = {}
        for key, network in window.networks.items():
            arcs = [tuple(a) for a in network.arcs]
            seed_arcs = set(zip(['entry', *plans[key]], [*plans[key], 'exit']))
            if not seed_arcs <= set(arcs):
                raise ValueError('initial route outside the protected candidate network')
            for arc in arcs:
                y[key, arc] = var(vtype=COPT.BINARY, name=f'y[{key},{arc}]', seed=arc in seed_arcs)
            for node in {v for a in arcs for v in a}:
                outgoing = cp.quicksum(y[key, a] for a in arcs if a[0] == node)
                incoming = cp.quicksum(y[key, a] for a in arcs if a[1] == node)
                rhs = 1 if node == network.origin else -1 if node == 'exit' else 0
                model.addConstr(outgoing - incoming == rhs)
                if isinstance(node, int):
                    model.addConstr(incoming <= 1)

        activation, alpha, service, total, arrivals, queue, failure = {}, {}, {}, {}, {}, {}, {}
        by_station = defaultdict(list)
        for rid in ordered:
            req = requests[rid]
            by_station[req.station].append(rid)
            active = 1 if req.arc is None else y[user_key_text(req.user_key), tuple(req.arc)]
            activation[rid] = active
            seed_active = 1 if isinstance(active, int) else start_values[active.index]
            for n in possible[rid]:
                for b in range(counts[req.station]):
                    alpha[rid, b, n] = var(vtype=COPT.BINARY, name=f'a[{rid},{b},{n}]')
                service[rid, n] = cp.quicksum(alpha[rid, b, n] for b in range(counts[req.station]))
            total[rid] = cp.quicksum(service[rid, n] for n in possible[rid])
            model.addConstr(total[rid] <= active)
            arrivals[rid] = [(t, 1 if q is None else service[q, m]) for t, q, m in cases[rid]]
            if req.predecessors:
                model.addConstr(cp.quicksum(v for _, v in arrivals[rid]) <= 1)
            for n in possible[rid]:
                eligible = cp.quicksum(v for t, v in arrivals[rid] if t <= n * delta <= deadline(req, t))
                previous = cp.quicksum(service[rid, m] for m in possible[rid] if m < n)
                seed_queue = seed_active if not req.predecessors else 0
                qvar = var(vtype=COPT.BINARY, name=f'Q[{rid},{n}]', seed=seed_queue)
                queue[rid, n] = qvar
                model.addConstr(qvar <= active)
                model.addConstr(qvar <= eligible)
                model.addConstr(qvar <= 1 - previous)
                model.addConstr(qvar >= active + eligible - previous - 1)
                model.addConstr(service[rid, n] <= qvar)
            if req.kind == 'reservation':
                arrived = cp.quicksum(v for _, v in arrivals[rid])
                fvar = var(vtype=COPT.BINARY, name=f'failed[{rid}]',
                           seed=seed_active if not req.predecessors else 0)
                failure[rid] = fvar
                model.addConstr(fvar <= active)
                model.addConstr(fvar <= arrived)
                model.addConstr(fvar <= 1 - total[rid])
                model.addConstr(fvar >= active + arrived - total[rid] - 1)
        by_user = defaultdict(list)
        for rid, value in failure.items():
            by_user[user_key_text(requests[rid].user_key)].append(value)
        for values in by_user.values():
            model.addConstr(cp.quicksum(values) <= 1)

        def priority(q, r, shared, precedes=1):
            for n in shared:
                model.addConstr(service[r, n] <= 2 - queue[q, n] + service[q, n] - precedes)

        period_sets = {rid: set(values) for rid, values in possible.items()}
        for ids in by_station.values():
            for q, r in combinations(ids, 2):
                qr, rr = requests[q], requests[r]
                if qr.user_key is not None and qr.user_key == rr.user_key:
                    continue
                shared = sorted(period_sets[q] & period_sets[r])
                if not shared:
                    continue
                if qr.kind != rr.kind:
                    priority(q, r, shared) if qr.kind == 'reservation' else priority(r, q, shared)
                    continue
                qc, rc = arrivals[q], arrivals[r]
                if len(qc) == 1 and len(rc) == 1:
                    if qc[0][0] < rc[0][0]:
                        priority(q, r, shared)
                    elif rc[0][0] < qc[0][0]:
                        priority(r, q, shared)
                    continue
                if max(t for t, _ in qc) < min(t for t, _ in rc):
                    priority(q, r, shared)
                    continue
                if max(t for t, _ in rc) < min(t for t, _ in qc):
                    priority(r, q, shared)
                    continue
                ranks = {t: rank for rank, t in enumerate(sorted({t for t, _ in qc + rc}), 1)}
                qrank = cp.quicksum(ranks[t] * value for t, value in qc)
                rrank = cp.quicksum(ranks[t] * value for t, value in rc)
                qseed = ranks[qc[0][0]] if not qr.predecessors else 0
                rseed = ranks[rc[0][0]] if not rr.predecessors else 0
                bound = len(ranks) + 1
                for first, second, diff, seed in [(q, r, qrank - rrank, qseed < rseed),
                                                   (r, q, rrank - qrank, rseed < qseed)]:
                    earlier = var(vtype=COPT.BINARY, name=f'earlier[{first},{second}]', seed=seed)
                    model.addConstr(diff <= bound * (1 - earlier) - 1)
                    model.addConstr(diff >= -bound * earlier)
                    priority(first, second, shared, earlier)

        power, soc = {}, {}
        charging = cp.LinExpr()
        for i, count in enumerate(counts):
            for b in range(count):
                initial_soc = window.state.slot_soc[i][b]
                for n in range(h + 1):
                    soc[i, b, n] = var(lb=0, ub=1, name=f'S[{i},{b},{n}]', seed=initial_soc)
                model.addConstr(soc[i, b, 0] == initial_soc)
                for n in range(h):
                    power[i, b, n] = var(lb=0, ub=params.slot_power_limit(i, b), name=f'P[{i},{b},{n}]')
                    swaps = [(rid, alpha[rid, b, n]) for rid in by_station[i] if (rid, b, n) in alpha]
                    # Since SOC <= 1, this combines at-most-one and full-delivery.
                    model.addConstr(cp.quicksum(a for _, a in swaps) <= soc[i, b, n])
                    reduction = cp.quicksum((1 - requests[rid].return_soc) * a for rid, a in swaps)
                    model.addConstr(soc[i, b, n + 1] == soc[i, b, n] - reduction +
                                    params.station.charging_efficiency * delta /
                                    params.battery_capacity_kwh * power[i, b, n])
                    charging += delta * price_at(params, 'electricity_price', i, n) * power[i, b, n]
            for n in range(h):
                model.addConstr(cp.quicksum(power[i, b, n] for b in range(count)) <= params.station_power_limit(i))
        income = cp.quicksum(params.battery_capacity_kwh * (1 - requests[rid].return_soc) *
                             price_at(params, 'swap_service_price', requests[rid].station, n) * a
                             for (rid, _, n), a in alpha.items())
        failed_cost = params.reservation_failure_penalty * cp.quicksum(failure.values())
        model.setObjective(income - charging - failed_cost, COPT.MAXIMIZE)

        # Check the complete deterministic MIP start against every generated row.
        seed_residual = 0.
        for constraint in model.getConstrs():
            row = model.getRow(constraint)
            lhs = row.getConstant() + sum(row.getCoeff(k) * start_values[row.getVar(k).index]
                                          for k in range(row.getSize()))
            lo, hi = constraint.getInfo(COPT.Info.LB), constraint.getInfo(COPT.Info.UB)
            seed_residual = max(seed_residual, lo - lhs, lhs - hi)
        if seed_residual > solver.feasibility_tol:
            raise ValueError(f'initial MIP solution violates constraints by {seed_residual}')
        if model.ismip:
            model.setMipStart(variables, start_values)
            model.loadMipStart()
        diagnostics = dict(model_variables=int(model.getAttr(COPT.Attr.Cols)),
                           model_constraints=int(model.getAttr(COPT.Attr.Rows)),
                           model_binary_variables=int(model.getAttr(COPT.Attr.Bins)),
                           initial_max_constraint_residual=seed_residual)
        build_seconds = perf_counter() - started
        if build_only:
            return dict(status='built', diagnostics=diagnostics, build_seconds=build_seconds)
        model.solve()
        names = {COPT.OPTIMAL: 'optimal', COPT.TIMEOUT: 'time_limit', COPT.INTERRUPTED: 'interrupted',
                 COPT.INFEASIBLE: 'infeasible', COPT.INF_OR_UNB: 'infeasible_or_unbounded',
                 COPT.UNBOUNDED: 'unbounded', COPT.NUMERICAL: 'numerical', COPT.IMPRECISE: 'suboptimal',
                 COPT.NODELIMIT: 'node_limit', COPT.ITERLIMIT: 'iteration_limit'}
        status = names.get(model.status, f'copt_status_{model.status}')
        has_solution = bool(model.hasmipsol if model.ismip else model.haslpsol)
        bound = float(model.bestbnd) if model.ismip else (float(model.objval) if has_solution else None)
        upper = bound if valid_upper_bound(bound, status) else None
        result = dict(status=status, has_incumbent=has_solution, best_bound=upper,
                      raw_best_bound=bound if bound is not None and isfinite(bound) else None,
                      incumbent_objective=float(model.objval) if has_solution else None,
                      relative_gap=float(model.bestgap) if has_solution and model.ismip else
                      (0. if has_solution else None), solve_seconds=float(model.solvingtime),
                      build_seconds=build_seconds, diagnostics=diagnostics, solution=None)
        if has_solution:
            def value(expr):
                if isinstance(expr, (int, float)):
                    return float(expr)
                return expr.x if isinstance(expr, cp.Var) else expr.getValue()
            paths = {}
            for key, network in window.networks.items():
                chosen = {a[0]: a[1] for a in map(tuple, network.arcs) if y[key, a].x > .5}
                node, route, seen = network.origin, [], set()
                while node != 'exit':
                    if node in seen or node not in chosen:
                        raise ValueError('invalid full-day route')
                    seen.add(node)
                    node = chosen[node]
                    if isinstance(node, int):
                        route.append(node)
                paths[key] = route
            selected_services = sorted([ServiceDecision(rid, requests[rid].station, b, n)
                                        for (rid, b, n), a in alpha.items() if a.x > .5],
                                       key=lambda a: (a.period, a.station, a.slot, a.request_id))
            served = {a.request_id: a.period for a in selected_services}
            outcomes = {}
            for rid, req in requests.items():
                selected = value(activation[rid]) > .5
                chosen = [t for t, expr in arrivals[rid] if value(expr) > .5] if selected else []
                arrival = chosen[0] if chosen else None
                outcomes[rid] = dict(selected=selected, arrival_time=arrival,
                                     deadline=deadline(req, arrival) if arrival is not None else None,
                                     served=rid in served, service_period=served.get(rid),
                                     failed=rid in failure and failure[rid].x > .5)
            solution = MPCSolution(status, float(model.objval),
                dict(income=value(income), charging_cost=value(charging), adjustment_cost=0.,
                     failure_cost=value(failed_cost)), paths, selected_services,
                [[[power[i, b, n].x for n in range(h)] for b in range(count)] for i, count in enumerate(counts)],
                [[[soc[i, b, n].x for n in range(h + 1)] for b in range(count)] for i, count in enumerate(counts)],
                outcomes, result['relative_gap'], result['solve_seconds'],
                build_seconds=build_seconds, wall_seconds=perf_counter() - started, diagnostics=diagnostics)
            result['solution'] = solution.to_dict()
            if upper is not None and upper + max(1e-4, abs(solution.objective) * 1e-8) < solution.objective:
                raise ValueError('maximization bound below the incumbent')
        result['wall_seconds'] = perf_counter() - started
        return result
    finally:
        del model
        environment.close()
