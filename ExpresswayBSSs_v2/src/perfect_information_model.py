"""Independent full-day MILP; service/SOC/FCFS formulation follows mpc_model.

There is one protected route per user, no forecast, no repeated replanning
penalty and no terminal reward. This module never imports or changes solve_mpc.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from math import isfinite
import json
from pathlib import Path
from time import perf_counter

from .domain import MPCSolution, ServiceDecision, user_key_text
from .parameters import price_at, slots_at
from .perfect_information import grid_time
from .perfect_information_checks import model_fingerprints
from .perfect_information_strengthening import (add_strengthening, add_total_service_order,
                                               canonicalize_slots, cooldown_sets)


def node_order(node):
    return (0, 0) if node == 'entry' else (2, 0) if node == 'exit' else (1, int(node))


def request_timing(params, window):
    requests = {r.request_id: r for r in sorted(window.requests, key=lambda r: r.request_id)}
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
        for q in sorted(req.predecessors):
            visit(q)
        values = [(grid_time(n * delta + req.travel_time, delta), q, n)
                  for q in sorted(req.predecessors) for n in possible[q]] if req.predecessors else [
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


def solve_perfect(params, window, plans, *, log_path=None, build_only=False,
                  warm_start=None, audit_path=None, expected_fingerprints=None,
                  formulation='baseline', root_cut_rounds=-1, relaxation_only=False,
                  restriction='none', slot_order='none'):
    import coptpy as cp
    from coptpy import COPT

    started = perf_counter()
    if formulation not in {'baseline', 'strengthened', 'strengthened_compact'}:
        raise ValueError('unknown perfect-information formulation')
    if slot_order not in {'none', 'total_services'}:
        raise ValueError('unknown slot ordering')
    if formulation == 'strengthened' and slot_order != 'none':
        raise ValueError('first-use and total-service orderings cannot be combined')
    if restriction not in {'none', 'routes', 'services'}:
        raise ValueError('unknown diagnostic restriction')
    if restriction != 'none' and warm_start is None:
        raise ValueError('diagnostic restrictions require a warm start')
    if restriction != 'none' and set(warm_start.paths) != set(window.networks):
        raise ValueError('fixed route user set differs from the model')
    bound_scope = 'original_full_problem' if restriction == 'none' else 'restricted_subproblem_only'
    strengthened = formulation != 'baseline'
    permutation = []
    if formulation == 'strengthened' and warm_start is not None:
        warm_start, permutation = canonicalize_slots(params, window, warm_start)
    if slot_order == 'total_services' and warm_start is not None:
        warm_start, permutation = canonicalize_slots(params, window, warm_start, order=slot_order)
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
        if root_cut_rounds >= 0:
            model.setParam(COPT.Param.RootCutRounds, root_cut_rounds)
        variables, start_values = [], []

        def var(*, seed=0., **kwargs):
            value = model.addVar(**kwargs)
            if value.index != len(start_values):
                raise ValueError('unexpected solver variable indexing')
            variables.append(value)
            start_values.append(float(seed))
            return value

        y = {}
        for key, network in sorted(window.networks.items()):
            arcs = sorted(map(tuple, network.arcs), key=lambda a: (node_order(a[0]), node_order(a[1])))
            seed_arcs = set(zip(['entry', *plans[key]], [*plans[key], 'exit']))
            if not seed_arcs <= set(arcs):
                raise ValueError('initial route outside the protected candidate network')
            fixed_arcs = (set(zip(['entry', *warm_start.paths[key]], [*warm_start.paths[key], 'exit']))
                          if restriction != 'none' else None)
            for arc in arcs:
                bounds = dict(lb=float(arc in fixed_arcs), ub=float(arc in fixed_arcs)) if fixed_arcs is not None else {}
                y[key, arc] = var(vtype=COPT.BINARY, name=f'y[{key},{arc}]', seed=arc in seed_arcs, **bounds)
            for node in sorted({v for a in arcs for v in a}, key=node_order):
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
                qvar = var(vtype=COPT.CONTINUOUS if strengthened else COPT.BINARY, lb=0., ub=1.,
                           name=f'Q[{rid},{n}]', seed=seed_queue)
                queue[rid, n] = qvar
                model.addConstr(qvar <= active)
                model.addConstr(qvar <= eligible)
                model.addConstr(qvar <= 1 - previous)
                model.addConstr(qvar >= active + eligible - previous - 1)
                model.addConstr(service[rid, n] <= qvar)
            if req.kind == 'reservation':
                arrived = cp.quicksum(v for _, v in arrivals[rid])
                fvar = var(vtype=COPT.CONTINUOUS if strengthened else COPT.BINARY, lb=0., ub=1., name=f'failed[{rid}]',
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
        earlier_variables = []
        for station in sorted(by_station):
            ids = sorted(by_station[station])
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
                    earlier_variables.append((earlier, diff))
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
        prefix, strengthening = {}, {}
        if strengthened:
            prefix, strengthening = add_strengthening(cp, model, var, params, window, requests,
                alpha, service, total, activation, arrivals, failure, first_use=formulation == 'strengthened')
        order_rows = (add_total_service_order(cp, model, params, window, requests, alpha)
                      if slot_order == 'total_services' else 0)
        if restriction == 'services':
            # Freeze the entire served set and its periods, not the battery slots.
            # All unlisted request-period pairs are explicitly fixed to zero.
            fixed = {(d.request_id, d.period) for d in warm_start.services}
            for key, expression in service.items():
                model.addConstr(expression == float(key in fixed))
        model.setObjective(income - charging - failed_cost, COPT.MAXIMIZE)
        model.update()

        def start_value(expr):
            if isinstance(expr, (int, float)):
                return float(expr)
            if isinstance(expr, cp.Var):
                return start_values[expr.index]
            return expr.getConstant() + sum(expr.getCoeff(k) * start_values[expr.getVar(k).index]
                                            for k in range(expr.getSize()))

        if warm_start is not None:
            if set(warm_start.paths) != set(window.networks):
                raise ValueError('warm start user set differs from the model')
            for key in sorted(window.networks):
                route = warm_start.paths[key]
                arcs = set(zip(['entry', *route], [*route, 'exit']))
                if not arcs <= {a for k, a in y if k == key}:
                    raise ValueError(f'warm start route is outside the candidate network: {key}')
                for (k, a), variable in y.items():
                    if k == key:
                        start_values[variable.index] = float(a in arcs)
            for variable in alpha.values():
                start_values[variable.index] = 0.
            seen = set()
            for decision in warm_start.services:
                rid = decision.request_id
                key = (rid, decision.slot, decision.period)
                if (rid in seen or key not in alpha or requests[rid].station != decision.station):
                    raise ValueError(f'invalid or duplicate warm start service: {rid}')
                seen.add(rid)
                start_values[alpha[key].index] = 1.
            for values, target, size in [(warm_start.power, power, h), (warm_start.soc, soc, h + 1)]:
                if (len(values) != len(counts) or any(len(values[i]) != count for i, count in enumerate(counts))
                        or any(len(row) != size for station in values for row in station)):
                    raise ValueError('warm start battery/time dimensions differ from the model')
                for (i, b, n), variable in target.items():
                    start_values[variable.index] = float(values[i][b][n])
            for rid in ordered:
                active = start_value(activation[rid]) > .5
                for n in possible[rid]:
                    eligible = sum(start_value(v) for t, v in arrivals[rid]
                                   if t <= n * delta <= deadline(requests[rid], t))
                    previous = sum(start_value(service[rid, m]) for m in possible[rid] if m < n)
                    start_values[queue[rid, n].index] = float(active and eligible > .5 and previous < .5)
                if rid in failure:
                    arrived = sum(start_value(v) for _, v in arrivals[rid])
                    start_values[failure[rid].index] = float(active and arrived > .5 and start_value(total[rid]) < .5)
            for variable, diff in earlier_variables:
                start_values[variable.index] = float(start_value(diff) < -.5)
            count_by_slot_period = defaultdict(int)
            for decision in warm_start.services:
                count_by_slot_period[decision.station, decision.slot, decision.period] += 1
            for (i, b, n), variable in prefix.items():
                start_values[variable.index] = sum(count_by_slot_period[i, b, m] for m in range(n + 1))

        # Check the complete deterministic MIP start against every generated row.
        seed_residual = 0.
        for variable, value in zip(variables, start_values):
            if not isfinite(value):
                raise ValueError('nonfinite warm start value')
            seed_residual = max(seed_residual, variable.lb - value, value - variable.ub,
                                abs(value - round(value)) if variable.vtype != COPT.CONTINUOUS else 0.)
        for constraint in model.getConstrs():
            row = model.getRow(constraint)
            lhs = row.getConstant() + sum(row.getCoeff(k) * start_values[row.getVar(k).index]
                                          for k in range(row.getSize()))
            lo, hi = constraint.getInfo(COPT.Info.LB), constraint.getInfo(COPT.Info.UB)
            seed_residual = max(seed_residual, lo - lhs, lhs - hi)
        if seed_residual > solver.feasibility_tol:
            raise ValueError(f'initial MIP solution violates constraints by {seed_residual}')
        initial_objective = start_value(income - charging - failed_cost)
        if warm_start is not None and abs(initial_objective - warm_start.objective) > max(1e-4, abs(warm_start.objective) * 1e-8):
            raise ValueError('warm start objective does not match the rebuilt model')
        if model.ismip:
            # Explicit mode ensures a supplied full start is processed, not left
            # to the solver's automatic choice of initial-solution handling.
            model.setParam(COPT.Param.MipStartMode, 1)
            model.setMipStart(variables, start_values)
            model.loadMipStart()
        fingerprints = model_fingerprints(model)
        if expected_fingerprints is not None and fingerprints != expected_fingerprints:
            raise ValueError('model differs from the independently verified build')
        diagnostics = dict(model_variables=int(model.getAttr(COPT.Attr.Cols)),
                           model_constraints=int(model.getAttr(COPT.Attr.Rows)),
                           model_binary_variables=int(model.getAttr(COPT.Attr.Bins)),
                           initial_max_constraint_residual=seed_residual,
                           initial_objective=initial_objective, mip_start_mode=1,
                           warm_start_used=warm_start is not None, fingerprints=fingerprints,
                           formulation=formulation, root_cut_rounds=root_cut_rounds,
                           strengthening=strengthening, warm_start_slot_permutation=permutation,
                           restriction=restriction, bound_scope=bound_scope,
                           slot_order=slot_order, total_service_order_rows=order_rows,
                           fixed_route_variables=len(y) if restriction != 'none' else 0,
                           fixed_service_periods=len(service) if restriction == 'services' else 0,
                           service_assignment_variables=len(alpha))
        if audit_path is not None:
            Path(audit_path).write_text(json.dumps(diagnostics, indent=2, allow_nan=False), encoding='utf-8')
        build_seconds = perf_counter() - started
        if build_only:
            return dict(status='built', diagnostics=diagnostics, build_seconds=build_seconds)
        if relaxation_only:
            model.solveLP()
            solved = model.lpstatus == COPT.OPTIMAL
            data = dict(status='lp_optimal' if solved else f'lp_status_{model.lpstatus}',
                        lp_bound=float(model.lpobjval) if solved else None,
                        solve_seconds=float(model.solvingtime), build_seconds=build_seconds,
                        diagnostics=diagnostics)
            if model.haslpsol:
                values, _, _, _ = model.getLpSolution()
                fractional = defaultdict(int)
                for v, x in zip(variables, values):
                    if v.vtype != COPT.CONTINUOUS and abs(x - round(x)) > 1e-7:
                        fractional[v.name.split('[')[0]] += 1
                violations = [sum(values[alpha[key].index] for key in keys) - upper
                              for _, _, _, keys, upper in cooldown_sets(params, window, requests, alpha)]
                data.update(fractional_integer_variables=dict(fractional),
                            violated_cooldown_cliques=sum(v > 1e-6 for v in violations),
                            max_cooldown_violation=max([0., *violations]))
            return data
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
                      bound_scope=bound_scope,
                      raw_best_bound=bound if bound is not None and isfinite(bound) else None,
                      incumbent_objective=float(model.objval) if has_solution else None,
                      relative_gap=float(model.bestgap) if has_solution and model.ismip else
                      (0. if has_solution else None), solve_seconds=float(model.solvingtime),
                      build_seconds=build_seconds, diagnostics=diagnostics, solution=None)
        if warm_start is not None and (not has_solution or result['incumbent_objective'] < initial_objective - max(1e-4, abs(initial_objective) * 1e-8)):
            raise ValueError('solver failed to preserve the validated warm start incumbent')
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
