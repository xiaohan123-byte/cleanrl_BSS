"""Joint path, swap-assignment and charging MILP with optional terminal value.

The formulation follows section 3 of the revised paper. Arrival eligibility is
an exact sum of predecessor service decisions. FCFS comparisons use integer
ranks of the finite possible arrival times, so they require no arbitrary
big-M bound in physical time and no positive time epsilon.
"""

from __future__ import annotations
from .parameters import slots_at, price_at, execution_period_limit

import math
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from typing import Any

from .domain import MPCSolution, MPCWindow, ServiceDecision, user_key_text


class MPCSolveError(RuntimeError):
    """The optimizer returned no feasible incumbent; execution must stop."""


def _computed_grid_time(value: float, delta: float) -> float:
    """Canonicalise grid arithmetic only; never call on external timestamps."""
    boundary = round(value / delta) * delta
    return boundary if abs(value - boundary) <= 4 * max(math.ulp(value), math.ulp(boundary)) else value


def solve_mpc(params: Any, window: MPCWindow, *, terminal_model=None, feature_spec=None,
              route_mode="joint", charging_mode="joint", diagnostic_dir=None) -> MPCSolution:
    """Solve one prediction window; only its first stage may be executed.

    ``power`` and ``soc`` use offsets into this window; services use absolute
    period indices. A time-limited feasible incumbent is returned with its gap.
    There is deliberately no heuristic fallback when no incumbent exists.
    """
    started_at = time.perf_counter()
    if route_mode not in {"joint", "dayahead", "fixed"}:
        raise ValueError("route_mode must be joint or dayahead/fixed")
    if charging_mode not in {"joint", "baseline"}:
        raise ValueError("charging_mode must be joint or baseline")
    if terminal_model is not None and terminal_model.configuration_fingerprint is not None:
        from .terminal_value import terminal_configuration_fingerprint
        if terminal_model.configuration_fingerprint != terminal_configuration_fingerprint(params):
            raise ValueError("terminal model was trained for a different physical/demand configuration")
    try:
        import coptpy as cp
        from coptpy import COPT
    except ImportError as exc:
        raise MPCSolveError("COPT (coptpy) is required to solve the baseline MILP") from exc

    ell, horizon = window.ell, window.horizon
    if horizon < 1 or ell < 0 or ell >= execution_period_limit(params) or (not getattr(params, "finish_pending_after_demand", False) and ell + horizon > params.num_periods):
        raise ValueError("MPC window must lie within the configured operating periods")
    if window.state.period != ell:
        raise ValueError("MPC initial state and window period differ")
    delta = params.interval_hours
    periods = range(ell, ell + horizon)
    end = (ell + horizon) * delta
    nstations = params.station.num_stations
    slot_counts = [slots_at(params, i) for i in range(nstations)]
    requests = {r.request_id: r for r in window.requests}
    if len(requests) != len(window.requests):
        raise ValueError("Candidate request IDs must be unique")
    for req in requests.values():
        if req.kind not in ("reservation", "random"):
            raise ValueError(f"Unknown request kind: {req.kind}")
        if not 0 <= req.station < nstations or not 0 <= req.return_soc < 1:
            raise ValueError(f"Invalid station or return SOC for {req.request_id}")
        if req.predecessors and req.travel_time <= 0:
            raise ValueError("Interstation travel times must be strictly positive")
        if not req.predecessors and req.arrival_time is None:
            raise ValueError(f"Request {req.request_id} needs a known arrival time")
        if any(q not in requests for q in req.predecessors):
            raise ValueError(f"Missing predecessor of {req.request_id}")
        if req.observed and req.request_id not in window.state.waiting:
            raise ValueError("An observed request must exist in the real waiting state")

    def deadline(req: Any, arrival: float) -> float:
        if not req.predecessors and req.deadline is not None:
            return req.deadline  # supplied deadlines retain their exact value
        due = arrival + params.max_wait_hours
        if req.predecessors:
            return _computed_grid_time(due, delta)
        return due

    # Acyclic dependency order and possible arrival times prune alpha before
    # building the model. This is feasibility pruning, not a service policy.
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    possible_periods: dict[str, list[int]] = {}
    possible_cases: dict[str, list[tuple[float, str | None, int | None]]] = {}

    def visit(rid: str) -> None:
        if rid in visiting:
            raise ValueError("Request predecessor graph contains a cycle")
        if rid in visited:
            return
        visiting.add(rid)
        req = requests[rid]
        for predecessor in req.predecessors:
            visit(predecessor)
        if req.predecessors:
            cases = [(_computed_grid_time(m * delta + req.travel_time, delta), q, m)
                     for q in req.predecessors for m in possible_periods[q]]
        else:
            cases = [(req.arrival_time, None, None)]
        possible_cases[rid] = cases
        possible_periods[rid] = [n for n in periods
                                if (n != ell or req.observed)
                                and any(t <= n * delta <= deadline(req, t)
                                        for t, _, _ in cases)]
        visiting.remove(rid)
        visited.add(rid)
        ordered.append(rid)

    for rid in requests:
        visit(rid)

    environment = None
    model = None
    initializing = True
    try:
        config = cp.EnvrConfig()
        if not params.solver.output_flag:
            config.set("nobanner", "1")
        environment = cp.Envr(config)
        model = environment.createModel("bss_discrete_terminal" if terminal_model is not None else "bss_discrete_zero_terminal")
        initializing = False
        solver = params.solver
        model.setParam(COPT.Param.Logging, solver.output_flag)
        model.setParam(COPT.Param.Threads, solver.threads)
        model.setParam(COPT.Param.TimeLimit, solver.time_limit_sec)
        model.setParam(COPT.Param.RelGap, solver.mip_gap)
        # Do not let COPT's default absolute gap bypass a requested zero gap.
        model.setParam(COPT.Param.AbsGap, 0.0)
        model.setParam(COPT.Param.FeasTol, solver.feasibility_tol)
        model.setParam(COPT.Param.IntTol, min(1e-8, solver.feasibility_tol))

        y: dict[tuple[str, tuple], Any] = {}
        changed: dict[str, Any] = {}
        for key, network in window.networks.items():
            arcs = [tuple(arc) for arc in network.arcs]
            if not arcs or len(set(arcs)) != len(arcs):
                raise ValueError(f"User {key} has no path network or duplicate arcs")
            nodes = {node for arc in arcs for node in arc}
            if network.origin not in nodes or "exit" not in nodes:
                raise ValueError(f"User {key} network requires its origin and exit")
            for arc in arcs:
                y[key, arc] = model.addVar(vtype=COPT.BINARY, name=f"y[{key},{arc}]")
            for node in nodes:
                outflow = cp.quicksum(y[key, arc] for arc in arcs if arc[0] == node)
                inflow = cp.quicksum(y[key, arc] for arc in arcs if arc[1] == node)
                rhs = 1 if node == network.origin else -1 if node == "exit" else 0
                model.addConstr(outflow - inflow == rhs, name=f"flow[{key},{node}]")
            differences = []
            reference = set(network.reference_stations)
            frozen = set(network.frozen_stations)
            if route_mode in {"dayahead", "fixed"}:
                user = window.state.users.get(key)
                if user is None:
                    raise ValueError("fixed dayahead paths need user state")
                od = params.od_pairs[params.od_index(user.user_key[0])]
                direction = 1 if od.exit_km > od.entry_km else -1
                excluded = set(user.completed_stations)
                if user.waiting_request_id is not None:
                    excluded.add(window.state.waiting[user.waiting_request_id].station)
                frozen = {i for i in user.day_ahead if i not in excluded
                          and direction * (params.station.positions_km[i] - user.position_km) >= -1e-9}
            for station in range(nstations):
                x = cp.quicksum(y[key, arc] for arc in arcs if arc[1] == station)
                model.addConstr(x <= 1)
                if not network.can_update or route_mode in {"dayahead", "fixed"}:
                    model.addConstr(x == int(station in frozen), name=f"freeze[{key},{station}]")
                differences.append(1 - x if station in reference else x)
            chi = model.addVar(vtype=COPT.BINARY, name=f"changed[{key}]")
            changed[key] = chi
            for difference in differences:
                model.addConstr(chi >= difference)
            model.addConstr(chi <= cp.quicksum(differences))

        activation: dict[str, Any] = {}
        alpha: dict[tuple[str, int, int], Any] = {}
        service: dict[tuple[str, int], Any] = {}
        total: dict[str, Any] = {}
        arrival_cases: dict[str, list[tuple[float, Any]]] = {}
        queue: dict[tuple[str, int], Any] = {}
        failures: dict[str, Any] = {}
        by_station: dict[int, list[str]] = defaultdict(list)

        for rid in ordered:
            req = requests[rid]
            by_station[req.station].append(rid)
            if req.arc is None:
                activation[rid] = 1
            else:
                activation[rid] = y[user_key_text(req.user_key), tuple(req.arc)]
            for n in possible_periods[rid]:
                for slot in range(slot_counts[req.station]):
                    alpha[rid, slot, n] = model.addVar(vtype=COPT.BINARY, name=f"a[{rid},{slot},{n}]")
                service[rid, n] = cp.quicksum(alpha[rid, slot, n] for slot in range(slot_counts[requests[rid].station]))
            total[rid] = cp.quicksum(service[rid, n] for n in possible_periods[rid])
            model.addConstr(total[rid] <= activation[rid], name=f"once[{rid}]")
            cases = [(t, 1 if q is None else service[q, m]) for t, q, m in possible_cases[rid]]
            arrival_cases[rid] = cases
            # At most one predecessor is served; this is implied by path flow,
            # but explicit here also validates manually constructed windows.
            if req.predecessors:
                model.addConstr(cp.quicksum(value for _, value in cases) <= 1)
            for n in possible_periods[rid]:
                eligible = cp.quicksum(value for t, value in cases
                                       if t <= n * delta <= deadline(req, t))
                previous = cp.quicksum(service[rid, m] for m in possible_periods[rid] if m < n)
                qvar = model.addVar(vtype=COPT.BINARY, name=f"Q[{rid},{n}]")
                queue[rid, n] = qvar
                model.addConstr(qvar <= activation[rid])
                model.addConstr(qvar <= eligible)
                model.addConstr(qvar <= 1 - previous)
                model.addConstr(qvar >= activation[rid] + eligible - previous - 1)
                model.addConstr(service[rid, n] <= qvar)
            if req.kind == "reservation":
                expired = cp.quicksum(value for t, value in cases if deadline(req, t) < end)
                fail = model.addVar(vtype=COPT.BINARY, name=f"failed[{rid}]")
                failures[rid] = fail
                model.addConstr(fail <= activation[rid])
                model.addConstr(fail <= expired)
                model.addConstr(fail <= 1 - total[rid])
                model.addConstr(fail >= activation[rid] + expired - total[rid] - 1)

        per_user_failures: dict[str, list[Any]] = defaultdict(list)
        for rid, failure in failures.items():
            if requests[rid].user_key is not None:
                per_user_failures[user_key_text(requests[rid].user_key)].append(failure)
        for key, values in per_user_failures.items():
            model.addConstr(cp.quicksum(values) <= 1, name=f"one_failure[{key}]")

        def priority(q: str, r: str, shared: list[int], precedes: Any = 1) -> None:
            for n in shared:
                model.addConstr(service[r, n] <= 2 - queue[q, n] + service[q, n] - precedes,
                                name=f"priority[{q},{r},{n}]")

        for station, ids in by_station.items():
            for q, r in combinations(ids, 2):
                qr, rr = requests[q], requests[r]
                shared = sorted(set(possible_periods[q]) & set(possible_periods[r]))
                if not shared:
                    continue
                if qr.user_key is not None and qr.user_key == rr.user_key:
                    continue  # same user's alternative incoming arcs are exclusive
                if qr.kind != rr.kind:
                    priority(q, r, shared) if qr.kind == "reservation" else priority(r, q, shared)
                    continue
                qc, rc = arrival_cases[q], arrival_cases[r]
                if len(qc) == 1 and len(rc) == 1:
                    # A single variable case need not occur; Q still gates it.
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
                # Integer ranks preserve every strict comparison of the finite
                # possible arrival times. Rank 0 means no predecessor service.
                # Distinct external timestamps stay distinct even one ULP apart.
                ranks = {t: rank for rank, t in enumerate(sorted({t for t, _ in qc + rc}), start=1)}
                rank = len(ranks)
                qrank = cp.quicksum(ranks[t] * value for t, value in qc)
                rrank = cp.quicksum(ranks[t] * value for t, value in rc)
                bound = rank + 1
                for first, second, difference in ((q, r, qrank - rrank), (r, q, rrank - qrank)):
                    earlier = model.addVar(vtype=COPT.BINARY, name=f"earlier[{first},{second}]")
                    model.addConstr(difference <= bound * (1 - earlier) - 1)
                    model.addConstr(difference >= -bound * earlier)
                    priority(first, second, shared, earlier)

        power: dict[tuple[int, int, int], Any] = {}
        soc: dict[tuple[int, int, int], Any] = {}
        eta = params.station.charging_efficiency
        charging_cost = cp.LinExpr()
        for station in range(nstations):
            for slot in range(slot_counts[station]):
                for n in range(ell, ell + horizon + 1):
                    soc[station, slot, n] = model.addVar(lb=0, ub=1, name=f"S[{station},{slot},{n}]")
                model.addConstr(soc[station, slot, ell] == window.state.slot_soc[station][slot])
                for n in periods:
                    pvar = model.addVar(lb=0, ub=params.slot_power_limit(station, slot),
                                        name=f"P[{station},{slot},{n}]")
                    power[station, slot, n] = pvar
                    swaps = [(rid, alpha[rid, slot, n]) for rid in by_station[station]
                             if (rid, slot, n) in alpha]
                    model.addConstr(cp.quicksum(a for _, a in swaps) <= 1)
                    for _, avar in swaps:
                        model.addConstr(soc[station, slot, n] >= avar)
                    reduction = cp.quicksum((1 - requests[rid].return_soc) * a for rid, a in swaps)
                    model.addConstr(soc[station, slot, n + 1] == soc[station, slot, n] - reduction
                                    + eta * delta / params.battery_capacity_kwh * pvar)
                    if charging_mode == "baseline":
                        from .terminal_value import bounded_relu
                        limit = min(params.slot_power_limit(station, slot),
                                    params.station_power_limit(station) / slot_counts[station])
                        full_step = params.battery_capacity_kwh / (eta * delta)
                        required = full_step * (1 - soc[station, slot, n] + reduction)
                        excess = bounded_relu(model, cp, COPT, required - limit,
                                              -limit, full_step - limit,
                                              f"base_power_excess[{station},{slot},{n}]")
                        model.addConstr(pvar == required - excess,
                                        name=f"baseline_power[{station},{slot},{n}]")
                    charging_cost += delta * price_at(params, "electricity_price", station, n) * pvar
            for n in periods:
                model.addConstr(cp.quicksum(power[station, slot, n] for slot in range(slot_counts[station]))
                                <= params.station_power_limit(station))

        income = cp.quicksum(params.battery_capacity_kwh * (1 - requests[rid].return_soc)
                             * price_at(params, "swap_service_price", requests[rid].station, n) * avar
                             for (rid, _, n), avar in alpha.items())
        adjustment_cost = params.path_adjustment_penalty * cp.quicksum(changed.values())
        failure_cost = params.reservation_failure_penalty * cp.quicksum(failures.values())
        terminal_expression = 0.0
        terminal_expressions, terminal_bounds, terminal_records = [], [], {}
        if terminal_model is not None and ell + horizon < params.num_periods:
            from .terminal_features import FeatureSpec
            from .terminal_value import embed_value
            if terminal_model.kind == "simple_inventory":
                terminal_expression = terminal_model.inventory_coefficient * params.battery_capacity_kwh * \
                    cp.quicksum(soc[i, b, ell + horizon] for i in range(nstations) for b in range(slot_counts[i]))
            else:
                feature_spec = feature_spec or FeatureSpec(params, terminal_model.variant)
                terminal_expressions, terminal_bounds, terminal_records = feature_spec.build_mip(
                    model, cp, COPT, window, soc, y, total, arrival_cases, service, failures, deadline)
                terminal_expression, _ = embed_value(model, cp, COPT, terminal_model,
                    terminal_expressions, terminal_bounds, feature_spec.names)
        model.setObjective(income - charging_cost - adjustment_cost - failure_cost + terminal_expression,
                           COPT.MAXIMIZE)
        build_seconds = time.perf_counter() - started_at
        model.solve()
        status_names = {COPT.OPTIMAL: "optimal", COPT.TIMEOUT: "time_limit", COPT.INTERRUPTED: "interrupted",
                        COPT.IMPRECISE: "suboptimal", COPT.INFEASIBLE: "infeasible",
                        COPT.INF_OR_UNB: "infeasible_or_unbounded", COPT.UNBOUNDED: "unbounded",
                        COPT.NODELIMIT: "node_limit", COPT.ITERLIMIT: "iteration_limit",
                        COPT.UNFINISHED: "unfinished", COPT.NUMERICAL: "numerical",
                        COPT.UNSTARTED: "unstarted"}
        status = status_names.get(model.status, f"copt_status_{model.status}")
        # A window with no requests or path choices is a pure LP. Its valid
        # solution is reported by HasLpSol, not HasMipSol.
        is_mip = bool(model.ismip)
        has_solution = model.hasmipsol if is_mip else model.haslpsol
        if not has_solution:
            details = {"period": ell, "horizon": horizon, "status": status,
                       "build_seconds": build_seconds, "solver_seconds": float(model.solvingtime),
                       "wall_seconds": time.perf_counter() - started_at,
                       "model_variables": int(model.getAttr(COPT.Attr.Cols)),
                       "model_constraints": int(model.getAttr(COPT.Attr.Rows)),
                       "model_binary_variables": int(model.getAttr(COPT.Attr.Bins)),
                       "terminal_feature_dimension": len(terminal_expressions),
                       "terminal_cases": terminal_records.get("case_count", 0),
                       "route_mode": route_mode, "charging_mode": charging_mode}
            if diagnostic_dir is not None:
                target = Path(diagnostic_dir)
                target.mkdir(parents=True, exist_ok=True)
                stem = target / f"failed_period_{ell:03d}"
                serialized = dict(details, state=window.state.to_dict(),
                                  requests=[r.__dict__ for r in window.requests])
                stem.with_suffix(".json").write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
                try:
                    model.write(str(stem.with_suffix(".mps")))
                except Exception as diagnostic_error:
                    stem.with_suffix(".export_error.txt").write_text(str(diagnostic_error), encoding="utf-8")
            error = MPCSolveError(f"MPC solve returned {status} without a feasible incumbent at period {ell}")
            error.diagnostics = details
            raise error

        def value(expression: Any) -> float:
            if isinstance(expression, (int, float)):
                return float(expression)
            return expression.x if isinstance(expression, cp.Var) else expression.getValue()

        paths: dict[str, list[int]] = {}
        for key, network in window.networks.items():
            selected = {arc[0]: arc[1] for arc in map(tuple, network.arcs) if y[key, arc].x > 0.5}
            node = network.origin
            route: list[int] = []
            seen = set()
            while node != "exit":
                if node in seen or node not in selected:
                    raise MPCSolveError(f"Invalid selected path for {key}")
                seen.add(node)
                node = selected[node]
                if isinstance(node, int):
                    route.append(node)
            paths[key] = route
        services = [ServiceDecision(request_id=rid, station=requests[rid].station, slot=slot, period=n)
                    for (rid, slot, n), avar in alpha.items() if avar.x > 0.5]
        services.sort(key=lambda item: (item.period, item.station, item.slot, item.request_id))
        served = {item.request_id: item.period for item in services}
        outcomes: dict[str, dict] = {}
        for rid, req in requests.items():
            chosen_cases = [t for t, expression in arrival_cases[rid] if value(expression) > 0.5]
            arrival = chosen_cases[0] if chosen_cases else None
            due = deadline(req, arrival) if arrival is not None else None
            selected = value(activation[rid]) > 0.5
            outcomes[rid] = {
                "selected": selected, "arrival_time": arrival, "deadline": due,
                "served": rid in served, "service_period": served.get(rid),
                "failed": rid in failures and value(failures[rid]) > 0.5,
                "waiting_at_end": selected and rid not in served and arrival is not None
                                  and arrival <= end <= due,
            }
        terms = {"income": value(income), "charging_cost": value(charging_cost),
                 "adjustment_cost": value(adjustment_cost), "failure_cost": value(failure_cost)}
        evaluated_terminal = value(terminal_expression)
        numeric_features = [value(expression) for expression in terminal_expressions]
        diagnostics = {"terminal_feature_dimension": len(numeric_features),
                       "terminal_cases": terminal_records.get("case_count", 0),
                       "model_variables": int(model.getAttr(COPT.Attr.Cols)),
                       "model_constraints": int(model.getAttr(COPT.Attr.Rows)),
                       "model_binary_variables": int(model.getAttr(COPT.Attr.Bins))}
        if terminal_model is not None:
            terms["terminal_value"] = evaluated_terminal
            if numeric_features:
                independently_evaluated = terminal_model.predict(numeric_features)
                error = abs(independently_evaluated - evaluated_terminal)
                diagnostics["terminal_network_evaluation_error"] = error
                if error > max(1e-4, 1e-7 * abs(independently_evaluated)):
                    raise MPCSolveError(f"terminal network/MIP evaluation mismatch: {error}")
                for feature, (low, high) in zip(numeric_features, terminal_bounds):
                    if not low - 1e-6 <= feature <= high + 1e-6:
                        raise MPCSolveError("terminal feature exceeds its mathematical bounds")
        result = MPCSolution(
            status=status, objective=float(model.objval),
            objective_terms=terms,
            paths=paths, services=services,
            power=[[[max(0.0, power[i, b, n].x) for n in periods] for b in range(slot_counts[i])] for i in range(nstations)],
            soc=[[[min(1.0, max(0.0, soc[i, b, n].x)) for n in range(ell, ell + horizon + 1)]
                  for b in range(slot_counts[i])] for i in range(nstations)],
            request_outcomes=outcomes, mip_gap=float(model.bestgap) if is_mip else 0.0,
            solve_seconds=float(model.solvingtime),
        )
        result.terminal_features = numeric_features
        result.terminal_value = evaluated_terminal
        result.build_seconds = build_seconds
        result.wall_seconds = time.perf_counter() - started_at
        result.diagnostics = diagnostics
        return result
    except cp.CoptError as exc:
        context = "Unable to initialize COPT" if initializing else f"COPT failed at period {ell}"
        raise MPCSolveError(f"{context}: {exc}") from exc
    finally:
        active_error = sys.exc_info()[0] is not None
        # COPT's Python model has no dispose/close method; its native resources
        # are owned by the Python model and released with the local references.
        del model
        if environment is not None:
            try:
                environment.close()
            except cp.CoptError as exc:
                # A connection-close error must not hide the original license,
                # model-construction or solve failure.
                if not active_error:
                    raise MPCSolveError(f"Unable to close COPT environment: {exc}") from exc
