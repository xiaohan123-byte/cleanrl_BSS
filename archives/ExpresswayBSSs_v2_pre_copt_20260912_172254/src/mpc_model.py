"""Joint path, swap-assignment and charging MILP without terminal value.

The formulation follows section 3 of the revised paper. Arrival eligibility is
an exact sum of predecessor service decisions. FCFS comparisons use integer
ranks of the finite possible arrival times, so they require no arbitrary
big-M bound in physical time and no positive time epsilon.
"""

from __future__ import annotations

import math
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


def solve_mpc(params: Any, window: MPCWindow) -> MPCSolution:
    """Solve one prediction window; only its first stage may be executed.

    ``power`` and ``soc`` use offsets into this window; services use absolute
    period indices. A time-limited feasible incumbent is returned with its gap.
    There is deliberately no heuristic fallback when no incumbent exists.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise MPCSolveError("Gurobi is required to solve the baseline MILP") from exc

    ell, horizon = window.ell, window.horizon
    if horizon < 1 or ell < 0 or ell + horizon > params.num_periods:
        raise ValueError("MPC window must lie within the configured operating periods")
    if window.state.period != ell:
        raise ValueError("MPC initial state and window period differ")
    delta = params.interval_hours
    periods = range(ell, ell + horizon)
    end = (ell + horizon) * delta
    nstations, nslots = params.station.num_stations, params.station.num_slots
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

    try:
        model = gp.Model("bss_discrete_zero_terminal")
    except gp.GurobiError as exc:
        raise MPCSolveError(f"Unable to initialize Gurobi: {exc}") from exc
    try:
        solver = params.solver
        model.Params.OutputFlag = solver.output_flag
        model.Params.Threads = solver.threads
        model.Params.TimeLimit = solver.time_limit_sec
        model.Params.MIPGap = solver.mip_gap
        model.Params.FeasibilityTol = solver.feasibility_tol
        model.Params.IntFeasTol = min(1e-8, solver.feasibility_tol)

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
                y[key, arc] = model.addVar(vtype=GRB.BINARY, name=f"y[{key},{arc}]")
            for node in nodes:
                outflow = gp.quicksum(y[key, arc] for arc in arcs if arc[0] == node)
                inflow = gp.quicksum(y[key, arc] for arc in arcs if arc[1] == node)
                rhs = 1 if node == network.origin else -1 if node == "exit" else 0
                model.addConstr(outflow - inflow == rhs, name=f"flow[{key},{node}]")
            differences = []
            reference = set(network.reference_stations)
            frozen = set(network.frozen_stations)
            for station in range(nstations):
                x = gp.quicksum(y[key, arc] for arc in arcs if arc[1] == station)
                model.addConstr(x <= 1)
                if not network.can_update:
                    model.addConstr(x == int(station in frozen), name=f"freeze[{key},{station}]")
                differences.append(1 - x if station in reference else x)
            chi = model.addVar(vtype=GRB.BINARY, name=f"changed[{key}]")
            changed[key] = chi
            for difference in differences:
                model.addConstr(chi >= difference)
            model.addConstr(chi <= gp.quicksum(differences))

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
                for slot in range(nslots):
                    alpha[rid, slot, n] = model.addVar(vtype=GRB.BINARY, name=f"a[{rid},{slot},{n}]")
                service[rid, n] = gp.quicksum(alpha[rid, slot, n] for slot in range(nslots))
            total[rid] = gp.quicksum(service[rid, n] for n in possible_periods[rid])
            model.addConstr(total[rid] <= activation[rid], name=f"once[{rid}]")
            cases = [(t, 1 if q is None else service[q, m]) for t, q, m in possible_cases[rid]]
            arrival_cases[rid] = cases
            # At most one predecessor is served; this is implied by path flow,
            # but explicit here also validates manually constructed windows.
            if req.predecessors:
                model.addConstr(gp.quicksum(value for _, value in cases) <= 1)
            for n in possible_periods[rid]:
                eligible = gp.quicksum(value for t, value in cases
                                       if t <= n * delta <= deadline(req, t))
                previous = gp.quicksum(service[rid, m] for m in possible_periods[rid] if m < n)
                qvar = model.addVar(vtype=GRB.BINARY, name=f"Q[{rid},{n}]")
                queue[rid, n] = qvar
                model.addConstr(qvar <= activation[rid])
                model.addConstr(qvar <= eligible)
                model.addConstr(qvar <= 1 - previous)
                model.addConstr(qvar >= activation[rid] + eligible - previous - 1)
                model.addConstr(service[rid, n] <= qvar)
            if req.kind == "reservation":
                expired = gp.quicksum(value for t, value in cases if deadline(req, t) < end)
                fail = model.addVar(vtype=GRB.BINARY, name=f"failed[{rid}]")
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
            model.addConstr(gp.quicksum(values) <= 1, name=f"one_failure[{key}]")

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
                qrank = gp.quicksum(ranks[t] * value for t, value in qc)
                rrank = gp.quicksum(ranks[t] * value for t, value in rc)
                bound = rank + 1
                for first, second, difference in ((q, r, qrank - rrank), (r, q, rrank - qrank)):
                    earlier = model.addVar(vtype=GRB.BINARY, name=f"earlier[{first},{second}]")
                    model.addConstr(difference <= bound * (1 - earlier) - 1)
                    model.addConstr(difference >= -bound * earlier)
                    priority(first, second, shared, earlier)

        power: dict[tuple[int, int, int], Any] = {}
        soc: dict[tuple[int, int, int], Any] = {}
        eta = params.station.charging_efficiency
        charging_cost = gp.LinExpr()
        for station in range(nstations):
            for slot in range(nslots):
                for n in range(ell, ell + horizon + 1):
                    soc[station, slot, n] = model.addVar(lb=0, ub=1, name=f"S[{station},{slot},{n}]")
                model.addConstr(soc[station, slot, ell] == window.state.slot_soc[station][slot])
                for n in periods:
                    pvar = model.addVar(lb=0, ub=params.slot_power_limit(station, slot),
                                        name=f"P[{station},{slot},{n}]")
                    power[station, slot, n] = pvar
                    swaps = [(rid, alpha[rid, slot, n]) for rid in by_station[station]
                             if (rid, slot, n) in alpha]
                    model.addConstr(gp.quicksum(a for _, a in swaps) <= 1)
                    for _, avar in swaps:
                        model.addConstr(soc[station, slot, n] >= avar)
                    reduction = gp.quicksum((1 - requests[rid].return_soc) * a for rid, a in swaps)
                    model.addConstr(soc[station, slot, n + 1] == soc[station, slot, n] - reduction
                                    + eta * delta / params.battery_capacity_kwh * pvar)
                    charging_cost += delta * params.electricity_price[station][n] * pvar
            for n in periods:
                model.addConstr(gp.quicksum(power[station, slot, n] for slot in range(nslots))
                                <= params.station_power_limit(station))

        income = gp.quicksum(params.battery_capacity_kwh * (1 - requests[rid].return_soc)
                             * params.swap_service_price[requests[rid].station][n] * avar
                             for (rid, _, n), avar in alpha.items())
        adjustment_cost = params.path_adjustment_penalty * gp.quicksum(changed.values())
        failure_cost = params.reservation_failure_penalty * gp.quicksum(failures.values())
        model.setObjective(income - charging_cost - adjustment_cost - failure_cost, GRB.MAXIMIZE)
        model.optimize()
        status_names = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit", GRB.INTERRUPTED: "interrupted",
                        GRB.SUBOPTIMAL: "suboptimal", GRB.INFEASIBLE: "infeasible",
                        GRB.INF_OR_UNBD: "infeasible_or_unbounded", GRB.UNBOUNDED: "unbounded",
                        GRB.NODE_LIMIT: "node_limit", GRB.ITERATION_LIMIT: "iteration_limit"}
        status = status_names.get(model.Status, f"gurobi_status_{model.Status}")
        if model.SolCount == 0:
            raise MPCSolveError(f"MPC solve returned {status} without a feasible incumbent at period {ell}")

        def value(expression: Any) -> float:
            if isinstance(expression, (int, float)):
                return float(expression)
            return expression.X if isinstance(expression, gp.Var) else expression.getValue()

        paths: dict[str, list[int]] = {}
        for key, network in window.networks.items():
            selected = {arc[0]: arc[1] for arc in map(tuple, network.arcs) if y[key, arc].X > 0.5}
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
                    for (rid, slot, n), avar in alpha.items() if avar.X > 0.5]
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
        return MPCSolution(
            status=status, objective=float(model.ObjVal),
            objective_terms={"income": value(income), "charging_cost": value(charging_cost),
                             "adjustment_cost": value(adjustment_cost), "failure_cost": value(failure_cost)},
            paths=paths, services=services,
            power=[[[max(0.0, power[i, b, n].X) for n in periods] for b in range(nslots)] for i in range(nstations)],
            soc=[[[min(1.0, max(0.0, soc[i, b, n].X)) for n in range(ell, ell + horizon + 1)]
                  for b in range(nslots)] for i in range(nstations)],
            request_outcomes=outcomes, mip_gap=float(model.MIPGap) if model.IsMIP else 0.0, solve_seconds=float(model.Runtime),
        )
    except gp.GurobiError as exc:
        raise MPCSolveError(f"Gurobi failed at period {ell}: {exc}") from exc
    finally:
        model.dispose()
