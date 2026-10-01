"""Exact continuous charging subproblem with all service/slot decisions fixed.

Substitute fixed swaps into the original full-SOC, SOC balance and power rows.
Keep all stations in one LP, the full horizon, cyclic prices and no terminal
salvage/target. Route and queue feasibility must also pass the original replay.
"""
from copy import deepcopy
from math import fsum
from time import perf_counter

from .parameters import price_at, slots_at
from .perfect_information_checks import model_fingerprints


def solve_fixed_assignment_charging(params, window, initial, *, log_path=None):
    import coptpy as cp
    from coptpy import COPT

    started = perf_counter()
    if window.ell != 0 or window.state.period != 0:
        raise ValueError('charging diagnosis requires the complete demand day')
    h, delta = window.horizon, params.interval_hours
    counts = [slots_at(params, i) for i in range(params.station.num_stations)]
    requests = {r.request_id: r for r in window.requests}
    fixed, seen = {}, set()
    for d in initial.services:
        key = (d.station, d.slot, d.period)
        if (d.request_id in seen or d.request_id not in requests or key in fixed
                or requests[d.request_id].station != d.station or not 0 <= d.station < len(counts)
                or not 0 <= d.slot < counts[d.station] or not 0 <= d.period < h):
            raise ValueError('invalid fixed service assignment')
        fixed[key] = requests[d.request_id]
        seen.add(d.request_id)
    config = cp.EnvrConfig()
    config.set('nobanner', '1')
    environment = cp.Envr(config)
    model = None
    try:
        model = environment.createModel('fixed_assignment_charging_lp')
        for key, val in [('Logging', params.solver.output_flag), ('Threads', params.solver.threads),
                         ('TimeLimit', params.solver.time_limit_sec), ('FeasTol', params.solver.feasibility_tol)]:
            model.setParam(getattr(COPT.Param, key), val)
        if log_path:
            model.setLogFile(str(log_path))
        power, soc = {}, {}
        cost = cp.LinExpr()
        for i, count in enumerate(counts):
            for b in range(count):
                for n in range(h + 1):
                    soc[i, b, n] = model.addVar(lb=0., ub=1., name=f'S[{i},{b},{n}]')
                model.addConstr(soc[i, b, 0] == window.state.slot_soc[i][b])
                for n in range(h):
                    power[i, b, n] = model.addVar(lb=0., ub=params.slot_power_limit(i, b), name=f'P[{i},{b},{n}]')
                    req = fixed.get((i, b, n))
                    deficit = 0. if req is None else 1. - req.return_soc
                    if req is not None:
                        model.addConstr(soc[i, b, n] == 1.)
                    model.addConstr(soc[i, b, n + 1] == soc[i, b, n] - deficit +
                        params.station.charging_efficiency * delta / params.battery_capacity_kwh * power[i, b, n])
                    cost += delta * price_at(params, 'electricity_price', i, n) * power[i, b, n]
            for n in range(h):
                model.addConstr(cp.quicksum(power[i, b, n] for b in range(count)) <= params.station_power_limit(i))
        model.setObjective(cost, COPT.MINIMIZE)
        model.update()
        diagnostics = dict(model_variables=model.getAttr(COPT.Attr.Cols),
            model_constraints=model.getAttr(COPT.Attr.Rows), model_binary_variables=model.getAttr(COPT.Attr.Bins),
            fingerprints=model_fingerprints(model), fixed_service_count=len(fixed),
            bound_scope='fixed_assignment_only')
        if model.ismip:
            raise ValueError('charging-only diagnosis unexpectedly contains integer variables')
        build_seconds = perf_counter() - started
        model.solveLP()
        optimal = model.lpstatus == COPT.OPTIMAL
        result = dict(status='optimal' if optimal else f'lp_status_{model.lpstatus}',
                      diagnostics=diagnostics, solve_seconds=float(model.solvingtime),
                      build_seconds=build_seconds, bound_scope='fixed_assignment_only',
                      has_incumbent=optimal, charging_cost=None, charging_cost_lower_bound=None, solution=None)
        if optimal:
            values, _, _, _ = model.getLpSolution()
            sol = deepcopy(initial)
            sol.power = [[[values[power[i, b, n].index] for n in range(h)] for b in range(count)]
                         for i, count in enumerate(counts)]
            sol.soc = [[[values[soc[i, b, n].index] for n in range(h + 1)] for b in range(count)]
                       for i, count in enumerate(counts)]
            solved_cost = float(model.lpobjval)
            if solved_cost > initial.objective_terms['charging_cost'] + 1e-4:
                raise ValueError('optimal LP is worse than the known feasible charging schedule')
            income = fsum(params.battery_capacity_kwh * (1-r.return_soc) *
                          price_at(params, 'swap_service_price', i, n) for (i, b, n), r in fixed.items())
            if abs(income - initial.objective_terms['income']) > 1e-4:
                raise ValueError('fixed service income differs from the incumbent')
            sol.objective_terms.update(income=income, charging_cost=solved_cost)
            t = sol.objective_terms
            sol.objective = t['income'] - t['charging_cost'] - t['failure_cost'] - t['adjustment_cost']
            sol.status, sol.mip_gap = 'fixed_assignment_lp_optimal', None
            sol.solve_seconds, sol.build_seconds = result['solve_seconds'], build_seconds
            sol.wall_seconds, sol.diagnostics = perf_counter() - started, diagnostics
            result.update(charging_cost=solved_cost, charging_cost_lower_bound=solved_cost,
                          incumbent_objective=sol.objective, solution=sol.to_dict())
        result['wall_seconds'] = perf_counter() - started
        return result
    finally:
        del model
        environment.close()
