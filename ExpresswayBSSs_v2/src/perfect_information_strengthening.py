"""Valid recharge cuts and lossless relabeling of identical battery slots."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from math import ceil

from .parameters import slots_at


def slot_groups(params, window, station):
    groups = defaultdict(list)
    for b in range(slots_at(params, station)):
        groups[(window.state.slot_soc[station][b], params.slot_power_limit(station, b))].append(b)
    return [group for _, group in sorted(groups.items()) if len(group) > 1]


def canonicalize_slots(params, window, solution, *, order='first_use'):
    """Relabel complete trajectories within identical (SOC, power) classes."""
    if order not in {'first_use', 'total_services'}:
        raise ValueError('unknown slot ordering')
    result = deepcopy(solution)
    first = {}
    counts = defaultdict(int)
    for decision in solution.services:
        key = (decision.station, decision.slot)
        counts[key] += 1
        first[key] = min(first.get(key, (window.horizon + 1, '')), (decision.period, decision.request_id))
    mapping = {}
    for i in range(params.station.num_stations):
        for group in slot_groups(params, window, i):
            ordered = (sorted(group, key=lambda b: (-counts[i, b], b)) if order == 'total_services' else
                       sorted(group, key=lambda b: (*first.get((i, b), (window.horizon + 1, '')), b)))
            for new, old in zip(group, ordered):
                mapping[i, old] = new
                result.power[i][new] = deepcopy(solution.power[i][old])
                result.soc[i][new] = deepcopy(solution.soc[i][old])
    result.services = [replace(d, slot=mapping.get((d.station, d.slot), d.slot)) for d in solution.services]
    return result, [{'station': i, 'old_slot': old, 'new_slot': new}
                    for (i, old), new in sorted(mapping.items()) if old != new]


def add_total_service_order(cp, model, params, window, requests, alpha):
    """Order whole-day counts, never counts at every intermediate boundary.

Within each identical slot class, a single global permutation of service,
power and SOC trajectories sorts these totals. Thus every original feasible
operating plan has an equally valuable representative. No auxiliary variables.
"""
    by_slot = defaultdict(list)
    for (rid, b, n), variable in alpha.items():
        by_slot[requests[rid].station, b].append(variable)
    rows = 0
    for i in range(params.station.num_stations):
        for group in slot_groups(params, window, i):
            totals = {b: cp.quicksum(by_slot[i, b]) for b in group}
            for before, after in zip(group, group[1:]):
                model.addConstr(totals[before] >= totals[after], name=f'total_service_order[{i},{before},{after}]')
                rows += 1
    return rows


def recharge_periods(return_soc, soc_gain, horizon):
    # Subtract only a numerical margin: this weakens, never strengthens, cuts
    # at floating-point representations of exact grid boundaries.
    deficit = max(0., 1. - return_soc - 1e-10)
    return max(1, ceil(deficit / soc_gain)) if soc_gain > 0 else (horizon + 1 if deficit > 0 else 1)


def cooldown_sets(params, window, requests, alpha_keys):
    """Yield single-slot interval cliques, only at possible service boundaries.

    A swap at n occupies its slot through n+k-1, where k is the earliest full
    recharge time at maximum slot power. Intersecting intervals cannot share a
    physical slot, regardless of station power competition or electricity price.
    """
    events = defaultdict(list)
    for rid, b, n in alpha_keys:
        events[requests[rid].station, b].append((rid, b, n))
    for (i, b), keys in sorted(events.items()):
        gain = params.station.charging_efficiency * params.interval_hours * params.slot_power_limit(i, b) / params.battery_capacity_kwh
        durations = {rid: recharge_periods(requests[rid].return_soc, gain, window.horizon) for rid, _, _ in keys}
        initial_delay = 0 if window.state.slot_soc[i][b] >= 1. - 1e-10 else recharge_periods(window.state.slot_soc[i][b], gain, window.horizon)
        for t in sorted({n for _, _, n in keys}):
            covered = [key for key in keys if key[2] <= t < key[2] + durations[key[0]]]
            if t < initial_delay or any(key[2] < t for key in covered):
                yield i, b, t, covered, 0 if t < initial_delay else 1


def add_strengthening(cp, model, var, params, window, requests, alpha, service, total,
                      activation, arrivals, failure, *, first_use=True):
    extra_rows, cooldown_rows, symmetry_rows = 0, 0, 0
    for rid in sorted(requests):
        arrived = cp.quicksum(v for _, v in arrivals[rid])
        model.addConstr(total[rid] <= arrived)
        extra_rows += 1
        if rid in failure:
            # Complete the convex-hull inequalities for served+failed =
            # active AND arrived. The matching lower inequality already exists.
            model.addConstr(total[rid] + failure[rid] <= arrived)
            model.addConstr(total[rid] + failure[rid] <= activation[rid])
            extra_rows += 2
    for i, b, t, keys, upper in cooldown_sets(params, window, requests, alpha):
        model.addConstr(cp.quicksum(alpha[key] for key in keys) <= upper)
        cooldown_rows += 1

    if not first_use:
        return {}, dict(arrival_hull_rows=extra_rows, cooldown_rows=cooldown_rows,
                        first_use_rows=0, first_use_auxiliaries=0)

    swaps = defaultdict(list)
    times = defaultdict(set)
    for (rid, b, n), value in alpha.items():
        station = requests[rid].station
        swaps[station, b, n].append(value)
        times[station].add(n)
    prefix = {}
    for i in sorted(times):
        periods = sorted(times[i])
        for group in slot_groups(params, window, i):
            for before, after in zip(group, group[1:]):
                previous = 0.
                for n in periods:
                    used = var(lb=0., ub=n + 1., name=f'used_count[{i},{before},{n}]')
                    model.addConstr(used == previous + cp.quicksum(swaps[i, before, n]))
                    # Only first-use ordering, NOT cumulative-count dominance.
                    model.addConstr(cp.quicksum(swaps[i, after, n]) <= used)
                    prefix[i, before, n] = used
                    previous = used
                    symmetry_rows += 2
    return prefix, dict(arrival_hull_rows=extra_rows, cooldown_rows=cooldown_rows,
                        first_use_rows=symmetry_rows, first_use_auxiliaries=len(prefix))
