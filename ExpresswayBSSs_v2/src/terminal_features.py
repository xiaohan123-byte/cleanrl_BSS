"""Fixed business features and their exact decision-dependent MILP expressions.

Time is in hours. The fixed template set never depends on sampled scenarios.
Within each solve, terminal users are enumerated by selected path and last
predecessor service time. Thus all non-inventory candidate fields are constants
selected by binary decisions; no continuously variable time is rounded to a bin.
"""
from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np

from .candidate_network import enumerate_paths
from .domain import user_key_text
from .terminal_value import bounded_relu

SOC_THRESHOLDS = (0.25, 0.5, 0.75, 0.9, 0.95)
TIME_LIMITS = (0.25, 1.0, 3.0)


def _direction(params, user):
    od = params.od_pairs[params.od_index(user.user_key[0])]
    return 1 if od.exit_km > od.entry_km else -1


def _remaining(params, user, plan, position, completed, waiting_station=None):
    direction = _direction(params, user)
    excluded = set(completed)
    if waiting_station is not None:
        excluded.add(waiting_station)
    return tuple(i for i in (plan or []) if i not in excluded
                 and direction * (params.station.positions_km[i] - position) >= -1e-9)


def _time_class(arrival, deadline, now, waiting=None):
    if waiting is None:
        waiting = arrival <= now <= deadline
    if waiting:
        remaining = max(0.0, deadline - now)
        return (4 if remaining <= 5.0 / 60.0 else 5), remaining
    remaining = max(0.0, arrival - now)
    for index, high in enumerate(TIME_LIMITS):
        if remaining <= high:
            return index, remaining
    return 3, remaining


def _random_predictions(params, start, end, include_start=False):
    """Reuse the environment-independent, fixed absolute-time forecast series."""
    from .forecast import deterministic_random_requests
    lower = math.nextafter(start, -math.inf) if include_start else start
    records = deterministic_random_requests(params, lower, end)
    return [dict(record, deadline=record["arrival_time"] + params.max_wait_hours)
            for record in records]


class FeatureSpec:
    """The selected 12A/13A/18A/19A schema and fixed business scaling."""
    def __init__(self, params, variant="full"):
        if variant not in {"full", "inventory_only", "simple_inventory"}:
            raise ValueError("unknown terminal feature variant")
        self.params, self.variant = params, variant
        self.names = []
        self.index = {}
        self.groups = []
        self.group_offsets = {}
        self.random_offsets = {}
        self.inventory_indices = []
        self.nstations = params.station.num_stations
        self.nslots = params.station.num_slots
        endpoints = list(getattr(params.station, "positions_km", [0.0, 1.0]))
        for od in getattr(params, "od_pairs", []):
            endpoints.extend((od.entry_km, od.exit_km))
        self.network_length = max(1.0, max(endpoints) - min(endpoints))
        if variant == "simple_inventory":
            self._add("inventory_kwh")
            return
        for station in range(self.nstations):
            indices = [self._add(f"battery:{station}:mean_soc")]
            indices.extend(self._add(f"battery:{station}:hinge:{threshold:g}") for threshold in SOC_THRESHOLDS)
            self.inventory_indices.append(indices)
        if variant == "full":
            templates = set()
            for od in params.od_pairs:
                direction = 1 if od.exit_km > od.entry_km else -1
                stations = tuple(od.station_indices)
                for length in range(1, len(stations) + 1):
                    for seq in combinations(stations, length):
                        if any(abs(params.station.positions_km[b] - params.station.positions_km[a])
                               > params.range_km + 1e-8 for a, b in zip(seq, seq[1:])):
                            continue
                        if abs(od.exit_km - params.station.positions_km[seq[-1]]) / params.range_km                                 + params.min_exit_soc > 1 + 1e-8:
                            continue
                        templates.add((direction, seq))
            self.groups = sorted(templates, key=lambda item: (item[0], len(item[1]), item[1]))
            for group in self.groups:
                direction, seq = group
                prefix = "chain:" + str(direction) + ":" + ",".join(map(str, seq))
                start = len(self.names)
                for time_class in range(6):
                    for field in ("count", "not_entered", "next_return_soc", "remaining_hours"):
                        self._add(f"{prefix}:time:{time_class}:{field}")
                for field in ("position", "soc", "entry", "exit", "last_swap"):
                    self._add(f"{prefix}:{field}")
                for kind in ("dayahead", "published"):
                    for station in range(self.nstations):
                        self._add(f"{prefix}:{kind}:difference:{station}")
                for kind in ("dayahead", "published"):
                    self._add(f"{prefix}:{kind}:changed_count")
                for order in range(1, len(seq)):
                    self._add(f"{prefix}:later:{order}:return_soc")
                for order in range(1, len(seq)):
                    self._add(f"{prefix}:edge:{order}:travel_hours")
                self.group_offsets[group] = start
            for station in range(self.nstations):
                self.random_offsets[station] = len(self.names)
                for time_class in range(6):
                    for field in ("count", "return_soc", "remaining_hours"):
                        self._add(f"random:{station}:time:{time_class}:{field}")
        self.external_start = len(self.names)
        for name in ("time_fraction", "remaining_fraction", "phase:0", "phase:1", "phase:2"):
            self._add(name)
        for station in range(self.nstations):
            self._add(f"external:{station}:current_buy")
            self._add(f"external:{station}:current_swap")
        for k in range(24):
            for station in range(self.nstations):
                self._add(f"external:window:{k}:station:{station}:buy")
                self._add(f"external:window:{k}:station:{station}:swap")
                self._add(f"external:window:{k}:station:{station}:random_expected")
            self._add(f"external:window:{k}:valid_duration")

    def _add(self, name):
        self.index[name] = len(self.names)
        self.names.append(name)
        return len(self.names) - 1

    @property
    def dimension(self):
        return len(self.names)

    def to_dict(self):
        return {"schema_version": 1, "variant": self.variant, "names": list(self.names),
                "dimension": self.dimension, "chain_templates": [[d, list(s)] for d, s in self.groups],
                "soc_thresholds": list(SOC_THRESHOLDS), "time_limits_hours": list(TIME_LIMITS),
                "count_scale": 100.0, "time_scale_hours": 24.0,
                "inventory_count_scale": self.nslots, "network_length_km": self.network_length,
                "random_forecast_feature_extent": "state_time_to_operating_end"}

    def _external(self, period):
        p = self.params
        tau = period * p.interval_hours
        end = p.num_periods * p.interval_hours
        values = [tau / 24.0, max(0.0, end - tau) / 24.0]
        values.extend(float(period % p.path_update_interval == phase) for phase in range(3))
        for station in range(self.nstations):
            values.extend((p.electricity_price[station][period], p.swap_service_price[station][period])
                          if period < p.num_periods else (0.0, 0.0))
        for k in range(24):
            left, right = tau + k, min(tau + k + 1, end)
            length = max(0.0, right - left)
            for station in range(self.nstations):
                buy, swap, expected = 0.0, 0.0, 0.0
                if length:
                    for n in range(max(0, int(math.floor(left / p.interval_hours))),
                                   min(p.num_periods, int(math.ceil(right / p.interval_hours)))):
                        overlap = max(0.0, min(right, (n + 1) * p.interval_hours) - max(left, n * p.interval_hours))
                        buy += overlap * p.electricity_price[station][n]
                        swap += overlap * p.swap_service_price[station][n]
                    for hour in range(int(math.floor(left)), int(math.ceil(right))):
                        overlap = max(0.0, min(right, hour + 1.0) - max(left, float(hour)))
                        rate = p.random_rate_at(station, float(hour)) if hasattr(p, "random_rate_at")                             else p.random_arrival_rate_per_hour[station]
                        expected += overlap * rate
                    buy, swap = buy / length, swap / length
                values.extend((buy, swap, expected / 100.0))
            values.append(length)
        return values

    def _user_coefficients(self, user, chain, rho, arrival, due, motion, tau,
                           completed=(), published=None, later_rho=None, later_travel=None):
        if not chain or due < tau:
            return {}
        p = self.params
        group = (_direction(p, user), tuple(chain))
        if group not in self.group_offsets:
            raise ValueError(f"service chain not covered by fixed feature templates: {group}")
        start = self.group_offsets[group]
        time_class, remaining = _time_class(arrival, due, tau)
        coefficients = {start + 4 * time_class: 0.01,
                        start + 4 * time_class + 1: 0.0 if motion["entered"] else 0.01,
                        start + 4 * time_class + 2: rho / 100.0,
                        start + 4 * time_class + 3: remaining / 2400.0}
        od = p.od_pairs[p.od_index(user.user_key[0])]
        cursor = start + 24
        positions = (motion["position_km"], motion["soc"], od.entry_km, od.exit_km,
                     motion["last_swap_position_km"])
        for offset, value in enumerate(positions):
            coefficients[cursor + offset] = value / (100.0 if offset == 1 else 100.0 * self.network_length)
        cursor += 5
        waiting_station = chain[0] if arrival <= tau <= due else None
        current = _remaining(p, user, chain, motion["position_km"], completed, waiting_station)
        references = [user.day_ahead, published]
        flags = []
        for reference in references:
            if reference is None:
                differences = [0.0] * self.nstations
            else:
                ref = _remaining(p, user, reference, motion["position_km"], completed, waiting_station)
                differences = [float(i in current) - float(i in ref) for i in range(self.nstations)]
            for value in differences:
                coefficients[cursor] = value / 100.0
                cursor += 1
            flags.append(float(any(differences)))
        for value in flags:
            coefficients[cursor] = value / 100.0
            cursor += 1
        if later_rho is None:
            later_rho = [1 - abs(p.station.positions_km[b] - p.station.positions_km[a]) / p.range_km
                         for a, b in zip(chain, chain[1:])]
        if later_travel is None:
            later_travel = [abs(p.station.positions_km[b] - p.station.positions_km[a]) / p.vehicle_speed_kmh
                            for a, b in zip(chain, chain[1:])]
        for value in later_rho:
            coefficients[cursor] = value / 100.0
            cursor += 1
        for value in later_travel:
            coefficients[cursor] = value / 2400.0
            cursor += 1
        return {index: value for index, value in coefficients.items() if value != 0}

    def _random_coefficients(self, station, rho, arrival, due, tau):
        if due < tau:
            return {}
        time_class, remaining = _time_class(arrival, due, tau)
        start = self.random_offsets[station] + 3 * time_class
        return {start: 0.01, start + 1: rho / 100.0, start + 2: remaining / 2400.0}

    def encode_window(self, window):
        p = self.params
        state = window.state
        tau = state.period * p.interval_hours
        output = np.zeros(self.dimension, dtype=float)
        if self.variant == "simple_inventory":
            output[0] = p.battery_capacity_kwh * sum(map(sum, state.slot_soc))
            return output
        for station, indices in enumerate(self.inventory_indices):
            row = np.asarray(state.slot_soc[station])
            output[indices[0]] = row.sum() / self.nslots
            for index, threshold in zip(indices[1:], SOC_THRESHOLDS):
                output[index] = np.maximum(row - threshold, 0).sum() / (self.nslots * (1 - threshold))
        output[self.external_start:] = self._external(state.period)
        if self.variant != "full":
            return output
        for key, user in state.users.items():
            if user.status != "active":
                continue
            waiting = state.waiting.get(user.waiting_request_id)
            raw_plan = user.published_plan if user.entered else user.retained_plan
            chain = list(_remaining(p, user, raw_plan, user.position_km, user.completed_stations,
                                    waiting.station if waiting else None))
            if waiting:
                chain.insert(0, waiting.station)
            if not chain:
                continue
            if waiting:
                rho, arrival, due = waiting.return_soc, waiting.arrival_time, waiting.deadline
            else:
                candidates = [r for r in window.requests if r.user_key == user.user_key
                              and r.station == chain[0] and not r.predecessors]
                if not candidates:
                    raise ValueError(f"missing next request for retained user {key}")
                req = candidates[0]
                rho, arrival = req.return_soc, req.arrival_time
                due = req.deadline if req.deadline is not None else arrival + p.max_wait_hours
            motion = {"position_km": user.position_km, "soc": user.soc, "entered": user.entered,
                      "last_swap_position_km": user.last_swap_position_km}
            coefficients = self._user_coefficients(user, chain, rho, arrival, due, motion, tau,
                                                  user.completed_stations, user.published_plan)
            for index, value in coefficients.items():
                output[index] += value
        for req in state.waiting.values():
            if req.kind == "random":
                for index, value in self._random_coefficients(req.station, req.return_soc, req.arrival_time,
                                                            req.deadline, tau).items():
                    output[index] += value
        for req in _random_predictions(p, tau, p.num_periods * p.interval_hours):
            for index, value in self._random_coefficients(req["station"], req["return_soc"], req["arrival_time"],
                                                        req["deadline"], tau).items():
                output[index] += value
        return output

    def build_mip(self, model, cp, COPT, window, soc, y, total, arrival_cases, service,
                  failures, deadline):
        """Return exact feature expressions, safe bounds, and evaluation records."""
        p = self.params
        end_period = window.ell + window.horizon
        tau = end_period * p.interval_hours
        expressions = [0.0] * self.dimension
        bounds = [[0.0, 0.0] for _ in self.names]
        terms = defaultdict(list)
        case_records = []
        selectors = []
        if self.variant == "simple_inventory":
            expressions[0] = p.battery_capacity_kwh * cp.quicksum(soc[i, b, end_period]
                                for i in range(self.nstations) for b in range(self.nslots))
            bounds[0] = [0.0, p.battery_capacity_kwh * self.nstations * self.nslots]
            return expressions, bounds, {"cases": [], "selectors": []}
        for station, indices in enumerate(self.inventory_indices):
            expressions[indices[0]] = cp.quicksum(soc[station, b, end_period] for b in range(self.nslots)) / self.nslots
            bounds[indices[0]] = [0.0, 1.0]
            for index, threshold in zip(indices[1:], SOC_THRESHOLDS):
                pieces = [bounded_relu(model, cp, COPT, soc[station, b, end_period] - threshold,
                                       -threshold, 1 - threshold, f"soc_hinge[{station},{b},{threshold}]")
                          for b in range(self.nslots)]
                expressions[index] = cp.quicksum(pieces) / (self.nslots * (1 - threshold))
                bounds[index] = [0.0, 1.0]
        external = self._external(end_period)
        for offset, value in enumerate(external):
            expressions[self.external_start + offset] = value
            bounds[self.external_start + offset] = [value, value]
        if self.variant != "full":
            return expressions, bounds, {"cases": [], "selectors": []}
        requests = {r.request_id: r for r in window.requests}
        counter = [0]

        def and_selector(factors, label):
            constants = [factor for factor in factors if isinstance(factor, (int, float))]
            if any(value < 0.5 for value in constants):
                return 0.0
            variables = [factor for factor in factors if not isinstance(factor, (int, float))]
            if not variables:
                return 1.0
            if len(variables) == 1:
                return variables[0]
            counter[0] += 1
            z = model.addVar(vtype=COPT.BINARY, name=f"terminal_case[{label},{counter[0]}]")
            for factor in variables:
                model.addConstr(z <= factor)
            model.addConstr(z >= cp.quicksum(variables) - len(variables) + 1)
            selectors.append(z)
            return z

        def append_case(coefficients, selector, entity_bounds, record=None):
            if isinstance(selector, (int, float)) and selector == 0:
                return
            for index, coefficient in coefficients.items():
                terms[index].append(float(coefficient) * selector)
                low, high = entity_bounds.setdefault(index, [0.0, 0.0])
                entity_bounds[index] = [min(low, coefficient), max(high, coefficient)]
            case_records.append((selector, coefficients, record))

        def finish_entity(entity_bounds):
            for index, (low, high) in entity_bounds.items():
                bounds[index][0] += low
                bounds[index][1] += high

        for key, network in window.networks.items():
            user = window.state.users.get(key)
            if user is None:
                raise ValueError("full terminal features require complete user state")
            user_requests = [r for r in window.requests if r.user_key == user.user_key]
            request_by_arc = {tuple(r.arc): r for r in user_requests if r.arc is not None}
            waiting = window.state.waiting.get(user.waiting_request_id)
            waiting_request = next((r for r in user_requests if waiting and r.request_id == waiting.request_id), None)
            no_failure = 1 - cp.quicksum(failures[r.request_id] for r in user_requests if r.request_id in failures)
            entity_bounds = {}
            entity_selectors = []
            for path_index, path in enumerate(enumerate_paths(network.arcs, network.origin)):
                path_selector = and_selector([y[key, tuple(arc)] for arc in path], f"path:{key}:{path_index}")
                selected_requests = ([waiting_request] if waiting_request is not None else []) +                     [request_by_arc[tuple(arc)] for arc in path if isinstance(arc[1], int)]
                selected_stations = [r.station for r in selected_requests]
                if not selected_requests:
                    continue
                for next_index, req in enumerate(selected_requests):
                    rid = req.request_id
                    for arrival, predecessor_selector in arrival_cases[rid]:
                        # Each case means the selected path's earliest still-unserved request.
                        factors = [path_selector, 1 - total[rid], no_failure, predecessor_selector]
                        selector = and_selector(factors, f"{key}:{path_index}:{next_index}")
                        due = deadline(req, arrival)
                        if due < tau:
                            continue
                        from .forecast import forecast_motion
                        if next_index == 0 and req.observed:
                            motion = {"position_km": p.station.positions_km[req.station],
                                      "soc": req.return_soc, "entered": True,
                                      "last_swap_position_km": user.last_swap_position_km,
                                      "arrival_time": arrival}
                        elif next_index == 0:
                            motion = forecast_motion(p, user, req.station, tau,
                                                     now=window.ell * p.interval_hours)
                        else:
                            previous = selected_requests[next_index - 1]
                            departure_time = arrival - req.travel_time
                            motion = forecast_motion(p, user, req.station, tau,
                                                     departure_station=previous.station,
                                                     departure_time=departure_time,
                                                     now=window.ell * p.interval_hours)
                        # Request construction and motion prediction must share the same clock.
                        if not math.isclose(motion["arrival_time"], arrival, rel_tol=0, abs_tol=1e-7):
                            raise ValueError("terminal motion and candidate arrival disagree")
                        chain = selected_stations[next_index:]
                        completed = list(user.completed_stations) + selected_stations[:next_index]
                        published = selected_stations if motion["entered"] else None
                        coefficients = self._user_coefficients(
                            user, chain, req.return_soc, arrival, due, motion, tau, completed, published,
                            [later.return_soc for later in selected_requests[next_index + 1:]],
                            [later.travel_time for later in selected_requests[next_index + 1:]])
                        append_case(coefficients, selector, entity_bounds,
                                    {"user": key, "chain": chain, "arrival": arrival, "motion": motion})
                        entity_selectors.append(selector)
            if entity_selectors:
                model.addConstr(cp.quicksum(entity_selectors) <= 1, name=f"terminal_one_case[{key}]")
            finish_entity(entity_bounds)
        for req in window.requests:
            if req.kind != "random":
                continue
            arrival = req.arrival_time
            due = deadline(req, arrival)
            if due < tau:
                continue
            selector = 1 - total[req.request_id]
            coefficients = self._random_coefficients(req.station, req.return_soc, arrival, due, tau)
            entity_bounds = {}
            append_case(coefficients, selector, entity_bounds, {"random": req.request_id})
            finish_entity(entity_bounds)
        # Future fixed predictions beyond this solve have no within-window service.
        for req in _random_predictions(p, tau, p.num_periods * p.interval_hours, include_start=True):
            for index, value in self._random_coefficients(req["station"], req["return_soc"], req["arrival_time"],
                                                        req["deadline"], tau).items():
                expressions[index] += value
                bounds[index][0] += value
                bounds[index][1] += value
        for index, components in terms.items():
            expressions[index] += cp.quicksum(components)
        return expressions, bounds, {"cases": case_records, "selectors": selectors,
                                     "case_count": len(case_records), "feature_dimension": self.dimension}
