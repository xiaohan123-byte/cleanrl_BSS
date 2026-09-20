"""Jointly trainable set encoders and an exact, bounded MILP value graph."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from state_encoding import StateSpec, Normalizer
from src.parameters import slots_at
from src.candidate_network import enumerate_paths
from src.forecast import forecast_motion, deterministic_random_requests
from src.terminal_value import bounded_relu


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def physical_record(params):
    values = params.to_dict()
    for field in ("horizon", "solver", "seed", "source_metadata", "random_hourly_means", "random_arrival_rate_per_hour"):
        values.pop(field, None)
    return values


def make_contract(params, forecasts):
    meta = params.source_metadata
    return {"physical_hash": digest(physical_record(params)),
            "dataset_version": meta.get("dataset_version"), "sources": meta.get("sources", {}),
            "forecast_hashes": sorted({digest(f) for f in forecasts})}


class LearnedValue:
    kind = "learned_set_relu"
    variant = "learned_set"
    configuration_fingerprint = None  # validated below using a multi-weekday contract

    def __init__(self, record):
        if record.get("schema_version") != 2 or record.get("kind") != self.kind:
            raise ValueError("unsupported learned encoder/value schema")
        self.record = record
        self.normalizer = Normalizer(record["normalizer"])
        self.weights = {k: np.asarray(v, dtype=np.float64) for k, v in record["weights"].items()}
        if any(not np.isfinite(v).all() for v in self.weights.values()):
            raise ValueError("nonfinite learned weight")
        self.names = record["state_schema"]["names"]
        self.feature_names = self.names + ["continuation"]
        self.output_scale = 1000.
        self.hidden_units = 16

    @property
    def parameter_count(self):
        return sum(a.size for a in self.weights.values())

    def validate_params(self, params):
        c = self.record["contract"]
        if digest(physical_record(params)) != c["physical_hash"]:
            raise ValueError("learned value physical configuration differs")
        if params.source_metadata.get("dataset_version") != c["dataset_version"] or params.source_metadata.get("sources", {}) != c["sources"]:
            raise ValueError("learned value data sources differ")
        if digest(params.random_hourly_means) not in c["forecast_hashes"]:
            raise ValueError("forecast profile is not in the frozen historical-weekday contract")
        if StateSpec(params).schema() != self.record["state_schema"]:
            raise ValueError("learned state schema differs")

    def to_dict(self):
        return self.record

    def save(self, path):
        from bootstrap import enable_atomic_retries
        enable_atomic_retries()(path, self.record)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def layer(self, name, x, relu=True):
        result = np.asarray(x) @ self.weights[name+".weight"].T
        if name+".bias" in self.weights:
            result = result + self.weights[name+".bias"]
        return np.maximum(result, 0.) if relu else result

    def encode_user(self, rows):
        if not rows:
            return np.zeros(8)
        requests = self.layer("request", self.normalizer.transform("a", rows)).sum(axis=0)
        return self.layer("user", requests) - self.layer("user", np.zeros(8))

    def encode_random(self, rows):
        if not rows:
            return np.zeros(4)
        return self.layer("random", self.normalizer.transform("r", rows)).sum(axis=0)

    def encode(self, state):
        parts = [self.layer("battery", self.normalizer.transform("battery", row)).sum(axis=0)
                 for row in state["batteries"]]
        parts.append(sum((self.encode_user(u) for u in state["users"]), np.zeros(8)))
        parts.append(self.encode_random(state["random"]))
        parts.append(self.normalizer.transform("x", [state["external"]])[0])
        return np.concatenate(parts)

    def predict(self, features):
        if isinstance(features, dict):
            return self.predict([*self.encode(features), features["continuation"]])
        x = np.asarray(features, dtype=float)
        if x.shape[-1] != len(self.names)+1:
            raise ValueError("learned value input dimension mismatch")
        phi, continuation = x[..., :-1], x[..., -1]
        output = self.layer("skip", phi, False)[..., 0] + self.layer("output", self.layer("hidden", phi), False)[..., 0]
        result = self.output_scale * output * continuation
        return float(result) if np.ndim(result) == 0 else result

    def build_mip(self, model, cp, COPT, window, soc, y, total, arrival_cases, service, failures, deadline, spec):
        p = spec.params
        period = window.ell + window.horizon
        tau = period * p.interval_hours
        expressions, bounds = [], []
        normal = self.normalizer.record["battery"]
        for i in range(spec.nstations):
            for k in range(4):
                a = float(self.weights["battery.weight"][k, 0] / normal["scale"][0])
                b = float(self.weights["battery.bias"][k] - a*normal["mean"][0])
                lo, hi = min(b, b+a), max(b, b+a)
                pieces = [bounded_relu(model, cp, COPT, a*soc[i, slot, period]+b, lo, hi,
                                       f"learned_battery[{i},{slot},{k}]") for slot in range(slots_at(p, i))]
                expressions.append(cp.quicksum(pieces))
                bounds.append([len(pieces)*max(0., lo), len(pieces)*max(0., hi)])
        user_terms = [[] for _ in range(8)]
        user_bounds = np.zeros((8, 2))
        random_terms = [[] for _ in range(4)]
        random_bounds = np.zeros((4, 2))
        active = []
        raw_cases = []
        random_cases = []
        counter = 0

        def conjunction(factors, label):
            nonlocal counter
            if any(isinstance(v, (int, float)) and v < .5 for v in factors):
                return 0.
            variables = [v for v in factors if not isinstance(v, (int, float))]
            if not variables:
                return 1.
            if len(variables) == 1:
                return variables[0]
            counter += 1
            z = model.addVar(vtype=COPT.BINARY, name=f"learned_case[{label},{counter}]")
            for v in variables:
                model.addConstr(z <= v)
            model.addConstr(z >= cp.quicksum(variables)-len(variables)+1)
            return z

        for key, network in window.networks.items():
            user = window.state.users[key]
            requests = [r for r in window.requests if r.user_key == user.user_key]
            by_arc = {tuple(r.arc): r for r in requests if r.arc is not None}
            waiting = window.state.waiting.get(user.waiting_request_id)
            waiting_req = next((r for r in requests if waiting and r.request_id == waiting.request_id), None)
            no_failure = 1-cp.quicksum(failures[r.request_id] for r in requests if r.request_id in failures)
            entity_selectors, entity_values = [], [np.zeros(8)]
            for path_index, path in enumerate(enumerate_paths(network.arcs, network.origin)):
                path_selector = conjunction([y[key, tuple(arc)] for arc in path], f"path:{key}:{path_index}")
                chosen = ([waiting_req] if waiting_req is not None else []) + [by_arc[tuple(arc)] for arc in path if isinstance(arc[1], int)]
                stations = [r.station for r in chosen]
                for j, req in enumerate(chosen):
                    for arrival, predecessor in arrival_cases[req.request_id]:
                        due = deadline(req, arrival)
                        if due < tau:
                            continue
                        selector = conjunction([path_selector, 1-total[req.request_id], no_failure, predecessor], key)
                        if isinstance(selector, (float, int)) and not selector:
                            continue
                        if j == 0 and req.observed:
                            motion = {"position_km": p.station.positions_km[req.station], "soc": req.return_soc,
                                      "entered": True, "last_swap_position_km": user.last_swap_position_km,
                                      "arrival_time": arrival}
                        elif j == 0:
                            motion = forecast_motion(p, user, req.station, tau, now=window.ell*p.interval_hours)
                        else:
                            motion = forecast_motion(p, user, req.station, tau,
                                departure_station=chosen[j-1].station, departure_time=arrival-req.travel_time,
                                now=window.ell*p.interval_hours)
                        if abs(motion["arrival_time"]-arrival) > 1e-7:
                            raise ValueError("learned terminal motion and request clocks differ")
                        rows = spec.reservation_rows(user, stations[j:], req.return_soc, arrival, due, motion, tau,
                            completed=[*user.completed_stations, *stations[:j]],
                            published=stations if motion["entered"] else None,
                            later_rho=[r.return_soc for r in chosen[j+1:]],
                            later_travel=[r.travel_time for r in chosen[j+1:]])
                        values = self.encode_user(rows)
                        for k, v in enumerate(values):
                            if v:
                                user_terms[k].append(float(v)*selector)
                        entity_values.append(values)
                        entity_selectors.append(selector)
                        raw_cases.append((selector, rows))
            if entity_selectors:
                model.addConstr(cp.quicksum(entity_selectors) <= 1, name=f"learned_one_user_case[{key}]")
                active.extend(entity_selectors)
            candidates = np.asarray(entity_values)
            user_bounds[:, 0] += candidates.min(0)
            user_bounds[:, 1] += candidates.max(0)
        for req in window.requests:
            if req.kind != "random":
                continue
            due = deadline(req, req.arrival_time)
            if due < tau:
                continue
            row = spec.random_row(req.station, req.return_soc, req.arrival_time, due, tau)
            selector = 1-total[req.request_id]
            values = self.encode_random([row])
            for k, v in enumerate(values):
                if v:
                    random_terms[k].append(float(v)*selector)
            random_bounds[:, 0] += np.minimum(0., values)
            random_bounds[:, 1] += np.maximum(0., values)
            active.append(selector)
            random_cases.append((selector, row))
        # The terminal state is before service at tau but after admitting arrivals
        # at tau. These point arrivals are excluded from the half-open MPC window.
        for req in deterministic_random_requests(p, math.nextafter(tau, -math.inf), math.nextafter(tau, math.inf)):
            row = spec.random_row(req["station"], req["return_soc"], tau, tau+p.max_wait_hours, tau)
            values = self.encode_random([row])
            for k, v in enumerate(values):
                random_terms[k].append(float(v))
            random_bounds += values[:, None]
            active.append(1.)
            random_cases.append((1., row))
        expressions += [cp.quicksum(v) for v in user_terms+random_terms]
        bounds += user_bounds.tolist()+random_bounds.tolist()
        external = spec.external(period)
        xi = self.normalizer.transform("x", [external])[0]
        expressions += xi.tolist()
        bounds += [[float(v), float(v)] for v in xi]
        if len(expressions) != len(self.names):
            raise ValueError("MILP learned feature dimension differs")
        # Constants are collapsed before passing large affine rows to the solver.
        def affine(weights, bias):
            constant = float(bias)
            terms = []
            lo = hi = float(bias)
            for w, expression, (low, high) in zip(weights, expressions, bounds):
                w = float(w)
                lo += w*(low if w >= 0 else high)
                hi += w*(high if w >= 0 else low)
                if isinstance(expression, (float, int)):
                    constant += w*expression
                elif w:
                    terms.append(w*expression)
            return constant+cp.quicksum(terms), lo, hi
        value, low, high = affine(self.weights["skip.weight"][0], self.weights["skip.bias"][0])
        for k in range(16):
            z, lo, hi = affine(self.weights["hidden.weight"][k], self.weights["hidden.bias"][k])
            hidden = bounded_relu(model, cp, COPT, z, lo, hi, f"learned_head[{k}]")
            a = float(self.weights["output.weight"][0, k])
            value += a*hidden
            low += a*(max(0., lo) if a >= 0 else max(0., hi))
            high += a*(max(0., hi) if a >= 0 else max(0., lo))
        if tau < p.num_periods*p.interval_hours:
            continuation = 1.
        elif not active:
            continuation = 0.
            value = 0.
        else:
            continuation = model.addVar(vtype=COPT.BINARY, name="learned_continuation")
            for a in active:
                model.addConstr(continuation >= a)
            model.addConstr(continuation <= cp.quicksum(active))
            gated = model.addVar(lb=min(0., low), ub=max(0., high), name="learned_absorbing_value")
            model.addConstr(gated >= low*continuation)
            model.addConstr(gated <= high*continuation)
            model.addConstr(gated >= value-high*(1-continuation))
            model.addConstr(gated <= value-low*(1-continuation))
            value = gated
        expressions.append(continuation)
        bounds.append([0., 1.])
        records = {"case_count": len(raw_cases)+len(random_cases), "raw_cases": raw_cases,
                   "random_cases": random_cases, "external": external, "period": period, "soc": soc,
                   "slot_counts": spec.slot_counts, "continuation": continuation}
        return 1000.*value, expressions, bounds, records

    def audit_mip(self, records, get_value, features):
        raw = {"batteries": [[get_value(records["soc"][i,b,records["period"]]) for b in range(count)]
                              for i,count in enumerate(records["slot_counts"])],
               "users": [rows for selector, rows in records["raw_cases"] if get_value(selector) > .5],
               "random": [row for selector, row in records["random_cases"] if get_value(selector) > .5],
               "external": records["external"], "continuation": get_value(records["continuation"])}
        encoded = np.r_[self.encode(raw), raw["continuation"]]
        error = float(np.max(np.abs(encoded-np.asarray(features))))
        if error > 2e-5:
            raise ValueError(f"terminal learned state/MILP mismatch: {error}")
        return {"learned_state_encoding_error": error, "terminal_pending_users": len(raw["users"]),
                "terminal_pending_random": len(raw["random"]), "terminal_continuation": raw["continuation"]}
