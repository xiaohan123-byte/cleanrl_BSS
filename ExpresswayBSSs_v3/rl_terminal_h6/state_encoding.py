"""Observable structured states shared by neural training and terminal MILPs.

No private scenario object is accepted here. Future random arrivals are in xi,
not duplicated as requests. Arrival/deadline fields of later reservation stops
are zero with time_known=0, exactly as in paper section 4.
"""
from __future__ import annotations
import math
import numpy as np
from src.forecast import predict_travel_time, predicted_entry_time
from src.parameters import price_at, slots_at


A_BASE = ["station_km", "predecessor_km", "travel_hours", "origin_km", "destination_km",
          "return_soc", "arrival_minus_now", "deadline_minus_now", "waiting", "time_known",
          "predecessor_known", "vehicle_km", "vehicle_soc", "entered", "last_swap_km", "published_known"]
R_FIELDS = ["station_km", "return_soc", "arrival_minus_now", "deadline_minus_now", "waiting"]


class StateSpec:
    variant = "learned_set"

    def __init__(self, params):
        self.params = params
        self.nstations = params.station.num_stations
        self.slot_counts = [slots_at(params, i) for i in range(self.nstations)]
        self.a_names = A_BASE + [f"{kind}:{i}" for kind in ("dayahead", "published", "retained")
                                for i in range(self.nstations)]
        self.r_names = R_FIELDS.copy()
        self.x_names = ["absolute_hours", "hours_until_demand_end"]
        self.x_names += [f"phase:{k}" for k in range(params.path_update_interval)]
        self.x_names += [f"current:{i}:{kind}" for i in range(self.nstations) for kind in ("buy", "sale")]
        self.x_names += [f"future:{h}:{i}:{kind}" for h in range(24) for i in range(self.nstations)
                         for kind in ("buy", "sale", "random_expected")]
        self.names = ([f"battery:{i}:{k}" for i in range(self.nstations) for k in range(4)] +
                      [f"reservation:{k}" for k in range(8)] + [f"random:{k}" for k in range(4)] + self.x_names)
        self.dimension = len(self.names)
        self._external_cache = {}

    def schema(self):
        return {"version": 1, "variant": self.variant, "a_names": self.a_names, "r_names": self.r_names,
                "x_names": self.x_names, "names": self.names, "slot_counts": self.slot_counts,
                "aggregation": "station battery sum; request sum then user map minus user map(0); random sum",
                "random_requests": "observed waiting only; terminal retains within-window predictions once",
                "future_forecasts": "24 one-hour windows; original historical means; zero after demand end"}

    def external(self, period):
        if period in self._external_cache:
            return self._external_cache[period].copy()
        p = self.params
        tau = period * p.interval_hours
        end = p.num_periods * p.interval_hours
        out = [tau, max(0., end - tau)]
        out += [float(period % p.path_update_interval == k) for k in range(p.path_update_interval)]
        for i in range(self.nstations):
            out.extend([price_at(p, "electricity_price", i, period), price_at(p, "swap_service_price", i, period)])
        for h in range(24):
            left, right = tau + h, tau + h + 1.
            for i in range(self.nstations):
                buy = sale = demand = 0.
                for n in range(int(math.floor(left / p.interval_hours)), int(math.ceil(right / p.interval_hours))):
                    overlap = max(0., min(right, (n+1)*p.interval_hours) - max(left, n*p.interval_hours))
                    buy += overlap * price_at(p, "electricity_price", i, n)
                    sale += overlap * price_at(p, "swap_service_price", i, n)
                for hour in range(int(math.floor(left)), int(math.ceil(right))):
                    overlap = max(0., min(right, hour+1., end) - max(left, float(hour)))
                    if overlap:
                        demand += overlap * p.random_rate_at(i, float(hour))
                out.extend([buy, sale, demand])
        self._external_cache[period] = out
        return out.copy()

    def reservation_rows(self, user, chain, rho, arrival, due, motion, tau, *,
                         completed=None, published=None, later_rho=None, later_travel=None):
        if not chain:
            return []
        p = self.params
        completed = list(user.completed_stations if completed is None else completed)
        od = p.od_pairs[p.od_index(user.user_key[0])]
        # Canonical remaining paths include the current waiting stop once.
        published = None if published is None else [i for i in published if i not in completed]
        if published is not None and arrival <= tau <= due and chain[0] not in published:
            published = [chain[0], *published]
        paths = [[float(i in route) for i in range(self.nstations)]
                 for route in (user.day_ahead, published or [], chain)]
        path_bits = sum(paths, [])
        rows = []
        for j, station in enumerate(chain):
            predecessor = chain[j-1] if j else (completed[-1] if completed else None)
            travel = 0. if predecessor is None else predict_travel_time(
                p, user, p.station.positions_km[predecessor], p.station.positions_km[station])
            if j and later_travel is not None:
                travel = later_travel[j-1]
            returned = rho if j == 0 else (later_rho[j-1] if later_rho is not None else
                1 - abs(p.station.positions_km[station] - p.station.positions_km[predecessor]) / p.range_km)
            row = [p.station.positions_km[station], 0. if predecessor is None else p.station.positions_km[predecessor],
                   travel, od.entry_km, od.exit_km, returned,
                   arrival-tau if j == 0 else 0., due-tau if j == 0 else 0.,
                   float(j == 0 and arrival <= tau <= due), float(j == 0), float(predecessor is not None),
                   motion["position_km"], motion["soc"], float(motion["entered"]),
                   motion["last_swap_position_km"], float(published is not None), *path_bits]
            rows.append(row)
        return rows

    def random_row(self, station, rho, arrival, due, tau):
        return [self.params.station.positions_km[station], rho, arrival-tau, due-tau,
                float(arrival <= tau <= due)]

    def encode_window(self, window):
        return self.raw_state(window.state)

    def raw_state(self, state):
        p = self.params
        tau = state.period * p.interval_hours
        users = []
        for key, user in sorted(state.users.items()):
            if user.status != "active":
                continue
            waiting = state.waiting.get(user.waiting_request_id)
            plan = user.published_plan if user.entered else user.retained_plan
            direction = p.od_direction(p.od_index(user.user_key[0]))
            chain = [i for i in (plan or []) if i not in user.completed_stations and
                     direction*(p.station.positions_km[i]-user.position_km) >= -1e-9 and
                     (waiting is None or i != waiting.station)]
            if waiting:
                chain.insert(0, waiting.station)
            if not chain:
                continue
            if waiting:
                rho, arrival, due = waiting.return_soc, waiting.arrival_time, waiting.deadline
            else:
                start = tau if user.entered else predicted_entry_time(p, user, tau)
                arrival = start + predict_travel_time(p, user, user.position_km, p.station.positions_km[chain[0]])
                rho = (user.soc if user.entered else user.entry_soc) - abs(p.station.positions_km[chain[0]]-user.position_km)/p.range_km
                due = arrival + p.max_wait_hours
            motion = {"position_km": user.position_km, "soc": user.soc if user.entered else user.entry_soc,
                      "entered": user.entered, "last_swap_position_km": user.last_swap_position_km}
            users.append(self.reservation_rows(user, chain, rho, arrival, due, motion, tau,
                                              published=user.published_plan))
        random = [self.random_row(r.station, r.return_soc, r.arrival_time, r.deadline, tau)
                  for _, r in sorted(state.waiting.items()) if r.kind == "random"]
        return {"schema_version": 1, "period": state.period,
                "batteries": [list(row) for row in state.slot_soc], "users": users, "random": random,
                "external": self.external(state.period),
                "continuation": float(tau < p.num_periods*p.interval_hours or bool(users) or bool(random))}


class Normalizer:
    """Only initial training-batch observations fit these affine transforms."""
    def __init__(self, record):
        self.record = record

    @classmethod
    def fit(cls, states, spec):
        groups = {"battery": [[v] for s in states for row in s["batteries"] for v in row],
                  "a": [r for s in states for u in s["users"] for r in u],
                  "r": [r for s in states for r in s["random"]],
                  "x": [s["external"] for s in states]}
        sizes = {"battery": 1, "a": len(spec.a_names), "r": len(spec.r_names), "x": len(spec.x_names)}
        binary = {"battery": [], "a": [8, 9, 10, 13, 15, *range(16, len(spec.a_names))],
                  "r": [4], "x": list(range(2, 2+spec.params.path_update_interval))}
        record = {"version": 1, "fit_source": "initial_training_batch_only", "sample_count": len(states)}
        for name, values in groups.items():
            a = np.asarray(values, dtype=float).reshape(-1, sizes[name])
            means, scales = [], []
            for col in range(sizes[name]):
                column = a[:, col]
                if name == "a" and col in (1, 2):
                    column = column[a[:, 10] > .5]
                if name == "a" and col in (6, 7):
                    column = column[a[:, 9] > .5]
                means.append(float(column.mean()) if len(column) else 0.)
                sd = float(column.std()) if len(column) else 0.
                scales.append(sd if sd > 1e-6 else 1.)
            for col in binary[name]:
                means[col], scales[col] = 0., 1.
            record[name] = {"mean": means, "scale": scales}
        return cls(record)

    def transform(self, name, values):
        mean = np.asarray(self.record[name]["mean"])
        scale = np.asarray(self.record[name]["scale"])
        x = np.asarray(values, dtype=float).reshape(-1, len(mean))
        result = (x-mean)/scale
        if name == "a" and len(x):
            result[:, [1, 2]] *= x[:, 10:11]
            result[:, [6, 7]] *= x[:, 9:10]
        return result
