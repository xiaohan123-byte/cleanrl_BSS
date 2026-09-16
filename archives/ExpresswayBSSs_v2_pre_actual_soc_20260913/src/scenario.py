"""Reproducible synthetic truth with a separate, copy-isolated observation API."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .candidate_network import generate_candidate_network, get_feasible_arcs
from .parameters import BusinessParameters


@dataclass(frozen=True)
class ObservationView:
    now: float
    reservations: list[dict]
    random_history: list[dict]

    def to_dict(self) -> dict:
        return copy.deepcopy({"now": self.now, "reservations": self.reservations, "random_history": self.random_history})


@dataclass
class SyntheticScenario:
    """Simulation truth. Only observation_at() output is passed to predictors.

    Baseline reservations enter at their announced entry time and SOC. Actual
    downstream arrival times are produced by vehicle execution after service.
    """

    params: dict | BusinessParameters
    reservations: list[dict]
    actual_random_requests: list[dict]
    seed: int = 42

    def __post_init__(self) -> None:
        self.params = copy.deepcopy(self.params.to_dict() if isinstance(self.params, BusinessParameters) else self.params)
        self.reservations = copy.deepcopy(self.reservations)
        self.actual_random_requests = sorted(copy.deepcopy(self.actual_random_requests), key=lambda record: (record["arrival_time"], record["request_id"]))
        self.validate()

    def validate(self) -> None:
        params = BusinessParameters.from_dict(self.params)
        keys = set()
        for reservation in self.reservations:
            key = tuple(reservation["user_key"])
            if len(key) != 2 or key in keys or key[0] != reservation["od_id"]:
                raise ValueError("reservation user keys must be unique (od_id, user_id) pairs")
            keys.add(key)
            params.od_index(reservation["od_id"])
            if not math.isfinite(reservation["entry_time"]) or reservation["entry_time"] < 0:
                raise ValueError("reservation entry time must be finite and nonnegative")
            params.soc_bin(reservation["entry_soc"])
        ids = set()
        for request in self.actual_random_requests:
            if request["request_id"] in ids:
                raise ValueError("random request IDs must be unique")
            ids.add(request["request_id"])
            if request["station"] not in params.station.station_ids:
                raise ValueError("random request station is unknown")
            if not math.isfinite(request["arrival_time"]) or request["arrival_time"] < 0 or not math.isfinite(request["return_soc"]) or not 0 <= request["return_soc"] < 1:
                raise ValueError("random request time or SOC is invalid")

    def initial_reservations(self) -> list[dict]:
        # Whitelist fields so private fields in loaded records cannot leak.
        fields = ("user_key", "od_id", "entry_time", "entry_soc")
        return [{field: copy.deepcopy(record[field]) for field in fields} for record in self.reservations]

    def observation_at(self, now: float) -> ObservationView:
        if not math.isfinite(now) or now < 0:
            raise ValueError("observation time must be finite and nonnegative")
        reservations = self.initial_reservations()
        for record in reservations:
            if record["entry_time"] <= now:
                record["actual_entry_time"] = record["entry_time"]
                record["actual_entry_soc"] = record["entry_soc"]
        return ObservationView(float(now), reservations, copy.deepcopy([
            record for record in self.actual_random_requests if record["arrival_time"] <= now
        ]))

    def arrivals_between(self, start: float, end: float) -> list[dict]:
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError("arrival interval must have finite ordered endpoints")
        return copy.deepcopy([record for record in self.actual_random_requests if start <= record["arrival_time"] < end])

    def to_dict(self) -> dict:
        return copy.deepcopy({"schema_version": 1, "params": self.params, "seed": self.seed,
                              "reservations": self.reservations, "actual_random_requests": self.actual_random_requests})

    @classmethod
    def from_dict(cls, data: dict) -> "SyntheticScenario":
        if data.get("schema_version") != 1:
            raise ValueError("unsupported baseline scenario schema")
        return cls(data["params"], data["reservations"], data["actual_random_requests"], data["seed"])


def generate_synthetic_scenario(params: BusinessParameters, seed: int | None = None) -> SyntheticScenario:
    params.validate()
    scenario_seed = params.seed if seed is None else int(seed)
    # Reservation draws, actual random requests and forecasts use distinct streams.
    reservation_rng = np.random.default_rng(np.random.SeedSequence([scenario_seed, 101]))
    actual_rng = np.random.default_rng(np.random.SeedSequence([scenario_seed, 202]))
    network = generate_candidate_network(params)
    duration = params.num_periods * params.interval_hours
    reservations = []
    counts = {od.od_id: 0 for od in params.od_pairs}
    feasible_ods = [index for index in range(len(params.od_pairs))
                    if any(get_feasible_arcs(network, index, high) for _, high in params.soc_bins)]
    if params.num_reservations and not feasible_ods:
        raise ValueError("configured road has no reachable candidate path for any SOC bin")
    for _ in range(params.num_reservations):
        od_index = feasible_ods[int(reservation_rng.integers(len(feasible_ods)))]
        od = params.od_pairs[od_index]
        # Paper assumes every input reservation has a complete reachable path.
        # Reject unreachable samples; do not alter its specified network pruning.
        for attempt in range(10000):
            soc = float(reservation_rng.uniform(params.soc_bins[0][0], params.soc_bins[-1][1]))
            if get_feasible_arcs(network, od_index, soc):
                break
        else:
            raise ValueError(f"cannot sample a reachable reservation for OD {od.od_id}")
        counts[od.od_id] += 1
        reservations.append({"user_key": [od.od_id, counts[od.od_id] - 1], "od_id": od.od_id,
                             "entry_time": float(reservation_rng.uniform(0., min(params.reservation_entry_window_hours, duration))), "entry_soc": soc})
    reservations.sort(key=lambda record: (record["entry_time"], record["user_key"]))
    random_requests = []
    for station, rate in enumerate(params.random_arrival_rate_per_hour):
        count = int(actual_rng.poisson(rate * duration))
        times = sorted(float(value) for value in actual_rng.uniform(0., duration, count))
        for index, arrival in enumerate(times):
            random_requests.append({"request_id": f"random:{station}:{index}", "station": station,
                                    "arrival_time": arrival, "return_soc": float(actual_rng.uniform(.05, .4))})
    return SyntheticScenario(params, reservations, random_requests, scenario_seed)


def save_scenario(scenario: SyntheticScenario, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_scenario(path: str | Path) -> SyntheticScenario:
    return SyntheticScenario.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
