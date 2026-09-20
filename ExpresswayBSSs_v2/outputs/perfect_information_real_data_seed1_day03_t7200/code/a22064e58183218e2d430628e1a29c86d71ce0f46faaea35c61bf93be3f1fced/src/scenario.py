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

    The legacy baseline has exact entry reports. Terminal experiments keep
    private actual entry values and fixed physical-segment travel multipliers.
    Only entry values already observed are exposed by observation_at().
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
            if not math.isfinite(reservation["entry_soc"]) or not 0 <= reservation["entry_soc"] <= 1:
                raise ValueError("reservation entry SOC must be finite and in [0, 1]")
            actual_time = reservation.get("actual_entry_time", reservation["entry_time"])
            actual_soc = reservation.get("actual_entry_soc", reservation["entry_soc"])
            if not math.isfinite(actual_time) or actual_time < 0 or not math.isfinite(actual_soc) or not 0 <= actual_soc <= 1:
                raise ValueError("actual reservation entry values are invalid")
            if params.terminal_experiment:
                duration = params.num_periods * params.interval_hours
                low, high = params.reservation_entry_soc_range
                report_low, report_high = params.report_entry_soc_range
                if not 0 <= actual_time < duration or not 0 <= reservation["entry_time"] < duration:
                    raise ValueError("terminal experiment entries must stay within the operating day")
                if not low < actual_soc <= high or not report_low <= reservation["entry_soc"] <= report_high:
                    raise ValueError("terminal experiment actual or report SOC is outside its configured range")
                if abs(actual_time - reservation["entry_time"]) > params.entry_time_error_hours + params.time_epsilon:
                    raise ValueError("entry report time exceeds the configured error support")
                if abs(actual_soc - reservation["entry_soc"]) > params.entry_soc_error + params.time_epsilon:
                    raise ValueError("entry report SOC exceeds the configured error support")
                multipliers = reservation.get("segment_time_multipliers", [])
                error = params.travel_time_relative_error
                if len(multipliers) != len(params.physical_nodes()) - 1 or any(not math.isfinite(v) or not 1 - error <= v <= 1 + error for v in multipliers):
                    raise ValueError("private physical-segment multipliers are missing or invalid")
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
        truth = {tuple(record["user_key"]): record for record in self.reservations}
        for record in reservations:
            actual = truth[tuple(record["user_key"])]
            actual_time = actual.get("actual_entry_time", actual["entry_time"])
            if actual_time <= now:
                record["actual_entry_time"] = actual_time
                record["actual_entry_soc"] = actual.get("actual_entry_soc", actual["entry_soc"])
        return ObservationView(float(now), reservations, copy.deepcopy([
            record for record in self.actual_random_requests if record["arrival_time"] <= now
        ]))

    def arrivals_between(self, start: float, end: float) -> list[dict]:
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError("arrival interval must have finite ordered endpoints")
        return copy.deepcopy([record for record in self.actual_random_requests if start <= record["arrival_time"] < end])

    def to_dict(self) -> dict:
        return copy.deepcopy({"schema_version": 2, "params": self.params, "seed": self.seed,
                              "reservations": self.reservations, "actual_random_requests": self.actual_random_requests})

    @classmethod
    def from_dict(cls, data: dict) -> "SyntheticScenario":
        if data.get("schema_version") != 2:
            raise ValueError("unsupported baseline scenario schema")
        return cls(data["params"], data["reservations"], data["actual_random_requests"], data["seed"])


def generate_synthetic_scenario(params: BusinessParameters, seed: int | None = None) -> SyntheticScenario:
    params.validate()
    scenario_seed = params.seed if seed is None else int(seed)
    if params.terminal_experiment:
        return _generate_terminal_scenario(params, scenario_seed)
    # Reservation draws, actual random requests and forecasts use distinct streams.
    reservation_rng = np.random.default_rng(np.random.SeedSequence([scenario_seed, 101]))
    actual_rng = np.random.default_rng(np.random.SeedSequence([scenario_seed, 202]))
    network = generate_candidate_network(params)
    duration = params.num_periods * params.interval_hours
    reservations = []
    counts = {od.od_id: 0 for od in params.od_pairs}
    low, high = params.reservation_entry_soc_range
    feasible_ods = [index for index in range(len(params.od_pairs))
                    if get_feasible_arcs(network, index, high)]
    if params.num_reservations and not feasible_ods:
        raise ValueError("configured road has no reachable candidate path within reservation_entry_soc_range")
    for _ in range(params.num_reservations):
        od_index = feasible_ods[int(reservation_rng.integers(len(feasible_ods)))]
        od = params.od_pairs[od_index]
        # Paper assumes every input reservation has a complete reachable path.
        # Reject unreachable samples; do not alter its specified network pruning.
        for attempt in range(10000):
            soc = float(reservation_rng.uniform(low, high))
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


def _generate_terminal_scenario(params: BusinessParameters, scenario_seed: int, *, day_id: int | None = None, include_random: bool = True) -> SyntheticScenario:
    """Generate the confirmed independent processes without feasibility filtering.

    Sampling a report uniformly from its intersected support is exactly the
    conditional distribution obtained by redrawing only an out-of-bounds noise.
    Actual OD and SOC draws are never rejected to make a route feasible.
    """
    def rng(stream: int):
        return np.random.default_rng(np.random.SeedSequence(
            [scenario_seed, stream] if day_id is None else [scenario_seed, stream, day_id]))

    od_rng, hour_rng, offset_rng, soc_rng = [rng(k) for k in (110, 111, 112, 113)]
    report_time_rng, report_soc_rng, travel_rng = [rng(k) for k in (120, 121, 122)]
    count_rng, arrival_rng, return_rng = [rng(k) for k in (210, 211, 212)]
    duration = params.num_periods * params.interval_hours
    od_weights = np.asarray(params.od_sampling_weights, dtype=float)
    od_weights /= od_weights.sum()
    hourly = np.asarray(params.reservation_hourly_weights, dtype=float)
    hourly /= hourly.sum()
    low, high = params.reservation_entry_soc_range
    report_low, report_high = params.report_entry_soc_range
    counts = {od.od_id: 0 for od in params.od_pairs}
    reservations = []
    for _ in range(params.num_reservations):
        od = params.od_pairs[int(od_rng.choice(len(params.od_pairs), p=od_weights))]
        hour = int(hour_rng.choice(len(hourly), p=hourly))
        actual_time = hour + float(offset_rng.uniform(0., min(1., duration - hour)))
        actual_soc = high - float(soc_rng.random()) * (high - low)
        time_low = max(0., actual_time - params.entry_time_error_hours)
        time_high = min(duration, actual_time + params.entry_time_error_hours)
        report_time = float(report_time_rng.uniform(time_low, time_high)) if time_high > time_low else actual_time
        report_time = min(report_time, float(np.nextafter(duration, 0.)))
        soc_low = max(report_low, actual_soc - params.entry_soc_error)
        soc_high = min(report_high, actual_soc + params.entry_soc_error)
        if soc_high < soc_low:
            raise ValueError("actual SOC has no report inside the noise and report bounds")
        report_soc = float(report_soc_rng.uniform(soc_low, soc_high)) if soc_high > soc_low else soc_low
        multipliers = travel_rng.uniform(1 - params.travel_time_relative_error,
                                         1 + params.travel_time_relative_error,
                                         len(params.physical_nodes()) - 1).tolist()
        user_id = counts[od.od_id]
        counts[od.od_id] += 1
        reservations.append({"user_key": [od.od_id, user_id], "od_id": od.od_id,
                             "entry_time": report_time, "entry_soc": report_soc,
                             "actual_entry_time": actual_time, "actual_entry_soc": actual_soc,
                             "segment_time_multipliers": multipliers})
    # The public order itself must not reveal private actual-entry ordering.
    reservations.sort(key=lambda record: (record["entry_time"], record["user_key"]))
    random_requests = []
    soc_low, soc_high = params.random_return_soc_range
    for station in (params.station.station_ids if include_random else []):
        index = 0
        for period in range(params.num_periods):
            start = period * params.interval_hours
            end = (period + 1) * params.interval_hours
            count = int(count_rng.poisson(params.random_rate_at(station, start + params.time_epsilon) * (end - start)))
            for arrival in sorted(float(v) for v in arrival_rng.uniform(start, end, count)):
                random_requests.append({"request_id": f"random:{station}:{index}", "station": station,
                                        "arrival_time": arrival,
                                        "return_soc": float(return_rng.uniform(soc_low, soc_high))})
                index += 1
    return SyntheticScenario(params, reservations, random_requests, scenario_seed)


def save_scenario(scenario: SyntheticScenario, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenario.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_scenario(path: str | Path) -> SyntheticScenario:
    return SyntheticScenario.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
