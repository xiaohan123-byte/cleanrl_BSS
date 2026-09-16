"""Business inputs for the five-minute, zero-terminal-value MPC baseline.

Time is measured in hours, power in kW, energy in kWh, and prices in yuan/kWh.
Demand rates are hourly rates and therefore do not change with grid resolution.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

ENTRY_NODE = "entry"
EXIT_NODE = "exit"
NodeId = Union[int, str]


@dataclass
class StationParameters:
    num_stations: int = 6
    station_ids: list[int] = field(default_factory=lambda: list(range(6)))
    positions_km: list[float] = field(default_factory=lambda: [80., 180., 280., 380., 480., 580.])
    num_slots: int = 5
    initial_slot_soc: list[list[float]] = field(default_factory=lambda: [[1., .9, .6, .4, .2] for _ in range(6)])
    charging_efficiency: float = .95
    slot_power_limits_kw: list[list[float]] = field(default_factory=lambda: [[60.] * 5 for _ in range(6)])
    station_power_limits_kw: list[float] = field(default_factory=lambda: [240.] * 6)


@dataclass
class ODPairParameters:
    od_id: int = 0
    entry_km: float = 0.
    exit_km: float = 430.
    station_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3])


@dataclass
class SolverParameters:
    threads: int = 1
    time_limit_sec: float = 30.
    output_flag: int = 0
    mip_gap: float = 0.
    feasibility_tol: float = 1e-8


def _default_ods() -> list[ODPairParameters]:
    return [ODPairParameters(), ODPairParameters(1, 0., 680., list(range(6)))]


@dataclass
class BusinessParameters:
    num_periods: int = 144
    interval_hours: float = 1 / 12
    horizon: int = 48
    path_update_interval: int = 3
    station: StationParameters = field(default_factory=StationParameters)
    od_pairs: list[ODPairParameters] = field(default_factory=_default_ods)
    vehicle_speed_kmh: float = 75.
    range_km: float = 300.
    battery_capacity_kwh: float = 100.
    soc_bins: list[list[float]] = field(default_factory=lambda: [[.30, .50], [.50, .75], [.75, 1.]])
    min_exit_soc: float = .10
    electricity_price: list[list[float]] = field(default_factory=list)
    swap_service_price: list[list[float]] = field(default_factory=list)
    path_adjustment_penalty: float = 1.
    reservation_failure_penalty: float = 1000.
    min_swap_spacing_km: float = 100.
    max_wait_hours: float = .25
    time_epsilon: float = 1e-9
    num_reservations: int = 6
    reservation_entry_window_hours: float = 2.
    random_arrival_rate_per_hour: list[float] = field(default_factory=list)
    seed: int = 42
    solver: SolverParameters = field(default_factory=SolverParameters)

    def __post_init__(self) -> None:
        hourly = [.35, .35, .65, 1.10, 1.10, .65, .40, .35, .35, .65, 1.10, .65]
        if not self.electricity_price:
            prices = [hourly[int(n * self.interval_hours + 1e-9) % len(hourly)] for n in range(self.num_periods)]
            self.electricity_price = [list(prices) for _ in self.station.station_ids]
        if not self.swap_service_price:
            self.swap_service_price = [[1.2] * self.num_periods for _ in self.station.station_ids]
        if not self.random_arrival_rate_per_hour:
            self.random_arrival_rate_per_hour = [.2] * self.station.num_stations

    def validate(self) -> None:
        for name in ("num_periods", "horizon", "path_update_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("interval_hours", "vehicle_speed_kmh", "range_km", "battery_capacity_kwh", "time_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_wait_hours", "min_swap_spacing_km", "path_adjustment_penalty", "reservation_failure_penalty", "reservation_entry_window_hours"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.min_exit_soc <= 1:
            raise ValueError("min_exit_soc must be in [0, 1]")
        if not isinstance(self.num_reservations, int) or self.num_reservations < 0:
            raise ValueError("num_reservations must be a nonnegative integer")
        st = self.station
        if st.num_stations <= 0 or st.num_slots <= 0:
            raise ValueError("station and slot counts must be positive")
        if st.station_ids != list(range(st.num_stations)):
            raise ValueError("station_ids must be contiguous indices starting at zero")
        if len(st.positions_km) != st.num_stations or any(not math.isfinite(x) for x in st.positions_km):
            raise ValueError("positions_km must contain one finite position per station")
        if any(a >= b for a, b in zip(st.positions_km, st.positions_km[1:])):
            raise ValueError("station positions must increase downstream")
        if not 0 < st.charging_efficiency <= 1:
            raise ValueError("charging_efficiency must be in (0, 1]")
        for name, lower, upper in (("initial_slot_soc", 0., 1.), ("slot_power_limits_kw", 0., math.inf)):
            matrix = getattr(st, name)
            if len(matrix) != st.num_stations or any(len(row) != st.num_slots for row in matrix):
                raise ValueError(f"{name} must have shape [station][slot]")
            if any(not math.isfinite(x) or not lower <= x <= upper for row in matrix for x in row):
                raise ValueError(f"invalid values in {name}")
        if len(st.station_power_limits_kw) != st.num_stations or any(not math.isfinite(x) or x < 0 for x in st.station_power_limits_kw):
            raise ValueError("station_power_limits_kw must contain one nonnegative finite limit per station")
        if not self.od_pairs or len({od.od_id for od in self.od_pairs}) != len(self.od_pairs):
            raise ValueError("OD pairs must be nonempty with unique od_id values")
        for od in self.od_pairs:
            if not math.isfinite(od.entry_km) or not math.isfinite(od.exit_km) or od.entry_km >= od.exit_km:
                raise ValueError("OD entry must be strictly before exit")
            if od.station_indices != sorted(set(od.station_indices)):
                raise ValueError("OD station_indices must be unique and downstream ordered")
            if any(i not in st.station_ids or not od.entry_km < st.positions_km[i] < od.exit_km for i in od.station_indices):
                raise ValueError("OD stations must lie strictly between entry and exit")
        if not self.soc_bins:
            raise ValueError("soc_bins cannot be empty")
        for h, bounds in enumerate(self.soc_bins):
            if len(bounds) != 2 or not 0 <= bounds[0] < bounds[1] <= 1:
                raise ValueError("SOC bins must be ordered intervals within [0, 1]")
            if h and abs(self.soc_bins[h - 1][1] - bounds[0]) > self.time_epsilon:
                raise ValueError("SOC bins must be contiguous")
        for name in ("electricity_price", "swap_service_price"):
            matrix = getattr(self, name)
            if len(matrix) != st.num_stations or any(len(row) < self.num_periods for row in matrix):
                raise ValueError(f"{name} must contain every station and operating period")
            if any(not math.isfinite(x) or x < 0 for row in matrix for x in row):
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(self.random_arrival_rate_per_hour) != st.num_stations or any(not math.isfinite(x) or x < 0 for x in self.random_arrival_rate_per_hour):
            raise ValueError("random_arrival_rate_per_hour needs one nonnegative finite hourly rate per station")
        sv = self.solver
        if sv.threads < 0 or sv.time_limit_sec <= 0 or not math.isfinite(sv.time_limit_sec) or sv.output_flag not in (0, 1):
            raise ValueError("invalid solver threads, time limit, or output flag")
        if not 0 <= sv.mip_gap <= 1 or not 1e-9 <= sv.feasibility_tol <= 1e-2:
            raise ValueError("invalid solver MIP gap or feasibility tolerance")

    def od_index(self, od_id: int) -> int:
        for index, od in enumerate(self.od_pairs):
            if od.od_id == od_id:
                return index
        raise ValueError(f"unknown od_id {od_id}")

    def od_nodes(self, od_index: int) -> list[NodeId]:
        return [ENTRY_NODE, *self.od_pairs[od_index].station_indices, EXIT_NODE]

    def node_position_km(self, od_index: int, node: NodeId) -> float:
        od = self.od_pairs[od_index]
        if node == ENTRY_NODE:
            return od.entry_km
        if node == EXIT_NODE:
            return od.exit_km
        if node not in od.station_indices:
            raise ValueError(f"station {node} is not in OD {od.od_id}")
        return self.station.positions_km[int(node)]

    def distance_km(self, od_index: int, origin: NodeId, destination: NodeId) -> float:
        distance = self.node_position_km(od_index, destination) - self.node_position_km(od_index, origin)
        if distance <= 0:
            raise ValueError("arcs must travel strictly downstream")
        return distance

    def soc_consumption(self, od_index: int, origin: NodeId, destination: NodeId) -> float:
        return self.distance_km(od_index, origin, destination) / self.range_km

    def travel_time_hours(self, od_index: int, origin: NodeId, destination: NodeId) -> float:
        return self.distance_km(od_index, origin, destination) / self.vehicle_speed_kmh

    def travel_periods(self, od_index: int, origin: NodeId, destination: NodeId) -> float:
        return self.travel_time_hours(od_index, origin, destination) / self.interval_hours

    def soc_bin(self, soc: float) -> int:
        for index, (lo, hi) in enumerate(self.soc_bins):
            if lo <= soc < hi or index == len(self.soc_bins) - 1 and lo <= soc <= hi:
                return index
        raise ValueError(f"entry_soc={soc} is outside the configured SOC bins")

    def slot_power_limit(self, station: int, slot: int) -> float:
        return self.station.slot_power_limits_kw[station][slot]

    def station_power_limit(self, station: int) -> float:
        return self.station.station_power_limits_kw[station]

    def electricity_price_at(self, station: int, period: int) -> float:
        return self.electricity_price[station][period]

    def swap_service_price_at(self, station: int, period: int) -> float:
        return self.swap_service_price[station][period]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BusinessParameters":
        payload = copy.deepcopy(data)
        if "station" in payload:
            payload["station"] = StationParameters(**payload["station"])
        if "od_pairs" in payload:
            payload["od_pairs"] = [ODPairParameters(**od) for od in payload["od_pairs"]]
        if "solver" in payload:
            payload["solver"] = SolverParameters(**payload["solver"])
        result = cls(**payload)
        result.validate()
        return result

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "BusinessParameters":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def get_default_parameters() -> BusinessParameters:
    params = BusinessParameters()
    params.validate()
    return params
