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
    num_slots_by_station: list[int] = field(default_factory=list)


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
    reservation_entry_soc_range: list[float] = field(default_factory=lambda: [.3, 1.])
    random_arrival_rate_per_hour: list[float] = field(default_factory=list)
    seed: int = 42
    solver: SolverParameters = field(default_factory=SolverParameters)
    terminal_experiment: bool = False
    physical_node_positions_km: list[float] = field(default_factory=list)
    od_sampling_weights: list[float] = field(default_factory=list)
    reservation_hourly_weights: list[float] = field(default_factory=list)
    random_hourly_means: list[list[float]] = field(default_factory=list)
    entry_time_error_hours: float = 0.
    entry_soc_error: float = 0.
    report_entry_soc_range: list[float] = field(default_factory=lambda: [0., 1.])
    travel_time_relative_error: float = 0.
    random_return_soc_range: list[float] = field(default_factory=lambda: [.05, .4])
    random_soc_prediction: float = .225
    source_metadata: dict = field(default_factory=dict)
    finish_pending_after_demand: bool = False

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
        if st.num_slots_by_station and (len(st.num_slots_by_station) != st.num_stations or
                any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in st.num_slots_by_station)):
            raise ValueError("num_slots_by_station needs one positive integer per station")
        if not isinstance(self.finish_pending_after_demand, bool):
            raise ValueError("finish_pending_after_demand must be boolean")
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
            if len(matrix) != st.num_stations or any(len(row) != slots_at(self, i) for i, row in enumerate(matrix)):
                raise ValueError(f"{name} must have shape [station][slot]")
            if any(not math.isfinite(x) or not lower <= x <= upper for row in matrix for x in row):
                raise ValueError(f"invalid values in {name}")
        if len(st.station_power_limits_kw) != st.num_stations or any(not math.isfinite(x) or x < 0 for x in st.station_power_limits_kw):
            raise ValueError("station_power_limits_kw must contain one nonnegative finite limit per station")
        if not self.od_pairs or len({od.od_id for od in self.od_pairs}) != len(self.od_pairs):
            raise ValueError("OD pairs must be nonempty with unique od_id values")
        for od in self.od_pairs:
            if not math.isfinite(od.entry_km) or not math.isfinite(od.exit_km) or od.entry_km == od.exit_km:
                raise ValueError("OD entry and exit must be finite and distinct")
            direction = 1 if od.exit_km > od.entry_km else -1
            if od.station_indices != sorted(set(od.station_indices), reverse=direction < 0):
                raise ValueError("OD station_indices must be unique and ordered in the driving direction")
            if any(i not in st.station_ids or not min(od.entry_km, od.exit_km) < st.positions_km[i] < max(od.entry_km, od.exit_km) for i in od.station_indices):
                raise ValueError("OD stations must lie strictly between entry and exit")
        bounds = self.reservation_entry_soc_range
        try:
            valid_range = (len(bounds) == 2 and all(math.isfinite(x) for x in bounds)
                           and 0 <= bounds[0] < bounds[1] <= 1)
        except (TypeError, ValueError):
            valid_range = False
        if not valid_range:
            raise ValueError("reservation_entry_soc_range must contain finite bounds 0 <= lower < upper <= 1")
        for name in ("electricity_price", "swap_service_price"):
            matrix = getattr(self, name)
            if len(matrix) != st.num_stations or any(len(row) < self.num_periods for row in matrix):
                raise ValueError(f"{name} must contain every station and operating period")
            if any(not math.isfinite(x) or x < 0 for row in matrix for x in row):
                raise ValueError(f"{name} must be finite and nonnegative")
        if len(self.random_arrival_rate_per_hour) != st.num_stations or any(not math.isfinite(x) or x < 0 for x in self.random_arrival_rate_per_hour):
            raise ValueError("random_arrival_rate_per_hour needs one nonnegative finite hourly rate per station")
        if not isinstance(self.terminal_experiment, bool):
            raise ValueError("terminal_experiment must be boolean")
        for name in ("entry_time_error_hours", "entry_soc_error", "travel_time_relative_error"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.travel_time_relative_error >= 1:
            raise ValueError("travel_time_relative_error must be less than one")
        for name in ("report_entry_soc_range", "random_return_soc_range"):
            interval = getattr(self, name)
            if len(interval) != 2 or any(not math.isfinite(v) for v in interval) or not 0 <= interval[0] < interval[1] <= 1:
                raise ValueError(f"{name} must be increasing bounds in [0, 1]")
        if not math.isfinite(self.random_soc_prediction) or not 0 <= self.random_soc_prediction < 1:
            raise ValueError("random_soc_prediction must be in [0, 1)")
        if self.physical_node_positions_km:
            nodes = self.physical_node_positions_km
            if any(not math.isfinite(v) for v in nodes) or nodes != sorted(set(nodes)):
                raise ValueError("physical node positions must be finite, unique and increasing")
            required = set(st.positions_km) | {x for od in self.od_pairs for x in (od.entry_km, od.exit_km)}
            if not required.issubset(set(nodes)):
                raise ValueError("physical nodes must include every station and OD endpoint")
        for name, size in (("od_sampling_weights", len(self.od_pairs)), ("reservation_hourly_weights", math.ceil(self.num_periods * self.interval_hours))):
            weights = getattr(self, name)
            if weights and (len(weights) != size or any(not math.isfinite(v) or v < 0 for v in weights) or sum(weights) <= 0):
                raise ValueError(f"{name} has invalid length or weights")
        if self.random_hourly_means:
            hours = math.ceil(self.num_periods * self.interval_hours)
            if len(self.random_hourly_means) != st.num_stations or any(len(row) != hours for row in self.random_hourly_means):
                raise ValueError("random_hourly_means must have shape [station][hour]")
            if any(not math.isfinite(v) or v < 0 for row in self.random_hourly_means for v in row):
                raise ValueError("random_hourly_means must be finite and nonnegative")
        if self.terminal_experiment and (not self.od_sampling_weights or not self.reservation_hourly_weights or not self.random_hourly_means):
            raise ValueError("terminal experiment requires OD, entry-hour and random-hour weights")
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

    def od_direction(self, od_index: int) -> int:
        od = self.od_pairs[od_index]
        return 1 if od.exit_km > od.entry_km else -1

    def physical_nodes(self) -> list[float]:
        if self.physical_node_positions_km:
            return list(self.physical_node_positions_km)
        return sorted(set(self.station.positions_km) | {x for od in self.od_pairs for x in (od.entry_km, od.exit_km)})

    def random_rate_at(self, station: int, time_hours: float) -> float:
        if self.finish_pending_after_demand and not 0 <= time_hours < self.num_periods * self.interval_hours:
            return 0.
        if self.random_hourly_means:
            if not 0 <= time_hours < self.num_periods * self.interval_hours:
                return 0.
            return self.random_hourly_means[station][int(time_hours)]
        return self.random_arrival_rate_per_hour[station]

    def distance_km(self, od_index: int, origin: NodeId, destination: NodeId) -> float:
        distance = self.od_direction(od_index) * (self.node_position_km(od_index, destination) - self.node_position_km(od_index, origin))
        if distance <= 0:
            raise ValueError("arcs must travel strictly downstream")
        return distance

    def soc_consumption(self, od_index: int, origin: NodeId, destination: NodeId) -> float:
        return self.distance_km(od_index, origin, destination) / self.range_km

    def travel_time_hours(self, od_index: int, origin: NodeId, destination: NodeId) -> float:
        return self.distance_km(od_index, origin, destination) / self.vehicle_speed_kmh

    def travel_periods(self, od_index: int, origin: NodeId, destination: NodeId) -> float:
        return self.travel_time_hours(od_index, origin, destination) / self.interval_hours

    def slot_power_limit(self, station: int, slot: int) -> float:
        return self.station.slot_power_limits_kw[station][slot]

    def station_power_limit(self, station: int) -> float:
        return self.station.station_power_limits_kw[station]

    def electricity_price_at(self, station: int, period: int) -> float:
        return price_at(self, "electricity_price", station, period)

    def swap_service_price_at(self, station: int, period: int) -> float:
        return price_at(self, "swap_service_price", station, period)

    def to_dict(self) -> dict:
        payload = asdict(self)
        # Preserve historical serialized inputs and their fingerprints when the
        # new options are disabled; old scenario archives remain reproducible.
        if not self.finish_pending_after_demand:
            payload.pop("finish_pending_after_demand")
        if not self.station.num_slots_by_station:
            payload["station"].pop("num_slots_by_station")
        return payload

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


def slots_at(params, station: int) -> int:
    """Number of real slots; also supports legacy lightweight test parameters."""
    counts = getattr(params.station, "num_slots_by_station", [])
    return counts[station] if counts else params.station.num_slots


def price_at(params, name: str, station: int, period: int) -> float:
    if period < 0:
        raise ValueError("negative price period")
    if getattr(params, "finish_pending_after_demand", False):
        period %= params.num_periods
    return getattr(params, name)[station][period]


def execution_period_limit(params) -> int:
    """Conservative safety bound, not a truncated optimization horizon."""
    if not getattr(params, "finish_pending_after_demand", False):
        return params.num_periods
    travel = max(abs(od.exit_km - od.entry_km) for od in params.od_pairs) / params.vehicle_speed_kmh
    travel *= 1 + params.travel_time_relative_error
    waiting = max(len(od.station_indices) for od in params.od_pairs) * params.max_wait_hours
    return params.num_periods + math.ceil((travel + waiting) / params.interval_hours) + 1


def prediction_horizon(params, period: int, requested: int) -> int:
    return requested if getattr(params, "finish_pending_after_demand", False) else min(requested, params.num_periods - period)


def validate_complete_result(result: dict) -> None:
    """A completed demand day must include every pending service when enabled."""
    params = BusinessParameters.from_dict(result["parameter_snapshot"])
    count = len(result["rounds"])
    if not params.finish_pending_after_demand:
        if count != params.num_periods:
            raise ValueError("partial trajectory cannot be reported as a full-day experiment")
        return
    state = result["final_state"]
    if (not result.get("completed") or not params.num_periods <= count <= execution_period_limit(params)
            or state["period"] != count or state["waiting"]
            or any(u["status"] == "active" for u in state["users"].values())):
        raise ValueError("demand day has an incomplete cleanup trajectory")


def get_default_parameters() -> BusinessParameters:
    params = BusinessParameters()
    params.validate()
    return params
