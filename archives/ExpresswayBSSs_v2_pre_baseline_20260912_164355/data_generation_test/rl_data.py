# -*- coding: utf-8 -*-
"""可复现的六站模拟输入与确定性 Mock 信号。

本模块不训练或加载 RL 模型。它只提供：

* 带 seed 的完整 ``SyntheticScenario``（真实执行器/oracle 可持有）；
* 无未来真实信息的 ``ObservationView``（Mock、reference、MPC 可持有）；
* ``MockRLProvider``，以低 SOC 槽优先规则模拟请求功率和终端价值；
* 兼容既有调用的 ``generate_mock_data``、``RLSignals`` 和四参数
  ``get_signals(params, period_ell, horizon, soc_obs)``。

所有外部参数、价格、能量上限、预约、随机需求和车辆快照均写入同一份
synthetic 输入快照。当前 schema 为 2；它只适用于代码联调，不能用于论文
性能结论。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np

try:  # 作为包的一部分导入
    from data_generation_test.parameter import (
        BusinessParameters,
        get_default_parameters,
    )
except ImportError:  # 直接运行脚本
    from parameter import BusinessParameters, get_default_parameters

try:
    # 与连续事件执行器共享同一投影规则；本模块只负责把 H 步张量逐步适配
    # 为执行器需要的 [station][slot] 单区间动作。
    from src.event_engine import project_requested_power as _project_interval_power
except ImportError:  # pragma: no cover - 仅支持独立复制该模块时的降级导入
    _project_interval_power = None


JsonDict = Dict[str, Any]

SCHEMA_VERSION = 2
GENERATOR_VERSION = "six-station-synthetic-v2"
NUM_MOCK_RESERVATIONS = 6
REQUEST_PROBABILITY = 0.20
DEFAULT_MOCK_DATA_PATH = (
    Path(__file__).resolve().parent / "output" / "mock_rl_data.json"
)

__all__ = [
    "DEFAULT_MOCK_DATA_PATH",
    "GENERATOR_VERSION",
    "NUM_MOCK_RESERVATIONS",
    "SCHEMA_VERSION",
    "ObservationView",
    "SyntheticScenario",
    "RLSignals",
    "RLProvider",
    "MockRLProvider",
    "generate_mock_data",
    "generate_synthetic_scenario",
    "load_mock_data",
    "load_synthetic_scenario",
    "materialize_request",
    "project_requested_power",
    "save_mock_data",
]


def _json_copy(value: Any) -> Any:
    """拷贝只含 JSON 类型的结构，避免调用者修改场景内部真值。"""
    return copy.deepcopy(value)


def _rounded(value: float) -> float:
    """稳定序列化连续时刻，仍保留远小于时间区间的精度。"""
    return round(float(value), 9)


def _request_deadline(params: BusinessParameters, arrival_time: float) -> float:
    return _rounded(arrival_time + params.max_wait_hours)


def materialize_request(
    request: Mapping[str, Any], params: BusinessParameters
) -> JsonDict:
    """为事件执行器补齐规范请求字段，而不改变输入对象。

    schema 2 已持久化 ``deadline``，这里仍保留推导逻辑，以便旧输入和测试
    构造的最小 request 可接入同一执行接口。
    """
    out = dict(request)
    if "arrival_time" not in out:
        raise ValueError("request 缺少 arrival_time")
    if "arrival_soc" not in out and "return_soc" not in out:
        raise ValueError("request 缺少 arrival_soc 或 return_soc")
    out.setdefault("kind", "random")
    out.setdefault("return_soc", out.get("arrival_soc"))
    out.setdefault("deadline", _request_deadline(params, float(out["arrival_time"])))
    if float(out["deadline"]) < float(out["arrival_time"]):
        raise ValueError("request.deadline 不得早于 arrival_time")
    return out


def _make_random_request(
    rng: np.random.Generator,
    params: BusinessParameters,
    prefix: str,
    station: int,
    period: int,
    suffix: str,
) -> JsonDict:
    arrival_time = _rounded(
        (period + float(rng.random())) * params.interval_hours
    )
    arrival_soc = _rounded(float(rng.uniform(0.20, 0.90)))
    return materialize_request(
        {
            "request_id": f"{prefix}{station}_{period}_{suffix}",
            "station": station,
            "kind": "random",
            "arrival_time": arrival_time,
            "arrival_soc": arrival_soc,
        },
        params,
    )


def _generate_request_set(
    rng: np.random.Generator,
    params: BusinessParameters,
    prefix: str,
) -> List[List[List[JsonDict]]]:
    """生成 [station][period][request] 的 0--1 随机请求集。

    每个站全天至少有一个请求。预测、实际和日前预测分别从独立的 PCG64
    流调用此函数，因此实际流不从预测流推导。
    """
    all_requests: List[List[List[JsonDict]]] = []
    for station in range(params.station.num_stations):
        per_station: List[List[JsonDict]] = []
        has_request = False
        for period in range(params.num_periods):
            current: List[JsonDict] = []
            if float(rng.random()) < REQUEST_PROBABILITY:
                current.append(
                    _make_random_request(
                        rng, params, prefix, station, period, "0"
                    )
                )
                has_request = True
            per_station.append(current)
        if not has_request:
            anchor_period = int(rng.integers(0, params.num_periods))
            per_station[anchor_period].append(
                _make_random_request(
                    rng,
                    params,
                    prefix,
                    station,
                    anchor_period,
                    "coverage",
                )
            )
        for requests in per_station:
            requests.sort(key=lambda item: (item["arrival_time"], item["request_id"]))
        all_requests.append(per_station)
    return all_requests


def _od_by_id(params: BusinessParameters, od_id: int) -> Any:
    for od in params.od_pairs:
        if od.od_id == od_id:
            return od
    raise ValueError(f"未知 od_id={od_id}")


def _make_reservations(
    rng: np.random.Generator, params: BusinessParameters
) -> List[JsonDict]:
    """生成六个预约；第一个是覆盖最远 O-D 的固定锚点。"""
    reservations: List[JsonDict] = []
    for reservation_id in range(NUM_MOCK_RESERVATIONS):
        if reservation_id == 0:
            od_id = 1
            day_ahead_entry_time = 0.0
            day_ahead_entry_soc = 1.0
            actual_entry_time = 0.0
            actual_entry_soc = 1.0
        else:
            od = params.od_pairs[int(rng.integers(0, len(params.od_pairs)))]
            od_id = od.od_id
            day_ahead_entry_time = _rounded(float(rng.uniform(0.0, 2.0)))
            day_ahead_entry_soc = _rounded(float(rng.uniform(0.30, 1.0)))
            actual_entry_time = _rounded(
                min(
                    2.0,
                    max(0.0, day_ahead_entry_time + float(rng.uniform(-0.20, 0.20))),
                )
            )
            actual_entry_soc = _rounded(
                min(
                    1.0,
                    max(
                        params.soc_bins[0][0],
                        day_ahead_entry_soc + float(rng.uniform(-0.05, 0.05)),
                    ),
                )
            )
        reservations.append(
            {
                "reservation_id": reservation_id,
                "request_id": f"reservation_{reservation_id}",
                "submission_order": reservation_id,
                "kind": "reservation",
                "od_id": od_id,
                "user_key": [od_id, reservation_id],
                "path_order": 0,
                "day_ahead_entry_time": day_ahead_entry_time,
                "day_ahead_entry_soc": day_ahead_entry_soc,
                "actual_entry_time": actual_entry_time,
                "actual_entry_soc": actual_entry_soc,
                "return_soc": actual_entry_soc,
            }
        )
    return reservations


def _vehicle_trajectory(
    reservation: Mapping[str, Any], params: BusinessParameters
) -> List[JsonDict]:
    """生成外生的预约车辆位置/SOC/ETA 快照；不由优化结果反向更新。"""
    od = _od_by_id(params, int(reservation["od_id"]))
    entry_time = float(reservation["actual_entry_time"])
    entry_soc = float(reservation["actual_entry_soc"])
    result: List[JsonDict] = []
    for tick in range(params.num_periods + 1):
        now = _rounded(tick * params.interval_hours)
        elapsed = max(0.0, now - entry_time)
        position = min(od.exit_km, od.entry_km + params.vehicle_speed_kmh * elapsed)
        distance = max(0.0, position - od.entry_km)
        vehicle_soc = max(params.min_exit_soc, entry_soc - distance / params.range_km)
        result.append(
            {
                "time": now,
                "reservation_id": int(reservation["reservation_id"]),
                "od_id": int(reservation["od_id"]),
                "state": "enroute" if now >= entry_time and position < od.exit_km else (
                    "completed" if position >= od.exit_km else "future"
                ),
                "position_km": _rounded(position),
                "vehicle_soc": _rounded(vehicle_soc),
                "eta_to_exit_hours": _rounded(
                    max(0.0, (od.exit_km - position) / params.vehicle_speed_kmh)
                ),
            }
        )
    return result


@dataclass(frozen=True)
class ObservationView:
    """优化链可见的受限快照。

    没有 ``actual_random_requests``、完整车辆轨迹或 ``SyntheticScenario``
    引用。未来预测可以出现；未来真实到达和未来车辆轨迹不可以出现。
    """

    now: float
    metadata: JsonDict
    parameter_snapshot: JsonDict
    stations: List[JsonDict]
    reservations: List[JsonDict]
    revealed_reservation_entries: List[JsonDict]
    predicted_random_requests: List[List[List[JsonDict]]]
    actual_random_history: List[JsonDict]
    vehicle_snapshots: Dict[str, List[JsonDict]]

    def to_dict(self) -> JsonDict:
        return {
            "now": self.now,
            "metadata": _json_copy(self.metadata),
            "parameter_snapshot": _json_copy(self.parameter_snapshot),
            "stations": _json_copy(self.stations),
            "reservations": _json_copy(self.reservations),
            "revealed_reservation_entries": _json_copy(
                self.revealed_reservation_entries
            ),
            "predicted_random_requests": _json_copy(self.predicted_random_requests),
            "actual_random_history": _json_copy(self.actual_random_history),
            "vehicle_snapshots": _json_copy(self.vehicle_snapshots),
        }


@dataclass
class SyntheticScenario:
    """完整模拟真值容器，仅供真实执行器/oracle 持有。"""

    _payload: JsonDict

    def __post_init__(self) -> None:
        self._payload = _json_copy(self._payload)
        if int(self._payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                "mock 数据 schema 已过期；请使用 generate_mock_data(...)/--regenerate "
                f"重新生成 schema {SCHEMA_VERSION} 输入"
            )
        metadata = self._payload.get("metadata", {})
        if (
            metadata.get("data_source") != "synthetic"
            or metadata.get("signal_source") != "mock"
        ):
            raise ValueError("SyntheticScenario 只接受 synthetic/mock 输入")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SyntheticScenario":
        return cls(dict(payload))

    def to_dict(self) -> JsonDict:
        """返回完整真值副本；不要把该对象传给 Mock/reference/MPC。"""
        return _json_copy(self._payload)

    @property
    def metadata(self) -> JsonDict:
        return _json_copy(self._payload["metadata"])

    def observation_at(self, now: float) -> ObservationView:
        """构造时点 ``now`` 的受限可见视图。"""
        now = _rounded(now)
        if now < -1e-9:
            raise ValueError("now 不能小于 0")
        epsilon = float(self._payload["parameter_snapshot"].get("time_epsilon", 1e-9))

        visible_reservations: List[JsonDict] = []
        revealed_entries: List[JsonDict] = []
        for reservation in self._payload["reservations"]:
            planned = {
                key: value
                for key, value in reservation.items()
                if not key.startswith("actual_") and key != "return_soc"
            }
            visible_reservations.append(planned)
            if float(reservation["actual_entry_time"]) <= now + epsilon:
                revealed_entries.append(
                    {
                        "reservation_id": reservation["reservation_id"],
                        "request_id": reservation["request_id"],
                        "od_id": reservation["od_id"],
                        "user_key": _json_copy(reservation["user_key"]),
                        "arrival_time": reservation["actual_entry_time"],
                        "arrival_soc": reservation["actual_entry_soc"],
                        "return_soc": reservation["actual_entry_soc"],
                        "kind": "reservation",
                    }
                )

        actual_history: List[JsonDict] = []
        for per_station in self._payload["actual_random_requests"]:
            for per_period in per_station:
                actual_history.extend(
                    _json_copy(request)
                    for request in per_period
                    if float(request["arrival_time"]) <= now + epsilon
                )
        actual_history.sort(key=lambda item: (item["arrival_time"], item["request_id"]))

        snapshots: Dict[str, List[JsonDict]] = {}
        for reservation_id, trajectory in self._payload["vehicle_trajectories"].items():
            snapshots[str(reservation_id)] = [
                _json_copy(point)
                for point in trajectory
                if float(point["time"]) <= now + epsilon
                and point["state"] != "future"
            ]

        metadata = _json_copy(self._payload["metadata"])
        metadata["observation_time"] = now
        return ObservationView(
            now=now,
            metadata=metadata,
            parameter_snapshot=_json_copy(self._payload["parameter_snapshot"]),
            stations=_json_copy(self._payload["stations"]),
            reservations=visible_reservations,
            revealed_reservation_entries=revealed_entries,
            predicted_random_requests=_json_copy(
                self._payload["predicted_random_requests"]
            ),
            actual_random_history=actual_history,
            vehicle_snapshots=snapshots,
        )


def generate_synthetic_scenario(
    params: BusinessParameters, seed: Optional[int] = None
) -> SyntheticScenario:
    """用独立 PCG64 子流生成完整、可复现的六站模拟场景。"""
    params.validate()
    used_seed = params.seed if seed is None else int(seed)
    seed_sequence = np.random.SeedSequence(used_seed)
    reservation_ss, predicted_ss, actual_ss, forecast_ss = seed_sequence.spawn(4)

    reservations = _make_reservations(np.random.default_rng(reservation_ss), params)
    predicted = _generate_request_set(np.random.default_rng(predicted_ss), params, "P")
    actual = _generate_request_set(np.random.default_rng(actual_ss), params, "A")
    day_ahead = _generate_request_set(np.random.default_rng(forecast_ss), params, "D")
    trajectories = {
        str(item["reservation_id"]): _vehicle_trajectory(item, params)
        for item in reservations
    }

    station_snapshot = [
        {
            "station": i,
            "position_km": params.station.positions_km[i],
            "initial_slot_soc": list(params.station.initial_slot_soc[i]),
            "slot_power_limit_kw": params.station.slot_power_limit_kw,
            "charging_efficiency": params.station.charging_efficiency,
            "electricity_price": list(params.electricity_price[i]),
            "swap_service_price": list(params.swap_service_price[i]),
            "station_energy_limit_kwh": list(params.station_energy_limit_kwh[i]),
        }
        for i in range(params.station.num_stations)
    ]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": used_seed,
        "generator_version": GENERATOR_VERSION,
        "data_source": "synthetic",
        "signal_source": "mock",
        "rng": "numpy.SeedSequence(seed).spawn(4) + PCG64",
        "streams": ["reservations", "predicted", "actual", "day_ahead"],
    }
    payload: JsonDict = {
        "schema_version": SCHEMA_VERSION,
        "seed": used_seed,
        "generator_version": GENERATOR_VERSION,
        "data_source": "synthetic",
        "signal_source": "mock",
        "metadata": metadata,
        "parameter_snapshot": params.to_dict(),
        "stations": station_snapshot,
        "reservations": reservations,
        "predicted_random_requests": predicted,
        "actual_random_requests": actual,
        "day_ahead_random_forecast": day_ahead,
        "initial_slot_soc": _json_copy(params.station.initial_slot_soc),
        "vehicle_trajectories": trajectories,
    }
    return SyntheticScenario(payload)


def generate_mock_data(
    params: BusinessParameters, seed: Optional[int] = None
) -> JsonDict:
    """兼容旧 API：返回 schema 2 的完整 JSON 字典。"""
    return generate_synthetic_scenario(params, seed).to_dict()


def save_mock_data(
    data: Union[Mapping[str, Any], SyntheticScenario],
    path: Union[str, Path] = DEFAULT_MOCK_DATA_PATH,
) -> None:
    """保存完整模拟输入；写入前验证 schema/source 元数据。"""
    scenario = (
        data if isinstance(data, SyntheticScenario) else SyntheticScenario.from_dict(data)
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(scenario.to_dict(), handle, ensure_ascii=False, indent=2)


def load_mock_data(path: Union[str, Path] = DEFAULT_MOCK_DATA_PATH) -> JsonDict:
    """兼容旧 API：加载完整模拟字典，并拒绝旧 schema。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SyntheticScenario.from_dict(payload).to_dict()


def load_synthetic_scenario(
    path: Union[str, Path] = DEFAULT_MOCK_DATA_PATH,
) -> SyntheticScenario:
    return SyntheticScenario.from_dict(load_mock_data(path))


def project_requested_power(
    params: BusinessParameters,
    requested_power: Sequence[Sequence[Sequence[float]]],
    start_period: int = 0,
) -> List[List[List[float]]]:
    """执行统一动作投影：逐槽裁剪，再按站级区间能量同比缩放。"""
    station = params.station
    if len(requested_power) != station.num_stations:
        raise ValueError("requested_power 站点维度与参数不一致")
    if station.num_stations == 0 or station.num_slots == 0:
        return []
    horizon: Optional[int] = None
    result: List[List[List[float]]] = []
    for i, station_values in enumerate(requested_power):
        if len(station_values) != station.num_slots:
            raise ValueError(f"requested_power[{i}] 槽位维度与参数不一致")
        out_station: List[List[float]] = []
        for values in station_values:
            if horizon is None:
                horizon = len(values)
            elif len(values) != horizon:
                raise ValueError("requested_power 的预测时域长度不一致")
            out_station.append(
                [
                    min(station.slot_power_limit_kw, max(0.0, float(value)))
                    for value in values
                ]
            )
        result.append(out_station)
    if horizon is None:
        return result

    for h in range(horizon):
        period = start_period + h
        if not 0 <= period < params.num_periods:
            for i in range(station.num_stations):
                for b in range(station.num_slots):
                    result[i][b][h] = 0.0
            continue
        one_interval = [
            [result[i][b][h] for b in range(station.num_slots)]
            for i in range(station.num_stations)
        ]
        energy_limit = [
            params.station_energy_limit_at(i, period)
            for i in range(station.num_stations)
        ]
        if _project_interval_power is not None:
            projected = _project_interval_power(
                one_interval,
                slot_power_limit_kw=station.slot_power_limit_kw,
                station_energy_limit_kwh=energy_limit,
                interval_hours=params.interval_hours,
            ).power_kw
        else:  # pragma: no cover - see import fallback above
            projected = one_interval
            for i in range(station.num_stations):
                requested_kwh = params.interval_hours * sum(projected[i])
                if requested_kwh > energy_limit[i] and requested_kwh > 0.0:
                    scale = energy_limit[i] / requested_kwh
                    projected[i] = [value * scale for value in projected[i]]
        for i in range(station.num_stations):
            for b in range(station.num_slots):
                result[i][b][h] = projected[i][b]
    return result


@dataclass
class RLSignals:
    """本轮 MPC 所需的 Mock 信号；无训练模型或 checkpoint 状态。"""

    start_period: int
    horizon: int
    requested_power: List[List[List[float]]]
    terminal_soc_value: List[List[float]]
    outside_swap_lambda: List[float]
    signal_source: str = "mock"
    schema_version: int = SCHEMA_VERSION
    metadata: JsonDict = field(default_factory=dict)

    def outside_swap_value(self, station_index: int, rho: float) -> float:
        return self.outside_swap_lambda[station_index] * (rho - 1.0)

    def to_dict(self) -> JsonDict:
        return {
            "start_period": self.start_period,
            "horizon": self.horizon,
            "requested_power": _json_copy(self.requested_power),
            "terminal_soc_value": _json_copy(self.terminal_soc_value),
            "outside_swap_lambda": list(self.outside_swap_lambda),
            "signal_source": self.signal_source,
            "schema_version": self.schema_version,
            "metadata": _json_copy(self.metadata),
        }

    def validate(self, params: BusinessParameters) -> None:
        if self.signal_source != "mock":
            raise ValueError("本轮只允许 signal_source='mock'")
        if self.horizon <= 0 or self.start_period < 0:
            raise ValueError("RLSignals 的 start_period/horizon 非法")
        if len(self.requested_power) != params.station.num_stations:
            raise ValueError("RLSignals.requested_power 站点维度不匹配")
        if len(self.terminal_soc_value) != params.station.num_stations:
            raise ValueError("RLSignals.terminal_soc_value 站点维度不匹配")
        if len(self.outside_swap_lambda) != params.station.num_stations:
            raise ValueError("RLSignals.outside_swap_lambda 站点维度不匹配")
        for i in range(params.station.num_stations):
            if len(self.requested_power[i]) != params.station.num_slots:
                raise ValueError("RLSignals.requested_power 槽位维度不匹配")
            if len(self.terminal_soc_value[i]) != params.station.num_slots:
                raise ValueError("RLSignals.terminal_soc_value 槽位维度不匹配")
            for b in range(params.station.num_slots):
                if len(self.requested_power[i][b]) != self.horizon:
                    raise ValueError("RLSignals.requested_power 预测时域不匹配")
        projected = project_requested_power(
            params, self.requested_power, self.start_period
        )
        for i in range(params.station.num_stations):
            for b in range(params.station.num_slots):
                for h in range(self.horizon):
                    if not math.isclose(
                        self.requested_power[i][b][h],
                        projected[i][b][h],
                        abs_tol=1e-9,
                    ):
                        raise ValueError("RLSignals.requested_power 未经过统一动作投影")


@runtime_checkable
class RLProvider(Protocol):
    """终端信号协议；本轮唯一实现是 ``MockRLProvider``。"""

    def get_signals(
        self,
        params: BusinessParameters,
        period_ell: int,
        horizon: int,
        soc_obs: Sequence[Sequence[float]],
        *,
        observation: Optional[Union[ObservationView, Mapping[str, Any]]] = None,
        rolling_state: Optional[Any] = None,
    ) -> RLSignals:
        ...


_OBSERVATION_ALLOWED_KEYS = frozenset(
    {
        "now",
        "metadata",
        "parameter_snapshot",
        "stations",
        "reservations",
        "revealed_reservation_entries",
        "predicted_random_requests",
        "actual_random_history",
        "vehicle_snapshots",
    }
)
_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "scenario",
        "ground_truth",
        "actual_random_requests",
        "vehicle_trajectories",
        "mip_result",
        "mpc_result",
        "y",
        "alpha",
        "z",
        "pending",
    }
)


def _validate_observation(
    observation: Union[ObservationView, Mapping[str, Any]]
) -> JsonDict:
    if isinstance(observation, ObservationView):
        payload = observation.to_dict()
    elif isinstance(observation, Mapping):
        keys = set(observation)
        disallowed = keys - _OBSERVATION_ALLOWED_KEYS
        if disallowed:
            raise ValueError(
                "Mock observation 含未白名单字段：" + ", ".join(sorted(disallowed))
            )
        payload = _json_copy(dict(observation))
    else:
        raise TypeError("observation 必须是 ObservationView 或字段白名单 Mapping")

    def check(value: Any) -> None:
        if isinstance(value, Mapping):
            forbidden = _FORBIDDEN_CONTEXT_KEYS.intersection(value.keys())
            if forbidden:
                raise ValueError(
                    "Mock 输入禁止包含求解结果/未来真值："
                    + ", ".join(sorted(forbidden))
                )
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)

    check(payload)
    return payload


class MockRLProvider:
    """低 SOC 优先、无训练过程的确定性 Mock 信号提供者。"""

    def __init__(
        self, params: BusinessParameters, soc_value_coeff: Optional[float] = None
    ) -> None:
        params.validate()
        self.params = params
        self.soc_value_coeff = (
            params.battery_capacity_kwh
            if soc_value_coeff is None
            else float(soc_value_coeff)
        )

    def get_signals(
        self,
        params: BusinessParameters,
        period_ell: int,
        horizon: int,
        soc_obs: Sequence[Sequence[float]],
        *,
        observation: Optional[Union[ObservationView, Mapping[str, Any]]] = None,
        rolling_state: Optional[Any] = None,
    ) -> RLSignals:
        """保持旧四参数调用，并可接收受限 ObservationView。"""
        del rolling_state  # 明确不读取求解状态，也不允许回流本轮决策。
        params.validate()
        if observation is not None:
            _validate_observation(observation)
        if period_ell < 0 or horizon <= 0:
            raise ValueError("period_ell 必须非负且 horizon 必须为正")
        station = params.station
        if len(soc_obs) != station.num_stations:
            raise ValueError("soc_obs 站点维度与参数不一致")
        sim_soc: List[List[float]] = []
        for i, row in enumerate(soc_obs):
            if len(row) != station.num_slots:
                raise ValueError(f"soc_obs[{i}] 槽位维度与参数不一致")
            parsed = [float(value) for value in row]
            if any(value < 0.0 or value > 1.0 for value in parsed):
                raise ValueError("soc_obs 必须全部位于 [0, 1]")
            sim_soc.append(parsed)

        raw_power = [
            [[0.0 for _ in range(horizon)] for _ in range(station.num_slots)]
            for _ in range(station.num_stations)
        ]
        for h in range(horizon):
            period = period_ell + h
            if period >= params.num_periods:
                continue
            for i in range(station.num_stations):
                for b in sorted(
                    range(station.num_slots), key=lambda item: (sim_soc[i][item], item)
                ):
                    required_kw = (
                        params.battery_capacity_kwh
                        * (1.0 - sim_soc[i][b])
                        / (station.charging_efficiency * params.interval_hours)
                    )
                    raw_power[i][b][h] = min(
                        station.slot_power_limit_kw, max(0.0, required_kw)
                    )
            projected = project_requested_power(params, raw_power, period_ell)
            for i in range(station.num_stations):
                for b in range(station.num_slots):
                    sim_soc[i][b] = min(
                        1.0,
                        sim_soc[i][b]
                        + projected[i][b][h]
                        * params.interval_hours
                        * station.charging_efficiency
                        / params.battery_capacity_kwh,
                    )

        requested_power = project_requested_power(params, raw_power, period_ell)
        terminal_soc_value: List[List[float]] = []
        outside_swap_lambda: List[float] = []
        for i in range(station.num_stations):
            lo = min(period_ell, params.num_periods - 1)
            hi = min(period_ell + horizon, params.num_periods)
            prices = params.electricity_price[i][lo:hi] or [
                params.electricity_price[i][-1]
            ]
            lam = self.soc_value_coeff * sum(prices) / len(prices)
            terminal_soc_value.append([lam] * station.num_slots)
            outside_swap_lambda.append(lam)

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "seed": params.seed,
            "generator_version": GENERATOR_VERSION,
            "data_source": "synthetic",
            "signal_source": "mock",
        }
        signals = RLSignals(
            start_period=period_ell,
            horizon=horizon,
            requested_power=requested_power,
            terminal_soc_value=terminal_soc_value,
            outside_swap_lambda=outside_swap_lambda,
            signal_source="mock",
            schema_version=SCHEMA_VERSION,
            metadata=metadata,
        )
        signals.validate(params)
        return signals

    def get_signals_for_observation(
        self,
        observation: Union[ObservationView, Mapping[str, Any]],
        rolling_state: Any,
        horizon: Optional[int] = None,
    ) -> RLSignals:
        """供新主流程使用的显式受限入口。"""
        payload = _validate_observation(observation)
        if isinstance(rolling_state, Mapping):
            soc_obs = rolling_state.get("soc_obs")
        else:
            soc_obs = getattr(rolling_state, "soc_obs", None)
        if soc_obs is None:
            raise TypeError("rolling_state 必须提供 soc_obs")
        now = float(payload["now"])
        period_ell = int(math.floor(now / self.params.interval_hours))
        return self.get_signals(
            self.params,
            period_ell,
            self.params.horizon if horizon is None else horizon,
            soc_obs,
            observation=observation,
            rolling_state=rolling_state,
        )


def _print_summary(data: Mapping[str, Any], params: BusinessParameters) -> None:
    metadata = data["metadata"]
    # Keep CLI output ASCII so ``conda run`` also works in legacy GBK shells.
    print("=== six-station synthetic/mock data ===")
    print(
        f"schema={data['schema_version']}, seed={data['seed']}, "
        f"source={metadata['data_source']}/{metadata['signal_source']}"
    )
    print(
        f"stations={params.station.num_stations}, periods={params.num_periods}, "
        f"reservations={len(data['reservations'])}"
    )
    for i in range(params.station.num_stations):
        predicted = sum(len(items) for items in data["predicted_random_requests"][i])
        actual = sum(len(items) for items in data["actual_random_requests"][i])
        print(f"  station {i}: predicted={predicted}, actual={actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成可复现六站 synthetic/mock 输入")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(DEFAULT_MOCK_DATA_PATH))
    args = parser.parse_args()

    params = get_default_parameters()
    data = generate_mock_data(params, seed=args.seed)
    save_mock_data(data, args.output)
    _print_summary(data, params)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
