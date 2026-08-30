# -*- coding: utf-8 -*-
"""
模拟数据与 RL 接口模块：为 MPC 滚动优化提供确定性 mock 数据与 RL 信号。

本文件对应论文 paper/mian11_fixed.tex 第 2.3.1 节（滚动时域信息，
137-187 行）与第 3 节 RL state/action/参考 rollout（686-720 行、
329-367 行）的接口要求，按 plan.md 第 22-27 行实施。真实 RL 策略
（actor/critic）尚未训练，本模块用完全确定性的 mock 实现打通
``src/mpc_model.py``、``src/dayahead_plan.py`` 与 ``run_mpc.py`` 的数据
与信号接口；未来真实 RL 只需实现同一 ``RLProvider`` 协议即可替换
``MockRLProvider``，MPC 侧代码无需修改。

内容概览
--------
1. ``generate_mock_data(params, seed=None)``：用
   ``random.Random(seed or params.seed)`` 生成全部模拟数据，包括：
   - 四名预约用户（论文集合 K^p）：od_id、用户 id、提交顺序、
     日前预约入口时刻 bar_t_A（小时，浮点，分布在 [0, num_periods-2)）、
     日前预约入口 SOC bar_soc_A（[0.3, 1.0]）、实际入口时刻 t_A 与实际
     入口 SOC soc_A（在日前值上加小扰动：时刻偏差 -0.5..+1.0 小时，
     可含逾期情形；SOC 偏差 ±0.05，并截断到 [最低 SOC 档下界, 1.0]
     以保证可查询 params.soc_bin）。
   - 逐请求随机需求预测 ``predicted_random_requests``：论文中时刻 ell
     对时段 q 生成的预测随机请求集合 R_{i,q}，每站每时段 0..3 个请求，
     每个含请求 id、预测到站时刻（落在 [q, q+1)）与预测到站 SOC。
   - 实际随机到达 ``actual_random_requests``：与预测独立的实际集合
     （论文中时段 q 的实际到达集合），结构同上。生成后强制站 0 时段 0
     的实际请求数超过该站初始满电电池数，构造 FCFS（先到先服务）
     超需求验证场景，并在 ``fcfs_stress_test`` 字段中记录。
   - 初始逐槽 SOC ``initial_slot_soc``：直接取
     ``params.station.initial_slot_soc``（确定性给定）。
   - 日前随机需求预测 ``day_ahead_random_forecast``：供日前计划
     （dayahead_plan.py）使用的需求预测，结构与逐请求预测一致。
2. ``save_mock_data(data, path)`` / ``load_mock_data(path)``：JSON 落盘
   与读取，默认路径 ``data_generation_test/output/mock_rl_data.json``
   （以本文件位置解析，与当前工作目录无关）。
3. RL 接口（供 src/mpc_model.py 对接）：
   - ``RLSignals`` dataclass：逐槽请求功率 ``requested_power``
     （[站][槽][h] 的 H 步序列，h=0 对应当前时段 ell，单位 kW）、
     终端 SOC 边际价值 ``terminal_soc_value``（[站][槽]，即论文
     lambda^S_{i,b}，单位：金额/单位 SOC）、逐站域外换电价值斜率
     ``outside_swap_lambda``（[站]），以及方法
     ``outside_swap_value(station_index, rho)`` 计算论文
     Delta V^out_i(rho) = lambda_i * (rho - 1)。
   - ``RLProvider`` 协议（typing.Protocol）：
     ``get_signals(params, period_ell, horizon, soc_obs) -> RLSignals``，
     其中 ``soc_obs`` 为当前观测逐槽 SOC（[站][槽]）。
   - ``MockRLProvider``：确定性实现。请求功率采用"受槽位 60 kW 与
     站级 240 kW 限制的确定性充电策略"：每个站、每个预测步内按 SOC
     从低到高（并列按槽号升序）依次分配功率，直到电池补满或站级
     能量预算耗尽；步间按充电效率递推名义 SOC。模拟线性 critic：
     lambda^S_{i,b} = 窗口平均电价 x soc_value_coeff（默认
     = 电池容量 E_B，即"一块满电电池按均价估值"），恒为正；
     域外换电价值 Delta V^out_i(rho) = lambda_i * (rho - 1)，
     rho < 1 时为负。终端接口以非零值进入 MPC 目标。
4. CLI：``python data_generation_test/rl_data.py [--seed N] [--output PATH]``
   生成、保存并打印摘要（预约数、各站各时段预测/实际随机请求数、
   FCFS 超需求场景说明）。

确定性
------
同一 seed 重跑输出完全一致：全部随机量来自单个
``random.Random(seed or params.seed)`` 实例，且抽样顺序固定
（预约用户 -> 逐请求预测 -> 实际到达 -> FCFS 强制 -> 日前预测）。

用法示例
--------
>>> from data_generation_test.parameter import get_default_parameters
>>> from data_generation_test.rl_data import (
...     generate_mock_data, MockRLProvider,
... )
>>> params = get_default_parameters()
>>> data = generate_mock_data(params, seed=42)
>>> len(data["reservations"])
4
>>> provider = MockRLProvider(params)
>>> signals = provider.get_signals(
...     params, period_ell=0, horizon=params.horizon,
...     soc_obs=data["initial_slot_soc"],
... )
>>> signals.terminal_soc_value[0][0] > 0
True
>>> signals.outside_swap_value(0, 0.6) < 0
True
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Sequence, Union, runtime_checkable

try:  # 作为包的一部分导入（run_mpc.py 等）
    from data_generation_test.parameter import (
        BusinessParameters,
        get_default_parameters,
    )
except ImportError:  # 直接以脚本方式运行（python data_generation_test/rl_data.py）
    from parameter import BusinessParameters, get_default_parameters

__all__ = [
    "DEFAULT_MOCK_DATA_PATH",
    "NUM_MOCK_RESERVATIONS",
    "generate_mock_data",
    "save_mock_data",
    "load_mock_data",
    "RLSignals",
    "RLProvider",
    "MockRLProvider",
]

# 默认输出路径：相对本文件位置解析，与运行时工作目录无关。
DEFAULT_MOCK_DATA_PATH = (
    Path(__file__).resolve().parent / "output" / "mock_rl_data.json"
)

# mock 预约用户数量（论文预约用户集合 K^p 的规模）。
NUM_MOCK_RESERVATIONS = 4

# 每站每时段随机请求数上限（0..3 个）。
MAX_REQUESTS_PER_STATION_PERIOD = 3

# 问题 2 的临时 mock 规避条件：入口到首个沿线站的最短行驶时间必须
# 严格大于决策周期 Delta。当前 Delta=1 h，完整假设见根目录 note.md。

# FCFS 验证场景：强制 (FCFS_STRESS_STATION, FCFS_STRESS_PERIOD) 时段的
# 实际随机请求数超过该站初始满电电池数。
FCFS_STRESS_STATION = 0
FCFS_STRESS_PERIOD = 0

# 预约实际入口时刻相对日前值的扰动范围（小时，负值为提前、正值为逾期）。
RESERVATION_TIME_NOISE_LO = -0.5
RESERVATION_TIME_NOISE_HI = 1.0
# 预约实际入口 SOC 相对日前值的扰动范围。
RESERVATION_SOC_NOISE = 0.05


# ----------------------------------------------------------------------
# 模拟数据生成
# ----------------------------------------------------------------------
def _generate_request_set(
    rng: random.Random,
    params: BusinessParameters,
    id_prefix: str,
) -> List[List[List[Dict]]]:
    """生成 [站][时段] 的随机请求集合，每站每时段 0..3 个请求。

    每个请求含：
    - ``request_id``：``{id_prefix}{站}_{时段}_{序号}``；
    - ``arrival_time``：预测/实际到站时刻（小时，落在 [q, q+1)）；
    - ``arrival_soc``：到站 SOC（[0.2, 0.9]）。
    """
    st = params.station
    requests: List[List[List[Dict]]] = []
    for i in range(st.num_stations):
        per_station: List[List[Dict]] = []
        for q in range(params.num_periods):
            n = rng.randint(0, MAX_REQUESTS_PER_STATION_PERIOD)
            period_requests = []
            for k in range(n):
                arrival_time = q * params.delta_hours + rng.random() * params.delta_hours
                period_requests.append(
                    {
                        "request_id": f"{id_prefix}{i}_{q}_{k}",
                        "arrival_time": arrival_time,
                        "arrival_soc": rng.uniform(0.2, 0.9),
                    }
                )
            # 按到站时刻升序、请求 id 字典序固定排序（FCFS 依据）。
            period_requests.sort(key=lambda r: (r["arrival_time"], r["request_id"]))
            per_station.append(period_requests)
        requests.append(per_station)
    return requests


def generate_mock_data(
    params: BusinessParameters, seed: int = None
) -> Dict:
    """生成全部确定性 mock 数据（同一 seed 输出完全一致）。

    参数
    ----
    params : BusinessParameters
        业务参数对象（见 data_generation_test/parameter.py）。
    seed : int, optional
        随机种子；为 None 时使用 ``params.seed``。

    返回
    ----
    dict
        可直接 ``json.dumps`` 的字典，键名清单见模块 docstring 与
        ``save_mock_data``；主要键：
        ``schema_version``、``seed``、``reservations``、
        ``predicted_random_requests``、``actual_random_requests``、
        ``day_ahead_random_forecast``、``initial_slot_soc``、
        ``fcfs_stress_test``、``timing_workaround``。
    """
    # 当前滚动实现到 ell 才观测上一时段进入的预约用户。为避免用户在路径
    # 发布前已经到达首站，mock 场景暂时强制入口到首个沿线站的时间 > Delta。
    first_station_times = []
    for od_index, od in enumerate(params.od_pairs):
        if not od.station_indices:
            continue
        first_station_times.append(
            params.travel_time_from_entry_hours(od_index, od.station_indices[0])
        )
    if not first_station_times:
        raise ValueError("mock 数据至少需要一个含换电站的 O-D 对")
    min_first_station_time = min(first_station_times)
    if min_first_station_time <= params.delta_hours:
        raise ValueError(
            "mock 时序规避条件不满足：入口到第一个可能换电站的最短"
            f"行驶时间为 {min_first_station_time:.6f} h，必须严格大于 "
            f"决策周期 Delta={params.delta_hours:.6f} h；见 note.md"
        )

    rng = random.Random(seed if seed is not None else params.seed)
    st = params.station
    used_seed = seed if seed is not None else params.seed

    # ---- 1. 四名预约用户（日前信息 + 实际入口时刻/SOC 偏差） ----
    reservations: List[Dict] = []
    for k in range(NUM_MOCK_RESERVATIONS):
        od_id = params.od_pairs[rng.randrange(len(params.od_pairs))].od_id
        bar_t_a = rng.uniform(0.0, params.num_periods - 2)  # 日前预约入口时刻
        bar_soc_a = rng.uniform(0.3, 1.0)  # 日前预约入口 SOC
        t_a = max(0.0, bar_t_a + rng.uniform(RESERVATION_TIME_NOISE_LO,
                                             RESERVATION_TIME_NOISE_HI))
        soc_a = min(1.0, max(params.soc_bins[0][0], bar_soc_a + rng.uniform(
            -RESERVATION_SOC_NOISE, RESERVATION_SOC_NOISE)))
        reservations.append(
            {
                "reservation_id": k,  # 用户 id，同时作为提交顺序（0 最先提交）
                "submission_order": k,
                "od_id": od_id,
                "day_ahead_entry_time": bar_t_a,  # \bar t_A^{p,k} (h)
                "day_ahead_entry_soc": bar_soc_a,  # \bar soc_A^{p,k}
                "actual_entry_time": t_a,  # t_A^{p,k} (h)，可逾期
                "actual_entry_soc": soc_a,  # soc_A^{p,k}
            }
        )

    # ---- 2. 逐请求随机需求预测（滚动 MPC 使用的 R_{i,q}） ----
    predicted_random_requests = _generate_request_set(rng, params, "P")

    # ---- 3. 实际随机到达（与预测独立抽样） ----
    actual_random_requests = _generate_request_set(rng, params, "A")

    # ---- 4. 强制 FCFS 超需求场景：站 FCFS_STRESS_STATION 时段
    # FCFS_STRESS_PERIOD 的实际请求数 > 该站初始满电电池数 ----
    initial_slot_soc = [list(row) for row in st.initial_slot_soc]
    i0, q0 = FCFS_STRESS_STATION, FCFS_STRESS_PERIOD
    full_batteries = sum(1 for s in initial_slot_soc[i0] if s >= 1.0 - 1e-9)
    target_count = full_batteries + 2  # 严格超过剩余满电电池数
    period_reqs = actual_random_requests[i0][q0]
    while len(period_reqs) < target_count:
        k = len(period_reqs)
        period_reqs.append(
            {
                "request_id": f"A{i0}_{q0}_{k}",
                "arrival_time": q0 * params.delta_hours
                + rng.random() * params.delta_hours,
                "arrival_soc": rng.uniform(0.2, 0.9),
            }
        )
    period_reqs.sort(key=lambda r: (r["arrival_time"], r["request_id"]))
    fcfs_stress_test = {
        "station": i0,
        "period": q0,
        "num_actual_requests": len(period_reqs),
        "initial_full_batteries": full_batteries,
        "description": (
            f"站 {i0} 时段 {q0} 实际随机请求数 {len(period_reqs)} > "
            f"初始满电电池数 {full_batteries}，用于验证随机请求严格 FCFS "
            f"前缀：只能服务先到站的 {full_batteries} 个请求。"
        ),
    }

    # ---- 5. 日前随机需求预测（供日前计划使用） ----
    day_ahead_random_forecast = _generate_request_set(rng, params, "D")

    return {
        "schema_version": 1,
        "seed": used_seed,
        "reservations": reservations,
        "predicted_random_requests": predicted_random_requests,
        "actual_random_requests": actual_random_requests,
        "day_ahead_random_forecast": day_ahead_random_forecast,
        "initial_slot_soc": initial_slot_soc,
        "fcfs_stress_test": fcfs_stress_test,
        "timing_workaround": {
            "min_first_swap_travel_hours": min_first_station_time,
            "decision_period_hours": params.delta_hours,
            "note": "临时规避条件；一般化时序处理见根目录 note.md",
        },
    }


def save_mock_data(
    data: Dict, path: Union[str, Path] = DEFAULT_MOCK_DATA_PATH
) -> None:
    """将 mock 数据保存为 JSON（UTF-8，带缩进）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_mock_data(path: Union[str, Path] = DEFAULT_MOCK_DATA_PATH) -> Dict:
    """从 JSON 文件加载 mock 数据。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# RL 接口
# ----------------------------------------------------------------------
@dataclass
class RLSignals:
    """RL 层提供给单轮 MPC 的全部信号。

    字段
    ----
    start_period : int
        当前滚动决策时段 ell。
    horizon : int
        信号覆盖的预测步数 H；``requested_power[*][*][h]`` 对应
        时段 ``start_period + h``（h = 0..H-1）。
    requested_power : List[List[List[float]]]
        逐槽请求功率 \\widehat P_{i,b,q}（kW），形状
        [站][槽][h]，h = 0..H-1。满足 0 <= P <= 槽位功率上限，
        且逐站逐时段 Delta * sum_b P <= 站级能量上限。
    terminal_soc_value : List[List[float]]
        终端 SOC 边际价值 lambda^S_{i,b}（金额/单位 SOC），
        形状 [站][槽]，即论文式 (marginal_value_soc)；进入 MPC 目标的
        线性项 lambda^S_{i,b} * S_{i,b,ell+H}。
    outside_swap_lambda : List[float]
        逐站域外换电价值斜率 lambda_i（金额/单位 SOC），形状 [站]。

    方法
    ----
    outside_swap_value(station_index, rho)
        域外换电价值 Delta V^out_i(rho) = lambda_i * (rho - 1)，
        即论文式 (outside_swap_value)；rho < 1 时为负。
    """

    start_period: int
    horizon: int
    requested_power: List[List[List[float]]]
    terminal_soc_value: List[List[float]]
    outside_swap_lambda: List[float]

    def outside_swap_value(self, station_index: int, rho: float) -> float:
        """Delta V^out_i(rho) = lambda_i * (rho - 1)。"""
        return self.outside_swap_lambda[station_index] * (rho - 1.0)


@runtime_checkable
class RLProvider(Protocol):
    """RL 信号提供者协议：真实 actor/critic 与 MockRLProvider 均实现。

    未来真实 RL 训练完成后，用同一签名的实现替换 MockRLProvider 即可，
    MPC 侧（src/mpc_model.py）无需修改。
    """

    def get_signals(
        self,
        params: BusinessParameters,
        period_ell: int,
        horizon: int,
        soc_obs: Sequence[Sequence[float]],
    ) -> RLSignals:
        """返回当前滚动时刻的 RL 信号。

        参数
        ----
        params : BusinessParameters
            业务参数对象。
        period_ell : int
            当前滚动决策时段 ell。
        horizon : int
            预测时域步数 H（控制期 ell..ell+H-1）。
        soc_obs : Sequence[Sequence[float]]
            当前观测逐槽 SOC，形状 [站][槽]。
        """
        ...


class MockRLProvider:
    """确定性 mock RL 实现（论文参考 rollout 的简化确定性版本）。

    - 请求功率：受槽位功率上限与站级能量上限限制的确定性充电策略，并为
      MPC 在满电 SOC 容差内的微量补足预留 ``p_tol`` 功率裕量。
      每个预测步内，按 SOC 从低到高（并列按槽号升序）依次为各槽分配
      功率 min(槽位上限, 补满所需功率, 站级剩余预算/Delta)，直到补满
      或站级预算耗尽；步间按充电效率递推名义 SOC（不模拟换电）。
    - 模拟线性 critic：lambda^S_{i,b} = 窗口平均电价 x soc_value_coeff，
      恒为正；Delta V^out_i(rho) = lambda_i * (rho - 1)。

    参数
    ----
    params : BusinessParameters
        业务参数对象（用于读取功率/能量上限与电价）。
    soc_value_coeff : float, optional
        单位 SOC 价值系数（kWh 量纲）。默认取电池容量 E_B，即
        "一块满电电池按窗口均价估值"，保证终端项非零且量级合理。
    """

    def __init__(
        self, params: BusinessParameters, soc_value_coeff: float = None
    ) -> None:
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
    ) -> RLSignals:
        """确定性生成 H 步请求功率序列、终端 SOC 边际价值与域外斜率。"""
        st = params.station
        delta = params.delta_hours
        eta = st.charging_efficiency
        e_b = params.battery_capacity_kwh
        p_tol = params.full_power_tolerance_kw()
        # 每槽最多可能由 MPC 在容差带内补足 p_tol；actor 请求侧预留相同
        # 裕量，保证校正后的实际 P 仍满足单槽和站级物理上限。
        slot_limit = st.slot_power_limit_kw - p_tol
        station_energy_budget = (
            st.station_power_limit_kw - st.num_slots * p_tol
        ) * delta  # kWh

        # ---- 逐槽请求功率：确定性充电策略的 H 步 rollout ----
        sim_soc = [list(row) for row in soc_obs]
        requested_power: List[List[List[float]]] = [
            [[0.0] * horizon for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        for h in range(horizon):
            for i in range(st.num_stations):
                remaining_kwh = station_energy_budget
                # 按 SOC 从低到高分配，并列按槽号升序（确定性）。
                order = sorted(
                    range(st.num_slots), key=lambda b: (sim_soc[i][b], b)
                )
                for b in order:
                    need_kw = params.power_needed_to_full_kw(sim_soc[i][b], i)
                    p = min(slot_limit, need_kw, remaining_kwh / delta)
                    p = max(0.0, p)
                    requested_power[i][b][h] = p
                    remaining_kwh -= p * delta
                    sim_soc[i][b] = min(
                        1.0, sim_soc[i][b] + p * delta * eta / e_b
                    )

        # ---- 模拟线性 critic：终端 SOC 边际价值（恒正） ----
        terminal_soc_value: List[List[float]] = []
        outside_swap_lambda: List[float] = []
        for i in range(st.num_stations):
            price_row = params.electricity_price[i]
            lo = min(period_ell, params.num_periods)
            hi = min(period_ell + horizon, params.num_periods)
            window = price_row[lo:hi] if hi > lo else price_row[-1:]
            price_avg = sum(window) / len(window)
            lam = self.soc_value_coeff * price_avg  # 金额/单位 SOC，> 0
            terminal_soc_value.append([lam] * st.num_slots)
            outside_swap_lambda.append(lam)

        return RLSignals(
            start_period=period_ell,
            horizon=horizon,
            requested_power=requested_power,
            terminal_soc_value=terminal_soc_value,
            outside_swap_lambda=outside_swap_lambda,
        )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _print_summary(data: Dict, params: BusinessParameters) -> None:
    """打印 mock 数据摘要（预约、预测/实际随机请求计数、FCFS 场景）。"""
    print("=== Mock 数据摘要 ===")
    print(f"seed = {data['seed']}，schema_version = {data['schema_version']}")

    print(f"\n预约用户（{len(data['reservations'])} 名）:")
    for r in data["reservations"]:
        overdue = r["actual_entry_time"] - r["day_ahead_entry_time"]
        print(
            f"  用户 {r['reservation_id']} (O-D {r['od_id']}): "
            f"日前 bar_t_A={r['day_ahead_entry_time']:.2f} h, "
            f"bar_soc_A={r['day_ahead_entry_soc']:.3f}; "
            f"实际 t_A={r['actual_entry_time']:.2f} h "
            f"(偏差 {overdue:+.2f} h), soc_A={r['actual_entry_soc']:.3f}"
        )

    st = params.station
    print("\n各站各时段随机请求数（预测 / 实际）:")
    for i in range(st.num_stations):
        pred_counts = [len(q) for q in data["predicted_random_requests"][i]]
        act_counts = [len(q) for q in data["actual_random_requests"][i]]
        print(f"  站 {i}: 预测 {pred_counts}")
        print(f"       实际 {act_counts}")

    fcfs = data["fcfs_stress_test"]
    print("\nFCFS 超需求验证场景:")
    print(f"  {fcfs['description']}")

    da_counts = sum(
        len(reqs)
        for per_station in data["day_ahead_random_forecast"]
        for reqs in per_station
    )
    print(f"\n日前随机需求预测请求总数: {da_counts}")
    print(f"初始逐槽 SOC (站0): {data['initial_slot_soc'][0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 MPC-RL 分层优化的确定性 mock 数据（JSON）。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子；缺省使用 BusinessParameters.seed (42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_MOCK_DATA_PATH),
        help=f"输出 JSON 路径（默认 {DEFAULT_MOCK_DATA_PATH}）",
    )
    args = parser.parse_args()

    params = get_default_parameters()
    data = generate_mock_data(params, seed=args.seed)
    save_mock_data(data, args.output)
    _print_summary(data, params)
    print(f"\n已保存: {args.output}")


if __name__ == "__main__":
    main()
