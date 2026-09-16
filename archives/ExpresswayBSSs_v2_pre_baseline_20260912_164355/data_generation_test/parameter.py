# -*- coding: utf-8 -*-
"""
业务参数模块：高速公路换电站运营 MPC-RL 分层优化的集中参数定义。

本文件对应论文 paper/main.tex 第 2-3 节的符号定义与 docs/archive/plan_mpc.md 第 1 节
“业务参数与模拟数据”的要求，集中保存后续候选网络生成
(candidate_network.py)、模拟数据生成 (rl_data.py)、日前计划
(dayahead_plan.py)、MPC 模型 (src/paper_mpc.py) 与滚动执行 (run_mpc.py)
所需的全部业务参数。

内容概览
--------
1. ``StationParameters``：换电站参数——站点编号、沿线位置 (km)、槽位数、
   初始逐槽 SOC、单槽功率上限和充电效率 eta。
2. ``ODPairParameters``：O-D 对参数——入口/出口位置 (km)、沿线站点索引。
   论文中 O-D 对 p 的节点集合 V^p = {o^p} ∪ S^p ∪ {d^p}，本模块用整数
   站点索引表示换电站节点，用字符串 "entry"/"exit" 表示入口/出口节点。
3. ``SolverParameters``：Gurobi 求解参数——线程数、时限、输出开关。
4. ``BusinessParameters``：聚合全部业务参数，并提供派生计算方法：
   - ``node_position_km`` / ``distance_km``：节点位置与站间距离 D_{i,j}^p；
   - ``soc_consumption``：归一化 SOC 消耗 v_{i,j}^p = 距离 / 续航；
   - ``travel_time_hours`` / ``travel_periods``：行驶时间 tau_{p,i} 及
     仅为兼容旧代码保留的时段折算；
   - ``interval_hours``、``max_wait_hours`` 和逐站逐区间
     ``station_energy_limit_kwh``：连续事件内核的时间与能量参数；
   - ``soc_bin``：按入口 SOC 查询所属离线分档（对应论文候选网络分档）；
   - ``reservation_failure_penalty``：每次预约换电未能由站内满电库存履约
     时计入的违约成本，使随机需求预测或功率轨迹不再把模型推入不可行。
5. ``get_default_parameters()``：构造可快速求解的非平凡默认实例。

默认实例（六站 synthetic/mock 联调）
------------------------------
- 12 个一小时运行时段（索引 0..11），预测时域 H = 4；
- 6 个换电站（索引 0..5），各 5 个槽位，位于 80/180/280/380/480/580 km；
- 电池容量 E_B = 100 kWh，续航 300 km（SOC 消耗 v = 距离/300）；
- 车速 75 km/h，连续区间长度 sigma = 1 h，最大等待时长 0.25 h；
- 2 个 O-D 对，入口均在 0 km，出口分别在 430 km 和 680 km：
  O-D 0 覆盖站 0--3，O-D 1 覆盖站 0--5；
- SOC 分档 [0.30,0.50)、[0.50,0.75)、[0.75,1.00]，出口最低 SOC 0.10；
- 充电效率 0.95，单槽功率上限 60 kW；每站每区间充电能量上限 240 kWh，
  不施加新的站级瞬时功率上限；
- 分时电价为 12 个时段的 synthetic 表，换电服务价 1.2 元/kWh
  （各站各时段相同）；
- 路径调整惩罚 kappa = 50，终端价值权重 beta = 1.0，
  最小换电间距下界 D_min = 100 km，随机种子 42；
- 初始逐槽 SOC（确定性给定，不随机生成）：每站 5 槽
  [1.0, 0.9, 0.6, 0.4, 0.2]，6 站相同；
- Gurobi 单线程、30 秒时限、默认关闭求解日志。

用法示例
--------
>>> from data_generation_test.parameter import get_default_parameters
>>> params = get_default_parameters()
>>> params.validate()                     # 不合法时抛出带明确信息的 ValueError
>>> params.soc_consumption(1, "entry", 2)  # O-D 1 入口(0km)到站2(280km)的SOC消耗
0.9333333333333333
>>> params.travel_periods(0, 0, "exit")    # O-D 0 站0(80km)到出口(430km)所需时段数
4.666666666666667
>>> params.soc_bin(0.62)                   # 0.62 属于第 1 档 [0.50,0.75)
1
>>> d = params.to_dict()                   # JSON 可序列化
>>> BusinessParameters.from_dict(d) == params
True

所有字段均为标准库类型（int/float/str/list/dict），``to_dict`` 的输出可
直接 ``json.dumps``；``save_json``/``load_json`` 提供落盘与读取。
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

__all__ = [
    "StationParameters",
    "ODPairParameters",
    "SolverParameters",
    "BusinessParameters",
    "get_default_parameters",
]

# 节点标识：换电站用整数站点索引，入口/出口用以下保留字符串。
ENTRY_NODE = "entry"
EXIT_NODE = "exit"
NodeId = Union[int, str]


@dataclass
class StationParameters:
    """换电站网络参数（对全部 O-D 对共享的一组物理站点）。

    station_ids 与 positions_km 等长且一一对应，位置按高速下游方向严格
    递增。站点索引一律使用 0..num_stations-1 的连续整数。
    """

    num_stations: int = 6
    station_ids: List[int] = field(default_factory=lambda: list(range(6)))
    positions_km: List[float] = field(
        default_factory=lambda: [80.0, 180.0, 280.0, 380.0, 480.0, 580.0]
    )
    num_slots: int = 5
    # 初始逐槽 SOC：initial_slot_soc[i][b] 为站点 i 槽位 b 的初始 SOC。
    initial_slot_soc: List[List[float]] = field(
        default_factory=lambda: [[1.0, 0.9, 0.6, 0.4, 0.2] for _ in range(6)]
    )
    slot_power_limit_kw: float = 60.0  # 单槽充电功率上限 \bar P_i (kW)
    charging_efficiency: float = 0.95  # 充电效率 eta_i



@dataclass
class ODPairParameters:
    """O-D 对参数：入口/出口位置及沿线换电站索引（按下游顺序）。"""

    od_id: int = 0
    entry_km: float = 0.0
    exit_km: float = 230.0
    station_indices: List[int] = field(default_factory=lambda: [0, 1])


@dataclass
class SolverParameters:
    """Gurobi 求解参数：单线程、30 秒时限、默认关闭输出。"""

    threads: int = 1
    time_limit_sec: float = 30.0
    output_flag: int = 0  # 0 关闭 Gurobi 日志，1 开启
    mip_gap: float = 0.0
    feasibility_tol: float = 1e-8


@dataclass
class BusinessParameters:
    """聚合的业务参数对象，后续各模块统一以此作为参数入口。

    节点标识约定：换电站节点用整数站点索引 i（0..num_stations-1），
    O-D 对的入口、出口节点分别用字符串 "entry"、"exit"。
    """

    # ---- 时段与预测时域 ----
    num_periods: int = 12  # 运行时段总数（索引 0..num_periods-1）
    interval_hours: float = 1.0  # 时段长度 sigma (h)
    horizon: int = 4  # MPC 预测时域 H（控制步数）

    # ---- 站点与 O-D 对 ----
    station: StationParameters = field(default_factory=StationParameters)
    od_pairs: List[ODPairParameters] = field(default_factory=list)

    # ---- 车辆与电池 ----
    # 默认模拟车速取 75 km/h，使入口到最近首站的时间严格大于 1 h。
    # 这是当前测试数据的临时时序规避条件，不是一般业务约束；见 note.md。
    vehicle_speed_kmh: float = 75.0  # 车速 (km/h)
    range_km: float = 300.0  # 满电续航 (km)，SOC 消耗 v = 距离/range_km
    battery_capacity_kwh: float = 100.0  # 额定电池能量 E_B (kWh)

    # ---- SOC 分档与出口要求 ----
    # soc_bins[h] = [下界, 上界]；除最后一档为闭区间外均为左闭右开。
    soc_bins: List[List[float]] = field(
        default_factory=lambda: [[0.30, 0.50], [0.50, 0.75], [0.75, 1.00]]
    )
    min_exit_soc: float = 0.10  # 到达出口的最低 SOC \underline s_d^p

    # ---- 价格（[站点][时段] 二维列表，单位：金额/电量） ----
    electricity_price: List[List[float]] = field(default_factory=list)  # e_{i,q}
    swap_service_price: List[List[float]] = field(default_factory=list)  # pi^sw_{i,q}
    # 每站每外层区间充电电量上限 (kWh)，替代旧的站级瞬时功率上限。
    station_energy_limit_kwh: List[List[float]] = field(default_factory=list)

    # ---- 目标函数系数 ----
    path_adjustment_penalty: float = 1.0  # 路径调整惩罚 kappa > 0
    reservation_failure_penalty: float = 1000.0  # 每次预约换电违约成本 c_fail
    terminal_value_weight: float = 1.0  # RL 终端价值权重 beta >= 0

    # ---- 候选网络 ----
    min_swap_spacing_km: float = 100.0  # 最小换电间距下界 D_min > 0
    max_wait_hours: float = 0.25  # 请求到站后的最大等待时长
    time_epsilon: float = 1e-9

    # ---- 其他 ----
    seed: int = 42  # 随机种子
    data_source: str = "synthetic"
    generator_version: str = "six-station-synthetic-v2"
    solver: SolverParameters = field(default_factory=SolverParameters)

    # ------------------------------------------------------------------
    # 校验
    # ------------------------------------------------------------------
    def validate(self) -> None:
        """校验参数合法性；不合法时抛出带明确信息的 ValueError。"""
        st = self.station

        if self.num_periods <= 0:
            raise ValueError(f"num_periods 必须为正整数，当前为 {self.num_periods}")
        if self.interval_hours <= 0:
            raise ValueError(
                f"interval_hours 必须为正数，当前为 {self.interval_hours}"
            )
        if self.horizon <= 0:
            raise ValueError(f"horizon(H) 必须为正整数，当前为 {self.horizon}")

        if st.num_stations <= 0:
            raise ValueError(f"num_stations 必须为正整数，当前为 {st.num_stations}")
        if st.station_ids != list(range(st.num_stations)):
            raise ValueError(
                f"station_ids 必须为连续索引 0..{st.num_stations - 1}，"
                f"当前为 {st.station_ids}"
            )
        if len(st.positions_km) != st.num_stations:
            raise ValueError(
                f"positions_km 长度 {len(st.positions_km)} 与 "
                f"num_stations={st.num_stations} 不一致"
            )
        if any(
            st.positions_km[i] >= st.positions_km[i + 1]
            for i in range(st.num_stations - 1)
        ):
            raise ValueError(
                f"positions_km 必须沿下游严格递增，当前为 {st.positions_km}"
            )
        if st.num_slots <= 0:
            raise ValueError(f"num_slots 必须为正整数，当前为 {st.num_slots}")
        if len(st.initial_slot_soc) != st.num_stations:
            raise ValueError(
                f"initial_slot_soc 行数 {len(st.initial_slot_soc)} 与 "
                f"num_stations={st.num_stations} 不一致"
            )
        for i, row in enumerate(st.initial_slot_soc):
            if len(row) != st.num_slots:
                raise ValueError(
                    f"initial_slot_soc[{i}] 长度 {len(row)} 与 "
                    f"num_slots={st.num_slots} 不一致"
                )
            for b, soc in enumerate(row):
                if not 0.0 <= soc <= 1.0:
                    raise ValueError(
                        f"initial_slot_soc[{i}][{b}]={soc} 超出 [0,1] 范围"
                    )
        if st.slot_power_limit_kw <= 0:
            raise ValueError(
                f"slot_power_limit_kw 必须为正数，当前为 {st.slot_power_limit_kw}"
            )
        if not 0.0 < st.charging_efficiency <= 1.0:
            raise ValueError(
                f"charging_efficiency 必须在 (0,1] 内，当前为 {st.charging_efficiency}"
            )

        if not self.od_pairs:
            raise ValueError("od_pairs 不能为空")
        seen_od_ids = set()
        for od in self.od_pairs:
            if od.od_id in seen_od_ids:
                raise ValueError(f"od_id={od.od_id} 重复")
            seen_od_ids.add(od.od_id)
            if not od.entry_km < od.exit_km:
                raise ValueError(
                    f"O-D {od.od_id}: entry_km({od.entry_km}) 必须小于 "
                    f"exit_km({od.exit_km})"
                )
            if not od.station_indices:
                raise ValueError(f"O-D {od.od_id}: station_indices 不能为空")
            prev = od.entry_km
            for idx in od.station_indices:
                if not 0 <= idx < st.num_stations:
                    raise ValueError(
                        f"O-D {od.od_id}: 站点索引 {idx} 超出 "
                        f"0..{st.num_stations - 1} 范围"
                    )
                pos = st.positions_km[idx]
                if not prev < pos:
                    raise ValueError(
                        f"O-D {od.od_id}: 站点 {idx} 位置 {pos} km 未沿下游递增"
                    )
                prev = pos
            if not prev < od.exit_km:
                raise ValueError(
                    f"O-D {od.od_id}: 出口 {od.exit_km} km 必须位于末站 {prev} km 下游"
                )

        if self.vehicle_speed_kmh <= 0:
            raise ValueError(
                f"vehicle_speed_kmh 必须为正数，当前为 {self.vehicle_speed_kmh}"
            )
        if self.range_km <= 0:
            raise ValueError(f"range_km 必须为正数，当前为 {self.range_km}")
        if self.battery_capacity_kwh <= 0:
            raise ValueError(
                f"battery_capacity_kwh 必须为正数，当前为 {self.battery_capacity_kwh}"
            )

        if not self.soc_bins:
            raise ValueError("soc_bins 不能为空")
        prev_upper = None
        for h, (lo, hi) in enumerate(self.soc_bins):
            if not 0.0 <= lo < hi <= 1.0:
                raise ValueError(
                    f"soc_bins[{h}]=[{lo},{hi}] 必须满足 0<=下界<上界<=1"
                )
            if prev_upper is not None and lo != prev_upper:
                raise ValueError(
                    f"soc_bins[{h}] 下界 {lo} 与上一档上界 {prev_upper} 不连续"
                )
            prev_upper = hi
        if self.soc_bins[-1][1] != 1.0:
            raise ValueError(
                f"最后一档 SOC 上界必须为 1.0，当前为 {self.soc_bins[-1][1]}"
            )
        if not 0.0 <= self.min_exit_soc < 1.0:
            raise ValueError(
                f"min_exit_soc 必须在 [0,1) 内，当前为 {self.min_exit_soc}"
            )

        for name, table in (
            ("electricity_price", self.electricity_price),
            ("swap_service_price", self.swap_service_price),
        ):
            if len(table) != st.num_stations:
                raise ValueError(
                    f"{name} 行数 {len(table)} 与 num_stations={st.num_stations} 不一致"
                )
            for i, row in enumerate(table):
                if len(row) != self.num_periods:
                    raise ValueError(
                        f"{name}[{i}] 长度 {len(row)} 与 "
                        f"num_periods={self.num_periods} 不一致"
                    )
                if any(p < 0 for p in row):
                    raise ValueError(f"{name}[{i}] 存在负数价格")

        if len(self.station_energy_limit_kwh) != st.num_stations:
            raise ValueError(
                "station_energy_limit_kwh 行数 "
                f"{len(self.station_energy_limit_kwh)} 与 "
                f"num_stations={st.num_stations} 不一致"
            )
        for i, row in enumerate(self.station_energy_limit_kwh):
            if len(row) != self.num_periods:
                raise ValueError(
                    f"station_energy_limit_kwh[{i}] 长度 {len(row)} 与 "
                    f"num_periods={self.num_periods} 不一致"
                )
            if any(limit <= 0.0 for limit in row):
                raise ValueError(
                    f"station_energy_limit_kwh[{i}] 必须全部为正数"
                )

        if self.path_adjustment_penalty <= 0:
            raise ValueError(
                f"path_adjustment_penalty(kappa) 必须为正数，当前为 "
                f"{self.path_adjustment_penalty}"
            )
        if self.reservation_failure_penalty <= 0:
            raise ValueError(
                "reservation_failure_penalty(c_fail) 必须为正数，当前为 "
                f"{self.reservation_failure_penalty}"
            )
        if self.terminal_value_weight < 0:
            raise ValueError(
                f"terminal_value_weight(beta) 必须非负，当前为 "
                f"{self.terminal_value_weight}"
            )
        if self.min_swap_spacing_km <= 0:
            raise ValueError(
                f"min_swap_spacing_km(D_min) 必须为正数，当前为 "
                f"{self.min_swap_spacing_km}"
            )
        if self.max_wait_hours <= 0:
            raise ValueError(
                f"max_wait_hours 必须为正数，当前为 {self.max_wait_hours}"
            )
        if self.time_epsilon <= 0:
            raise ValueError(
                f"time_epsilon 必须为正数，当前为 {self.time_epsilon}"
            )
        if self.data_source != "synthetic":
            raise ValueError("本轮仅接受 data_source='synthetic' 的可复现输入")
        if not self.generator_version:
            raise ValueError("generator_version 不能为空")
        if self.solver.threads <= 0:
            raise ValueError(
                f"solver.threads 必须为正整数，当前为 {self.solver.threads}"
            )
        if self.solver.time_limit_sec <= 0:
            raise ValueError(
                f"solver.time_limit_sec 必须为正数，当前为 {self.solver.time_limit_sec}"
            )
        if self.solver.output_flag not in (0, 1):
            raise ValueError(
                f"solver.output_flag 必须为 0 或 1，当前为 {self.solver.output_flag}"
            )
        if self.solver.mip_gap < 0:
            raise ValueError("solver.mip_gap 必须非负")
        if self.solver.feasibility_tol <= 0:
            raise ValueError("solver.feasibility_tol 必须为正数")

    # ------------------------------------------------------------------
    # 节点与几何派生量
    # ------------------------------------------------------------------
    def od_nodes(self, od_index: int) -> List[NodeId]:
        """返回 O-D 对的完整节点序列 [entry, 各站点索引..., exit]。"""
        od = self.od_pairs[od_index]
        return [ENTRY_NODE, *od.station_indices, EXIT_NODE]

    def node_position_km(self, od_index: int, node: NodeId) -> float:
        """返回节点位置 (km)。node 为站点索引 (int)、"entry" 或 "exit"。"""
        od = self.od_pairs[od_index]
        if node == ENTRY_NODE:
            return od.entry_km
        if node == EXIT_NODE:
            return od.exit_km
        if isinstance(node, int) and 0 <= node < self.station.num_stations:
            if node not in od.station_indices:
                raise ValueError(
                    f"站点 {node} 不在 O-D {od.od_id} 沿线站点 "
                    f"{od.station_indices} 中"
                )
            return self.station.positions_km[node]
        raise ValueError(f"O-D {od.od_id}: 无法识别的节点 {node!r}")

    def distance_km(self, od_index: int, from_node: NodeId, to_node: NodeId) -> float:
        """节点 i 到下游节点 j 的行驶距离 D_{i,j}^p (km)，必须下游。"""
        d = self.node_position_km(od_index, to_node) - self.node_position_km(
            od_index, from_node
        )
        if d <= 0:
            raise ValueError(
                f"O-D {self.od_pairs[od_index].od_id}: 节点 {from_node!r} -> "
                f"{to_node!r} 不是下游弧（距离 {d} km）"
            )
        return d

    def soc_consumption(self, od_index: int, from_node: NodeId, to_node: NodeId) -> float:
        """归一化 SOC 消耗 v_{i,j}^p = 距离 / 满电续航。"""
        return self.distance_km(od_index, from_node, to_node) / self.range_km

    def travel_time_hours(
        self, od_index: int, from_node: NodeId, to_node: NodeId
    ) -> float:
        """节点间行驶时间 (h)。"""
        return self.distance_km(od_index, from_node, to_node) / self.vehicle_speed_kmh

    def travel_periods(
        self, od_index: int, from_node: NodeId, to_node: NodeId
    ) -> float:
        """节点间行驶时间折算的运行时段数（行驶小时 / Delta）。"""
        return self.travel_time_hours(od_index, from_node, to_node) / self.interval_hours

    def travel_time_from_entry_hours(self, od_index: int, node: NodeId) -> float:
        """入口到节点的行驶时间 tau_{p,i} (h)。入口自身为 0。"""
        if node == ENTRY_NODE:
            return 0.0
        return self.travel_time_hours(od_index, ENTRY_NODE, node)

    def arrival_period(
        self, od_index: int, node: NodeId, entry_time: float
    ) -> int:
        """论文 q_{A,i}^{p,k} = [t_A + tau_{p,i}]：入口时刻 entry_time
        （单位：时段，连续时刻按 [q,q+1) 归并）出发的用户到达节点的时段。"""
        tau_periods = self.travel_time_from_entry_hours(od_index, node) / self.interval_hours
        return math.floor(entry_time + tau_periods)

    def period_of(self, time_in_hours: float) -> int:
        """连续时刻（小时）归入时段 [q,q+1)，返回时段索引 q。"""
        return math.floor(time_in_hours / self.interval_hours)

    # ------------------------------------------------------------------
    # SOC 分档
    # ------------------------------------------------------------------
    def soc_bin(self, soc: float) -> int:
        """返回入口 SOC 所属分档索引。除最后一档为闭区间外均左闭右开。"""
        if not 0.0 <= soc <= 1.0:
            raise ValueError(f"soc={soc} 超出 [0,1] 范围")
        for h, (lo, hi) in enumerate(self.soc_bins):
            if h == len(self.soc_bins) - 1:
                if lo <= soc <= hi:
                    return h
            elif lo <= soc < hi:
                return h
        raise ValueError(
            f"soc={soc} 不属于任何分档 {self.soc_bins}（分档未覆盖该区间）"
        )

    # ------------------------------------------------------------------
    # 充电相关派生量
    # ------------------------------------------------------------------
    def power_needed_to_full_kw(self, soc: float, station_index: int = 0) -> float:
        """兼容接口：在一个外层区间内补满所需的理论功率。

        station_index 保留给未来逐站效率差异；当前各站效率相同。
        """
        return (
            self.battery_capacity_kwh
            * (1.0 - soc)
            / (self.station.charging_efficiency * self.interval_hours)
        )

    def station_energy_limit_at(self, station_index: int, period: int) -> float:
        """返回站点在指定外层区间的充电能量上限（kWh）。"""
        return self.station_energy_limit_kwh[station_index][period]

    def electricity_price_at(self, station_index: int, period: int) -> float:
        """站点 station_index 在时段 period 的购电价格 e_{i,q}。"""
        return self.electricity_price[station_index][period]

    def swap_service_price_at(self, station_index: int, period: int) -> float:
        """站点 station_index 在时段 period 的换电服务价 pi^sw_{i,q}。"""
        return self.swap_service_price[station_index][period]

    # ------------------------------------------------------------------
    # JSON 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        """转换为仅含标准库类型、可直接 json.dumps 的字典。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "BusinessParameters":
        """由 to_dict 生成的字典重建参数对象。"""
        payload = dict(data)
        station_data = dict(payload["station"])
        if "station_energy_limit_kwh" not in payload:
            # 缺省时补默认站级能量上限（与 get_default_parameters 一致）。
            interval_hours = float(payload.get("interval_hours", 1.0))
            count = int(station_data.get("num_stations", 0))
            periods = int(payload.get("num_periods", 0))
            payload["station_energy_limit_kwh"] = [
                [240.0 * interval_hours for _ in range(periods)]
                for _ in range(count)
            ]
        station = StationParameters(**station_data)
        od_pairs = [ODPairParameters(**od) for od in payload["od_pairs"]]
        solver = SolverParameters(**payload["solver"])
        rest = {
            k: v
            for k, v in payload.items()
            if k not in ("station", "od_pairs", "solver")
        }
        return cls(station=station, od_pairs=od_pairs, solver=solver, **rest)

    def save_json(self, path: Union[str, Path]) -> None:
        """保存为 JSON 文件（UTF-8，带缩进）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "BusinessParameters":
        """从 JSON 文件加载参数对象。"""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def get_default_parameters() -> BusinessParameters:
    """构造可复现的六站、12 区间 synthetic 默认实例。"""
    num_periods = 12
    interval_hours = 1.0
    tou_price = [
        0.35, 0.35, 0.65, 1.10, 1.10, 0.65,
        0.40, 0.35, 0.35, 0.65, 1.10, 0.65,
    ]
    station = StationParameters()
    od_pairs = [
        ODPairParameters(
            od_id=0,
            entry_km=0.0,
            exit_km=430.0,
            station_indices=[0, 1, 2, 3],
        ),
        ODPairParameters(
            od_id=1,
            entry_km=0.0,
            exit_km=680.0,
            station_indices=[0, 1, 2, 3, 4, 5],
        ),
    ]
    params = BusinessParameters(
        num_periods=num_periods,
        interval_hours=interval_hours,
        horizon=4,
        station=station,
        od_pairs=od_pairs,
        electricity_price=[list(tou_price) for _ in range(station.num_stations)],
        swap_service_price=[
            [1.2] * num_periods for _ in range(station.num_stations)
        ],
        station_energy_limit_kwh=[
            [240.0 * interval_hours] * num_periods
            for _ in range(station.num_stations)
        ],
        max_wait_hours=0.25,
        seed=42,
        data_source="synthetic",
        generator_version="six-station-synthetic-v2",
        solver=SolverParameters(
            threads=1,
            time_limit_sec=30.0,
            output_flag=0,
            mip_gap=0.0,
            feasibility_tol=1e-8,
        ),
    )
    params.validate()
    return params


def _print_summary(params: BusinessParameters) -> None:
    """打印默认实例的关键参数摘要。"""
    st = params.station
    print("=== 业务参数摘要 ===")
    print(
        f"时段: {params.num_periods} 个 × {params.interval_hours} h，预测时域 H = {params.horizon}"
    )
    print(f"换电站: {st.num_stations} 站 × {st.num_slots} 槽，位置 {st.positions_km} km")
    print(f"初始逐槽 SOC (站0): {st.initial_slot_soc[0]}")
    for od in params.od_pairs:
        print(
            f"O-D {od.od_id}: 入口 {od.entry_km} km -> 出口 {od.exit_km} km，"
            f"沿线站点 {od.station_indices}"
        )
    print(
        f"车速 {params.vehicle_speed_kmh} km/h，续航 {params.range_km} km，"
        f"电池容量 {params.battery_capacity_kwh} kWh"
    )
    print(f"SOC 分档: {params.soc_bins}，出口最低 SOC = {params.min_exit_soc}")
    print(
        f"充电效率 {st.charging_efficiency}，单槽功率上限 {st.slot_power_limit_kw} kW，"
        f"站级区间能量上限 {params.station_energy_limit_kwh[0][0]} kWh"
    )
    print(f"分时电价 (站0): {params.electricity_price[0]}")
    print(f"换电服务价 (站0): {params.swap_service_price[0]} 元/kWh")
    print(
        f"kappa = {params.path_adjustment_penalty}，"
        f"c_fail = {params.reservation_failure_penalty}，"
        f"beta = {params.terminal_value_weight}，"
        f"D_min = {params.min_swap_spacing_km} km，seed = {params.seed}"
    )
    print(
        f"Gurobi: 线程 {params.solver.threads}，时限 {params.solver.time_limit_sec} s，"
        f"输出开关 {params.solver.output_flag}"
    )


if __name__ == "__main__":
    params = get_default_parameters()
    params.validate()
    _print_summary(params)

    # 派生方法演示
    print("\n=== 派生量示例 ===")
    print(
        "O-D 1 入口->站2 SOC 消耗 v =",
        round(params.soc_consumption(1, ENTRY_NODE, 2), 4),
    )
    print(
        "O-D 0 站0->出口 行驶时段数 =",
        params.travel_periods(0, 0, EXIT_NODE),
    )
    print(
        "入口时刻 0.5、O-D 0 站1 到达时段 q =",
        params.arrival_period(0, 1, 0.5),
    )
    print("SOC 0.62 所属分档 =", params.soc_bin(0.62))
    print(
        "SOC 0.4 补满所需功率 =",
        round(params.power_needed_to_full_kw(0.4), 2),
        "kW",
    )

    # JSON 序列化往返验证
    restored = BusinessParameters.from_dict(params.to_dict())
    assert restored == params, "to_dict/from_dict 往返不一致"
    json.dumps(params.to_dict(), ensure_ascii=False)
    print("\nJSON 序列化往返一致，校验通过。")
