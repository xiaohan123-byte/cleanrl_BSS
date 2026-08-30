# -*- coding: utf-8 -*-
r"""
MPC 主体模块：论文 2.2.4 节“Optimization Model”的 Gurobi 混合整数规划实现。

对应论文 paper/mian11_fixed.tex 第 137-235 行（滚动时域机制、用户集合、
有效入口信息式 eq:effective_arrival_information）与第 235-625 行（完整
优化模型：目标式 eq:objective、收益式 eq:income、充电成本式
eq:chargingcost、调整成本式 eq:adjustment_cost、RL 终端价值式
eq:rl_terminal_value、约束 1-6 式 eq:flow 到 eq:continuous_soc_domains），
按 plan.md 第 55-74 行实施。依赖仅限标准库、gurobipy 与
data_generation_test 下的 parameter / candidate_network / rl_data 三个
模块（RLSignals/RLProvider 协议）以及 src/dayahead_plan.py 的落盘格式。

预约用户索引约定
----------------
- 论文中的预约用户索引始终是 ``(p,k)``。本模块公开
  ``UserKey = (od_id, user_id)``，并用它作为所有路径变量、调整变量、
  预约事件和结果字典的用户标识；不同 O-D 可以合法复用同一个
  ``user_id``，同一 ``UserKey`` 在一个窗口内必须唯一。
- ``ReservationObservation`` 与 ``FixedCommitment`` 仍保留原有
  ``od_id`` / ``user_id`` 字段，兼容既有构造代码；二者的
  ``user_key`` 只读属性负责生成复合键。

时域索引约定（plan.md 一致性修正）
----------------------------------
- ``H = params.horizon`` 为总控制步数；当前滚动决策时段为 ``ell``。
- 控制期（充电与换电服务发生的时段）为 ``q = ell, ..., ell+H-1``，共 H 步；
  代码中用窗口内偏移 ``h = q - ell ∈ 0..H-1``。
- SOC 状态：``S[i,b,h]`` 为时段 ``ell+h`` 结束（换电后）的槽位 SOC，
  ``h = 0..H-1``；``S[i,b,-1]`` 为初始观测 ``S^obs[ell-1]``（固定界实现，
  对应 Constraint 6 式 eq:initial_battery_soc）；终端状态
  ``S[ell+H]`` 即 ``S[i,b,H-1]``（控制期之外不再充电/服务）。
- 论文符号表记 T={ell,...,ell+H}；按 plan.md 修正，充电功率 P、服务就绪
  g、换电事件均只定义在 h=0..H-1，终端 SOC 取 S[ell+H]=S[h=H-1]。
- “域外事件”判定：预约换电到站时段 ``q_{A,i} > ell+H-1``（即
  ``q >= ell+H``，超出控制期）时不进入逐槽 SOC 转移，仅以
  ``DeltaV^out_i(rho) * y[j,i]`` 进入终端价值式 eq:rl_terminal_value
  的第二项。

论文公式到代码的映射表
----------------------
- eq:flow（Constraint 1 流平衡）：
  ``build_model`` 中 ``flow[...]`` 约束，逐决策用户在
  ``get_feasible_arcs(network, od_id, 有效入口SOC)`` 生成的个体可行弧集
  A^{p,k} 上建二元 y 变量。
- eq:station_visit_indicator / eq:path_adjustment_indicator
  （Constraint 2 路径调整指示 d）：``visit[...]`` 表达式与
  ``adj_pos[...]`` / ``adj_neg[...]`` 约束；基准站点访问指示
  bar_x 由 ReservationObservation.baseline_path_arcs 预先计算。
- eq:power_saturation（Constraint 3 请求功率截断，g 两分支）：
  indicator 约束 ``pow_defer_*``（g=0: P=P_hat 且
  P_hat<=P_need-p_tol）与 ``pow_fill_*``（g=1: P=P_need 且
  P_hat>=P_need-p_tol），其中 ``P_need = E_B(1-S_prev)/(eta*Delta)``，
  ``p_tol`` 由统一的满电 SOC 容差换算得到。
- eq:continuous_charging_transition（充电后 SOC）：``charge[...]``。
- eq:return_soc / eq:swap_event_sets / eq:event_activation_and_soc /
  eq:total_reservation_swap（Constraint 4 三类事件与激活量）：
  事件构造见 ``_build_events``；``res_served[...]`` 与
  ``res_shortage[...]`` 显式表示预约站内履约/缺供，``res_priority[...]``
  强制有缺供时全部满电库存先服务预约；随机严格 FCFS 前缀
  ``fcfs_lb[...]`` / ``fcfs_ub[...]`` 只使用预约后的物理余量；
  ``ready_cnt[...]``（F=sum g）与
  ``ready_rel_lb/ub[...]``（g 与满电状态双向绑定，式
  eq:service_ready_relation）。
- eq:battery_event_assignment（Constraint 5 事件-槽匹配）：
  ``match_event[...]``（每激活事件恰一块电池）、
  ``match_ready[...]``（只用服务就绪槽）、
  ``match_min_slot[...]``（最小可用槽位）、
  ``match_no_cross[...]``（无交叉规范化）；换电后 SOC 转移
  （式 eq:continuous_swap_transition）用 indicator 约束
  ``swap_idle[...]``（used=0 => S=S_pre）与 ``swap_fire[...]``
  （alpha=1 => S=rho）。
- eq:initial_battery_soc（Constraint 6 初始状态）：S[i,b,-1] 的固定界；
  变量域同式 eq:binary_variables 到 eq:continuous_soc_domains；
  不设置终端满电库存下界。
- eq:income（I^A、I^R）、eq:chargingcost（C_ch）、
  eq:adjustment_cost（C_adj）、预约违约成本（C_fail）、
  eq:rl_terminal_value（Phi^RL）、
  eq:objective（总目标）：``build_model`` 末尾的目标表达式与
  ``MPCResult`` 的目标分项字段。

事件顺序（规范化匹配的依据，固定）
----------------------------------
- 预约事件按“已发布固定承诺、本轮新到预约、尚未发布未来预约”分级，再按
  （到站时段 q、O-D ID、用户 ID、入弧起点位置、入弧起点名字典序）排序；
  随机事件按（到站时段 q、预测到站时刻、
  请求 ID）排序；同一（站, 时段）内先预约后随机；槽位始终升序。

关键实现决定
------------
1. 满电边界硬约束：令 ``eps_soc=params.full_soc_tolerance``、
   ``p_tol=E_B*eps_soc/(eta*Delta)``。请求功率距补满功率不超过
   ``p_tol`` 时自动微量补足到 SOC 1 并强制 ``g=1``；``g=0`` 时
   ``S_pre<=1-eps_soc``。因此 ``P_hat=P_need`` 以及
   ``S_prev=1,P_hat=0`` 均不能再被经济目标主动标为不就绪。actor 请求
   在单槽和站级上限内预留相应裕量，MPC 再显式约束校正后的实际功率。
2. 域外事件：q >= ell+H 的预约换电不进事件集合与 SOC 转移。决策预约
   仅在入弧 y 被选择且用户全部域内事件均履约时计入
   RLSignals.outside_swap_value(i, rho)；固定承诺的域外事件已含于参考
   状态，若用户在域内失败则显式撤销对应参考价值。
3. q < ell 钳位：因离散时段归并，本轮新进入用户（t_A ∈ (ell-1, ell]）
   到达首站的时段理论上可能落在 ell-1（时刻已过）。此时把事件时段钳位
   到 ell（视为当前时段内服务），属离散化一致性修正；固定承诺事件
   q < ell 则视为输入错误（应已在早前时段履约）。
4. 价格延展：q >= params.num_periods 时电价与服务价按最后一个运营时段
   的价格延展（滚动末期窗口超出运营日的情形），不改变运营日内的数值。
5. RL 请求功率由 actor 可行域参数化满足扣除 ``p_tol`` 裕量后的槽位/
   站级上限（式 eq:rl_action_projection）；MPC 对容差内缺口补足后，仍
   显式施加原始单槽和站级功率上限。
6. 预约缺供不再令模型不可行：每个激活预约事件要么由站内满电池履约，
   要么计一次 ``reservation_failure_penalty``。硬约束仍保证预约先占用
   全部可用库存，随机请求只能按 FCFS 使用剩余满电池；预测违约仅表示
   当前窗口的风险，不会在滚动执行前提前结算。

数据文件接口
------------
- 候选网络：``data_generation_test/output/candidate_network.json``
  （candidate_network.load_candidate_network）；在线用
  ``get_feasible_arcs(network, od_index, entry_soc)`` 生成 A^{p,k}。
- 日前计划：``data_generation_test/output/dayahead_plan.json``
  （dayahead_plan.load_dayahead_plan）；用 ``reservations[i]`` 的
  ``od_id`` 与 ``path_arcs``（日前基准路径 ȳ）构造
  ``ReservationObservation``（见 ``make_reservation_observation``）。
- RL 信号：实现 ``data_generation_test.rl_data.RLProvider`` 协议的对象
  （缺省 MockRLProvider），返回 ``RLSignals``：
  ``requested_power[站][槽][h]`` 对应对段 ell+h（h=0..H-1，单位 kW）；
  ``terminal_soc_value[站][槽]`` 即 lambda^S；``outside_swap_value(i,
  rho)`` 即 DeltaV^out_i(rho)。

最小调用示例
------------
>>> from data_generation_test.parameter import get_default_parameters
>>> from data_generation_test.rl_data import MockRLProvider
>>> from src.mpc_model import (
...     MPCController, MPCWindowInput, RollingState, RandomRequest)
>>> params = get_default_parameters()
>>> ctl = MPCController.from_files(params)          # 默认输出目录下的 JSON
>>> mock = __import__("json").load(open(
...     "data_generation_test/output/mock_rl_data.json", encoding="utf-8"))
>>> signals = ctl.rl_provider.get_signals(
...     params, period_ell=0, horizon=params.horizon,
...     soc_obs=mock["initial_slot_soc"])
>>> obs = [ctl.make_reservation_observation(          # 本轮新进入用户 0
...     0, mock["reservations"][0]["actual_entry_time"],
...     mock["reservations"][0]["actual_entry_soc"], True)]
>>> reqs = [RandomRequest(station=i, period=0, request_id=r["request_id"],
...          arrival_time=r["arrival_time"], arrival_soc=r["arrival_soc"])
...         for i in range(3) for r in mock["predicted_random_requests"][i][0]]
>>> window = MPCWindowInput(params=params,
...     rolling_state=RollingState(soc_obs=mock["initial_slot_soc"],
...                                period_ell=0),
...     reservations=obs, random_requests=reqs, fixed_commitments=[],
...     rl_signals=signals)
>>> result = ctl.solve_step(window)                   # 最优解
>>> result.is_optimal
True
>>> exec_pkg = ctl.execute_period(result)             # 首阶段执行包
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:  # 作为包导入（from src.mpc_model import ...）
    from data_generation_test.parameter import (
        ENTRY_NODE,
        EXIT_NODE,
        BusinessParameters,
        NodeId,
    )
    from data_generation_test.candidate_network import (
        DEFAULT_OUTPUT_PATH as DEFAULT_NETWORK_PATH,
        get_feasible_arcs,
        load_candidate_network,
    )
    from data_generation_test.rl_data import (
        MockRLProvider,
        RLProvider,
        RLSignals,
    )
    from src.dayahead_plan import (
        DEFAULT_PLAN_PATH,
        load_dayahead_plan,
    )
except ImportError:  # 直接作为脚本运行：python src/mpc_model.py
    sys.path.insert(0, str(_REPO_ROOT))
    from data_generation_test.parameter import (  # type: ignore
        ENTRY_NODE,
        EXIT_NODE,
        BusinessParameters,
        NodeId,
    )
    from data_generation_test.candidate_network import (  # type: ignore
        DEFAULT_OUTPUT_PATH as DEFAULT_NETWORK_PATH,
        get_feasible_arcs,
        load_candidate_network,
    )
    from data_generation_test.rl_data import (  # type: ignore
        MockRLProvider,
        RLProvider,
        RLSignals,
    )
    from src.dayahead_plan import (  # type: ignore
        DEFAULT_PLAN_PATH,
        load_dayahead_plan,
    )

__all__ = [
    "MPCError",
    "MPCInputError",
    "MPCInfeasibleError",
    "MPCNoSolutionError",
    "UserKey",
    "Arc",
    "ReservationObservation",
    "RandomRequest",
    "FixedSwapEvent",
    "FixedCommitment",
    "RollingState",
    "MPCWindowInput",
    "FirstStageExecution",
    "MPCResult",
    "MPCModelBundle",
    "MPCController",
]

UserKey = Tuple[int, int]
"""预约用户复合键 ``(od_id, user_id)``，对应论文索引 ``(p,k)``。"""

Arc = Tuple[NodeId, NodeId]

_EPS = 1e-9

# 事件类别
_EVENT_DEC = "dec"  # 本轮可调整预约（激活量 = y）
_EVENT_FIX = "fix"  # 已发布固定承诺（恒激活）
_EVENT_RAND = "rand"  # 预测随机请求（激活量 = z）


# ----------------------------------------------------------------------
# 异常
# ----------------------------------------------------------------------
class MPCError(RuntimeError):
    """MPC 求解相关异常基类。"""


class MPCInputError(ValueError):
    """MPC 窗口输入不合法。"""


class MPCInfeasibleError(MPCError):
    """模型在预约违约松弛后仍不可行；iis_constraints 列出 IIS 约束名。"""

    def __init__(self, message: str, iis_constraints: Optional[List[str]] = None):
        self.iis_constraints = list(iis_constraints or [])
        if self.iis_constraints:
            shown = self.iis_constraints[:50]
            suffix = "..." if len(self.iis_constraints) > 50 else ""
            message = (
                f"{message}；IIS 约束（{len(self.iis_constraints)} 个）: "
                f"{shown}{suffix}"
            )
        super().__init__(message)


class MPCNoSolutionError(MPCError):
    """求解结束但无可行 incumbent（如时限内未找到解）。"""


# ----------------------------------------------------------------------
# 公开输入 dataclass
# ----------------------------------------------------------------------
@dataclass
class ReservationObservation:
    """本轮参与路径优化的预约用户（论文 K_ell^p = K_arr ∪ K_fut）。

    effective_entry_time / effective_entry_soc 为有效入口信息
    （式 eq:effective_arrival_information）：本轮新进入用户取实际值
    (t_A, soc_A)，未来用户取日前值 (bar_t_A, bar_soc_A)。
    """

    od_id: int
    user_id: int
    effective_entry_time: float  # 有效入口时刻（小时）
    effective_entry_soc: float  # 有效入口 SOC
    baseline_path_arcs: List[Arc]  # 日前基准路径 ȳ（[from, to] 弧列表）
    is_new_arrival: bool  # True=K_arr（本轮新进入，求解后发布路径）

    @property
    def user_key(self) -> UserKey:
        """返回论文 ``(p,k)`` 对应的 ``(od_id, user_id)`` 复合键。"""
        return (self.od_id, self.user_id)


@dataclass
class RandomRequest:
    """预测随机请求（论文 R_{i,q} 的元素，MPC 中取预测值）。"""

    station: int
    period: int  # 预测到站所属时段 q = [t_R]（绝对时段索引）
    request_id: str
    arrival_time: float  # 预测到站时刻（小时，落在 [q, q+1)）
    arrival_soc: float  # 预测到站 SOC（即退回 SOC）


@dataclass
class FixedSwapEvent:
    """固定承诺的剩余待履约换电事件。"""

    station: int
    period: int  # 到站时段 q_{A,i}^{fix}（绝对时段索引，>= ell）
    return_soc: float  # 退回 SOC（按式 eq:return_soc 预先计算）


@dataclass
class FixedCommitment:
    """已发布在途固定承诺（论文 K_ell^{fix,p}）：路径不再调整。"""

    od_id: int
    user_id: int
    fixed_path_arcs: List[Arc]  # 已发布固定路径 y^{fix}
    remaining_events: List[FixedSwapEvent]  # 尚未完成的换电事件

    @property
    def user_key(self) -> UserKey:
        """返回论文 ``(p,k)`` 对应的 ``(od_id, user_id)`` 复合键。"""
        return (self.od_id, self.user_id)


@dataclass
class RollingState:
    """当前观测状态。"""

    soc_obs: List[List[float]]  # S^obs[ell-1]，形状 [站][槽]
    period_ell: int  # 当前滚动决策时刻 ell


@dataclass
class MPCWindowInput:
    """单轮 MPC 窗口的全部输入。"""

    params: BusinessParameters
    rolling_state: RollingState
    reservations: List[ReservationObservation] = field(default_factory=list)
    random_requests: List[RandomRequest] = field(default_factory=list)
    fixed_commitments: List[FixedCommitment] = field(default_factory=list)
    rl_signals: Optional[RLSignals] = None


@dataclass
class FirstStageExecution:
    """首阶段（时段 ell）执行包：唯一被实际执行的部分。"""

    period: int  # = ell
    power_kw: List[List[float]]  # P_act[i][b]（式 eq:actual_first_step_charging）
    ready: List[List[int]]  # g*[i][b] 服务就绪指示
    available_full: List[int]  # F*[i] 可用满电电池数
    assignments: List[Dict[str, Any]]  # 事件-槽匹配（按 站, 槽 升序）


@dataclass
class MPCResult:
    """单轮 MPC 求解结果。

    SOC 轨迹索引：soc[i][b][h] 为时段 ell+h 换电后的槽位 SOC，
    h=0..H-1；terminal_soc[i][b] 即终端状态 S[ell+H]。
    """

    period_ell: int
    horizon: int
    status: str  # Gurobi 状态名（OPTIMAL / TIME_LIMIT / ...）
    is_optimal: bool
    solve_time_sec: float
    # ---- 目标分项（主目标 = I^A + I^R - C_ch - C_adj - C_fail
    #                         + beta*Phi^RL） ----
    objective_total: float
    income_reservation: float  # I^A
    income_random: float  # I^R
    charging_cost: float  # C_ch
    adjustment_cost: float  # C_adj
    reservation_failure_cost: float  # C_fail
    terminal_value: float  # Phi^RL（未乘 beta）
    terminal_value_weight: float  # beta
    # ---- 路径决策 ----
    paths: Dict[UserKey, List[Arc]]  # 全部决策用户的预测路径（(p,k) -> 弧序）
    publish_paths: Dict[UserKey, List[Arc]]  # 仅本轮新到用户的待发布路径
    path_adjusted: Dict[UserKey, int]  # d_{p,k}
    # ---- 解值 ----
    y: Dict[Tuple[UserKey, NodeId, NodeId], int]  # ((od_id,user_id), from, to)
    z: Dict[str, int]  # request_id -> 是否服务
    reservation_active: Dict[int, int]  # 预约 event_id -> 是否仍有效到达
    reservation_served: Dict[int, int]  # 预约 event_id -> 站内库存是否履约
    reservation_failed: Dict[int, int]  # 预约 event_id -> 是否发生预测违约
    alpha: Dict[Tuple[int, int, int, int], int]  # (站, 槽, event_id, h)
    power: List[List[List[float]]]  # P[i][b][h]
    ready: List[List[List[int]]]  # g[i][b][h]
    available: List[List[int]]  # F[i][h]
    soc_pre: List[List[List[float]]]  # S^pre[i][b][h]
    soc: List[List[List[float]]]  # S[i][b][h]，h=0..H-1
    terminal_soc: List[List[float]]  # S[ell+H]
    events: List[Dict[str, Any]]  # 全部换电事件清单（含 event_id）
    first_stage: FirstStageExecution


# ----------------------------------------------------------------------
# 内部结构
# ----------------------------------------------------------------------
@dataclass
class _SwapEvent:
    """窗口内换电事件（Constraint 4 的统一表示）。"""

    event_id: int
    kind: str  # _EVENT_DEC / _EVENT_FIX / _EVENT_RAND
    station: int
    h: int  # 窗口内偏移 h = q - ell
    period: int  # 绝对时段 q
    return_soc: float
    sort_key: Tuple  # 规范化事件顺序排序键
    user_key: Optional[UserKey] = None  # dec / fix；论文复合索引 (p,k)
    arc: Optional[Arc] = None  # dec: 入弧 (j, i)
    request_id: Optional[str] = None  # rand
    arrival_time: Optional[float] = None  # rand

    def describe(self) -> Dict[str, Any]:
        od_id = self.user_key[0] if self.user_key is not None else None
        user_id = self.user_key[1] if self.user_key is not None else None
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "station": self.station,
            "period": self.period,
            "return_soc": self.return_soc,
            # JSON 不保留 tuple 类型；输出层统一为 list，确保保存/重载后
            # 与内存结果逐字段相等。
            "user_key": (
                list(self.user_key) if self.user_key is not None else None
            ),
            "od_id": od_id,
            "user_id": user_id,
            "arc": list(self.arc) if self.arc is not None else None,
            "request_id": self.request_id,
            "arrival_time": self.arrival_time,
        }


@dataclass
class _DecUser:
    """决策预约用户的建模上下文。"""

    obs: ReservationObservation
    od_index: int
    arcs: List[Arc]  # 个体可行弧集 A^{p,k}
    baseline_visits: Dict[int, int]  # bar_x：基准路径的站点访问指示

    @property
    def user_key(self) -> UserKey:
        """决策用户的论文复合索引 ``(p,k)``。"""
        return self.obs.user_key


@dataclass
class MPCModelBundle:
    """build_model 的返回：模型与全部变量/表达式句柄（供提取结果）。

    一般无需直接使用；``solve_step`` 内部调用 ``build_model`` 并完成求解
    与结果提取。事件清单见 ``events``，目标分项线性表达式见
    ``obj_parts``（求解后可 ``.getValue()``）。
    """

    model: gp.Model
    window: MPCWindowInput
    horizon: int
    dec_users: List[_DecUser]
    events: List[_SwapEvent]
    events_at: Dict[Tuple[int, int], List[_SwapEvent]]  # (站, h) -> 有序事件
    out_events: List[Tuple[UserKey, Arc, float]]  # 域外事件 ((p,k), 入弧, rho)
    fixed_out_events: List[Tuple[UserKey, int, float]]
    y: Dict[Tuple[UserKey, Arc], gp.Var]
    d: Dict[UserKey, gp.Var]
    z: Dict[str, gp.Var]
    reservation_active: Dict[int, gp.Var]
    reservation_served: Dict[int, gp.Var]
    dec_boundary_alive: Dict[UserKey, gp.Var]
    fixed_boundary_alive: Dict[UserKey, gp.Var]
    out_event_active: Dict[Tuple[UserKey, Arc], gp.Var]
    residual_full: Dict[Tuple[int, int], gp.Var]
    reservation_shortage: Dict[Tuple[int, int], gp.Var]
    alpha: Dict[Tuple[int, int, int, int], gp.Var]
    g: Dict[Tuple[int, int, int], gp.Var]
    F: Dict[Tuple[int, int], gp.Var]
    P: Dict[Tuple[int, int, int], gp.Var]
    S_pre: Dict[Tuple[int, int, int], gp.Var]
    S: Dict[Tuple[int, int, int], gp.Var]  # h = -1..H-1（h=-1 为固定初始态）
    obj_parts: Dict[str, gp.LinExpr]


# ----------------------------------------------------------------------
# MPC 控制器
# ----------------------------------------------------------------------
class MPCController:
    """逐电池槽位 MPC 控制器（论文 2.2.4 节优化模型）。

    参数
    ----
    params : BusinessParameters
        业务参数对象（data_generation_test.parameter）。
    candidate_network : dict
        离线候选网络（candidate_network.generate/load 的输出）。
    rl_provider : RLProvider, optional
        RL 信号提供者；缺省 MockRLProvider(params)。
    dayahead_plan : dict, optional
        日前计划（dayahead_plan.load_dayahead_plan 的输出），用于
        ``make_reservation_observation`` 查询 od_id 与基准路径。
    """

    def __init__(
        self,
        params: BusinessParameters,
        candidate_network: dict,
        rl_provider: Optional[RLProvider] = None,
        dayahead_plan: Optional[dict] = None,
    ) -> None:
        params.validate()
        self.params = params
        self.network = candidate_network
        self.rl_provider = (
            rl_provider if rl_provider is not None else MockRLProvider(params)
        )
        self.dayahead_plan = dayahead_plan

    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        params: BusinessParameters,
        candidate_network_path: Any = DEFAULT_NETWORK_PATH,
        dayahead_plan_path: Any = DEFAULT_PLAN_PATH,
        rl_provider: Optional[RLProvider] = None,
    ) -> "MPCController":
        """从 JSON 文件构造控制器。dayahead_plan_path 可为 None。"""
        network = load_candidate_network(candidate_network_path)
        plan = (
            load_dayahead_plan(dayahead_plan_path)
            if dayahead_plan_path is not None
            else None
        )
        return cls(params, network, rl_provider=rl_provider, dayahead_plan=plan)

    # ------------------------------------------------------------------
    def _plan_record(self, user_id: int, *, od_id: Optional[int] = None) -> dict:
        """按 ``(od_id, user_id)`` 查询日前记录。

        ``od_id`` 省略时保留旧接口行为，但只允许 ``user_id`` 在日前计划中
        全局唯一；若不同 O-D 复用了同一编号，调用者必须显式传入 ``od_id``。
        """
        if self.dayahead_plan is None:
            raise MPCInputError("未加载日前计划，无法按 (od_id, user_id) 查询基准路径")
        matches = [
            rec
            for rec in self.dayahead_plan["reservations"]
            if rec["reservation_id"] == user_id
            and (od_id is None or rec["od_id"] == od_id)
        ]
        if not matches:
            scope = f"od_id={od_id}, " if od_id is not None else ""
            raise MPCInputError(
                f"日前计划中不存在 {scope}reservation_id={user_id}"
            )
        if len(matches) > 1:
            od_ids = sorted(rec["od_id"] for rec in matches)
            raise MPCInputError(
                f"reservation_id={user_id} 在多个 O-D {od_ids} 中重复；"
                "请显式传入 od_id"
            )
        rec = matches[0]
        if not rec["accepted"]:
            raise MPCInputError(
                f"预约 ({rec['od_id']}, {user_id}) 日前已被拒绝，无基准路径"
            )
        return rec

    def baseline_arcs_of(
        self, user_id: int, *, od_id: Optional[int] = None
    ) -> List[Arc]:
        """日前基准路径弧集 ȳ；重复 user_id 时必须给出 ``od_id``。"""
        rec = self._plan_record(user_id, od_id=od_id)
        return [tuple(a) for a in rec["path_arcs"]]

    def make_reservation_observation(
        self,
        user_id: int,
        effective_entry_time: float,
        effective_entry_soc: float,
        is_new_arrival: bool,
        *,
        od_id: Optional[int] = None,
    ) -> ReservationObservation:
        """按日前记录构造观测；重复 user_id 时必须给出 ``od_id``。"""
        rec = self._plan_record(user_id, od_id=od_id)
        return ReservationObservation(
            od_id=rec["od_id"],
            user_id=user_id,
            effective_entry_time=float(effective_entry_time),
            effective_entry_soc=float(effective_entry_soc),
            baseline_path_arcs=[tuple(a) for a in rec["path_arcs"]],
            is_new_arrival=bool(is_new_arrival),
        )

    # ------------------------------------------------------------------
    # 输入校验
    # ------------------------------------------------------------------
    def _od_index_of(self, od_id: int) -> int:
        for idx, od in enumerate(self.params.od_pairs):
            if od.od_id == od_id:
                return idx
        raise MPCInputError(f"参数中不存在 od_id={od_id}")

    def _validate_window(self, window: MPCWindowInput) -> None:
        p = self.params
        st = p.station
        ell = window.rolling_state.period_ell
        H = p.horizon
        if window.rl_signals is None:
            raise MPCInputError("rl_signals 不能为空")
        if not 0 <= ell <= p.num_periods - 1:
            raise MPCInputError(
                f"滚动时刻 ell={ell} 超出 [0, {p.num_periods - 1}]"
            )
        soc_obs = window.rolling_state.soc_obs
        if len(soc_obs) != st.num_stations or any(
            len(row) != st.num_slots for row in soc_obs
        ):
            raise MPCInputError(
                f"soc_obs 形状应为 [{st.num_stations}][{st.num_slots}]"
            )
        for i, row in enumerate(soc_obs):
            for b, s in enumerate(row):
                if not 0.0 <= s <= 1.0:
                    raise MPCInputError(f"soc_obs[{i}][{b}]={s} 超出 [0,1]")

        sig = window.rl_signals
        if sig.start_period != ell:
            raise MPCInputError(
                f"rl_signals.start_period={sig.start_period} 与 ell={ell} 不一致"
            )
        if sig.horizon != H:
            raise MPCInputError(
                f"rl_signals.horizon={sig.horizon} 与 params.horizon={H} 不一致"
            )
        rp = sig.requested_power
        if len(rp) != st.num_stations or any(
            len(rp[i]) != st.num_slots for i in range(st.num_stations)
        ):
            raise MPCInputError("requested_power 形状应为 [站][槽][H]")
        p_tol = p.full_power_tolerance_kw()
        request_slot_cap = st.slot_power_limit_kw - p_tol
        request_station_cap = st.station_power_limit_kw - st.num_slots * p_tol
        for i in range(st.num_stations):
            for b in range(st.num_slots):
                if len(rp[i][b]) != H:
                    raise MPCInputError(
                        f"requested_power[{i}][{b}] 长度应为 H={H}"
                    )
                for h, v in enumerate(rp[i][b]):
                    if not -_EPS <= v <= request_slot_cap + 1e-8:
                        raise MPCInputError(
                            f"requested_power[{i}][{b}][{h}]={v} 超出 "
                            f"[0, 预留补足裕量后的槽位上限 {request_slot_cap}]"
                        )
            for h in range(H):
                requested_total = sum(rp[i][b][h] for b in range(st.num_slots))
                if requested_total > request_station_cap + 1e-8:
                    raise MPCInputError(
                        f"requested_power 站 {i} 时域步 {h} 合计 "
                        f"{requested_total} kW 超过预留补足裕量后的站级上限 "
                        f"{request_station_cap} kW"
                    )
        if len(sig.terminal_soc_value) != st.num_stations or any(
            len(sig.terminal_soc_value[i]) != st.num_slots
            for i in range(st.num_stations)
        ):
            raise MPCInputError("terminal_soc_value 形状应为 [站][槽]")
        if len(sig.outside_swap_lambda) != st.num_stations:
            raise MPCInputError("outside_swap_lambda 长度应为站数")

        seen_users: set[UserKey] = set()
        for obs in window.reservations:
            key = obs.user_key
            if key in seen_users:
                raise MPCInputError(f"决策预约用户 {key} 重复")
            seen_users.add(key)
            self._od_index_of(obs.od_id)
            if obs.effective_entry_time < -_EPS:
                raise MPCInputError(
                    f"用户 {key} 有效入口时刻 "
                    f"{obs.effective_entry_time} 为负"
                )
            if not 0.0 <= obs.effective_entry_soc <= 1.0:
                raise MPCInputError(
                    f"用户 {key} 有效入口 SOC "
                    f"{obs.effective_entry_soc} 超出 [0,1]"
                )

        seen_req = set()
        for req in window.random_requests:
            if not 0 <= req.station < st.num_stations:
                raise MPCInputError(f"随机请求 {req.request_id} 站点非法")
            if req.request_id in seen_req:
                raise MPCInputError(f"随机请求 id {req.request_id!r} 重复")
            seen_req.add(req.request_id)
            if not 0.0 <= req.arrival_soc <= 1.0:
                raise MPCInputError(
                    f"随机请求 {req.request_id} 到站 SOC 超出 [0,1]"
                )

        for fc in window.fixed_commitments:
            key = fc.user_key
            if key in seen_users:
                raise MPCInputError(f"预约用户复合键 {key} 在窗口中重复")
            seen_users.add(key)
            self._od_index_of(fc.od_id)
            for ev in fc.remaining_events:
                if not 0 <= ev.station < st.num_stations:
                    raise MPCInputError(
                        f"固定承诺用户 {key} 的事件站点非法"
                    )
                if ev.period < ell:
                    raise MPCInputError(
                        f"固定承诺用户 {key} 在站 {ev.station} 的事件"
                        f"时段 {ev.period} 早于当前 ell={ell}（应已履约）"
                    )
                if not 0.0 <= ev.return_soc <= 1.0:
                    raise MPCInputError(
                        f"固定承诺用户 {key} 退回 SOC 超出 [0,1]"
                    )

    # ------------------------------------------------------------------
    # 事件构造（Constraint 4，式 eq:swap_event_sets）
    # ------------------------------------------------------------------
    def _node_pos(self, od_index: int, node: NodeId) -> float:
        return self.params.node_position_km(od_index, node)

    def _build_events(
        self, window: MPCWindowInput
    ) -> Tuple[
        List[_DecUser],
        List[_SwapEvent],
        List[Tuple[UserKey, Arc, float]],
        List[Tuple[UserKey, int, float]],
    ]:
        """构造决策用户上下文、窗口内事件及决策/固定域外事件。"""
        p = self.params
        ell = window.rolling_state.period_ell
        H = p.horizon
        delta = p.delta_hours

        dec_users: List[_DecUser] = []
        events: List[_SwapEvent] = []
        out_events: List[Tuple[UserKey, Arc, float]] = []
        fixed_out_events: List[Tuple[UserKey, int, float]] = []

        # ---- 决策预约用户（dec）：个体可行弧集 + 域内/域外事件 ----
        for obs in window.reservations:
            od_index = self._od_index_of(obs.od_id)
            try:
                arcs = get_feasible_arcs(
                    self.network, od_index, obs.effective_entry_soc
                )
            except ValueError as exc:
                raise MPCInputError(
                    f"用户 {obs.user_id}（O-D {obs.od_id}）在有效入口 SOC="
                    f"{obs.effective_entry_soc} 下无可行路径: {exc}"
                ) from exc
            baseline_visits: Dict[int, int] = {}
            for a in obs.baseline_path_arcs:
                if isinstance(a[1], int):
                    baseline_visits[a[1]] = 1
            du = _DecUser(
                obs=obs,
                od_index=od_index,
                arcs=arcs,
                baseline_visits=baseline_visits,
            )
            dec_users.append(du)
            user_key = du.user_key

            t_periods = obs.effective_entry_time / delta
            for (j, i) in arcs:
                if not isinstance(i, int):
                    continue  # 终点为出口的弧不产生换电事件
                v = p.soc_consumption(od_index, j, i)
                rho = (obs.effective_entry_soc if j == ENTRY_NODE else 1.0) - v
                q = p.arrival_period(od_index, i, t_periods)
                if q >= ell + H:
                    # 域外事件：仅进入终端价值第二项。
                    out_events.append((user_key, (j, i), rho))
                    continue
                q = max(q, ell)  # q < ell 的离散化钳位（见模块文档）
                events.append(
                    _SwapEvent(
                        event_id=-1,
                        kind=_EVENT_DEC,
                        station=i,
                        h=q - ell,
                        period=q,
                        return_soc=rho,
                        sort_key=(
                            q,
                            1 if obs.is_new_arrival else 2,
                            obs.od_id,
                            obs.user_id,
                            self._node_pos(od_index, j),
                            str(j),
                        ),
                        user_key=user_key,
                        arc=(j, i),
                    )
                )

        # ---- 固定承诺（fix）：恒激活 ----
        for fc in window.fixed_commitments:
            od_index = self._od_index_of(fc.od_id)
            user_key = fc.user_key
            in_arc_from: Dict[int, NodeId] = {}
            for a in fc.fixed_path_arcs:
                if isinstance(a[1], int):
                    in_arc_from[a[1]] = a[0]
            for fev in fc.remaining_events:
                q = fev.period
                if q >= ell + H:
                    # 域外固定事件正常履约时已含于参考状态；若用户在域内
                    # 失败，目标中需撤销该参考价值。
                    fixed_out_events.append(
                        (user_key, fev.station, fev.return_soc)
                    )
                    continue
                j = in_arc_from.get(fev.station, ENTRY_NODE)
                events.append(
                    _SwapEvent(
                        event_id=-1,
                        kind=_EVENT_FIX,
                        station=fev.station,
                        h=q - ell,
                        period=q,
                        return_soc=fev.return_soc,
                        sort_key=(
                            q,
                            0,
                            fc.od_id,
                            fc.user_id,
                            self._node_pos(od_index, j),
                            str(j),
                        ),
                        user_key=user_key,
                        arc=(j, fev.station),
                    )
                )

        # ---- 预测随机请求（rand）：窗口外请求忽略 ----
        for req in window.random_requests:
            q = req.period
            if not ell <= q <= ell + H - 1:
                continue
            events.append(
                _SwapEvent(
                    event_id=-1,
                    kind=_EVENT_RAND,
                    station=req.station,
                    h=q - ell,
                    period=q,
                    return_soc=req.arrival_soc,
                    sort_key=(q, req.arrival_time, req.request_id),
                    request_id=req.request_id,
                    arrival_time=req.arrival_time,
                )
            )

        # ---- 固定事件顺序：fixed > new dec > future dec，再按
        # (q, O-D, 用户, 入弧) 排序；随机请求始终在预约之后。 ----
        events.sort(key=lambda e: (e.kind == _EVENT_RAND, e.sort_key))
        for eid, ev in enumerate(events):
            ev.event_id = eid

        return dec_users, events, out_events, fixed_out_events

    # ------------------------------------------------------------------
    # 价格（q 超出运营日时按最后时段延展，见模块文档）
    # ------------------------------------------------------------------
    def _e_price(self, i: int, q: int) -> float:
        row = self.params.electricity_price[i]
        return row[q] if q < len(row) else row[-1]

    def _pi_price(self, i: int, q: int) -> float:
        row = self.params.swap_service_price[i]
        return row[q] if q < len(row) else row[-1]

    # ------------------------------------------------------------------
    # 建模
    # ------------------------------------------------------------------
    def build_model(self, window: MPCWindowInput) -> MPCModelBundle:
        """按论文 2.2.4 节完整构建本轮 MPC 的 Gurobi 模型（不求解）。"""
        self._validate_window(window)
        p = self.params
        st = p.station
        n_sta, n_slot = st.num_stations, st.num_slots
        ell = window.rolling_state.period_ell
        H = p.horizon
        delta = p.delta_hours
        eta = st.charging_efficiency
        e_b = p.battery_capacity_kwh
        soc_obs = window.rolling_state.soc_obs
        sig = window.rl_signals
        coeff = e_b / (eta * delta)  # P_need = coeff * (1 - S_prev)
        charge_coef = eta * delta / e_b  # S_pre = S_prev + charge_coef * P
        eps_soc = p.full_soc_tolerance

        dec_users, events, out_events, fixed_out_events = self._build_events(
            window
        )
        events_at: Dict[Tuple[int, int], List[_SwapEvent]] = {}
        for ev in events:
            events_at.setdefault((ev.station, ev.h), []).append(ev)

        model = gp.Model(f"MPC_ell{ell}")
        model.Params.Threads = p.solver.threads
        model.Params.TimeLimit = p.solver.time_limit_sec
        model.Params.OutputFlag = p.solver.output_flag
        # 必须明显小于 eps_soc，避免求解容差重新放行“S_pre=1 但 g=0”。
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        model.ModelSense = GRB.MAXIMIZE

        # ---------------- 变量 ----------------
        # y：预约路径（式 eq:binary_variables，仅建在个体可行弧集 A^{p,k} 上）
        y: Dict[Tuple[UserKey, Arc], gp.Var] = {}
        for du in dec_users:
            user_key = du.user_key
            for a in du.arcs:
                y[(user_key, a)] = model.addVar(
                    vtype=GRB.BINARY,
                    name=f"y[{user_key[0]},{user_key[1]},{a[0]},{a[1]}]",
                )
        # d：路径调整指示
        d: Dict[UserKey, gp.Var] = {
            du.user_key: model.addVar(
                vtype=GRB.BINARY,
                name=f"d[{du.user_key[0]},{du.user_key[1]}]",
            )
            for du in dec_users
        }
        # z：预测随机请求服务指示
        z: Dict[str, gp.Var] = {}
        for ev in events:
            if ev.kind == _EVENT_RAND:
                z[ev.request_id] = model.addVar(
                    vtype=GRB.BINARY, name=f"z[{ev.request_id}]"
                )
        # 预约事件由站内物理库存履约的指示。未能匹配满电池的激活预约
        # 计为显式预测违约并进入目标成本，而不是令整个窗口不可行。
        reservation_active: Dict[int, gp.Var] = {
            ev.event_id: model.addVar(
                vtype=GRB.BINARY, name=f"res_active[{ev.event_id}]"
            )
            for ev in events
            if ev.kind in (_EVENT_DEC, _EVENT_FIX)
        }
        reservation_served: Dict[int, gp.Var] = {
            ev.event_id: model.addVar(
                vtype=GRB.BINARY, name=f"res_served[{ev.event_id}]"
            )
            for ev in events
            if ev.kind in (_EVENT_DEC, _EVENT_FIX)
        }
        # 只有成功存活到预测边界的预约，才会发生域外换电。决策预约的
        # out_event_active = y AND boundary_alive；固定承诺的边界存活
        # 用于在域内失败时撤销参考轨迹已包含的域外价值。
        dec_boundary_alive: Dict[UserKey, gp.Var] = {
            user_key: model.addVar(
                vtype=GRB.BINARY,
                name=f"dec_boundary_alive[{user_key[0]},{user_key[1]}]",
            )
            for user_key in sorted({item[0] for item in out_events})
        }
        fixed_boundary_alive: Dict[UserKey, gp.Var] = {
            user_key: model.addVar(
                vtype=GRB.BINARY,
                name=f"fix_boundary_alive[{user_key[0]},{user_key[1]}]",
            )
            for user_key in sorted({item[0] for item in fixed_out_events})
        }
        out_event_active: Dict[Tuple[UserKey, Arc], gp.Var] = {
            (user_key, arc): model.addVar(
                vtype=GRB.BINARY,
                name=(
                    f"out_active[{user_key[0]},{user_key[1]},"
                    f"{arc[0]},{arc[1]}]"
                ),
            )
            for user_key, arc, _ in out_events
        }
        # alpha：事件-槽匹配
        alpha: Dict[Tuple[int, int, int, int], gp.Var] = {}
        for (i, h), evs in events_at.items():
            for ev in evs:
                for b in range(n_slot):
                    alpha[(i, b, ev.event_id, h)] = model.addVar(
                        vtype=GRB.BINARY,
                        name=f"alpha[{i},{b},{ev.event_id},{h}]",
                    )
        # g / F：服务就绪指示与可用电池数
        g: Dict[Tuple[int, int, int], gp.Var] = {}
        F: Dict[Tuple[int, int], gp.Var] = {}
        residual_full: Dict[Tuple[int, int], gp.Var] = {}
        reservation_shortage: Dict[Tuple[int, int], gp.Var] = {}
        for i in range(n_sta):
            for h in range(H):
                F[(i, h)] = model.addVar(
                    vtype=GRB.INTEGER, lb=0, ub=n_slot, name=f"F[{i},{h}]"
                )
                residual_full[(i, h)] = model.addVar(
                    vtype=GRB.INTEGER,
                    lb=0,
                    ub=n_slot,
                    name=f"F_after_res[{i},{h}]",
                )
                reservation_shortage[(i, h)] = model.addVar(
                    vtype=GRB.BINARY, name=f"res_shortage[{i},{h}]"
                )
                for b in range(n_slot):
                    g[(i, b, h)] = model.addVar(
                        vtype=GRB.BINARY, name=f"g[{i},{b},{h}]"
                    )
        # P / S_pre / S：充电功率与 SOC（S 含 h=-1 固定初始态）
        P: Dict[Tuple[int, int, int], gp.Var] = {}
        S_pre: Dict[Tuple[int, int, int], gp.Var] = {}
        S: Dict[Tuple[int, int, int], gp.Var] = {}
        for i in range(n_sta):
            for b in range(n_slot):
                # Constraint 6（式 eq:initial_battery_soc）：固定界实现
                S[(i, b, -1)] = model.addVar(
                    lb=soc_obs[i][b],
                    ub=soc_obs[i][b],
                    name=f"S[{i},{b},-1]",
                )
                for h in range(H):
                    P[(i, b, h)] = model.addVar(
                        lb=0.0, name=f"P[{i},{b},{h}]"
                    )
                    S_pre[(i, b, h)] = model.addVar(
                        lb=0.0, ub=1.0, name=f"Spre[{i},{b},{h}]"
                    )
                    S[(i, b, h)] = model.addVar(
                        lb=0.0, ub=1.0, name=f"S[{i},{b},{h}]"
                    )

        # 预约事件有效链：决策用户的首站事件由入弧 y 激活，后续站事件还
        # 必须以前一站已由站内库存履约为前提；固定承诺按剩余事件顺序
        # 逐项传递。这样一次失败不会在下游重复计罚或虚构服务/退回 SOC。
        dec_events_by_user_station: Dict[Tuple[UserKey, int], List[_SwapEvent]] = {}
        fixed_events_by_user: Dict[UserKey, List[_SwapEvent]] = {}
        for ev in events:
            if ev.kind == _EVENT_DEC:
                dec_events_by_user_station.setdefault(
                    (ev.user_key, ev.station), []
                ).append(ev)
            elif ev.kind == _EVENT_FIX:
                fixed_events_by_user.setdefault(ev.user_key, []).append(ev)

        for ev in events:
            if ev.kind != _EVENT_DEC:
                continue
            active = reservation_active[ev.event_id]
            y_in = y[(ev.user_key, ev.arc)]
            previous_node = ev.arc[0]
            if previous_node == ENTRY_NODE:
                model.addConstr(
                    active == y_in,
                    name=f"res_active_entry[{ev.event_id}]",
                )
            else:
                previous_service = gp.quicksum(
                    reservation_served[prev.event_id]
                    for prev in dec_events_by_user_station.get(
                        (ev.user_key, previous_node), []
                    )
                )
                model.addConstr(
                    active <= y_in,
                    name=f"res_active_arc[{ev.event_id}]",
                )
                model.addConstr(
                    active <= previous_service,
                    name=f"res_active_prev_ub[{ev.event_id}]",
                )
                model.addConstr(
                    active >= y_in + previous_service - 1.0,
                    name=f"res_active_prev_lb[{ev.event_id}]",
                )

        for user_key, user_events in fixed_events_by_user.items():
            user_events.sort(key=lambda ev: ev.sort_key)
            for pos, ev in enumerate(user_events):
                rhs = (
                    1.0
                    if pos == 0
                    else reservation_served[user_events[pos - 1].event_id]
                )
                model.addConstr(
                    reservation_active[ev.event_id] == rhs,
                    name=f"res_active_fix[{user_key[0]},{user_key[1]},{pos}]",
                )

        # 预测边界存活：A=1 当且仅当该用户全部域内激活事件均已履约。
        # f_e=active_e-served_e 因 served<=active 且二元而是失败指示。
        # 未选候选事件 active=served=0，不会误杀用户的边界存活状态。
        for kind, alive_vars in (
            (_EVENT_DEC, dec_boundary_alive),
            (_EVENT_FIX, fixed_boundary_alive),
        ):
            for user_key, alive in alive_vars.items():
                user_events = [
                    ev
                    for ev in events
                    if ev.kind == kind and ev.user_key == user_key
                ]
                failures = [
                    reservation_active[ev.event_id]
                    - reservation_served[ev.event_id]
                    for ev in user_events
                ]
                kind_tag = "dec" if kind == _EVENT_DEC else "fix"
                if not failures:
                    model.addConstr(
                        alive == 1,
                        name=(
                            f"{kind_tag}_boundary_no_inner["
                            f"{user_key[0]},{user_key[1]}]"
                        ),
                    )
                    continue
                model.addConstr(
                    alive >= 1.0 - gp.quicksum(failures),
                    name=(
                        f"{kind_tag}_boundary_lb["
                        f"{user_key[0]},{user_key[1]}]"
                    ),
                )
                for pos, failure in enumerate(failures):
                    model.addConstr(
                        alive <= 1.0 - failure,
                        name=(
                            f"{kind_tag}_boundary_ub["
                            f"{user_key[0]},{user_key[1]},{pos}]"
                        ),
                    )

        # w_out = y_out AND A_boundary：路径弧虽已选择，但若用户在域内
        # 首次违约，其所有域外换电都不会发生，也不应继续获得/承担价值。
        for user_key, arc, _ in out_events:
            active = out_event_active[(user_key, arc)]
            y_out = y[(user_key, arc)]
            alive = dec_boundary_alive[user_key]
            model.addConstr(
                active <= y_out,
                name=(
                    f"out_active_y_ub[{user_key[0]},{user_key[1]},"
                    f"{arc[0]},{arc[1]}]"
                ),
            )
            model.addConstr(
                active <= alive,
                name=(
                    f"out_active_alive_ub[{user_key[0]},{user_key[1]},"
                    f"{arc[0]},{arc[1]}]"
                ),
            )
            model.addConstr(
                active >= y_out + alive - 1.0,
                name=(
                    f"out_active_lb[{user_key[0]},{user_key[1]},"
                    f"{arc[0]},{arc[1]}]"
                ),
            )

        # ---------------- Constraint 1：流平衡（式 eq:flow） ----------------
        for du in dec_users:
            user_key = du.user_key
            out_arcs: Dict[NodeId, List[Arc]] = {}
            in_arcs: Dict[NodeId, List[Arc]] = {}
            for a in du.arcs:
                out_arcs.setdefault(a[0], []).append(a)
                in_arcs.setdefault(a[1], []).append(a)
            for n in p.od_nodes(du.od_index):
                expr = gp.quicksum(
                    y[(user_key, a)] for a in out_arcs.get(n, [])
                )
                expr -= gp.quicksum(
                    y[(user_key, a)] for a in in_arcs.get(n, [])
                )
                rhs = (
                    1.0
                    if n == ENTRY_NODE
                    else (-1.0 if n == EXIT_NODE else 0.0)
                )
                model.addConstr(
                    expr == rhs,
                    name=f"flow[{user_key[0]},{user_key[1]},{n}]",
                )

        # ---------------- Constraint 2：路径调整指示 ----------------
        # （式 eq:station_visit_indicator / eq:path_adjustment_indicator）
        for du in dec_users:
            user_key = du.user_key
            od = p.od_pairs[du.od_index]
            for i in od.station_indices:
                x_i = gp.quicksum(
                    y[(user_key, a)] for a in du.arcs if a[1] == i
                )
                bar_x = float(du.baseline_visits.get(i, 0))
                model.addConstr(
                    d[user_key] >= x_i - bar_x,
                    name=f"adj_pos[{user_key[0]},{user_key[1]},{i}]",
                )
                model.addConstr(
                    d[user_key] >= bar_x - x_i,
                    name=f"adj_neg[{user_key[0]},{user_key[1]},{i}]",
                )

        # ---------------- Constraint 3：功率截断与充电转移 ----------------
        # （式 eq:power_saturation / eq:continuous_charging_transition）
        for i in range(n_sta):
            for b in range(n_slot):
                for h in range(H):
                    p_hat = float(sig.requested_power[i][b][h])
                    gv = g[(i, b, h)]
                    Pv = P[(i, b, h)]
                    S_prev = S[(i, b, h - 1)]
                    # g=0 => P=P_hat 且 P_hat <= P_need-p_tol；因此
                    # S_pre <= 1-eps_soc，物理上未满。
                    model.addGenConstrIndicator(
                        gv, 0, Pv == p_hat, name=f"pow_defer_eq[{i},{b},{h}]"
                    )
                    model.addGenConstrIndicator(
                        gv,
                        0,
                        S_prev + charge_coef * p_hat <= 1.0 - eps_soc,
                        name=f"pow_defer_le[{i},{b},{h}]",
                    )
                    # g=1 => P=P_need 且 P_hat >= P_need-p_tol；容差内微量
                    # 补足到 SOC 1，并强制开放服务。
                    model.addGenConstrIndicator(
                        gv,
                        1,
                        Pv + coeff * S_prev == coeff,
                        name=f"pow_fill_eq[{i},{b},{h}]",
                    )
                    model.addGenConstrIndicator(
                        gv,
                        1,
                        S_prev + charge_coef * p_hat >= 1.0 - eps_soc,
                        name=f"pow_fill_ge[{i},{b},{h}]",
                    )
                    # 充电转移：S_pre = S_prev + (eta*Delta/E_B) * P
                    model.addConstr(
                        S_pre[(i, b, h)] == S_prev + charge_coef * Pv,
                        name=f"charge[{i},{b},{h}]",
                    )
                    # 服务就绪等价关系（式 eq:service_ready_relation）：
                    # g=1 => S_pre=1；g=0 => S_pre<=1-eps_soc。
                    model.addConstr(
                        gv <= S_pre[(i, b, h)],
                        name=f"ready_rel_lb[{i},{b},{h}]",
                    )
                    model.addConstr(
                        S_pre[(i, b, h)] <= 1.0 - eps_soc * (1.0 - gv),
                        name=f"ready_rel_ub[{i},{b},{h}]",
                    )
                    model.addConstr(
                        Pv <= st.slot_power_limit_kw,
                        name=f"slot_power_cap[{i},{b},{h}]",
                    )
        for i in range(n_sta):
            for h in range(H):
                model.addConstr(
                    gp.quicksum(P[(i, b, h)] for b in range(n_slot))
                    <= st.station_power_limit_kw,
                    name=f"station_power_cap[{i},{h}]",
                )
        # F = sum_b g（式 eq:available_battery_count）
        for i in range(n_sta):
            for h in range(H):
                model.addConstr(
                    F[(i, h)]
                    == gp.quicksum(g[(i, b, h)] for b in range(n_slot)),
                    name=f"ready_cnt[{i},{h}]",
                )

        # ---------------- Constraint 4：事件激活、预约违约与服务优先 ----------------
        for i in range(n_sta):
            for h in range(H):
                evs = events_at.get((i, h), [])
                res_evs = [
                    ev for ev in evs
                    if ev.kind in (_EVENT_DEC, _EVENT_FIX)
                ]
                rand_evs = [ev for ev in evs if ev.kind == _EVENT_RAND]

                active_terms = []
                for ev in res_evs:
                    active = reservation_active[ev.event_id]
                    active_terms.append(active)
                    model.addConstr(
                        reservation_served[ev.event_id] <= active,
                        name=f"res_service_active[{i},{ev.event_id},{h}]",
                    )

                q_res = gp.quicksum(active_terms)
                q_res_served = gp.quicksum(
                    reservation_served[ev.event_id] for ev in res_evs
                )
                q_res_failed = q_res - q_res_served
                remain = residual_full[(i, h)]
                shortage = reservation_shortage[(i, h)]

                # q_res_served=min(q_res,F)，remain=max(F-q_res,0)。因此
                # 预约有缺供时所有物理满电池均先给预约，随机余量严格为0。
                model.addConstr(
                    q_res_served + remain == F[(i, h)],
                    name=f"residual_full_def[{i},{h}]",
                )
                if res_evs:
                    model.addConstr(
                        q_res_failed <= len(res_evs) * shortage,
                        name=f"res_shortage_ub[{i},{h}]",
                    )
                    model.addConstr(
                        q_res_failed >= shortage,
                        name=f"res_shortage_lb[{i},{h}]",
                    )
                    model.addConstr(
                        remain <= n_slot * (1 - shortage),
                        name=f"res_priority[{i},{h}]",
                    )
                    # 激活预约按既定事件顺序服务最长前缀；未激活的候选入弧
                    # 不占顺序位置，也不会被计作失败。
                    for later_pos, later in enumerate(res_evs):
                        for earlier in res_evs[:later_pos]:
                            earlier_active = reservation_active[
                                earlier.event_id
                            ]
                            model.addConstr(
                                reservation_served[later.event_id]
                                <= reservation_served[earlier.event_id]
                                + 1.0
                                - earlier_active,
                                name=(
                                    f"res_order[{i},{h},{earlier.event_id},"
                                    f"{later.event_id}]"
                                ),
                            )
                else:
                    model.addConstr(
                        shortage == 0, name=f"res_no_shortage[{i},{h}]"
                    )

                # 随机严格 FCFS 前缀（式 eq:automatic_random_service）；
                # rand_evs 已按 (到站时刻, 请求 id) 排序（见 _build_events）。
                for r, ev in enumerate(rand_evs, start=1):
                    zv = z[ev.request_id]
                    model.addConstr(
                        r * zv <= remain,
                        name=f"fcfs_lb[{i},{h},{ev.request_id}]",
                    )
                    model.addConstr(
                        remain - n_slot * zv <= r - 1,
                        name=f"fcfs_ub[{i},{h},{ev.request_id}]",
                    )

        # ---------------- Constraint 5：事件-槽匹配与 SOC 转移 ----------------
        for (i, h), evs in events_at.items():
            # 5a：每个激活事件恰好分配一块电池
            for ev in evs:
                lhs = gp.quicksum(
                    alpha[(i, b, ev.event_id, h)] for b in range(n_slot)
                )
                if ev.kind == _EVENT_DEC:
                    rhs = reservation_served[ev.event_id]
                elif ev.kind == _EVENT_FIX:
                    rhs = reservation_served[ev.event_id]
                else:
                    rhs = z[ev.request_id]
                model.addConstr(
                    lhs == rhs, name=f"match_event[{i},{ev.event_id},{h}]"
                )
            # 5b：禁止使用未就绪槽；每槽每时段至多服务一次
            for b in range(n_slot):
                model.addConstr(
                    gp.quicksum(
                        alpha[(i, b, ev.event_id, h)] for ev in evs
                    )
                    <= g[(i, b, h)],
                    name=f"match_ready[{i},{b},{h}]",
                )
            # 5c：最小可用槽位（小编号就绪槽不可被跳过）
            for b in range(n_slot):
                for bp in range(b):
                    model.addConstr(
                        gp.quicksum(
                            alpha[(i, b, ev.event_id, h)] for ev in evs
                        )
                        <= gp.quicksum(
                            alpha[(i, bp, ev.event_id, h)] for ev in evs
                        )
                        + 1
                        - g[(i, bp, h)],
                        name=f"match_min_slot[{i},{b},{bp},{h}]",
                    )
            # 5d：无交叉（事件顺序与槽位顺序一致）
            for idx_a in range(len(evs)):
                for idx_c in range(idx_a + 1, len(evs)):
                    ea, ec = evs[idx_a], evs[idx_c]
                    for b in range(n_slot):
                        for bp in range(b):
                            model.addConstr(
                                alpha[(i, b, ea.event_id, h)]
                                + alpha[(i, bp, ec.event_id, h)]
                                <= 1,
                                name=(
                                    f"match_no_cross[{i},{h},"
                                    f"{ea.event_id},{ec.event_id},{b},{bp}]"
                                ),
                            )
        # 换电后 SOC 转移（式 eq:continuous_swap_transition）：对全部
        # (i,b,h) 施加——没有事件的 (站, 时段) 退化为 S = S_pre。
        for i in range(n_sta):
            for h in range(H):
                evs = events_at.get((i, h), [])
                for b in range(n_slot):
                    used_b = gp.quicksum(
                        alpha[(i, b, ev.event_id, h)] for ev in evs
                    )
                    # used=0（全部 alpha=0）=> S = S_pre；用辅助二元变量
                    # used 作指示变量。
                    uv = model.addVar(
                        vtype=GRB.BINARY, name=f"used[{i},{b},{h}]"
                    )
                    model.addConstr(
                        uv == used_b, name=f"used_def[{i},{b},{h}]"
                    )
                    model.addGenConstrIndicator(
                        uv,
                        0,
                        S[(i, b, h)] == S_pre[(i, b, h)],
                        name=f"swap_idle[{i},{b},{h}]",
                    )
                    for ev in evs:
                        model.addGenConstrIndicator(
                            alpha[(i, b, ev.event_id, h)],
                            1,
                            S[(i, b, h)] == ev.return_soc,
                            name=f"swap_fire[{i},{b},{ev.event_id},{h}]",
                        )

        # ---------------- 目标（式 eq:objective 及各分项） ----------------
        # I^A：由站内满电库存实际履约的预约换电收益。允许违约后固定预约
        # 收益也不再是常数，必须同样乘站内履约指示。
        income_a = gp.LinExpr()
        for ev in events:
            if ev.kind in (_EVENT_DEC, _EVENT_FIX):
                income_a += (
                    e_b
                    * self._pi_price(ev.station, ev.period)
                    * (1.0 - ev.return_soc)
                    * reservation_served[ev.event_id]
                )
        # I^R：预测随机服务收益
        income_r = gp.LinExpr()
        for ev in events:
            if ev.kind == _EVENT_RAND:
                income_r += (
                    e_b
                    * self._pi_price(ev.station, ev.period)
                    * (1.0 - ev.return_soc)
                    * z[ev.request_id]
                )
        # C_ch：购电成本（式 eq:chargingcost）
        cost_ch = gp.LinExpr()
        for i in range(n_sta):
            for h in range(H):
                e_iq = self._e_price(i, ell + h)
                for b in range(n_slot):
                    cost_ch += e_iq * delta * P[(i, b, h)]
        # C_adj：路径调整成本（式 eq:adjustment_cost）
        cost_adj = p.path_adjustment_penalty * gp.quicksum(
            d[du.user_key] for du in dec_users
        )
        # C_fail：激活预约事件未能由站内物理满电库存履约的违约成本。
        failure_count = gp.LinExpr()
        for ev in events:
            if ev.kind in (_EVENT_DEC, _EVENT_FIX):
                failure_count += (
                    reservation_active[ev.event_id]
                    - reservation_served[ev.event_id]
                )
        cost_fail = p.reservation_failure_penalty * failure_count
        # Phi^RL：终端 SOC 线性价值 + 域外事件价值（式 eq:rl_terminal_value）
        phi = gp.LinExpr()
        for i in range(n_sta):
            for b in range(n_slot):
                phi += sig.terminal_soc_value[i][b] * S[(i, b, H - 1)]
        for user_key, arc, rho in out_events:
            phi += sig.outside_swap_value(
                arc[1], rho
            ) * out_event_active[
                (user_key, arc)
            ]
        for user_key, station, rho in fixed_out_events:
            # 固定域外事件正常履约时已在参考轨迹中，是被删除的常数；
            # 若用户在域内失败，A_fix=0，此校正恰好撤销该参考价值。
            phi += sig.outside_swap_value(station, rho) * (
                fixed_boundary_alive[user_key] - 1.0
            )

        obj_parts = {
            "income_reservation": income_a,
            "income_random": income_r,
            "charging_cost": cost_ch,
            "adjustment_cost": cost_adj,
            "reservation_failure_cost": cost_fail,
            "terminal_value": phi,
        }
        main_obj = (
            income_a
            + income_r
            - cost_ch
            - cost_adj
            - cost_fail
            + p.terminal_value_weight * phi
        )
        # g 已由物理满电状态硬约束决定，不再用经济目标或次级目标选择。
        model.setObjective(main_obj, GRB.MAXIMIZE)

        return MPCModelBundle(
            model=model,
            window=window,
            horizon=H,
            dec_users=dec_users,
            events=events,
            events_at=events_at,
            out_events=out_events,
            fixed_out_events=fixed_out_events,
            y=y,
            d=d,
            z=z,
            reservation_active=reservation_active,
            reservation_served=reservation_served,
            dec_boundary_alive=dec_boundary_alive,
            fixed_boundary_alive=fixed_boundary_alive,
            out_event_active=out_event_active,
            residual_full=residual_full,
            reservation_shortage=reservation_shortage,
            alpha=alpha,
            g=g,
            F=F,
            P=P,
            S_pre=S_pre,
            S=S,
            obj_parts=obj_parts,
        )

    # ------------------------------------------------------------------
    # 求解与结果提取
    # ------------------------------------------------------------------
    def solve_step(self, window: MPCWindowInput) -> MPCResult:
        """构建并求解本轮 MPC，返回完整结果。

        异常
        ----
        MPCInputError
            输入不合法（形状、范围、未知 od_id、逾期固定承诺等）。
        MPCInfeasibleError
            模型在预约违约松弛后仍不可行；消息含 IIS 约束名。
        MPCNoSolutionError
            求解结束但无可行 incumbent。
        """
        bundle = self.build_model(window)
        model = bundle.model
        t0 = time.perf_counter()
        model.optimize()
        solve_time = time.perf_counter() - t0

        status = model.Status
        status_name = _status_name(status)
        if status == GRB.INFEASIBLE:
            model.computeIIS()
            iis = [c.ConstrName for c in model.getConstrs() if c.IISConstr]
            try:
                iis += [
                    gc.GenConstrName
                    for gc in model.getGenConstrs()
                    if gc.IISGenConstr
                ]
            except (AttributeError, gp.GurobiError):
                pass
            raise MPCInfeasibleError(
                f"MPC 模型不可行（ell={window.rolling_state.period_ell}）",
                iis_constraints=iis,
            )
        if model.SolCount == 0:
            raise MPCNoSolutionError(
                f"求解结束但无可行 incumbent（状态 {status_name}，"
                f"ell={window.rolling_state.period_ell}）"
            )
        return self._extract_result(bundle, status_name, solve_time)

    # ------------------------------------------------------------------
    @staticmethod
    def _order_path(arcs: List[Arc]) -> List[Arc]:
        """把选中的弧整理为 entry -> ... -> exit 的有序路径。"""
        nxt = {f: t for f, t in arcs}
        path: List[Arc] = []
        node: NodeId = ENTRY_NODE
        while node != EXIT_NODE:
            t = nxt[node]
            path.append((node, t))
            node = t
        return path

    def _extract_result(
        self, bundle: MPCModelBundle, status_name: str, solve_time: float
    ) -> MPCResult:
        model = bundle.model
        window = bundle.window
        p = self.params
        st = p.station
        n_sta, n_slot = st.num_stations, st.num_slots
        ell = window.rolling_state.period_ell
        H = bundle.horizon

        def val(v: gp.Var) -> float:
            return float(v.X)

        # ---- 路径 ----
        paths: Dict[UserKey, List[Arc]] = {}
        publish: Dict[UserKey, List[Arc]] = {}
        y_sol: Dict[Tuple[UserKey, NodeId, NodeId], int] = {}
        for du in bundle.dec_users:
            user_key = du.user_key
            chosen = [
                a for a in du.arcs if val(bundle.y[(user_key, a)]) > 0.5
            ]
            for a in du.arcs:
                y_sol[(user_key, a[0], a[1])] = int(
                    round(val(bundle.y[(user_key, a)]))
                )
            path = self._order_path(chosen)
            paths[user_key] = path
            if du.obs.is_new_arrival:
                publish[user_key] = path
        d_sol = {
            du.user_key: int(round(val(bundle.d[du.user_key])))
            for du in bundle.dec_users
        }
        z_sol = {
            rid: int(round(val(v))) for rid, v in bundle.z.items()
        }
        reservation_active_sol = {
            eid: int(round(val(v)))
            for eid, v in bundle.reservation_active.items()
        }
        reservation_served_sol = {
            eid: int(round(val(v)))
            for eid, v in bundle.reservation_served.items()
        }
        reservation_failed_sol: Dict[int, int] = {}
        event_records: List[Dict[str, Any]] = []
        for ev in bundle.events:
            record = ev.describe()
            if ev.kind in (_EVENT_DEC, _EVENT_FIX):
                active = reservation_active_sol[ev.event_id]
                failed = active - reservation_served_sol[ev.event_id]
                reservation_failed_sol[ev.event_id] = failed
                record["reservation_active"] = active
                record["reservation_served"] = reservation_served_sol[
                    ev.event_id
                ]
                record["reservation_failed"] = failed
            event_records.append(record)
        alpha_sol = {
            key: int(round(val(v))) for key, v in bundle.alpha.items()
        }

        power = [
            [
                [val(bundle.P[(i, b, h)]) for h in range(H)]
                for b in range(n_slot)
            ]
            for i in range(n_sta)
        ]
        ready = [
            [
                [int(round(val(bundle.g[(i, b, h)]))) for h in range(H)]
                for b in range(n_slot)
            ]
            for i in range(n_sta)
        ]
        available = [
            [int(round(val(bundle.F[(i, h)]))) for h in range(H)]
            for i in range(n_sta)
        ]
        soc_pre = [
            [
                [val(bundle.S_pre[(i, b, h)]) for h in range(H)]
                for b in range(n_slot)
            ]
            for i in range(n_sta)
        ]
        soc = [
            [
                [val(bundle.S[(i, b, h)]) for h in range(H)]
                for b in range(n_slot)
            ]
            for i in range(n_sta)
        ]
        # 解后硬校验：g 必须与充电后的物理满电状态一致，F 必须等于逐槽
        # g 之和。用 eps_soc 中点区分两支；模型已保证 g=0 至多
        # 1-eps_soc、g=1 等于 1，且求解可行性容差远小于该间隔。
        ready_threshold = 1.0 - 0.5 * p.full_soc_tolerance
        for i in range(n_sta):
            for h in range(H):
                physical_ready = [
                    int(soc_pre[i][b][h] >= ready_threshold)
                    for b in range(n_slot)
                ]
                model_ready = [ready[i][b][h] for b in range(n_slot)]
                if model_ready != physical_ready:
                    raise MPCError(
                        f"求解结果的满电状态与 g 不一致：站 {i}、步 {h}，"
                        f"g={model_ready}，S_pre="
                        f"{[soc_pre[i][b][h] for b in range(n_slot)]}"
                    )
                if available[i][h] != sum(model_ready):
                    raise MPCError(
                        f"求解结果的 F 与 sum(g) 不一致：站 {i}、步 {h}，"
                        f"F={available[i][h]}，sum(g)={sum(model_ready)}"
                    )
        terminal_soc = [[soc[i][b][H - 1] for b in range(n_slot)]
                        for i in range(n_sta)]

        # ---- 首阶段执行包（h = 0，时段 ell） ----
        assignments: List[Dict[str, Any]] = []
        for (i, b, eid, h), a_val in sorted(alpha_sol.items()):
            if h != 0 or a_val != 1:
                continue
            ev = bundle.events[eid]
            od_id = ev.user_key[0] if ev.user_key is not None else None
            user_id = ev.user_key[1] if ev.user_key is not None else None
            assignments.append(
                {
                    "station": i,
                    "slot": b,
                    "event_id": eid,
                    "kind": ev.kind,
                    "user_key": ev.user_key,
                    "od_id": od_id,
                    "user_id": user_id,
                    "request_id": ev.request_id,
                    "return_soc": ev.return_soc,
                }
            )
        assignments.sort(key=lambda a: (a["station"], a["slot"]))
        first_stage = FirstStageExecution(
            period=ell,
            power_kw=[[power[i][b][0] for b in range(n_slot)]
                      for i in range(n_sta)],
            ready=[[ready[i][b][0] for b in range(n_slot)]
                   for i in range(n_sta)],
            available_full=[available[i][0] for i in range(n_sta)],
            assignments=assignments,
        )

        parts = {k: float(expr.getValue())
                 for k, expr in bundle.obj_parts.items()}
        beta = p.terminal_value_weight
        total = (
            parts["income_reservation"]
            + parts["income_random"]
            - parts["charging_cost"]
            - parts["adjustment_cost"]
            - parts["reservation_failure_cost"]
            + beta * parts["terminal_value"]
        )

        return MPCResult(
            period_ell=ell,
            horizon=H,
            status=status_name,
            is_optimal=(model.Status == GRB.OPTIMAL),
            solve_time_sec=solve_time,
            objective_total=float(model.ObjVal),
            income_reservation=parts["income_reservation"],
            income_random=parts["income_random"],
            charging_cost=parts["charging_cost"],
            adjustment_cost=parts["adjustment_cost"],
            reservation_failure_cost=parts["reservation_failure_cost"],
            terminal_value=parts["terminal_value"],
            terminal_value_weight=beta,
            paths=paths,
            publish_paths=publish,
            path_adjusted=d_sol,
            y=y_sol,
            z=z_sol,
            reservation_active=reservation_active_sol,
            reservation_served=reservation_served_sol,
            reservation_failed=reservation_failed_sol,
            alpha=alpha_sol,
            power=power,
            ready=ready,
            available=available,
            soc_pre=soc_pre,
            soc=soc,
            terminal_soc=terminal_soc,
            events=event_records,
            first_stage=first_stage,
        )

    # ------------------------------------------------------------------
    def execute_period(self, result: MPCResult) -> FirstStageExecution:
        """返回首阶段执行包（时段 ell 的功率、就绪指示与事件-槽匹配）。"""
        return result.first_stage


def _status_name(status: int) -> str:
    names = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INTERRUPTED: "INTERRUPTED",
    }
    return names.get(status, f"STATUS_{status}")
