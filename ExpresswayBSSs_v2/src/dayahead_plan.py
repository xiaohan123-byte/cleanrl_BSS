# -*- coding: utf-8 -*-
r"""
日前基准路径生成模块：论文 2.2.3 节“Day-Ahead Baseline Path Generation”的实现。

对应论文 paper/mian11_fixed.tex 第 217-233 行与 plan.md 第 42-53 行。
本模块仅使用 Python 标准库与 data_generation_test 下的三个模块
（parameter / candidate_network / rl_data），不依赖 numpy、pandas 或
Gurobi。

当前公开 ``generate_dayahead_plan`` / ``validate_dayahead_plan`` 已委托给
``src.continuous_dayahead``：它使用共享连续事件内核生成 schema-2、
``continuous_event_v2`` 的六站 synthetic/mock 日前计划。下文保留的离散
库存递推、schema-1 格式和 actor 接入说明只服务于私有 legacy 辅助函数，
不是当前公开 API 的语义。

内容概览
--------
运营日前，预约用户提交 (O-D 对 p, 预计入口时刻 bar_t_A, 预计入口 SOC
bar_soc_A)。本模块据此：
1. 从 mock_data["initial_slot_soc"] 出发，对各站逐槽预测 SOC 库存做
   全运营日递推：每个时段先充电、再按“预约优先、随机 FCFS”顺序服务；
   本时段退回的电池从下一时段起才参与充电。
2. 按提交顺序（reservation_id 升序）逐个事务式试排预约：在个体可行弧集
   A^{p,k}（按预约 SOC 精确过滤，见 candidate_network.get_feasible_arcs）
   上从入口向出口贪心选站，逐步为该用户临时预留满电电池。每次试加候选
   事件后，不仅要求该事件当期库存充足，还要求整条模拟轨迹中的既有预约和
   本请求临时预约全部可履约；任一步无满足库存与下游可达性要求的候选站，
   则拒绝该请求并完整回滚其全部临时预留。
3. 输出日前计划（JSON 可序列化 dict）：每个预约的接受/拒绝结果、拒绝
   原因、日前基准路径（节点序列与弧列表）、各次换电的预计时段与退回 SOC，
   以及各站基准访问量和预测库存轨迹。

日前库存递推（每个时段 q 内的事件顺序）
--------------------------------------
1. 充电：由 rl_provider（缺省 MockRLProvider）逐期给出请求功率
   \widehat P_{i,b,q}，按论文式 (power_saturation) 做满电补足截断：
   补满所需功率 P_need = E_B(1-S)/(eta*Delta)；
   - 若 P_hat >= P_need - p_tol（p_tol 对应统一的满电 SOC 容差，含恰好
     相等及数值上极小的补足缺口）：
     P = P_need，充电后 SOC = 1；
   - 否则：P = P_hat，SOC 增加 eta*Delta*P/E_B，并至少比 1 低
     full_soc_tolerance。
   部署时该请求功率由训练好的 actor 的均值策略在日前预测状态上逐期生成
   （见下文“未来真实 actor 接入方式”）。
2. 预约换电（优先）：该站该时段全部已接受预约（含当前请求已确定的临时
   预留）按（提交顺序, 路径内换电序号）依次服务；每次服务取走编号最小
   的服务就绪（SOC=1）槽位的满电电池，退回电池（SOC=rho）进入同一槽位。
3. 随机换电（FCFS）：日前随机需求预测 day_ahead_random_forecast 中该站
   该时段的请求按 (arrival_time, request_id) 排序，先到先服务，直到没有
   剩余满电电池；同样发出满电电池、退回电池进入同一槽位。
退回电池在当期不再被充电或服务，其 SOC 从下一时段起参与充电递推。

预约试排规则（论文 2.2.3 节规则 1-3）
------------------------------------
a. 用 get_feasible_arcs(candidate_network, od_id, day_ahead_entry_soc)
   得到个体可行弧集 A^{p,k}（日前用预约 SOC 精确过滤）。
b. 车辆从入口出发时 SOC = 预约入口 SOC，每次换电后恢复为 1。若当前节点
   存在直达出口弧（入口按精确 SOC、换电站按换电后满电判断，均要求到达
   SOC >= min_exit_soc；候选网络已施加“能直达出口则仅保留直达出弧”
   规则），则路径结束。
c. 否则，候选下一站 = 当前节点可行出弧指向的换电站中，满足以下两条者：
   - 库存：预计到站时段 q = floor(bar_t_A/Delta + tau_{p,i}/Delta) 为该
     用户再预留一块满电电池后，“预约服务后、随机服务前”的满电余量
     仍 >= 0（余量由含本次临时预留的库存递推计算）；
   - 可达：从该站沿可行弧集仍能到达出口（存在可行下游延续路径）。
d. 选站：按上述满电余量降序，并列时更靠近出口（位置大）优先，再按站
   ID 升序。
e. 选定后把 (站, 时段, 退回 SOC) 加入该请求的临时预留，继续直到出口。
f. 任一步无候选站，或个体可行弧集本身不存在完整路径（get_feasible_arcs
   抛 ValueError），则拒绝该请求；临时预留只在完整路径生成后才转为正式
   预留，因此回滚是天然的（事务式）。
退回 SOC（论文式 return_soc）：首站 rho = bar_soc_A - v(o,i)，后续站
rho = 1 - v(j,i)。

遗留私有分支的输入 / 输出格式
----------------------------
generate_dayahead_plan(params, candidate_network, mock_data,
                       rl_provider=None, output_path=None) -> dict

- params: data_generation_test.parameter.BusinessParameters。
- candidate_network: generate_candidate_network 的输出 dict（或
  load_candidate_network 读取的 JSON）。
- mock_data: rl_data.generate_mock_data / load_mock_data 的 dict，使用
  其中 "reservations"（键 reservation_id/od_id/day_ahead_entry_time/
  day_ahead_entry_soc/...）、"day_ahead_random_forecast"
  （[站][时段][请求{request_id, arrival_time, arrival_soc}]）和
  "initial_slot_soc"（[站][槽]）。
- rl_provider: 实现 rl_data.RLProvider 协议的对象；缺省
  MockRLProvider(params)。每期调用
  get_signals(params, period_ell=q, horizon=1, soc_obs=当前预测 SOC)，
  取 requested_power[i][b][0] 作为该期请求功率。

返回 dict（schema_version = 1，可直接 json.dumps）主要键：
- "schema_version", "generator", "seed", "num_periods",
  "num_sim_periods"（库存递推覆盖的时段数，>= num_periods）；
- "reservations": 每个预约一条记录：
    reservation_id, od_id, day_ahead_entry_time, day_ahead_entry_soc,
    accepted (bool), reject_reason (str 或 None),
    path_nodes (如 ["entry", 1, "exit"]，被拒时为 []),
    path_arcs ([[from, to], ...]，被拒时为 []),
    swap_stations / swap_periods / return_socs （逐次换电的站、预计时段、
    退回 SOC 的平行列表）;
- "baseline_station_visits": [站][时段] 已接受预约的换电次数
  （基准访问量）;
- "inventory_trajectory": 全部预约处理完后、含全部已接受预约与日前随机
  预测的库存递推轨迹：
    num_sim_periods,
    full_after_reservation[i][q] （充电+预约服务后、随机服务前满电数，
    可为负——仅表示若如此安排将出现的缺口；正式计划中已接受预约保证
    >= 0),
    full_after_random[i][q] （随机 FCFS 服务后的满电数，>= 0),
    slot_soc_end[i][q][b] （时段 q 结束后槽位 b 的预测 SOC),
    service_log[i][q] （服务记录列表，每条 {kind, ref, slot,
    return_soc}；kind 为 "reservation"/"random"，ref 为 reservation_id
    或 request_id）。

自检：validate_dayahead_plan(plan, params, candidate_network, mock_data,
rl_provider=None) 校验库存无负值、被拒请求无残留预留（重算轨迹一致且
已接受预约事件全部被服务）、当期退回电池当期不再被充电或服务（服务槽位
期末 SOC 恰为退回 SOC 且每槽每时段至多服务一次）、接受路径在个体可行
弧集内且首尾为 entry/exit、弧首尾相接、退回 SOC 与换电时段符合论文公式。
任何一项不满足即抛 ValueError。

调用示例
--------
作为库：

>>> from data_generation_test.parameter import get_default_parameters
>>> from data_generation_test.candidate_network import (
...     generate_candidate_network)
>>> from data_generation_test.rl_data import generate_mock_data
>>> from src.dayahead_plan import (
...     generate_dayahead_plan, validate_dayahead_plan, save_dayahead_plan)
>>> params = get_default_parameters()
>>> network = generate_candidate_network(params)
>>> mock = generate_mock_data(params, seed=42)
>>> plan = generate_dayahead_plan(params, network, mock)
>>> validate_dayahead_plan(plan, params, network, mock)
>>> save_dayahead_plan(plan)   # 默认 data_generation_test/output/dayahead_plan.json

命令行（加载或生成 candidate_network.json 与 mock_rl_data.json，生成
日前计划、自检、保存并打印摘要）：

    python src/dayahead_plan.py [--seed N] [--output PATH]
                                [--network PATH] [--mock-data PATH]

未来真实 actor 接入方式
----------------------
部署时，日前充电功率由训练完成的同一 actor 的均值策略在日前预测状态上
逐期生成（论文 2.2.3 节：“日前充电功率由训练完成的同一 actor 的均值
策略在日前预测状态上逐期生成”）。接入时无需修改本模块逻辑：只需实现
data_generation_test.rl_data.RLProvider 协议（方法
get_signals(params, period_ell, horizon, soc_obs) -> RLSignals，内部用
actor 均值策略 mu_phi 对 soc_obs 逐期 rollout），并把该对象作为
rl_provider 参数传入 generate_dayahead_plan / validate_dayahead_plan 即可；
目前默认使用确定性的 MockRLProvider 打通接口。

确定性
------
全部计算无随机源：库存递推、候选站排序（满电余量、位置、站 ID）、槽位
分配（编号最小的就绪槽位）均为确定性规则，MockRLProvider 亦确定性；同一
seed 生成的 mock 数据重跑，输出完全一致。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:  # 作为包导入（from src.dayahead_plan import ...）
    from data_generation_test.parameter import (
        ENTRY_NODE,
        EXIT_NODE,
        BusinessParameters,
        NodeId,
        get_default_parameters,
    )
    from data_generation_test.candidate_network import (
        DEFAULT_OUTPUT_PATH as DEFAULT_NETWORK_PATH,
        generate_candidate_network,
        get_feasible_arcs,
        load_candidate_network,
        save_candidate_network,
    )
    from data_generation_test.rl_data import (
        DEFAULT_MOCK_DATA_PATH,
        MockRLProvider,
        RLProvider,
        generate_mock_data,
        load_mock_data,
        save_mock_data,
    )
except ImportError:  # 直接作为脚本运行：python src/dayahead_plan.py
    sys.path.insert(0, str(_REPO_ROOT))
    from data_generation_test.parameter import (  # type: ignore
        ENTRY_NODE,
        EXIT_NODE,
        BusinessParameters,
        NodeId,
        get_default_parameters,
    )
    from data_generation_test.candidate_network import (  # type: ignore
        DEFAULT_OUTPUT_PATH as DEFAULT_NETWORK_PATH,
        generate_candidate_network,
        get_feasible_arcs,
        load_candidate_network,
        save_candidate_network,
    )
    from data_generation_test.rl_data import (  # type: ignore
        DEFAULT_MOCK_DATA_PATH,
        MockRLProvider,
        RLProvider,
        generate_mock_data,
        load_mock_data,
        save_mock_data,
    )

from src.continuous_dayahead import (
    SCHEMA_VERSION as CONTINUOUS_SCHEMA_VERSION,
    generate_continuous_dayahead_plan,
    validate_continuous_dayahead_plan,
)

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_PLAN_PATH",
    "generate_dayahead_plan",
    "validate_dayahead_plan",
    "save_dayahead_plan",
    "load_dayahead_plan",
]

SCHEMA_VERSION = CONTINUOUS_SCHEMA_VERSION

# 默认输出路径：data_generation_test/output/dayahead_plan.json
#（以本文件位置解析，与运行时工作目录无关）。
DEFAULT_PLAN_PATH = (
    _REPO_ROOT / "data_generation_test" / "output" / "dayahead_plan.json"
)

_EPS = 1e-9  # 浮点比较容差

# 内部换电事件元组：(站 i, 时段 q, 退回 SOC, reservation_id, 路径内换电序号)
_Event = Tuple[int, int, float, int, int]


# ----------------------------------------------------------------------
# 日前库存递推
# ----------------------------------------------------------------------
def _simulate_inventory(
    params: BusinessParameters,
    initial_slot_soc: Sequence[Sequence[float]],
    day_ahead_forecast: Sequence[Sequence[Sequence[Dict]]],
    reservation_events: Sequence[_Event],
    rl_provider: Optional[RLProvider],
    num_sim_periods: int,
) -> Dict:
    """对给定预约换电事件集合做全运营日逐站逐槽库存递推。

    参数
    ----
    reservation_events : 预约换电事件列表，元素为
        (station, period, return_soc, reservation_id, seq_in_path)。
        同一 (站, 时段) 内按 (reservation_id, seq_in_path) 排序服务
        （预约优先于随机请求，预约之间按提交顺序）。
    num_sim_periods : 递推覆盖的时段数；q >= params.num_periods 时没有
        RL 请求功率（视为 0）与随机需求预测（视为空），库存仅随服务变化。

    返回
    ----
    dict，键：full_after_reservation / full_after_random /
    slot_soc_end / service_log / unmet_reservation_events，
    结构见模块 docstring“输入/输出格式”。
    """
    st = params.station
    n_sta, n_slot = st.num_stations, st.num_slots
    eta, delta, e_b = st.charging_efficiency, params.delta_hours, params.battery_capacity_kwh
    p_tol = params.full_power_tolerance_kw()

    soc: List[List[float]] = [list(row) for row in initial_slot_soc]

    # 预约事件按 (站, 时段) 分组，组内按 (reservation_id, seq) 排序。
    res_by: Dict[Tuple[int, int], List[_Event]] = {}
    for ev in reservation_events:
        res_by.setdefault((ev[0], ev[1]), []).append(ev)
    for key in res_by:
        res_by[key].sort(key=lambda e: (e[3], e[4]))

    full_after_res: List[List[int]] = [
        [0] * num_sim_periods for _ in range(n_sta)
    ]
    full_after_rand: List[List[int]] = [
        [0] * num_sim_periods for _ in range(n_sta)
    ]
    slot_soc_end: List[List[List[float]]] = [
        [[0.0] * n_slot for _ in range(num_sim_periods)] for _ in range(n_sta)
    ]
    service_log: List[List[List[Dict]]] = [
        [[] for _ in range(num_sim_periods)] for _ in range(n_sta)
    ]
    unmet = 0

    for q in range(num_sim_periods):
        # ---- 1. 充电（当期退回的电池尚未进入库存，天然不参与当期充电） ----
        if rl_provider is not None and q < params.num_periods:
            signals = rl_provider.get_signals(
                params, period_ell=q, horizon=1, soc_obs=soc
            )
            p_hat = [
                [signals.requested_power[i][b][0] for b in range(n_slot)]
                for i in range(n_sta)
            ]
        else:
            p_hat = [[0.0] * n_slot for _ in range(n_sta)]
        for i in range(n_sta):
            for b in range(n_slot):
                p_need = params.power_needed_to_full_kw(soc[i][b], i)
                p_req = max(0.0, p_hat[i][b])
                if p_req >= p_need - p_tol:
                    # 容差内立即补满（含恰好相等）：P = P_need，SOC = 1。
                    soc[i][b] = 1.0
                else:
                    # 请求不足：严格执行 RL 请求。
                    soc[i][b] = min(
                        1.0, soc[i][b] + eta * delta * p_req / e_b
                    )

        for i in range(n_sta):
            # ---- 2. 预约换电（优先）----
            events = res_by.get((i, q), [])
            n_ready = sum(1 for b in range(n_slot) if soc[i][b] >= 1.0 - _EPS)
            # “预约服务后、随机服务前”的满电余量（允许在试排评估中为负，
            # 表示库存不足；正式接受路径保证 >= 0）。
            full_after_res[i][q] = n_ready - len(events)
            for (st_i, ev_q, rho, res_id, seq) in events:
                slot = _smallest_ready_slot(soc[i])
                if slot is None:
                    unmet += 1  # 库存不足，事件无法履约（仅试排中可能出现）
                    continue
                soc[i][slot] = rho
                service_log[i][q].append(
                    {
                        "kind": "reservation",
                        "ref": res_id,
                        "slot": slot,
                        "return_soc": rho,
                    }
                )

            # ---- 3. 随机请求 FCFS ----
            reqs: List[Dict] = []
            if q < params.num_periods:
                reqs = sorted(
                    day_ahead_forecast[i][q],
                    key=lambda r: (r["arrival_time"], r["request_id"]),
                )
            for r in reqs:
                slot = _smallest_ready_slot(soc[i])
                if slot is None:
                    break  # 无剩余满电电池：FCFS 前缀终止
                soc[i][slot] = r["arrival_soc"]
                service_log[i][q].append(
                    {
                        "kind": "random",
                        "ref": r["request_id"],
                        "slot": slot,
                        "return_soc": r["arrival_soc"],
                    }
                )
            full_after_rand[i][q] = sum(
                1 for b in range(n_slot) if soc[i][b] >= 1.0 - _EPS
            )
            for b in range(n_slot):
                slot_soc_end[i][q][b] = soc[i][b]

    return {
        "num_sim_periods": num_sim_periods,
        "full_after_reservation": full_after_res,
        "full_after_random": full_after_rand,
        "slot_soc_end": slot_soc_end,
        "service_log": service_log,
        "unmet_reservation_events": unmet,
    }


def _smallest_ready_slot(slot_soc: Sequence[float]) -> Optional[int]:
    """返回编号最小的服务就绪（SOC=1）槽位；没有则返回 None。"""
    for b, s in enumerate(slot_soc):
        if s >= 1.0 - _EPS:
            return b
    return None


# ----------------------------------------------------------------------
# 单个预约的事务式试排
# ----------------------------------------------------------------------
def _try_plan_reservation(
    params: BusinessParameters,
    network: dict,
    initial_slot_soc: Sequence[Sequence[float]],
    day_ahead_forecast: Sequence[Sequence[Sequence[Dict]]],
    reservation: Dict,
    accepted_events: Sequence[_Event],
    rl_provider: RLProvider,
    num_sim_periods: int,
) -> Dict:
    """按论文 2.2.3 节规则为单个预约事务式试排日前基准路径。

    成功时返回 accepted=True 的记录，并把新换电事件放在
    record["_new_events"]（内部键，调用方提交后移除）；失败时返回
    accepted=False 与 reject_reason，不修改 accepted_events（回滚天然
    成立：临时预留只在完整路径生成后提交）。每个候选事件必须同时满足：
    当前候选站/时段的预约服务后余量非负，且加入该事件后的整条预测轨迹
    没有任何未履约预约，从而不会破坏既有预约或本请求此前的临时预约。
    """
    res_id = reservation["reservation_id"]
    od_id = reservation["od_id"]
    od_index = _od_index_of(params, od_id)
    od = params.od_pairs[od_index]
    entry_time = reservation["day_ahead_entry_time"]  # 小时
    entry_soc = reservation["day_ahead_entry_soc"]
    entry_time_periods = entry_time / params.delta_hours

    base = {
        "reservation_id": res_id,
        "od_id": od_id,
        "day_ahead_entry_time": entry_time,
        "day_ahead_entry_soc": entry_soc,
    }

    # a. 个体可行弧集（按预约 SOC 精确过滤）。
    try:
        arcs = get_feasible_arcs(network, od_index, entry_soc)
    except ValueError as exc:
        return {
            **base,
            "accepted": False,
            "reject_reason": f"个体可行弧集不存在完整路径: {exc}",
            "path_nodes": [],
            "path_arcs": [],
            "swap_stations": [],
            "swap_periods": [],
            "return_socs": [],
        }
    arc_set = set(arcs)
    positions: Dict[NodeId, float] = {
        n: params.node_position_km(od_index, n)
        for n in params.od_nodes(od_index)
    }

    # c. 下游可达性：能沿可行弧到达 exit 的节点集合（反向 BFS）。
    reachable_exit = {EXIT_NODE}
    changed = True
    while changed:
        changed = False
        for f, t in arc_set:
            if t in reachable_exit and f not in reachable_exit:
                reachable_exit.add(f)
                changed = True

    current: NodeId = ENTRY_NODE
    path_arcs: List[Tuple[NodeId, NodeId]] = []
    tentative: List[_Event] = []  # 本请求的临时预留
    swap_stations: List[int] = []
    swap_periods: List[int] = []
    return_socs: List[float] = []

    while True:
        # b. 能直达出口且到达 SOC >= min_exit_soc（候选网络已保证：存在
        #    (current, exit) 弧即满足出口 SOC 要求，且此时仅有直达出弧）。
        if (current, EXIT_NODE) in arc_set:
            path_arcs.append((current, EXIT_NODE))
            break

        # c. 候选下一站：库存预留后余量 >= 0 且存在下游延续路径。
        outgoing = sorted(
            (t for (f, t) in arc_set if f == current and t != EXIT_NODE),
            key=lambda n: (positions[n], str(n)),
        )
        candidates = []
        for j in outgoing:
            if j not in reachable_exit:
                continue
            q = params.arrival_period(od_index, j, entry_time_periods)
            v = params.soc_consumption(od_index, current, j)
            # 退回 SOC：首站 = 入口 SOC - v(o,i)，后续站 = 1 - v(j,i)。
            rho = (entry_soc if current == ENTRY_NODE else 1.0) - v
            ev: _Event = (j, q, rho, res_id, len(tentative))
            sim = _simulate_inventory(
                params,
                initial_slot_soc,
                day_ahead_forecast,
                list(accepted_events) + tentative + [ev],
                rl_provider,
                num_sim_periods,
            )
            margin = sim["full_after_reservation"][j][q]
            # margin 保留为候选站排序依据；同时必须保证加入该事件后，
            # 整条模拟轨迹中的既有预约和本请求临时预约均仍可履约。
            if margin >= 0 and sim["unmet_reservation_events"] == 0:
                candidates.append((j, q, rho, margin, ev))

        # f. 无候选站 -> 拒绝（临时预留随函数返回丢弃，回滚完成）。
        if not candidates:
            return {
                **base,
                "accepted": False,
                "reject_reason": (
                    f"节点 {current!r} 处无满足库存与下游可达性要求的"
                    f"候选站（已试排 {len(tentative)} 次换电，全部回滚）"
                ),
                "path_nodes": [],
                "path_arcs": [],
                "swap_stations": [],
                "swap_periods": [],
                "return_socs": [],
            }

        # d. 选站：满电余量降序 -> 更靠近出口（位置大）-> 站 ID 升序。
        candidates.sort(key=lambda c: (-c[3], -positions[c[0]], c[0]))
        j, q, rho, _margin, ev = candidates[0]
        # e. 临时预留一块满电电池，继续向出口推进。
        tentative.append(ev)
        path_arcs.append((current, j))
        swap_stations.append(j)
        swap_periods.append(q)
        return_socs.append(rho)
        current = j

    path_nodes: List[NodeId] = [path_arcs[0][0]] + [a[1] for a in path_arcs]
    return {
        **base,
        "accepted": True,
        "reject_reason": None,
        "path_nodes": list(path_nodes),
        "path_arcs": [[a[0], a[1]] for a in path_arcs],
        "swap_stations": swap_stations,
        "swap_periods": swap_periods,
        "return_socs": return_socs,
        "_new_events": tentative,
    }


def _od_index_of(params: BusinessParameters, od_id: int) -> int:
    """按 od_id 查 O-D 对在 params.od_pairs 中的下标。"""
    for idx, od in enumerate(params.od_pairs):
        if od.od_id == od_id:
            return idx
    raise ValueError(f"参数中不存在 od_id={od_id}")


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def _generate_legacy_dayahead_plan(
    params: BusinessParameters,
    candidate_network: dict,
    mock_data: Dict,
    rl_provider: Optional[RLProvider] = None,
    output_path=None,
) -> Dict:
    """生成日前基准路径计划（论文 2.2.3 节）。

    参数与返回结构见模块 docstring“输入/输出格式”。``output_path``
    不为 None 时同时保存 JSON。全部计算确定性：同一 seed 的输入重跑
    输出一致。
    """
    params.validate()
    if rl_provider is None:
        rl_provider = MockRLProvider(params)

    initial_slot_soc = [list(row) for row in mock_data["initial_slot_soc"]]
    forecast = mock_data["day_ahead_random_forecast"]
    reservations = sorted(
        mock_data["reservations"], key=lambda r: r["reservation_id"]
    )

    # 库存递推需覆盖全部预约可能的到站时段（含超出运营时段的部分）。
    max_q = params.num_periods - 1
    for r in reservations:
        od_index = _od_index_of(params, r["od_id"])
        t_periods = r["day_ahead_entry_time"] / params.delta_hours
        for i in params.od_pairs[od_index].station_indices:
            max_q = max(max_q, params.arrival_period(od_index, i, t_periods))
    num_sim_periods = max_q + 1

    # 按提交顺序（reservation_id 升序）逐个事务式试排。
    accepted_events: List[_Event] = []
    results: List[Dict] = []
    for r in reservations:
        record = _try_plan_reservation(
            params,
            candidate_network,
            initial_slot_soc,
            forecast,
            r,
            accepted_events,
            rl_provider,
            num_sim_periods,
        )
        if record["accepted"]:
            accepted_events.extend(record.pop("_new_events"))
        results.append(record)

    # 全部预约处理完毕后，用全部已接受事件重算最终预测库存轨迹。
    final_sim = _simulate_inventory(
        params,
        initial_slot_soc,
        forecast,
        accepted_events,
        rl_provider,
        num_sim_periods,
    )

    # 各站基准访问量：[站][时段] 已接受预约的换电次数。
    st = params.station
    visits = [[0] * num_sim_periods for _ in range(st.num_stations)]
    for (i, q, _rho, _rid, _seq) in accepted_events:
        visits[i][q] += 1

    plan = {
        "schema_version": SCHEMA_VERSION,
        "generator": "src/dayahead_plan.py",
        "seed": mock_data.get("seed"),
        "num_periods": params.num_periods,
        "num_sim_periods": num_sim_periods,
        "reservations": results,
        "baseline_station_visits": visits,
        "inventory_trajectory": {
            "num_sim_periods": final_sim["num_sim_periods"],
            "full_after_reservation": final_sim["full_after_reservation"],
            "full_after_random": final_sim["full_after_random"],
            "slot_soc_end": final_sim["slot_soc_end"],
            "service_log": final_sim["service_log"],
        },
    }

    if output_path is not None:
        save_dayahead_plan(plan, output_path)
    return plan


# ----------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------
def generate_dayahead_plan(
    params: BusinessParameters,
    candidate_network: dict,
    mock_data: Dict,
    rl_provider: Optional[RLProvider] = None,
    output_path=None,
) -> Dict:
    """Generate the schema-2 continuous-event day-ahead baseline.

    The historical integer-period implementation remains in this file only
    for source-level migration reference.  New calls always use the shared
    continuous event kernel.
    """

    plan = generate_continuous_dayahead_plan(
        params, candidate_network, mock_data, rl_provider=rl_provider
    )
    if output_path is not None:
        save_dayahead_plan(plan, output_path)
    return plan


def _validate_legacy_dayahead_plan(
    plan: Dict,
    params: BusinessParameters,
    candidate_network: dict,
    mock_data: Dict,
    rl_provider: Optional[RLProvider] = None,
) -> None:
    """校验日前计划的一致性；任何一项不满足即抛 ValueError。

    校验内容：
    1. 库存无负值：逐槽预测 SOC 均在 [0,1] 内，随机服务后满电数 >= 0；
    2. 被拒请求无残留预留：用计划中 accepted=True 的记录重建事件集合并
       重新递推，轨迹与存储值一致，且全部已接受预约事件均被服务
       （unmet_reservation_events == 0）；
    3. 当期退回电池当期不再被充电或服务：每站每时段每个槽位至多出现
       一次服务记录，且被服务槽位的期末 SOC 恰等于退回 SOC（服务后
       当期未再发生变化）；
    4. 接受路径合法：每条弧在该预约（按日前 SOC 精确过滤）的个体可行
       弧集内，路径首节点为 "entry"、末节点为 "exit" 且弧首尾相接；
       退回 SOC 与论文式 return_soc 一致，换电时段与 arrival_period
       一致。
    """
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version 应为 {SCHEMA_VERSION}，当前为 "
            f"{plan.get('schema_version')}"
        )
    if rl_provider is None:
        rl_provider = MockRLProvider(params)

    num_sim_periods = plan["num_sim_periods"]
    traj = plan["inventory_trajectory"]
    st = params.station

    # ---- 1. 库存无负值 / SOC 范围 ----
    for i in range(st.num_stations):
        for q in range(num_sim_periods):
            if traj["full_after_random"][i][q] < 0:
                raise ValueError(
                    f"站 {i} 时段 {q} 随机服务后满电数为负: "
                    f"{traj['full_after_random'][i][q]}"
                )
            for b, s in enumerate(traj["slot_soc_end"][i][q]):
                if not -_EPS <= s <= 1.0 + _EPS:
                    raise ValueError(
                        f"站 {i} 时段 {q} 槽 {b} 预测 SOC={s} 超出 [0,1]"
                    )

    # ---- 2. 被拒请求无残留预留：重建事件 -> 重算轨迹 -> 比对 ----
    events: List[_Event] = []
    for rec in plan["reservations"]:
        if not rec["accepted"]:
            if rec["path_arcs"] or rec["swap_stations"] or rec["return_socs"]:
                raise ValueError(
                    f"被拒请求 {rec['reservation_id']} 存在残留路径/预留"
                )
            continue
        if not (
            len(rec["swap_stations"])
            == len(rec["swap_periods"])
            == len(rec["return_socs"])
        ):
            raise ValueError(
                f"预约 {rec['reservation_id']} 的换电站/时段/退回 SOC "
                f"列表长度不一致"
            )
        for seq, (i, q, rho) in enumerate(
            zip(
                rec["swap_stations"],
                rec["swap_periods"],
                rec["return_socs"],
            )
        ):
            events.append((i, q, rho, rec["reservation_id"], seq))

    resim = _simulate_inventory(
        params,
        mock_data["initial_slot_soc"],
        mock_data["day_ahead_random_forecast"],
        events,
        rl_provider,
        num_sim_periods,
    )
    if resim["unmet_reservation_events"] != 0:
        raise ValueError(
            f"存在 {resim['unmet_reservation_events']} 个已接受预约事件"
            f"因库存不足未被服务"
        )
    for key in (
        "full_after_reservation",
        "full_after_random",
        "slot_soc_end",
        "service_log",
    ):
        if resim[key] != traj[key]:
            raise ValueError(
                f"重算轨迹与存储轨迹不一致（键 {key!r}）：存在残留预留"
                f"或递推不可复现"
            )

    # ---- 3. 当期退回电池当期不再被充电或服务 ----
    for i in range(st.num_stations):
        for q in range(num_sim_periods):
            seen_slots = set()
            for log in traj["service_log"][i][q]:
                b = log["slot"]
                if b in seen_slots:
                    raise ValueError(
                        f"站 {i} 时段 {q} 槽位 {b} 被服务多次"
                    )
                seen_slots.add(b)
                if abs(traj["slot_soc_end"][i][q][b] - log["return_soc"]) > _EPS:
                    raise ValueError(
                        f"站 {i} 时段 {q} 槽位 {b} 期末 SOC="
                        f"{traj['slot_soc_end'][i][q][b]} 与退回 SOC="
                        f"{log['return_soc']} 不一致：当期退回电池当期"
                        f"又被充电或服务"
                    )

    # ---- 4. 接受路径在个体可行弧集内、首尾正确、公式一致 ----
    for rec in plan["reservations"]:
        if not rec["accepted"]:
            continue
        od_index = _od_index_of(params, rec["od_id"])
        entry_soc = rec["day_ahead_entry_soc"]
        entry_t_periods = rec["day_ahead_entry_time"] / params.delta_hours
        feas = set(get_feasible_arcs(candidate_network, od_index, entry_soc))

        arcs = [tuple(a) for a in rec["path_arcs"]]
        if not arcs:
            raise ValueError(f"预约 {rec['reservation_id']} 已接受但路径为空")
        if arcs[0][0] != ENTRY_NODE or arcs[-1][1] != EXIT_NODE:
            raise ValueError(
                f"预约 {rec['reservation_id']} 路径首尾不是 entry/exit: "
                f"{arcs}"
            )
        for a in arcs:
            if a not in feas:
                raise ValueError(
                    f"预约 {rec['reservation_id']} 的弧 {a} 不在个体可行"
                    f"弧集内"
                )
        for a, b in zip(arcs, arcs[1:]):
            if a[1] != b[0]:
                raise ValueError(
                    f"预约 {rec['reservation_id']} 路径弧不相接: {arcs}"
                )
        nodes = [arcs[0][0]] + [a[1] for a in arcs]
        if nodes != list(rec["path_nodes"]):
            raise ValueError(
                f"预约 {rec['reservation_id']} 的 path_nodes 与 path_arcs "
                f"不一致"
            )

        # 退回 SOC 与换电时段符合论文公式。
        prev: NodeId = ENTRY_NODE
        for seq, (i, q, rho) in enumerate(
            zip(
                rec["swap_stations"],
                rec["swap_periods"],
                rec["return_socs"],
            )
        ):
            v = params.soc_consumption(od_index, prev, i)
            expected_rho = (entry_soc if prev == ENTRY_NODE else 1.0) - v
            if abs(rho - expected_rho) > _EPS:
                raise ValueError(
                    f"预约 {rec['reservation_id']} 第 {seq} 次换电退回 SOC="
                    f"{rho} 与论文式 return_soc={expected_rho} 不一致"
                )
            expected_q = params.arrival_period(od_index, i, entry_t_periods)
            if q != expected_q:
                raise ValueError(
                    f"预约 {rec['reservation_id']} 第 {seq} 次换电时段 q={q}"
                    f" 与 arrival_period={expected_q} 不一致"
                )
            prev = i


# ----------------------------------------------------------------------
# JSON 落盘与读取
# ----------------------------------------------------------------------
def validate_dayahead_plan(
    plan: Dict,
    params: BusinessParameters,
    candidate_network: dict,
    mock_data: Dict,
    rl_provider: Optional[RLProvider] = None,
) -> None:
    """Validate a schema-2 plan by replaying the shared event kernel."""

    validate_continuous_dayahead_plan(
        plan,
        params,
        candidate_network,
        mock_data,
        rl_provider=rl_provider,
    )


def save_dayahead_plan(plan: Dict, path=DEFAULT_PLAN_PATH) -> None:
    """保存日前计划为 JSON（UTF-8，带缩进），自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)


def load_dayahead_plan(path=DEFAULT_PLAN_PATH) -> Dict:
    """从 JSON 文件加载日前计划。"""
    with Path(path).open("r", encoding="utf-8") as f:
        plan = json.load(f)
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"不支持的 schema_version: {plan.get('schema_version')}；"
            "请用 run_mpc.py --regenerate 生成 schema-2 连续事件日前计划"
        )
    return plan


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _print_summary(plan: Dict, params: BusinessParameters) -> None:
    """打印日前计划摘要。"""
    print("=== 日前基准路径计划摘要 ===")
    print(
        f"seed = {plan['seed']}，运营时段 {plan['num_periods']}，"
        f"库存递推覆盖 {plan['num_sim_periods']} 个时段"
    )
    n_acc = sum(1 for r in plan["reservations"] if r["accepted"])
    print(f"\n预约结果：接受 {n_acc} / {len(plan['reservations'])}")
    for r in plan["reservations"]:
        if r["accepted"]:
            swaps = ", ".join(
                f"站{i}@时段{q}(rho={rho:.3f})"
                for i, q, rho in zip(
                    r["swap_stations"], r["swap_periods"], r["return_socs"]
                )
            )
            print(
                f"  用户 {r['reservation_id']} (O-D {r['od_id']}): 接受，"
                f"路径 {r['path_nodes']}"
                + (f"；换电 {swaps}" if swaps else "；直达出口不换电")
            )
        else:
            print(
                f"  用户 {r['reservation_id']} (O-D {r['od_id']}): 拒绝 —— "
                f"{r['reject_reason']}"
            )

    st = params.station
    print("\n各站基准访问量（[时段] 已接受预约换电次数）:")
    for i in range(st.num_stations):
        print(f"  站 {i}: {plan['baseline_station_visits'][i]}")

    traj = plan["inventory_trajectory"]
    print("\n各站各时段满电电池数（预约服务后 / 随机服务后）:")
    for i in range(st.num_stations):
        print(
            f"  站 {i}: {traj['full_after_reservation'][i]} / "
            f"{traj['full_after_random'][i]}"
        )
    min_soc = min(
        s
        for i in range(st.num_stations)
        for row in traj["slot_soc_end"][i]
        for s in row
    )
    print(f"\n预测库存轨迹最小 SOC = {min_soc:.4f}（应 >= 0）")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="生成日前基准路径计划（论文 2.2.3 节）。"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（仅在需要重新生成 mock 数据时使用；缺省用参数 seed=42）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_PLAN_PATH),
        help=f"日前计划输出路径（默认 {DEFAULT_PLAN_PATH}）",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=str(DEFAULT_NETWORK_PATH),
        help=f"候选网络 JSON 路径（默认 {DEFAULT_NETWORK_PATH}，不存在则生成）",
    )
    parser.add_argument(
        "--mock-data",
        type=str,
        default=str(DEFAULT_MOCK_DATA_PATH),
        help=f"mock 数据 JSON 路径（默认 {DEFAULT_MOCK_DATA_PATH}，不存在则生成）",
    )
    args = parser.parse_args(argv)

    params = get_default_parameters()

    # 加载或生成候选网络。
    network_path = Path(args.network)
    if network_path.exists():
        network = load_candidate_network(network_path)
        print(f"已加载候选网络: {network_path}")
    else:
        network = generate_candidate_network(params)
        save_candidate_network(network, network_path)
        print(f"候选网络不存在，已生成并保存: {network_path}")

    # 加载或生成 mock 数据。
    mock_path = Path(args.mock_data)
    if mock_path.exists():
        mock_data = load_mock_data(mock_path)
        print(f"已加载 mock 数据: {mock_path}")
    else:
        mock_data = generate_mock_data(params, seed=args.seed)
        save_mock_data(mock_data, mock_path)
        print(f"mock 数据不存在，已生成并保存: {mock_path}")

    rl_provider = MockRLProvider(params)
    plan = generate_dayahead_plan(
        params, network, mock_data, rl_provider=rl_provider
    )
    validate_dayahead_plan(
        plan, params, network, mock_data, rl_provider=rl_provider
    )
    save_dayahead_plan(plan, args.output)
    _print_summary(plan, params)

    # 确定性自检：同输入重算，输出必须完全一致。
    plan2 = generate_dayahead_plan(
        params, network, mock_data, rl_provider=rl_provider
    )
    if plan2 != plan:
        raise ValueError("确定性自检失败：同输入重算结果不一致")

    # 落盘往返一致性检查。
    loaded = load_dayahead_plan(args.output)
    if loaded != plan:
        raise ValueError("JSON 保存/加载往返不一致")
    print(f"\n日前计划已生成、自检通过并保存至 {args.output}（往返一致）。")


if __name__ == "__main__":
    main()
