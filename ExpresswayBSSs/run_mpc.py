# -*- coding: utf-8 -*-
r"""
滚动 MPC 运行入口：论文“滚动时域优化、首阶段执行”机制（第 624 行与第
719-794 行 Exec_q 执行算子）的驱动脚本，按 plan.md 第 76-87 行实施。

流程
----
1. 生成（--regenerate 或文件缺失时）或加载三个输入文件：候选网络
   ``candidate_network.json``、模拟数据 ``mock_rl_data.json``、日前计划
   ``dayahead_plan.json``（默认均位于 ``data_generation_test/output/``）。
2. 从 ell=0 到 num_periods-1 逐时段滚动，每轮：
   a. 把全部已接受预约划分为互斥四态：未进入（pending，含逾期未到——
      有效 ETA 更新为 max(日前 ETA, ell) 留在 K_fut，避免从模型集合消失）、
      本轮新进入（arr，实际入口时刻 t_A ∈ (ell-1, ell]）、已发布在途
      （fix，固定承诺）、已完成（done）；
   b. 取窗口内（时段 ell..ell+H-1）预测随机请求，调 RLProvider 获取
      RLSignals，构造 MPCWindowInput 并 solve_step（完整 H 步模型）；
   c. 只发布本轮新进入用户的路径（result.publish_paths），未来用户路径
      丢弃；
   d. 只执行第一期充电功率与换电服务（result.first_stage）；
   e. 用实际数据更新物理 SOC（Exec_q）：本期实际到站预约事件（新发布
      路径按实际入口信息计算、q 钳位到 >= ell，加固定承诺本期事件）与
      实际随机到达按“预约优先、随机 FCFS 前缀”
      （预约沿用户路径传播 alive 状态并优先使用满电库存，不足部分在首个
      失败事件显式计违约；同周期下游事件随即失活；随机服务数为预约后的
      满电余量与实际随机请求数的较小者）
      服务；槽位分配用与 MPC 相同的最小可用槽位规范化规则；预测随机请求
      绝不写入真实状态；
   f. 新发布且未完成的预约路径转为固定承诺（剩余事件含到站时段与退回
      SOC），进入下一轮 MPCWindowInput.fixed_commitments。
3. 每轮自检（不通过抛带明确信息的异常）：SOC ∈ [0,1]、各站电池数守恒、
   物理满电集合与 g/F 一致、滚动状态首尾衔接（本轮 S_obs
   == 上轮执行后 SOC）。
4. 输出逐轮记录与最终汇总（累计实际收益 I^A/I^R、实际充电成本、实际
   调整成本、预约服务失败成本及总目标）到
   ``mpc_run_result.json``（--output 可配），JSON 可重新加载。
5. 同 seed 重跑结果一致（solve_time_sec 等诊断字段不参与比较）。

用户身份与结果 schema
------------------------
论文中预约用户的唯一身份为 ``(od_id, user_id)``，其中输入 JSON
的 ``reservation_id`` 对应 ``user_id``。本文件内部始终使用 ``UserKey``
二元组，因此不同 O-D 下相同 ``user_id`` 的预约可同时存在、发布和
履约，不会相互覆盖。

输出 ``schema_version = 2``。与 v1 不兼容的用户相关格式如下：

- ``published_paths`` 是 ``[{od_id, user_id, path}, ...]``，不再是仅以
  ``user_id`` 为键的对象；
- ``path_adjusted`` 是 ``[{od_id, user_id, adjusted}, ...]``；
- ``user_states`` 中的 ``pending/arr/fixed/completed`` 均为
  ``[{od_id, user_id}, ...]``；
- ``actual.reservation_swaps`` 的每条记录同时含 ``od_id`` 与
  ``user_id``。

.. note::
   本次复合键改造故意保留现有的新进入判定
   ``ell - 1 < t_A <= ell``。默认模拟数据由数据生成端规避该边界语义；
   该判定不属于本次身份键修改范围。

用法
----
    python run_mpc.py --seed 42 --regenerate
    python run_mpc.py                      # 复用已生成的输入文件
    python run_mpc.py --time-limit 60 --solver-log
    python run_mpc.py --network PATH --mock-data PATH --plan PATH \
        --output PATH

依赖：标准库 + gurobipy + data_generation_test 三模块 + src/dayahead_plan.py
+ src/mpc_model.py。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_generation_test.parameter import (  # noqa: E402
    ENTRY_NODE,
    BusinessParameters,
    NodeId,
    get_default_parameters,
)
from data_generation_test.candidate_network import (  # noqa: E402
    DEFAULT_OUTPUT_PATH as DEFAULT_NETWORK_PATH,
    generate_candidate_network,
    load_candidate_network,
    save_candidate_network,
    validate_candidate_network,
)
from data_generation_test.rl_data import (  # noqa: E402
    DEFAULT_MOCK_DATA_PATH,
    MockRLProvider,
    generate_mock_data,
    load_mock_data,
    save_mock_data,
)
from src.dayahead_plan import (  # noqa: E402
    DEFAULT_PLAN_PATH,
    generate_dayahead_plan,
    load_dayahead_plan,
    save_dayahead_plan,
    validate_dayahead_plan,
)
from src.mpc_model import (  # noqa: E402
    FixedCommitment,
    FixedSwapEvent,
    MPCController,
    MPCResult,
    MPCWindowInput,
    RandomRequest,
    RollingState,
    UserKey,
)

__all__ = ["DEFAULT_RESULT_PATH", "main", "run_rolling_mpc"]

SCHEMA_VERSION = 2

# 默认输出路径：data_generation_test/output/mpc_run_result.json
DEFAULT_RESULT_PATH = (
    _REPO_ROOT / "data_generation_test" / "output" / "mpc_run_result.json"
)

_EPS = 1e-9


def _reservation_key(record: Dict[str, Any]) -> UserKey:
    """将日前计划/mock 预约记录转为论文唯一用户键。"""
    return int(record["od_id"]), int(record["reservation_id"])


def _index_reservations(
    records: Sequence[Dict[str, Any]],
    *,
    source: str,
    accepted_only: bool = False,
) -> Dict[UserKey, Dict[str, Any]]:
    """以 ``(od_id, reservation_id)`` 建立索引，并显式拒绝真正的复合键重复。"""
    indexed: Dict[UserKey, Dict[str, Any]] = {}
    for record in records:
        if accepted_only and not record["accepted"]:
            continue
        key = _reservation_key(record)
        if key in indexed:
            raise ValueError(
                f"{source} 中预约复合键 (od_id={key[0]}, user_id={key[1]}) "
                "重复"
            )
        indexed[key] = record
    return indexed


def _user_record(key: UserKey) -> Dict[str, int]:
    """转为无 tuple 键、可直接 JSON 序列化的用户身份记录。"""
    return {"od_id": key[0], "user_id": key[1]}


def _user_records(keys: Iterable[UserKey]) -> List[Dict[str, int]]:
    """按复合键排序输出用户身份记录。"""
    return [_user_record(key) for key in sorted(keys)]


# ----------------------------------------------------------------------
# 路径换电事件计算（按实际/有效入口信息，式 eq:return_soc 与 arrival_period）
# ----------------------------------------------------------------------
def _path_swap_events(
    params: BusinessParameters,
    od_index: int,
    path_arcs: Sequence[Tuple[NodeId, NodeId]],
    entry_time: float,
    entry_soc: float,
    ell: int,
) -> List[FixedSwapEvent]:
    """由入口--出口路径弧序计算换电事件（站, 时段钳位到 >= ell, 退回 SOC）。"""
    events: List[FixedSwapEvent] = []
    t_periods = entry_time / params.delta_hours
    for (j, i) in path_arcs:
        if not isinstance(i, int):
            continue
        v = params.soc_consumption(od_index, j, i)
        rho = (entry_soc if j == ENTRY_NODE else 1.0) - v
        q = max(params.arrival_period(od_index, i, t_periods), ell)
        events.append(FixedSwapEvent(station=i, period=q, return_soc=rho))
    return events


# ----------------------------------------------------------------------
# 首阶段执行（式 eq:actual_execution_operator 的 Exec_q 算子）
# ----------------------------------------------------------------------
def _execute_first_stage(
    params: BusinessParameters,
    ell: int,
    soc_obs: Sequence[Sequence[float]],
    result: MPCResult,
    res_events: List[Dict[str, Any]],
    actual_random: Sequence[Sequence[Dict]],
) -> Dict[str, Any]:
    """执行时段 ell 的充电与换电服务，返回新 SOC 与服务记录。

    res_events：本期实际到站预约事件，元素含 od_id / user_id /
    station / return_soc / sort_key，并可含同一用户路径上的 path_order。
    预约按全局业务优先级与路径先后顺序执行；若同一用户在本时段的上游
    事件失败，则其本时段下游事件失活，既不服务也不重复计罚。
    actual_random：[站] 的实际随机请求列表（已到时段 ell 的）。
    """
    st = params.station
    n_sta, n_slot = st.num_stations, st.num_slots
    eta, delta, e_b = st.charging_efficiency, params.delta_hours, params.battery_capacity_kwh
    fs = result.first_stage

    new_soc: List[List[float]] = []
    ready_slots_by_station: List[List[int]] = []
    initial_full_by_station: List[int] = []
    served_res: List[Dict[str, Any]] = []
    failed_res: List[Dict[str, Any]] = []
    served_rand: List[Dict[str, Any]] = []
    rejected_rand: List[Dict[str, Any]] = []

    # ---- 1. 充电并逐站交叉检查物理满电状态与模型 g/F ----
    for i in range(n_sta):
        row = [
            soc_obs[i][b] + eta * delta * fs.power_kw[i][b] / e_b
            for b in range(n_slot)
        ]
        ready_threshold = 1.0 - 0.5 * params.full_soc_tolerance
        physical_ready_slots = [
            b for b in range(n_slot) if row[b] >= ready_threshold
        ]
        model_ready_slots = [
            b for b in range(n_slot) if fs.ready[i][b] == 1
        ]
        if model_ready_slots != physical_ready_slots:
            raise RuntimeError(
                f"时段 {ell} 站 {i} 的物理满电槽与 g 不一致："
                f"physical={physical_ready_slots}，g={model_ready_slots}，"
                f"充电后 SOC={row}"
            )
        if fs.available_full[i] != len(physical_ready_slots):
            raise RuntimeError(
                f"时段 {ell} 站 {i} 的 F_obs={fs.available_full[i]} 与物理"
                f"满电数 {len(physical_ready_slots)} 不一致"
            )
        # 执行层以物理满电状态为最终真相；在上述交叉检查通过后，它与
        # MPC 的 g/F 完全一致，满电池不能被经济目标主动关闭。
        ready_slots_by_station.append(list(physical_ready_slots))
        initial_full_by_station.append(len(physical_ready_slots))
        new_soc.append(row)

    # ---- 2. 预约优先，并在同一时段传播用户 alive 状态 ----
    #
    # sort_key 的末项是入弧/站点在 O-D 路径上的位置，前缀是业务优先级。
    # 显式 path_order（由滚动驱动生成）优先用于同一用户的上下游顺序；
    # 兼容测试或外部调用中只有 sort_key 的旧记录。
    def reservation_order_key(
        indexed_event: Tuple[int, Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        event_index, event = indexed_event
        sort_key = tuple(event["sort_key"])
        path_order = event.get(
            "path_order", sort_key[-1] if sort_key else event_index
        )
        path_position = sort_key[-1] if sort_key else event_index
        return (*sort_key[:-1], path_order, path_position, event_index)

    failed_users_this_period: Set[UserKey] = set()
    active_reservation_count = [0 for _ in range(n_sta)]
    failed_event_refs: List[Dict[str, Any]] = []
    for _, ev in sorted(enumerate(res_events), key=reservation_order_key):
        user_key = (ev["od_id"], ev["user_id"])
        if user_key in failed_users_this_period:
            # 上游事件已失败：本事件不再实际到达，不服务也不重复计罚。
            continue
        i = ev["station"]
        active_reservation_count[i] += 1
        ready_slots = ready_slots_by_station[i]
        if not ready_slots:
            failed_users_this_period.add(user_key)
            failed_event_refs.append(ev)
            continue
        b = ready_slots.pop(0)  # 编号最小的服务就绪槽位
        new_soc[i][b] = ev["return_soc"]
        served_res.append(
            {
                "period": ell,
                "station": i,
                "slot": b,
                "od_id": ev["od_id"],
                "user_id": ev["user_id"],
                "return_soc": ev["return_soc"],
            }
        )

    for ev in failed_event_refs:
        i = ev["station"]
        failed_res.append(
            {
                "period": ell,
                "station": i,
                "od_id": ev["od_id"],
                "user_id": ev["user_id"],
                "return_soc": ev["return_soc"],
                "available_full": initial_full_by_station[i],
                "reservation_demand": active_reservation_count[i],
                "reason": "insufficient_full_battery_after_charging",
            }
        )

    # ---- 3. 仅用预约后的物理余量服务实际随机 FCFS 前缀 ----
    for i in range(n_sta):
        ready_slots = ready_slots_by_station[i]
        reqs = sorted(
            actual_random[i], key=lambda r: (r["arrival_time"], r["request_id"])
        )
        n_rand = min(len(reqs), len(ready_slots))
        for ref in reqs[:n_rand]:
            if not ready_slots:
                raise RuntimeError(
                    f"时段 {ell} 站 {i} 服务事件 {ref} 时无可用满电槽位"
                )
            b = ready_slots.pop(0)  # 编号最小的服务就绪槽位
            new_soc[i][b] = ref["arrival_soc"]
            served_rand.append(
                {
                    "station": i,
                    "slot": b,
                    "request_id": ref["request_id"],
                    "return_soc": ref["arrival_soc"],
                }
            )
        rejected_rand.extend(
            {"station": i, "request_id": r["request_id"]} for r in reqs[n_rand:]
        )
        # 未换出的就绪电池在模型中应为精确 SOC 1；这里消除求解器的微小
        # 浮点残差，避免其进入下一滚动时段后被误判为未满。
        for b in ready_slots:
            new_soc[i][b] = 1.0
    return {
        "new_soc": new_soc,
        "served_res": served_res,
        "failed_res": failed_res,
        "served_rand": served_rand,
        "rejected_rand": rejected_rand,
    }


# ----------------------------------------------------------------------
# 输入文件的生成或加载
# ----------------------------------------------------------------------
def _load_or_generate(
    params: BusinessParameters, args: argparse.Namespace
) -> Tuple[dict, dict, dict]:
    network_path = Path(args.network)
    if args.regenerate or not network_path.exists():
        network = generate_candidate_network(params)
        validate_candidate_network(network, params)
        save_candidate_network(network, network_path)
        print(f"候选网络已生成: {network_path}")
    else:
        network = load_candidate_network(network_path)
        print(f"已加载候选网络: {network_path}")

    mock_path = Path(args.mock_data)
    if args.regenerate or not mock_path.exists():
        mock = generate_mock_data(params, seed=args.seed)
        save_mock_data(mock, mock_path)
        print(f"mock 数据已生成 (seed={args.seed}): {mock_path}")
    else:
        mock = load_mock_data(mock_path)
        print(f"已加载 mock 数据: {mock_path}")

    plan_path = Path(args.plan)
    if args.regenerate or not plan_path.exists():
        provider = MockRLProvider(params)
        plan = generate_dayahead_plan(params, network, mock, rl_provider=provider)
        validate_dayahead_plan(plan, params, network, mock, rl_provider=provider)
        save_dayahead_plan(plan, plan_path)
        print(f"日前计划已生成: {plan_path}")
    else:
        plan = load_dayahead_plan(plan_path)
        print(f"已加载日前计划: {plan_path}")
    return network, mock, plan


# ----------------------------------------------------------------------
# 滚动主流程
# ----------------------------------------------------------------------
def run_rolling_mpc(
    params: BusinessParameters,
    network: dict,
    mock: dict,
    plan: dict,
    rl_provider=None,
) -> Dict[str, Any]:
    """从 ell=0 滚动到 num_periods-1，返回可 JSON 序列化的结果 dict。"""
    st = params.station
    n_sta, n_slot = st.num_stations, st.num_slots
    H = params.horizon
    n_periods = params.num_periods
    e_b = params.battery_capacity_kwh
    delta = params.delta_hours
    kappa = params.path_adjustment_penalty
    failure_penalty = params.reservation_failure_penalty

    ctl = MPCController(
        params, network,
        rl_provider=rl_provider if rl_provider is not None else MockRLProvider(params),
        dayahead_plan=plan,
    )

    accepted = _index_reservations(
        plan["reservations"], source="日前计划", accepted_only=True
    )
    mock_res = _index_reservations(mock["reservations"], source="mock 数据")
    missing_mock = sorted(set(accepted) - set(mock_res))
    if missing_mock:
        missing_text = ", ".join(
            f"(od_id={key[0]}, user_id={key[1]})" for key in missing_mock
        )
        raise ValueError(f"已接受预约在 mock 数据中缺失: {missing_text}")

    soc_obs: List[List[float]] = [list(row) for row in mock["initial_slot_soc"]]
    commitments: Dict[UserKey, FixedCommitment] = {}
    completed: Set[UserKey] = set()
    failed: Set[UserKey] = set()

    rounds: List[Dict[str, Any]] = []
    totals = {
        "income_reservation": 0.0,
        "income_random": 0.0,
        "charging_cost": 0.0,
        "adjustment_cost": 0.0,
        "reservation_failure_cost": 0.0,
    }
    n_res_served = 0
    n_res_failed = 0
    n_rand_served = 0

    for ell in range(n_periods):
        # ---- 自检：滚动状态首尾衔接 ----
        if rounds:
            prev_end = rounds[-1]["soc_obs_end"]
            for i in range(n_sta):
                for b in range(n_slot):
                    if abs(prev_end[i][b] - soc_obs[i][b]) > 1e-6:
                        raise RuntimeError(
                            f"滚动状态首尾不衔接：站 {i} 槽 {b} 上轮结束 "
                            f"SOC={prev_end[i][b]} != 本轮观测 {soc_obs[i][b]}"
                        )
        # 自检：SOC 范围与电池数守恒
        for i in range(n_sta):
            if len(soc_obs[i]) != n_slot:
                raise RuntimeError(f"站 {i} 电池数不守恒：{len(soc_obs[i])}")
            for b, s in enumerate(soc_obs[i]):
                if not -_EPS <= s <= 1.0 + _EPS:
                    raise RuntimeError(
                        f"时段 {ell} 站 {i} 槽 {b} SOC={s} 超出 [0,1]"
                    )

        # ---- a. 预约四态划分 ----
        pending: List[UserKey] = []
        arr: List[UserKey] = []
        for user_key in sorted(accepted):
            if user_key in commitments or user_key in completed or user_key in failed:
                continue
            t_a = mock_res[user_key]["actual_entry_time"]
            # NOTE: 按本轮要求保留原有 (ell-1, ell] 进入判定；
            # 默认数据生成端负责规避该边界语义。
            if ell - 1.0 < t_a <= float(ell):
                arr.append(user_key)
            elif t_a > float(ell):
                pending.append(user_key)  # 含逾期未到（日前 ETA <= ell）
        fixed_ids = sorted(commitments)

        # ---- b. 构造窗口并求解 ----
        observations = []
        for user_key in arr:
            r = mock_res[user_key]
            observations.append(ctl.make_reservation_observation(
                user_key[1], r["actual_entry_time"], r["actual_entry_soc"],
                True, od_id=user_key[0]))
        for user_key in pending:
            r = mock_res[user_key]
            # 逾期未到：有效 ETA 更新到当前窗口（论文 eq 的 fut 分支取日前
            # 值；此处 max 保证不早于 ell，避免从模型集合消失）。
            eff_t = max(r["day_ahead_entry_time"], float(ell))
            if eff_t <= ell + H:  # 论文 K_fut: bar_t_A ∈ (ell, ell+H]
                observations.append(ctl.make_reservation_observation(
                    user_key[1], eff_t, r["day_ahead_entry_soc"], False,
                    od_id=user_key[0]))
        random_reqs: List[RandomRequest] = []
        for i in range(n_sta):
            for q in range(ell, min(ell + H, n_periods)):
                for r in mock["predicted_random_requests"][i][q]:
                    random_reqs.append(RandomRequest(
                        i, q, r["request_id"], r["arrival_time"],
                        r["arrival_soc"]))
        signals = ctl.rl_provider.get_signals(
            params, period_ell=ell, horizon=H, soc_obs=soc_obs)
        window = MPCWindowInput(
            params=params,
            rolling_state=RollingState(soc_obs=soc_obs, period_ell=ell),
            reservations=observations,
            random_requests=random_reqs,
            fixed_commitments=[commitments[user_key] for user_key in fixed_ids],
            rl_signals=signals,
        )
        result = ctl.solve_step(window)
        if not result.is_optimal:
            raise RuntimeError(
                f"时段 {ell} 未获得最优解（状态 {result.status}）"
            )

        # ---- c. 只发布本轮新进入用户的路径 ----
        published: Dict[UserKey, List[Tuple[NodeId, NodeId]]] = {
            user_key: [tuple(a) for a in path]
            for user_key, path in result.publish_paths.items()
        }

        # ---- d+e. 首阶段执行（Exec_q） ----
        # 本期实际到站预约事件：新发布路径（实际入口信息）+ 固定承诺本期事件。
        res_events: List[Dict[str, Any]] = []
        for user_key, path in published.items():
            od_id, user_id = user_key
            od_index = ctl._od_index_of(od_id)
            r = mock_res[user_key]
            path_events = _path_swap_events(
                params, od_index, path,
                r["actual_entry_time"], r["actual_entry_soc"], ell,
            )
            for path_order, fev in enumerate(path_events):
                if fev.period == ell:
                    res_events.append({
                        "od_id": od_id,
                        "user_id": user_id,
                        "station": fev.station,
                        "return_soc": fev.return_soc,
                        "path_order": path_order,
                        "sort_key": (
                            1,
                            od_id,
                            user_id,
                            params.node_position_km(od_index, fev.station),
                        ),
                    })
        for user_key in fixed_ids:
            commitment = commitments[user_key]
            for path_order, fev in enumerate(commitment.remaining_events):
                if fev.period == ell:
                    res_events.append({
                        "od_id": user_key[0],
                        "user_id": user_key[1],
                        "station": fev.station,
                        "return_soc": fev.return_soc,
                        "path_order": path_order,
                        "sort_key": (
                            0,
                            user_key[0],
                            user_key[1],
                            params.node_position_km(
                                ctl._od_index_of(commitment.od_id),
                                fev.station),
                        ),
                    })
        actual_random = [
            mock["actual_random_requests"][i][ell] for i in range(n_sta)
        ]
        exec_out = _execute_first_stage(
            params, ell, soc_obs, result, res_events, actual_random)
        new_soc = exec_out["new_soc"]
        failed_keys: Set[UserKey] = {
            (event["od_id"], event["user_id"])
            for event in exec_out["failed_res"]
        }
        failed.update(failed_keys)
        for user_key in failed_keys:
            commitments.pop(user_key, None)

        # ---- f. 新发布路径转为固定承诺；推进既有承诺 ----
        for user_key, path in published.items():
            if user_key in failed_keys:
                continue
            od_id, user_id = user_key
            od_index = ctl._od_index_of(od_id)
            r = mock_res[user_key]
            remaining = [
                fev for fev in _path_swap_events(
                    params, od_index, path,
                    r["actual_entry_time"], r["actual_entry_soc"], ell)
                if fev.period > ell
            ]
            if remaining:
                commitments[user_key] = FixedCommitment(
                    od_id=od_id, user_id=user_id,
                    fixed_path_arcs=path, remaining_events=remaining)
            else:
                completed.add(user_key)
        for user_key in list(commitments):
            if user_key in failed_keys:
                del commitments[user_key]
                continue
            if user_key in published:
                continue  # 本轮刚发布，剩余事件已按上式更新
            commitments[user_key].remaining_events = [
                fev for fev in commitments[user_key].remaining_events
                if fev.period > ell
            ]
            if not commitments[user_key].remaining_events:
                completed.add(user_key)
                del commitments[user_key]

        # ---- 本轮实际结算 ----
        pi = params.swap_service_price
        e_price = params.electricity_price
        income_a = sum(
            e_b * pi[ev["station"]][ell] * (1.0 - ev["return_soc"])
            for ev in exec_out["served_res"]
        )
        income_r = sum(
            e_b * pi[ev["station"]][ell] * (1.0 - ev["return_soc"])
            for ev in exec_out["served_rand"]
        )
        cost_ch = sum(
            e_price[i][ell] * delta * result.first_stage.power_kw[i][b]
            for i in range(n_sta) for b in range(n_slot)
        )
        # 实际调整成本：只计本轮正式发布且偏离基准的用户
        cost_adj = kappa * sum(
            result.path_adjusted.get(user_key, 0) for user_key in published
        )
        cost_fail = failure_penalty * len(exec_out["failed_res"])
        reward = income_a + income_r - cost_ch - cost_adj - cost_fail
        totals["income_reservation"] += income_a
        totals["income_random"] += income_r
        totals["charging_cost"] += cost_ch
        totals["adjustment_cost"] += cost_adj
        totals["reservation_failure_cost"] += cost_fail
        n_res_served += len(exec_out["served_res"])
        n_res_failed += len(exec_out["failed_res"])
        n_rand_served += len(exec_out["served_rand"])

        published_records = [
            {
                **_user_record(user_key),
                "path": [[arc[0], arc[1]] for arc in path],
            }
            for user_key, path in sorted(published.items())
        ]
        path_adjusted_records = [
            {
                **_user_record(user_key),
                "adjusted": result.path_adjusted.get(user_key, 0),
            }
            for user_key in sorted(published)
        ]

        rounds.append({
            "ell": ell,
            "status": result.status,
            "objective": {
                "total": result.objective_total,
                "income_reservation": result.income_reservation,
                "income_random": result.income_random,
                "charging_cost": result.charging_cost,
                "adjustment_cost": result.adjustment_cost,
                "reservation_failure_cost": result.reservation_failure_cost,
                "terminal_value": result.terminal_value,
            },
            "published_paths": published_records,
            "path_adjusted": path_adjusted_records,
            "predicted": {
                "random_served": sum(result.z.values()),
                "random_requests": len(result.z),
                "reservation_failures": sum(result.reservation_failed.values()),
                "at_risk_reservation_events": [
                    event
                    for event in result.events
                    if event.get("reservation_failed") == 1
                ],
                "window_events": len(result.events),
            },
            "actual": {
                "reservation_swaps": exec_out["served_res"],
                "reservation_failures": exec_out["failed_res"],
                "random_served": exec_out["served_rand"],
                "random_rejected": exec_out["rejected_rand"],
            },
            "power_kw": result.first_stage.power_kw,
            "ready": result.first_stage.ready,
            "available_full": result.first_stage.available_full,
            "soc_obs_start": [list(row) for row in soc_obs],
            "soc_obs_end": [list(row) for row in new_soc],
            "terminal_soc_pred": result.terminal_soc,
            "num_fixed_commitments": len(commitments),
            "user_states": {
                "pending": _user_records(pending),
                "arr": _user_records(arr),
                "fixed": _user_records(fixed_ids),
                "completed": _user_records(completed),
                "failed": _user_records(failed),
            },
            "realized": {
                "income_reservation": income_a,
                "income_random": income_r,
                "charging_cost": cost_ch,
                "adjustment_cost": cost_adj,
                "reservation_failure_cost": cost_fail,
                "reward": reward,
            },
            "solve_time_sec": result.solve_time_sec,  # 诊断字段，不参与复现比较
        })

        # 状态推进
        soc_obs = new_soc

    summary = {
        "total_income_reservation": totals["income_reservation"],
        "total_income_random": totals["income_random"],
        "total_charging_cost": totals["charging_cost"],
        "total_adjustment_cost": totals["adjustment_cost"],
        "total_reservation_failure_cost": totals["reservation_failure_cost"],
        "total_reward": (
            totals["income_reservation"] + totals["income_random"]
            - totals["charging_cost"] - totals["adjustment_cost"]
            - totals["reservation_failure_cost"]
        ),
        "total_actual_reservation_swaps": n_res_served,
        "total_actual_reservation_failures": n_res_failed,
        "total_actual_random_served": n_rand_served,
        "num_accepted_reservations": len(accepted),
        "num_completed_reservations": len(completed),
        "num_failed_reservations": len(failed),
        "accepted_reservations": _user_records(accepted),
        "completed_reservations": _user_records(completed),
        "failed_reservations": _user_records(failed),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "run_mpc.py",
        "seed": mock.get("seed"),
        "num_periods": n_periods,
        "horizon": H,
        "service_policy": {
            "reservation_priority": True,
            "reservation_failure_penalty": failure_penalty,
            "reservation_failure_unit": "per_unfulfilled_swap_event",
            "random_policy": "mandatory_fcfs_when_full_battery_available",
            "full_soc_tolerance": params.full_soc_tolerance,
            "full_power_tolerance_kw": params.full_power_tolerance_kw(),
        },
        "rounds": rounds,
        "summary": summary,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _print_round_brief(result: Dict[str, Any]) -> None:
    print("=== 逐轮摘要 ===")
    for rd in result["rounds"]:
        pub = rd["published_paths"]
        pub_str = (
            "; ".join(
                f"O-D{record['od_id']}/用户{record['user_id']}: "
                f"{record['path']}"
                for record in pub
            ) or "无"
        )
        print(
            f"ell={rd['ell']}: 目标={rd['objective']['total']:.3f} "
            f"(I^A={rd['objective']['income_reservation']:.2f}, "
            f"I^R={rd['objective']['income_random']:.2f}, "
            f"C_ch={rd['objective']['charging_cost']:.2f}, "
            f"C_adj={rd['objective']['adjustment_cost']:.2f}, "
            f"C_fail={rd['objective']['reservation_failure_cost']:.2f}, "
            f"Phi={rd['objective']['terminal_value']:.2f}); "
            f"发布: {pub_str}; "
            f"实际服务 预约{len(rd['actual']['reservation_swaps'])} "
            f"(失败{len(rd['actual']['reservation_failures'])}) "
            f"随机{len(rd['actual']['random_served'])}"
            f"(拒绝{len(rd['actual']['random_rejected'])}); "
            f"固定承诺 {rd['num_fixed_commitments']}"
        )
    s = result["summary"]
    print("\n=== 最终汇总 ===")
    print(
        f"累计实际收益: 预约 {s['total_income_reservation']:.3f} + "
        f"随机 {s['total_income_random']:.3f}；"
        f"充电成本 {s['total_charging_cost']:.3f}；"
        f"调整成本 {s['total_adjustment_cost']:.3f}；"
        f"预约失败成本 {s['total_reservation_failure_cost']:.3f}；"
        f"总目标 {s['total_reward']:.3f}"
    )
    print(
        f"实际换电: 预约 {s['total_actual_reservation_swaps']} 次"
        f"（失败 {s['total_actual_reservation_failures']} 次），"
        f"随机 {s['total_actual_random_served']} 次"
    )


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="滚动运行 MPC（论文滚动时域优化、首阶段执行）。"
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="mock 数据随机种子（默认 42）")
    parser.add_argument("--regenerate", action="store_true",
                        help="强制重新生成候选网络/mock 数据/日前计划")
    parser.add_argument("--network", type=str,
                        default=str(DEFAULT_NETWORK_PATH),
                        help="候选网络 JSON 路径")
    parser.add_argument("--mock-data", type=str,
                        default=str(DEFAULT_MOCK_DATA_PATH),
                        help="mock 数据 JSON 路径")
    parser.add_argument("--plan", type=str, default=str(DEFAULT_PLAN_PATH),
                        help="日前计划 JSON 路径")
    parser.add_argument("--output", type=str,
                        default=str(DEFAULT_RESULT_PATH),
                        help="结果 JSON 输出路径")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="覆盖 Gurobi 求解时限（秒）")
    parser.add_argument("--solver-log", action="store_true",
                        help="开启 Gurobi 求解日志")
    args = parser.parse_args(argv)

    params = get_default_parameters()
    if args.time_limit is not None:
        params.solver.time_limit_sec = args.time_limit
    if args.solver_log:
        params.solver.output_flag = 1
    params.validate()

    network, mock, plan = _load_or_generate(params, args)
    result = run_rolling_mpc(params, network, mock, plan)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _print_round_brief(result)

    # JSON 可重新加载自检
    with out_path.open("r", encoding="utf-8") as f:
        reloaded = json.load(f)
    if reloaded != result:
        raise RuntimeError("结果 JSON 保存/加载往返不一致")
    print(f"\n结果已保存并可重新加载: {out_path}")
    return result


if __name__ == "__main__":
    main()
