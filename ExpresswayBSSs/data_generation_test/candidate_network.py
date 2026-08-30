# -*- coding: utf-8 -*-
"""
离线候选网络模块：论文 2.2.2 节“基于 SOC 分档的离线候选网络”的实现。

对应论文 paper/mian11_fixed.tex 第 189-213 行（2.2.2 节）与 plan.md
第 29-40 行的要求。本模块仅使用 Python 标准库。

内容概览
--------
对每位预约用户，MPC 需要在入口--出口路径（换电站序列）上定义二元决策
变量。为避免每个滚动时刻重复枚举全部下游组合，本模块在运营前离线生成
候选网络：把用户到达高速入口时的 SOC 划分为若干分档（见
``BusinessParameters.soc_bins``），对每个 O-D 对 p 和每个 SOC 分档 h
生成候选弧集 A^{h,p} 及其完整路径列表。在线使用时，再以用户的精确入口
SOC 对所在分档的离线弧集（超集）做过滤，得到个体可行弧集 A^{p,k}。

算法步骤（论文 Step 1-4 加 plan.md 一致性约束）
---------------------------------------------
对每个 O-D 对 p（节点集 V^p = {o^p} ∪ S^p ∪ {d^p}，入口 "entry"、出口
"exit"、换电站为整数站点索引）和每个 SOC 分档 h = [lo, hi]：

- Step 1 生成原始弧（以分档上限 hi 为出发电量）：
  * 入口弧 o→j（j 为换电站）：若 hi - v(o,j) >= 0（能到达）；
  * 站间弧 i→j（i 在 j 上游）：若 v(i,j) <= 1（换电后满电 1 出发）；
  * 站→出口弧 i→d：若 v(i,d) + min_exit_soc <= 1；
  * 入口直达出口弧 o→d（plan.md 一致性约束）：若 hi - v(o,d) >=
    min_exit_soc。
- Step 2 枚举完整路径：用标准库 DFS 枚举原始网络中全部 o→d 完整路径；
  邻接节点按（位置 km，节点字符串）升序，保证枚举顺序确定。
- Step 3 剪枝过近换电弧：从完整路径中找出距离 < D_min 且终点不是出口
  的弧（入口--首站弧和站间弧；指向出口的末段弧不受约束），按距离升序
  （并列按弧端点位置、节点字符串固定排序）依次考察：试探删除该弧后，
  若剩余弧中仍存在至少一条对分档下界 lo 用户可行的完整 o→d 路径，
  则删除之，否则保留。由此同时保证论文“存在不含该弧的方案”与 plan.md
  “下界用户仍保有完整路径”两个条件（下界可行的路径对档内任意 SOC 均
  可行）。
- Step 4 形成候选网络：在剪枝后的弧集上重新枚举全部完整路径，将这些
  路径包含的弧合并为该档候选弧集 A^{h,p}；不在任何完整路径上的弧随之
  丢弃。

在线过滤（``get_feasible_arcs``，对应论文个体可行弧集 A^{p,k}）
--------------------------------------------------------------
1. 按精确入口 SOC 查询所属分档 h，取离线候选弧集 A^{h,p} 作为超集；
2. 精确 SOC 过滤：入口弧 o→j 要求 soc - v(o,j) >= 0（直达出口弧要求
   soc - v(o,d) >= min_exit_soc）；站间弧与站→出口弧的生成条件与入口
   SOC 无关，直接保留；
3. 直达出口优先规则（防止为服务收益进行不必要换电）：任一节点若能
   直接以满足出口最低 SOC 的方式到达出口（入口按精确 SOC 判断，换电站
   按换电后满电 1 判断），则该节点仅保留直达出口的出弧；
4. 清理无法组成任何完整 o→d 路径的弧：反复剔除出度或入度为 0 的非端点
   节点（换电站）及其关联弧，直到稳定；若最终不存在完整路径则抛出
   ValueError，否则返回确定性排序后的可行弧列表。

公开 API
--------
- ``generate_candidate_network(params)``：离线生成全部 O-D × SOC 档的
  候选网络，返回可直接 ``json.dumps`` 的字典；
- ``get_feasible_arcs(network, od_index, entry_soc)``：在线精确 SOC
  过滤，返回可行弧 ``(from_node, to_node)`` 元组列表；
- ``validate_candidate_network(network, params)``：校验弧向下游、路径
  首尾正确、每档存在完整路径、下界用户可行、直达出口优先规则等，非法
  抛出 ValueError；
- ``save_candidate_network(network, path)`` / ``load_candidate_network
  (path)``：JSON 落盘与读取（int 站点索引与 "entry"/"exit" 字符串节点
  往返一致）。

JSON 结构（schema_version = 1）
-------------------------------
- 顶层：``schema_version``、``generator``、``range_km``、
  ``min_exit_soc``、``min_swap_spacing_km``、``soc_bins``、
  ``od_networks``；
- 每个 ``od_networks`` 元素：``od_index``、``od_id``、``entry_km``、
  ``exit_km``、``nodes``（["entry", 站点 int..., "exit"]）、
  ``node_positions_km``（字符串键 "entry"/"exit"/"<站点索引>"）、
  ``bins``；
- 每个 ``bins`` 元素：``soc_bin_index``、``soc_lower``、``soc_upper``、
  ``raw_arcs``、``removed_arcs``（Step 3 剪除）、``candidate_arcs``、
  ``arc_distance_km``、``arc_soc_consumption``（后两者以
  "from->to" 字符串为键）、``complete_paths``（路径 = 弧 [from, to]
  列表的列表）。

用法
----
作为库：

>>> from data_generation_test.parameter import get_default_parameters
>>> from data_generation_test.candidate_network import (
...     generate_candidate_network, get_feasible_arcs)
>>> params = get_default_parameters()
>>> network = generate_candidate_network(params)
>>> arcs = get_feasible_arcs(network, od_index=1, entry_soc=0.6)

命令行（生成、校验、保存并打印摘要）：

    python data_generation_test/candidate_network.py [--output PATH]

默认输出 ``data_generation_test/output/candidate_network.json``。

所有枚举与排序均使用固定排序键（位置 km、节点字符串、距离），不使用任何
随机源，保证相同参数重跑结果完全一致。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:  # 作为包导入
    from .parameter import (
        ENTRY_NODE,
        EXIT_NODE,
        BusinessParameters,
        NodeId,
        get_default_parameters,
    )
except ImportError:  # 直接作为脚本运行：python data_generation_test/candidate_network.py
    from parameter import (  # type: ignore
        ENTRY_NODE,
        EXIT_NODE,
        BusinessParameters,
        NodeId,
        get_default_parameters,
    )

__all__ = [
    "SCHEMA_VERSION",
    "Arc",
    "generate_candidate_network",
    "get_feasible_arcs",
    "validate_candidate_network",
    "save_candidate_network",
    "load_candidate_network",
    "DEFAULT_OUTPUT_PATH",
]

SCHEMA_VERSION = 1

# 默认输出路径：data_generation_test/output/candidate_network.json
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "output" / "candidate_network.json"
)

Arc = Tuple[NodeId, NodeId]

_EPS = 1e-9  # 浮点比较容差


# ----------------------------------------------------------------------
# 节点 / 弧的 JSON 编码辅助
# ----------------------------------------------------------------------
def _node_key(node: NodeId) -> str:
    """节点 -> JSON 字符串键（"entry"/"exit"/"<站点索引>"）。"""
    return str(node)


def _parse_node(key: str) -> NodeId:
    """JSON 字符串键 -> 节点；int 站点索引与字符串节点往返一致。"""
    if key == ENTRY_NODE or key == EXIT_NODE:
        return key
    return int(key)


def _arc_key(arc: Arc) -> str:
    """弧 -> "from->to" 字符串键。"""
    return f"{_node_key(arc[0])}->{_node_key(arc[1])}"


def _parse_arc_key(key: str) -> Arc:
    """"from->to" 字符串键 -> 弧元组。"""
    a, b = key.split("->")
    return (_parse_node(a), _parse_node(b))


def _arc_sort_key(positions: Dict[NodeId, float], arc: Arc) -> Tuple:
    """弧的固定排序键：起点位置、终点位置、起点字符串、终点字符串。"""
    return (
        positions[arc[0]],
        positions[arc[1]],
        _node_key(arc[0]),
        _node_key(arc[1]),
    )


# ----------------------------------------------------------------------
# 路径枚举
# ----------------------------------------------------------------------
def _enumerate_complete_paths(
    arcs: Sequence[Arc],
    positions: Dict[NodeId, float],
    arc_soc: Optional[Dict[Arc, float]] = None,
    entry_soc: Optional[float] = None,
    min_exit_soc: float = 0.0,
) -> List[List[Arc]]:
    """DFS 枚举 arcs 中全部 entry -> exit 完整路径（确定性顺序）。

    邻接节点按（位置 km，节点字符串）升序访问；图为按位置严格递增的
    DAG，不存在环。若给定 ``entry_soc``，则只保留对入口弧满足精确 SOC
    约束的路径：入口→站 要求 entry_soc - v >= 0，入口→出口 要求
    entry_soc - v >= min_exit_soc；站间弧与站→出口弧的生成条件与入口
    SOC 无关，假定已由弧集构造保证。
    """
    adj: Dict[NodeId, List[NodeId]] = {}
    for f, t in arcs:
        adj.setdefault(f, []).append(t)
    for f in adj:
        adj[f].sort(key=lambda n: (positions[n], _node_key(n)))

    paths: List[List[Arc]] = []

    def dfs(node: NodeId, path: List[Arc]) -> None:
        if node == EXIT_NODE:
            paths.append(list(path))
            return
        for nb in adj.get(node, []):
            if entry_soc is not None and node == ENTRY_NODE:
                assert arc_soc is not None
                need = min_exit_soc if nb == EXIT_NODE else 0.0
                if entry_soc - arc_soc[(node, nb)] < need - _EPS:
                    continue
            path.append((node, nb))
            dfs(nb, path)
            path.pop()

    dfs(ENTRY_NODE, [])
    return paths


def _path_feasible_at_soc(
    path: Sequence[Arc],
    arc_soc: Dict[Arc, float],
    entry_soc: float,
    min_exit_soc: float,
) -> bool:
    """判断完整路径在给定入口 SOC 下是否可行（仅入口弧与 SOC 相关）。"""
    first = path[0]
    need = min_exit_soc if first[1] == EXIT_NODE else 0.0
    return entry_soc - arc_soc[first] >= need - _EPS


def _soc_bin_of(soc_bins: Sequence[Sequence[float]], soc: float) -> int:
    """与 BusinessParameters.soc_bin 相同的分档查询（基于网络内存储的分档）。"""
    if not 0.0 <= soc <= 1.0:
        raise ValueError(f"entry_soc={soc} 超出 [0,1] 范围")
    for h, (lo, hi) in enumerate(soc_bins):
        if h == len(soc_bins) - 1:
            if lo <= soc <= hi:
                return h
        elif lo <= soc < hi:
            return h
    raise ValueError(f"entry_soc={soc} 不属于任何分档 {list(soc_bins)}")


# ----------------------------------------------------------------------
# 离线候选网络生成
# ----------------------------------------------------------------------
def generate_candidate_network(params: BusinessParameters) -> dict:
    """按论文 2.2.2 节四步算法生成全部 O-D 对 × SOC 分档的离线候选网络。

    返回仅含标准库类型、可直接 ``json.dumps`` 的字典；结构见模块文档。
    """
    params.validate()
    d_min = params.min_swap_spacing_km
    min_exit = params.min_exit_soc

    od_networks: List[dict] = []
    for od_index, od in enumerate(params.od_pairs):
        nodes = params.od_nodes(od_index)
        positions: Dict[NodeId, float] = {
            n: params.node_position_km(od_index, n) for n in nodes
        }
        stations = list(od.station_indices)

        bins_out: List[dict] = []
        for h, (soc_lo, soc_hi) in enumerate(params.soc_bins):
            # ---- Step 1: 生成原始弧（以分档上限为出发电量） ----
            raw_arcs: List[Arc] = []
            # 入口 → 换电站 j：soc_hi - v(o,j) >= 0（能到达）
            for j in stations:
                v = params.soc_consumption(od_index, ENTRY_NODE, j)
                if soc_hi - v >= -_EPS:
                    raw_arcs.append((ENTRY_NODE, j))
            # 入口 → 出口直达弧（plan.md 一致性约束）：
            # soc_hi - v(o,d) >= min_exit_soc
            v_od = params.soc_consumption(od_index, ENTRY_NODE, EXIT_NODE)
            if soc_hi - v_od >= min_exit - _EPS:
                raw_arcs.append((ENTRY_NODE, EXIT_NODE))
            for i in stations:
                # 站间弧 i → j（i 在 j 上游）：v(i,j) <= 1
                for j in stations:
                    if positions[i] < positions[j]:
                        v = params.soc_consumption(od_index, i, j)
                        if v <= 1.0 + _EPS:
                            raw_arcs.append((i, j))
                # 站 → 出口：v(i,d) + min_exit_soc <= 1
                v_id = params.soc_consumption(od_index, i, EXIT_NODE)
                if v_id + min_exit <= 1.0 + _EPS:
                    raw_arcs.append((i, EXIT_NODE))
            raw_arcs.sort(key=lambda a: _arc_sort_key(positions, a))

            arc_soc: Dict[Arc, float] = {
                a: params.soc_consumption(od_index, a[0], a[1]) for a in raw_arcs
            }
            arc_dist: Dict[Arc, float] = {
                a: params.distance_km(od_index, a[0], a[1]) for a in raw_arcs
            }

            # ---- Step 2: DFS 枚举全部完整路径 ----
            paths0 = _enumerate_complete_paths(raw_arcs, positions)

            # ---- Step 3: 剪枝距离 < D_min 的非出口弧 ----
            # 候选短弧：出现在完整路径中、终点不是出口、距离 < D_min。
            short_arcs = sorted(
                {
                    a
                    for p in paths0
                    for a in p
                    if a[1] != EXIT_NODE and arc_dist[a] < d_min - _EPS
                },
                key=lambda a: (
                    arc_dist[a],
                    positions[a[0]],
                    positions[a[1]],
                    _node_key(a[0]),
                    _node_key(a[1]),
                ),
            )
            current = set(raw_arcs)
            removed: List[Arc] = []
            for a in short_arcs:
                if a not in current:
                    continue
                current.discard(a)
                # 删除后仍须存在对分档下界用户可行的完整路径
                #（下界可行 => 档内任意 SOC 可行，同时满足论文
                #  “仍存在不含该弧的完整方案”条件）。
                if _enumerate_complete_paths(
                    sorted(current, key=lambda x: _arc_sort_key(positions, x)),
                    positions,
                    arc_soc=arc_soc,
                    entry_soc=soc_lo,
                    min_exit_soc=min_exit,
                ):
                    removed.append(a)
                else:
                    current.add(a)

            # ---- Step 4: 合并保留路径的弧为候选弧集 A^{h,p} ----
            kept_arcs = sorted(current, key=lambda a: _arc_sort_key(positions, a))
            final_paths = _enumerate_complete_paths(kept_arcs, positions)
            candidate_arcs = sorted(
                {a for p in final_paths for a in p},
                key=lambda a: _arc_sort_key(positions, a),
            )

            bins_out.append(
                {
                    "soc_bin_index": h,
                    "soc_lower": soc_lo,
                    "soc_upper": soc_hi,
                    "raw_arcs": [[a[0], a[1]] for a in raw_arcs],
                    "removed_arcs": [[a[0], a[1]] for a in removed],
                    "candidate_arcs": [[a[0], a[1]] for a in candidate_arcs],
                    "arc_distance_km": {
                        _arc_key(a): arc_dist[a] for a in candidate_arcs
                    },
                    "arc_soc_consumption": {
                        _arc_key(a): arc_soc[a] for a in candidate_arcs
                    },
                    "complete_paths": [
                        [[a[0], a[1]] for a in p] for p in final_paths
                    ],
                }
            )

        od_networks.append(
            {
                "od_index": od_index,
                "od_id": od.od_id,
                "entry_km": od.entry_km,
                "exit_km": od.exit_km,
                "nodes": list(nodes),
                "node_positions_km": {_node_key(n): positions[n] for n in nodes},
                "bins": bins_out,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "data_generation_test/candidate_network.py",
        "range_km": params.range_km,
        "min_exit_soc": params.min_exit_soc,
        "min_swap_spacing_km": params.min_swap_spacing_km,
        "soc_bins": [list(b) for b in params.soc_bins],
        "od_networks": od_networks,
    }


# ----------------------------------------------------------------------
# 在线精确 SOC 过滤
# ----------------------------------------------------------------------
def _get_od_info(network: dict, od_index: int) -> dict:
    """按 od_index 取出 O-D 子网络，不存在时抛 ValueError。"""
    for od_info in network.get("od_networks", []):
        if od_info.get("od_index") == od_index:
            return od_info
    raise ValueError(f"候选网络中不存在 od_index={od_index}")


def get_feasible_arcs(network: dict, od_index: int, entry_soc: float) -> List[Arc]:
    """在线过滤：按精确入口 SOC 从离线候选网络得到个体可行弧集 A^{p,k}。

    步骤（详见模块文档）：
    1. 按 ``entry_soc`` 所属分档取离线候选弧集作为超集；
    2. 精确 SOC 过滤入口弧（站弧 o→j：soc - v >= 0；直达出口弧：
       soc - v >= min_exit_soc）；
    3. 直达出口优先规则：任一节点若能直接以满足出口最低 SOC 的方式到达
       出口，则仅保留其直达出口的出弧；
    4. 反复剔除出/入度为 0 的非端点节点，清理无法组成完整路径的弧。

    返回确定性排序的 ``(from_node, to_node)`` 元组列表；结果保证至少
    存在一条完整 entry→exit 路径，否则抛出 ValueError。
    """
    od_info = _get_od_info(network, od_index)
    soc_bins = [tuple(b) for b in network["soc_bins"]]
    h = _soc_bin_of(soc_bins, entry_soc)
    bin_info = od_info["bins"][h]
    min_exit = network["min_exit_soc"]

    positions: Dict[NodeId, float] = {
        _parse_node(k): v for k, v in od_info["node_positions_km"].items()
    }
    arc_soc: Dict[Arc, float] = {
        _parse_arc_key(k): v for k, v in bin_info["arc_soc_consumption"].items()
    }
    superset: List[Arc] = [tuple(a) for a in bin_info["candidate_arcs"]]

    # ---- 精确 SOC 过滤（仅入口弧与 entry_soc 相关） ----
    arcs: List[Arc] = []
    for f, t in superset:
        if f == ENTRY_NODE:
            need = min_exit if t == EXIT_NODE else 0.0
            if entry_soc - arc_soc[(f, t)] >= need - _EPS:
                arcs.append((f, t))
        else:
            # 站间弧（v<=1）与站→出口弧（1 - v >= min_exit_soc）的生成
            # 条件与入口 SOC 无关，直接保留。
            arcs.append((f, t))
    arc_set = set(arcs)

    # ---- 直达出口优先规则 ----
    # 入口：若能直接满足出口最低 SOC（弧 (entry, exit) 已通过过滤），
    # 仅保留直达出口弧；换电站：换电后满电 1，若 (i, exit) 在弧集中则
    # 仅保留该出弧。
    if (ENTRY_NODE, EXIT_NODE) in arc_set:
        arc_set = {a for a in arc_set if a[0] != ENTRY_NODE or a[1] == EXIT_NODE}
    stations = [n for n in positions if isinstance(n, int)]
    for i in stations:
        if (i, EXIT_NODE) in arc_set:
            arc_set = {a for a in arc_set if a[0] != i or a[1] == EXIT_NODE}

    # ---- 清理无法组成完整路径的弧（反复剔除出/入度为 0 的非端点节点） ----
    while True:
        out_deg: Dict[NodeId, int] = {n: 0 for n in positions}
        in_deg: Dict[NodeId, int] = {n: 0 for n in positions}
        for f, t in arc_set:
            out_deg[f] += 1
            in_deg[t] += 1
        dead = {
            n
            for n in stations
            if (out_deg[n] == 0 or in_deg[n] == 0)
            and any(n in a for a in arc_set)
        }
        if not dead:
            break
        arc_set = {a for a in arc_set if a[0] not in dead and a[1] not in dead}

    # ---- 校验仍存在完整路径 ----
    kept = sorted(arc_set, key=lambda a: _arc_sort_key(positions, a))
    paths = _enumerate_complete_paths(kept, positions)
    if not paths:
        raise ValueError(
            f"O-D {od_info.get('od_id', od_index)}（od_index={od_index}）在入口 "
            f"SOC={entry_soc}（分档 {h}）下不存在完整 entry->exit 可行路径"
        )
    return kept


# ----------------------------------------------------------------------
# 校验
# ----------------------------------------------------------------------
def validate_candidate_network(network: dict, params: BusinessParameters) -> None:
    """校验候选网络结构与一致性规则；任何一项不满足即抛 ValueError。

    校验内容：
    - schema 版本及全局参数（SOC 分档、min_exit_soc、D_min、range_km）与
      ``params`` 一致；
    - 每个 O-D 的节点集合与位置与 ``params`` 一致；
    - 全部候选弧向下游（位置严格递增）、端点属于节点集合，弧距离与 SOC
      消耗与 ``params`` 计算一致；
    - 候选弧集恰为存储的完整路径之并；每条完整路径起于 entry、止于
      exit、弧首尾相接，且在分档上限 SOC 下可行；
    - 每个分档至少存在一条完整路径，且至少一条对分档下界 SOC 用户可行；
    - Step 3 剪除的弧均为距离 < D_min 的非出口弧；
    - 直达出口优先规则：对每个 O-D、每个分档的下界与中点 SOC，
      ``get_feasible_arcs`` 的结果中，任一能直达出口的节点只有直达出口
      的出弧，结果弧集是离线候选弧集的子集、全部向下游且含完整路径。
    """
    if network.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version 应为 {SCHEMA_VERSION}，当前为 "
            f"{network.get('schema_version')}"
        )
    if [list(b) for b in params.soc_bins] != network.get("soc_bins"):
        raise ValueError("soc_bins 与参数不一致")
    for name, value in (
        ("min_exit_soc", params.min_exit_soc),
        ("min_swap_spacing_km", params.min_swap_spacing_km),
        ("range_km", params.range_km),
    ):
        if abs(network.get(name, float("nan")) - value) > _EPS:
            raise ValueError(f"{name} 与参数不一致")

    od_infos = network.get("od_networks")
    if not isinstance(od_infos, list) or len(od_infos) != len(params.od_pairs):
        raise ValueError("od_networks 数量与参数 od_pairs 不一致")

    d_min = params.min_swap_spacing_km
    min_exit = params.min_exit_soc

    for od_index, od in enumerate(params.od_pairs):
        od_info = _get_od_info(network, od_index)
        label = f"O-D {od.od_id}（od_index={od_index}）"

        # 节点集合与位置
        expected_nodes = params.od_nodes(od_index)
        if [n for n in od_info["nodes"]] != expected_nodes:
            raise ValueError(f"{label}: nodes 与参数不一致: {od_info['nodes']}")
        positions: Dict[NodeId, float] = {}
        for k, v in od_info["node_positions_km"].items():
            n = _parse_node(k)
            expected_pos = params.node_position_km(od_index, n)
            if abs(v - expected_pos) > _EPS:
                raise ValueError(
                    f"{label}: 节点 {k!r} 位置 {v} 与参数 {expected_pos} 不一致"
                )
            positions[n] = v

        if len(od_info["bins"]) != len(params.soc_bins):
            raise ValueError(f"{label}: 分档数量与参数不一致")

        for h, (soc_lo, soc_hi) in enumerate(params.soc_bins):
            bin_info = od_info["bins"][h]
            blabel = f"{label} 分档 {h} [{soc_lo},{soc_hi}]"
            if bin_info.get("soc_bin_index") != h:
                raise ValueError(f"{blabel}: soc_bin_index 错误")

            candidate: List[Arc] = [tuple(a) for a in bin_info["candidate_arcs"]]
            cand_set = set(candidate)
            arc_soc: Dict[Arc, float] = {
                _parse_arc_key(k): v
                for k, v in bin_info["arc_soc_consumption"].items()
            }
            arc_dist: Dict[Arc, float] = {
                _parse_arc_key(k): v
                for k, v in bin_info["arc_distance_km"].items()
            }

            # 弧端点合法、向下游、距离与 SOC 消耗一致
            for a in candidate:
                if a[0] not in positions or a[1] not in positions:
                    raise ValueError(f"{blabel}: 弧 {a} 端点不在节点集合中")
                if not positions[a[0]] < positions[a[1]]:
                    raise ValueError(f"{blabel}: 弧 {a} 不是下游弧")
                if abs(arc_dist[a] - params.distance_km(od_index, *a)) > _EPS:
                    raise ValueError(f"{blabel}: 弧 {a} 距离与参数不一致")
                if abs(arc_soc[a] - params.soc_consumption(od_index, *a)) > _EPS:
                    raise ValueError(f"{blabel}: 弧 {a} SOC 消耗与参数不一致")

            # 完整路径：首尾正确、弧相接、属于候选弧集、上限 SOC 可行
            paths: List[List[Arc]] = [
                [tuple(a) for a in p] for p in bin_info["complete_paths"]
            ]
            if not paths:
                raise ValueError(f"{blabel}: 不存在完整路径")
            union: set = set()
            for p in paths:
                if p[0][0] != ENTRY_NODE or p[-1][1] != EXIT_NODE:
                    raise ValueError(f"{blabel}: 路径 {p} 首尾不是 entry/exit")
                for a in p:
                    if a not in cand_set:
                        raise ValueError(f"{blabel}: 路径弧 {a} 不在候选弧集中")
                    union.add(a)
                for a, b in zip(p, p[1:]):
                    if a[1] != b[0]:
                        raise ValueError(f"{blabel}: 路径 {p} 弧不相接")
                if not _path_feasible_at_soc(p, arc_soc, soc_hi, min_exit):
                    raise ValueError(
                        f"{blabel}: 路径 {p} 在分档上限 SOC 下不可行"
                    )
            if union != cand_set:
                raise ValueError(f"{blabel}: 候选弧集不是完整路径之并")

            # 分档下界用户至少保有一条完整路径
            if not any(
                _path_feasible_at_soc(p, arc_soc, soc_lo, min_exit) for p in paths
            ):
                raise ValueError(
                    f"{blabel}: 分档下界 SOC={soc_lo} 用户无完整路径"
                )

            # 剪除弧均为距离 < D_min 的非出口弧
            raw_set = {tuple(a) for a in bin_info["raw_arcs"]}
            for a in bin_info["removed_arcs"]:
                a = tuple(a)
                if a not in raw_set or a in cand_set:
                    raise ValueError(f"{blabel}: 剪除弧 {a} 状态不一致")
                if a[1] == EXIT_NODE:
                    raise ValueError(f"{blabel}: 剪除了指向出口的弧 {a}")
                if not params.distance_km(od_index, *a) < d_min - _EPS:
                    raise ValueError(
                        f"{blabel}: 剪除弧 {a} 距离不小于 D_min={d_min}"
                    )

            # 直达出口优先规则（在线过滤结果，按下界与中点 SOC 检验）
            for soc in (soc_lo, (soc_lo + soc_hi) / 2.0):
                feas = get_feasible_arcs(network, od_index, soc)
                feas_set = set(feas)
                if not feas_set <= cand_set:
                    raise ValueError(
                        f"{blabel}: SOC={soc} 的可行弧不是离线候选弧集的子集"
                    )
                for a in feas:
                    if not positions[a[0]] < positions[a[1]]:
                        raise ValueError(f"{blabel}: 可行弧 {a} 不是下游弧")
                out_map: Dict[NodeId, List[NodeId]] = {}
                for f, t in feas:
                    out_map.setdefault(f, []).append(t)
                for n, outs in out_map.items():
                    if EXIT_NODE in outs and len(outs) > 1:
                        raise ValueError(
                            f"{blabel}: SOC={soc} 时节点 {n!r} 能直达出口"
                            f"但保留了其他出弧 {outs}"
                        )
                if not _enumerate_complete_paths(feas, positions):
                    raise ValueError(
                        f"{blabel}: SOC={soc} 的可行弧中无完整路径"
                    )


# ----------------------------------------------------------------------
# JSON 落盘与读取
# ----------------------------------------------------------------------
def save_candidate_network(network: dict, path=DEFAULT_OUTPUT_PATH) -> None:
    """保存候选网络为 JSON 文件（UTF-8，带缩进），自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(network, f, ensure_ascii=False, indent=2)


def load_candidate_network(path=DEFAULT_OUTPUT_PATH) -> dict:
    """从 JSON 文件加载候选网络。

    int 站点索引在弧列表 ``[from, to]`` 中保持 int 类型；以字符串为键的
    映射（``node_positions_km``、``arc_distance_km``、
    ``arc_soc_consumption``）在使用处通过 ``_parse_node`` /
    ``_parse_arc_key`` 还原，保证与生成时的内存表示往返一致。
    """
    with Path(path).open("r", encoding="utf-8") as f:
        network = json.load(f)
    if network.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"不支持的 schema_version: {network.get('schema_version')}"
        )
    return network


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _print_summary(network: dict) -> None:
    """打印候选网络规模摘要。"""
    print("=== 候选网络摘要 ===")
    print(
        f"SOC 分档: {network['soc_bins']}，min_exit_soc = "
        f"{network['min_exit_soc']}，D_min = {network['min_swap_spacing_km']} km"
    )
    for od_info in network["od_networks"]:
        print(
            f"O-D {od_info['od_id']}（od_index={od_info['od_index']}）: "
            f"入口 {od_info['entry_km']} km -> 出口 {od_info['exit_km']} km，"
            f"节点 {od_info['nodes']}"
        )
        for bin_info in od_info["bins"]:
            print(
                f"  分档 {bin_info['soc_bin_index']} "
                f"[{bin_info['soc_lower']},{bin_info['soc_upper']}]: "
                f"原始弧 {len(bin_info['raw_arcs'])}，剪除 "
                f"{len(bin_info['removed_arcs'])}，候选弧 "
                f"{len(bin_info['candidate_arcs'])}，完整路径 "
                f"{len(bin_info['complete_paths'])}"
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="生成基于 SOC 分档的离线候选网络（论文 2.2.2 节）"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"输出 JSON 路径（默认 {DEFAULT_OUTPUT_PATH}）",
    )
    args = parser.parse_args(argv)

    params = get_default_parameters()
    network = generate_candidate_network(params)
    validate_candidate_network(network, params)
    save_candidate_network(network, args.output)
    _print_summary(network)

    # 落盘往返一致性检查
    loaded = load_candidate_network(args.output)
    if loaded != network:
        raise ValueError("JSON 保存/加载往返不一致")
    print(f"\n候选网络已生成、校验并保存至 {args.output}（往返一致）。")


if __name__ == "__main__":
    main()
