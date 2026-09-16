# -*- coding: utf-8 -*-
r"""
日前基准路径生成模块：论文 3.2 节“Day-Ahead Baseline Path Generation”。

对应论文 paper/main.tex 第 178-195 行（3.2 节）。公开接口
``generate_dayahead_plan`` / ``validate_dayahead_plan`` 直接委托
``src.continuous_dayahead``：以共享连续事件内核生成并校验 schema-2、
``continuous_event_v2`` 的日前计划。另提供 ``save_dayahead_plan`` /
``load_dayahead_plan`` 的 JSON 落盘与读取，以及命令行入口::

    python src/dayahead_plan.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent

try:  # 作为包导入（from src.dayahead_plan import ...）
    from data_generation_test.parameter import (
        BusinessParameters,
        get_default_parameters,
    )
    from data_generation_test.candidate_network import (
        DEFAULT_OUTPUT_PATH as DEFAULT_NETWORK_PATH,
        generate_candidate_network,
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
        BusinessParameters,
        get_default_parameters,
    )
    from data_generation_test.candidate_network import (  # type: ignore
        DEFAULT_OUTPUT_PATH as DEFAULT_NETWORK_PATH,
        generate_candidate_network,
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

# ----------------------------------------------------------------------
# 公开接口

def generate_dayahead_plan(
    params: BusinessParameters,
    candidate_network: dict,
    mock_data: Dict,
    rl_provider: Optional[RLProvider] = None,
    output_path=None,
) -> Dict:
    """生成 schema-2 连续事件日前基准计划（委托 src.continuous_dayahead）。"""


    plan = generate_continuous_dayahead_plan(
        params, candidate_network, mock_data, rl_provider=rl_provider
    )
    if output_path is not None:
        save_dayahead_plan(plan, output_path)
    return plan


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
        description="生成日前基准路径计划（论文 3.2 节）。"
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
