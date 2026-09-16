# -*- coding: utf-8 -*-
r"""
滚动 MPC 运行入口：论文“滚动时域优化、连续事件执行”机制的驱动脚本。

当前公开入口 ``main`` / ``run_rolling_mpc`` 委托
``src.continuous_runner.run_continuous_rolling_mpc``：它加载或生成六站
schema-2 synthetic/mock 输入和连续日前计划，以本机 Gurobi 运行
paper/main.tex 第 3.3 节及附录 B--D 的路径--事件联合 MILP
（``src.paper_mpc.solve_paper_mpc``），RL 功率和终端参数仍来自 Mock。
输出 schema-3 结果 JSON、真实事件账本与统计文件；所有外部字段和输出
都来自带 seed 的模拟数据，此入口不训练或加载 RL 模型。

用法
----
    python run_mpc.py --seed 42 --regenerate
    python run_mpc.py                      # 复用已生成的输入文件
    python run_mpc.py --time-limit 60 --solver-log
    python run_mpc.py --network PATH --mock-data PATH --plan PATH         --output PATH

依赖：标准库 + gurobipy + data_generation_test 三模块 + src/ 下连续事件
架构模块（continuous_runner / paper_mpc / event_engine / domain 等）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data_generation_test.parameter import (  # noqa: E402
    BusinessParameters,
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
from src.continuous_runner import run_continuous_rolling_mpc
from src.result_statistics import (
    build_result_statistics,
    write_statistics_artifacts,
)

__all__ = ["DEFAULT_RESULT_PATH", "main", "run_rolling_mpc"]

SCHEMA_VERSION = 3

# 默认输出路径：data_generation_test/output/mpc_run_result.json
DEFAULT_RESULT_PATH = (
    _REPO_ROOT / "data_generation_test" / "output" / "mpc_run_result.json"
)

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
# CLI
# ----------------------------------------------------------------------
def run_rolling_mpc(
    params: BusinessParameters,
    network: dict,
    mock: dict,
    plan: dict,
    rl_provider=None,
) -> Dict[str, Any]:
    """运行 schema-3 连续事件滚动 MPC（委托 src.continuous_runner）。"""


    return run_continuous_rolling_mpc(
        params, network, mock, plan, rl_provider=rl_provider
    )


def _print_round_brief(result: Dict[str, Any]) -> None:
    if result.get("run_mode") in {
        "continuous_event_mock_path_search",
        "paper_gurobi_continuous_event_mpc",
    }:
        print("=== 连续事件滚动摘要 ===")
        for rd in result["rounds"]:
            actual = rd["actual"]
            print(
                f"q={rd['period']}: {rd['status']}; "
                f"预测目标={rd['prediction']['objective_total']:.3f}; "
                f"实际预约服务 {len(actual['reservation_services'])}，"
                f"随机服务 {len(actual['random_services'])}，"
                f"预约超时 {len(actual['reservation_timeouts'])}，"
                f"路径发布 {sum(bool(item.get('publication_event_id')) for item in rd.get('path_decisions', []))}，"
                f"首区间回放={'通过' if rd['replay']['matches'] else '失败'}"
            )
        summary = result["summary"]
        print(
            "\n=== 实际账本汇总 ===\n"
            f"预约收入 {summary['total_income_reservation']:.3f}；"
            f"随机收入 {summary['total_income_random']:.3f}；"
            f"充电成本 {summary['total_charging_cost']:.3f}；"
            f"路径调整成本 {summary['total_adjustment_cost']:.3f}；"
            f"预约失败成本 {summary['total_reservation_failure_cost']:.3f}；"
            f"总收益 {summary['total_reward']:.3f}"
        )
        return


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

    # 统计只从已实现事件账本汇总；滚动预测目标因窗口重叠不参与累计。
    statistics = build_result_statistics(
        reloaded,
        mock=mock,
        plan=plan,
        strict=True,
        source_sha256=hashlib.sha256(out_path.read_bytes()).hexdigest(),
    )
    result_stem = out_path.stem
    statistics_stem = (
        result_stem[:-7] if result_stem.endswith("_result") else result_stem
    )
    statistics_paths = write_statistics_artifacts(
        statistics,
        out_path.parent,
        stem=statistics_stem,
    )
    print(
        "统计文件已生成: "
        + "；".join(str(path) for path in statistics_paths.values())
    )
    return result


if __name__ == "__main__":
    main()
