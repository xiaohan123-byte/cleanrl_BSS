"""Deterministic six-station stress cases for waiting and reservation failure.

These cases exercise the realised continuous-time executor directly.  They do
not use a learned RL policy or an optimiser, so they are small enough to serve
as reproducible diagnostics when a normal synthetic run happens to contain no
queueing or reservation timeout.

Run from the repository root with::

    python -m src.cost_stress_scenarios

The default outputs are
``data_generation_test/output/cost_stress_test_results.json`` and
``data_generation_test/output/cost_stress_test_results.md``.
"""

from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.accounting import RealizedLedger
from src.domain import RequestKind, RollingState, SlotState, WaitingRequest
from src.event_engine import ContinuousEventEngine, ExecutionResult
from src.time_grid import TimeGrid


STRESS_SCHEMA_VERSION = 1
NUM_STATIONS = 6
BATTERY_CAPACITY_KWH = 100.0
CHARGING_EFFICIENCY = 0.95
INITIAL_SOC = 0.9
CHARGING_POWER_KW = 60.0
ARRIVAL_TIME = 0.0
DEADLINE = 0.25
RETURN_SOC = 0.2
RESERVATION_FAILURE_PENALTY = 1000.0
DEFAULT_OUTPUT_DIR = Path("data_generation_test/output")
DEFAULT_OUTPUT_STEM = "cost_stress_test_results"


def _initial_state() -> RollingState:
    """Return a six-station, one-slot state with no full battery at station 0."""

    return RollingState(
        now=0.0,
        slots=[
            [SlotState(station, 0, INITIAL_SOC if station == 0 else 1.0)]
            for station in range(NUM_STATIONS)
        ],
    )


def _request(scenario_id: str) -> WaitingRequest:
    return WaitingRequest(
        request_id=f"{scenario_id}-reservation",
        kind=RequestKind.RESERVATION,
        station=0,
        arrival_time=ARRIVAL_TIME,
        deadline=DEADLINE,
        return_soc=RETURN_SOC,
        user_key=(0, 0),
        event_id=f"reservation:stress:{scenario_id}",
    )


def _engine() -> ContinuousEventEngine:
    # A one-hour interval places the 0.25 h deadline inside the interval, so
    # service and timeout postings both have unambiguous realised timestamps.
    return ContinuousEventEngine(
        TimeGrid(interval_hours=1.0, num_intervals=2),
        battery_capacity_kwh=BATTERY_CAPACITY_KWH,
        charging_efficiency=CHARGING_EFFICIENCY,
        max_wait_hours=DEADLINE - ARRIVAL_TIME,
        slot_power_limit_kw=CHARGING_POWER_KW,
    )


def _account(execution: ExecutionResult) -> RealizedLedger:
    """Book only realised entries, with non-target prices held at zero."""

    ledger = RealizedLedger(
        TimeGrid(interval_hours=1.0, num_intervals=2),
        energy_price=0.0,
        reservation_service_price=0.0,
        random_service_price=0.0,
        reservation_failure_penalty=RESERVATION_FAILURE_PENALTY,
        path_adjustment_cost=0.0,
        battery_capacity_kwh=BATTERY_CAPACITY_KWH,
    )
    ledger.submit_many(execution.ledger_entries)
    return ledger


def _scenario_result(
    scenario_id: str,
    description: str,
    requested_station_zero_power_kw: float,
) -> Dict[str, Any]:
    requested_power = [[requested_station_zero_power_kw]] + [
        [0.0] for _ in range(NUM_STATIONS - 1)
    ]
    execution = _engine().simulate_interval(
        _initial_state(),
        interval_index=0,
        requested_power=requested_power,
        arrivals=[_request(scenario_id)],
        realized=True,
    )
    ledger = _account(execution)
    financial = ledger.summary()
    service_waits = [event.wait_hours for event in execution.services]
    timeout_waits = [event.wait_hours for event in execution.timeouts]
    waiting_ids = [
        request.event_id or request.request_id
        for request in execution.state.all_waiting_requests()
    ]
    return {
        "scenario_id": scenario_id,
        "description": description,
        "input": {
            "num_stations": NUM_STATIONS,
            "slots_per_station": 1,
            "station_0_initial_soc": INITIAL_SOC,
            "station_0_power_kw": requested_station_zero_power_kw,
            "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
            "charging_efficiency": CHARGING_EFFICIENCY,
            "request_kind": RequestKind.RESERVATION.value,
            "request_station": 0,
            "arrival_time_hours": ARRIVAL_TIME,
            "deadline_hours": DEADLINE,
            "return_soc": RETURN_SOC,
            "reservation_failure_penalty": RESERVATION_FAILURE_PENALTY,
        },
        "observed": {
            "service_count": len(execution.services),
            "timeout_count": len(execution.timeouts),
            "service_wait_hours": service_waits,
            "timeout_wait_hours": timeout_waits,
            "maximum_observed_wait_hours": max(service_waits + timeout_waits, default=0.0),
            # Waiting is a physical statistic in the current objective, not a
            # separately priced financial component.
            "waiting_cost_defined": False,
            "waiting_cost": 0.0,
            "final_waiting_request_ids": waiting_ids,
            "station_0_final_soc": execution.state.slots[0][0].soc,
            "financial": financial,
        },
        "events": {
            "services": [event.to_dict() for event in execution.services],
            "timeouts": [event.to_dict() for event in execution.timeouts],
            "ledger_postings": [posting.to_dict() for posting in ledger.postings],
        },
    }


def run_waiting_then_service_scenario() -> Dict[str, Any]:
    """Charge from SOC 0.9 at 60 kW, then serve the waiting reservation."""

    result = _scenario_result(
        "waiting_then_service",
        "Station 0 has no full battery at arrival; its slot charges until the request is served.",
        CHARGING_POWER_KW,
    )
    expected_wait = (
        (1.0 - INITIAL_SOC)
        * BATTERY_CAPACITY_KWH
        / (CHARGING_EFFICIENCY * CHARGING_POWER_KW)
    )
    observed = result["observed"]
    result["expected"] = {
        "service_count": 1,
        "timeout_count": 0,
        "wait_hours": expected_wait,
        "reservation_failure_cost": 0.0,
    }
    result["checks"] = {
        "six_station_state": result["input"]["num_stations"] == NUM_STATIONS,
        "wait_is_positive": observed["maximum_observed_wait_hours"] > 0.0,
        "served_before_deadline": (
            observed["service_count"] == 1 and observed["timeout_count"] == 0
        ),
        "wait_matches_charging_equation": (
            len(observed["service_wait_hours"]) == 1
            and isclose(
                observed["service_wait_hours"][0],
                expected_wait,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        ),
        "no_failure_cost": isclose(
            observed["financial"]["reservation_failure_cost"], 0.0, abs_tol=1e-12
        ),
    }
    result["checks"]["all_passed"] = all(result["checks"].values())
    return result


def run_reservation_timeout_scenario() -> Dict[str, Any]:
    """Hold the non-full station-0 slot at 0 kW until the request times out."""

    result = _scenario_result(
        "reservation_timeout",
        "Station 0 has no full battery and receives zero power, so the reservation reaches its deadline.",
        0.0,
    )
    observed = result["observed"]
    result["expected"] = {
        "service_count": 0,
        "timeout_count": 1,
        "wait_hours": DEADLINE - ARRIVAL_TIME,
        "reservation_failure_cost": RESERVATION_FAILURE_PENALTY,
    }
    result["checks"] = {
        "six_station_state": result["input"]["num_stations"] == NUM_STATIONS,
        "not_served": observed["service_count"] == 0,
        "one_reservation_timeout": observed["timeout_count"] == 1,
        "timeout_at_deadline": (
            len(observed["timeout_wait_hours"]) == 1
            and isclose(
                observed["timeout_wait_hours"][0],
                DEADLINE - ARRIVAL_TIME,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "failure_cost_is_1000": isclose(
            observed["financial"]["reservation_failure_cost"],
            RESERVATION_FAILURE_PENALTY,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "reward_is_minus_failure_cost": isclose(
            observed["financial"]["reward_delta"],
            -RESERVATION_FAILURE_PENALTY,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
    }
    result["checks"]["all_passed"] = all(result["checks"].values())
    return result


def build_cost_stress_results() -> Dict[str, Any]:
    """Run both deterministic cases and return one JSON-serialisable report."""

    scenarios = [
        run_waiting_then_service_scenario(),
        run_reservation_timeout_scenario(),
    ]
    return {
        "schema_version": STRESS_SCHEMA_VERSION,
        "generator": "src.cost_stress_scenarios",
        "run_mode": "SIX_STATION_DETERMINISTIC_MOCK_STRESS",
        "num_stations": NUM_STATIONS,
        "rl_implemented": False,
        "path_optimization_used": False,
        "purpose": "Expose positive waiting time and reservation-failure cost in controlled physical executions.",
        "objective_note": (
            "The current accounting model has no monetary waiting-cost term; "
            "waiting is reported in hours. Only reservation timeout creates the configured failure cost."
        ),
        "scenarios": scenarios,
        "checks": {
            "scenario_count": len(scenarios) == 2,
            "all_scenarios_passed": all(
                scenario["checks"]["all_passed"] for scenario in scenarios
            ),
        },
    }


def cost_stress_markdown(results: Mapping[str, Any]) -> str:
    """Render the two diagnostics as a concise Chinese Markdown report."""

    by_id = {item["scenario_id"]: item for item in results["scenarios"]}
    waiting = by_id["waiting_then_service"]
    timeout = by_id["reservation_timeout"]
    wait_hours = waiting["observed"]["service_wait_hours"][0]
    timeout_hours = timeout["observed"]["timeout_wait_hours"][0]
    failure_cost = timeout["observed"]["financial"]["reservation_failure_cost"]
    reward = timeout["observed"]["financial"]["reward_delta"]
    return "\n".join(
        [
            "# 六站 Mock 等待与预约失败成本压力测试",
            "",
            "本报告直接调用连续时间物理事件引擎和真实账本；RL、外部信号与路径优化均未参与。",
            "",
            "| 场景 | 站点 0 初始 SOC | 功率 | 服务 | 超时 | 等待时间（h） | 预约失败成本 | 净收益 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| 充电后服务 | {INITIAL_SOC:.3f} | {CHARGING_POWER_KW:.1f} kW | "
                f"1 | 0 | {wait_hours:.12f} | 0.00 | "
                f"{waiting['observed']['financial']['reward_delta']:.2f} |"
            ),
            (
                f"| 零功率预约超时 | {INITIAL_SOC:.3f} | 0.0 kW | "
                f"0 | 1 | {timeout_hours:.12f} | {failure_cost:.2f} | {reward:.2f} |"
            ),
            "",
            "## 结论",
            "",
            (
                f"- 等待场景在 `t={wait_hours:.12f} h` 获得满电电池并服务。这个值来自 "
                "`(1-0.9)×100/(0.95×60)`。"
            ),
            (
                f"- 超时场景在 `t={timeout_hours:.2f} h` 触发预约失败，真实账本计入 "
                f"`{failure_cost:.2f}` 的预约失败成本。"
            ),
            "- 当前目标函数没有独立的等待货币成本；等待时间作为物理统计量输出，因此等待成本保持 0。",
            f"- 两个场景均使用 {NUM_STATIONS} 个站点，全部自动检查通过：`{results['checks']['all_scenarios_passed']}`。",
            "",
        ]
    )


def write_cost_stress_artifacts(
    results: Mapping[str, Any],
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    *,
    stem: str = DEFAULT_OUTPUT_STEM,
) -> Dict[str, Path]:
    """Write deterministic JSON and Markdown sidecars."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / f"{stem}.json",
        "markdown": destination / f"{stem}.md",
    }
    with paths["json"].open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with paths["markdown"].open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(cost_stress_markdown(results))
    return paths


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Generate deterministic six-station waiting/failure stress results."
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="directory for JSON and Markdown artifacts",
    )
    parser.add_argument("--stem", default=DEFAULT_OUTPUT_STEM, help="artifact filename stem")
    args = parser.parse_args(argv)

    results = build_cost_stress_results()
    paths = write_cost_stress_artifacts(results, args.output_dir, stem=args.stem)
    print("压力测试已生成：" + "，".join(f"{name}={path}" for name, path in paths.items()))
    return results


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OUTPUT_STEM",
    "NUM_STATIONS",
    "RESERVATION_FAILURE_PENALTY",
    "STRESS_SCHEMA_VERSION",
    "build_cost_stress_results",
    "cost_stress_markdown",
    "run_reservation_timeout_scenario",
    "run_waiting_then_service_scenario",
    "write_cost_stress_artifacts",
]
