# -*- coding: utf-8 -*-
"""连续事件 MPC 输入契约与半开时间边界测试。

这些用例不调用旧离散 Gurobi 模型。它们验证新的 replay-first MPC 分支：
Mock 功率是固定参数，服务与等待由共享 ContinuousEventEngine 决定。
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from data_generation_test.parameter import get_default_parameters
from data_generation_test.rl_data import RLSignals
from src.domain import CandidateRequest, RollingState, SlotState
from src.mpc_model import (
    EventMPCModelBundle,
    EventMPCWindowInput,
    MPCController,
    MPCInputError,
)
import src.mpc_model as mpc_model


def _state(params, *, first_slot_soc: float = 1.0) -> RollingState:
    slots = [
        [
            SlotState(
                station=i,
                slot=b,
                soc=(first_slot_soc if (i, b) == (0, 0) else 0.2),
                last_update_time=0.0,
            )
            for b in range(params.station.num_slots)
        ]
        for i in range(params.station.num_stations)
    ]
    return RollingState(now=0.0, slots=slots)


def _signals(params, period: int, *, power_000: float = 0.0) -> RLSignals:
    power = [
        [[0.0 for _ in range(params.horizon)] for _ in range(params.station.num_slots)]
        for _ in range(params.station.num_stations)
    ]
    power[0][0][0] = power_000
    return RLSignals(
        start_period=period,
        horizon=params.horizon,
        requested_power=power,
        terminal_soc_value=[
            [0.0 for _ in range(params.station.num_slots)]
            for _ in range(params.station.num_stations)
        ],
        outside_swap_lambda=[0.0 for _ in range(params.station.num_stations)],
        signal_source="mock",
    )


class EventMPCInputContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = get_default_parameters()
        self.params.horizon = 1
        self.params.validate()
        self.controller = MPCController(self.params, candidate_network={})

    def test_near_right_endpoint_remains_in_current_half_open_interval(self) -> None:
        # 不能把正常的 t_end-5e-8 事件吸附到下一轮。
        near_end = 1.0 - 5e-8
        request = CandidateRequest(
            request_id="near-end",
            kind="random",
            station=0,
            arrival_time=near_end,
            deadline=near_end + 0.1,
            return_soc=0.3,
        )
        window = EventMPCWindowInput(
            params=self.params,
            rolling_state=_state(self.params),
            period_ell=0,
            rl_signals=_signals(self.params, 0, power_000=60.0),
            event_requests=[request],
        )
        bundle = self.controller.build_model(window)
        self.assertIsInstance(bundle, EventMPCModelBundle)
        self.assertIsNone(bundle.model)
        self.assertEqual(bundle.candidate_event_count, 1)

        result = self.controller.solve_step(window)
        self.assertEqual(result.request_outcomes["near-end"], "SERVED_IN_HORIZON")
        self.assertEqual(len(result.first_stage.assignments), 1)
        self.assertAlmostEqual(
            result.first_stage.assignments[0]["occurred_at"], near_end, places=12
        )
        # 60 kW 是新 Mock 合法的逐槽上限，不再受旧 p_tol 裕量限制。
        self.assertEqual(result.first_stage.power_kw[0][0], 60.0)

    def test_exact_prediction_right_endpoint_is_pending(self) -> None:
        request = CandidateRequest(
            request_id="at-end",
            kind="reservation",
            station=0,
            arrival_time=1.0,
            deadline=1.25,
            return_soc=0.3,
        )
        result = self.controller.solve_step(
            EventMPCWindowInput(
                params=self.params,
                rolling_state=_state(self.params),
                period_ell=0,
                rl_signals=_signals(self.params, 0),
                event_requests=[request],
            )
        )
        self.assertEqual(result.request_outcomes["at-end"], "PENDING_AT_HORIZON")
        self.assertEqual(result.first_stage.assignments, [])
        self.assertIn("at-end", result.pending_request_ids)

    def test_interval_energy_limit_is_checked_without_station_power_cap(self) -> None:
        self.params.station_energy_limit_kwh[0][0] = 10.0
        self.params.validate()
        with self.assertRaisesRegex(MPCInputError, "请求能量"):
            self.controller.build_model(
                EventMPCWindowInput(
                    params=self.params,
                    rolling_state=_state(self.params),
                    period_ell=0,
                    rl_signals=_signals(self.params, 0, power_000=60.0),
                    event_requests=[],
                )
            )

    def test_only_mock_signal_source_is_accepted(self) -> None:
        signals = _signals(self.params, 0)
        signals.signal_source = "trained"
        with self.assertRaisesRegex(MPCInputError, "仅允许 Mock"):
            self.controller.build_model(
                EventMPCWindowInput(
                    params=self.params,
                    rolling_state=_state(self.params),
                    period_ell=0,
                    rl_signals=signals,
                )
            )

    def test_path_search_adjustment_cost_is_prediction_only_objective_term(self) -> None:
        result = self.controller.solve_step(
            EventMPCWindowInput(
                params=self.params,
                rolling_state=_state(self.params),
                period_ell=0,
                rl_signals=_signals(self.params, 0),
                planned_adjustment_cost=7.0,
                reference_context={"path_search_enabled": True},
            )
        )
        self.assertEqual(result.status, "EVENT_PATH_ENUM_REPLAY")
        self.assertEqual(result.adjustment_cost, 7.0)
        self.assertAlmostEqual(result.objective_total, -7.0)
        self.assertEqual(
            result.model_statistics["model_kind"],
            "event_path_enumeration_mock",
        )

    def test_event_path_branch_does_not_require_gurobi_runtime(self) -> None:
        window = EventMPCWindowInput(
            params=self.params,
            rolling_state=_state(self.params),
            period_ell=0,
            rl_signals=_signals(self.params, 0),
            reference_context={"path_search_enabled": True},
        )
        with patch.object(mpc_model, "gp", None), patch.object(
            mpc_model, "GRB", None
        ):
            result = self.controller.solve_step(window)
        self.assertEqual(result.status, "EVENT_PATH_ENUM_REPLAY")


if __name__ == "__main__":
    unittest.main()
