# -*- coding: utf-8 -*-
"""连续事件 MPC 首区间重放与内部事件分段测试。"""

from __future__ import annotations

import unittest

from data_generation_test.parameter import get_default_parameters
from data_generation_test.rl_data import RLSignals
from src.domain import CandidateRequest, RollingState, SlotState
from src.mpc_model import EventMPCWindowInput, MPCController


def _params():
    params = get_default_parameters()
    params.horizon = 1
    params.validate()
    return params


def _state(params, first_soc: float) -> RollingState:
    return RollingState(
        now=0.0,
        slots=[
            [
                SlotState(
                    station=i,
                    slot=b,
                    soc=(first_soc if (i, b) == (0, 0) else 0.2),
                    last_update_time=0.0,
                )
                for b in range(params.station.num_slots)
            ]
            for i in range(params.station.num_stations)
        ],
    )


def _signals(params, power: float) -> RLSignals:
    requested = [
        [[0.0] for _ in range(params.station.num_slots)]
        for _ in range(params.station.num_stations)
    ]
    requested[0][0][0] = power
    return RLSignals(
        start_period=0,
        horizon=1,
        requested_power=requested,
        terminal_soc_value=[
            [0.0 for _ in range(params.station.num_slots)]
            for _ in range(params.station.num_stations)
        ],
        outside_swap_lambda=[0.0 for _ in range(params.station.num_stations)],
        signal_source="mock",
    )


class EventMPCReplayTests(unittest.TestCase):
    def test_full_at_prediction_end_is_carried_and_next_round_serves_once(self) -> None:
        params = _params()
        # (1 - 0.43) * 100 / (0.95 * 60) == 1 h，精确在右端充满。
        state = _state(params, first_soc=0.43)
        request = CandidateRequest(
            request_id="deadline-at-end",
            kind="reservation",
            station=0,
            arrival_time=0.0,
            deadline=1.0,
            return_soc=0.3,
        )
        controller = MPCController(params, candidate_network={})
        result = controller.solve_step(
            EventMPCWindowInput(
                params=params,
                rolling_state=state,
                period_ell=0,
                rl_signals=_signals(params, 60.0),
                event_requests=[request],
            )
        )

        self.assertEqual(result.request_outcomes["deadline-at-end"], "PENDING_AT_HORIZON")
        self.assertEqual(result.first_stage.assignments, [])
        carried = result.terminal_state.slots[0][0]
        self.assertEqual(carried.soc, 1.0)
        self.assertFalse(carried.ready)
        self.assertEqual(carried.completion_due_at, 1.0)
        self.assertTrue(controller.replay_first_interval(result).matches)

        zeros = [
            [0.0 for _ in range(params.station.num_slots)]
            for _ in range(params.station.num_stations)
        ]
        next_interval = result.event_engine.simulate_interval(
            result.terminal_state, 1, zeros, (), realized=False
        )
        self.assertEqual(
            [(event.request.request_id, event.occurred_at) for event in next_interval.services],
            [("deadline-at-end", 1.0)],
        )
        self.assertEqual(next_interval.timeouts, [])

    def test_swap_creates_new_charge_episode_within_same_interval(self) -> None:
        params = _params()
        state = _state(params, first_soc=1.0)
        request = CandidateRequest(
            request_id="charge-after-swap",
            kind="random",
            station=0,
            arrival_time=0.1,
            deadline=0.35,
            return_soc=0.4,
        )
        controller = MPCController(params, candidate_network={})
        result = controller.solve_step(
            EventMPCWindowInput(
                params=params,
                rolling_state=state,
                period_ell=0,
                rl_signals=_signals(params, 60.0),
                event_requests=[request],
            )
        )

        segments = [
            item for item in result.first_stage.charging_segments
            if item["station"] == 0 and item["slot"] == 0
        ]
        self.assertEqual(len(segments), 1)
        self.assertAlmostEqual(segments[0]["start_time"], 0.1)
        self.assertAlmostEqual(segments[0]["end_time"], 1.0)
        self.assertAlmostEqual(segments[0]["energy_kwh"], 54.0)
        # 只有换入低 SOC 后的 0.9 h 充电计入该槽成本。
        self.assertAlmostEqual(
            result.charging_cost,
            54.0 * params.electricity_price[0][0],
        )
        self.assertAlmostEqual(result.terminal_state.slots[0][0].soc, 0.913)
        self.assertFalse(result.terminal_state.slots[0][0].ready)
        self.assertTrue(controller.replay_first_interval(result).matches)


if __name__ == "__main__":
    unittest.main()
