# -*- coding: utf-8 -*-
"""“有满电池就必须按 FCFS 服务”的回归测试。"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from data_generation_test.candidate_network import (
    generate_candidate_network,
    validate_candidate_network,
)
from data_generation_test.parameter import get_default_parameters
from data_generation_test.rl_data import MockRLProvider, RLSignals, generate_mock_data
from run_mpc import _execute_first_stage, run_rolling_mpc
from src.dayahead_plan import generate_dayahead_plan, validate_dayahead_plan
from src.mpc_model import (
    FirstStageExecution,
    FixedCommitment,
    FixedSwapEvent,
    MPCController,
    MPCWindowInput,
    ReservationObservation,
    RollingState,
)


class ExecutionPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.params = get_default_parameters()
        st = self.params.station
        self.soc = [
            [0.5 for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        self.soc[0][0] = 1.0
        self.soc[0][1] = 1.0
        self.power = [
            [0.0 for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        self.ready = [
            [0 for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        self.ready[0][0] = 1
        self.ready[0][1] = 1
        self.requests = [[] for _ in range(st.num_stations)]
        self.requests[0] = [
            {"request_id": "late", "arrival_time": 0.8, "arrival_soc": 0.4},
            {"request_id": "first", "arrival_time": 0.1, "arrival_soc": 0.3},
            {"request_id": "second", "arrival_time": 0.2, "arrival_soc": 0.35},
        ]

    def _result(self, ready=None, available=None):
        selected_ready = self.ready if ready is None else ready
        fs = FirstStageExecution(
            period=0,
            power_kw=self.power,
            ready=selected_ready,
            available_full=(
                [sum(row) for row in selected_ready]
                if available is None
                else available
            ),
            assignments=[],
        )
        return SimpleNamespace(first_stage=fs)

    def test_all_physical_full_batteries_serve_fcfs_prefix(self) -> None:
        out = _execute_first_stage(
            self.params,
            0,
            self.soc,
            self._result(),
            res_events=[],
            actual_random=self.requests,
        )
        self.assertEqual(
            [item["request_id"] for item in out["served_rand"]],
            ["first", "second"],
        )
        self.assertEqual(
            [item["request_id"] for item in out["rejected_rand"]],
            ["late"],
        )

    def test_withheld_physical_full_battery_is_rejected(self) -> None:
        ready = [list(row) for row in self.ready]
        ready[0][1] = 0
        with self.assertRaisesRegex(RuntimeError, "物理满电槽与 g 不一致"):
            _execute_first_stage(
                self.params,
                0,
                self.soc,
                self._result(ready=ready),
                res_events=[],
                actual_random=self.requests,
            )

    def test_reservation_shortage_is_penalizable_and_blocks_random(self) -> None:
        reservations = [
            {
                "station": 0,
                "od_id": 0,
                "user_id": user_id,
                "return_soc": 0.3,
                "sort_key": (0, user_id, 0),
            }
            for user_id in range(3)
        ]
        out = _execute_first_stage(
            self.params,
            0,
            self.soc,
            self._result(),
            res_events=reservations,
            actual_random=self.requests,
        )
        self.assertEqual(len(out["served_res"]), 2)
        self.assertEqual(
            [item["user_id"] for item in out["failed_res"]], [2]
        )
        self.assertEqual(out["served_rand"], [])
        self.assertEqual(
            [item["request_id"] for item in out["rejected_rand"]],
            ["first", "second", "late"],
        )

    def test_same_period_downstream_event_is_inactive_after_failure(self) -> None:
        st = self.params.station
        soc = [
            [0.5 for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        soc[1][0] = 1.0
        ready = [
            [0 for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        ready[1][0] = 1
        reservations = [
            {
                "station": 0,
                "od_id": 0,
                "user_id": 99,
                "return_soc": 0.2,
                "path_order": 0,
                "sort_key": (0, 0, 99, 0.0),
            },
            {
                "station": 1,
                "od_id": 0,
                "user_id": 99,
                "return_soc": 0.3,
                "path_order": 1,
                "sort_key": (0, 0, 99, 1.0),
            },
        ]
        out = _execute_first_stage(
            self.params,
            0,
            soc,
            self._result(ready=ready),
            res_events=reservations,
            actual_random=[[] for _ in range(st.num_stations)],
        )
        self.assertEqual(
            [
                (item["station"], item["user_id"])
                for item in out["failed_res"]
            ],
            [(0, 99)],
        )
        self.assertEqual(out["served_res"], [])
        self.assertEqual(out["new_soc"][1][0], 1.0)


class Seed42IntegrationTest(unittest.TestCase):
    def test_no_false_shortage_and_fcfs_in_every_round(self) -> None:
        params = get_default_parameters()
        network = generate_candidate_network(params)
        validate_candidate_network(network, params)
        mock = generate_mock_data(params, seed=42)
        provider = MockRLProvider(params)
        plan = generate_dayahead_plan(
            params, network, mock, rl_provider=provider
        )
        validate_dayahead_plan(
            plan, params, network, mock, rl_provider=provider
        )
        result = run_rolling_mpc(
            params, network, mock, plan, rl_provider=provider
        )
        self.assertEqual(
            json.loads(json.dumps(result, ensure_ascii=False)),
            result,
        )

        threshold = 1.0 - 0.5 * params.full_soc_tolerance
        eta = params.station.charging_efficiency
        factor = eta * params.delta_hours / params.battery_capacity_kwh
        rejected_total = 0
        predicted_failure_rounds = 0
        for rd in result["rounds"]:
            ell = rd["ell"]
            for i in range(params.station.num_stations):
                charged_soc = [
                    rd["soc_obs_start"][i][b] + factor * rd["power_kw"][i][b]
                    for b in range(params.station.num_slots)
                ]
                physical_ready = [int(s >= threshold) for s in charged_soc]
                self.assertEqual(rd["ready"][i], physical_ready)
                self.assertEqual(rd["available_full"][i], sum(physical_ready))

                res_count = sum(
                    item["station"] == i
                    for item in rd["actual"]["reservation_swaps"]
                )
                requests = sorted(
                    mock["actual_random_requests"][i][ell],
                    key=lambda item: (item["arrival_time"], item["request_id"]),
                )
                capacity = max(0, rd["available_full"][i] - res_count)
                split = min(len(requests), capacity)
                served_ids = [
                    item["request_id"]
                    for item in rd["actual"]["random_served"]
                    if item["station"] == i
                ]
                rejected_ids = [
                    item["request_id"]
                    for item in rd["actual"]["random_rejected"]
                    if item["station"] == i
                ]
                self.assertEqual(
                    served_ids, [item["request_id"] for item in requests[:split]]
                )
                self.assertEqual(
                    rejected_ids, [item["request_id"] for item in requests[split:]]
                )
                rejected_total += len(rejected_ids)

            predicted_failure_rounds += int(
                rd["predicted"]["reservation_failures"] > 0
            )
            expected_objective = (
                rd["objective"]["income_reservation"]
                + rd["objective"]["income_random"]
                - rd["objective"]["charging_cost"]
                - rd["objective"]["adjustment_cost"]
                - rd["objective"]["reservation_failure_cost"]
                + params.terminal_value_weight
                * rd["objective"]["terminal_value"]
            )
            self.assertAlmostEqual(
                rd["objective"]["total"], expected_objective, places=5
            )

        self.assertTrue(all(rd["status"] == "OPTIMAL" for rd in result["rounds"]))
        self.assertGreater(predicted_failure_rounds, 0)
        self.assertEqual(rejected_total, 0)
        self.assertEqual(result["summary"]["total_actual_reservation_failures"], 0)
        self.assertEqual(result["summary"]["total_actual_random_served"], 35)
        self.assertEqual(
            result["service_policy"]["random_policy"],
            "mandatory_fcfs_when_full_battery_available",
        )


class ReservationAliveChainTest(unittest.TestCase):
    @staticmethod
    def _zero_signals(params, outside_lambda=None) -> RLSignals:
        st = params.station
        zeros = [
            [[0.0 for _ in range(params.horizon)] for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        return RLSignals(
            start_period=0,
            horizon=params.horizon,
            requested_power=zeros,
            terminal_soc_value=[
                [0.0 for _ in range(st.num_slots)]
                for _ in range(st.num_stations)
            ],
            outside_swap_lambda=(
                [0.0 for _ in range(st.num_stations)]
                if outside_lambda is None
                else list(outside_lambda)
            ),
        )

    def test_fixed_user_is_not_served_downstream_after_failure(self) -> None:
        params = get_default_parameters()
        network = generate_candidate_network(params)
        st = params.station
        soc = [
            [0.2 for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        soc[1][0] = 1.0  # 下游站有电，但用户应在上游失败后失活。
        zeros = [
            [[0.0 for _ in range(params.horizon)] for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        signals = RLSignals(
            start_period=0,
            horizon=params.horizon,
            requested_power=zeros,
            terminal_soc_value=[
                [0.0 for _ in range(st.num_slots)]
                for _ in range(st.num_stations)
            ],
            outside_swap_lambda=[0.0 for _ in range(st.num_stations)],
        )
        fixed = FixedCommitment(
            od_id=1,
            user_id=99,
            fixed_path_arcs=[("entry", 0), (0, 1), (1, "exit")],
            remaining_events=[
                FixedSwapEvent(station=0, period=0, return_soc=0.2),
                FixedSwapEvent(station=1, period=1, return_soc=0.2),
            ],
        )
        window = MPCWindowInput(
            params=params,
            rolling_state=RollingState(soc_obs=soc, period_ell=0),
            fixed_commitments=[fixed],
            rl_signals=signals,
        )
        result = MPCController(params, network).solve_step(window)
        fixed_events = [event for event in result.events if event["kind"] == "fix"]
        self.assertEqual(
            [
                (
                    event["reservation_active"],
                    event["reservation_served"],
                    event["reservation_failed"],
                )
                for event in fixed_events
            ],
            [(1, 0, 1), (0, 0, 0)],
        )
        self.assertAlmostEqual(
            result.reservation_failure_cost,
            params.reservation_failure_penalty,
        )

    def test_decision_outside_value_requires_boundary_alive(self) -> None:
        for has_inner_inventory in (False, True):
            with self.subTest(has_inner_inventory=has_inner_inventory):
                params = get_default_parameters()
                params.horizon = 2
                network = generate_candidate_network(params)
                st = params.station
                soc = [
                    [0.2 for _ in range(st.num_slots)]
                    for _ in range(st.num_stations)
                ]
                if has_inner_inventory:
                    soc[0][0] = 1.0
                observation = ReservationObservation(
                    od_id=1,
                    user_id=77,
                    effective_entry_time=0.0,
                    effective_entry_soc=0.3,
                    baseline_path_arcs=[
                        ("entry", 0),
                        (0, 1),
                        (1, "exit"),
                    ],
                    is_new_arrival=False,
                )
                window = MPCWindowInput(
                    params=params,
                    rolling_state=RollingState(soc_obs=soc, period_ell=0),
                    reservations=[observation],
                    rl_signals=self._zero_signals(params),
                )
                bundle = MPCController(params, network).build_model(window)
                bundle.model.optimize()
                user_key = (1, 77)
                self.assertEqual(
                    round(bundle.dec_boundary_alive[user_key].X),
                    int(has_inner_inventory),
                )
                selected_out = [
                    (key, var)
                    for key, var in bundle.out_event_active.items()
                    if bundle.y[key].X > 0.5
                ]
                self.assertEqual(len(selected_out), 1)
                self.assertEqual(
                    round(selected_out[0][1].X),
                    int(has_inner_inventory),
                )
                self.assertTrue(
                    all(
                        var.X < 0.5
                        for key, var in bundle.out_event_active.items()
                        if bundle.y[key].X < 0.5
                    )
                )

    def test_fixed_outside_reference_value_is_reversed_after_failure(self) -> None:
        params = get_default_parameters()
        params.horizon = 2
        network = generate_candidate_network(params)
        st = params.station
        soc = [
            [0.2 for _ in range(st.num_slots)]
            for _ in range(st.num_stations)
        ]
        fixed = FixedCommitment(
            od_id=1,
            user_id=88,
            fixed_path_arcs=[("entry", 0), (0, 1), (1, "exit")],
            remaining_events=[
                FixedSwapEvent(station=0, period=0, return_soc=0.2),
                FixedSwapEvent(station=1, period=2, return_soc=0.2),
            ],
        )
        signals = self._zero_signals(
            params, outside_lambda=[10.0 for _ in range(st.num_stations)]
        )
        window = MPCWindowInput(
            params=params,
            rolling_state=RollingState(soc_obs=soc, period_ell=0),
            fixed_commitments=[fixed],
            rl_signals=signals,
        )
        bundle = MPCController(params, network).build_model(window)
        bundle.model.optimize()
        self.assertEqual(round(bundle.fixed_boundary_alive[(1, 88)].X), 0)
        expected_reversal = -signals.outside_swap_value(1, 0.2)
        self.assertAlmostEqual(
            bundle.obj_parts["terminal_value"].getValue(),
            expected_reversal,
        )


if __name__ == "__main__":
    unittest.main()
