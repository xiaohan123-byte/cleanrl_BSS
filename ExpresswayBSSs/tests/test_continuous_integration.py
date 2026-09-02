from __future__ import annotations

import copy
import json
import unittest

from data_generation_test.candidate_network import generate_candidate_network
from data_generation_test.parameter import get_default_parameters
from data_generation_test.rl_data import (
    DEFAULT_MOCK_DATA_PATH,
    MockRLProvider,
    generate_mock_data,
    load_mock_data,
)
from run_mpc import DEFAULT_RESULT_PATH, run_rolling_mpc
from src.dayahead_plan import (
    DEFAULT_PLAN_PATH,
    generate_dayahead_plan,
    load_dayahead_plan,
    validate_dayahead_plan,
)


class ContinuousSixStationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.params = get_default_parameters()
        cls.network = generate_candidate_network(cls.params)
        cls.mock = generate_mock_data(cls.params, seed=42)
        cls.provider = MockRLProvider(cls.params)
        cls.plan = generate_dayahead_plan(
            cls.params, cls.network, cls.mock, rl_provider=cls.provider
        )

    def test_dayahead_schema_two_replays_with_continuous_kernel(self) -> None:
        self.assertEqual(self.params.station.num_stations, 6)
        self.assertEqual(self.plan["schema_version"], 2)
        self.assertEqual(self.plan["engine"], "continuous_event_v2")
        validate_dayahead_plan(
            self.plan, self.params, self.network, self.mock, rl_provider=self.provider
        )
        self.assertEqual(len(self.plan["baseline_station_visits"]), 6)
        for record in self.plan["reservations"]:
            if not record["accepted"]:
                continue
            self.assertEqual(
                len(record["swap_stations"]), len(record["swap_times"])
            )
            self.assertEqual(
                len(record["swap_stations"]), len(record["event_ids"])
            )

    def test_rolling_runner_optimises_mock_paths_and_uses_actual_ledger_only(self) -> None:
        result = run_rolling_mpc(
            self.params,
            self.network,
            self.mock,
            self.plan,
            rl_provider=self.provider,
        )
        self.assertEqual(result["schema_version"], 3)
        self.assertEqual(result["run_mode"], "paper_gurobi_continuous_event_mpc")
        self.assertEqual(result["num_stations"], 6)
        self.assertEqual(len(result["rounds"]), self.params.num_periods)
        self.assertTrue(
            all(
                round_["status"] == "PAPER_GUROBI_OPTIMAL"
                for round_ in result["rounds"]
            )
        )
        self.assertTrue(
            all(
                round_["model_statistics"]["solver_backend"] == "gurobi"
                and round_["model_statistics"]["selected_pattern_replay_validated"]
                and round_["model_statistics"]["global_milp_optimality_claimed"]
                for round_ in result["rounds"]
            )
        )
        self.assertTrue(all(round_["replay"]["matches"] for round_ in result["rounds"]))
        self.assertEqual(json.loads(json.dumps(result)), result)

        # Reservation 0 is visible at t=0 and its next trajectory point is
        # visible at t=1.  The runner must refresh live en-route facts rather
        # than retaining only the entry snapshot.
        user_zero = result["rounds"][1]["state_end"]["enroute"]["1:0"]
        self.assertAlmostEqual(user_zero["current_position"], 75.0)
        self.assertAlmostEqual(user_zero["vehicle_soc"], 0.75)

        terminal = {
            ":".join(str(part) for part in key)
            for key in (
                result["summary"]["completed_reservations"]
                + result["summary"]["failed_reservations"]
            )
        }
        self.assertFalse(terminal.intersection(result["final_state"]["enroute"]))

        ledger = result["ledger"]
        event_ids = [posting["entry"]["event_id"] for posting in ledger["postings"]]
        self.assertEqual(len(event_ids), len(set(event_ids)))
        self.assertAlmostEqual(
            result["summary"]["total_reward"], ledger["summary"]["reward_delta"]
        )
        forbidden = {
            "pending_at_horizon",
            "terminal_soc_value",
            "outside_delivery_value",
        }
        self.assertFalse(
            forbidden.intersection(
                posting["entry"]["event_type"] for posting in ledger["postings"]
            )
        )

        # The selected route is committed to the physical rollout, not merely
        # reported by the forecast.
        user_zero_services = [
            posting["entry"]["station"]
            for posting in ledger["postings"]
            if posting["entry"]["event_type"] == "reservation_service"
            and posting["entry"]["user_key"] == [1, 0]
        ]
        self.assertTrue(user_zero_services)
        self.assertGreater(result["summary"]["total_path_publications"], 0)
        self.assertEqual(
            result["summary"]["total_adjustment_cost"],
            result["summary"]["total_path_publications"]
            * self.params.path_adjustment_penalty,
        )

    def test_default_artifacts_are_current_six_station_mock_outputs(self) -> None:
        """默认 CLI 输入/输出不应保留已废弃的离散 schema。"""
        mock = load_mock_data(DEFAULT_MOCK_DATA_PATH)
        plan = load_dayahead_plan(DEFAULT_PLAN_PATH)
        with DEFAULT_RESULT_PATH.open("r", encoding="utf-8") as handle:
            result = json.load(handle)

        self.assertEqual(mock["schema_version"], 2)
        self.assertEqual(mock["parameter_snapshot"]["station"]["num_stations"], 6)
        self.assertEqual(plan["schema_version"], 2)
        self.assertEqual(plan["engine"], "continuous_event_v2")
        self.assertEqual(result["schema_version"], 3)
        self.assertEqual(result["run_mode"], "paper_gurobi_continuous_event_mpc")
        self.assertEqual(result["num_stations"], 6)

    def test_dayahead_rejects_a_request_that_only_becomes_ready_after_arrival(self) -> None:
        """日前接纳不能把线上等待窗当作库存可用性。"""
        params = copy.deepcopy(self.params)
        # Give every slot 60 kW (no station-level scaling): from SOC 0.3 it
        # becomes full at about 1.228 h, after the first-station arrival at
        # 1.067 h but before its online deadline at 1.317 h.
        params.station_energy_limit_kwh = [
            [300.0] * params.num_periods
            for _ in range(params.station.num_stations)
        ]
        params.validate()
        mock = generate_mock_data(params, seed=42)
        mock["reservations"] = [
            {
                "reservation_id": 0,
                "request_id": "strict-arrival-probe",
                "od_id": 0,
                "day_ahead_entry_time": 0.0,
                "day_ahead_entry_soc": 0.5,
            }
        ]
        mock["initial_slot_soc"] = [
            [0.3] * params.station.num_slots
            for _ in range(params.station.num_stations)
        ]

        plan = generate_dayahead_plan(
            params,
            generate_candidate_network(params),
            mock,
            rl_provider=MockRLProvider(params),
        )
        self.assertFalse(plan["reservations"][0]["accepted"])
        self.assertIn("strict inventory", plan["reservations"][0]["reject_reason"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
