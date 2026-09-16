# -*- coding: utf-8 -*-
"""连续事件滚动主流程的集成回归测试。"""

from __future__ import annotations

import json
import unittest

from data_generation_test.candidate_network import (
    generate_candidate_network,
    validate_candidate_network,
)
from data_generation_test.parameter import get_default_parameters
from data_generation_test.rl_data import MockRLProvider, generate_mock_data
from run_mpc import run_rolling_mpc
from src.dayahead_plan import generate_dayahead_plan, validate_dayahead_plan


class Seed42IntegrationTest(unittest.TestCase):
    def test_continuous_mock_run_replays_and_keeps_actual_ledger(self) -> None:
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
        self.assertEqual(json.loads(json.dumps(result, ensure_ascii=False)), result)
        self.assertEqual(result["run_mode"], "paper_gurobi_continuous_event_mpc")
        self.assertEqual(result["num_stations"], 6)
        self.assertTrue(all(round_["replay"]["matches"] for round_ in result["rounds"]))
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
                for round_ in result["rounds"]
            )
        )
        self.assertGreater(result["summary"]["total_path_publications"], 0)
        self.assertAlmostEqual(
            result["summary"]["total_reward"],
            result["ledger"]["summary"]["reward_delta"],
        )
        self.assertFalse(result["summary"]["final_waiting_request_ids"])


if __name__ == "__main__":
    unittest.main()
