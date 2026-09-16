import unittest
from copy import deepcopy
from unittest.mock import patch

from src.domain import MPCSolution, ServiceDecision
from src.parameters import BusinessParameters, ODPairParameters, StationParameters
from src.rolling_runner import run_rolling_mpc
from src.scenario import SyntheticScenario


def parameters(periods=4):
    return BusinessParameters(
        num_periods=periods, horizon=2, num_reservations=0, min_swap_spacing_km=0.,
        station=StationParameters(num_stations=2, station_ids=[0, 1], positions_km=[5., 10.],
            num_slots=1, initial_slot_soc=[[1.], [1.]], charging_efficiency=1.,
            slot_power_limits_kw=[[60.], [60.]], station_power_limits_kw=[60., 60.]),
        od_pairs=[ODPairParameters(0, 0., 15., [0, 1])], vehicle_speed_kmh=60.,
        random_arrival_rate_per_hour=[0., 0.])


def answer(params, window, paths=None, services=None):
    return MPCSolution("optimal", 9999., {"income": 9999.}, paths or {}, services or [],
                       [[[0.] * window.horizon] for _ in range(2)], [])


class RollingRunnerTests(unittest.TestCase):
    def test_only_actual_first_interval_actions_affect_reward(self):
        p = parameters()
        scenario = SyntheticScenario(p, [], [
            dict(request_id="R", station=0, arrival_time=2 / 60, return_soc=.2)])
        observed = []
        def scripted(params, window):
            observed.append(list(window.state.waiting))
            service = [ServiceDecision("R", 0, 0, 1)] if window.ell == 1 else []
            return answer(params, window, services=service)
        with patch("src.rolling_runner.solve_mpc", side_effect=scripted):
            result = run_rolling_mpc(p, scenario)
        self.assertEqual(observed[0], [])
        self.assertEqual(observed[1], ["R"])
        self.assertEqual(result["summary"]["random_services"], 1)
        self.assertAlmostEqual(result["summary"]["total_reward"], 96.)
        self.assertEqual(result["final_state"]["slot_soc"][0], [.2])
        self.assertEqual(result["rounds"][1]["state_before"]["slot_soc"][0], [1.])
        self.assertEqual(result["rounds"][2]["state_before"]["slot_soc"][0], [.2])
        self.assertNotIn("ledger", result["rounds"][1]["state_before"])

    def test_future_plan_retained_and_entry_publication_free_then_one_change_fee(self):
        p = parameters(6)
        scenario = SyntheticScenario(p, [
            dict(user_key=[0, 0], od_id=0, entry_time=.2, entry_soc=.9)], [])
        def scripted(params, window):
            route = [0] if window.ell < 3 else ([1] if window.ell < 5 else [])
            return answer(params, window, paths={"0:0": route})
        with patch("src.rolling_runner.solve_mpc", side_effect=scripted):
            result = run_rolling_mpc(p, scenario)
        rows = result["rounds"]
        self.assertEqual(rows[0]["state_after"]["users"]["0:0"]["retained_plan"], [0])
        self.assertIsNone(rows[1]["state_after"]["users"]["0:0"]["published_plan"])
        self.assertEqual(rows[2]["state_after"]["users"]["0:0"]["published_plan"], [0])
        changes = [item for item in result["ledger"] if item["type"] == "path_adjustment"]
        self.assertEqual([item["period"] for item in changes], [3])
        self.assertEqual(result["summary"]["adjustment_cost"], p.path_adjustment_penalty)

    def test_solver_and_horizon_overrides_preserve_scenario_truth(self):
        p = parameters()
        scenario = SyntheticScenario(p, [], [])
        p.horizon = 3
        p.solver.time_limit_sec = 2.
        with patch("src.rolling_runner.solve_mpc", side_effect=answer):
            result = run_rolling_mpc(p, scenario)
        self.assertEqual(result["rounds"][0]["horizon"], 3)
        self.assertEqual(scenario.params["horizon"], 2)

    def test_changed_physical_parameters_rejected(self):
        p = parameters()
        scenario = SyntheticScenario(p, [], [])
        p.battery_capacity_kwh = 150.
        with self.assertRaisesRegex(ValueError, "physical parameters"):
            run_rolling_mpc(p, scenario)


if __name__ == "__main__":
    unittest.main()
