import copy
import tempfile
import unittest
from pathlib import Path

from src.candidate_network import generate_candidate_network
from src.dayahead_plan import generate_dayahead_plan
from src.forecast import build_forecast
from src.parameters import BusinessParameters, get_default_parameters
from src.scenario import SyntheticScenario, generate_synthetic_scenario, load_scenario, save_scenario


class ScenarioTests(unittest.TestCase):
    def test_seed_reproduction_and_reachable_reservations(self):
        params = get_default_parameters()
        one = generate_synthetic_scenario(params)
        two = generate_synthetic_scenario(params)
        self.assertEqual(one.to_dict(), two.to_dict())
        self.assertNotEqual(one.to_dict(), generate_synthetic_scenario(params, seed=43).to_dict())
        plan = generate_dayahead_plan(params, generate_candidate_network(params), one.initial_reservations())
        self.assertEqual(len(plan), params.num_reservations)

    def test_future_truth_is_not_visible_or_used_by_forecast(self):
        params = get_default_parameters()
        reservation = {"user_key": [0, 0], "od_id": 0, "entry_time": .5, "entry_soc": .9}
        actual = [{"request_id": "past", "station": 0, "arrival_time": 0., "return_soc": .2},
                  {"request_id": "secret", "station": 1, "arrival_time": 1., "return_soc": .3}]
        one = SyntheticScenario(params, [reservation], actual)
        two = SyntheticScenario(params, [reservation], actual[:1])
        obs = one.observation_at(0.)
        self.assertEqual(obs, two.observation_at(0.))
        self.assertNotIn("actual_entry_time", obs.reservations[0])
        self.assertEqual(one.observation_at(.5).reservations[0]["actual_entry_time"], .5)
        self.assertEqual(build_forecast(params, obs, 0, 48), build_forecast(params, two.observation_at(0.), 0, 48))
        self.assertFalse(any(request["request_id"] in {"past", "secret"} for request in build_forecast(params, obs, 0, 48).random_requests))

    def test_copy_isolation_and_half_open_arrivals(self):
        params = get_default_parameters()
        records = [{"request_id": "at_end", "station": 0, "arrival_time": 1., "return_soc": .2}]
        scenario = SyntheticScenario(params, [], records)
        records[0]["return_soc"] = .9
        params.station.initial_slot_soc[0][0] = 0.
        self.assertEqual(scenario.actual_random_requests[0]["return_soc"], .2)
        self.assertEqual(scenario.params["station"]["initial_slot_soc"][0][0], 1.)
        self.assertEqual(scenario.arrivals_between(0., 1.), [])
        observed = scenario.observation_at(1.)
        observed.random_history[0]["return_soc"] = .8
        self.assertEqual(scenario.arrivals_between(1., 2.)[0]["return_soc"], .2)

    def test_hourly_rates_do_not_scale_with_grid_resolution(self):
        params = get_default_parameters()
        coarse = BusinessParameters.from_dict({"num_periods": 12, "interval_hours": 1., "horizon": 4})
        fine_truth = generate_synthetic_scenario(params)
        coarse_truth = generate_synthetic_scenario(coarse)
        self.assertEqual(fine_truth.actual_random_requests, coarse_truth.actual_random_requests)
        self.assertEqual(fine_truth.reservations, coarse_truth.reservations)
        self.assertTrue(all(0. <= record['entry_time'] <= 2. for record in fine_truth.reservations))
        self.assertEqual(build_forecast(params, fine_truth.observation_at(0.), 0, 48), build_forecast(coarse, coarse_truth.observation_at(0.), 0, 4))

    def test_json_roundtrip_and_parameter_validation(self):
        params = get_default_parameters()
        self.assertEqual(BusinessParameters.from_dict(params.to_dict()), params)
        self.assertEqual(params.num_periods * params.interval_hours, 12.)
        self.assertEqual(params.horizon * params.interval_hours, 4.)
        self.assertNotIn("terminal_value_weight", params.to_dict())
        scenario = generate_synthetic_scenario(params)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario.json"
            save_scenario(scenario, path)
            self.assertEqual(load_scenario(path).to_dict(), scenario.to_dict())
        bad = copy.deepcopy(params.to_dict())
        bad["station"]["station_power_limits_kw"][0] = -1
        with self.assertRaises(ValueError):
            BusinessParameters.from_dict(bad)


if __name__ == "__main__":
    unittest.main()
