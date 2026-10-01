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

    def test_reservation_soc_sampling_range_is_configurable(self):
        params = get_default_parameters()
        params.reservation_entry_soc_range = [.6, .61]
        scenario = generate_synthetic_scenario(params)
        self.assertTrue(all(.6 <= record["entry_soc"] < .61 for record in scenario.reservations))
        plans = generate_dayahead_plan(params, generate_candidate_network(params), scenario.initial_reservations())
        self.assertEqual(len(plans), params.num_reservations)

    def test_reject_invalid_reservation_soc_sampling_ranges(self):
        for bounds in ([], [.3], [.3, .6, 1.], [-.1, 1.], [.3, 1.1], [.5, .5], [.7, .3],
                       [float("nan"), 1.], [.3, float("inf")], [float("-inf"), 1.], None):
            with self.subTest(bounds=bounds):
                data = get_default_parameters().to_dict()
                data["reservation_entry_soc_range"] = bounds
                with self.assertRaisesRegex(ValueError, "reservation_entry_soc_range"):
                    BusinessParameters.from_dict(data)

    def test_external_reservation_soc_is_independent_of_sampling_range(self):
        params = get_default_parameters()
        params.reservation_entry_soc_range = [.3, .4]
        # SOC .28 reaches the first station at 80 km, below the sampling range.
        records = [{"user_key": [0, index], "od_id": 0, "entry_time": 0., "entry_soc": soc}
                   for index, soc in enumerate((.28, .9))]
        scenario = SyntheticScenario(params, records, [])
        plans = generate_dayahead_plan(params, generate_candidate_network(params), scenario.initial_reservations())
        self.assertEqual(plans["0:0"], [0, 3])
        self.assertEqual(len(plans), 2)

    def test_external_reservation_soc_must_be_finite_and_physical(self):
        params = get_default_parameters()
        for soc in (-.01, 1.01, float("nan"), float("inf"), float("-inf")):
            with self.subTest(soc=soc):
                record = {"user_key": [0, 0], "od_id": 0, "entry_time": 0., "entry_soc": soc}
                with self.assertRaisesRegex(ValueError, "entry SOC"):
                    SyntheticScenario(params, [record], [])
        for soc in (0., 1.):
            record = {"user_key": [0, 0], "od_id": 0, "entry_time": 0., "entry_soc": soc}
            self.assertEqual(SyntheticScenario(params, [record], []).reservations[0]["entry_soc"], soc)

    def test_sampling_range_without_reachable_od_is_rejected(self):
        params = get_default_parameters()
        params.reservation_entry_soc_range = [.1, .2]
        with self.assertRaisesRegex(ValueError, "no reachable candidate path within reservation_entry_soc_range"):
            generate_synthetic_scenario(params)

    def test_generation_excludes_ods_unreachable_at_sampling_upper_bound(self):
        params = get_default_parameters()
        od_type = type(params.od_pairs[0])
        params.od_pairs = [od_type(7, 0., 230., [0, 1]), od_type(8, 81., 430., [1, 2, 3])]
        params.reservation_entry_soc_range = [.27, .28]
        scenario = generate_synthetic_scenario(params)
        # Both ODs are feasible at full SOC, but only OD 7 is feasible here.
        self.assertEqual({record["od_id"] for record in scenario.reservations}, {7})

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
        self.assertNotIn("soc_bins", params.to_dict())
        self.assertEqual(params.reservation_entry_soc_range, [.3, 1.])
        scenario = generate_synthetic_scenario(params)
        self.assertEqual(scenario.to_dict()["schema_version"], 2)
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
