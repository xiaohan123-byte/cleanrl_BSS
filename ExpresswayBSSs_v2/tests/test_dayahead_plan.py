import unittest

from src.candidate_network import generate_candidate_network
from src.dayahead_plan import generate_dayahead_plan
from src.parameters import get_default_parameters


class DayaheadPlanTests(unittest.TestCase):
    def test_minimum_swaps_and_downstream_tie_break(self):
        params = get_default_parameters()
        reservation = {"user_key": [0, 3], "od_id": 0, "entry_time": 0., "entry_soc": 1.}
        plan = generate_dayahead_plan(params, generate_candidate_network(params), [reservation])
        # 180 km and 280 km each permit one swap; choose the latter.
        self.assertEqual(plan, {"0:3": [2]})

    def test_no_inventory_or_price_dependency(self):
        params = get_default_parameters()
        reservation = {"user_key": [1, 8], "od_id": 1, "entry_time": 0., "entry_soc": 1.}
        network = generate_candidate_network(params)
        expected = generate_dayahead_plan(params, network, [reservation])
        params.station.initial_slot_soc = [[0.] * params.station.num_slots for _ in params.station.station_ids]
        params.electricity_price = [[999.] * params.num_periods for _ in params.station.station_ids]
        self.assertEqual(generate_dayahead_plan(params, network, [reservation]), expected)

    def test_actual_soc_requires_two_swaps(self):
        params = get_default_parameters()
        reservation = {"user_key": [0, 0], "od_id": 0, "entry_time": 0., "entry_soc": .5}
        self.assertEqual(generate_dayahead_plan(params, generate_candidate_network(params), [reservation]),
                         {"0:0": [0, 3]})

    def test_actual_soc_allows_one_swap(self):
        params = get_default_parameters()
        reservation = {"user_key": [0, 0], "od_id": 0, "entry_time": 0., "entry_soc": .7}
        self.assertEqual(generate_dayahead_plan(params, generate_candidate_network(params), [reservation]),
                         {"0:0": [1]})

    def test_reject_physically_unreachable_actual_soc(self):
        params = get_default_parameters()
        # SOC .2 gives 60 km of range, less than the first station at 80 km.
        reservation = {"user_key": [0, 0], "od_id": 0, "entry_time": 0., "entry_soc": .2}
        with self.assertRaisesRegex(ValueError, "no complete reachable"):
            generate_dayahead_plan(params, generate_candidate_network(params), [reservation])

    def test_direct_exit_needs_no_swap(self):
        params = get_default_parameters()
        params.od_pairs = [type(params.od_pairs[0])(7, 0., 230., [0, 1])]
        reservation = {"user_key": [7, 0], "od_id": 7, "entry_time": 0., "entry_soc": 1.}
        self.assertEqual(generate_dayahead_plan(params, generate_candidate_network(params), [reservation]), {"7:0": []})


if __name__ == "__main__":
    unittest.main()
