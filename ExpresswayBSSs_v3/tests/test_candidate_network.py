import copy
import tempfile
import unittest
from pathlib import Path

from src.candidate_network import (
    enumerate_paths, generate_candidate_network, get_feasible_arcs,
    load_candidate_network, save_candidate_network,
)
from src.parameters import (
    BusinessParameters, ODPairParameters, StationParameters, get_default_parameters,
)


def small_parameters(positions, exit_km, spacing=50., min_exit_soc=.1):
    count = len(positions)
    return BusinessParameters(
        num_periods=6, horizon=3, range_km=100.,
        min_swap_spacing_km=spacing, min_exit_soc=min_exit_soc,
        station=StationParameters(
            num_stations=count, station_ids=list(range(count)),
            positions_km=positions, num_slots=1,
            initial_slot_soc=[[1.] for _ in positions],
            slot_power_limits_kw=[[60.] for _ in positions],
            station_power_limits_kw=[60.] * count),
        od_pairs=[ODPairParameters(0, 0., exit_km, list(range(count)))])


class CandidateNetworkTests(unittest.TestCase):
    def test_actual_soc_changes_which_short_first_leg_is_needed(self):
        network = generate_candidate_network(get_default_parameters())
        # The .5 user needs the 80 km station. The .7 user can instead start
        # at 180 km, so its short entry->80 km arc can be removed.
        low_arcs = get_feasible_arcs(network, 0, .5)
        high_arcs = get_feasible_arcs(network, 0, .7)
        self.assertIn(("entry", 0), low_arcs)
        self.assertIn((("entry", 0), (0, 2), (2, "exit")),
                      enumerate_paths(low_arcs))
        self.assertNotIn(("entry", 0), high_arcs)
        self.assertIn((("entry", 1), (1, "exit")),
                      enumerate_paths(high_arcs))

    def test_reachability_changes_at_physical_soc_threshold(self):
        network = generate_candidate_network(get_default_parameters())
        # Reaching 180 km requires .6 SOC; former bin boundaries play no role.
        self.assertIn(("entry", 0), get_feasible_arcs(network, 0, .599))
        self.assertNotIn(("entry", 1), get_feasible_arcs(network, 0, .599))
        self.assertNotIn(("entry", 0), get_feasible_arcs(network, 0, .6))
        self.assertIn(("entry", 1), get_feasible_arcs(network, 0, .6))

    def test_soc_below_former_bins_is_accepted_when_physically_reachable(self):
        network = generate_candidate_network(small_parameters([20.], 90.))
        self.assertEqual(enumerate_paths(get_feasible_arcs(network, 0, .2)),
                         [(("entry", 0), (0, "exit"))])
        self.assertEqual(get_feasible_arcs(network, 0, .199), [])
        self.assertEqual(get_feasible_arcs(network, 0, 0.), [])
        self.assertTrue(get_feasible_arcs(network, 0, 1.))

    def test_short_first_leg_needs_a_complete_alternative_before_removal(self):
        network = generate_candidate_network(small_parameters([40., 70.], 150.))
        # Reaching 40 km alone does not suffice: the later short 40->70 km
        # leg is also essential to finish the trip from entry SOC .5.
        self.assertEqual(enumerate_paths(get_feasible_arcs(network, 0, .5)),
                         [(("entry", 0), (0, 1), (1, "exit"))])

    def test_short_arc_is_removed_when_actual_soc_has_an_alternative(self):
        network = generate_candidate_network(small_parameters([20., 60.], 140.))
        self.assertEqual(enumerate_paths(get_feasible_arcs(network, 0, .6)),
                         [(("entry", 1), (1, "exit"))])

    def test_short_interstation_arc_is_retained_only_when_needed(self):
        network = generate_candidate_network(small_parameters([40., 70., 145.], 225.))
        # A .4 user must use 40->70 because 40->145 exceeds full range.
        self.assertEqual(enumerate_paths(get_feasible_arcs(network, 0, .4)),
                         [(("entry", 0), (0, 1), (1, 2), (2, "exit"))])
        high_arcs = get_feasible_arcs(network, 0, .8)
        self.assertNotIn((0, 1), high_arcs)
        self.assertIn((("entry", 1), (1, 2), (2, "exit")),
                      enumerate_paths(high_arcs))

    def test_direct_exit_requires_exit_soc_reserve(self):
        network = generate_candidate_network(small_parameters([20.], 70.))
        # .7 reaches the exit empty; the .1 reserve still requires a swap.
        self.assertEqual(enumerate_paths(get_feasible_arcs(network, 0, .7)),
                         [(("entry", 0), (0, "exit"))])
        self.assertNotIn(("entry", "exit"), get_feasible_arcs(network, 0, .799))
        self.assertEqual(get_feasible_arcs(network, 0, .8), [("entry", "exit")])

    def test_station_to_exit_also_requires_reserve(self):
        network = generate_candidate_network(small_parameters([20.], 115.))
        # A full battery at 20 km reaches 115 km with only .05 SOC.
        self.assertEqual(get_feasible_arcs(network, 0, 1.), [])

    def test_direct_exit_preserves_other_complete_routes(self):
        params = get_default_parameters()
        params.od_pairs = [ODPairParameters(0, 0., 230., [0, 1])]
        network = generate_candidate_network(params)
        arcs = get_feasible_arcs(network, 0, .95)
        self.assertIn(("entry", "exit"), arcs)
        self.assertIn(("entry", 1), arcs)
        self.assertIn((1, "exit"), arcs)
        self.assertGreater(len(enumerate_paths(arcs)), 1)
        self.assertNotIn(("entry", "exit"), get_feasible_arcs(network, 0, .80))

    def test_user_networks_do_not_mutate_shared_network(self):
        network = generate_candidate_network(get_default_parameters())
        saved = copy.deepcopy(network)
        expected_low = get_feasible_arcs(network, 0, .5)
        get_feasible_arcs(network, 0, .7)
        get_feasible_arcs(network, 1, 1.)
        self.assertEqual(get_feasible_arcs(network, 0, .5), expected_low)
        self.assertEqual(network, saved)
        self.assertNotIn("soc_bins", network)
        for od in network["od_networks"]:
            self.assertNotIn("soc_bins", od)
            self.assertTrue(all(isinstance(arc[0], int) for arc in od["station_arcs"]))

    def test_candidate_serialization_and_virtual_origin(self):
        network = generate_candidate_network(get_default_parameters())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.json"
            save_candidate_network(network, path)
            loaded = load_candidate_network(path)
            self.assertEqual(loaded, network)
            for soc in (.2, .5, .7, 1.):
                self.assertEqual(get_feasible_arcs(loaded, 0, soc),
                                 get_feasible_arcs(network, 0, soc))
        self.assertEqual(enumerate_paths([("virtual", 1), (1, "exit")], origin="virtual"),
                         [(("virtual", 1), (1, "exit"))])


if __name__ == "__main__":
    unittest.main()
