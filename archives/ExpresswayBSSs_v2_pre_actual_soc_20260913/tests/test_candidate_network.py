import tempfile
import unittest
from pathlib import Path

from src.candidate_network import (
    enumerate_paths, generate_candidate_network, get_candidate_arcs,
    get_feasible_arcs, load_candidate_network, save_candidate_network,
)
from src.parameters import (
    BusinessParameters, ODPairParameters, StationParameters, get_default_parameters,
)


def small_parameters(positions, exit_km, soc_bin, spacing=50., min_exit_soc=.1):
    count = len(positions)
    return BusinessParameters(
        num_periods=6, horizon=3, range_km=100., soc_bins=[list(soc_bin)],
        min_swap_spacing_km=spacing, min_exit_soc=min_exit_soc,
        station=StationParameters(
            num_stations=count, station_ids=list(range(count)),
            positions_km=positions, num_slots=1,
            initial_slot_soc=[[1.] for _ in positions],
            slot_power_limits_kw=[[60.] for _ in positions],
            station_power_limits_kw=[60.] * count),
        od_pairs=[ODPairParameters(0, 0., exit_km, list(range(count)))])


class CandidateNetworkTests(unittest.TestCase):
    def test_pruning_preserves_soc_lower_bound_path(self):
        params = get_default_parameters()
        network = generate_candidate_network(params)
        # SOC .5 reaches 80 km but not 180 km. Preserve its only first stop.
        self.assertIn(("entry", 0), get_candidate_arcs(network, 0, .5))
        paths = enumerate_paths(get_feasible_arcs(network, 0, .5))
        self.assertIn((("entry", 0), (0, 2), (2, "exit")), paths)

    def test_upper_soc_keeps_additional_first_leg(self):
        params = get_default_parameters()
        network = generate_candidate_network(params)
        # Protecting the bin's lower SOC must not replace the upper-SOC graph.
        self.assertIn(("entry", 1), get_candidate_arcs(network, 0, .7))
        paths = enumerate_paths(get_feasible_arcs(network, 0, .7))
        self.assertIn((("entry", 1), (1, "exit")), paths)

    def test_no_pruning_when_lower_bound_has_no_raw_path(self):
        params = small_parameters([40., 70.], 150., (.2, .8))
        network = generate_candidate_network(params)
        record = network["od_networks"][0]["soc_bins"][0]
        self.assertEqual(record["removed_arcs"], [])
        self.assertEqual(get_feasible_arcs(network, 0, .2), [])
        raw_paths = enumerate_paths([tuple(arc) for arc in record["raw_arcs"]])
        self.assertEqual(enumerate_paths(get_candidate_arcs(network, 0, .5)), raw_paths)
        self.assertIn((("entry", 0), (0, 1), (1, "exit")),
                      enumerate_paths(get_feasible_arcs(network, 0, .5)))

    def test_short_arc_is_removed_when_lower_bound_has_an_alternative(self):
        params = small_parameters([20., 60.], 140., (.6, 1.))
        network = generate_candidate_network(params)
        self.assertNotIn(("entry", 0), get_candidate_arcs(network, 0, .6))
        self.assertIn((("entry", 1), (1, "exit")),
                      enumerate_paths(get_feasible_arcs(network, 0, .6)))

    def test_short_interstation_arc_preserves_lower_bound_complete_path(self):
        params = small_parameters([40., 70., 145.], 225., (.4, .8))
        network = generate_candidate_network(params)
        # Upper SOC can start at 70 km. Lower SOC must start at 40 km and
        # use the short 40->70 leg, since 40->145 exceeds the full range.
        self.assertIn((0, 1), get_candidate_arcs(network, 0, .4))
        self.assertIn((("entry", 0), (0, 1), (1, 2), (2, "exit")),
                      enumerate_paths(get_feasible_arcs(network, 0, .4)))
        self.assertIn(("entry", 1), get_candidate_arcs(network, 0, .8))

    def test_lower_bound_direct_exit_requires_exit_soc_reserve(self):
        params = small_parameters([20.], 70., (.7, .9))
        network = generate_candidate_network(params)
        # SOC .7 reaches the exit empty, below the .1 reserve; the 20 km
        # stop therefore cannot be pruned in favor of the upper-SOC direct arc.
        self.assertIn(("entry", 0), get_candidate_arcs(network, 0, .7))
        self.assertEqual(enumerate_paths(get_feasible_arcs(network, 0, .7)),
                         [(("entry", 0), (0, "exit"))])
        self.assertIn(("entry", "exit"), get_feasible_arcs(network, 0, .9))

    def test_direct_exit_preserves_other_complete_routes(self):
        params = get_default_parameters()
        params.od_pairs = [type(params.od_pairs[0])(0, 0., 230., [0, 1])]
        network = generate_candidate_network(params)
        arcs = get_feasible_arcs(network, 0, .95)
        self.assertIn(("entry", "exit"), arcs)
        self.assertIn(("entry", 1), arcs)
        self.assertIn((1, "exit"), arcs)
        self.assertGreater(len(enumerate_paths(arcs)), 1)
        self.assertNotIn(("entry", "exit"), get_feasible_arcs(network, 0, .80))

    def test_retain_only_route_when_first_leg_is_short(self):
        params = get_default_parameters()
        params.od_pairs = [type(params.od_pairs[0])(0, 0., 230., [0])]
        network = generate_candidate_network(params)
        self.assertEqual(enumerate_paths(get_feasible_arcs(network, 0, .3)), [(('entry', 0), (0, 'exit'))])

    def test_candidate_serialization_and_virtual_origin(self):
        params = get_default_parameters()
        network = generate_candidate_network(params)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "network.json"
            save_candidate_network(network, path)
            self.assertEqual(load_candidate_network(path), network)
        self.assertEqual(enumerate_paths([("virtual", 1), (1, "exit")], origin="virtual"), [(('virtual', 1), (1, 'exit'))])


if __name__ == "__main__":
    unittest.main()
