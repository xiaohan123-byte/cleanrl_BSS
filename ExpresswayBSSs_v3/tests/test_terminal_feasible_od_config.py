"""Fixed OD-set correction preserves independent per-vehicle generation."""
import copy
import csv
import unittest
from pathlib import Path

from src.candidate_network import generate_candidate_network, get_feasible_arcs
from src.parameters import BusinessParameters
from src.scenario import generate_synthetic_scenario

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "ExpresswayBSSs/data_generation_optim/output"


class FeasibleODConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.old = BusinessParameters.load_json(ROOT / "configs/terminal_experiment.json")
        self.new = BusinessParameters.load_json(ROOT / "configs/terminal_experiment_feasible_od.json")

    def test_exact_fixed_subset_and_every_half_soc_path(self):
        self.assertEqual(len(self.old.od_pairs),30)
        self.assertEqual([od.od_id for od in self.new.od_pairs],
                         [od.od_id for od in self.old.od_pairs if od.od_id not in {6,7,8,9}])
        self.assertEqual(len(self.new.od_pairs),26)
        self.assertEqual(sum(self.new.od_direction(i)<0 for i in range(26)),15)
        network = generate_candidate_network(self.new)
        for index,od in enumerate(self.new.od_pairs):
            with self.subTest(od_id=od.od_id):
                self.assertTrue(get_feasible_arcs(network,index,.5))
                self.assertTrue(get_feasible_arcs(network,index,.500001))

    def test_population_weights_and_station_means_recomputed_together(self):
        with (DATA/"population_8e.csv").open(encoding="utf-8-sig",newline="") as stream:
            populations = {key:float(value) for key,value in next(csv.DictReader(stream)).items()}
        with (DATA/"subpaths_8e.csv").open(encoding="utf-8-sig",newline="") as stream:
            records = {int(row["PATH"][1:]):row for row in csv.DictReader(stream)}
        raw = [populations[records[od.od_id]["SOURCE"]] * populations[records[od.od_id]["ROOT"]]
               / abs(od.exit_km-od.entry_km)**2 for od in self.new.od_pairs]
        expected = [value/sum(raw) for value in raw]
        for actual,wanted in zip(self.new.od_sampling_weights,expected):
            self.assertAlmostEqual(actual,wanted,places=14)
        exposure = [sum(weight for od,weight in zip(self.new.od_pairs,expected) if station in od.station_indices)
                    for station in self.new.station.station_ids]
        shares = [value/sum(exposure) for value in exposure]
        hourly = self.old.source_metadata["random_hourly_weights"]
        for station,share in enumerate(shares):
            self.assertAlmostEqual(self.new.source_metadata["random_station_shares"][station],share,places=14)
            for hour,weight in enumerate(hourly):
                self.assertAlmostEqual(self.new.random_hourly_means[station][hour],share*100*weight/sum(hourly),places=13)
        self.assertAlmostEqual(sum(map(sum,self.new.random_hourly_means)),100.)
        self.assertNotEqual(self.old.source_metadata["random_station_shares"],shares)

    def test_only_intended_configuration_fields_change(self):
        old,new = self.old.to_dict(),self.new.to_dict()
        changed = {key for key in old if old[key]!=new[key]}
        self.assertEqual(changed,{"od_pairs","od_sampling_weights","random_hourly_means",
                                  "random_arrival_rate_per_hour","source_metadata"})
        self.assertEqual(new["reservation_entry_soc_range"],[.5,1.])
        self.assertEqual(new["report_entry_soc_range"],[.5,1.])
        self.assertEqual(new["source_metadata"]["generation_rule_version"],2)

    def test_report_noise_still_has_an_independent_stream(self):
        changed = copy.deepcopy(self.new)
        changed.entry_soc_error = .01
        changed.entry_time_error_hours = .05
        one = generate_synthetic_scenario(self.new,104)
        two = generate_synthetic_scenario(changed,104)
        project = lambda scenario: {tuple(row["user_key"]):(row["actual_entry_time"],row["actual_entry_soc"],row["segment_time_multipliers"])
                                    for row in scenario.reservations}
        self.assertEqual(project(one),project(two))
        self.assertEqual(one.actual_random_requests,two.actual_random_requests)


if __name__ == "__main__":
    unittest.main()
