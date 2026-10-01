"""Orchestration tests use temporary fake worker outputs, never solver results."""
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiment_control import ExperimentPaused, atomic_json
from src.experiment_suite import _Suite, _freeze_plan, run_formal_and_paper
from src.parameters import BusinessParameters
from src.terminal_features import FeatureSpec
from src.terminal_value import TerminalValueModel, terminal_configuration_fingerprint


class ExperimentSuiteTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.params = BusinessParameters(num_periods=2, horizon=1, num_reservations=0,
                                         random_arrival_rate_per_hour=[0.] * 6)
        self.config = self.root / "config.json"
        self.params.save_json(self.config)
        self.pilots = [{"completed_periods": 2, "worker_wall_seconds": 60.}] * 3

    def test_freeze_budget_and_separate_splits(self):
        plan = _freeze_plan(self.params, self.pilots, None)
        self.assertEqual(plan["budget"], {"train_days_per_iteration":3,"validation_days":3,
            "test_days":10,"outer_iterations":2,"training_replicates":2,"fit_epochs":200})
        splits = plan["seeds"]
        combined = splits["pilot"] + splits["validation"] + splits["test"]
        combined += [seed for rep in splits["training"] for iteration in rep for seed in iteration]
        self.assertEqual(len(combined), len(set(combined)))
        fast = _freeze_plan(self.params, [{"completed_periods":2,"worker_wall_seconds":20.}] * 3, None)
        self.assertEqual(fast["budget"]["test_days"], 20)
        self.assertEqual(fast["budget"]["outer_iterations"], 3)
        with self.assertRaises(ValueError):
            _freeze_plan(self.params, self.pilots[:2], None)

    def test_freeze_missing_worker_timing_uses_explicit_proxy_without_mutating_metrics(self):
        pilots = [{"completed_periods":2,"worker_wall_seconds":None,
                   "runtime_timing_incomplete":True,"known_runtime_seconds":4.,
                   "model_wall_seconds_total":60.},
                  {"completed_periods":2,"model_wall_seconds_total":90.},
                  {"completed_periods":2,"worker_wall_seconds":30.,"model_wall_seconds_total":5.}]
        plan = _freeze_plan(self.params,pilots,None)
        self.assertEqual(plan["pilot_mean_day_seconds"],60.)
        self.assertEqual([row["timing_seconds"] for row in plan["pilot_timing"]],[60.,90.,30.])
        self.assertEqual([row["timing_basis"] for row in plan["pilot_timing"]],
                         ["model_wall_seconds_total","model_wall_seconds_total","worker_wall_seconds"])
        self.assertIn("not complete process wall time",plan["pilot_timing"][0]["timing_limitation"])
        self.assertEqual(plan["budget"],_freeze_plan(self.params,self.pilots,None)["budget"])
        self.assertIsNone(pilots[0]["worker_wall_seconds"])
        self.assertEqual(plan["pilot_metrics"],pilots)

    def test_freeze_missing_or_invalid_proxy_is_never_zero_filled(self):
        for value in (None,0.,-1.,float("nan"),float("inf"),"60",True):
            with self.subTest(value=value):
                pilots=[{"completed_periods":2,"worker_wall_seconds":None,
                         "model_wall_seconds_total":value}]+self.pilots[1:]
                with self.assertRaisesRegex(ValueError,"finite positive"):
                    _freeze_plan(self.params,pilots,None)

    def test_plan_only_and_immutable_resume(self):
        def forbidden(job):
            self.fail("plan-only must not call the worker")
        first = run_formal_and_paper(self.config, self.root, None, forbidden, self.pilots, stage="plan")
        second = run_formal_and_paper(self.config, self.root, "2099-01-01T00:00:00+00:00",
                                      forbidden, [], stage="plan")
        self.assertEqual(first["plan"], second["plan"])
        path = self.root / "formal_suite/plan.json"
        altered = json.loads(path.read_text())
        altered["budget"]["test_days"] = 3
        atomic_json(path, altered)
        with self.assertRaisesRegex(ValueError, "modified frozen"):
            run_formal_and_paper(self.config, self.root, None, forbidden, [], stage="plan")

    def test_incomplete_worker_stops_and_retry_uses_same_seed(self):
        calls = []
        def stopped(job):
            calls.append(job)
            return {"worker_status":{"state":"failed","error":"deliberate test failure"}}
        result = run_formal_and_paper(self.config, self.root, None, stopped, self.pilots, stage="core")
        self.assertEqual(result["status"], "paused")
        self.assertEqual(len(calls), 1)
        result = run_formal_and_paper(self.config, self.root, None, stopped, [], stage="core")
        self.assertEqual(result["status"], "paused")
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(calls[0]["job_id"], "base/test/zero/seed_41000")
        self.assertEqual(result["completed_jobs"], 0)

    def test_completed_worker_is_reused_and_expired_deadline_does_not_launch(self):
        calls = []
        def finished(job):
            calls.append(job)
            return {"worker_status":"complete", "metrics":{"completed_periods":2,"net_profit_yuan":5.}}
        suite = _Suite(self.config,self.root,None,finished,self.pilots)
        one = suite._job("base",self.params,41000,"test/zero")
        again = suite._job("base",self.params,41000,"test/zero")
        self.assertEqual(one,again)
        self.assertEqual(len(calls),1)
        suite.deadline = "2000-01-01T00:00:00+00:00"
        with self.assertRaises(ExperimentPaused):
            suite._job("base",self.params,41001,"test/zero")
        self.assertEqual(len(calls),1)

    def test_validation_selects_model_and_current_policy_batches_stay_separate(self):
        calls, fits = [], []
        def tiny_plan(params,pilots,deadline):
            plan = _freeze_plan(params,pilots,deadline)
            plan["budget"].update(train_days_per_iteration=1,validation_days=1,test_days=1,
                                   training_replicates=1,outer_iterations=2,fit_epochs=1)
            plan["seeds"]["training"] = [[[11000],[11100]]]
            plan["seeds"]["validation"] = [21000]
            plan["seeds"]["test"] = [41000]
            return plan
        def worker(job):
            calls.append(job)
            directory = Path(job["output_dir"])
            directory.mkdir(parents=True,exist_ok=True)
            if job["capture_features"]:
                source_spec = FeatureSpec(self.params,variant=job["feature_variant"])
                dimension = source_spec.dimension
                atomic_json(directory/"feature_manifest.json",{"variant":source_spec.variant,"dimension":dimension,"names":source_spec.names})
                result = {"parameter_snapshot":self.params.to_dict(),"completed":True,
                          "method":{"terminal_kind":TerminalValueModel.load(job["model"]).kind if job["model"] else "zero",
                                    "feature_variant":source_spec.variant,"route_mode":"joint","charging_mode":"joint"},
                          "rounds":[{"state_features":[0.] * dimension,"reward":1.} for _ in range(2)],
                          "summary":{"total_reward":2.}}
                with gzip.open(directory/"result.json.gz","wt",encoding="utf-8") as stream:
                    json.dump(result,stream)
            score = 10. if "iteration_0" in job["job_id"] else 5.
            return {"worker_status":"complete","metrics":{"completed_periods":2,"net_profit_yuan":score}}
        def fake_fit(x,y,spec,**kwargs):
            fits.append((x.copy(),y.copy(),kwargs))
            model = TerminalValueModel(kind="relu",variant=spec.variant,feature_names=list(spec.names),
                configuration_fingerprint=terminal_configuration_fingerprint(spec.params),
                linear_weights=[0.] * spec.dimension,bias=float(len(fits)),
                hidden_weights=[[0.] * spec.dimension],hidden_bias=[0.],output_weights=[0.])
            return model,{"samples":len(y),"kind":"relu"}
        with patch("src.experiment_suite._freeze_plan",tiny_plan), patch("src.experiment_suite.fit_value_model",fake_fit):
            suite = _Suite(self.config,self.root,None,worker,self.pilots)
            zero = suite._test("base",self.params,"zero")
            selected,jobs = suite._learn("base",self.params,"relu","inventory_only",0,zero)
            self.assertEqual(selected.name,"iteration_0.json")
            self.assertEqual(len(fits),2)
            self.assertIsNone(fits[0][2]["previous"])
            self.assertIsNotNone(fits[1][2]["previous"])
            self.assertEqual(fits[0][1].tolist(),[2.,1.])
            self.assertEqual(fits[1][1].tolist(),[2.,1.])
            training = [job for job in calls if job["capture_features"]]
            self.assertEqual(len(training),2)
            self.assertIsNone(training[0]["model"])
            self.assertTrue(training[1]["model"].endswith("iteration_0.json"))
            self.assertNotEqual(training[0]["scenario"],training[1]["scenario"])
            before = len(calls)
            selected_again,_ = suite._learn("base",self.params,"relu","inventory_only",0,zero)
            self.assertEqual(selected_again,selected)
            self.assertEqual(len(calls),before)
            self.assertEqual(len(fits),2)
            group = suite.state["groups"]["base/relu_inventory_only_rep0"]
            self.assertEqual(group["statistics"]["net_profit_yuan"]["n"],1)
            self.assertEqual(group["paired_difference_from_zero"]["net_profit_yuan"]["n"],1)
            # Equal dimension is insufficient: swapped feature columns must fail.
            first_manifest = Path(training[0]["output_dir"])/"feature_manifest.json"
            altered = json.loads(first_manifest.read_text())
            altered["names"][0],altered["names"][1] = altered["names"][1],altered["names"][0]
            atomic_json(first_manifest,altered)
            with self.assertRaisesRegex(ValueError,"training feature manifest differs"):
                suite._training_batch("base",self.params,"relu_inventory_only_rep0",0,0,None,FeatureSpec(self.params,"inventory_only"))

    def test_completed_job_rejects_modified_scenario_with_same_seed_and_parameters(self):
        calls = []
        def finished(job):
            calls.append(job)
            return {"worker_status":"complete", "metrics":{"completed_periods":2,"net_profit_yuan":5.}}
        suite = _Suite(self.config,self.root,None,finished,self.pilots)
        suite._job("base",self.params,41000,"test/zero")
        scenario_path = Path(calls[0]["scenario"])
        data = json.loads(scenario_path.read_text())
        data["actual_random_requests"].append({"request_id":"tampered","station":0,"arrival_time":.1,"return_soc":.2})
        atomic_json(scenario_path,data)
        with self.assertRaisesRegex(ValueError,"completed job inputs have changed"):
            suite._job("base",self.params,41000,"test/zero")
        self.assertEqual(len(calls),1)

    def test_disk_model_rejects_wrong_kind_variant_or_configuration(self):
        def forbidden(job):
            self.fail("a mismatched saved model must fail before any worker starts")
        suite = _Suite(self.config,self.root,None,forbidden,self.pilots)
        spec = FeatureSpec(self.params,variant="full")
        directory = suite.root/"models/base/relu_full_rep0"
        atomic_json(directory/"iteration_0_fit.json",{"sampled_policy_fingerprint":None})
        valid = TerminalValueModel(kind="relu",variant="full",feature_names=list(spec.names),
            linear_weights=[0.]*spec.dimension,hidden_weights=[[0.]*spec.dimension],hidden_bias=[0.],output_weights=[0.],
            configuration_fingerprint=terminal_configuration_fingerprint(self.params)).to_dict()
        for field,value in (("kind","linear"),("variant","inventory_only"),("configuration_fingerprint","wrong")):
            with self.subTest(field=field):
                altered = dict(valid)
                altered[field] = value
                atomic_json(directory/"iteration_0.json",altered)
                with self.assertRaisesRegex(ValueError,"saved fit does not match"):
                    suite._learn("base",self.params,"relu","full",0,[])

    def test_feature_manifest_is_checked_before_it_can_be_overwritten(self):
        suite = _Suite(self.config,self.root,None,lambda job:None,self.pilots)
        spec = FeatureSpec(self.params,variant="full").to_dict()
        spec["count_scale"] = 99.
        path = suite.root/"models/base/relu_full_rep0/feature_spec.json"
        atomic_json(path,spec)
        with self.assertRaisesRegex(ValueError,"saved fixed feature specification differs"):
            suite._learn("base",self.params,"relu","full",0,[])
        self.assertEqual(json.loads(path.read_text())["count_scale"],99.)

    def test_sensitivity_changes_only_selected_resources_or_demand(self):
        params = BusinessParameters.load_json(Path(__file__).resolve().parents[1]/"configs/terminal_experiment.json")
        params.save_json(self.config)
        pilots = [{"completed_periods":288,"worker_wall_seconds":60.}] * 3
        suite = _Suite(self.config,self.root,None,lambda job:None,pilots)
        demand = suite._changed_configuration("demand",1.5)
        self.assertEqual(demand.num_reservations,150)
        self.assertAlmostEqual(sum(map(sum,demand.random_hourly_means)),150.)
        self.assertEqual(demand.od_sampling_weights,params.od_sampling_weights)
        self.assertEqual(demand.reservation_entry_soc_range,params.reservation_entry_soc_range)
        battery = suite._changed_configuration("battery",14)
        self.assertTrue(all(len(row)==14 for row in battery.station.initial_slot_soc))
        self.assertEqual(battery.station.station_power_limits_kw,params.station.station_power_limits_kw)
        power = suite._changed_configuration("power",.5)
        self.assertEqual(power.station.station_power_limits_kw,[480.] * 6)
        self.assertEqual(power.station.slot_power_limits_kw,params.station.slot_power_limits_kw)
        self.assertEqual(params.station.num_slots,21)


if __name__ == "__main__":
    unittest.main()
