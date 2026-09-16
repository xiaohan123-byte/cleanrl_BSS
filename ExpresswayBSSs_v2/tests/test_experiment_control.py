import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.domain import MPCSolution
from src.experiment_control import ExperimentPaused, RunJournal, check_deadline
from src.parameters import BusinessParameters, ODPairParameters, StationParameters
from src.rolling_runner import run_rolling_mpc
from src.scenario import SyntheticScenario
from src.value_training import fit_value_model, monte_carlo_samples


class ExperimentControlTests(unittest.TestCase):
    def test_deadline_requires_timezone_and_expired_deadline_pauses(self):
        with self.assertRaises(ValueError):
            check_deadline("2026-09-14T07:00:00")
        with self.assertRaises(ExperimentPaused):
            check_deadline("2000-01-01T00:00:00+00:00")

    def test_journal_recovers_only_complete_appends_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            journal=RunJournal(directory,{"seed":101})
            journal.initialize({"period":0},{})
            journal.append({"period":0,"reward":7})
            with journal.round_path.open("ab") as stream:
                stream.write(b'{"period":1')
            _,_,rows=journal.recover()
            self.assertEqual(rows,[{"period":0,"reward":7}])
            self.assertTrue(journal.round_path.read_bytes().endswith(b"\n"))
            changed=RunJournal(directory,{"seed":102})
            with self.assertRaises(ValueError):
                changed.recover()

    def test_resume_does_not_double_count_actual_interval(self):
        p=BusinessParameters(num_periods=3,horizon=1,num_reservations=0,
            station=StationParameters(num_stations=1,station_ids=[0],positions_km=[5.],
                num_slots=1,initial_slot_soc=[[1.]],charging_efficiency=1.,
                slot_power_limits_kw=[[60.]],station_power_limits_kw=[60.]),
            od_pairs=[ODPairParameters(0,0.,10.,[0])],random_arrival_rate_per_hour=[0.])
        scene=SyntheticScenario(p,[],[])
        seen=[]
        def solve(params,window,**kwargs):
            seen.append(window.ell)
            return MPCSolution("optimal",0.,{}, {}, [], [[[0.]*window.horizon]],
                               [[[1.]*(window.horizon+1)]])
        def interrupt(record):
            raise ExperimentPaused("simulated interruption after durable first interval")
        with tempfile.TemporaryDirectory() as directory, patch("src.rolling_runner.solve_mpc",side_effect=solve):
            with self.assertRaises(ExperimentPaused):
                run_rolling_mpc(p,scene,journal_dir=directory,progress=interrupt)
            result=run_rolling_mpc(p,scene,journal_dir=directory)
            self.assertEqual(seen,[0,1,2])
            self.assertEqual(len(result["rounds"]),3)
            self.assertEqual(len(result["ledger"]),3)
            self.assertEqual(len({e["event_id"] for e in result["ledger"]}),3)
            self.assertEqual(result["summary"]["total_reward"],0.)
            self.assertEqual(json.loads((Path(directory)/"status.json").read_text())["state"],"complete")

    def test_only_full_realised_return_labels_are_accepted(self):
        data={"parameter_snapshot":{"num_periods":3},"summary":{"total_reward":6.},
              "rounds":[{"state_features":[i],"reward":r} for i,r in enumerate([1.,-2.,7.])]}
        x,y=monte_carlo_samples(data)
        np.testing.assert_allclose(y,[6.,5.,7.])
        data["rounds"].pop()
        with self.assertRaises(ValueError):
            monte_carlo_samples(data)

    def test_ridge_currency_scale_and_unpenalized_intercept(self):
        class Spec:
            names=["x"]; dimension=1; variant="full"
        x=np.linspace(-1,1,100)[:,None]
        y=7000.+2000.*x[:,0]
        model,diagnostics=fit_value_model(x,y,Spec(),kind="linear")
        np.testing.assert_allclose(model.predict(x),y,atol=.1)
        self.assertAlmostEqual(model.bias,7.,places=8)
        self.assertEqual(diagnostics["target_scale_yuan"],1000.)
        self.assertLess(diagnostics["training_rmse_yuan"],.1)


if __name__=="__main__":
    unittest.main()
