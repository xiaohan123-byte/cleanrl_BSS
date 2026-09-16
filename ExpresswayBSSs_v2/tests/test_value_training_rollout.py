"""Tiny end-to-end training checks; these are not paper experiment results."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from src.parameters import BusinessParameters, ODPairParameters, StationParameters
from src.scenario import SyntheticScenario
from src.terminal_features import FeatureSpec
from src.terminal_value import TerminalValueModel
from src.rolling_runner import run_rolling_mpc
from src.result_statistics import build_result_statistics
from src.value_training import fit_value_model, monte_carlo_samples


class ValueTrainingRolloutTests(unittest.TestCase):
    def test_complete_zero_rollout_fit_save_reload_and_value_rollout(self):
        p=BusinessParameters(num_periods=4,horizon=2,num_reservations=0,
            random_arrival_rate_per_hour=[0.],
            station=StationParameters(num_stations=1,station_ids=[0],positions_km=[10.],
                num_slots=1,initial_slot_soc=[[1.]],charging_efficiency=.95,
                slot_power_limits_kw=[[60.]],station_power_limits_kw=[60.]),
            od_pairs=[ODPairParameters(0,0.,20.,[0])])
        p.solver.time_limit_sec=5.
        scene=SyntheticScenario(p,[],[
            {"request_id":"one","station":0,"arrival_time":0.,"return_soc":.2}])
        spec=FeatureSpec(p,variant="inventory_only")
        first=run_rolling_mpc(p,scene,feature_spec=spec)
        build_result_statistics(first)
        x,y=monte_carlo_samples(first)
        self.assertEqual(x.shape,(4,spec.dimension))
        for kind in ["linear","relu"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                model,fit=fit_value_model(x,y,spec,kind=kind,seed=9,epochs=2)
                path=Path(directory)/"model.json"
                model.save(path)
                loaded=TerminalValueModel.load(path)
                np.testing.assert_allclose(model.predict(x),loaded.predict(x))
                result=run_rolling_mpc(p,scene,terminal_model=loaded,feature_spec=spec)
                build_result_statistics(result)
                self.assertEqual(len(result["rounds"]),4)
                self.assertEqual(result["rounds"][-1]["solution"]["terminal_value"],0.)
                if kind=="relu":
                    warmed,info=fit_value_model(x,y,spec,kind=kind,previous=loaded,seed=10,epochs=1)
                    self.assertTrue(info["warm_started"])
                    self.assertTrue(info["optimizer_state_reset"])
                    self.assertTrue(np.isfinite(warmed.predict(x)).all())


if __name__=="__main__":
    unittest.main()
