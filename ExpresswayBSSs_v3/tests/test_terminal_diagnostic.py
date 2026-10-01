"""Small fixtures validate diagnostics without launching the six-station experiment."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnose_terminal_window import prepare_window, run_comparison, make_untrained_model, main
from src.candidate_network import generate_candidate_network
from src.domain import Reservation, RollingState
from src.experiment_control import fingerprint
from src.terminal_features import FeatureSpec
from tests.test_terminal_value import params


def fixture(root):
    p=params();p.solver.time_limit_sec=60.
    user=Reservation((0,0),0.,.8,0.,.8,entered=True,day_ahead=[0,1],retained_plan=[0,1],published_plan=[0,1])
    state=RollingState(0,[[1.],[1.]],users={"0:0":user})
    state_before=state.to_dict();state_before.pop("ledger");state_before["ledger_event_count"]=7
    pilot=root/"pilot";journal=pilot/"journal";journal.mkdir(parents=True)
    identity={"parameters":p.to_dict()}
    header={"fingerprint":fingerprint(identity),"identity":identity,
            "initial_state":state.to_dict(),"dayahead_plan":{"0:0":[0,1]}}
    (journal/"initial.json").write_text(json.dumps(header),encoding="utf-8")
    record={"period":0,"horizon":1,"state_before":state_before,
            "forecast":{"random_requests":[],"start_time":0.,"end_time":1/12},
            "solution":{"status":"optimal"}}
    (journal/"rounds.jsonl").write_text(json.dumps(record)+"\n",encoding="utf-8")
    (pilot/"network.json").write_text(json.dumps(generate_candidate_network(p)),encoding="utf-8")
    return journal


class DiagnosticTests(unittest.TestCase):
    def test_restores_snapshot_and_does_not_change_source_solver_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);journal=fixture(root)
            original=(journal/"initial.json").read_bytes()
            p,window,metadata=prepare_window(journal,0)
            self.assertEqual(p.solver.time_limit_sec,10.)
            self.assertEqual(metadata["original_parameter_snapshot"]["solver"]["time_limit_sec"],60.)
            self.assertEqual(window.state.ledger,[])
            self.assertEqual(metadata["original_ledger_event_count"],7)
            self.assertEqual((journal/"initial.json").read_bytes(),original)

    def test_dry_run_never_calls_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);journal=fixture(root)
            with patch("diagnose_terminal_window.solve_mpc") as solve, contextlib.redirect_stdout(io.StringIO()):
                code=main(["--journal",str(journal),"--period","0","--output-dir",str(root/"preview")])
            self.assertEqual(code,0);solve.assert_not_called()
            self.assertFalse((root/"preview"/"diagnostic_results.json").exists())

    def test_supplied_pilot_model_reuses_isolated_window_and_direct_value_check(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);journal=fixture(root)
            p,window,metadata=prepare_window(journal,0,solve_seconds=60.)
            spec=FeatureSpec(p,"full")
            model=make_untrained_model(spec,"linear")
            vector=spec.encode_window(window).tolist()
            value=float(model.predict(vector))
            solution=SimpleNamespace(status="optimal",mip_gap=0.,build_seconds=.01,
                solve_seconds=.02,wall_seconds=.03,terminal_features=vector,
                terminal_value=value,objective=value,diagnostics={"model_variables":1},
                to_dict=lambda:{"terminal_features":vector,"terminal_value":value})
            with patch("diagnose_terminal_window.solve_mpc",return_value=solution) as solve, contextlib.redirect_stdout(io.StringIO()):
                result=run_comparison(p,window,root/"diagnostic",metadata=metadata,
                    methods=("linear",),supplied_models={"linear":model},purpose="pilot-only fixture")
            self.assertTrue(result["trained_model"])
            self.assertEqual(result["purpose"],"pilot-only fixture")
            self.assertEqual(result["methods"][0]["network_mip_absolute_error"],0.)
            self.assertTrue(result["methods"][0]["input_window_unchanged"])
            self.assertIsNot(solve.call_args.args[1],window)
            envelope=json.loads((root/"diagnostic/linear/pilot_only_model.json").read_text(encoding="utf-8"))
            self.assertEqual(envelope["scope"],"pilot_only")
            self.assertFalse((root/"diagnostic/linear/untrained_model.json").exists())

    def test_three_small_methods_report_feasibility_and_value_consistency(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);journal=fixture(root)
            before={path:path.read_bytes() for path in journal.parent.rglob("*") if path.is_file()}
            p,window,metadata=prepare_window(journal,0)
            with contextlib.redirect_stdout(io.StringIO()):
                result=run_comparison(p,window,root/"diagnostic",metadata=metadata)
            self.assertEqual(result["state"],"complete")
            self.assertFalse(result["trained_model"])
            self.assertFalse(result["complete_operating_return"])
            self.assertEqual([row["method"] for row in result["methods"]],["zero","linear","relu"])
            for row in result["methods"]:
                self.assertTrue(row["feasible_incumbent"])
                self.assertTrue(row["input_window_unchanged"])
                self.assertGreater(row["diagnostics"]["model_variables"],0)
                if row["method"] != "zero":
                    self.assertLess(row["network_mip_absolute_error"],1e-6)
            for path,content in before.items():
                self.assertEqual(path.read_bytes(),content)
            with self.assertRaises(FileExistsError):
                run_comparison(p,window,root/"diagnostic",metadata=metadata)

if __name__=="__main__":unittest.main()
