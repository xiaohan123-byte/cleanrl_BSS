"""Isolated figure fixtures, never scientific results or solver runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.experiment_control import atomic_json
from src.experiment_plots import (COST_METRICS, IncompletePlot, _require_groups,
    _stat, _training_data, export_experiment_plots)
from src.experiment_reporting import _environment


class ExperimentPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="expressway_plot_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = self.root/"formal_suite"

    def fixture(self):
        plan = {"fixture":True,"budget":{"training_replicates":1,"outer_iterations":2,
                "fit_epochs":20,"validation_days":2},"seeds":{"test":[41000]},"target_scale_yuan":1000.}
        groups = {}
        for name,profit in (("zero",100.),("linear_full_rep0",115.),("relu_full_rep0",130.)):
            means = {"net_profit_yuan":profit,"service_income_yuan":profit+50.,
                     "charging_cost_yuan":30.,"adjustment_cost_yuan":15.,"reservation_failure_cost_yuan":5.}
            groups["base/"+name] = {"status":"complete","scenario_seeds":[41000],
                "statistics":{key:{"mean":value,"n":1,"ci95_low":value-10,"ci95_high":value+10}
                              for key,value in means.items()}}
        report = {"fixture":True,"status":"core_complete_paper_pending","groups":groups}
        atomic_json(self.suite/"plan.json",plan)
        atomic_json(self.suite/"report.json",report)
        return report,plan

    def test_single_fixture_figure_png_pdf_and_exact_values(self):
        # This is the only test that renders a figure; all artifacts are temporary.
        from PIL import Image
        from matplotlib.figure import Figure
        self.fixture()
        captured = []
        original = Figure.savefig
        def capture(figure,*args,**kwargs):
            captured.append(([bar.get_height() for bar in figure.axes[0].patches],
                             [bar.get_height() for bar in figure.axes[1].patches],
                             [text.get_text() for text in figure.texts]))
            return original(figure,*args,**kwargs)
        with patch.object(Figure,"savefig",capture):
            manifest = json.loads(Path(export_experiment_plots(self.root,["core"])).read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"],"complete")
        self.assertEqual(manifest["observed_suite_status"],"core_complete_paper_pending")
        self.assertEqual(manifest["complete_figure_count"],1)
        row = manifest["figures"][0]
        self.assertEqual([v["values"]["net_profit_yuan"]["mean"] for v in row["plotted_values"]],[100.,115.,130.])
        self.assertTrue(all(v["values"]["net_profit_yuan"]["ci95"] is None for v in row["plotted_values"]))
        self.assertEqual(captured[0][0],[100.,115.,130.])
        self.assertEqual(captured[0][1],[150.,165.,180.,-30.,-30.,-30.,-15.,-15.,-15.,-5.,-5.,-5.])
        self.assertTrue(any("FIXTURE" in text for text in captured[0][2]))
        self.assertEqual(len(captured),2)  # same figure exported in two formats
        png,pdf=Path(row["outputs"]["png"]),Path(row["outputs"]["pdf"])
        self.assertTrue(png.name.startswith("fixture_"))
        self.assertEqual(png.read_bytes()[:8],b"\x89PNG\r\n\x1a\n")
        self.assertEqual(pdf.read_bytes()[:5],b"%PDF-")
        with Image.open(png) as image:
            self.assertGreater(image.width,2000)
            self.assertGreater(image.height,800)
        self.assertEqual(len(list((self.root/"figures").glob("*.png"))),1)
        self.assertEqual(len(list((self.root/"figures").glob("*.pdf"))),1)

    def test_absent_report_only_pending_manifest(self):
        with patch("src.experiment_plots._plot_style",side_effect=AssertionError("must not render")):
            path=Path(export_experiment_plots(self.root))
        data=json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["status"],"pending")
        self.assertEqual(data["complete_figure_count"],0)
        self.assertEqual(list((self.root/"figures").iterdir()),[path])

    def test_incomplete_comparison_skips_without_rendering_or_zero_fill(self):
        report,plan=self.fixture()
        report["groups"]["base/relu_full_rep0"]["status"]="paused"
        atomic_json(self.suite/"report.json",report)
        with patch("src.experiment_plots._plot_style",side_effect=AssertionError("must not render")):
            data=json.loads(Path(export_experiment_plots(self.root,["core"])).read_text(encoding="utf-8"))
        self.assertEqual(data["status"],"pending")
        self.assertEqual(data["figures"][0]["status"],"skipped")
        self.assertNotIn("plotted_values",data["figures"][0])
        self.assertEqual(list((self.root/"figures").glob("*.png")),[])

    def test_ci_requires_two_distinct_scenarios_and_matching_test_set(self):
        report,plan=self.fixture()
        group=report["groups"]["base/zero"]
        self.assertIsNone(_stat(group,"net_profit_yuan")["ci95"])
        group["scenario_seeds"].append(41001)
        group["statistics"]["net_profit_yuan"]["n"]=2
        self.assertEqual(_stat(group,"net_profit_yuan")["ci95"],[90.,110.])
        with self.assertRaisesRegex(IncompletePlot,"paired test set"):
            _require_groups(report,["base/zero"],["net_profit_yuan"],plan["seeds"]["test"])

    def test_training_curves_require_complete_fits_and_matching_selection(self):
        _,plan=self.fixture()
        directory=self.suite/"models/base/relu_full_rep0"
        candidates=[]
        for iteration in range(2):
            validation={"iteration":iteration,"validation_net_profit_yuan":80.+iteration*20,
                        "validation_jobs":[f"val/{iteration}/1",f"val/{iteration}/2"]}
            fit={"iteration":iteration,"training_replicate":0,"only_current_policy_labels":True,
                 "diagnostics":{"kind":"relu","epochs":20,"target_scale_yuan":1000.,
                                "history":[{"epoch":1,"mse_scaled":.8},{"epoch":20,"mse_scaled":.2}]}}
            atomic_json(directory/f"iteration_{iteration}_fit.json",fit)
            atomic_json(directory/f"iteration_{iteration}_validation.json",validation)
            candidates.append(validation)
        selection={"kind":"relu","variant":"full","training_replicate":0,"candidates":candidates}
        atomic_json(directory/"selection.json",selection)
        rows,sources=_training_data(self.suite,"relu","full",0,plan)
        self.assertEqual([row["validation_net_profit_yuan"] for row in rows],[80.,100.])
        self.assertEqual(rows[0]["training_mse_scaled"],[.8,.2])
        self.assertEqual(len(sources),5)
        changed=dict(candidates[1],validation_net_profit_yuan=900.)
        atomic_json(directory/"iteration_1_validation.json",changed)
        with self.assertRaisesRegex(IncompletePlot,"provenance"):
            _training_data(self.suite,"relu","full",0,plan)

    def test_hardware_os_preferred_and_raw_platform_retained(self):
        atomic_json(self.root/"execution_environment.json",{"platform":"Windows-10-raw"})
        atomic_json(self.root/"hardware_environment.json",{
            "operating_system":{"Caption":"Microsoft Windows 11 Home China","Version":"10.0.26200"},
            "cpu":[{"Name":"i7-12650H"}],"gpu":{"name":"RTX 3060 Laptop GPU","memory_mib":6144},
            "training_device_protocol":"CPU"})
        environment=_environment(self.root)
        self.assertEqual(environment["operating_system"],"Microsoft Windows 11 Home China 10.0.26200")
        self.assertEqual(environment["platform"],"Windows-10-raw")
        self.assertEqual(environment["training_device_protocol"],"CPU")


if __name__=="__main__":
    unittest.main()
