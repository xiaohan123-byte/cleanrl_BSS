"""COPT integration contracts independent of the MILP's business formulation."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import Mock, patch

import coptpy as cp

from src.mpc_model import MPCSolveError, solve_mpc
from test_mpc_model import request, toy_parameters, window


class ModelProxy:
    """Run the real model, with controlled solve outcomes for error-path tests."""

    def __init__(self, model, overrides=None, solve_error=None):
        self.model = model
        self.overrides = overrides or {}
        self.solve_error = solve_error

    def __getattr__(self, name):
        if name in self.overrides:
            return self.overrides[name]
        return getattr(self.model, name)

    def solve(self):
        if self.solve_error is not None:
            raise self.solve_error
        return self.model.solve()


class COPTBackendTest(unittest.TestCase):
    def instrument_environment(self, overrides=None, solve_error=None):
        native_environment = cp.Envr
        environments, models = [], []

        def create_environment(config):
            native = native_environment(config)
            environment = Mock(wraps=native)

            def create_model(name):
                model = ModelProxy(native.createModel(name), overrides, solve_error)
                models.append(model)
                return model

            environment.createModel.side_effect = create_model
            environments.append(environment)
            return environment

        return patch.object(cp, "Envr", side_effect=create_environment), environments, models

    def test_missing_coptpy_reports_dependency_without_fallback(self):
        with patch.dict(sys.modules, {"coptpy": None}):
            with self.assertRaisesRegex(MPCSolveError, r"COPT \(coptpy\) is required"):
                solve_mpc(toy_parameters(), window([], [[0.2]]))

    def test_environment_initialization_error_is_actionable(self):
        with patch.object(cp, "Envr", side_effect=cp.CoptError(1, "license unavailable")):
            with self.assertRaisesRegex(MPCSolveError, "Unable to initialize COPT.*license unavailable"):
                solve_mpc(toy_parameters(), window([], [[0.2]]))

    def test_model_creation_error_closes_created_environment(self):
        environment = Mock()
        environment.createModel.side_effect = cp.CoptError(1, "model creation failed")
        with patch.object(cp, "Envr", return_value=environment):
            with self.assertRaisesRegex(MPCSolveError, "Unable to initialize COPT.*model creation failed"):
                solve_mpc(toy_parameters(), window([], [[0.2]]))
        environment.close.assert_called_once_with()

    def test_environment_close_error_does_not_hide_initialization_failure(self):
        environment = Mock()
        environment.createModel.side_effect = cp.CoptError(1, "model creation failed")
        environment.close.side_effect = cp.CoptError(1, "connection close failed")
        with patch.object(cp, "Envr", return_value=environment):
            with self.assertRaisesRegex(MPCSolveError, "Unable to initialize COPT.*model creation failed"):
                solve_mpc(toy_parameters(), window([], [[0.2]]))
        environment.close.assert_called_once_with()

    def test_time_limit_returns_feasible_incumbent_and_native_gap(self):
        instrumentation, environments, models = self.instrument_environment({
            "status": cp.COPT.TIMEOUT, "bestgap": 0.125,
        })
        with instrumentation:
            result = solve_mpc(toy_parameters(), window([request("ready")], [[1.0]]))
        self.assertEqual(result.status, "time_limit")
        self.assertEqual(result.mip_gap, 0.125)
        self.assertEqual([service.request_id for service in result.services], ["ready"])
        self.assertAlmostEqual(result.objective, 50.4)
        self.assertGreaterEqual(result.solve_seconds, 0.0)
        self.assertTrue(models[0].model.hasmipsol)
        environments[0].close.assert_called_once_with()

    def test_time_limit_without_incumbent_stops_and_closes_environment(self):
        instrumentation, environments, _ = self.instrument_environment({
            "status": cp.COPT.TIMEOUT, "hasmipsol": 0,
        })
        with instrumentation:
            with self.assertRaisesRegex(MPCSolveError, "time_limit without a feasible incumbent at period 0"):
                solve_mpc(toy_parameters(), window([request("ready")], [[1.0]]))
        environments[0].close.assert_called_once_with()

    def test_solver_failure_is_wrapped_and_closes_environment(self):
        instrumentation, environments, _ = self.instrument_environment(
            solve_error=cp.CoptError(1, "optimization failed"))
        with instrumentation:
            with self.assertRaisesRegex(MPCSolveError, "COPT failed at period 0.*optimization failed"):
                solve_mpc(toy_parameters(), window([], [[0.2]]))
        environments[0].close.assert_called_once_with()

    def test_lp_solution_and_requested_solver_tolerances(self):
        params = toy_parameters()
        params.solver.mip_gap = 0.0123
        params.solver.time_limit_sec = 0.75
        instrumentation, environments, models = self.instrument_environment()
        with instrumentation:
            result = solve_mpc(params, window([], [[0.2]], horizon=2))
        model = models[0].model
        self.assertFalse(model.ismip)
        self.assertFalse(model.hasmipsol)
        self.assertTrue(model.haslpsol)
        self.assertEqual(result.status, "optimal")
        self.assertEqual(result.mip_gap, 0.0)
        self.assertEqual(result.objective, 0.0)
        self.assertEqual(model.getParam(cp.COPT.Param.RelGap), 0.0123)
        self.assertEqual(model.getParam(cp.COPT.Param.AbsGap), 0.0)
        self.assertEqual(model.getParam(cp.COPT.Param.TimeLimit), 0.75)
        self.assertEqual(model.getParam(cp.COPT.Param.FeasTol), params.solver.feasibility_tol)
        self.assertEqual(model.getParam(cp.COPT.Param.IntTol), params.solver.feasibility_tol)
        self.assertEqual(model.getParam(cp.COPT.Param.Logging), 0)
        environments[0].close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
