"""Independent feature audit against the physical execution transition."""
import copy
import unittest
import numpy as np

from src.candidate_network import generate_candidate_network
from src.domain import MPCSolution, Reservation, RollingState, ServiceDecision, WaitingRequest, initial_state
from src.execution import advance_to_boundary, execute_step
from src.forecast import build_forecast, deterministic_random_requests
from src.mpc_model import solve_mpc
from src.request_builder import build_window
from src.scenario import SyntheticScenario
from src.terminal_features import FeatureSpec
from src.terminal_value import TerminalValueModel
from tests.test_terminal_execution import parameters, scenario


class TerminalConsistencyAudit(unittest.TestCase):
    def _params(self):
        p = parameters()
        p.random_hourly_means = [[0., 0.], [0., 0.]]
        p.solver.time_limit_sec = 5.
        p.solver.mip_gap = 0.
        p.validate()
        return p

    def _compare_transition(self, p, truth, state, horizon=1):
        road = generate_candidate_network(p)
        now = state.period * p.interval_hours
        state = advance_to_boundary(p, state,
            [r for r in truth.observation_at(now).random_history if r["arrival_time"] == now],
            scenario=truth).state
        forecast = build_forecast(p, truth.observation_at(now), state.period, horizon)
        window = build_window(p, state, road, forecast, horizon=horizon)
        for network in window.networks.values():
            network.can_update = False
        spec = FeatureSpec(p)
        value = TerminalValueModel("linear", spec.names, "full", np.zeros(spec.dimension), output_scale=1000.)
        solution = solve_mpc(p, window, terminal_model=value, feature_spec=spec)
        candidate = {r.request_id: r for r in window.requests}
        realised = state
        for offset in range(horizon):
            period = state.period + offset
            left, right = period * p.interval_hours, (period + 1) * p.interval_hours
            actions = []
            for action in solution.services:
                if action.period != period:
                    continue
                rid = action.request_id
                if rid not in realised.waiting:
                    req = candidate[rid]
                    matches = [w for w in realised.waiting.values()
                               if w.station == req.station and w.kind == req.kind and w.user_key == req.user_key]
                    self.assertEqual(len(matches), 1, "prediction/actual service identity is ambiguous")
                    rid = matches[0].request_id
                actions.append(ServiceDecision(rid, action.station, action.slot, period))
            step = MPCSolution("optimal", 0., {}, solution.paths if offset == 0 else {}, actions,
                [[[solution.power[i][b][offset]] for b in range(p.station.num_slots)]
                 for i in range(p.station.num_stations)], [])
            realised = execute_step(p, realised, step, truth.arrivals_between(left, right), scenario=truth).state
            observed_at_end = [r for r in truth.observation_at(right).random_history if r["arrival_time"] == right]
            realised = advance_to_boundary(p, realised, observed_at_end, scenario=truth).state
        terminal_time = realised.period * p.interval_hours
        next_forecast = build_forecast(p, truth.observation_at(terminal_time), realised.period, 1)
        next_window = build_window(p, realised, road, next_forecast, horizon=1)
        encoded = spec.encode_window(next_window)
        diff = np.abs(encoded - np.asarray(solution.terminal_features))
        bad = [(spec.names[index], float(solution.terminal_features[index]), float(encoded[index]))
               for index in np.flatnonzero(diff > 2e-7)]
        self.assertFalse(bad, f"MIP versus independently executed terminal state: {bad[:12]}")
        return spec, solution, realised, encoded

    def test_current_observed_segment_speed_survives_same_segment_projection(self):
        p = self._params()
        truth = scenario(p, actual_soc=.8, report_soc=.8, multipliers=[1.1, 1., 1.])
        user = Reservation((0, 0), 0., .8, 5., .75, True, day_ahead=[0, 1],
                           retained_plan=[0, 1], published_plan=[0, 1], actual_entry_time=0.,
                           observation_time=0., observed_segment_index=0, observed_speed_kmh=60 / 1.1)
        state = RollingState(0, [[1.], [1.]], {"0:0": user})
        _, _, end, _ = self._compare_transition(p, truth, state)
        self.assertAlmostEqual(end.users["0:0"].position_km, 5. + 5 / 1.1)

    def test_current_speed_does_not_extend_to_unobserved_future_segment(self):
        p = self._params()
        truth = scenario(p, actual_soc=.8, report_soc=.8, multipliers=[1.1, 1., 1.])
        user = Reservation((0, 0), 0., .8, 5., .75, True, day_ahead=[1],
                           retained_plan=[1], published_plan=[1], actual_entry_time=0.,
                           observation_time=0., observed_segment_index=0, observed_speed_kmh=60 / 1.1)
        state = RollingState(0, [[1.], [1.]], {"0:0": user})
        _, _, end, _ = self._compare_transition(p, truth, state, horizon=2)
        self.assertAlmostEqual(end.users["0:0"].position_km, 14.5)

    def test_future_entry_initializes_predicted_first_publication(self):
        p = self._params()
        truth = scenario(p, actual_time=.025, report_time=.025, actual_soc=.8, report_soc=.8,
                         multipliers=[1., 1., 1.])
        state = initial_state(p, truth.initial_reservations(), {"0:0": [0, 1]})
        # The retained plan differs from the original day-ahead plan.
        state.users["0:0"].day_ahead = [1]
        spec, _, end, encoded = self._compare_transition(p, truth, state)
        self.assertTrue(end.users["0:0"].entered)
        self.assertEqual(end.users["0:0"].published_plan, [0, 1])
        self.assertAlmostEqual(encoded[spec.index["chain:1:0,1:published:changed_count"]], 0.)
        self.assertAlmostEqual(encoded[spec.index["chain:1:0,1:dayahead:changed_count"]], .01)

    def test_predecessor_service_updates_remaining_chain_and_anchor(self):
        p = self._params()
        truth = scenario(p, actual_soc=.8, report_soc=.8, multipliers=[1., 1., 1.])
        waiting = WaitingRequest("first", 0, "reservation", 0., .25, .2, (0, 0))
        user = Reservation((0, 0), 0., .8, 10., .2, True, day_ahead=[0, 1],
                           retained_plan=[1], published_plan=[1], waiting_request_id="first",
                           last_swap_position_km=0., actual_entry_time=0.)
        state = RollingState(0, [[1.], [1.]], {"0:0": user}, {"first": waiting})
        spec, _, end, encoded = self._compare_transition(p, truth, state)
        self.assertEqual(end.users["0:0"].completed_stations, [0])
        self.assertEqual(end.users["0:0"].last_swap_position_km, 10.)
        self.assertAlmostEqual(encoded[spec.index["chain:1:1:time:0:count"]], .01)
        self.assertTrue(all(encoded[i] == 0 for i, name in enumerate(spec.names)
                            if name.startswith("chain:1:0,1:time:") and name.endswith(":count")))

    def test_waiting_deadline_at_terminal_boundary_is_still_retained(self):
        p = self._params()
        truth = scenario(p, actual_soc=.8, report_soc=.8, multipliers=[1., 1., 1.])
        waiting = WaitingRequest("first", 0, "reservation", 0., 1 / 12, .2, (0, 0))
        user = Reservation((0, 0), 0., .8, 10., .2, True, day_ahead=[0, 1],
                           retained_plan=[1], published_plan=[1], waiting_request_id="first")
        state = RollingState(0, [[.2], [1.]], {"0:0": user}, {"first": waiting})
        spec, _, end, encoded = self._compare_transition(p, truth, state)
        self.assertIn("first", end.waiting)
        self.assertAlmostEqual(encoded[spec.index["chain:1:0,1:time:4:count"]], .01)

    def test_random_prediction_inside_window_becomes_real_waiting_once(self):
        p = self._params()
        p.random_hourly_means = [[10., 0.], [0., 0.]]
        truth = SyntheticScenario(p, [], [dict(request_id="actual-r", station=0, arrival_time=.05, return_soc=.2)])
        spec, _, end, encoded = self._compare_transition(p, truth, RollingState(0, [[1.], [1.]]))
        self.assertIn("actual-r", end.waiting)
        self.assertAlmostEqual(encoded[spec.index["random:0:time:5:count"]], .01)

    def test_random_arrival_exactly_at_terminal_boundary_uses_pre_service_state(self):
        p = self._params()
        p.random_hourly_means = [[6., 0.], [0., 0.]]
        truth = SyntheticScenario(p, [], [dict(request_id="actual-r", station=0, arrival_time=1 / 12, return_soc=.2)])
        spec, _, end, encoded = self._compare_transition(p, truth, RollingState(0, [[1.], [1.]]))
        self.assertEqual(end.waiting["actual-r"].arrival_time, 1 / 12)
        self.assertAlmostEqual(encoded[spec.index["random:0:time:5:count"]], .01)

    def test_current_random_features_do_not_depend_on_mpc_horizon(self):
        p = self._params()
        p.random_hourly_means = [[1., 3.], [.25, .25]]
        truth = SyntheticScenario(p, [], [])
        state = RollingState(0, [[1.], [1.]])
        road = generate_candidate_network(p)
        spec = FeatureSpec(p)
        a = build_window(p, state, road, build_forecast(p, truth.observation_at(0.), 0, 1), horizon=1)
        b = build_window(p, state, road, build_forecast(p, truth.observation_at(0.), 0, 12), horizon=12)
        np.testing.assert_array_equal(spec.encode_window(a), spec.encode_window(b))


if __name__ == "__main__":
    unittest.main()
