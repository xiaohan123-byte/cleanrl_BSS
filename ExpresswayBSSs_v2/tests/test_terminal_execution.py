"""Regression checks for private truth, directed motion and terminal-day rules."""
import copy
import unittest

from src.accounting import summarize_ledger
from src.candidate_network import generate_candidate_network, get_feasible_arcs, validate_candidate_network
from src.dayahead_plan import generate_dayahead_plan
from src.domain import MPCSolution, Reservation, RollingState, ServiceDecision, WaitingRequest, initial_state
from src.execution import advance_to_boundary, execute_step
from src.forecast import build_forecast, deterministic_random_requests, forecast_motion, predicted_entry_time, predict_travel_time
from src.parameters import BusinessParameters, ODPairParameters, StationParameters
from src.path_state import apply_path_decisions
from src.request_builder import build_window
from src.scenario import SyntheticScenario


def parameters(reverse=False):
    p = BusinessParameters(
        num_periods=24, horizon=12, num_reservations=1, vehicle_speed_kmh=60.,
        range_km=100., min_swap_spacing_km=0., terminal_experiment=True,
        station=StationParameters(num_stations=2, station_ids=[0, 1], positions_km=[10., 30.],
                                  num_slots=1, initial_slot_soc=[[1.], [1.]],
                                  slot_power_limits_kw=[[60.], [60.]], station_power_limits_kw=[60., 60.]),
        od_pairs=[ODPairParameters(0, 50., 0., [1, 0]) if reverse else ODPairParameters(0, 0., 50., [0, 1])],
        od_sampling_weights=[1.], reservation_hourly_weights=[1., 1.],
        random_hourly_means=[[1., 3.], [.25, .25]], entry_time_error_hours=1 / 6,
        entry_soc_error=.05, report_entry_soc_range=[.5, 1.], reservation_entry_soc_range=[.5, 1.],
        travel_time_relative_error=.1, random_return_soc_range=[.1, .3], random_soc_prediction=.2)
    p.validate()
    return p


def scenario(p, actual_time=0., report_time=0., actual_soc=.76, report_soc=.8, multipliers=None):
    return SyntheticScenario(p, [dict(user_key=[0, 0], od_id=0,
        entry_time=report_time, entry_soc=report_soc,
        actual_entry_time=actual_time, actual_entry_soc=actual_soc,
        segment_time_multipliers=[1.1, .9, 1.] if multipliers is None else multipliers)], [])


def idle(p, paths=None, services=None):
    return MPCSolution("optimal", 0., {}, paths or {}, services or [],
                       [[[0.] for _ in range(p.station.num_slots)] for _ in range(p.station.num_stations)], [])


class TerminalExecutionTests(unittest.TestCase):
    def test_reverse_network_and_dayahead_keep_global_coordinates(self):
        p = parameters(True)
        network = generate_candidate_network(p)
        validate_candidate_network(network, p)
        self.assertEqual(p.distance_km(0, "entry", 1), 20.)
        arcs = get_feasible_arcs(network, 0, .55)
        self.assertIn(("entry", 1), arcs)
        self.assertIn((1, 0), arcs)
        self.assertEqual(generate_dayahead_plan(p, network, [dict(user_key=[0, 0], od_id=0, entry_soc=.6)])["0:0"], [0])

    def test_actual_entry_and_soc_are_not_announced_truth(self):
        p = parameters()
        truth = scenario(p, actual_time=.05, report_time=0.)
        state = initial_state(p, truth.initial_reservations(), {"0:0": [0, 1]})
        self.assertFalse(state.users["0:0"].entered)
        self.assertIsNone(state.users["0:0"].actual_entry_time)
        self.assertNotIn("segment_time_multipliers", state.users["0:0"].to_dict())
        at_zero = advance_to_boundary(p, state, scenario=truth)
        self.assertFalse(at_zero.state.users["0:0"].entered)
        result = execute_step(p, at_zero.state, idle(p), scenario=truth)
        user = result.state.users["0:0"]
        expected_distance = (p.interval_hours - .05) * 60 / 1.1
        self.assertAlmostEqual(user.position_km, expected_distance)
        self.assertAlmostEqual(user.soc, .76 - expected_distance / 100)
        self.assertEqual(user.entry_soc, .8)
        self.assertEqual(user.actual_entry_time, .05)
        self.assertEqual(user.published_plan, [0, 1])
        self.assertEqual(summarize_ledger(result.state.ledger)["adjustment_cost"], 0.)

    def test_future_segment_truth_does_not_change_current_visible_state(self):
        p = parameters()
        one = scenario(p, multipliers=[1.1, .9, 1.])
        two = scenario(p, multipliers=[1.1, 1.1, .9])
        state = initial_state(p, one.initial_reservations(), {"0:0": [1]})
        a = execute_step(p, advance_to_boundary(p, state, scenario=one).state, idle(p), scenario=one)
        b = execute_step(p, advance_to_boundary(p, state, scenario=two).state, idle(p), scenario=two)
        self.assertEqual(a.state.to_dict(), b.state.to_dict())
        user = a.state.users["0:0"]
        expected = p.interval_hours + (10 - user.position_km) / (60 / 1.1) + 20 / 60
        self.assertAlmostEqual(user.next_arrival_time, expected)

    def test_actual_motion_across_segments_is_additive(self):
        p = parameters()
        truth = scenario(p, multipliers=[1.1, .9, 1.])
        state = initial_state(p, truth.initial_reservations(), {"0:0": [1]})
        state = advance_to_boundary(p, state, scenario=truth).state
        for _ in range(6):
            state = execute_step(p, state, idle(p), scenario=truth).state
        waiting = state.waiting["A:0:0:0"]
        self.assertAlmostEqual(waiting.arrival_time, (10 * 1.1 + 20 * .9) / 60)
        self.assertAlmostEqual(waiting.return_soc, .76 - 30 / 100)
        self.assertAlmostEqual(waiting.deadline, waiting.arrival_time + .25)

    def test_reverse_execution_uses_correct_segment(self):
        p = parameters(True)
        truth = scenario(p, multipliers=[.9, 1., 1.1])
        state = initial_state(p, truth.initial_reservations(), {"0:0": [1, 0]})
        state = advance_to_boundary(p, state, scenario=truth).state
        for _ in range(5):
            state = execute_step(p, state, idle(p), scenario=truth).state
        waiting = state.waiting["A:0:0:0"]
        self.assertEqual(waiting.station, 1)
        self.assertAlmostEqual(waiting.arrival_time, 20 * 1.1 / 60)
        self.assertAlmostEqual(waiting.return_soc, .56)

    def test_conservative_soc_only_filters_paths_not_return_soc_forecast(self):
        p = parameters()
        truth = scenario(p, actual_time=.1, report_time=.1, actual_soc=.6, report_soc=.62)
        state = initial_state(p, truth.initial_reservations(), {"0:0": [1]})
        forecast = build_forecast(p, truth.observation_at(0.), 0, p.horizon)
        window = build_window(p, state, generate_candidate_network(p), forecast)
        self.assertNotIn(("entry", "exit"), window.networks["0:0"].arcs)
        request = next(r for r in window.requests if r.arc == ("entry", 1))
        self.assertAlmostEqual(request.return_soc, .62 - .3)

    def test_late_report_prediction_keeps_original_support(self):
        p = parameters()
        user = Reservation((0, 0), 1., .8, 0., .8)
        self.assertEqual(predicted_entry_time(p, user, .9), 1.)
        self.assertAlmostEqual(predicted_entry_time(p, user, 1.05), (1.05 + 1 + 1 / 6) / 2)
        self.assertAlmostEqual(predicted_entry_time(p, user, 1.1), (1.1 + 1 + 1 / 6) / 2)
        self.assertEqual(user.entry_time, 1.)
        self.assertFalse(user.entered)
        with self.assertRaisesRegex(ValueError, "report support"):
            predicted_entry_time(p, user, 1 + 1 / 6)

    def test_motion_prediction_only_uses_observed_current_segment(self):
        p = parameters()
        user = Reservation((0, 0), 0., .8, 5., .8, True,
                           observed_segment_index=0, observed_speed_kmh=50.)
        self.assertAlmostEqual(predict_travel_time(p, user, 5., 30.), 5 / 50 + 20 / 60)
        motion = forecast_motion(p, user, 1, .2, now=0.)
        self.assertAlmostEqual(motion["position_km"], 16.)
        self.assertAlmostEqual(motion["soc"], .69)
        from_swap = forecast_motion(p, user, 1, .2, departure_station=0, departure_time=0., now=0.)
        self.assertAlmostEqual(from_swap["position_km"], 22.)
        self.assertAlmostEqual(from_swap["soc"], .88)

    def test_fixed_forecast_retains_midpoint_anchors_across_windows(self):
        p = parameters()
        full = deterministic_random_requests(p, 0., 2.)
        arrivals = [r["arrival_time"] for r in full if r["station"] == 0]
        for observed, expected in zip(arrivals, [.5, 1 + .5 / 3, 1.5, 1 + 2.5 / 3]):
            self.assertAlmostEqual(observed, expected)
        self.assertEqual(len(arrivals), 4)
        self.assertTrue(all(r["return_soc"] == .2 for r in full))
        later = deterministic_random_requests(p, .5, 1.75)
        self.assertEqual(later, [r for r in full if .5 < r["arrival_time"] < 1.75])

    def test_entry_on_update_boundary_initializes_free_reference_first(self):
        p = parameters()
        truth = scenario(p, actual_time=.25, report_time=.2)
        state = initial_state(p, truth.initial_reservations(), {"0:0": [0, 1]})
        state.period = 3
        admission = advance_to_boundary(p, state, scenario=truth)
        self.assertEqual(admission.reward, 0.)
        self.assertEqual(admission.state.users["0:0"].published_plan, [0, 1])
        result = execute_step(p, admission.state, idle(p, paths={"0:0": [1]}), scenario=truth)
        self.assertEqual(summarize_ledger(result.state.ledger)["adjustment_cost"], 1.)

    def test_final_planned_swap_retires_user_before_highway_exit(self):
        p = parameters()
        request = WaitingRequest("A:0:0:0", 0, "reservation", 0., .25, .2, (0, 0))
        user = Reservation((0, 0), 0., .8, 10., .2, True, retained_plan=[0], published_plan=[0],
                           waiting_request_id=request.request_id)
        state = RollingState(0, [[1.], [1.]], {"0:0": user}, {request.request_id: request})
        result = execute_step(p, state, idle(p, paths={"0:0": []},
            services=[ServiceDecision(request.request_id, 0, 0, 0)]))
        self.assertEqual(result.state.users["0:0"].status, "completed")
        self.assertEqual(result.state.users["0:0"].position_km, 10.)
        with self.assertRaisesRegex(ValueError, "inactive"):
            apply_path_decisions(p, result.state, {"0:0": [1]})

    def test_entry_with_empty_plan_retires_before_next_optimization(self):
        p = parameters()
        truth = scenario(p, actual_soc=.8, report_soc=.8, multipliers=[1., 1., 1.])
        state = initial_state(p, truth.initial_reservations(), {"0:0": []})
        admission = advance_to_boundary(p, state, scenario=truth)
        self.assertEqual(admission.state.users["0:0"].status, "completed")
        self.assertEqual(admission.state.users["0:0"].published_plan, [])
        forecast = build_forecast(p, truth.observation_at(0.), 0, 1)
        window = build_window(p, admission.state, generate_candidate_network(p), forecast, horizon=1)
        self.assertNotIn("0:0", window.networks)
        self.assertEqual(summarize_ledger(admission.state.ledger)["adjustment_cost"], 0.)


    def test_entered_user_empty_published_plan_retires_without_exit_event(self):
        p = parameters()
        user = Reservation((0, 0), 0., .8, 5., .75, True, retained_plan=[1], published_plan=[1])
        state = RollingState(0, [[1.], [1.]], {"0:0": user}, {})
        events = apply_path_decisions(p, state, {"0:0": []})
        self.assertEqual(state.users["0:0"].status, "completed")
        self.assertEqual(state.users["0:0"].position_km, 5.)
        self.assertEqual([event["type"] for event in events], ["path_adjustment"])
        truth = scenario(p, multipliers=[1., 1., 1.])
        state.period = 3
        forecast = build_forecast(p, truth.observation_at(.25), 3, 1)
        window = build_window(p, state, generate_candidate_network(p), forecast, horizon=1)
        self.assertNotIn("0:0", window.networks)
        with self.assertRaisesRegex(ValueError, "inactive"):
            apply_path_decisions(p, state, {"0:0": [1]})

    def test_empty_plan_retirement_does_not_change_legacy_baseline(self):
        p = parameters()
        p.terminal_experiment = False
        user = Reservation((0, 0), 0., .8, 5., .75, True, retained_plan=[1], published_plan=[1])
        state = RollingState(0, [[1.], [1.]], {"0:0": user}, {})
        apply_path_decisions(p, state, {"0:0": []})
        self.assertEqual(state.users["0:0"].status, "active")
        truth = scenario(p, actual_time=.05, report_time=.05, multipliers=[1., 1., 1.])
        state = initial_state(p, truth.initial_reservations(), {"0:0": []})
        result = execute_step(p, state, idle(p))
        self.assertTrue(result.state.users["0:0"].entered)
        self.assertEqual(result.state.users["0:0"].status, "active")
        self.assertFalse(any(event["type"] == "reservation_exit" for event in result.events))

    def test_operating_end_does_not_settle_pending_reservation(self):
        p = parameters()
        request = WaitingRequest("A:0:0:0", 0, "reservation", 23 / 12, 2., .2, (0, 0))
        user = Reservation((0, 0), 0., .8, 10., .2, True, retained_plan=[0, 1], published_plan=[0, 1],
                           waiting_request_id=request.request_id)
        state = RollingState(23, [[1.], [1.]], {"0:0": user}, {request.request_id: request})
        result = execute_step(p, state, idle(p))
        self.assertEqual(result.state.period, 24)
        self.assertIn(request.request_id, result.state.waiting)
        self.assertEqual(result.state.users["0:0"].status, "active")
        self.assertEqual(summarize_ledger(result.state.ledger)["reservation_failures"], 0)


if __name__ == "__main__":
    unittest.main()
