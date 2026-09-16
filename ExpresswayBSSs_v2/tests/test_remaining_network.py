"""Independent paper-fidelity checks using actual generated candidate networks."""

import copy
import unittest
from types import SimpleNamespace

from src.candidate_network import enumerate_paths, generate_candidate_network, get_feasible_arcs
from src.domain import DomainError, Reservation, RollingState, WaitingRequest
from src.parameters import BusinessParameters, ODPairParameters, StationParameters, get_default_parameters
from src.mpc_model import solve_mpc
from src.path_state import apply_path_decisions, build_remaining_network
from src.request_builder import build_window


def make_state(params, user, period=3, waiting=None):
    return RollingState(period, copy.deepcopy(params.station.initial_slot_soc),
                        {f"{user.user_key[0]}:{user.user_key[1]}": user}, waiting or {})


class RemainingNetworkTests(unittest.TestCase):
    def test_spacing_uses_last_swap_not_moving_virtual_origin(self):
        params = get_default_parameters()
        user = Reservation((1, 0), 0., .9, 260., 1. - 80. / 300., True,
                           day_ahead=[1, 2, 4], retained_plan=[2, 4],
                           published_plan=[1, 2, 4], last_swap_position_km=180.,
                           completed_stations=[1])
        state = make_state(params, user)
        remaining = build_remaining_network(params, state, user, generate_candidate_network(params))
        # It is only 20 km from the current position to station 2, but the
        # actual previous swap was 100 km upstream, so this edge stays legal.
        self.assertIn((remaining.origin, 2), remaining.arcs)
        self.assertNotIn(1, remaining.reference_stations)
        self.assertNotIn(1, remaining.frozen_stations)
        for source, target in remaining.arcs:
            if isinstance(source, int):
                self.assertGreater(params.station.positions_km[source], user.position_km)
            if isinstance(target, int):
                self.assertGreater(params.station.positions_km[target], user.position_km)

    def test_online_short_leg_is_pruned_against_last_actual_swap(self):
        params = BusinessParameters.from_dict({
            "station": {"num_stations": 3, "station_ids": [0, 1, 2],
                        "positions_km": [100., 150., 250.], "num_slots": 1,
                        "initial_slot_soc": [[1.], [1.], [1.]],
                        "slot_power_limits_kw": [[60.], [60.], [60.]],
                        "station_power_limits_kw": [60., 60., 60.]},
            "od_pairs": [{"od_id": 0, "entry_km": 0., "exit_km": 430., "station_indices": [0, 1, 2]}],
        })
        user = Reservation((0, 0), 0., 1., 120., 1. - 20. / 300., True,
                           retained_plan=[2], published_plan=[0, 2],
                           last_swap_position_km=100., completed_stations=[0])
        state = make_state(params, user)
        remaining = build_remaining_network(params, state, user, generate_candidate_network(params))
        # Station 1 is 50 km beyond the last swap; station 2 gives an alternative.
        self.assertNotIn((remaining.origin, 1), remaining.arcs)
        self.assertIn((remaining.origin, 2), remaining.arcs)

    def test_published_short_first_leg_remains_feasible_across_rounds(self):
        params = get_default_parameters()
        params.horizon = 2
        network = generate_candidate_network(params)
        # A short stop retained from an earlier SOC forecast remains a valid
        # choice when the observed SOC now also permits reaching 180 km.
        # Admission and update rounds must not erase that existing plan.
        for period in (1, 3):
            with self.subTest(period=period):
                position = period * params.interval_hours * params.vehicle_speed_kmh
                user = Reservation(
                    (0, 0), 0., .7, position, .7 - position / params.range_km, True,
                    day_ahead=[1], retained_plan=[0, 2], published_plan=[0, 2])
                state = make_state(params, user, period=period)
                window = build_window(params, state, network, SimpleNamespace(random_requests=[]))
                remaining = window.networks["0:0"]
                self.assertIn((remaining.origin, 0), remaining.arcs)
                self.assertIn((remaining.origin, 1), remaining.arcs)
                solution = solve_mpc(params, window)
                self.assertEqual(solution.status, "optimal")
                self.assertEqual(solution.paths, {"0:0": [0, 2]})

    def test_complete_existing_plan_survives_pruning_for_every_user_state(self):
        params = BusinessParameters(
            num_periods=6, horizon=2,
            station=StationParameters(
                num_stations=4, station_ids=[0, 1, 2, 3],
                positions_km=[100., 150., 200., 300.], num_slots=1,
                initial_slot_soc=[[1.]] * 4,
                slot_power_limits_kw=[[60.]] * 4,
                station_power_limits_kw=[60.] * 4),
            od_pairs=[ODPairParameters(0, 0., 480., [0, 1, 2, 3])])
        network = generate_candidate_network(params)
        # Both 100->150 and 150->200 are short and otherwise removable.
        self.assertNotIn((0, 1), get_feasible_arcs(network, 0, 1.))
        self.assertNotIn((1, 2), get_feasible_arcs(network, 0, 1.))
        for period in (1, 3):
            for mode in ('future', 'driving', 'waiting'):
                with self.subTest(period=period, mode=mode):
                    request = None
                    if mode == 'future':
                        user = Reservation((0, 0), 1., 1., 0., 1., False,
                                           day_ahead=[2], retained_plan=[0, 1, 2, 3])
                    elif mode == 'driving':
                        user = Reservation((0, 0), 0., 1., 25., 1. - 25. / 300., True,
                                           retained_plan=[0, 1, 2, 3],
                                           published_plan=[0, 1, 2, 3])
                    else:
                        request = WaitingRequest('actual:reservation:0:0:0', 0,
                                                 'reservation', .01, .26, .2, (0, 0))
                        user = Reservation((0, 0), 0., 1., 100., .2, True,
                                           retained_plan=[0, 1, 2, 3],
                                           published_plan=[0, 1, 2, 3],
                                           waiting_request_id=request.request_id)
                    waiting = {request.request_id: request} if request else {}
                    state = make_state(params, user, period=period, waiting=waiting)
                    remaining = build_remaining_network(params, state, user, network)
                    stations = [1, 2, 3] if mode == 'waiting' else [0, 1, 2, 3]
                    nodes = [remaining.origin, *stations, 'exit']
                    planned_path = tuple(zip(nodes, nodes[1:]))
                    self.assertIn(planned_path,
                                  enumerate_paths(remaining.arcs, remaining.origin))
                    self.assertEqual(remaining.can_update, period == 3)

    def test_waiting_user_reuses_full_battery_arcs_removed_at_entry(self):
        params = BusinessParameters(
            num_periods=6, horizon=2, range_km=100., min_swap_spacing_km=50.,
            station=StationParameters(
                num_stations=3, station_ids=[0, 1, 2],
                positions_km=[40., 70., 145.], num_slots=1,
                initial_slot_soc=[[1.]] * 3,
                slot_power_limits_kw=[[60.]] * 3,
                station_power_limits_kw=[60.] * 3),
            od_pairs=[ODPairParameters(0, 0., 225., [0, 1, 2])])
        network = generate_candidate_network(params)
        self.assertNotIn((0, 1), get_feasible_arcs(network, 0, .8))
        request = WaitingRequest('actual:reservation:0:0:0', 0,
                                 'reservation', .01, .26, .4, (0, 0))
        user = Reservation((0, 0), 0., .8, 40., .4, True,
                           retained_plan=[0, 1, 2], published_plan=[0, 1, 2],
                           waiting_request_id=request.request_id)
        state = make_state(params, user, waiting={request.request_id: request})
        remaining = build_remaining_network(params, state, user, network)
        # Once served at 40 km, the user has a full battery but still needs
        # 70 km before 145 km. Its old entry network cannot supply this arc.
        self.assertEqual(enumerate_paths(remaining.arcs, remaining.origin),
                         [((0, 1), (1, 2), (2, 'exit'))])

    def test_protection_never_restores_a_physically_unreachable_first_leg(self):
        params = get_default_parameters()
        user = Reservation((0, 0), 0., .7, 0., .3, True,
                           retained_plan=[1], published_plan=[1])
        state = make_state(params, user)
        remaining = build_remaining_network(params, state, user, generate_candidate_network(params))
        self.assertIn((remaining.origin, 0), remaining.arcs)
        self.assertNotIn((remaining.origin, 1), remaining.arcs)
        self.assertTrue(enumerate_paths(remaining.arcs, remaining.origin))

    def test_waiting_station_is_origin_and_current_request_is_not_duplicated(self):
        params = get_default_parameters()
        request = WaitingRequest("actual:reservation:1:0:1", 1, "reservation", .1, .35, .2, (1, 0))
        user = Reservation((1, 0), 0., .9, 180., .2, True,
                           day_ahead=[0, 1, 4], retained_plan=[1, 4],
                           published_plan=[0, 1, 4], last_swap_position_km=80.,
                           completed_stations=[0], waiting_request_id=request.request_id)
        state = make_state(params, user, waiting={request.request_id: request})
        window = build_window(params, state, generate_candidate_network(params), SimpleNamespace(random_requests=[]))
        remaining = window.networks["1:0"]
        self.assertEqual(remaining.origin, 1)
        self.assertEqual(remaining.reference_stations, (4,))
        self.assertEqual(remaining.frozen_stations, (4,))
        at_current = [record for record in window.requests if record.station == 1]
        self.assertEqual(len(at_current), 1)
        self.assertEqual(at_current[0].request_id, request.request_id)
        self.assertEqual(at_current[0].arrival_time, .1)
        self.assertEqual(at_current[0].deadline, .35)
        outgoing = [record for record in window.requests if record.arc is not None and record.arc[0] == 1]
        self.assertTrue(outgoing)
        for record in outgoing:
            self.assertEqual(record.predecessors, (request.request_id,))
            self.assertIsNone(record.arrival_time)
            distance = params.station.positions_km[record.station] - 180.
            self.assertAlmostEqual(record.return_soc, 1. - distance / params.range_km)

    def test_unreachable_actual_vehicle_soc_rejects_remaining_network(self):
        params = get_default_parameters()
        user = Reservation((0, 0), 0., .9, 400., .01, True, published_plan=[], retained_plan=[])
        state = make_state(params, user)
        with self.assertRaisesRegex(DomainError, "no reachable remaining path"):
            build_remaining_network(params, state, user, generate_candidate_network(params))

    def test_future_freeze_uses_retained_plan_and_dayahead_stays_reference(self):
        params = get_default_parameters()
        user = Reservation((0, 0), 1., 1., 0., 1., False,
                           day_ahead=[2], retained_plan=[1])
        state = make_state(params, user, period=1)
        offline = generate_candidate_network(params)
        remaining = build_remaining_network(params, state, user, offline)
        self.assertEqual(remaining.frozen_stations, (1,))
        self.assertEqual(remaining.reference_stations, (2,))
        self.assertFalse(remaining.can_update)
        self.assertEqual(apply_path_decisions(params, state, {"0:0": [1]}), [])
        with self.assertRaisesRegex(DomainError, "publication clock"):
            apply_path_decisions(params, state, {"0:0": [2]})
        self.assertEqual(user.retained_plan, [1])
        self.assertIsNone(user.published_plan)
        state.period = 3
        self.assertTrue(build_remaining_network(params, state, user, offline).can_update)
        self.assertEqual(apply_path_decisions(params, state, {"0:0": [2]}), [])

    def test_enroute_direct_exit_keeps_reachable_swap_alternative(self):
        params = get_default_parameters()
        user = Reservation((0, 0), 0., 1., 300., .9, True,
                           retained_plan=[3], published_plan=[2, 3],
                           last_swap_position_km=280., completed_stations=[2])
        state = make_state(params, user)
        remaining = build_remaining_network(params, state, user, generate_candidate_network(params))
        self.assertIn((remaining.origin, "exit"), remaining.arcs)
        self.assertIn((remaining.origin, 3), remaining.arcs)
        self.assertIn((3, "exit"), remaining.arcs)
        self.assertEqual(len(enumerate_paths(remaining.arcs, remaining.origin)), 2)
        window = build_window(params, state, generate_candidate_network(params), SimpleNamespace(random_requests=[]))
        for record in window.requests:
            self.assertGreaterEqual(record.return_soc, 0.)
            self.assertLess(record.return_soc, 1.)
        first = next(record for record in window.requests if record.arc == (remaining.origin, 3))
        self.assertAlmostEqual(first.return_soc, user.soc - 80. / params.range_km)


if __name__ == "__main__":
    unittest.main()
