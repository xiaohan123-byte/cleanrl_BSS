"""Formula-level tests for the six-station paper event MILP.

The optimization tests use the public :func:`solve_paper_mpc` API and a real
Gurobi model.  They are skipped when the machine has no usable license; the
failure-path tests remain active and require an explicit solver error rather
than a silent replay/greedy fallback.
"""

from __future__ import annotations

import copy
import unittest
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
from unittest.mock import patch

from data_generation_test.candidate_network import generate_candidate_network
from data_generation_test.parameter import ENTRY_NODE, EXIT_NODE, get_default_parameters
from data_generation_test.rl_data import RLSignals
from src.domain import (
    CandidateRequest,
    EnrouteReservation,
    PhysicalRequestStatus,
    RequestKind,
    RollingState,
    SlotState,
    WaitingRequest,
)
from src.event_engine import ContinuousEventEngine
from src.path_state import station_sequence
from src.reference_rollout import ReservationEventRollout, VisibleEntry
import src.paper_mpc as paper_mpc
from src.paper_mpc import PaperMPCSolverUnavailable, solve_paper_mpc
from src.time_grid import TimeGrid


UserKey = Tuple[int, int]
Arc = Tuple[Any, Any]


def _has_usable_gurobi_license() -> bool:
    gp = paper_mpc.gp
    if gp is None:
        return False
    environment = None
    model = None
    try:
        environment = gp.Env(empty=True)
        environment.setParam("OutputFlag", 0)
        environment.start()
        model = gp.Model("paper_mpc_test_probe", env=environment)
        model.Params.OutputFlag = 0
        model.addVar(vtype=gp.GRB.BINARY, name="probe")
        model.update()
        return True
    except Exception:
        return False
    finally:
        if model is not None:
            try:
                model.dispose()
            except Exception:
                pass
        if environment is not None:
            try:
                environment.dispose()
            except Exception:
                pass


GUROBI_LICENSE_AVAILABLE = _has_usable_gurobi_license()


def _params(horizon: int = 4):
    params = get_default_parameters()
    params.horizon = horizon
    params.solver.threads = 1
    params.solver.output_flag = 0
    params.solver.time_limit_sec = 30.0
    params.validate()
    return params


def _grid(params) -> TimeGrid:
    return TimeGrid(params.interval_hours, num_intervals=params.num_periods)


def _engine(params, grid: TimeGrid) -> ContinuousEventEngine:
    return ContinuousEventEngine(
        grid,
        params.battery_capacity_kwh,
        params.station.charging_efficiency,
        params.max_wait_hours,
        slot_power_limit_kw=params.station.slot_power_limit_kw,
        station_energy_limit_kwh=params.station_energy_limit_kwh,
    )


def _signals(params, horizon: int, *, station_zero_power_kw: float = 0.0) -> RLSignals:
    requested = [
        [
            [0.0 for _ in range(horizon)]
            for _ in range(params.station.num_slots)
        ]
        for _ in range(params.station.num_stations)
    ]
    requested[0][0] = [float(station_zero_power_kw) for _ in range(horizon)]
    return RLSignals(
        start_period=0,
        horizon=horizon,
        requested_power=requested,
        terminal_soc_value=[
            [0.0 for _ in range(params.station.num_slots)]
            for _ in range(params.station.num_stations)
        ],
        outside_swap_lambda=[0.0 for _ in range(params.station.num_stations)],
        signal_source="mock",
    )


def _state(
    params,
    *,
    full_slots: Iterable[Tuple[int, int]] = (),
    station_zero_soc: float | None = None,
) -> RollingState:
    full = set(full_slots)
    slots = []
    for station in range(params.station.num_stations):
        row = []
        for slot in range(params.station.num_slots):
            soc = 1.0 if (station, slot) in full else 0.2
            if (station, slot) == (0, 0) and station_zero_soc is not None:
                soc = station_zero_soc
            row.append(SlotState(station, slot, soc, last_update_time=0.0))
        slots.append(row)
    return RollingState(now=0.0, slots=slots)


def _plan_record(user_key: UserKey, path: Sequence[Arc], *, entry_soc: float) -> Dict[str, Any]:
    return {
        "reservation_id": user_key[1],
        "request_id": f"reservation_{user_key[1]}",
        "od_id": user_key[0],
        "user_key": list(user_key),
        "day_ahead_entry_time": 0.0,
        "day_ahead_entry_soc": entry_soc,
        "accepted": True,
        "path_arcs": [[source, target] for source, target in path],
    }


def _restrict_od_zero_network(network: Mapping[str, Any], params, paths: Sequence[Sequence[Arc]]):
    """Keep a tiny exact path menu while retaining the production schema."""

    result = copy.deepcopy(network)
    od_info = next(item for item in result["od_networks"] if item["od_index"] == 0)
    arcs = []
    for path in paths:
        for arc in path:
            if tuple(arc) not in arcs:
                arcs.append(tuple(arc))

    def arc_key(arc: Arc) -> str:
        return f"{arc[0]}->{arc[1]}"

    arc_distance = {
        arc_key(arc): params.distance_km(0, arc[0], arc[1]) for arc in arcs
    }
    arc_soc = {
        arc_key(arc): params.soc_consumption(0, arc[0], arc[1]) for arc in arcs
    }
    encoded_paths = [
        [[source, target] for source, target in path] for path in paths
    ]
    for bin_info in od_info["bins"]:
        encoded_arcs = [[source, target] for source, target in arcs]
        bin_info["raw_arcs"] = copy.deepcopy(encoded_arcs)
        bin_info["removed_arcs"] = []
        bin_info["candidate_arcs"] = copy.deepcopy(encoded_arcs)
        bin_info["arc_distance_km"] = dict(arc_distance)
        bin_info["arc_soc_consumption"] = dict(arc_soc)
        bin_info["complete_paths"] = copy.deepcopy(encoded_paths)
    return result


def _solve_kwargs(
    *,
    params,
    network,
    state: RollingState,
    plan_records: Mapping[UserKey, Mapping[str, Any]],
    signals: RLSignals,
    current_rollouts: Mapping[UserKey, ReservationEventRollout] | None = None,
):
    grid = _grid(params)
    return {
        "params": params,
        "network": network,
        "state": state,
        "plan_records": plan_records,
        "visible_entries": [],
        "current_rollouts": current_rollouts or {},
        "terminal_users": set(),
        "forecast_random": [],
        "signals": signals,
        "engine": _engine(params, grid),
        "grid": grid,
        "period": 0,
        "horizon": signals.horizon,
        "max_patterns_per_station": 1_000,
        "output_flag": 0,
    }


@unittest.skipUnless(
    GUROBI_LICENSE_AVAILABLE,
    "requires a usable Gurobi license; unavailable-solver behavior is tested below",
)
class PaperMPCJointOptimizationTest(unittest.TestCase):
    def test_two_users_jointly_split_shared_inventory_and_pay_one_adjustment(self) -> None:
        params = _params(horizon=4)
        reference_path = (
            (ENTRY_NODE, 0),
            (0, 2),
            (2, EXIT_NODE),
        )
        alternate_path = (
            (ENTRY_NODE, 1),
            (1, EXIT_NODE),
        )
        network = _restrict_od_zero_network(
            generate_candidate_network(params),
            params,
            (reference_path, alternate_path),
        )
        users = ((0, 100), (0, 101))
        plan_records = {
            key: _plan_record(key, reference_path, entry_soc=0.7) for key in users
        }
        # One immediately available battery at each relevant station.  If both
        # users independently keep station 0, one times out there; the joint
        # optimum sends exactly one user through station 1 instead.
        state = _state(
            params,
            full_slots=((0, 0), (1, 0), (2, 0)),
        )
        result = solve_paper_mpc(
            **_solve_kwargs(
                params=params,
                network=network,
                state=state,
                plan_records=plan_records,
                signals=_signals(params, 4),
            )
        )

        self.assertEqual(result.status, "OPTIMAL")
        self.assertTrue(result.is_optimal)
        self.assertTrue(result.has_incumbent)
        self.assertEqual(
            sorted(station_sequence(path) for path in result.selected_paths.values()),
            [(0, 2), (1,)],
        )
        self.assertEqual(sum(result.path_adjusted.values()), 1)
        self.assertAlmostEqual(
            result.adjustment_cost,
            params.path_adjustment_penalty,
        )
        self.assertAlmostEqual(result.reservation_failure_cost, 0.0)
        self.assertEqual(
            sum(len(pattern.served_ids) for pattern in result.selected_station_patterns.values()),
            3,
        )
        self.assertTrue(
            all(not pattern.failed_ids for pattern in result.selected_station_patterns.values())
        )
        stats = result.model_statistics
        self.assertEqual(stats["model_kind"], "paper_continuous_event_station_pattern_milp")
        self.assertTrue(stats["station_pattern_space_complete"])
        self.assertTrue(stats["path_candidate_space_complete"])
        self.assertTrue(stats["fixed_mock_power"])
        self.assertTrue(stats["global_milp_optimality_claimed"])
        self.assertGreater(stats["paper_equation_constraint_counts"]["eq:flow"], 0)
        self.assertGreater(
            stats["paper_equation_constraint_counts"]["eq:path_adjustment_indicator"],
            0,
        )


@unittest.skipUnless(
    GUROBI_LICENSE_AVAILABLE,
    "requires a usable Gurobi license; unavailable-solver behavior is tested below",
)
class PaperMPCFixedPowerOutcomeTest(unittest.TestCase):
    WAITING_ID = "paper-test-carried-reservation"
    USER_KEY = (0, 909)

    def _waiting_case(self, station_zero_power_kw: float):
        params = _params(horizon=4)
        path = (
            (ENTRY_NODE, 0),
            (0, 2),
            (2, EXIT_NODE),
        )
        network = _restrict_od_zero_network(
            generate_candidate_network(params), params, (path,)
        )
        waiting = WaitingRequest(
            request_id=self.WAITING_ID,
            event_id=self.WAITING_ID,
            kind=RequestKind.RESERVATION,
            station=0,
            arrival_time=0.0,
            deadline=0.25,
            return_soc=0.2,
            user_key=self.USER_KEY,
            source_arc=(ENTRY_NODE, 0),
            path_order=0,
        )
        state = _state(params, station_zero_soc=0.9, full_slots=((2, 0),))
        state.waiting_queues = {
            0: {
                RequestKind.RESERVATION: [waiting],
                RequestKind.RANDOM: [],
            }
        }
        state.request_status[self.WAITING_ID] = PhysicalRequestStatus.WAITING
        state.enroute[f"{self.USER_KEY[0]}:{self.USER_KEY[1]}"] = EnrouteReservation(
            user_key=self.USER_KEY,
            current_position=params.station.positions_km[0],
            vehicle_soc=0.2,
            dayahead_initial_path=list(path),
            last_published_remaining_path=list(path),
            waiting_request_id=self.WAITING_ID,
        )
        carried_candidate = CandidateRequest(
            request_id=self.WAITING_ID,
            event_id=self.WAITING_ID,
            kind=RequestKind.RESERVATION,
            station=0,
            arrival_time=0.0,
            deadline=0.25,
            return_soc=0.2,
            user_key=self.USER_KEY,
            source_arc=(ENTRY_NODE, 0),
            path_order=0,
        )
        rollout = ReservationEventRollout(
            reservation_id=str(self.USER_KEY[1]),
            od_id=self.USER_KEY[0],
            user_key=self.USER_KEY,
            entry=VisibleEntry(0.0, 0.2, "visible_override"),
            path_arcs=path,
            events=(carried_candidate,),
        )
        result = solve_paper_mpc(
            **_solve_kwargs(
                params=params,
                network=network,
                state=state,
                plan_records={
                    self.USER_KEY: _plan_record(self.USER_KEY, path, entry_soc=0.7)
                },
                signals=_signals(
                    params,
                    4,
                    station_zero_power_kw=station_zero_power_kw,
                ),
                current_rollouts={self.USER_KEY: rollout},
            )
        )
        return params, result

    def test_fixed_60_kw_creates_positive_wait_then_service(self) -> None:
        _, result = self._waiting_case(60.0)
        self.assertEqual(result.request_outcomes[self.WAITING_ID], "served_in_horizon")
        service_times = dict(result.selected_station_patterns[0].service_times)
        expected_wait = (1.0 - 0.9) * 100.0 / (0.95 * 60.0)
        self.assertAlmostEqual(service_times[self.WAITING_ID], expected_wait, places=10)
        self.assertGreater(service_times[self.WAITING_ID], 0.0)
        self.assertAlmostEqual(result.reservation_failure_cost, 0.0)
        self.assertTrue(result.model_statistics["fixed_mock_power"])

    def test_fixed_zero_kw_causes_one_failure_and_deactivates_downstream(self) -> None:
        params, result = self._waiting_case(0.0)
        self.assertEqual(result.request_outcomes[self.WAITING_ID], "failed_in_horizon")
        self.assertIn(self.WAITING_ID, result.selected_station_patterns[0].failed_ids)
        downstream = [
            identifier
            for identifier in result.request_outcomes
            if identifier != self.WAITING_ID
        ]
        self.assertGreaterEqual(len(downstream), 1)
        self.assertTrue(
            all(result.request_outcomes[identifier] == "inactive" for identifier in downstream)
        )
        self.assertAlmostEqual(
            result.reservation_failure_cost,
            params.reservation_failure_penalty,
        )
        self.assertEqual(
            sum(len(pattern.failed_ids) for pattern in result.selected_station_patterns.values()),
            1,
        )


class PaperMPCSolverAvailabilityTest(unittest.TestCase):
    @staticmethod
    def _empty_case_kwargs():
        params = _params(horizon=1)
        network = generate_candidate_network(params)
        state = _state(params)
        return _solve_kwargs(
            params=params,
            network=network,
            state=state,
            plan_records={},
            signals=_signals(params, 1),
        )

    def test_missing_gurobi_has_no_replay_or_greedy_fallback(self) -> None:
        with patch.object(paper_mpc, "gp", None), patch.object(paper_mpc, "GRB", None):
            with self.assertRaisesRegex(
                PaperMPCSolverUnavailable,
                "(?i)gurobi.*(not installed|no fallback)",
            ):
                solve_paper_mpc(**self._empty_case_kwargs())

    @unittest.skipIf(
        paper_mpc.gp is None or GUROBI_LICENSE_AVAILABLE,
        "only applies when gurobipy imports but the local license cannot create a model",
    )
    def test_unusable_local_license_is_wrapped_as_solver_unavailable(self) -> None:
        with self.assertRaisesRegex(
            PaperMPCSolverUnavailable,
            "(?i)Gurobi.*(OS user|environment|license)",
        ):
            solve_paper_mpc(**self._empty_case_kwargs())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
