"""RL-specific independent transition, training, settlement and resume checks."""
import bootstrap
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from learned_value import LearnedValue, make_contract, digest
from state_encoding import StateSpec, Normalizer
from training import SetValueNetwork, fit, monte_carlo_states
from protocol import split_days, read, scenario_for, atomic, source_hashes, file_hash, build_plan
from src.candidate_network import generate_candidate_network
from src.domain import (Reservation, RollingState, WaitingRequest, initial_state, MPCSolution,
                        ServiceDecision)
from src.execution import advance_to_boundary, execute_step
from src.forecast import build_forecast
from src.mpc_model import solve_mpc
from src.parameters import slots_at
from src.request_builder import build_window
from src.rolling_runner import run_rolling_mpc
from src.result_statistics import build_result_statistics
from src.scenario import SyntheticScenario

# The tests use existing public toy fixture definitions, with isolated src imported first.
PROJECT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT))
from tests.test_terminal_execution import parameters, scenario
from tests.test_real_data_experiment import parameters as ragged_parameters


def network_model(p, states=None):
    spec = StateSpec(p)
    if states is None:
        states = [spec.raw_state(RollingState(0, copy.deepcopy(p.station.initial_slot_soc)))]
    torch.manual_seed(1)
    net = SetValueNetwork(spec)
    return LearnedValue({"schema_version": 2, "kind": "learned_set_relu", "variant": "learned_set",
        "state_schema": spec.schema(), "contract": make_contract(p, [p.random_hourly_means]),
        "normalizer": Normalizer.fit(states, spec).record,
        "weights": {k:v.detach().numpy().astype(float).tolist() for k,v in net.state_dict().items()}})


class LearnedTransitionTests(unittest.TestCase):
    def params(self):
        p = parameters()
        p.finish_pending_after_demand = True
        p.random_hourly_means = [[0., 0.], [0., 0.]]
        p.solver.time_limit_sec = 5.
        p.solver.mip_gap = 0.
        return p

    def compare(self, p, truth, state, horizon=1):
        road = generate_candidate_network(p)
        now = state.period*p.interval_hours
        state = advance_to_boundary(p, state, [r for r in truth.observation_at(now).random_history
                                             if r["arrival_time"] == now], scenario=truth).state
        window = build_window(p, state, road, build_forecast(p, truth.observation_at(now), state.period, horizon), horizon=horizon)
        for network in window.networks.values():
            network.can_update = False
        spec = StateSpec(p)
        value = network_model(p, [spec.raw_state(state)])
        solution = solve_mpc(p, window, terminal_model=value, feature_spec=spec)
        candidates = {r.request_id:r for r in window.requests}
        realised = state
        for offset in range(horizon):
            period = state.period+offset
            actions = []
            for action in solution.services:
                if action.period != period:
                    continue
                rid = action.request_id
                if rid not in realised.waiting:
                    req = candidates[rid]
                    matches = [w for w in realised.waiting.values() if w.station == req.station and w.kind == req.kind and w.user_key == req.user_key]
                    self.assertEqual(len(matches), 1)
                    rid = matches[0].request_id
                actions.append(ServiceDecision(rid, action.station, action.slot, period))
            step = MPCSolution("optimal", 0., {}, solution.paths if offset == 0 else {}, actions,
                [[[solution.power[i][b][offset]] for b in range(slots_at(p,i))] for i in p.station.station_ids], [])
            left, right = period*p.interval_hours, (period+1)*p.interval_hours
            realised = execute_step(p, realised, step, truth.arrivals_between(left, right), scenario=truth).state
            realised = advance_to_boundary(p, realised, [r for r in truth.observation_at(right).random_history
                                                       if r["arrival_time"] == right], scenario=truth).state
        raw = spec.raw_state(realised)
        expected = np.r_[value.encode(raw), raw["continuation"]]
        diff = np.abs(expected-np.asarray(solution.terminal_features))
        bad = [(name, float(d)) for name,d in zip(value.feature_names,diff) if d > 2e-6]
        self.assertFalse(bad, f"MILP differs from independent actual execution: {bad[:12]}")
        self.assertAlmostEqual(value.predict(raw), solution.terminal_value, delta=.001)
        return solution, realised

    def test_observed_speed_same_and_next_segment(self):
        for horizon in (1,2):
            with self.subTest(horizon=horizon):
                p = self.params()
                truth = scenario(p, actual_soc=.8, report_soc=.8, multipliers=[1.1,1.,1.])
                user = Reservation((0,0), 0., .8, 5., .75, True, day_ahead=[1], retained_plan=[1],
                    published_plan=[1], actual_entry_time=0., observation_time=0.,
                    observed_segment_index=0, observed_speed_kmh=60/1.1)
                self.compare(p, truth, RollingState(0, [[1.],[1.]], {"0:0":user}), horizon)

    def test_future_entry_publication_and_unknown_later_time(self):
        p = self.params()
        truth = scenario(p, actual_time=.025, report_time=.025, actual_soc=.8, report_soc=.8, multipliers=[1.,1.,1.])
        state = initial_state(p, truth.initial_reservations(), {"0:0":[0,1]})
        state.users["0:0"].day_ahead = [1]
        self.compare(p, truth, state)
        rows = StateSpec(p).raw_state(state)["users"][0]
        self.assertEqual(rows[1][6:10], [0.,0.,0.,0.])

    def test_predecessor_swap_and_chain(self):
        p = self.params()
        truth = scenario(p, actual_soc=.8, report_soc=.8, multipliers=[1.,1.,1.])
        waiting = WaitingRequest("first", 0, "reservation", 0., .25, .2, (0,0))
        user = Reservation((0,0), 0., .8, 10., .2, True, day_ahead=[0,1], retained_plan=[1],
            published_plan=[1], waiting_request_id="first", last_swap_position_km=0., actual_entry_time=0.)
        self.compare(p, truth, RollingState(0, [[1.],[1.]], {"0:0":user}, {"first":waiting}))

    def test_exact_deadline_retained(self):
        p = self.params()
        truth = scenario(p, actual_soc=.8, report_soc=.8, multipliers=[1.,1.,1.])
        waiting = WaitingRequest("first",0,"reservation",0.,1/12,.2,(0,0))
        user = Reservation((0,0),0.,.8,10.,.2,True,day_ahead=[0,1],retained_plan=[1],
                           published_plan=[1],waiting_request_id="first")
        _, end = self.compare(p,truth,RollingState(0,[[.2],[1.]],{"0:0":user},{"first":waiting}))
        self.assertIn("first", end.waiting)

    def test_random_inside_and_at_terminal_boundary(self):
        for rate, arrival in ((10.,.05),(6.,1/12)):
            with self.subTest(rate=rate):
                p=self.params()
                p.random_hourly_means=[[rate,0.],[0.,0.]]
                truth=SyntheticScenario(p,[],[dict(request_id="actual-r",station=0,arrival_time=arrival,return_soc=.2)])
                _, end=self.compare(p,truth,RollingState(0,[[1.],[1.]]))
                self.assertIn("actual-r",end.waiting)

    def test_cross_demand_boundary_value_and_absorption(self):
        p=ragged_parameters()
        p.terminal_experiment=True
        p.random_hourly_means=[[0.],[0.]]
        p.reservation_hourly_weights=[1.]
        p.od_sampling_weights=[1.]
        p.validate()
        truth=SyntheticScenario(p,[],[])
        req=WaitingRequest("late",0,"random",.9,1.4,.2)
        # Empty batteries force this request to remain pending at the day boundary.
        _,end=self.compare(p,truth,RollingState(3,[[.2],[1.,1.]],waiting={"late":req}),horizon=1)
        self.assertEqual(StateSpec(p).raw_state(end)["continuation"],1.)
        self.compare(p,truth,RollingState(4,[[1.],[1.,1.]]),horizon=2)


class TrainingProtocolTests(unittest.TestCase):
    def test_penalty_override_preserves_frozen_scene_and_rejects_old_model(self):
        dataset=PROJECT/"data/final_real_data_v1"
        source=read(dataset/"scenarios/day_43.json")
        original_hash=file_hash(dataset/"scenarios/day_43.json")
        p,_,_=scenario_for(dataset,43)
        old_model=network_model(p)
        for penalty in (100.,200.):
            with self.subTest(penalty=penalty):
                changed,scene,_=scenario_for(dataset,43,failure_penalty=penalty)
                expected=copy.deepcopy(source)
                expected["params"]["reservation_failure_penalty"]=penalty
                self.assertEqual(scene.to_dict(),expected)
                self.assertEqual(changed.reservation_failure_penalty,penalty)
                self.assertEqual(file_hash(dataset/"scenarios/day_43.json"),original_hash)
                with self.assertRaisesRegex(ValueError,"physical configuration"):
                    old_model.validate_params(changed)
                plan=build_plan(dataset,"test-code",failure_penalty=penalty)
                self.assertEqual(plan["business_overrides"]["reservation_failure_penalty"],penalty)
                self.assertEqual(plan["contract"]["physical_hash"],make_contract(changed,[changed.random_hourly_means])["physical_hash"])

    def test_changed_penalty_reaches_actual_ledger_and_mc_return_once(self):
        for penalty in (100.,200.):
            with self.subTest(penalty=penalty):
                p=parameters()
                p.finish_pending_after_demand=True
                p.horizon=6
                p.reservation_failure_penalty=penalty
                p.random_hourly_means=[[0.,0.],[0.,0.]]
                p.station.initial_slot_soc=[[.2],[.2]]
                p.station.slot_power_limits_kw=[[0.],[0.]]
                p.station.station_power_limits_kw=[0.,0.]
                truth=scenario(p,actual_soc=.55,report_soc=.55,multipliers=[1.,1.,1.])
                result=run_rolling_mpc(p,truth,feature_spec=StateSpec(p))
                build_result_statistics(result)
                self.assertEqual(result["summary"]["reservation_failures"],1)
                self.assertEqual(result["summary"]["reservation_failure_cost"],penalty)
                _,returns=monte_carlo_states(result)
                self.assertAlmostEqual(returns[0],result["summary"]["total_reward"])
                self.assertGreaterEqual(sum(r["solution"]["objective_terms"]["failure_cost"] for r in result["rounds"]),penalty)

    def test_cuda_joint_updates_export_and_frozen_normalizer(self):
        p=parameters()
        p.finish_pending_after_demand=True
        truth=scenario(p,actual_soc=.8,report_soc=.8,multipliers=[1.,1.,1.])
        state=initial_state(p,truth.initial_reservations(),{"0:0":[0,1]})
        spec=StateSpec(p)
        rng=np.random.default_rng(1)
        states=[]
        for i in range(40):
            s=state.clone()
            s.period=i%20
            s.slot_soc=rng.uniform(.1,1.,(2,1)).tolist()
            s.users["0:0"].entry_time=1.8
            s.users["0:0"].entry_soc=float(rng.uniform(.55,.95))
            s.waiting={"r":WaitingRequest("r",i%2,"random",s.period*p.interval_hours-.04,
                                            s.period*p.interval_hours+.2,float(rng.uniform(.1,.3)))}
            states.append(spec.raw_state(s))
        contract=make_contract(p,[p.random_hourly_means])
        targets=np.linspace(100,1000,len(states))
        model,diag=fit(states,targets,spec,contract,epochs=3,batch_size=16,device="cuda")
        self.assertTrue(all(v>0 for v in diag["parameter_change_l2_by_layer"].values()))
        second,d2=fit(states,targets,spec,contract,previous=model,iteration=1,epochs=2,batch_size=16,device="cuda")
        self.assertEqual(digest(model.normalizer.record),digest(second.normalizer.record))
        self.assertTrue(d2["warm_started"])
        model.validate_params(p)
        altered=copy.deepcopy(p)
        altered.battery_capacity_kwh=99.
        with self.assertRaisesRegex(ValueError,"physical"):
            model.validate_params(altered)

    def test_complete_mc_and_resume_include_drain(self):
        p=ragged_parameters()
        scene=SyntheticScenario(p,[],[dict(request_id="late",station=0,arrival_time=.9,return_soc=.2)])
        with tempfile.TemporaryDirectory() as temp:
            def interrupt(row):
                if row["period"] == 3:
                    raise RuntimeError("test interruption after durable boundary")
            with self.assertRaisesRegex(RuntimeError,"test interruption"):
                run_rolling_mpc(p,scene,feature_spec=StateSpec(p),journal_dir=temp,progress=interrupt)
            resumed=[]
            result=run_rolling_mpc(p,scene,feature_spec=StateSpec(p),journal_dir=temp,
                                   progress=lambda row:resumed.append(row["period"]))
            self.assertEqual(resumed,[4])
            build_result_statistics(result)
            states,returns=monte_carlo_states(result)
            self.assertEqual(len(states),6)
            self.assertEqual(returns[-1],0.)
            self.assertGreater(returns[4],0.)
            self.assertEqual(returns[0],result["summary"]["total_reward"])
            self.assertEqual(states[-1]["continuation"],0.)

    def test_fixed_split_and_compatible_weekday_contract(self):
        dataset=PROJECT/"data/final_real_data_v1"
        splits=split_days(read(dataset/"manifest.json"))
        self.assertEqual(splits["test"],[1,37,3,4,19,13,14])
        self.assertEqual(splits["validation"],[57,65,31,67,47,34,42])
        self.assertEqual(len(sum(splits["training_batches"],[])),56)
        for batch in splits["training_batches"]:
            self.assertEqual([sum((d-1)%7==w for d in batch) for w in range(7)],[2]*7)
        p,_,_=scenario_for(dataset,43)
        q,_,_=scenario_for(dataset,58)
        model=network_model(p)
        model.record["contract"]=make_contract(p,[p.random_hourly_means,q.random_hourly_means])
        model.validate_params(q)

    def test_controller_runs_exactly_91_days_and_selects_only_validation(self):
        import run
        dataset=PROJECT/"data/final_real_data_v1"
        splits=split_days(read(dataset/"manifest.json"))
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/"output"
            baseline=Path(temp)/"baseline"
            atomic(baseline/"status.json",{"state":"complete"})
            plan={"code_hash":digest(source_hashes(bootstrap.ROOT)),"dataset_manifest_hash":digest(read(dataset/"manifest.json")),
                  "split":splits,"selection":"validation only"}
            atomic(output/"plan.json",plan)
            atomic(output/"prepared.json",{"source":str(bootstrap.ROOT)})
            calls=[]
            class Child:
                pid=123
                def __init__(self,command,**unused):
                    self.command=command
                    calls.append(command)
                def wait(self):
                    command=self.command
                    job=Path(command[command.index("--job")+1])
                    iteration=int(command[command.index("--iteration")+1])
                    filename="model.json" if command[3]=="fit" else "metrics.json"
                    atomic(job/filename,{"iteration":iteration,"net_profit_yuan":float(iteration)})
                    atomic(job/"status.json",{"state":"complete","artifacts":{filename:file_hash(job/filename)}})
                    return 0
            args=SimpleNamespace(output=output,dataset=dataset,baseline=baseline,resume=False,failure_penalty=None)
            with patch.object(run.subprocess,"Popen",Child):
                run.run(args)
            days=[c for c in calls if c[3]=="rollout"]
            self.assertEqual((len(calls),len(days)),(95,91))
            tests=[c for c in days if "/test/" in c[c.index("--job")+1].replace("\\","/")]
            self.assertEqual([int(c[c.index("--day")+1]) for c in tests],splits["test"])
            self.assertEqual(read(output/"selection.json")["selected"]["iteration"],4)
            self.assertTrue(read(output/"status.json")["report_pending_user_request"])
            self.assertFalse((output/"test_summary.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
