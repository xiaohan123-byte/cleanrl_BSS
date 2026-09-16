"""Check terminal decisions, exact ReLU graphs and canonical state encoding."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.domain import CandidateRequest, MPCWindow, RollingState, UserNetwork, Reservation, WaitingRequest
from src.parameters import BusinessParameters, StationParameters, ODPairParameters, SolverParameters
from src.terminal_features import FeatureSpec
from src.terminal_value import TerminalValueModel
from src.mpc_model import solve_mpc
from src.candidate_network import generate_candidate_network
from src.forecast import Forecast, forecast_motion
from src.request_builder import build_window


def params(stations=2, slots=1):
    p = BusinessParameters(num_periods=24,horizon=2,interval_hours=1/12,
        station=StationParameters(num_stations=stations,station_ids=list(range(stations)),
            positions_km=[10.*(i+1) for i in range(stations)],num_slots=slots,
            initial_slot_soc=[[1.]*slots for _ in range(stations)],charging_efficiency=.9,
            slot_power_limits_kw=[[80.]*slots for _ in range(stations)],
            station_power_limits_kw=[80.*slots]*stations),
        od_pairs=[ODPairParameters(0,0.,10.*(stations+1),list(range(stations)))],
        vehicle_speed_kmh=60.,range_km=100.,battery_capacity_kwh=60.,
        min_swap_spacing_km=0.,num_reservations=1,
        electricity_price=[[.5]*24 for _ in range(stations)],
        swap_service_price=[[1.2]*24 for _ in range(stations)],random_arrival_rate_per_hour=[0.]*stations,
        solver=SolverParameters(threads=1,time_limit_sec=10.,mip_gap=0.,feasibility_tol=1e-9))
    p.validate();return p


def linear(spec, weights=None, bias=0., scale=1.):
    return TerminalValueModel(kind="linear",feature_names=spec.names,variant=spec.variant,
        linear_weights=np.zeros(spec.dimension) if weights is None else weights,bias=bias,output_scale=scale)


class TerminalValueTests(unittest.TestCase):
    def test_inventory_objective_changes_charging(self):
        p=params(stations=1);spec=FeatureSpec(p,"inventory_only")
        weights=np.zeros(spec.dimension);weights[0]=100.
        result=solve_mpc(p,MPCWindow(0,1,RollingState(0,[[.2]]),{},[]),terminal_model=linear(spec,weights),feature_spec=spec)
        self.assertAlmostEqual(result.power[0][0][0],80.)
        self.assertAlmostEqual(result.terminal_features[0],.3)
        self.assertAlmostEqual(result.terminal_value,30.)
        self.assertAlmostEqual(result.objective,30.-80./12*.5)

    def test_relu_exact_graph_for_both_output_signs(self):
        p=params(stations=1);spec=FeatureSpec(p,"inventory_only")
        weights=np.zeros((1,spec.dimension));weights[0,0]=1.
        net=TerminalValueModel("relu",spec.names,spec.variant,np.zeros(spec.dimension),0.,weights,[-.25],[1.],100.)
        result=solve_mpc(p,MPCWindow(0,1,RollingState(0,[[.2]]),{},[]),terminal_model=net,feature_spec=spec)
        self.assertAlmostEqual(result.power[0][0][0],80.)
        self.assertAlmostEqual(result.terminal_value,5.)
        net.output_weights[:]=-1.
        result=solve_mpc(p,MPCWindow(0,1,RollingState(0,[[.8]]),{},[]),terminal_model=net,feature_spec=spec)
        self.assertAlmostEqual(result.power[0][0][0],0.)
        self.assertAlmostEqual(result.terminal_value,-55.)

    def test_final_period_overrides_positive_terminal_network(self):
        p=params(stations=1);spec=FeatureSpec(p,"inventory_only")
        weights=np.zeros(spec.dimension);weights[0]=100000.
        result=solve_mpc(p,MPCWindow(23,1,RollingState(23,[[.2]]),{},[]),terminal_model=linear(spec,weights),feature_spec=spec)
        self.assertEqual(result.terminal_value,0.)
        self.assertAlmostEqual(result.power[0][0][0],0.)
        self.assertEqual(result.terminal_features,[])

    def test_full_chain_value_selects_path(self):
        p=params();p.path_adjustment_penalty=0.
        user=Reservation((0,0),0.,1.,0.,1.,entered=True,day_ahead=[0],retained_plan=[0],published_plan=[0])
        state=RollingState(0,[[1.],[1.]],users={"0:0":user})
        network=UserNetwork((0,0),"origin",[("origin",0),(0,"exit"),("origin",1),(1,"exit")],(0,),(0,),True)
        requests=[CandidateRequest("a",0,"reservation",.9,(0,0),("origin",0),arrival_time=1/6),
                  CandidateRequest("b",1,"reservation",.8,(0,0),("origin",1),arrival_time=1/3)]
        spec=FeatureSpec(p);weights=np.zeros(spec.dimension)
        weights[spec.index["chain:1:1:time:0:count"]]=1000.
        result=solve_mpc(p,MPCWindow(0,1,state,{"0:0":network},requests),terminal_model=linear(spec,weights),feature_spec=spec)
        self.assertEqual(result.paths["0:0"],[1])
        self.assertAlmostEqual(result.terminal_value,10.)

    def test_predicted_in_transit_features_equal_same_physical_state(self):
        p=params()
        user=Reservation((0,0),0.,.8,0.,.8,entered=True,day_ahead=[0,1],retained_plan=[0,1],published_plan=[0,1])
        state=RollingState(0,[[1.],[1.]],users={"0:0":user});network=generate_candidate_network(p)
        w=build_window(p,state,network,Forecast([],0.,1/12),horizon=1);w.networks["0:0"].can_update=False
        spec=FeatureSpec(p);result=solve_mpc(p,w,terminal_model=linear(spec),feature_spec=spec)
        predicted=state.clone();predicted.period=1
        predicted.slot_soc=[[row[-1] for row in station] for station in result.soc]
        motion=forecast_motion(p,user,0,1/12,now=0.)
        for name in ("position_km","soc","entered","last_swap_position_km"):
            setattr(predicted.users["0:0"],name,motion[name])
        next_window=build_window(p,predicted,network,Forecast([],1/12,2/12),horizon=1)
        np.testing.assert_allclose(result.terminal_features,spec.encode_window(next_window),atol=1e-9)

    def test_predecessor_service_updates_chain_and_waiting_clock(self):
        p=params();user=Reservation((0,0),0.,.8,10.,.2,entered=True,day_ahead=[0,1],retained_plan=[1],
                         published_plan=[1],waiting_request_id="first",last_swap_position_km=0.)
        wait=WaitingRequest("first",0,"reservation",0.,.25,.2,(0,0))
        state=RollingState(0,[[1.],[1.]],users={"0:0":user},waiting={"first":wait});network=generate_candidate_network(p)
        w=build_window(p,state,network,Forecast([],0.,1/6),horizon=2);w.networks["0:0"].can_update=False
        spec=FeatureSpec(p);result=solve_mpc(p,w,terminal_model=linear(spec),feature_spec=spec)
        first=[s for s in result.services if s.request_id=="first"]
        self.assertEqual([(s.request_id,s.period) for s in first],[("first",0)])
        req=next(r for r in w.requests if r.predecessors)
        predicted=state.clone();predicted.period=2
        predicted.slot_soc=[[row[-1] for row in station] for station in result.soc]
        predicted.waiting={req.request_id:WaitingRequest(req.request_id,1,"reservation",1/6,1/6+.25,.9,(0,0))}
        vehicle=predicted.users["0:0"];vehicle.position_km=20.;vehicle.soc=.9;vehicle.last_swap_position_km=10.
        vehicle.completed_stations=[0];vehicle.waiting_request_id=req.request_id;vehicle.retained_plan=[];vehicle.published_plan=[]
        next_window=build_window(p,predicted,network,Forecast([],1/6,1/4),horizon=1)
        expected=spec.encode_window(next_window)
        np.testing.assert_allclose(result.terminal_features,expected,atol=1e-9)
        self.assertAlmostEqual(expected[spec.index["chain:1:1:time:5:count"]],.01)

    def test_baseline_charging_uses_post_swap_soc_and_fixed_station_share(self):
        p=params(stations=1,slots=2);p.station.station_power_limits_kw=[80.]
        result=solve_mpc(p,MPCWindow(0,2,RollingState(0,[[.2,.95]]),{},[]),charging_mode="baseline")
        self.assertAlmostEqual(result.power[0][0][0],40.)
        self.assertAlmostEqual(result.power[0][1][0],40.)
        self.assertAlmostEqual(result.power[0][1][1],0.)
        wait=WaitingRequest("r",0,"random",0.,.25,.1)
        w=MPCWindow(0,1,RollingState(0,[[1.,1.]],waiting={"r":wait}),{},
                    [CandidateRequest("r",0,"random",.1,arrival_time=0.,observed=True,deadline=.25)])
        result=solve_mpc(p,w,charging_mode="baseline");self.assertEqual(len(result.services),1)
        swapped=result.services[0].slot
        self.assertAlmostEqual(result.power[0][swapped][0],40.)
        self.assertAlmostEqual(result.power[0][1-swapped][0],0.)

    def test_reverse_direction_terminal_features_match_physical_state(self):
        p=params();p.od_pairs=[ODPairParameters(0,30.,0.,[1,0])];p.validate()
        user=Reservation((0,0),0.,.8,30.,.8,entered=True,day_ahead=[1,0],retained_plan=[1,0],
                         published_plan=[1,0],last_swap_position_km=30.)
        state=RollingState(0,[[1.],[1.]],users={"0:0":user});network=generate_candidate_network(p)
        w=build_window(p,state,network,Forecast([],0.,1/12),horizon=1);w.networks["0:0"].can_update=False
        spec=FeatureSpec(p);result=solve_mpc(p,w,terminal_model=linear(spec),feature_spec=spec)
        self.assertEqual(result.paths["0:0"],[1,0])
        predicted=state.clone();predicted.period=1
        predicted.slot_soc=[[row[-1] for row in station] for station in result.soc]
        motion=forecast_motion(p,user,1,1/12,now=0.)
        self.assertAlmostEqual(motion["position_km"],25.)
        for name in ("position_km","soc","entered","last_swap_position_km"):
            setattr(predicted.users["0:0"],name,motion[name])
        next_window=build_window(p,predicted,network,Forecast([],1/12,2/12),horizon=1)
        np.testing.assert_allclose(result.terminal_features,spec.encode_window(next_window),atol=1e-9)

    def test_failed_user_removes_entire_terminal_chain(self):
        p=params();p.station.slot_power_limits_kw=[[0.],[0.]];p.station.station_power_limits_kw=[0.,0.]
        user=Reservation((0,0),0.,.8,10.,.2,entered=True,day_ahead=[0,1],retained_plan=[1],
                         published_plan=[1],waiting_request_id="first")
        wait=WaitingRequest("first",0,"reservation",0.,.1,.2,(0,0))
        state=RollingState(0,[[0.],[1.]],users={"0:0":user},waiting={"first":wait})
        network=generate_candidate_network(p)
        w=build_window(p,state,network,Forecast([],0.,5/12),horizon=5);w.networks["0:0"].can_update=False
        spec=FeatureSpec(p);result=solve_mpc(p,w,terminal_model=linear(spec),feature_spec=spec)
        self.assertTrue(result.request_outcomes["first"]["failed"])
        self.assertFalse(result.services)
        for index,name in enumerate(spec.names):
            if name.startswith("chain:"):
                self.assertAlmostEqual(result.terminal_features[index],0.)

    def test_new_published_path_is_not_counted_as_unpublished_at_terminal(self):
        p=params();p.path_adjustment_penalty=0.
        user=Reservation((0,0),.02,1.,0.,1.,entered=False,day_ahead=[0],retained_plan=[0],published_plan=None)
        state=RollingState(0,[[1.],[1.]],users={"0:0":user});network=generate_candidate_network(p)
        w=build_window(p,state,network,Forecast([],0.,1/12),horizon=1)
        spec=FeatureSpec(p);weights=np.zeros(spec.dimension)
        weights[spec.index["chain:1:1:time:1:count"]]=1000.
        result=solve_mpc(p,w,terminal_model=linear(spec,weights),feature_spec=spec)
        self.assertEqual(result.paths["0:0"],[1])
        for index,name in enumerate(spec.names):
            if "published:difference" in name or name.endswith("published:changed_count") or name.endswith("not_entered"):
                self.assertAlmostEqual(result.terminal_features[index],0.)
        self.assertAlmostEqual(result.terminal_features[spec.index["chain:1:1:dayahead:changed_count"]],.01)
        self.assertAlmostEqual(result.terminal_features[spec.index["chain:1:1:dayahead:difference:0"]],-.01)
        self.assertAlmostEqual(result.terminal_features[spec.index["chain:1:1:dayahead:difference:1"]],.01)

    def test_configuration_binding_allows_horizon_seed_but_rejects_changed_demand(self):
        from src.terminal_value import terminal_configuration_fingerprint
        p=params(stations=1);spec=FeatureSpec(p,"inventory_only");net=linear(spec)
        net.configuration_fingerprint=terminal_configuration_fingerprint(p)
        original=net.configuration_fingerprint
        p.horizon=1;p.seed+=1;p.solver.time_limit_sec=3.
        self.assertEqual(terminal_configuration_fingerprint(p),original)
        solve_mpc(p,MPCWindow(0,1,RollingState(0,[[.2]]),{},[]),terminal_model=net,feature_spec=spec)
        p.num_reservations+=1
        with self.assertRaisesRegex(ValueError,"different physical/demand configuration"):
            solve_mpc(p,MPCWindow(0,1,RollingState(0,[[.2]]),{},[]),terminal_model=net,feature_spec=spec)
        restored=TerminalValueModel.from_dict(net.to_dict())
        self.assertEqual(restored.configuration_fingerprint,original)
        old=net.to_dict();old.pop("configuration_fingerprint")
        self.assertIsNone(TerminalValueModel.from_dict(old).configuration_fingerprint)

    def test_serialization_roundtrip(self):
        p=params();spec=FeatureSpec(p,"inventory_only");rng=np.random.default_rng(7)
        net=TerminalValueModel("relu",spec.names,spec.variant,rng.normal(size=spec.dimension),.7,
                               rng.normal(size=(2,spec.dimension)),[.2,-.3],[.5,-.8],1000.)
        inputs=rng.normal(size=(3,spec.dimension))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"model.json";net.save(path)
            np.testing.assert_allclose(net.predict(inputs),TerminalValueModel.load(path).predict(inputs))

if __name__=="__main__":unittest.main()
