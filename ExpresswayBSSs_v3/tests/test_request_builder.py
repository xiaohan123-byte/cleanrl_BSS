import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from src.domain import Reservation, RollingState, WaitingRequest, UserNetwork
from src.request_builder import build_window

class RequestBuilderTest(unittest.TestCase):
    def test_waiting_request_identity_and_downstream_dependencies(self):
        params=NS(interval_hours=1/12,horizon=6,num_periods=20,
                  station=NS(positions_km=[0,10,20]),range_km=100,
                  vehicle_speed_kmh=60,max_wait_hours=.25)
        waiting=WaitingRequest('actual-stable-id',0,'reservation',.01,.26,.3,(0,0))
        user=Reservation((0,0),0,.8,0,.3,True,
                         waiting_request_id=waiting.request_id,retained_plan=[1,2])
        state=RollingState(1,[[1],[1],[1]],{'0:0':user},{waiting.request_id:waiting})
        network=UserNetwork((0,0),0,[(0,1),(1,2),(2,'exit')],(1,2),(1,2),False)
        with patch('src.request_builder.build_remaining_network',return_value=network):
            window=build_window(params,state,{},NS(random_requests=[]))
        real=next(r for r in window.requests if r.observed)
        first=next(r for r in window.requests if r.arc==(0,1))
        second=next(r for r in window.requests if r.arc==(1,2))
        self.assertEqual(real.request_id,waiting.request_id)
        self.assertEqual(real.deadline,.26)
        self.assertEqual(first.predecessors,(waiting.request_id,))
        self.assertEqual(second.predecessors,(first.request_id,))
        self.assertIsNone(first.arrival_time)
        self.assertAlmostEqual(first.return_soc,.9)
        self.assertEqual(len([r for r in window.requests if r.station==0]),1)

    def test_predicted_random_request_cannot_duplicate_observed_history(self):
        params=NS(interval_hours=1/12,horizon=4,num_periods=20,max_wait_hours=.25)
        state=RollingState(1,[[1]],seen_random_ids=['past'])
        forecast=NS(random_requests=[
            dict(request_id='past',station=0,arrival_time=.1,return_soc=.2),
            dict(request_id='new',station=0,arrival_time=.2,return_soc=.3)])
        window=build_window(params,state,{},forecast)
        self.assertEqual([r.request_id for r in window.requests],['new'])
        self.assertFalse(window.requests[0].observed)

    def test_computed_first_arrival_and_deadline_use_grid_arithmetic(self):
        delta = 1/12
        params = NS(interval_hours=delta,horizon=8,num_periods=20,
                    station=NS(positions_km=[10]),range_km=100,
                    vehicle_speed_kmh=60,max_wait_hours=.25)
        user = Reservation((0,0),0,.8,0,.8,True)
        state = RollingState(0,[[1]],{'0:0':user})
        network = UserNetwork((0,0),'origin',[('origin',0),(0,'exit')],(0,),(0,),True)
        with patch('src.request_builder.build_remaining_network',return_value=network):
            window = build_window(params,state,{},NS(random_requests=[]))
        request = window.requests[0]
        self.assertEqual(request.arrival_time,2*delta)
        self.assertEqual(request.deadline,5*delta)
