import unittest
from types import SimpleNamespace as NS
from src.domain import DomainError, Reservation, RollingState, WaitingRequest
from src.path_state import apply_path_decisions

def params():
    return NS(interval_hours=1/12,path_update_interval=3,vehicle_speed_kmh=60,
              station=NS(positions_km=[20,40,60]),
              od_pairs=[NS(od_id=0,station_indices=[0,1,2])])

class PublicationTest(unittest.TestCase):
    def user_state(self,entered=False):
        user=Reservation((0,0),1,.9,0,.9,entered,day_ahead=[0,2],
                         retained_plan=[0,2],published_plan=[0,2] if entered else None)
        return RollingState(0,[[1],[1],[1]],{'0:0':user})

    def test_future_plan_is_retained_without_publication(self):
        state=self.user_state()
        self.assertEqual(apply_path_decisions(params(),state,{'0:0':[1,2]}),[])
        self.assertEqual(state.users['0:0'].retained_plan,[1,2])
        self.assertEqual(state.users['0:0'].day_ahead,[0,2])
        self.assertIsNone(state.users['0:0'].published_plan)
        state.period=1
        with self.assertRaises(DomainError):
            apply_path_decisions(params(),state,{'0:0':[0,2]})
        state.period=3
        apply_path_decisions(params(),state,{'0:0':[0,2]})

    def test_changed_enroute_path_costs_once_per_user(self):
        state=self.user_state(True)
        events=apply_path_decisions(params(),state,{'0:0':[1]})
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['type'],'path_adjustment')
        self.assertEqual(apply_path_decisions(params(),state,{'0:0':[1]}),[])

    def test_completed_and_waiting_stations_are_not_path_changes(self):
        state=self.user_state(True)
        user=state.users['0:0']
        user.position_km=20
        user.waiting_request_id='actual'
        state.waiting['actual']=WaitingRequest('actual',0,'reservation',0,.25,.4,(0,0))
        self.assertEqual(apply_path_decisions(params(),state,{'0:0':[2]}),[])
        self.assertIn('actual',state.waiting)

    def test_first_publication_does_not_charge_adjustment(self):
        state=self.user_state()
        state.users['0:0'].entered=True
        self.assertEqual(apply_path_decisions(params(),state,{'0:0':[0,2]}),[])
        self.assertEqual(state.users['0:0'].published_plan,[0,2])
