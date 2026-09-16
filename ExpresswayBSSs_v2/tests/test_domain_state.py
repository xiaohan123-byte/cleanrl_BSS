import json
import unittest
from src.domain import DomainError, Reservation, RollingState, WaitingRequest

class DomainStateTest(unittest.TestCase):
    def test_round_trip_preserves_request_id_deadline_and_distinct_plans(self):
        r = WaitingRequest('A:0:3:0', 0, 'reservation', .03, .28, .2, (0, 3))
        user = Reservation((0,3), 0, .6, 20, .2, True,
                           day_ahead=[0,1], retained_plan=[1],
                           published_plan=[0,1], waiting_request_id=r.request_id)
        state = RollingState(2, [[1,.4]], {'0:3':user}, {r.request_id:r})
        restored = RollingState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(restored.waiting[r.request_id].deadline, .28)
        self.assertEqual(restored.users['0:3'].day_ahead, [0,1])
        self.assertEqual(restored.users['0:3'].retained_plan, [1])
        self.assertEqual(restored.users['0:3'].published_plan, [0,1])
        restored.users['0:3'].retained_plan.clear()
        self.assertEqual(state.users['0:3'].retained_plan,[1])

    def test_return_soc_must_be_strictly_below_one(self):
        with self.assertRaises(DomainError):
            WaitingRequest('bad',0,'random',0,.25,1)

    def test_waiting_request_cannot_be_orphaned(self):
        state = RollingState(0,[[1]],waiting={'a':WaitingRequest('a',0,'reservation',0,.25,.2,(0,0))})
        with self.assertRaises(DomainError):
            state.validate()

    def test_mutated_physical_records_are_revalidated(self):
        user = Reservation((0,0),0,.8,0,.8)
        state = RollingState(0,[[1]],{'0:0':user})
        user.soc = 1.2
        with self.assertRaises(DomainError):
            state.validate()
        user.soc = .8
        request = WaitingRequest('random',0,'random',0,.25,.2)
        state.waiting['random'] = request
        request.deadline = -.1
        with self.assertRaises(DomainError):
            state.validate()
