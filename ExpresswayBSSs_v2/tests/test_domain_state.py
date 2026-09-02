from __future__ import annotations

import unittest

from src.domain import (
    CandidateRequest,
    PhysicalRequestStatus,
    RequestKind,
    RollingState,
    SlotState,
    WaitingRequest,
)


class DomainStateTest(unittest.TestCase):
    def test_json_round_trip_preserves_deadline_and_queue_order(self) -> None:
        state = RollingState(
            now=1.0,
            slots=[[SlotState(0, 0, 0.4, last_update_time=1.0)]],
        )
        later = WaitingRequest(
            request_id="later",
            kind=RequestKind.RANDOM,
            station=0,
            arrival_time=0.8,
            deadline=1.3,
            return_soc=0.2,
        )
        earlier = WaitingRequest(
            request_id="earlier",
            kind=RequestKind.RANDOM,
            station=0,
            arrival_time=0.7,
            deadline=1.2,
            return_soc=0.3,
        )
        state.add_waiting(later)
        state.add_waiting(earlier)
        state.request_status[earlier.event_id] = PhysicalRequestStatus.WAITING

        restored = RollingState.from_dict(state.to_dict())
        queue = restored.queue_for(0, RequestKind.RANDOM)
        self.assertEqual([item.request_id for item in queue], ["earlier", "later"])
        self.assertEqual([item.deadline for item in queue], [1.2, 1.3])
        self.assertEqual(restored.to_dict(), state.to_dict())

    def test_reservation_candidate_uses_stable_path_event_id(self) -> None:
        candidate = CandidateRequest(
            request_id="r-7",
            kind=RequestKind.RESERVATION,
            station=3,
            arrival_time=2.0,
            deadline=2.25,
            return_soc=0.4,
            user_key=(1, 7),
            path_order=2,
        )
        self.assertEqual(candidate.event_id, "reservation:1:7:2:3")
        self.assertEqual(candidate.to_waiting_request().event_id, candidate.event_id)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
