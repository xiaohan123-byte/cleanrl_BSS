from __future__ import annotations

import unittest

from src.accounting import (
    DuplicateLedgerEventError,
    RealizedLedger,
    UnsupportedLedgerEventError,
)
from src.domain import LedgerEntry, LedgerEventType
from src.time_grid import TimeGrid


class RealizedLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = RealizedLedger(
            TimeGrid(1.0, num_intervals=3),
            energy_price=[2.0, 3.0, 4.0],
            reservation_service_price=[10.0, 11.0, 12.0],
            random_service_price=[5.0, 6.0, 7.0],
            reservation_failure_penalty=13.0,
        )

    def test_carried_service_is_booked_at_service_interval(self) -> None:
        entry = LedgerEntry(
            event_id="service:carried",
            event_type=LedgerEventType.RESERVATION_SERVICE,
            occurred_at=1.2,
            interval=1,
            arrival_time=0.8,
        )
        posting = self.ledger.submit(entry)
        self.assertEqual(posting.income_reservation, 11.0)
        self.assertEqual(self.ledger.reward_for_interval(1), 11.0)

    def test_charge_cost_and_timeout_wait_are_realised_once(self) -> None:
        charge = LedgerEntry(
            event_id="charge:0",
            event_type=LedgerEventType.CHARGING,
            occurred_at=1.0,
            interval=0,
            energy_kwh=3.0,
        )
        timeout = LedgerEntry(
            event_id="timeout:r",
            event_type=LedgerEventType.RESERVATION_TIMEOUT,
            occurred_at=0.25,
            interval=0,
            arrival_time=0.0,
            deadline=0.25,
            metadata={"wait_hours": 0.25},
        )
        self.ledger.submit(charge)
        self.ledger.submit(timeout)
        parts = self.ledger.components_for_interval(0)
        self.assertEqual(parts["charging_cost"], 6.0)
        self.assertEqual(parts["reservation_failure_cost"], 13.0)
        self.assertEqual(parts["reward_delta"], -19.0)
        with self.assertRaises(DuplicateLedgerEventError):
            self.ledger.submit(timeout)

    def test_prediction_only_quantities_are_rejected(self) -> None:
        with self.assertRaises(UnsupportedLedgerEventError):
            self.ledger.submit(
                {
                    "event_id": "pending:x",
                    "event_type": "pending_at_horizon",
                    "occurred_at": 0.0,
                    "interval": 0,
                }
            )

    def test_station_time_price_and_service_energy_factor(self) -> None:
        ledger = RealizedLedger(
            TimeGrid(1.0, num_intervals=3),
            energy_price=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            reservation_service_price=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            battery_capacity_kwh=100.0,
        )
        service = LedgerEntry(
            event_id="service:station-1",
            event_type=LedgerEventType.RESERVATION_SERVICE,
            occurred_at=1.2,
            interval=1,
            station=1,
            metadata={"return_soc": 0.2},
        )
        charge = LedgerEntry(
            event_id="charge:station-1",
            event_type=LedgerEventType.CHARGING,
            occurred_at=2.0,
            interval=1,
            station=1,
            energy_kwh=3.0,
        )
        self.assertEqual(ledger.submit(service).income_reservation, 40.0)
        self.assertEqual(ledger.submit(charge).charging_cost, 15.0)

    def test_path_publication_uses_configured_adjustment_cost(self) -> None:
        ledger = RealizedLedger(
            TimeGrid(1.0, num_intervals=1), path_adjustment_cost=17.5
        )
        publication = LedgerEntry(
            event_id="publish:1:7:0",
            event_type=LedgerEventType.PATH_PUBLISHED,
            occurred_at=0.0,
            interval=0,
            metadata={"path_changed": True},
        )
        self.assertEqual(ledger.submit(publication).adjustment_cost, 17.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
