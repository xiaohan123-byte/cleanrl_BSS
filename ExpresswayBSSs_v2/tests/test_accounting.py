import unittest
from copy import deepcopy

from src.accounting import (
    DuplicateLedgerEventError, LedgerError, UnsupportedLedgerEventError,
    append_events, event_components, summarize_ledger,
)
from src.parameters import BusinessParameters


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.p = BusinessParameters(num_periods=4, horizon=2)

    def service(self, request="R", period=0):
        return dict(event_id=f"service:{request}", type="random_service", period=period,
                    time=period * self.p.interval_hours, request_id=request, station=0, slot=0,
                    return_soc=.25, energy_kwh=75., unit_price=self.p.swap_service_price[0][period],
                    arrival_time=0., deadline=.25, realized=True)

    def test_revenue_and_grid_energy_cost_are_independent(self):
        service = self.service()
        charging = dict(event_id="charging:0:0:0", type="charging", period=0, time=0.,
                        station=0, slot=0, power_kw=60., energy_kwh=5., unit_price=.35,
                        start_soc=.25, end_soc=.2975, realized=True)
        ledger = []
        reward = append_events(self.p, ledger, [service, charging])
        self.assertAlmostEqual(reward, 90. - 1.75)
        self.assertAlmostEqual(summarize_ledger(ledger)["grid_energy_kwh"], 5.)

    def test_actual_service_uses_service_period_price(self):
        self.p.swap_service_price[0][1] = 2.
        components = event_components(self.p, self.service(period=1))
        self.assertEqual(components["income_random"], 150.)

    def test_prediction_and_tampered_financials_are_rejected(self):
        event = self.service()
        event["realized"] = False
        with self.assertRaises(UnsupportedLedgerEventError):
            event_components(self.p, event)
        for field, wrong in (("energy_kwh", 76.), ("unit_price", 9.), ("reward_delta", 500.)):
            event = self.service()
            event[field] = wrong
            with self.assertRaises(LedgerError):
                event_components(self.p, event)

    def test_duplicate_submission_is_atomic(self):
        ledger = []
        with self.assertRaises(DuplicateLedgerEventError):
            append_events(self.p, ledger, [self.service(), self.service()])
        self.assertEqual(ledger, [])
        append_events(self.p, ledger, [self.service()])
        copied = self.service()
        copied["event_id"] = "different-id-same-service"
        with self.assertRaises(DuplicateLedgerEventError):
            append_events(self.p, ledger, [copied])

    def test_failure_penalty_is_once_per_reservation(self):
        first = dict(event_id="timeout:A:0:0:0", type="reservation_failure", period=0,
                     time=.02, user_key="0:0", request_id="A:0:0:0", realized=True)
        ledger = []
        append_events(self.p, ledger, [first])
        second = deepcopy(first)
        second.update(event_id="timeout:A:0:0:1", request_id="A:0:0:1")
        with self.assertRaises(DuplicateLedgerEventError):
            append_events(self.p, ledger, [second])
        self.assertEqual(summarize_ledger(ledger)["reservation_failure_cost"], self.p.reservation_failure_penalty)

    def test_served_and_timed_out_outcomes_are_mutually_exclusive(self):
        ledger = []
        append_events(self.p, ledger, [self.service()])
        timeout = dict(event_id="timeout:R", type="random_timeout", period=0,
                       time=.02, request_id="R", station=0, realized=True)
        with self.assertRaises(DuplicateLedgerEventError):
            append_events(self.p, ledger, [timeout])
        ledger = []
        append_events(self.p, ledger, [timeout])
        with self.assertRaises(DuplicateLedgerEventError):
            append_events(self.p, ledger, [self.service()])

    def test_returned_battery_soc_must_be_strictly_below_full(self):
        event = self.service()
        event["return_soc"] = 1.
        with self.assertRaises(LedgerError):
            event_components(self.p, event)


if __name__ == "__main__":
    unittest.main()
