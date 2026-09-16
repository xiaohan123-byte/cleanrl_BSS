from __future__ import annotations

import unittest

from src.time_grid import TimeGrid, TimeGridError


class TimeGridTest(unittest.TestCase):
    def test_half_open_boundaries_and_prediction_end(self) -> None:
        grid = TimeGrid(interval_hours=1.0, num_intervals=4)
        self.assertEqual(grid.interval(1), (1.0, 2.0))
        self.assertEqual(grid.prediction_bounds(0, 2), (0.0, 2.0))
        self.assertEqual(grid.interval_of(0.0), 0)
        self.assertEqual(grid.interval_of(1.0), 1)
        self.assertTrue(grid.contains_execution_time(0.999999, 0))
        self.assertFalse(grid.contains_execution_time(1.0, 0))
        self.assertTrue(grid.is_terminal_event(2.0, 0, 2))
        self.assertFalse(grid.is_terminal_event(1.999999, 0, 2))

    def test_event_just_before_end_is_not_globally_snapped(self) -> None:
        grid = TimeGrid(interval_hours=1.0, num_intervals=3)
        before_end = 1.0 - 5e-8
        self.assertEqual(grid.interval_of(before_end), 0)
        self.assertTrue(grid.contains_execution_time(before_end, 0))
        self.assertFalse(grid.is_terminal_event(before_end, 0, 1))
        self.assertEqual(grid.interval_of(1.0), 1)
        self.assertFalse(grid.contains_execution_time(1.0, 0))

    def test_boundary_roundoff_requires_explicit_provenance(self) -> None:
        grid = TimeGrid(interval_hours=1.0)
        rounded = 1.0 + 5e-11
        self.assertEqual(grid.snap_boundary(rounded), rounded)
        self.assertEqual(grid.snap_boundary(rounded, proven_boundary=True), 1.0)
        self.assertEqual(grid.normalize_for_window(rounded, 1.0), rounded)
        self.assertEqual(
            grid.normalize_for_window(rounded, 1.0, proven_boundary=True), 1.0
        )

    def test_finite_grid_rejects_final_right_endpoint(self) -> None:
        grid = TimeGrid(interval_hours=1.0, num_intervals=2)
        with self.assertRaises(TimeGridError):
            grid.interval_of(2.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
