"""Canonical continuous-time interval helpers.

The online executor, reference rollout, and optimisation model all need the
same boundary convention: an interval is ``[start(q), end(q))``.  In
particular, an event at the prediction end belongs to the next rolling round.
This module intentionally keeps that rule small and explicit rather than
spreading ``floor(t / delta)`` expressions through the code base.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Tuple


class TimeGridError(ValueError):
    """Raised when a time or interval index is outside the grid contract."""


@dataclass(frozen=True)
class TimeGrid:
    """A fixed-width, half-open time grid.

    ``boundary_tolerance`` is only used to remove floating-point noise from
    arithmetic that is known to have produced a grid boundary (for example,
    repeated addition of ``interval_hours``).  It is *not* a guard band that
    snaps ordinary near-boundary business events.  Therefore an event at
    ``t_end - 5e-8`` remains in the current interval by default.
    """

    interval_hours: float
    num_intervals: int | None = None
    boundary_tolerance: float = 1e-10
    origin: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.interval_hours) or self.interval_hours <= 0:
            raise TimeGridError("interval_hours must be a positive finite number")
        if self.num_intervals is not None and self.num_intervals <= 0:
            raise TimeGridError("num_intervals must be positive when specified")
        if not isfinite(self.boundary_tolerance) or self.boundary_tolerance < 0:
            raise TimeGridError("boundary_tolerance must be a non-negative finite number")
        if not isfinite(self.origin):
            raise TimeGridError("origin must be finite")

    def _validate_index(self, q: int, *, permit_endpoint: bool = False) -> None:
        if not isinstance(q, int):
            raise TimeGridError(f"interval index must be int, got {type(q)!r}")
        if q < 0:
            raise TimeGridError("interval index must be non-negative")
        if self.num_intervals is not None:
            maximum = self.num_intervals if permit_endpoint else self.num_intervals - 1
            if q > maximum:
                raise TimeGridError(f"interval index {q} is outside the configured grid")

    def start(self, q: int) -> float:
        """Return the inclusive start of interval ``q``."""

        self._validate_index(q)
        return self.origin + q * self.interval_hours

    def end(self, q: int) -> float:
        """Return the exclusive end of interval ``q``."""

        self._validate_index(q)
        return self.origin + (q + 1) * self.interval_hours

    def interval(self, q: int) -> Tuple[float, float]:
        """Return ``(start, end)`` for the half-open interval ``q``."""

        return (self.start(q), self.end(q))

    def snap_boundary(self, t: float, *, proven_boundary: bool = False) -> float:
        """Normalize only a *proven* boundary calculation's round-off.

        This is deliberately conservative.  It does not use a business guard
        band, and hence does not change ownership of normal events close to an
        endpoint.  Callers with provenance that a value was computed as a
        boundary may use this helper before comparison.
        """

        if not isfinite(t):
            raise TimeGridError("time must be finite")
        if not proven_boundary:
            return t
        relative = (t - self.origin) / self.interval_hours
        nearest = round(relative)
        boundary = self.origin + nearest * self.interval_hours
        if abs(t - boundary) <= self.boundary_tolerance:
            return boundary
        return t

    def interval_of(self, t: float) -> int:
        """Return the interval containing ``t`` under the half-open rule.

        An exact boundary belongs to the interval beginning at that boundary.
        ``t`` may equal the configured final end only when a following interval
        exists; otherwise it is outside the finite grid.
        """

        if not isfinite(t):
            raise TimeGridError("time must be finite")
        if t < self.origin:
            raise TimeGridError(f"time {t} is before grid origin {self.origin}")
        relative = (t - self.origin) / self.interval_hours
        # ``int`` truncates toward zero; the precondition above makes it floor.
        q = int(relative)
        if self.num_intervals is not None and q >= self.num_intervals:
            raise TimeGridError(f"time {t} is at or after the configured grid end")
        return q

    def prediction_bounds(self, ell: int, horizon: int) -> Tuple[float, float]:
        """Return the half-open prediction window ``[t_ell, t_(ell+H))``."""

        self._validate_index(ell)
        if not isinstance(horizon, int) or horizon <= 0:
            raise TimeGridError("horizon must be a positive integer")
        if self.num_intervals is not None and ell + horizon > self.num_intervals:
            raise TimeGridError("prediction horizon extends beyond configured grid")
        return (self.start(ell), self.origin + (ell + horizon) * self.interval_hours)

    def normalize_for_window(
        self, t: float, t_end: float, *, proven_boundary: bool = False
    ) -> float:
        """Normalize round-off at ``t_end`` without a global near-end snap.

        The signature is retained as the single boundary-normalisation entry
        point for callers.  Values strictly before the end remain unchanged;
        only values within ``boundary_tolerance`` of the supplied endpoint are
        canonicalised to it when the caller explicitly supplies provenance.
        """

        if not isfinite(t) or not isfinite(t_end):
            raise TimeGridError("t_end must be finite")
        if proven_boundary and abs(t - t_end) <= self.boundary_tolerance:
            return t_end
        return self.snap_boundary(t, proven_boundary=proven_boundary)

    def contains_execution_time(self, t: float, ell: int) -> bool:
        """Whether ``t`` belongs to execution interval ``ell``.

        The right endpoint is excluded.  This is the predicate to use for
        realised arrivals, services, and deadlines.
        """

        start, end = self.interval(ell)
        return start <= t < end

    def is_terminal_event(self, t: float, ell: int, horizon: int) -> bool:
        """Whether an event is exactly at the prediction right endpoint."""

        _, end = self.prediction_bounds(ell, horizon)
        return t == end

    def current_event_upper_bound(self, t_end: float) -> float:
        """Return the strict current-window supremum for model-side bounds.

        Continuous execution uses a half-open set, so there is no finite
        largest valid real number.  Returning the endpoint itself lets callers
        formulate ``t < t_end`` directly (or use a solver's strict-equivalent
        guard) without inventing a broad, behaviour-changing time margin.
        """

        if not isfinite(t_end):
            raise TimeGridError("t_end must be finite")
        return t_end


__all__ = ["TimeGrid", "TimeGridError"]
