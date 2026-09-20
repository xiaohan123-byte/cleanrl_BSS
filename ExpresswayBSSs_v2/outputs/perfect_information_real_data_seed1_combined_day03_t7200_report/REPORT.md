# Perfect-information experiment: completed results

All seven selected feasible solutions passed the saved execution and accounting replay.
Reservation failure penalty: 200 yuan per user. These are time-limited feasible solutions, not certified optima.

Selection: days 1, 37, 4, 19, 13, 14 use the original 1800-second runs; day 3 uses its separately authorized 7200-second re-solve.
Input and source hashes were checked against both saved snapshots. All settings other than the day subset and time limit match.
Original experiment results and attempt markers are preserved.

| Day | Limit (s) | Feasible profit (yuan) | Solver upper bound (yuan) | Gap (%) | Replay |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 1800 | 33473.52 | 54950.89 | 39.08 | passed |
| 37 | 1800 | 35156.50 | 54880.51 | 35.94 | passed |
| 3 | 7200 | 29442.07 | 55829.98 | 47.26 | passed |
| 4 | 1800 | 36330.51 | 53638.21 | 32.27 | passed |
| 19 | 1800 | 34457.85 | 55868.25 | 38.32 | passed |
| 13 | 1800 | 34622.24 | 54165.13 | 36.08 | passed |
| 14 | 1800 | 29463.22 | 57802.83 | 49.03 | passed |

Seven-day mean feasible profit: **33277.99 yuan**.
Seven-day mean solver upper bound: **55305.11 yuan** (same selected runs).
Mean reservation failure rate: 11.29%; random service rate: 53.07%.
Mean wait: 13.99 min (equal weighting across days); path adjustments: 0.00.
Gap target (0.01%) achieved: 0/7; maximum gap: 49.03%.

Selected seven whole-day solves: mean 2571.49 s, P95 5580.07 s.
P95 uses linear interpolation across the seven whole-day solve durations; these are not MPC iteration times.
All eight actual attempts, including the original day-3 attempt: 19800.45 s of solver time.
Original uniform 1800-second experiment: 6/7 verified incumbents; seven-day feasible mean remains unavailable; upper-bound mean 55322.03 yuan.

For every selected day: reservations completed + failed = 200; random requests served + timed out = 200; no final waiting requests or active users.
Maximum absolute objective replay residual: 7.301e-09 yuan.
Per-day tolerances and physical residuals are retained in summary.json.

Environment: Windows 10.0.26200, 8 logical processors, Python 3.10.19, COPT 8.0.6, NumPy 2.2.6.
Solver: 8 threads, RelGap=0.0001, AbsGap=0, feasibility tolerance 1e-8.

Existing zero-terminal paper results use a 1000-yuan failure penalty. These 200-yuan results do not establish a paired comparison under identical objectives.
This status-request report does not modify the paper. No experiment was restarted.

Full decisions and replay ledgers remain in the source directories listed in summary.json.
