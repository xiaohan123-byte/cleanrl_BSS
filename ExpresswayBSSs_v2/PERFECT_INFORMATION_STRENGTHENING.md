# Exact strengthening of the perfect-information MILP

The `strengthened` and `strengthened_compact` profiles preserve the business model, objective,
full 186-period horizon and valid upper-bound interpretation. The baseline
profile remains available. These changes are not a new policy or demand sample.

## Arrival/service hull

For a request let A denote selected route activation, R denote the sum of
arrival indicators, T its total service indicator, and F its failure indicator.
For integer route and slot assignments, A and R are binary (the model already
enforces R <= 1), and T is binary. A request cannot be served before arrival,
so T <= R is valid. A reservation satisfies T+F = A AND R; the old lower bound
F >= A+R-T-1 is supplemented with T+F <= A and T+F <= R. Thus no integer
solution is removed, while fractional service cannot multiply one fractional
arrival across several possible service periods.

Queue and failure variables are bounded continuous variables in this profile.
Their existing lower/upper inequalities force their unique zero/one values
whenever route and assignment variables are integral. Precedence comparison
variables remain binary. This eliminates redundant integrality declarations.

## Recharge interval cliques

For slot b, its maximum SOC gain per period is
`g_b = efficiency * interval_hours * slot_power_limit / battery_capacity`.
After a swap at n returning SOC rho, the next swap on the same slot cannot
occur before `n + ceil((1-rho)/g_b)`. Charging starts during period n, exactly
as in the original SOC recursion. Each candidate assignment therefore occupies
the interval from n through n+k-1. For any time t, all candidate occupied
intervals containing t form a clique: their assignment indicators sum to <=1.
The analogous initial unavailability interval has capacity zero. Slot power
heterogeneity is handled individually; shared station power can only delay
recharge further and does not invalidate these necessary inequalities.

A 1e-10 SOC margin is subtracted from the deficit before rounding to avoid
excluding boundary cases because of floating point arithmetic. Zero-power
slots stay occupied through the horizon after any non-full return. These
inequalities follow from full delivery, SOC evolution and slot power limits;
they do not impose immediate or uninterrupted charging.

## Identical-slot first-use ordering

Only slots within the same station with identical initial SOC and identical
power limit are grouped. Prices, charging efficiency, capacity, and the station
power constraint already treat these slots identically. A global permutation
of the entire service/power/SOC trajectory within each group leaves every
physical decision and objective unchanged. Label the slots by nondecreasing
time of their first swap (unused slots last). This proves every original
feasible solution has a representative satisfying this ordering.

For consecutive labels b,b+1 and each possible service boundary n, add
`swap[b+1,n] <= sum_{m<=n} swap[b,m]`. Continuous cumulative-count variables
implement the right side with sparse recursion. This only orders FIRST use;
it does not order total or cumulative service counts between slots. Counts
may cross after first use, and power/SOC curves are not forced to remain sorted.
The imported incumbent is relabeled by this exact global permutation, then
replayed and checked against every row and bound. No service time, route, SOC
amount, power amount or income is changed.

## Solver and verification

The compact profile omits the first-use ordering and its cumulative-count
variables. Initial full-size diagnosis found the first-use version's LP did
not finish within 180 seconds with the default LP method, versus about 3
seconds for the baseline. That diagnostic and its source snapshot are retained
under `outputs/perfect_information_strengthening_day01_20260929`. No full-day
integer experiment was run with that heavier profile. The current trial uses
the compact profile after a separate LP and warm-start validation.

`RootCutRounds=8` is the initial test setting to move beyond the previous long
root-cut phase; it limits rounds, not elapsed root time, and does not relax
feasibility or the requested gap. The total solve limit remains 3600 seconds.
This combined experiment tests the profile, not a causal attribution to each
individual modification. No claim of reaching 1% is made in advance.

The checks compare exact small-instance integer optima and execution replays,
verify a fractional recharge counterexample is removed, compare the baseline
matrix to its previously verified fingerprint, replay/reconstruct the imported
full-size incumbent, and compare ordered strengthened matrices under two Python
hash layouts. Two LP-only solves measure the actual relaxation bounds. The
LP tests are diagnostics, not repeated daily integer experiments or new seeds.

## Day 1 diagnostic results, 2026-09-29

The compact profile passed the checks in
`outputs/perfect_information_strengthening_compact_day01_20260929/report.json`.
All 26 perfect-information regression tests passed, including exact small
integer optima, execution replay, and crossing service-count trajectories.

| Quantity | Baseline | Compact |
|---|---:|---:|
| Variables | 175088 | 175088 |
| Binary variables | 81838 | 77797 |
| Rows | 129471 | 155534 |
| Optimal LP objective | 55005.4318963 | 55005.4318963 |
| LP solve seconds | 2.901 | 3.382 |
| Violated recharge cliques in LP solution | 265 | 0 |

The compact model removes the observed fractional recharge violations but
does not reduce this day's initial LP bound. It adds 2549 arrival/hull rows
and 23514 cooldown rows and removes 4041 redundant binary declarations.
The original 46183.8960213 incumbent remains feasible; its largest checked
residual is 5.47e-13. Ordered matrix hashes match under both hash layouts,
and the baseline still exactly matches the previously verified matrix.

The authorized full-day trial uses this compact profile, RootCutRounds=8,
3600 seconds, gap=0.01, 16 threads, seed=1, day=1, failure penalty=200 and
the original 3600-second incumbent. Its purpose is to test search progress
under the full time budget; the LP diagnosis alone establishes no MIP speedup.
