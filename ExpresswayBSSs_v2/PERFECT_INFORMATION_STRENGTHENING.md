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

## Whole-day service-count ordering, 2026-10-01

The independent `slot_order='total_services'` option adds, within each station
and each class of identical initial SOC and slot power limit,

`sum_(request,period) alpha[request,b,period] >= sum_(request,period) alpha[request,b+1,period]`.

For the frozen data every slot within a station is identical, so this adds
exactly `sum_i (B_i-1) = 250-11 = 239` rows and no new variables. It orders
WHOLE-DAY totals, not cumulative counts at every period, SOC, or first use.
The first-use profile cannot be combined with this option: two independent
canonicalization rules need not have a common representative.

Proof: simultaneously permute complete assignment, charging-power and SOC
trajectories within an identical-slot class. Initial conditions, slot bounds,
SOC equations, station power sums, all service/queue decisions and the
objective remain unchanged. Choose the permutation that sorts service totals
in descending order. Every original feasible operational plan retains a
representative, including every optimum. Thus the ordered full model's
maximization bound is valid for the original full problem. Warm starts are
permuted in precisely this way and replayed before use.

The same argument also applies to fractional LP trajectories and real-valued
service totals. Consequently, these count-order inequalities alone do not
change the optimal initial LP objective. Their intended benefit is reducing
permutation symmetry in integer search; their effects on presolve, cuts and
runtime must be measured, and speedup is not guaranteed.

`run_perfect_information_count_order.py` compares the existing compact profile
with and without these rows, leaving paths and service choices free. Both use
the exact same count-canonicalized feasible start, 16 threads, RootCutRounds=8,
600-second MIP limits and RelGap=0.01. Separate relaxation checks have 60-second
limits. The baseline matrix must match the prior full-model fingerprint; the
MIP matrices must exactly match their corresponding checked LP builds. Seed=1,
test day=1, reservation failure penalty=200. There is no station decomposition.

Before this comparison, `src/perfect_information_charging.py` substitutes the
fixed service/request/slot/time decisions into the original full-delivery and
SOC equations. Its only variables are continuous charging power and SOC. It
keeps all 11 stations in one LP and all 186 periods, with the original cyclic
prices, efficiency, power limits and no terminal inventory target or salvage.
Its cost bound is conditional on the fixed assignment, not an original-problem
profit bound. Both physical replay and unchanged service assignments are checked.

All 34 perfect-information regression tests passed before launching the trial.
The added tests cover exact small optimum equivalence, zero extra variables,
the precise number of ordering rows, crossing intermediate service counts,
heterogeneous-slot classes and continuous charging optimum/replay checks.
Inputs, code, logs and results are frozen under
`outputs/pi_count_order_day01_f200_t600_20261001/`.

### Measured results

The fixed-assignment LP has 93250 continuous variables and 49381 rows, with
zero integer variables. It reached its optimum in **0.125 seconds**, with
2.971 seconds of model construction/audit. The optimal charging cost was
11706.850287675 yuan and profit remained 46766.938986392 yuan. Thus the supplied
incumbent already had optimal charging for its fixed assignment. The code
submitted one all-station LP; COPT automatically recognized 11 independent
blocks, all included in the reported solve time. No separate station trials
were run.

| Quantity | Compact baseline | Compact + count ordering |
|---|---:|---:|
| Original variables | 175088 | 175088 |
| Original binary variables | 77797 | 77797 |
| Original rows | 155534 | 155773 |
| Standalone LP solve seconds | 2.701, optimal | 60.051, time limit |
| Standalone certified optimal LP objective | 55005.4318963 | not obtained within limit |
| Presolved MIP columns | 134528 | 140227 |
| Presolved MIP binaries / other integers | 79267 / 0 | 84163 / 228 |
| First root LP bound logged at about | 51 seconds | 113 seconds |
| MIP solve seconds | 600.055 | 600.030 |
| Verified profit | 46766.9389864 | 46766.9389864 |
| MIP upper bound | 54992.3264959 | 55005.4318125 |
| MIP gap | 14.9573% | 14.9776% |
| Nodes reported | 1 | 1 |

Both MIPs used byte-identical starts, with SHA256
`2b631988f8e47eaf47f4f85fbde06507794f27e24418c60b662f3ce64ad7d700`.
The count-order option did not improve the feasible objective or the final
bound within this budget. It increased LP/root work and the presolved size;
adding few rows was not enough to guarantee faster optimization. It remains
an opt-in diagnostic feature, disabled by default. This result concerns the
current instance and formulation; it does not show that all symmetry-handling
methods are ineffective.

All three integer-feasible trajectories passed the original physical replay
and a separate ledger/conservation check. Each completed 199 reservations,
failed 1, served 157 random requests and timed out 43, with no unsettled users.
The maximum independent accounting residual was below 5.3e-9 yuan. The charging
LP preserved all 585 request/time/slot decisions. The ordered MIP also passed
explicit whole-day count checks.

The two MIP bounds apply to the original full problem by the permutation
argument above. The charging LP bound only applies to the fixed assignment.
The previously obtained full-problem upper bound 54930.165935947 remains
stronger than either new bound. The best known original-problem interval is
therefore unchanged at [46766.938986392, 54930.165935947], gap **14.8611%**.

`comparison.json` and `comparison.txt` contain the checked summary. The
`best_feasible/days/day_01/solver_result.json` export can be reused as a warm
start by the existing runner. Neither experiment changed the paper's results.
