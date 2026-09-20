# Full-day perfect-information experiment

This implements the agreed Section 5.1 offline experiment. Seven frozen days
`[1,37,3,4,19,13,14]` share seed 1 and the same original day-ahead plans as the
online comparisons. It never loads online experiment trajectories. The default
is one MILP per day, 1800 seconds, relative gap 0.0001, absolute gap zero and
8 threads (or fewer on a smaller machine).

The model knows actual entries, SOC, fixed physical-segment travel multipliers
and random requests. Initial paths use actual SOC with original day-ahead path
protection. A path is selected before entry and remains fixed. Downstream
arrival depends on the preceding service time. All demand is settled over 186
15-minute intervals; cyclic prices and the original queue, battery and power
rules are retained. There is no unpublished-plan penalty, terminal value,
salvage or forced charging. See `PERFECT_INFORMATION_PATH_COVERAGE.md` for the
structural coverage argument and its assumptions.

The reservation failure penalty is set by `--failure-penalty` (default 200),
which overrides the frozen scenario value of 1000 without touching the frozen
inputs; the applied value is frozen into `run.json` and each day's saved
parameter snapshot.

`src/perfect_information_model.py` is independent of the shared MPC solver. Its
sparse arc/service/FCFS constraints follow that solver, with a single full-day
objective. Combining `sum(slot assignments) <= SOC <= 1` enforces both one swap
per battery/boundary and delivery only when full. A complete deterministic
zero-service MIP start is checked against every generated linear constraint.
It is also replayed through the existing physics/accounting code.

From the project directory, using the existing py310 environment:

```powershell
& 'D:\Users\miniconda\envs\py310\python.exe' -B run_perfect_information.py --prepare-only
& 'D:\Users\miniconda\envs\py310\python.exe' -u -B run_perfect_information.py
```

The preparation checks frozen input hashes, refreshes structural path coverage,
and copies source files into an immutable experiment snapshot. The controller
runs one hidden worker at a time from that snapshot, releasing native solver
memory between days. Other experiments must finish before this controller is
started; it does not stop or reconfigure other processes.

Default output: `outputs/perfect_information_real_data_seed1_v1`.

- `run.json`, `coverage.json`, `code/`: configuration, environment, input hashes
  and source evidence.
- `controller_status.json`, `worker_logs/`: lifecycle and console logs.
- `days/day_NN/status.json`: attempt marker and current phase.
- `days/day_NN/solver.log`, `solver_result.json`: raw solver status, incumbent,
  bound, gap, timings and full decisions, saved before replay.
- `days/day_NN/initial_check.json`, `replay.json.gz`, `result.json`: seed audit,
  full incumbent ledger and SOC checks, compact verified per-day result.

A solver bound is stored independently of an incumbent. Invalid statuses,
nonfinite bounds and the solver's infinity sentinel are not published as valid
bounds. Service metrics require a successful replay. Replay tolerances are
1e-8 hours for arrival, 1e-8 for return SOC, 1e-6 for accumulated battery SOC,
and max(1e-4 yuan, 1e-8 times absolute objective) for each accounting component.
The shared engine retains its own feasibility and power checks. Residuals are
saved; decisions are not reoptimized or replaced by a fallback policy.

Completed identical dates can be reused. A date marked running, interrupted or
failed is never automatically solved again. Input/code/solver changes require
a separate output version; missing markers alongside artifacts are errors.
An interrupted controller's lock must be inspected before any deliberate
recovery. Do not delete a live lock or clear attempt markers to rerun a date.

Per the user's instruction, starting the formal experiment ends the assistant's
active work: no ongoing monitoring or automatic reporting/paper editing.
On the user's later progress request, inspect saved status/results. After the
controller has ended, explicitly aggregate with:

```powershell
& 'D:\Users\miniconda\envs\py310\python.exe' -B run_perfect_information.py --report-only
```

This produces CSV, JSON, a report and two LaTeX rows. Missing dates invalidate
the corresponding seven-day means. The bound row has no service metrics; the
incumbent row includes only verified decisions. P95 is computed across seven
whole-day solves, not MPC rounds. The latest paper must then be read and its
perfect-information text/rows updated without changing zero-terminal or RL rows.

Tests: `python -B -m unittest discover -s tests -p 'test_perfect_information*.py' -v`.
