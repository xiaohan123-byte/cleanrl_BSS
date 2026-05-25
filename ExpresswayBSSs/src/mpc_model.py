"""
MPC model for highway battery swapping station operation.

The formulation follows paper/mian8_fixed_fee.tex.  The MPC layer optimizes
reservation assignment, random-user service, and SOC inventory transitions.
Charging decisions and terminal inventory values are external parameters so
that an RL actor/critic can be connected later without changing the model.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Tuple

import pandas as pd
import tyro
from gurobipy import GRB, LinExpr, Model, quicksum, tuplelist


# Tuple-key conventions used throughout the model.  These aliases are the
# quickest way to check whether an expression is using the intended indices.
#
# fA[n,p,t]          reservation demand with initial SOC n, OD/path p, depart t
# fR[n,i,t]          walk-in demand returning a battery of SOC n at station i, time t
# P[n,i,t]           RL-provided charging SOC increment for batteries in state n
# lambda[i,n]        RL critic marginal terminal value for one battery
# w_initial[n,i]     inventory at the start of the prediction horizon
# historical_yA[...] reservation flows that departed before this horizon
KeyFA = Tuple[int, str, int]
KeyFR = Tuple[int, str, int]
KeyP = Tuple[int, str, int]
KeyLambda = Tuple[str, int]
KeyWInitial = Tuple[int, str]
KeyHistoricalYA = Tuple[int, str, int, str, str]


class TeeStream:
    """Write output to both console and a log file."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def setup_run_log_file(num_soc: int, num_stations: int) -> tuple[str, Any]:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(module_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_name = f"MPC_{num_soc}_{num_stations}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_name)
    return log_path, open(log_path, "a", encoding="utf-8")


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_data_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str

    module_dir = os.path.dirname(os.path.abspath(__file__))
    root = _project_root()
    candidates = [
        os.path.join(root, path_str),
        os.path.join(module_dir, path_str),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def _literal_tuple(key: str) -> tuple:
    value = ast.literal_eval(key)
    if not isinstance(value, tuple):
        raise ValueError(f"JSON key is not a tuple: {key}")
    return value


def _to_float_dict(raw: Mapping[Any, Any]) -> dict:
    return {key: float(value) for key, value in raw.items()}


@dataclass
class MPCConfig:
    """Configuration for one MPC solve."""

    # System scale and prediction horizon.  T_set is the decision-period set
    # {0,...,H-1}; W_set is the inventory-state set {0,...,H}.
    num_stations: int = 3
    TimePeriods: int = 8
    max_tau: int = 5
    N_soc: int = 5

    # Objective weights:
    # alpha penalizes unserved reservation demand; eta enforces a minimum
    # reservation service rate; beta_terminal weights the RL critic value.
    alpha: float = 300.0
    eta: float = 0.8
    beta_terminal: float = 1.0
    service_fee: float = 0.6

    # SOC discretization uses R_range / N_soc as the mileage represented by one
    # SOC level.  speed converts origin-to-node distance into travel periods.
    speed: float = 100.0
    R_range: float = 400.0

    E: Tuple[float, ...] = (
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.9,
        0.9,
        1.4,
        1.4,
        1.4,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        1.4,
        1.4,
        1.4,
        0.9,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.9,
        0.9,
        1.4,
        1.4,
        1.4,
        0.9,
        0.9,
        0.9,
    )

    path_data_file: str = "data_generation_optim/output/data_6e.json"
    dist_file: str = "data_generation_optim/output/dist_6e.csv"
    demand_a_file: str = "data_generation_optim/output/6e_flow_100_0.30.json"
    demand_r_file: str = "data_generation_rl/output/3s6e_station_demand_poisson_0d.json"

    # require_terminal_inventory keeps the optional safety stock constraint
    # w[N,i,H] >= lower_bound_i from the paper.
    gurobi_time_limit: int = 3600
    mip_gap: float = 0.01
    log_to_console: bool = True
    require_terminal_inventory: bool = True

    stations: List[str] = field(init=False)
    N_set: List[int] = field(init=False)
    reservation_soc_set: List[int] = field(init=False)
    random_soc_set: List[int] = field(init=False)
    T_set: List[int] = field(init=False)
    W_set: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.path_data_file = _resolve_data_path(self.path_data_file)
        self.dist_file = _resolve_data_path(self.dist_file)
        self.demand_a_file = _resolve_data_path(self.demand_a_file)
        self.demand_r_file = _resolve_data_path(self.demand_r_file)

        self.stations = [f"s{i}" for i in range(1, self.num_stations + 1)]
        self.N_set = list(range(self.N_soc + 1))
        # Reservation users are assumed to enter the expressway with enough SOC.
        self.reservation_soc_set = list(range(2, self.N_soc + 1))
        # Walk-in users are assumed to request swapping only at low SOC.
        # For N_soc=5 this is {0,1,2}, matching range(0, N-2).
        self.random_soc_set = list(range(0, max(0, self.N_soc - 2)))
        self.T_set = list(range(self.TimePeriods))
        self.W_set = list(range(self.TimePeriods + 1))


def load_mpc_data(config: MPCConfig) -> dict:
    """Load network data and default demand files for the MPC model.

    Main returned symbols:
    A[p][n]      feasible arcs in the SOC-wise expanded network A^{n,p}
    Sp[p]        intermediate BSS set S^p
    tau[p,i]     travel time from origin of path p to node i
    rho[n,p,i,j] returned battery SOC after traversing arc i -> j
    fA/fR        default demand dictionaries; tests or RL envs may override them
    """

    with open(config.path_data_file, "r", encoding="utf-8") as file:
        path_data_raw = json.load(file)

    path_set = list(path_data_raw.keys())
    dist = pd.read_csv(config.dist_file, header=0, index_col=0)

    A: dict[str, dict[int, tuplelist]] = {}
    Sp: dict[str, list[str]] = {}
    source: dict[str, str] = {}
    dest: dict[str, str] = {}
    tau: dict[tuple[str, str], int] = {}
    tau_max: dict[str, int] = {}

    for p in path_set:
        source[p] = path_data_raw[p]["source"]
        dest[p] = path_data_raw[p]["root"]
        Sp[p] = list(path_data_raw[p].get("stations", []))
        A[p] = {}
        for n in config.N_set:
            # strategy[str(n)] already encodes SOC-feasible arcs for OD p and
            # initial SOC n, so infeasible station choices never become vars.
            arcs = path_data_raw[p].get("strategy", {}).get(str(n), [])
            A[p][n] = tuplelist((str(i), str(j)) for i, j in arcs)

        max_for_p = 0
        for node in [source[p], *Sp[p], dest[p]]:
            if source[p] in dist.index and node in dist.columns:
                value = int(math.ceil(float(dist.loc[source[p], node]) / config.speed))
            else:
                value = 0
            tau[(p, node)] = value
            max_for_p = max(max_for_p, value)
        tau_max[p] = max_for_p

    ratio = config.R_range / config.N_soc
    energy_consumption: dict[tuple[str, str, str], int] = {}
    rho: dict[tuple[int, str, str, str], int] = {}
    for p in path_set:
        for i, j in path_data_raw[p].get("arcs", []):
            if i in dist.index and j in dist.columns:
                consumption = int(math.ceil(float(dist.loc[i, j]) / ratio))
            else:
                consumption = 0
            energy_consumption[(p, i, j)] = consumption
        for n in config.N_set:
            for i, j in A[p][n]:
                consumption = energy_consumption.get((p, i, j), 0)
                # Before the first swap the vehicle starts with initial SOC n.
                # After each intermediate BSS departure it has just received a
                # full battery, so the start SOC for non-origin arcs is N_soc.
                start_soc = n if i == source[p] else config.N_soc
                rho[(n, p, i, j)] = max(0, min(config.N_soc, start_soc - consumption))

    fA: dict[KeyFA, float] = {}
    if os.path.exists(config.demand_a_file):
        with open(config.demand_a_file, "r", encoding="utf-8") as file:
            for key_str, value in json.load(file).items():
                n, p, t = _literal_tuple(key_str)
                # The reservation file contains all tuple keys, but this MPC
                # horizon only keeps t in T_set and n >= 2 by assumption.
                if int(n) in config.reservation_soc_set and int(t) in config.T_set:
                    fA[(int(n), str(p), int(t))] = float(value)

    fR: dict[KeyFR, float] = {}
    if os.path.exists(config.demand_r_file):
        with open(config.demand_r_file, "r", encoding="utf-8") as file:
            for key_str, value in json.load(file).items():
                n, i, t = _literal_tuple(key_str)
                # Walk-in demand may be provided by data or replaced by
                # random_test/RL simulation.  Keep only low-SOC demand classes.
                if int(n) in config.random_soc_set and int(t) in config.T_set:
                    fR[(int(n), str(i), int(t))] = float(value)

    return {
        "path_data_raw": path_data_raw,
        "path_set": path_set,
        "dist": dist,
        "tau": tau,
        "tau_max": tau_max,
        "fA": fA,
        "fR": fR,
        "A": A,
        "Sp": Sp,
        "source": source,
        "dest": dest,
        "energy_consumption": energy_consumption,
        "rho": rho,
    }


def default_initial_inventory(cfg: MPCConfig, full_per_station: int = 21) -> dict[KeyWInitial, float]:
    return {
        (n, i): float(full_per_station if n == cfg.N_soc else 0)
        for n in cfg.N_set
        for i in cfg.stations
    }


def default_terminal_inventory(cfg: MPCConfig, min_full_per_station: int = 2) -> dict[str, float]:
    return {i: float(min_full_per_station) for i in cfg.stations}


def default_charging_power(cfg: MPCConfig, increment: int = 1) -> dict[KeyP, float]:
    # Placeholder actor action: every non-full battery gains one SOC level per
    # period.  A real RL actor should replace this dictionary.
    return {
        (n, i, t): float(0 if n == cfg.N_soc else increment)
        for n in cfg.N_set
        for i in cfg.stations
        for t in cfg.T_set
    }


def default_terminal_value(cfg: MPCConfig) -> dict[KeyLambda, float]:
    # Placeholder critic gradient.  With all zeros, the MPC ignores terminal
    # value and relies only on the optional terminal safety-stock constraint.
    return {(i, n): 0.0 for i in cfg.stations for n in cfg.N_set}


def default_electricity_price(cfg: MPCConfig) -> dict[tuple[str, int], float]:
    return {(i, t): float(cfg.E[t % len(cfg.E)]) for i in cfg.stations for t in cfg.T_set}


def prepare_mpc_inputs(
    data: dict,
    cfg: MPCConfig,
    *,
    fA: Optional[Mapping[KeyFA, float]] = None,
    fR: Optional[Mapping[KeyFR, float]] = None,
    w_initial: Optional[Mapping[KeyWInitial, float]] = None,
    w_terminal_min: Optional[Mapping[str, float]] = None,
    P: Optional[Mapping[KeyP, float]] = None,
    lambda_terminal: Optional[Mapping[KeyLambda, float]] = None,
    historical_yA: Optional[Mapping[KeyHistoricalYA, float]] = None,
    electricity_price: Optional[Mapping[tuple[str, int], float]] = None,
) -> dict:
    """Merge defaults with externally supplied RL/MPC inputs.

    This is the intended handoff boundary between an RL environment and MPC:
    actor -> P, critic -> lambda_terminal, simulator/state -> inventory and
    demand.  Any missing field falls back to a deterministic placeholder.
    """

    return {
        "fA": dict(data.get("fA", {}) if fA is None else fA),
        "fR": dict(data.get("fR", {}) if fR is None else fR),
        "w_initial": dict(default_initial_inventory(cfg) if w_initial is None else w_initial),
        "w_terminal_min": dict(default_terminal_inventory(cfg) if w_terminal_min is None else w_terminal_min),
        "P": dict(default_charging_power(cfg) if P is None else P),
        "lambda_terminal": dict(default_terminal_value(cfg) if lambda_terminal is None else lambda_terminal),
        "historical_yA": dict({} if historical_yA is None else historical_yA),
        "electricity_price": dict(default_electricity_price(cfg) if electricity_price is None else electricity_price),
    }


def _charging_target_soc(cfg: MPCConfig, m: int, p_value: float) -> int:
    # P is treated as a discrete SOC increment in the current implementation.
    # If later P is a physical kW value, convert it to an SOC increment before
    # passing it into BuildMPC.
    return max(0, min(cfg.N_soc, m + int(round(p_value))))


def _charging_indicator(cfg: MPCConfig, P: Mapping[KeyP, float], m: int, i: str, t: int, r: int) -> int:
    target = _charging_target_soc(cfg, m, float(P.get((m, i, t), 0.0)))
    return 1 if target == r else 0


def _reservation_current_arrivals(data: dict, cfg: MPCConfig, yA: dict, i: str, t: int) -> LinExpr:
    """Q_i,t^{A,cur}: current-horizon reservation swaps arriving at station i."""
    expr = LinExpr()
    for p in data["path_set"]:
        for n in cfg.reservation_soc_set:
            for j, ii in data["A"][p][n].select("*", i):
                if ii != i:
                    continue
                departure = t - data["tau"].get((p, i), 0)
                key = (n, p, departure, j, i)
                if departure in cfg.T_set and key in yA:
                    expr.add(yA[key])
    return expr


def _reservation_current_arrivals_with_rho(
    data: dict, cfg: MPCConfig, yA: dict, i: str, t: int, r: int
) -> LinExpr:
    """Current reservation arrivals at station i that return a battery with SOC r."""
    expr = LinExpr()
    for p in data["path_set"]:
        for n in cfg.reservation_soc_set:
            for j, ii in data["A"][p][n].select("*", i):
                if ii != i or data["rho"].get((n, p, j, i), -1) != r:
                    continue
                departure = t - data["tau"].get((p, i), 0)
                key = (n, p, departure, j, i)
                if departure in cfg.T_set and key in yA:
                    expr.add(yA[key])
    return expr


def _historical_arrivals(
    data: dict,
    cfg: MPCConfig,
    historical_yA: Mapping[KeyHistoricalYA, float],
    i: str,
    t: int,
    r: Optional[int] = None,
) -> float:
    """Q_i,t^{A,his}: already-dispatched reservation swaps arriving this horizon."""
    total = 0.0
    for (n, p, departure, j, ii), value in historical_yA.items():
        if ii != i:
            continue
        if departure >= 0:
            continue
        if departure + data["tau"].get((p, i), 0) != t:
            continue
        if r is not None and data["rho"].get((n, p, j, i), -1) != r:
            continue
        total += float(value)
    return total


def BuildMPC(data: dict, cfg: MPCConfig, mpc_inputs: Optional[dict] = None) -> Model:
    """Build the Gurobi model for one MPC horizon.

    Decision variables:
    yA[n,p,t,i,j]  reservation flow on feasible arc (i,j)
    yR[n,i,t]      number of walk-in users served at station i
    w[n,i,t]       inventory of batteries with SOC n at inventory state point t
    """

    inputs = prepare_mpc_inputs(data, cfg) if mpc_inputs is None else prepare_mpc_inputs(data, cfg, **mpc_inputs)
    fA: Mapping[KeyFA, float] = inputs["fA"]
    fR: Mapping[KeyFR, float] = inputs["fR"]
    w_initial: Mapping[KeyWInitial, float] = inputs["w_initial"]
    w_terminal_min: Mapping[str, float] = inputs["w_terminal_min"]
    P: Mapping[KeyP, float] = inputs["P"]
    lambda_terminal: Mapping[KeyLambda, float] = inputs["lambda_terminal"]
    historical_yA: Mapping[KeyHistoricalYA, float] = inputs["historical_yA"]
    electricity_price: Mapping[tuple[str, int], float] = inputs["electricity_price"]

    model = Model("MPC")

    # yA follows the SOC-wise expanded network.  Each positive path flow is
    # repeated on every arc of the chosen route, so a single served user may
    # appear as several positive yA arc variables in the solution JSON.
    yA = {}
    for p in data["path_set"]:
        for n in cfg.reservation_soc_set:
            for t in cfg.T_set:
                for i, j in data["A"][p][n]:
                    yA[(n, p, t, i, j)] = model.addVar(
                        lb=0.0, vtype=GRB.INTEGER, name=f"yA[{n},{p},{t},{i},{j}]"
                    )

    yR = {}
    for n in cfg.random_soc_set:
        for i in cfg.stations:
            for t in cfg.T_set:
                yR[(n, i, t)] = model.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"yR[{n},{i},{t}]")

    w = {}
    for n in cfg.N_set:
        for i in cfg.stations:
            for t in cfg.W_set:
                w[(n, i, t)] = model.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"w[{n},{i},{t}]")

    model.update()

    # Reservation flow constraints, paper eq. (flow):
    # 1) flow leaving the origin cannot exceed reservation demand fA;
    # 2) every intermediate station conserves flow, producing a complete path.
    for p in data["path_set"]:
        origin = data["source"][p]
        for n in cfg.reservation_soc_set:
            for t in cfg.T_set:
                outgoing = data["A"][p][n].select(origin, "*")
                if outgoing:
                    model.addConstr(
                        quicksum(yA[(n, p, t, origin, j)] for _, j in outgoing) <= fA.get((n, p, t), 0.0),
                        name=f"reservation_demand[{n},{p},{t}]",
                    )
                for i in data["Sp"][p]:
                    in_arcs = data["A"][p][n].select("*", i)
                    out_arcs = data["A"][p][n].select(i, "*")
                    if in_arcs or out_arcs:
                        model.addConstr(
                            quicksum(yA[(n, p, t, ii, jj)] for ii, jj in out_arcs)
                            == quicksum(yA[(n, p, t, ii, jj)] for ii, jj in in_arcs),
                            name=f"flow_balance[{n},{p},{t},{i}]",
                        )

    served_reservation = LinExpr()
    reservation_demand_total = 0.0
    for p in data["path_set"]:
        origin = data["source"][p]
        for n in cfg.reservation_soc_set:
            for t in cfg.T_set:
                reservation_demand_total += float(fA.get((n, p, t), 0.0))
                for _, j in data["A"][p][n].select(origin, "*"):
                    served_reservation.add(yA[(n, p, t, origin, j)])

    if reservation_demand_total > 0 and cfg.eta > 0:
        # Soft priority for reservation users: the objective penalizes unserved
        # reservations, and this hard constraint enforces a minimum service rate.
        model.addConstr(
            served_reservation >= cfg.eta * reservation_demand_total,
            name="reservation_service_rate",
        )

    # Boundary condition w[n,i,0] = observed inventory at the start of the
    # rolling horizon.
    for n in cfg.N_set:
        for i in cfg.stations:
            model.addConstr(w[(n, i, 0)] == w_initial.get((n, i), 0.0), name=f"initial_w[{n},{i}]")

    # Inventory transition and full-battery capacity by period.
    # Timing convention:
    #   w[...,t] is inventory at the start of period t;
    #   charging with P[...,t] produces available_full during period t;
    #   user swaps during period t consume full batteries and return low-SOC
    #   batteries, yielding w[...,t+1].
    for t in cfg.T_set:
        for i in cfg.stations:
            # F_i,t in the paper: full batteries available after period-t
            # charging transition, before serving current swaps.
            available_full = quicksum(
                w[(m, i, t)] * _charging_indicator(cfg, P, m, i, t, cfg.N_soc) for m in cfg.N_set
            )
            q_cur = _reservation_current_arrivals(data, cfg, yA, i, t)
            q_his = _historical_arrivals(data, cfg, historical_yA, i, t)
            q_random = quicksum(yR[(n, i, t)] for n in cfg.random_soc_set)

            # Physical capacity: all users swapping at i,t share the same pool
            # of available full batteries.
            model.addConstr(q_cur + q_his + q_random <= available_full, name=f"full_capacity[{i},{t}]")

            for r in cfg.N_set:
                charged_to_r = quicksum(
                    w[(m, i, t)] * _charging_indicator(cfg, P, m, i, t, r) for m in cfg.N_set
                )
                if r == cfg.N_soc:
                    # Full SOC inventory: charged batteries that become full,
                    # minus every reservation/walk-in swap that takes one full
                    # battery from station i in period t.
                    model.addConstr(
                        w[(r, i, t + 1)] == charged_to_r - q_cur - q_his - q_random,
                        name=f"transition_full[{i},{t}]",
                    )
                else:
                    # Non-full SOC inventory: returned batteries with SOC r,
                    # plus existing station batteries that charge into SOC r.
                    returned_cur = _reservation_current_arrivals_with_rho(data, cfg, yA, i, t, r)
                    returned_his = _historical_arrivals(data, cfg, historical_yA, i, t, r)
                    returned_random = yR[(r, i, t)] if r in cfg.random_soc_set else 0.0
                    model.addConstr(
                        w[(r, i, t + 1)] == returned_cur + returned_his + returned_random + charged_to_r,
                        name=f"transition_soc[{r},{i},{t}]",
                    )

    for n in cfg.random_soc_set:
        for i in cfg.stations:
            for t in cfg.T_set:
                # Walk-in service is optional but cannot exceed exogenous demand.
                model.addConstr(yR[(n, i, t)] <= fR.get((n, i, t), 0.0), name=f"random_bound[{n},{i},{t}]")

    if cfg.require_terminal_inventory:
        for i in cfg.stations:
            # Optional safety stock: RL terminal value can shape all SOC states,
            # while this constraint only guarantees enough full batteries.
            model.addConstr(
                w[(cfg.N_soc, i, cfg.TimePeriods)] >= w_terminal_min.get(i, 0.0),
                name=f"terminal_full[{i}]",
            )

    # Objective: max I - C1 - C2 + beta * Phi_RL.
    income = LinExpr()
    for p in data["path_set"]:
        for n in cfg.reservation_soc_set:
            for t in cfg.T_set:
                for j, i in data["A"][p][n]:
                    if i in cfg.stations:
                        # The service fee is charged per replenished SOC unit.
                        # rho is the returned SOC when the user reaches station i.
                        gain = cfg.N_soc - data["rho"].get((n, p, j, i), cfg.N_soc)
                        income.add(yA[(n, p, t, j, i)] * gain * cfg.service_fee)
    for n in cfg.random_soc_set:
        for i in cfg.stations:
            for t in cfg.T_set:
                income.add(yR[(n, i, t)] * (cfg.N_soc - n) * cfg.service_fee)

    penalty = cfg.alpha * (reservation_demand_total - served_reservation)

    charging_cost = LinExpr()
    for t in cfg.T_set:
        for i in cfg.stations:
            price = float(electricity_price.get((i, t), cfg.E[t % len(cfg.E)]))
            for n in cfg.N_set:
                # Charging cost uses start-of-period inventory and exogenous
                # charging action P, as in C2 of the paper.
                charging_cost.add(w[(n, i, t)] * float(P.get((n, i, t), 0.0)) * price)

    # Linearized RL critic value at horizon end.  lambda_terminal is fixed
    # before solving, so the term remains linear.
    terminal_value = quicksum(
        float(lambda_terminal.get((i, n), 0.0)) * w[(n, i, cfg.TimePeriods)]
        for i in cfg.stations
        for n in cfg.N_set
    )

    model.setObjective(income - penalty - charging_cost + cfg.beta_terminal * terminal_value, GRB.MAXIMIZE)

    model._yA = yA
    model._yR = yR
    model._w = w
    model._mpc_inputs = inputs
    model._objective_terms = {
        "reservation_demand_total": reservation_demand_total,
        "served_reservation": served_reservation,
        "income": income,
        "penalty": penalty,
        "charging_cost": charging_cost,
        "terminal_value": terminal_value,
    }
    model.update()
    return model


def extract_solution(model: Model, data: dict, config: MPCConfig) -> Optional[dict]:
    """Extract a compact, JSON-friendly solution."""

    if model.SolCount == 0:
        print(f"No feasible solution available. Solver status: {model.status}")
        return None

    solution = {
        "status": int(model.status),
        "objective_value": float(model.ObjVal),
        "runtime": float(model.Runtime),
        "gap": float(model.MIPGap) if model.IsMIP and model.SolCount > 0 else 0.0,
    }

    terms = getattr(model, "_objective_terms", {})
    for name, expr in terms.items():
        if isinstance(expr, (LinExpr, int, float)):
            solution[name] = float(expr.getValue() if isinstance(expr, LinExpr) else expr)

    yA_sol = {}
    for key, var in model._yA.items():
        if var.X > 1e-6:
            yA_sol[key] = float(var.X)
    solution["yA"] = yA_sol

    yR_sol = {}
    for key, var in model._yR.items():
        if var.X > 1e-6:
            yR_sol[key] = float(var.X)
    solution["yR"] = yR_sol

    w_sol = {}
    for n in config.N_set:
        for i in config.stations:
            w_sol[(n, i)] = [float(model._w[(n, i, t)].X) for t in config.W_set]
    solution["w"] = w_sol
    return solution


def _stringify_tuple_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stringify_tuple_keys(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_stringify_tuple_keys(item) for item in value]
    return value


def save_solution(solution: Optional[dict], output_dir: Optional[str] = None, filename: str = "mpc_solution.json") -> Optional[str]:
    if solution is None:
        return None

    module_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = os.path.join(module_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(_stringify_tuple_keys(solution), file, ensure_ascii=False, indent=2)
    print(f"Solution saved to: {filepath}")
    return filepath


def print_model_summary(model: Model) -> None:
    model.update()
    integer_count = sum(1 for var in model.getVars() if var.vType == GRB.INTEGER)
    print("\n" + "=" * 60)
    print("MPC Model Summary")
    print("=" * 60)
    print(f"Model name: {model.ModelName}")
    print(f"Variables: {model.NumVars}")
    print(f"Integer variables: {integer_count}")
    print(f"Constraints: {model.NumConstrs}")
    print("Objective sense: MAXIMIZE")
    print("=" * 60)


def solve_mpc(
    cfg: MPCConfig,
    *,
    data: Optional[dict] = None,
    mpc_inputs: Optional[dict] = None,
    output_file: Optional[str] = None,
) -> Optional[dict]:
    data = load_mpc_data(cfg) if data is None else data
    model = BuildMPC(data, cfg, mpc_inputs)
    model.setParam(GRB.Param.TimeLimit, cfg.gurobi_time_limit)
    model.setParam(GRB.Param.MIPGap, cfg.mip_gap)
    model.setParam(GRB.Param.OutputFlag, 1 if cfg.log_to_console else 0)
    model.optimize()
    solution = extract_solution(model, data, cfg)
    if output_file:
        save_solution(solution, filename=output_file)
    return solution


def main() -> None:
    cfg = tyro.cli(MPCConfig)
    log_path, log_file = setup_run_log_file(cfg.N_soc, cfg.num_stations)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)

    try:
        print(f"Log file: {log_path}")
        print(f"MPC Model - N_soc={cfg.N_soc}, H={cfg.TimePeriods}, stations={cfg.num_stations}")
        data = load_mpc_data(cfg)
        model = BuildMPC(data, cfg)
        print_model_summary(model)
        model.setParam(GRB.Param.TimeLimit, cfg.gurobi_time_limit)
        model.setParam(GRB.Param.MIPGap, cfg.mip_gap)
        model.setParam(GRB.Param.LogFile, log_path)
        model.optimize()
        solution = extract_solution(model, data, cfg)
        save_solution(solution)
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
