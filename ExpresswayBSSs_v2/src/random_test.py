"""
Random smoke test for the MPC model before the RL layer is implemented.

Reservation demand is loaded from data_generation_optim/output/6e_flow_100_0.30.json.
Random walk-in demand, initial inventory, charging increments, and terminal
critic values are generated locally.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Optional

from gurobipy import GRB

from mpc_model import (
    BuildMPC,
    MPCConfig,
    default_electricity_price,
    default_terminal_inventory,
    extract_solution,
    load_mpc_data,
    prepare_mpc_inputs,
    save_solution,
)


def random_initial_inventory(cfg: MPCConfig, rng: random.Random) -> dict:
    inventory = {}
    for i in cfg.stations:
        total = rng.randint(24, 36)
        weights = [rng.random() + (2.0 if n == cfg.N_soc else 0.2) for n in cfg.N_set]
        weight_sum = sum(weights)
        counts = [int(total * w / weight_sum) for w in weights]
        while sum(counts) < total:
            counts[rng.randrange(len(counts))] += 1
        for n, count in zip(cfg.N_set, counts):
            inventory[(n, i)] = float(count)
    return inventory


def random_walk_in_demand(cfg: MPCConfig, rng: random.Random, max_demand: int) -> dict:
    demand = {}
    for n in cfg.random_soc_set:
        for i in cfg.stations:
            for t in cfg.T_set:
                demand[(n, i, t)] = float(rng.randint(0, max_demand))
    return demand


def random_charging_power(cfg: MPCConfig, rng: random.Random, max_increment: int) -> dict:
    power = {}
    for n in cfg.N_set:
        for i in cfg.stations:
            for t in cfg.T_set:
                if n == cfg.N_soc:
                    power[(n, i, t)] = 0.0
                else:
                    power[(n, i, t)] = float(rng.randint(0, max_increment))
    return power


def random_terminal_value(cfg: MPCConfig, rng: random.Random, max_abs_value: float) -> dict:
    return {
        (i, n): rng.uniform(-max_abs_value, max_abs_value)
        for i in cfg.stations
        for n in cfg.N_set
    }


def write_inputs_snapshot(path: str, inputs: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serializable = {section: {str(k): v for k, v in values.items()} for section, values in inputs.items()}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random MPC smoke test.")
    parser.add_argument("--horizon", type=int, default=4, help="MPC prediction horizon.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--time-limit", type=int, default=60, help="Gurobi time limit in seconds.")
    parser.add_argument("--no-solve", action="store_true", help="Only build the model, do not optimize.")
    parser.add_argument("--output", default="random_test_solution.json", help="Solution filename under src/output.")
    parser.add_argument("--max-random-demand", type=int, default=4, help="Upper bound for generated walk-in demand.")
    parser.add_argument("--max-charge-increment", type=int, default=1, help="Upper bound for generated charging SOC increment.")
    parser.add_argument("--terminal-value-scale", type=float, default=0.0, help="Absolute bound for random RL terminal values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    cfg = MPCConfig(
        TimePeriods=args.horizon,
        gurobi_time_limit=args.time_limit,
        log_to_console=True,
    )
    data = load_mpc_data(cfg)

    mpc_inputs = prepare_mpc_inputs(
        data,
        cfg,
        fA=data["fA"],
        fR=random_walk_in_demand(cfg, rng, args.max_random_demand),
        w_initial=random_initial_inventory(cfg, rng),
        w_terminal_min=default_terminal_inventory(cfg),
        P=random_charging_power(cfg, rng, args.max_charge_increment),
        lambda_terminal=random_terminal_value(cfg, rng, args.terminal_value_scale),
        historical_yA={},
        electricity_price=default_electricity_price(cfg),
    )

    model = BuildMPC(data, cfg, mpc_inputs)
    print("Random MPC smoke test")
    print(f"  seed: {args.seed}")
    print(f"  horizon: {cfg.TimePeriods}")
    print(f"  stations: {cfg.stations}")
    print(f"  reservation demand entries: {len(mpc_inputs['fA'])}")
    print(f"  random demand entries: {len(mpc_inputs['fR'])}")
    print(f"  variables: {model.NumVars}")
    print(f"  constraints: {model.NumConstrs}")

    snapshot_path = os.path.join(os.path.dirname(__file__), "output", "random_test_inputs.json")
    write_inputs_snapshot(snapshot_path, mpc_inputs)
    print(f"  input snapshot: {snapshot_path}")

    if args.no_solve:
        print("Model built successfully; skipped solve because --no-solve was set.")
        return

    model.setParam(GRB.Param.TimeLimit, args.time_limit)
    model.setParam(GRB.Param.MIPGap, cfg.mip_gap)
    model.optimize()

    solution = extract_solution(model, data, cfg)
    output_path: Optional[str] = None
    if solution is not None:
        output_path = save_solution(solution, filename=args.output)
    print(f"  solver status: {model.status}")
    if solution is not None:
        print(f"  objective: {solution['objective_value']:.4f}")
        print(f"  solution: {output_path}")


if __name__ == "__main__":
    main()
