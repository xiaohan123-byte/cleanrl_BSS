"""Command-line entry point for the discrete zero-terminal-value baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.candidate_network import generate_candidate_network, save_candidate_network
from src.dayahead_plan import generate_dayahead_plan
from src.parameters import BusinessParameters
from src.rolling_runner import run_rolling_mpc
from src.scenario import generate_synthetic_scenario, load_scenario, save_scenario
from src.result_statistics import build_result_statistics, write_statistics_artifacts

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs' / 'baseline.json'
DEFAULT_RESULT_PATH = ROOT / 'outputs' / 'mpc_run_result.json'

def main(argv=None):
    parser = argparse.ArgumentParser(description='Five-minute joint MPC without terminal value.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--scenario', type=Path, help='Load a baseline scenario; its parameter snapshot is authoritative.')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--horizon', type=int, help='Prediction length in five-minute periods.')
    parser.add_argument('--periods', type=int, help='Operating length in periods (also changes generated scenario duration).')
    parser.add_argument('--time-limit', type=float, help='COPT seconds per rolling solve.')
    parser.add_argument('--solver-log', action='store_true')
    parser.add_argument('--output', type=Path, default=DEFAULT_RESULT_PATH)
    args = parser.parse_args(argv)
    if args.scenario and (args.seed is not None or args.periods is not None):
        parser.error('--seed/--periods change generated scenarios; omit them when loading --scenario')
    if args.scenario:
        scenario = load_scenario(args.scenario)
        params = BusinessParameters.from_dict(scenario.params)
    else:
        with args.config.open(encoding='utf-8-sig') as stream:
            configuration = json.load(stream)
        if args.periods is not None:
            configuration['num_periods'] = args.periods
        if args.seed is not None:
            configuration['seed'] = args.seed
        params = BusinessParameters.from_dict(configuration)
    if args.horizon is not None:
        params.horizon = args.horizon
    if args.time_limit is not None:
        params.solver.time_limit_sec = args.time_limit
    if args.solver_log:
        params.solver.output_flag = 1
    params.validate()
    if not args.scenario:
        scenario = generate_synthetic_scenario(params, params.seed)
    network = generate_candidate_network(params)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # The complete inputs are saved before solving for reproducible failures.
    save_scenario(scenario, output.with_name(output.stem + '_scenario.json'))
    save_candidate_network(network, output.with_name(output.stem + '_network.json'))
    plans = generate_dayahead_plan(params, network, scenario.initial_reservations())
    output.with_name(output.stem + '_dayahead.json').write_text(
        json.dumps(plans, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Running {params.num_periods} periods, H={params.horizon}, '
          f'{params.station.num_stations} stations; solver = COPT; terminal value = 0', flush=True)
    def progress(record):
        if (record['period'] + 1) % 12 == 0 or record['period'] + 1 == params.num_periods:
            print(f"Round {record['period'] + 1}/{params.num_periods}: "
                  f"{record['solution']['status']}, gap={record['solution']['mip_gap']:.3g}", flush=True)
    result = run_rolling_mpc(params, scenario, network, progress=progress)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    statistics = build_result_statistics(result)
    write_statistics_artifacts(statistics, output.parent, output.stem + '_statistics')
    summary = result['summary']
    print(f"Finished {len(result['rounds'])} rounds; actual net revenue={summary['total_reward']:.6f}")
    print(f'Results: {output}')
    return result

if __name__ == '__main__':
    main()
