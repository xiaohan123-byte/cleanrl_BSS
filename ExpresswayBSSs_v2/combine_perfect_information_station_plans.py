"""Combine two identical service schedules by station; no optimization call."""
import argparse
from copy import deepcopy
from math import fsum
from pathlib import Path

from run_perfect_information import (digest, fingerprint, gzip_json, now, read_json,
                                    unpack_solution, write_json)
from run_perfect_information_diagnostics import compact_replay, verify_fixed_decisions
from src.parameters import BusinessParameters, price_at
from src.perfect_information import build_perfect_window, replay_solution
from src.scenario import SyntheticScenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    run, comparison = read_json(root / 'run.json'), read_json(root / 'comparison.json')
    if comparison['status'] != 'passed':
        raise ValueError('both diagnostics must be verified first')
    saved = read_json(root / 'warm_start.json')
    old = unpack_solution(saved['solution'])
    params = BusinessParameters.from_dict(saved['parameter_snapshot'])
    new_source = root / 'services/solver_result.json'
    new = unpack_solution(read_json(new_source)['solution'])
    verify_fixed_decisions(old, new, 'services')
    def station_cost(solution):
        return [fsum(v * params.interval_hours * price_at(params, 'electricity_price', i, n)
                     for row in station for n, v in enumerate(row)) for i, station in enumerate(solution.power)]
    prior_cost, new_cost = station_cost(old), station_cost(new)
    keep_old = [i for i, (a, b) in enumerate(zip(prior_cost, new_cost)) if a < b - 1e-7]
    combined = deepcopy(new)
    for i in keep_old:
        combined.power[i], combined.soc[i] = deepcopy(old.power[i]), deepcopy(old.soc[i])
    combined.services = sorted([d for d in new.services if d.station not in keep_old] +
                               [d for d in old.services if d.station in keep_old],
                               key=lambda d: (d.period, d.station, d.slot, d.request_id))
    combined.objective_terms['charging_cost'] = sum(a if i in keep_old else b
                                                   for i, (a, b) in enumerate(zip(prior_cost, new_cost)))
    terms = combined.objective_terms
    combined.objective = terms['income'] - terms['charging_cost'] - terms['failure_cost'] - terms['adjustment_cost']
    combined.status, combined.mip_gap = 'feasible_stationwise_combination', None
    combined.solve_seconds = combined.build_seconds = combined.wall_seconds = 0.
    combined.diagnostics = dict(method='stationwise minimum cost under identical fixed service schedule',
                                stations_retaining_old_plan_zero_based=keep_old,
                                source_sha256s=[digest(root / 'warm_start.json'), digest(new_source)],
                                postprocessor_sha256=digest(Path(__file__)), extra_solver_calls=0)
    dataset = Path(run['identity']['dataset_dir'])
    for relative, expected in run['identity']['inputs'].items():
        if digest(dataset / relative) != expected:
            raise ValueError('frozen input changed')
    scene = SyntheticScenario.from_dict(read_json(dataset / 'scenarios/day_01.json'))
    plans = read_json(dataset / 'dayahead/day_01.json')
    window = build_perfect_window(params, scene, plans)
    fixed = verify_fixed_decisions(old, combined, 'services')
    replay = replay_solution(params, scene, plans, window, combined)
    export = root / 'stationwise'
    day = export / 'days/day_01'
    day.mkdir(parents=True, exist_ok=True)
    identity = dict(run['identity'], postprocessing=combined.diagnostics)
    run_id = fingerprint(identity)
    write_json(export / 'run.json', dict(run, run_id=run_id, identity=identity,
                                       created_at=now(), role='verified_feasible_solution_export'))
    result = dict(run_id=run_id, day_id=1, status=combined.status, has_incumbent=True,
        has_verified_incumbent=True, incumbent_objective=combined.objective, best_bound=None, relative_gap=None,
        bound_scope='no_bound_feasible_postprocessing', solve_seconds=0., parameter_snapshot=params.to_dict(),
        solution=combined.to_dict(), audit=compact_replay(replay), fixed_decisions_check=fixed,
        diagnostics=combined.diagnostics, improvement=combined.objective-old.objective)
    write_json(day / 'solver_result.json', result)
    summary = {k: v for k, v in result.items() if k not in {'solution', 'parameter_snapshot'}}
    write_json(day / 'result.json', summary)
    gzip_json(day / 'replay.json.gz', replay)
    print(f'Combined profit={combined.objective:.9f}; improvement={result["improvement"]:.9f}; '
          f'old-plan stations={keep_old}; replay={replay["status"]}')


if __name__ == '__main__':
    main()
