"""Two serial restricted MIPs, both seeded from the same verified incumbent.

Subproblem bounds are NEVER bounds on the unrestricted perfect-information model.
An attempted case is not resumed or rerun automatically. Run in a fresh directory.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from run_perfect_information import (ROOT, check_warm_start_parameters, digest, fingerprint,
    frozen_inputs, gzip_json, now, read_json, source_files, unpack_solution, write_json)
from src.parameters import BusinessParameters
from src.perfect_information import build_perfect_window, replay_solution
from src.perfect_information_model import solve_perfect
from src.scenario import SyntheticScenario


def compact_replay(replay):
    return {k: v for k, v in replay.items() if k not in {'ledger', 'state_checks', 'final_state'}}


def verify_fixed_decisions(initial, solution, restriction):
    if initial.paths != solution.paths:
        raise ValueError('a fixed path changed')
    old = {d.request_id: (d.station, d.period) for d in initial.services}
    new = {d.request_id: (d.station, d.period) for d in solution.services}
    if restriction == 'services' and old != new:
        raise ValueError('a fixed service event changed')
    return dict(paths_unchanged=True, service_plan_unchanged=old == new,
                added_service_requests=len(new.keys() - old.keys()),
                removed_service_requests=len(old.keys() - new.keys()),
                moved_service_requests=sum(new[k] != old[k] for k in old.keys() & new.keys()))


def worker(output, restriction):
    run = read_json(output / 'run.json')
    identity = run['identity']
    directory = output / restriction
    directory.mkdir()  # Refuse to overwrite even an incomplete attempt.
    marker = dict(run_id=run['run_id'], restriction=restriction, started_at=now(), pid=os.getpid())
    write_json(directory / 'status.json', dict(marker, state='running', phase='validate'))
    try:
        for relative, expected in identity['source_hashes'].items():
            if digest(ROOT / relative) != expected:
                raise ValueError('source snapshot changed')
        dataset = Path(identity['dataset_dir'])
        for relative, expected in identity['inputs'].items():
            if digest(dataset / relative) != expected:
                raise ValueError('frozen input changed')
        if digest(output / 'warm_start.json') != identity['warm_start']['sha256']:
            raise ValueError('warm start changed')
        saved = read_json(output / 'warm_start.json')
        scene = SyntheticScenario.from_dict(read_json(dataset / 'scenarios/day_01.json'))
        params = BusinessParameters.from_dict(scene.params)
        params.reservation_failure_penalty = identity['failure_penalty']
        check_warm_start_parameters(saved['parameter_snapshot'], params)
        params.solver.time_limit_sec = identity['time_limit_sec']
        params.solver.mip_gap = identity['mip_gap']
        params.solver.threads = identity['threads']
        params.solver.feasibility_tol = 1e-8
        params.solver.output_flag = 1
        plans = read_json(dataset / 'dayahead/day_01.json')
        window = build_perfect_window(params, scene, plans)
        initial = unpack_solution(saved['solution'])
        initial_audit = compact_replay(replay_solution(params, scene, plans, window, initial))
        write_json(directory / 'initial_check.json', initial_audit)
        write_json(directory / 'status.json', dict(marker, state='running', phase='build_and_solve'))
        result = solve_perfect(params, window, plans, warm_start=initial,
            restriction=restriction, formulation=identity['formulation'],
            root_cut_rounds=identity['root_cut_rounds'], log_path=directory / 'solver.log',
            audit_path=directory / 'model_audit.json')
        result.update(run_id=run['run_id'], day_id=1, parameter_snapshot=params.to_dict())
        write_json(directory / 'solver_result.json', result)
        if not result['has_incumbent']:
            raise ValueError('restricted optimization lost its feasible incumbent')
        write_json(directory / 'status.json', dict(marker, state='running', phase='replay'))
        solution = unpack_solution(result['solution'])
        fixed_audit = verify_fixed_decisions(initial, solution, restriction)
        replay = replay_solution(params, scene, plans, window, solution)
        if restriction == 'services':
            for metric in ['reservations_completed', 'reservation_failures', 'random_services',
                           'random_timeouts', 'income', 'reservation_failure_cost', 'mean_wait_minutes']:
                if abs(replay['metrics'][metric] - initial_audit['metrics'][metric]) > 1e-5:
                    raise ValueError(f'fixed service economics/outcome changed: {metric}')
        gzip_json(directory / 'replay.json.gz', replay)
        summary = {k: v for k, v in result.items() if k not in {'solution', 'parameter_snapshot'}}
        summary.update(audit=compact_replay(replay), fixed_decisions_check=fixed_audit,
                       has_verified_incumbent=True, finished_at=now(),
                       improvement=solution.objective - initial.objective)
        write_json(directory / 'result.json', summary)
        write_json(directory / 'status.json', dict(marker, state='finished', finished_at=now()))
    except BaseException as error:
        write_json(directory / 'status.json', dict(marker, state='failed', error=str(error),
                                                   traceback=traceback.format_exc(), finished_at=now()))
        raise


def prepare(args):
    dataset, output = args.dataset_dir.resolve(), args.output_dir.resolve()
    saved = read_json(args.warm_start)
    previous = read_json(args.warm_start.resolve().parents[2] / 'run.json')
    inputs = frozen_inputs(dataset)
    if (saved['day_id'] != 1 or not saved['has_incumbent'] or saved['run_id'] != previous['run_id']
            or previous['identity']['inputs'] != inputs):
        raise ValueError('expected a feasible solution from frozen test day 1')
    params = BusinessParameters.from_dict(read_json(dataset / 'scenarios/day_01.json')['params'])
    params.reservation_failure_penalty = 200.
    check_warm_start_parameters(saved['parameter_snapshot'], params)
    sources = sorted(set([*source_files(), Path(__file__).resolve()]))
    identity = dict(version='fixed_decision_diagnostics_v1', dataset_dir=str(dataset), inputs=inputs,
        source_hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
        seed=1, day_id=1, failure_penalty=200., time_limit_sec=args.time_limit,
        mip_gap=.01, threads=16, formulation='strengthened_compact', root_cut_rounds=8,
        restrictions=['routes', 'services'], bound_scope='restricted_subproblem_only',
        warm_start=dict(source=str(args.warm_start.resolve()), sha256=digest(args.warm_start),
                        source_run_id=saved['run_id'], objective=saved['incumbent_objective']))
    run_id = fingerprint(identity)
    if (output / 'run.json').exists():
        raise ValueError('this diagnostic directory was already attempted; use a fresh output')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.warm_start, output / 'warm_start.json')
    if digest(output / 'warm_start.json') != identity['warm_start']['sha256']:
        raise ValueError('warm start changed while freezing')
    snapshot = output / 'code'
    for relative, expected in identity['source_hashes'].items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        if digest(target) != expected:
            raise ValueError('source changed while freezing')
    run = dict(run_id=run_id, identity=identity, snapshot=str(snapshot), created_at=now(),
               environment=dict(python=sys.version, executable=sys.executable,
                                coptpy=importlib.metadata.version('coptpy'), python_hash_seed=1))
    write_json(output / 'run.json', run)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/final_real_data_v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--time-limit', type=float, default=600.)
    parser.add_argument('--worker', choices=['routes', 'services'])
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        worker(output, args.worker)
        return
    if args.warm_start is None or not 0 < args.time_limit <= 600:
        parser.error('a warm start and a time limit in (0,600] are required')
    run = prepare(args)
    write_json(output / 'status.json', dict(state='running', started_at=now(), pid=os.getpid()))
    results, failures = {}, []
    for restriction in ['routes', 'services']:
        command = [sys.executable, '-u', '-B', str(Path(run['snapshot']) / Path(__file__).name),
                   '--output-dir', str(output), '--worker', restriction]
        with (output / f'{restriction}.log').open('w', encoding='utf-8') as stream:
            completed = subprocess.run(command, cwd=run['snapshot'], stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONHASHSEED='1'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if completed.returncode:
            failures.append(restriction)
        else:
            results[restriction] = read_json(output / restriction / 'result.json')
        print(f'{restriction}: returncode={completed.returncode}', flush=True)
    write_json(output / 'report.json', dict(run_id=run['run_id'], status='failed' if failures else 'passed',
        finished_at=now(), failed_cases=failures, original_objective=run['identity']['warm_start']['objective'],
        bound_warning='All reported bounds/gaps are for restricted subproblems, not the original full problem.',
        results=results))
    write_json(output / 'status.json', dict(state='failed' if failures else 'finished', finished_at=now()))
    if failures:
        raise RuntimeError(f'diagnostic case failures: {failures}')


if __name__ == '__main__':
    main()
