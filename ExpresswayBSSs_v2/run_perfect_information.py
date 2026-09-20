"""Seven serial full-day solves; persist results without automatic reporting.

Run normally to start/resume untouched pending days. --prepare-only freezes
inputs/code without solving; --report-only aggregates saved results on request.
An attempted interrupted/failed day is never automatically solved a second time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

from src.atomic_io import retry_atomic_writer
from src.domain import MPCSolution, ServiceDecision
from src.experiment_control import atomic_json, fingerprint
from src.parameters import BusinessParameters, execution_period_limit
from src.perfect_information import build_perfect_window, no_service_solution, replay_solution
from src.perfect_information_model import solve_perfect
from src.scenario import SyntheticScenario

ROOT = Path(__file__).resolve().parent
DAYS = [1, 37, 3, 4, 19, 13, 14]
write_json = retry_atomic_writer(atomic_json)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    return sorted([*ROOT.joinpath('src').glob('*.py'), Path(__file__).resolve(),
                   ROOT / 'check_perfect_information_path_coverage.py'])


def frozen_inputs(dataset):
    manifest = read_json(dataset / 'manifest.json')
    if manifest['seed'] != 1 or manifest['test_day_ids_by_weekday'] != DAYS:
        raise ValueError('dataset seed/test split differs from the agreed experiment')
    inputs = {name: digest(dataset / name) for name in ['manifest.json', 'base_config.json', 'poisson_70days.json']}
    if fingerprint(read_json(dataset / 'poisson_70days.json')) != manifest['poisson_dataset_hash']:
        raise ValueError('Poisson dataset hash mismatch')
    for name, expected in manifest['source_hashes'].items():
        relative = 'sources/' + name
        inputs[relative] = digest(dataset / relative)
        if inputs[relative] != expected:
            raise ValueError(f'frozen source data changed: {name}')
    coverage = read_json(ROOT / 'outputs/perfect_information_path_coverage/check.json')
    if coverage['dataset_manifest_hash'] != fingerprint(manifest):
        raise ValueError('dataset does not match the existing path coverage evidence')
    for day in DAYS:
        name = f'scenarios/day_{day:02d}.json'
        scene = read_json(dataset / name)
        if fingerprint(scene) != manifest['scenario_hashes'][name]:
            raise ValueError(f'frozen scene changed: {day}')
        if len(scene['reservations']) != 200 or len(scene['actual_random_requests']) != 200:
            raise ValueError('each day requires exactly 200 reservations and 200 random requests')
        p = BusinessParameters.from_dict(scene['params'])
        if (not p.terminal_experiment or not p.finish_pending_after_demand or p.interval_hours != .25
                or p.num_periods != 96 or execution_period_limit(p) != 186 or p.seed != 1):
            raise ValueError('unexpected frozen physical/time configuration')
        for relative in [name, f'dayahead/day_{day:02d}.json']:
            if fingerprint(read_json(dataset / relative)) != coverage['input_hashes'][relative]:
                raise ValueError(f'input differs from the protected-path coverage proof: {relative}')
            inputs[relative] = digest(dataset / relative)
    return inputs


def prepare(args):
    dataset, output = args.dataset_dir.resolve(), args.output_dir.resolve()
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files()}
    identity = dict(dataset_dir=str(dataset), inputs=frozen_inputs(dataset), source_hashes=sources,
                    days=args.days, solver=dict(time_limit_sec=args.time_limit, mip_gap=args.gap,
                                               threads=args.threads, absolute_gap=0., feasibility_tol=1e-8),
                    failure_penalty=args.failure_penalty, seed=1,
                    version='perfect_information_fixed_protected_routes_v1')
    run_id = fingerprint(identity)
    run_path = output / 'run.json'
    if run_path.exists():
        existing = read_json(run_path)
        if existing['run_id'] != run_id:
            raise ValueError('existing output uses different inputs/code/configuration; choose a new output directory')
        return existing
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'code' / run_id
    for relative, expected in sources.items():
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if digest(destination) != expected:
            raise ValueError('source changed while taking snapshot')
    # Recheck coverage against current source; this never solves or writes input data.
    from check_perfect_information_path_coverage import check_dataset
    coverage = check_dataset(dataset)
    write_json(output / 'coverage.json', coverage)
    record = dict(run_id=run_id, identity=identity, created_at=now(), snapshot=str(snapshot),
                  environment=dict(python=sys.version, executable=sys.executable,
                                   numpy=importlib.metadata.version('numpy'),
                                   coptpy=importlib.metadata.version('coptpy'),
                                   platform=platform.platform(), cpu=platform.processor(),
                                   logical_processors=os.cpu_count()),
                  reporting='manual, only on user request; no automatic paper changes')
    write_json(run_path, record)
    return record


def day_disposition(day_dir, run_id):
    """Only an entirely unattempted date can invoke the optimizer."""
    status_file = day_dir / 'status.json'
    if not status_file.exists():
        if day_dir.exists() and any(day_dir.iterdir()):
            raise ValueError('day artifacts exist without an attempt marker')
        return 'pending'
    status = read_json(status_file)
    if status.get('run_id') != run_id:
        raise ValueError('daily output identity mismatch')
    if status['state'] == 'finished':
        result = read_json(day_dir / 'result.json')
        if result['run_id'] != run_id:
            raise ValueError('daily result identity mismatch')
        return 'reuse'
    return 'blocked'


def gzip_json(path, value):
    pending = path.with_name(path.name + '.tmp')
    with gzip.open(pending, 'wt', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    os.replace(pending, path)


def worker(args):
    output = args.output_dir.resolve()
    run = read_json(output / 'run.json')
    identity, day = run['identity'], args.worker_day
    day_dir = output / 'days' / f'day_{day:02d}'
    if day not in identity['days'] or day_disposition(day_dir, run['run_id']) != 'pending':
        raise ValueError('refusing to repeat an attempted day')
    # Check both the frozen worker and inputs immediately before the one solve.
    for relative, expected in identity['source_hashes'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'snapshot mismatch: {relative}')
    dataset = Path(identity['dataset_dir'])
    for relative, expected in identity['inputs'].items():
        if digest(dataset / relative) != expected:
            raise ValueError(f'dataset mismatch: {relative}')
    day_dir.mkdir(parents=True, exist_ok=True)
    marker = dict(run_id=run['run_id'], day_id=day, pid=os.getpid(), started_at=now())
    write_json(day_dir / 'status.json', dict(marker, state='running', phase='prepare'))
    try:
        scene = SyntheticScenario.from_dict(read_json(dataset / f'scenarios/day_{day:02d}.json'))
        params = BusinessParameters.from_dict(scene.params)
        params.reservation_failure_penalty = float(identity['failure_penalty'])
        config = identity['solver']
        params.solver.time_limit_sec = config['time_limit_sec']
        params.solver.mip_gap = config['mip_gap']
        params.solver.threads = config['threads']
        params.solver.feasibility_tol = config['feasibility_tol']
        params.solver.output_flag = 1
        plans = read_json(dataset / f'dayahead/day_{day:02d}.json')
        window = build_perfect_window(params, scene, plans)
        initial = no_service_solution(params, window, plans)
        initial_check = replay_solution(params, scene, plans, window, initial)
        write_json(day_dir / 'initial_check.json', {k: v for k, v in initial_check.items()
                                                   if k not in {'ledger', 'state_checks', 'final_state'}})
        del initial_check
        write_json(day_dir / 'status.json', dict(marker, state='running', phase='build_and_solve'))
        result = solve_perfect(params, window, plans, log_path=day_dir / 'solver.log')
        result.update(run_id=run['run_id'], day_id=day, parameter_snapshot=params.to_dict())
        write_json(day_dir / 'solver_result.json', result)
        write_json(day_dir / 'status.json', dict(marker, state='running', phase='replay'))
        audit, verified = None, False
        if result['has_incumbent']:
            payload = dict(result['solution'])
            payload['services'] = [ServiceDecision(**v) for v in payload['services']]
            replay = replay_solution(params, scene, plans, window, MPCSolution(**payload))
            gzip_json(day_dir / 'replay.json.gz', replay)
            audit = {k: v for k, v in replay.items() if k not in {'ledger', 'state_checks', 'final_state'}}
            verified = True
        summary = {k: v for k, v in result.items() if k not in {'solution', 'parameter_snapshot'}}
        summary.update(has_verified_incumbent=verified, audit=audit, finished_at=now())
        write_json(day_dir / 'result.json', summary)
        final_state = 'finished' if result['status'] in {'optimal', 'time_limit', 'node_limit', 'iteration_limit'} else 'failed'
        write_json(day_dir / 'status.json', dict(marker, state=final_state, finished_at=now(),
                                                solver_status=result['status'], has_verified_incumbent=verified))
        print(json.dumps(dict(day_id=day, state=final_state, has_verified_incumbent=verified)), flush=True)
        return 0 if final_state == 'finished' else 1
    except BaseException as error:
        write_json(day_dir / 'status.json', dict(marker, state='failed', finished_at=now(),
                                                error=str(error), traceback=traceback.format_exc()))
        raise


def run_controller(args, run):
    output = args.output_dir.resolve()
    lock_path = output / 'controller.lock'
    with lock_path.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), started_at=now(), run_id=run['run_id']), stream)
    try:
        write_json(output / 'controller_status.json', dict(state='running', pid=os.getpid(), run_id=run['run_id'], started_at=now()))
        outcomes = []
        for day in run['identity']['days']:
            day_dir = output / 'days' / f'day_{day:02d}'
            disposition = day_disposition(day_dir, run['run_id'])
            if disposition != 'pending':
                outcomes.append(dict(day_id=day, action=disposition))
                continue
            # Logs live outside the day folder so an unstarted worker still has
            # no daily attempt artifacts. Each child releases all native memory.
            logs = output / 'worker_logs'
            logs.mkdir(exist_ok=True)
            with (logs / f'day_{day:02d}.log').open('w', encoding='utf-8') as stream:
                command = [sys.executable, '-u', '-B', str(Path(run['snapshot']) / Path(__file__).name),
                           '--worker-day', str(day), '--output-dir', str(output)]
                completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                           cwd=run['snapshot'], creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            outcomes.append(dict(day_id=day, action='attempted', returncode=completed.returncode))
        write_json(output / 'controller_status.json', dict(state='finished', run_id=run['run_id'],
                                                          finished_at=now(), outcomes=outcomes))
    except BaseException as error:
        write_json(output / 'controller_status.json', dict(state='failed', run_id=run['run_id'],
                                                          error=str(error), traceback=traceback.format_exc(), finished_at=now()))
        raise
    finally:
        lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/final_real_data_v1')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/perfect_information_real_data_seed1_v1')
    parser.add_argument('--days', type=int, nargs='+', default=DAYS)
    parser.add_argument('--time-limit', type=float, default=1800.)
    parser.add_argument('--gap', type=float, default=.0001)
    parser.add_argument('--threads', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--failure-penalty', type=float, default=200.,
                        help='reservation failure penalty overriding the frozen scenario value')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--report-only', action='store_true')
    modes.add_argument('--worker-day', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_day is not None:
        return worker(args)
    if args.report_only:
        from src.perfect_information_reporting import report
        report(args.output_dir)
        return 0
    if (len(set(args.days)) != len(args.days) or not set(args.days) <= set(DAYS)
            or args.time_limit <= 0 or not 0 <= args.gap <= 1 or args.threads < 1
            or args.failure_penalty < 0):
        parser.error('invalid experiment days or solver configuration')
    run = prepare(args)
    if args.prepare_only:
        print(json.dumps(dict(state='prepared', run_id=run['run_id'], output=str(args.output_dir.resolve()))))
        return 0
    run_controller(args, run)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
