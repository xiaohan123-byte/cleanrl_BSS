"""Charging LP diagnosis and a controlled full-model count-order comparison.

One fixed-assignment LP, two relaxation checks, then two controlled MIPs.
An extended comparison can reuse the validated common start and model audits.
No station decomposition, repeated seeds, RL, or automatic paper edits.
The two full MIPs use the very same canonicalized feasible start.
"""
import argparse
from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from run_perfect_information import (ROOT, check_warm_start_parameters, digest, fingerprint,
                                    frozen_inputs, gzip_json, now, read_json, source_files,
                                    unpack_solution, write_json)
from run_perfect_information_diagnostics import compact_replay
from src.parameters import BusinessParameters
from src.perfect_information import build_perfect_window, replay_solution
from src.perfect_information_charging import solve_fixed_assignment_charging
from src.perfect_information_model import solve_perfect
from src.perfect_information_strengthening import canonicalize_slots, slot_groups
from src.scenario import SyntheticScenario

CASES = ['charging_lp', 'baseline_lp', 'count_lp', 'baseline_mip', 'count_mip']


def check_counts(params, window, solution):
    counts = Counter((d.station, d.slot) for d in solution.services)
    for i in range(params.station.num_stations):
        for group in slot_groups(params, window, i):
            if any(counts[i, a] < counts[i, b] for a, b in zip(group, group[1:])):
                raise ValueError('whole-day service counts are not ordered')
    return [[counts[i, b] for b in range(len(station))] for i, station in enumerate(solution.power)]


def worker(output, case):
    run = read_json(output / 'run.json')
    identity = run['identity']
    directory = output / case
    directory.mkdir()
    marker = dict(run_id=run['run_id'], case=case, pid=os.getpid(), started_at=now())
    write_json(directory / 'status.json', dict(marker, state='running', phase='validate'))
    try:
        for relative, expected in identity['source_hashes'].items():
            if digest(ROOT / relative) != expected:
                raise ValueError('source snapshot changed')
        dataset = Path(identity['dataset_dir'])
        for relative, expected in identity['inputs'].items():
            if digest(dataset / relative) != expected:
                raise ValueError('frozen input changed')
        source = output / ('warm_start.json' if case == 'charging_lp' else 'common_start.json')
        expected = (identity['warm_start']['sha256'] if case == 'charging_lp'
                    else read_json(output / 'common_start_meta.json')['sha256'])
        if digest(source) != expected:
            raise ValueError('initial solution changed')
        saved = read_json(source)
        initial = unpack_solution(saved['solution'])
        scene = SyntheticScenario.from_dict(read_json(dataset / 'scenarios/day_01.json'))
        params = BusinessParameters.from_dict(scene.params)
        params.reservation_failure_penalty = 200.
        check_warm_start_parameters(saved['parameter_snapshot'], params)
        params.solver.time_limit_sec = identity['mip_time_limit'] if case.endswith('_mip') else identity['lp_time_limit']
        params.solver.mip_gap = .01
        params.solver.threads = 16
        params.solver.feasibility_tol = 1e-8
        params.solver.output_flag = 1
        plans = read_json(dataset / 'dayahead/day_01.json')
        window = build_perfect_window(params, scene, plans)
        initial_audit = compact_replay(replay_solution(params, scene, plans, window, initial))
        write_json(directory / 'initial_check.json', initial_audit)
        write_json(directory / 'status.json', dict(marker, state='running', phase='build_and_solve'))
        if case == 'charging_lp':
            result = solve_fixed_assignment_charging(params, window, initial, log_path=directory / 'solver.log')
        else:
            order = 'total_services' if case.startswith('count') else 'none'
            expected_fp = identity.get('expected_model_fingerprints', {}).get(case)
            if expected_fp is None:
                expected_fp = (identity['original_full_model_fingerprints'] if case == 'baseline_lp' else
                           read_json(output / case.replace('_mip', '_lp') / 'result.json')['diagnostics']['fingerprints']
                           if case.endswith('_mip') else None)
            result = solve_perfect(params, window, plans, warm_start=initial, formulation='strengthened_compact',
                root_cut_rounds=8, slot_order=order, relaxation_only=case.endswith('_lp'),
                expected_fingerprints=expected_fp, log_path=directory / 'solver.log',
                audit_path=directory / 'model_audit.json')
        result.update(run_id=run['run_id'], day_id=1, case=case, parameter_snapshot=params.to_dict(),
                      initial_objective=initial.objective, initial_solution_sha256=expected)
        write_json(directory / 'solver_result.json', result)
        audit, count_check = None, None
        if result.get('solution'):
            write_json(directory / 'status.json', dict(marker, state='running', phase='replay'))
            sol = unpack_solution(result['solution'])
            if case == 'charging_lp' and (sol.paths != initial.paths or sol.services != initial.services):
                raise ValueError('fixed battery assignment changed')
            replay = replay_solution(params, scene, plans, window, sol)
            gzip_json(directory / 'replay.json.gz', replay)
            audit = compact_replay(replay)
            if case == 'count_mip':
                count_check = check_counts(params, window, sol)
            if case == 'charging_lp':
                canonical, permutation = canonicalize_slots(params, window, sol, order='total_services')
                canonical_check = compact_replay(replay_solution(params, scene, plans, window, canonical))
                check_counts(params, window, canonical)
                common = dict(solution=canonical.to_dict(), parameter_snapshot=params.to_dict(), source_case=case)
                write_json(output / 'common_start.json', common)
                write_json(output / 'common_start_meta.json', dict(sha256=digest(output / 'common_start.json'),
                    objective=canonical.objective, permutation=permutation, replay=canonical_check))
        if case == 'charging_lp' and result['status'] != 'optimal':
            raise ValueError('fixed-assignment LP did not establish an optimum')
        if case.endswith('_mip') and audit is None:
            raise ValueError('full MIP did not preserve its verified incumbent')
        summary = {k: v for k, v in result.items() if k not in {'solution', 'parameter_snapshot'}}
        summary.update(audit=audit, has_verified_incumbent=audit is not None,
                       ordered_counts=count_check, finished_at=now())
        write_json(directory / 'result.json', summary)
        if case == 'count_lp':
            base = read_json(output / 'baseline_lp/result.json')
            a, b = base['diagnostics'], result['diagnostics']
            if (a['model_variables'] != b['model_variables'] or a['model_binary_variables'] != b['model_binary_variables']
                    or b['model_constraints']-a['model_constraints'] != 239 or b['total_service_order_rows'] != 239):
                raise ValueError('count-order matrix did not add exactly 239 rows and zero variables')
            if base['status'] == result['status'] == 'lp_optimal' and abs(base['lp_bound']-result['lp_bound']) > 1e-4:
                raise ValueError('LP optima differ: count ordering should only select a permutation')
        write_json(directory / 'status.json', dict(marker, state='finished', finished_at=now()))
    except BaseException as error:
        write_json(directory / 'status.json', dict(marker, state='failed', error=str(error),
                                                   traceback=traceback.format_exc(), finished_at=now()))
        raise


def comparison_sources():
    return sorted(set([*source_files(), ROOT / 'run_perfect_information_diagnostics.py',
        ROOT / 'summarize_perfect_information_count_order.py',
        *ROOT.joinpath('tests').glob('test_perfect_information*.py'), Path(__file__).resolve()]))


def freeze_run(output, identity):
    snapshot = output / 'code'
    for relative, expected in identity['source_hashes'].items():
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if digest(destination) != expected:
            raise ValueError('source changed while freezing')
    run = dict(run_id=fingerprint(identity), identity=identity, snapshot=str(snapshot), created_at=now(),
               environment=dict(python=sys.version, executable=sys.executable,
                                coptpy=importlib.metadata.version('coptpy'), python_hash_seed=1))
    write_json(output / 'run.json', run)
    return run


def validate_reuse(reference, inputs):
    previous = read_json(reference / 'run.json')
    identity = previous['identity']
    if previous['run_id'] != fingerprint(identity):
        raise ValueError('reference identity changed')
    if (read_json(reference / 'status.json')['state'] != 'finished'
            or read_json(reference / 'report.json')['status'] != 'passed'
            or read_json(reference / 'comparison.json')['status'] != 'passed'):
        raise ValueError('reference comparison is not fully validated')
    expected = dict(seed=1, day_id=1, failure_penalty=200., mip_gap=.01, threads=16,
                    formulation='strengthened_compact', root_cut_rounds=8,
                    baseline_slot_order='none', candidate_slot_order='total_services')
    if identity['inputs'] != inputs or any(identity[k] != v for k, v in expected.items()):
        raise ValueError('reference data or comparison parameters differ')
    for source in [*source_files(), ROOT / 'run_perfect_information_diagnostics.py']:
        relative = source.relative_to(ROOT).as_posix()
        if identity['source_hashes'].get(relative) != digest(source):
            raise ValueError(f'reference model or replay source changed: {relative}')
    meta = read_json(reference / 'common_start_meta.json')
    if digest(reference / 'common_start.json') != meta['sha256']:
        raise ValueError('reference common initial solution changed')
    fingerprints = {}
    for case in ['baseline_mip', 'count_mip']:
        result = read_json(reference / case / 'result.json')
        audit = read_json(reference / case / 'model_audit.json')
        if (result['run_id'] != previous['run_id'] or not result['has_verified_incumbent']
                or result['initial_solution_sha256'] != meta['sha256']
                or result['diagnostics']['fingerprints'] != audit['fingerprints']):
            raise ValueError(f'inconsistent reference evidence: {case}')
        fingerprints[case] = audit['fingerprints']
    return previous, meta, fingerprints


def prepare_reuse(args):
    reference = args.reuse_comparison.resolve()
    dataset, output = args.dataset_dir.resolve(), args.output_dir.resolve()
    inputs = frozen_inputs(dataset)
    previous, meta, fingerprints = validate_reuse(reference, inputs)
    saved = read_json(reference / 'common_start.json')
    params = BusinessParameters.from_dict(read_json(dataset / 'scenarios/day_01.json')['params'])
    params.reservation_failure_penalty = 200.
    check_warm_start_parameters(saved['parameter_snapshot'], params)
    evidence = ['run.json', 'report.json', 'comparison.json', 'common_start_meta.json',
                'baseline_mip/result.json', 'baseline_mip/model_audit.json',
                'count_mip/result.json', 'count_mip/model_audit.json']
    identity = dict(previous['identity'], version='total_count_order_extended_v1',
        dataset_dir=str(dataset), inputs=inputs, mip_time_limit=args.mip_time_limit,
        cases=['baseline_mip', 'count_mip'],
        source_hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in comparison_sources()},
        expected_model_fingerprints=fingerprints,
        warm_start=dict(source=str(reference / 'common_start.json'), sha256=meta['sha256'],
                        objective=meta['objective'], source_run_id=previous['run_id']),
        reused_comparison=dict(source=str(reference), run_id=previous['run_id'],
            evidence_hashes={name: digest(reference / name) for name in evidence},
            note='Fresh MIP searches from the same common start; no search tree continuation or repeated LP diagnostics.'))
    if output.exists() and any(output.iterdir()):
        raise ValueError('refusing to overwrite an attempted comparison')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference / 'common_start.json', output / 'common_start.json')
    if digest(output / 'common_start.json') != meta['sha256']:
        raise ValueError('initial solution changed while freezing')
    for name, expected in identity['reused_comparison']['evidence_hashes'].items():
        destination = output / 'reference_evidence' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference / name, destination)
        if digest(destination) != expected:
            raise ValueError('reference evidence changed while freezing')
    shutil.copyfile(output / 'reference_evidence/common_start_meta.json', output / 'common_start_meta.json')
    return freeze_run(output, identity)


def prepare(args):
    dataset, output = args.dataset_dir.resolve(), args.output_dir.resolve()
    saved = read_json(args.warm_start)
    previous = read_json(args.warm_start.resolve().parents[2] / 'run.json')
    inputs = frozen_inputs(dataset)
    if (saved['day_id'] != 1 or not saved['has_incumbent'] or saved['run_id'] != previous['run_id']
            or previous['identity']['inputs'] != inputs):
        raise ValueError('initial solution must use the same frozen day and data')
    params = BusinessParameters.from_dict(read_json(dataset / 'scenarios/day_01.json')['params'])
    params.reservation_failure_penalty = 200.
    check_warm_start_parameters(saved['parameter_snapshot'], params)
    original = read_json(args.original_audit)
    sources = comparison_sources()
    identity = dict(version='charging_and_total_count_order_v1', dataset_dir=str(dataset), inputs=inputs,
        source_hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
        seed=1, day_id=1, failure_penalty=200., mip_time_limit=args.mip_time_limit,
        lp_time_limit=args.lp_time_limit, mip_gap=.01, threads=16, formulation='strengthened_compact',
        root_cut_rounds=8, cases=CASES, baseline_slot_order='none', candidate_slot_order='total_services',
        original_full_model_fingerprints=original['fingerprints'],
        original_audit_sha256=digest(args.original_audit),
        warm_start=dict(source=str(args.warm_start.resolve()), sha256=digest(args.warm_start),
                        source_run_id=saved['run_id'], objective=saved['incumbent_objective']))
    if (output / 'run.json').exists():
        raise ValueError('refusing to overwrite an attempted comparison')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.warm_start, output / 'warm_start.json')
    if digest(output / 'warm_start.json') != identity['warm_start']['sha256']:
        raise ValueError('initial solution changed while freezing')
    return freeze_run(output, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/final_real_data_v1')
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--original-audit', type=Path)
    parser.add_argument('--reuse-comparison', type=Path,
                        help='Reuse a completed, validated comparison without repeating its charging/LP diagnostics')
    parser.add_argument('--prior-full-result', type=Path, help='Also run the independent final summary')
    parser.add_argument('--mip-time-limit', type=float, default=600.)
    parser.add_argument('--lp-time-limit', type=float, default=60.)
    parser.add_argument('--worker', choices=CASES)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        worker(output, args.worker)
        return
    if (not 0 < args.mip_time_limit <= 3600 or not 0 < args.lp_time_limit <= 60
            or (not args.reuse_comparison and (not args.warm_start or not args.original_audit))):
        parser.error('supply initial solution, original audit, and valid time budgets')
    run = prepare_reuse(args) if args.reuse_comparison else prepare(args)
    write_json(output / 'status.json', dict(state='running', pid=os.getpid(), started_at=now()))
    results = {}
    try:
        for case in run['identity']['cases']:
            command = [sys.executable, '-u', '-B', str(Path(run['snapshot']) / Path(__file__).name),
                       '--output-dir', str(output), '--worker', case]
            with (output / f'{case}.log').open('w', encoding='utf-8') as stream:
                completed = subprocess.run(command, cwd=run['snapshot'], stdout=stream, stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONHASHSEED='1'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if completed.returncode:
                raise RuntimeError(f'{case} failed; inspect its status and log; no automatic retry')
            results[case] = read_json(output / case / 'result.json')
            print(f'{case}: {results[case]["status"]}, solve={results[case]["solve_seconds"]:.3f}s', flush=True)
        write_json(output / 'report.json', dict(status='passed', run_id=run['run_id'], results=results,
            original_objective=run['identity']['warm_start']['objective'], finished_at=now(),
            bounds='Charging LP is conditional; both unrestricted MIP bounds apply to the original full problem.'))
        if args.prior_full_result:
            with (output / 'summary.log').open('w', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-u', '-B',
                    str(Path(run['snapshot']) / 'summarize_perfect_information_count_order.py'),
                    str(output), '--prior-full-result', str(args.prior_full_result.resolve())],
                    cwd=run['snapshot'], stdout=stream, stderr=subprocess.STDOUT, check=True,
                    env=dict(os.environ, PYTHONHASHSEED='1'), creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        write_json(output / 'status.json', dict(state='finished', finished_at=now()))
    except BaseException as error:
        write_json(output / 'status.json', dict(state='failed', error=str(error), traceback=traceback.format_exc(),
                                               finished_at=now()))
        raise


if __name__ == '__main__':
    main()
