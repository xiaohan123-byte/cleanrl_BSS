"""Validate the exact reformulation and diagnose its continuous relaxation.

Two LP solves and one build-only check; no full-day integer optimization here.
The mathematical equivalence argument is in PERFECT_INFORMATION_STRENGTHENING.md.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_perfect_information import ROOT, digest, frozen_inputs, read_json, source_files, unpack_solution, write_json
from src.parameters import BusinessParameters
from src.perfect_information import build_perfect_window, replay_solution
from src.perfect_information_model import solve_perfect
from src.perfect_information_strengthening import canonicalize_slots
from src.scenario import SyntheticScenario


def worker(args):
    saved = read_json(args.warm_start)
    params = BusinessParameters.from_dict(saved['parameter_snapshot'])
    params.solver.time_limit_sec = args.lp_time_limit
    params.solver.output_flag = 1
    scene = SyntheticScenario.from_dict(read_json(args.dataset_dir / f"scenarios/day_{saved['day_id']:02d}.json"))
    plans = read_json(args.dataset_dir / f"dayahead/day_{saved['day_id']:02d}.json")
    window = build_perfect_window(params, scene, plans)
    warm = unpack_solution(saved['solution'])
    replay_warm = canonicalize_slots(params, window, warm)[0] if args.formulation == 'strengthened' else warm
    replay = replay_solution(params, scene, plans, window, replay_warm)
    result = solve_perfect(params, window, plans, warm_start=warm, formulation=args.formulation,
                           build_only=args.build_only, relaxation_only=not args.build_only,
                           log_path=args.worker_output.with_suffix('.solver.log'))
    result['warm_start_replay'] = {k: v for k, v in replay.items() if k not in {'ledger', 'state_checks', 'final_state'}}
    result['python_hash_seed'] = os.environ.get('PYTHONHASHSEED')
    write_json(args.worker_output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--warm-start', type=Path, required=True)
    parser.add_argument('--baseline-audit', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/final_real_data_v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--lp-time-limit', type=float, default=180.)
    parser.add_argument('--formulation', choices=['baseline', 'strengthened', 'strengthened_compact'], default='baseline')
    parser.add_argument('--candidate', choices=['strengthened', 'strengthened_compact'], default='strengthened_compact')
    parser.add_argument('--worker-output', type=Path)
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    if args.worker_output:
        worker(args)
        return
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files()}
    inputs = frozen_inputs(args.dataset_dir)
    if inputs != read_json(args.warm_start.resolve().parents[2] / 'run.json')['identity']['inputs']:
        raise ValueError('warm start uses different data')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'report.json').exists():
        raise ValueError('refusing to overwrite a completed validation')
    results = {}
    for formulation, seed, build_only in [('baseline', '1', False), (args.candidate, '1', False), (args.candidate, '17', True)]:
        label = f'{formulation}_hash{seed}'
        destination = args.output_dir / f'{label}.json'
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--warm-start', str(args.warm_start.resolve()),
                   '--baseline-audit', str(args.baseline_audit.resolve()), '--dataset-dir', str(args.dataset_dir.resolve()),
                   '--output-dir', str(args.output_dir.resolve()), '--lp-time-limit', str(args.lp_time_limit),
                   '--formulation', formulation, '--worker-output', str(destination.resolve())]
        if build_only:
            command.append('--build-only')
        with (args.output_dir / f'{label}.log').open('w', encoding='utf-8') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONHASHSEED=seed),
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        results[label] = read_json(destination)
        print(json.dumps(dict(check=label, status=results[label]['status'], lp_bound=results[label].get('lp_bound'),
                              solve_seconds=results[label].get('solve_seconds'))), flush=True)
    if results['baseline_hash1']['diagnostics']['fingerprints'] != read_json(args.baseline_audit)['fingerprints']:
        raise ValueError('baseline model algebra/order has changed')
    selected = results[f'{args.candidate}_hash1']
    fixed = selected['diagnostics']['fingerprints']
    if fixed != results[f'{args.candidate}_hash17']['diagnostics']['fingerprints']:
        raise ValueError('strengthened model depends on Python hash layout')
    if results['baseline_hash1']['status'] != 'lp_optimal' or selected['status'] != 'lp_optimal':
        raise ValueError('LP diagnosis did not establish both optimal relaxation bounds')
    if selected['lp_bound'] > results['baseline_hash1']['lp_bound'] + 1e-4:
        raise ValueError('strengthening unexpectedly weakens the maximization LP bound')
    if hashes != {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files()}:
        raise ValueError('sources changed during validation')
    saved = read_json(args.warm_start)
    report = dict(status='passed', formulation=args.candidate, day_id=saved['day_id'],
                  failure_penalty=saved['parameter_snapshot']['reservation_failure_penalty'], source_hashes=hashes,
                  inputs=inputs, warm_start_sha256=digest(args.warm_start), fixed_fingerprints=fixed,
                  diagnostics=results, equivalence_argument='PERFECT_INFORMATION_STRENGTHENING.md',
                  note='Exact integer-feasible projection up to identical-slot relabeling; LP constraints intentionally strengthened.')
    write_json(args.output_dir / 'report.json', report)
    print(json.dumps(dict(status='passed', report=str(args.output_dir / 'report.json'))), flush=True)


if __name__ == '__main__':
    main()
