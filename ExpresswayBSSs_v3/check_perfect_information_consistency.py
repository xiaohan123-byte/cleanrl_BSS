"""Build (never solve) the fixed day under different Python hash layouts.

Compare the exact named algebra to the original frozen model, and require the
new model's ordered matrix and full reconstructed warm start to be identical.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from run_perfect_information import (ROOT, check_warm_start_parameters, digest, frozen_inputs,
                                     read_json, source_files, unpack_solution, write_json)
from src.parameters import BusinessParameters
from src.perfect_information import build_perfect_window, replay_solution
from src.perfect_information_checks import model_fingerprints
from src.perfect_information_model import solve_perfect
from src.scenario import SyntheticScenario


def build(args):
    saved = read_json(args.warm_start)
    day = saved['day_id']
    scene = SyntheticScenario.from_dict(read_json(args.dataset_dir / f'scenarios/day_{day:02d}.json'))
    params = BusinessParameters.from_dict(saved['parameter_snapshot'])
    params.solver.output_flag = 0
    check_warm_start_parameters(saved['parameter_snapshot'], params)
    plans = read_json(args.dataset_dir / f'dayahead/day_{day:02d}.json')
    window = build_perfect_window(params, scene, plans)
    warm = unpack_solution(saved['solution'])
    replay = replay_solution(params, scene, plans, window, warm)
    if args.worker_kind == 'fixed':
        result = solve_perfect(params, window, plans, build_only=True, warm_start=warm)
        fingerprints = result['diagnostics']['fingerprints']
    else:
        old_run = read_json(args.warm_start.parents[2] / 'run.json')
        source = args.warm_start.parents[2] / 'code' / old_run['run_id'] / 'src/perfect_information_model.py'
        if digest(source) != old_run['identity']['source_hashes']['src/perfect_information_model.py']:
            raise ValueError('original model snapshot hash mismatch')
        spec = importlib.util.spec_from_file_location('src._legacy_perfect_information_model', source)
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        import coptpy as cp
        original_environment = cp.Envr
        captured = {}

        class ModelProxy:
            def __init__(self, model):
                self.model = model

            def __getattr__(self, name):
                return getattr(self.model, name)

            def setObjective(self, *values, **kwargs):
                result = self.model.setObjective(*values, **kwargs)
                captured.update(model_fingerprints(self.model))
                return result

        class EnvironmentProxy:
            def __init__(self, *values, **kwargs):
                self.environment = original_environment(*values, **kwargs)

            def createModel(self, *values, **kwargs):
                return ModelProxy(self.environment.createModel(*values, **kwargs))

            def close(self):
                self.environment.close()

        with patch.object(cp, 'Envr', EnvironmentProxy):
            result = legacy.solve_perfect(params, window, plans, build_only=True)
        fingerprints = captured
    write_json(args.worker_output, dict(kind=args.worker_kind, python_hash_seed=os.environ['PYTHONHASHSEED'],
               fingerprints=fingerprints, diagnostics=result['diagnostics'], build_seconds=result['build_seconds'],
               warm_start_replay={k: v for k, v in replay.items() if k not in {'ledger', 'state_checks', 'final_state'}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=ROOT / 'data/final_real_data_v1')
    parser.add_argument('--warm-start', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker-kind', choices=['fixed', 'legacy'])
    parser.add_argument('--worker-output', type=Path)
    args = parser.parse_args()
    args.warm_start = args.warm_start.resolve()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.worker_kind:
        build(args)
        return
    before = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files()}
    inputs = frozen_inputs(args.dataset_dir)
    previous = read_json(args.warm_start.parents[2] / 'run.json')
    if previous['identity']['inputs'] != inputs:
        raise ValueError('original run uses different inputs')
    # Unchanged shared business logic is required for a source-level comparison.
    for relative, expected in previous['identity']['source_hashes'].items():
        if relative not in {'run_perfect_information.py', 'src/perfect_information_model.py'} and digest(ROOT / relative) != expected:
            raise ValueError(f'shared logic changed since the original solve: {relative}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'report.json').exists():
        raise ValueError('refusing to overwrite an existing consistency report')
    results = []
    for kind, hash_seed in [('legacy', '1'), ('legacy', '17'), ('fixed', '1'), ('fixed', '17')]:
        name = f'{kind}_hash{hash_seed}'
        destination = args.output_dir / f'{name}.json'
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--dataset-dir', str(args.dataset_dir),
                   '--warm-start', str(args.warm_start), '--output-dir', str(args.output_dir),
                   '--worker-kind', kind, '--worker-output', str(destination)]
        with (args.output_dir / f'{name}.log').open('w', encoding='utf-8') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, PYTHONHASHSEED=hash_seed),
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        results.append(read_json(destination))
        print(json.dumps(dict(build=name, state='checked', fingerprints=results[-1]['fingerprints'])), flush=True)
    fixed = [r for r in results if r['kind'] == 'fixed']
    if len({r['fingerprints']['semantic_sha256'] for r in results}) != 1:
        raise ValueError('old/new models have different named algebra')
    if fixed[0]['fingerprints'] != fixed[1]['fingerprints']:
        raise ValueError('fixed model still depends on Python hash seed')
    after = {p.relative_to(ROOT).as_posix(): digest(p) for p in source_files()}
    if before != after:
        raise ValueError('sources changed during model checks')
    saved = read_json(args.warm_start)
    report = dict(status='passed', day_id=saved['day_id'], failure_penalty=saved['parameter_snapshot']['reservation_failure_penalty'],
                  source_hashes=before, inputs=inputs, warm_start_sha256=digest(args.warm_start),
                  warm_start_objective=saved['incumbent_objective'], fixed_fingerprints=fixed[0]['fingerprints'], builds=results,
                  note='Four build-only checks, zero optimization runs; Python hash seeds are not experiment random seeds.')
    write_json(args.output_dir / 'report.json', report)
    print(json.dumps(dict(status='passed', report=str(args.output_dir / 'report.json'))), flush=True)


if __name__ == '__main__':
    main()
