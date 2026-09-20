"""Frozen split, artifact identities, and verified scenario access."""
from __future__ import annotations
import bootstrap
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from learned_value import digest, make_contract
from src.scenario import load_scenario
from src.parameters import BusinessParameters
from src.candidate_network import generate_candidate_network
from src.dayahead_plan import generate_dayahead_plan

atomic = bootstrap.enable_atomic_retries()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"immutable experiment identity changed: {path}")
    else:
        atomic(path, value)


def split_days(manifest):
    tests = manifest["test_day_ids_by_weekday"]
    pool = manifest["training_validation_pool"]
    rng_val = np.random.default_rng(np.random.SeedSequence([1, 320]))
    validation = [int(rng_val.choice([d for d in pool if (d-1)%7 == w])) for w in range(7)]
    rng_train = np.random.default_rng(np.random.SeedSequence([1, 321]))
    shuffled = [rng_train.permutation([d for d in pool if (d-1)%7 == w and d not in validation]).tolist() for w in range(7)]
    batches = [[d for weekday in shuffled for d in weekday[2*b:2*b+2]] for b in range(4)]
    assert len(set(tests+validation+sum(batches, []))) == 70
    assert sorted(tests+validation+sum(batches, [])) == list(range(1, 71))
    return {"test": tests, "validation": validation, "training_batches": batches}


def scenario_for(dataset, day, *, failure_penalty=None):
    dataset = Path(dataset)
    relative = f"scenarios/day_{day:02d}.json"
    manifest = read(dataset/"manifest.json")
    record = read(dataset/relative)
    if digest(record) != manifest["scenario_hashes"][relative]:
        raise ValueError(f"frozen scenario changed: day {day}")
    scenario = load_scenario(dataset/relative)
    if failure_penalty is not None:
        if failure_penalty not in (100., 200.):
            raise ValueError("this experiment supports failure penalties of 100 or 200 yuan")
        # This is a business-parameter override, not a resampling of the frozen
        # input. Apply it to both the execution scenario and optimizer parameters.
        scenario.params["reservation_failure_penalty"] = float(failure_penalty)
        scenario.validate()
    params = BusinessParameters.from_dict(scenario.params)
    params.horizon = 6
    params.solver.time_limit_sec = 120.
    params.solver.mip_gap = .0001
    params.solver.threads = 16
    params.solver.output_flag = 0
    params.validate()
    network = generate_candidate_network(params)
    plans = generate_dayahead_plan(params, network, scenario.initial_reservations())
    frozen_plans = read(dataset/f"dayahead/day_{day:02d}.json")
    if plans != frozen_plans:
        raise ValueError("day-ahead plan differs from the baseline frozen plan")
    return params, scenario, network


def load_result(directory):
    directory = Path(directory)
    status = read(directory/"status.json")
    if status["state"] != "complete":
        raise ValueError(f"incomplete trajectory: {directory}")
    for name, expected in status["artifacts"].items():
        if file_hash(directory/name) != expected:
            raise ValueError(f"artifact changed: {directory/name}")
    with gzip.open(directory/"result.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def save_result(directory, result):
    directory = Path(directory)
    path = directory/"result.json.gz"
    temporary = path.with_suffix(".gz.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as f:
        json.dump(result, f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def source_hashes(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".md", ".json"}
            and "__pycache__" not in p.parts and "scratch" not in p.parts}


def build_plan(dataset, code_hash, *, failure_penalty):
    dataset = Path(dataset)
    manifest = read(dataset/"manifest.json")
    splits = split_days(manifest)
    first = splits["training_batches"][0]
    # Obtain all seven historical profiles from training days, never test outcomes.
    params, _, _ = scenario_for(dataset, first[0], failure_penalty=failure_penalty)
    profiles = [scenario_for(dataset, first[2*w], failure_penalty=failure_penalty)[0].random_hourly_means for w in range(7)]
    from state_encoding import StateSpec
    return {"version": f"rl_h6_learned_set_seed1_failure{int(failure_penalty)}_v1", "seed": 1, "H": 6, "beta": 1,
            "business_overrides": {"reservation_failure_penalty": float(failure_penalty)},
            "source_failure_penalty": float(read(dataset/"base_config.json")["reservation_failure_penalty"]),
            "comparison_requirement": "zero-terminal comparator must use the same failure penalty; historical results with a different penalty are not comparable",
            "dataset_manifest_hash": digest(manifest), "code_hash": code_hash, "split": splits,
            "rollouts": {"training": 56, "validation": 28, "test": 7, "total": 91},
            "contract": make_contract(params, profiles), "state_schema": StateSpec(params).schema(),
            "solver": {"time_limit_sec": 120, "mip_gap": .0001, "threads": 16},
            "training": {"iterations": 4, "epochs": 200, "batch_size": 256, "lr": .001,
                         "optimizer": "Adam", "weight_decay": 0, "device": "cuda",
                         "normalization": "first training batch only; frozen thereafter",
                         "targets": "full undiscounted MC including drain; absorbing state zero",
                         "warm_start": "previous weights; reset optimizer; current batch only"},
            "selection": "maximum seven-day validation mean actual profit; ties choose earlier iteration",
            "data_description": "Poisson synthetic scenarios based on real observed means",
            "post_launch": "no assistant monitoring; summarize test results only on later user request"}
