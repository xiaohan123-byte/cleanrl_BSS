"""Check fixed clairvoyant route coverage without reading any MPC trajectories.

This checks the seven frozen input days and representative routing states,
not optimization performance. See PERFECT_INFORMATION_PATH_COVERAGE.md for
the structural proof and its assumptions. No solver or random draw is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from src.candidate_network import build_user_arcs, enumerate_paths, generate_candidate_network
from src.dayahead_plan import generate_dayahead_plan
from src.experiment_control import fingerprint
from src.parameters import BusinessParameters

ROOT = Path(__file__).resolve().parent
SOURCES = ("src/candidate_network.py", "src/path_state.py", "src/dayahead_plan.py",
           "src/execution.py", "src/parameters.py", "src/scenario.py", "src/forecast.py",
           "check_perfect_information_path_coverage.py")


def sequences(arcs, origin):
    return {tuple(b for _, b in path if isinstance(b, int))
            for path in enumerate_paths(arcs, origin)}


def short_legs(params, od_index, path):
    previous_node, previous_position = "entry", params.od_pairs[od_index].entry_km
    result = set()
    for station in path:
        position = params.station.positions_km[station]
        if abs(position - previous_position) < params.min_swap_spacing_km - 1e-9:
            result.add((previous_node, station))
        previous_node, previous_position = station, position
    return result


def representative_states(params, od_index, actual_soc, initial):
    """Include passed-station boundaries, open intervals, and waiting origins.

    Between two station boundaries, reachability from the last departure is
    independent of the virtual-origin position: the travelled SOC cancels.
    Spacing also uses that departure anchor, not the virtual origin.
    Empty remaining routes retire immediately in the current execution code.
    """
    od = params.od_pairs[od_index]
    direction = params.od_direction(od_index)
    cases = set()
    for path in initial:
        for k in range(len(path)):
            prefix, suffix = path[:k], path[k:]
            anchor = od.entry_km if not prefix else params.station.positions_km[prefix[-1]]
            departure_soc = actual_soc if not prefix else 1.
            target = params.station.positions_km[suffix[0]]
            points = [anchor] + [x for x in params.station.positions_km
                                if direction * (x - anchor) > 0
                                and direction * (target - x) > 0] + [target]
            points.sort(reverse=direction < 0)
            positions = points[:-1] + [(a + b) / 2 for a, b in zip(points, points[1:])]
            for position in positions:
                cases.add((prefix, suffix, anchor, departure_soc, position, "origin"))
            if prefix:
                # At the waiting station, subsequent arcs use the full battery
                # received upon service. The waiting request itself is retained.
                cases.add((prefix, suffix, anchor, departure_soc, anchor, prefix[-1]))
    return cases


def check_dataset(directory):
    started = perf_counter()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    totals = dict(users=0, initial_paths=0, reported_paths=0, states=0, suffix_paths=0)
    days, input_hashes = [], {}
    earliest_entry = float("inf")
    for day in manifest["test_day_ids_by_weekday"]:
        relative = f"scenarios/day_{day:02d}.json"
        scene = json.loads((directory / relative).read_text(encoding="utf-8"))
        digest = fingerprint(scene)
        if digest != manifest["scenario_hashes"][relative]:
            raise ValueError(f"changed frozen scenario: {relative}")
        input_hashes[relative] = digest
        plan_relative = f"dayahead/day_{day:02d}.json"
        plans = json.loads((directory / plan_relative).read_text(encoding="utf-8"))
        input_hashes[plan_relative] = fingerprint(plans)
        params = BusinessParameters.from_dict(scene["params"])
        if not params.terminal_experiment:
            raise ValueError("proof uses the terminal-experiment motion and retirement rules")
        if params.path_adjustment_penalty < 0:
            raise ValueError("removing actual adjustments must not lower profit")
        network = generate_candidate_network(params)
        public = [{k: u[k] for k in ("user_key", "od_id", "entry_time", "entry_soc")}
                  for u in scene["reservations"]]
        if generate_dayahead_plan(params, network, public) != plans:
            raise ValueError("saved initial plans differ from the declared shared day-ahead rule")
        counts = {key: 0 for key in totals}
        for user in scene["reservations"]:
            counts["users"] += 1
            key = ":".join(map(str, user["user_key"]))
            od_index = params.od_index(user["od_id"])
            od = params.od_pairs[od_index]
            actual_soc = user["actual_entry_soc"]
            actual_time = user["actual_entry_time"]
            earliest_entry = min(earliest_entry, actual_time)
            if actual_time <= 0:
                raise ValueError("free pre-entry initialization at time zero is not available")
            conservative_soc = max(.5, user["entry_soc"] - params.entry_soc_error)
            if actual_soc + 1e-9 < conservative_soc:
                raise ValueError("original plan need not be physically feasible under actual SOC")
            plan = plans[key]
            initial = sequences(build_user_arcs(network, od_index, "entry", od.entry_km,
                                actual_soc, protected_stations=plan), "entry")
            if tuple(plan) not in initial:
                raise AssertionError(f"initial protected plan lost: day={day}, user={key}")
            allowed_short = short_legs(params, od_index, plan)
            if any(not short_legs(params, od_index, path) <= allowed_short for path in initial):
                raise AssertionError("a new short leg entered the initial protected network")
            reported = sequences(build_user_arcs(network, od_index, "entry", od.entry_km,
                                 conservative_soc, protected_stations=plan), "entry")
            if not reported <= initial:
                raise AssertionError(f"pre-entry coverage fails: day={day}, user={key}")
            counts["initial_paths"] += len(initial)
            counts["reported_paths"] += len(reported)
            for prefix, suffix, anchor, departure_soc, position, origin in representative_states(
                    params, od_index, actual_soc, initial):
                soc = departure_soc - abs(position - anchor) / params.range_km
                options = sequences(build_user_arcs(network, od_index, origin, position, soc,
                                    last_swap_position_km=anchor, protected_stations=suffix), origin)
                if suffix not in options:
                    raise AssertionError("a physically feasible protected suffix was removed")
                counts["states"] += 1
                counts["suffix_paths"] += len(options)
                bad = [prefix + option for option in options if prefix + option not in initial]
                if bad:
                    raise AssertionError(f"continuation coverage fails: day={day}, user={key}, "
                                         f"position={position}, bad={bad[0]}")
        for key, value in counts.items():
            totals[key] += value
        days.append(dict(day_id=day, **counts))
        print(json.dumps(dict(day_id=day, state="passed", **counts)), flush=True)
    for name, digest in hashes.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("routing source changed during the coverage check")
    return dict(status="passed", check_kind="structural-proof cross-check; not a solver run",
                zero_terminal_trajectories_used=False, seed=manifest["seed"],
                dataset_manifest_hash=fingerprint(manifest), source_hashes=hashes,
                input_hashes=input_hashes, test_days=manifest["test_day_ids_by_weekday"],
                earliest_actual_entry_hours=earliest_entry, totals=totals, per_day=days,
                conclusion="fixed clairvoyant paths cover online realized route sequences",
                required_initialization="actual entry SOC plus the original protected day-ahead plan",
                required_objective="realized net profit; no repeated unpublished-plan penalties",
                elapsed_seconds=perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/final_real_data_v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check_dataset(args.dataset_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status=report["status"], **report["totals"])), flush=True)


if __name__ == "__main__":
    main()
