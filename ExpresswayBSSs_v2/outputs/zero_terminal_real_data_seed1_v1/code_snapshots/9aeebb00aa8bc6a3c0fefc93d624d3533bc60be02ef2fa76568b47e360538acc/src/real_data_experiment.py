"""Frozen, seed-1 scenarios derived from the supplied 11-station observations.

The 70 days are Poisson synthetic samples of observed weekday hourly means,
not 70 days of measured orders. Test truth is never passed to the forecast.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .candidate_network import generate_candidate_network, get_feasible_arcs
from .dayahead_plan import generate_dayahead_plan
from .experiment_control import atomic_json, fingerprint
from .parameters import BusinessParameters, StationParameters, ODPairParameters, SolverParameters
from .scenario import SyntheticScenario, _generate_terminal_scenario

SOURCE_FILES = (
    "poi_node_6e.csv", "distance_stations.csv", "11s_configuration_info.csv",
    "11s_elec_price_info.csv", "11s_service_fee_info.csv", "distance_6e_entrance.csv",
    "population_6e.csv", "subpaths_6e.csv", "flowVolume.csv", "demand_info.csv",
)
DATASET_VERSION = "observed_means_11s6e_poisson70_seed1_v1"
HORIZONS = (4, 5, 6, 7, 8)
SEED = 1


def rng(stream: int, day_id: int = 0):
    return np.random.default_rng(np.random.SeedSequence([SEED, stream, day_id]))


def _rows(directory, name):
    with (Path(directory) / name).open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(k is None or v is None or not v.strip() for r in rows for k, v in r.items()):
        raise ValueError(f"missing or malformed cells in {name}")
    return rows


def _by_key(rows, key, expected):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows) or set(result) != set(expected):
        raise ValueError(f"duplicate or unexpected {key} keys")
    return result


def load_inputs(directory):
    directory = Path(directory)
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    stations = [f"s{i}" for i in range(1, 12)]
    entrances = [f"e{i}" for i in range(1, 7)]
    poi = _by_key(_rows(directory, "poi_node_6e.csv"), "node_idx", stations + entrances)
    distances = _by_key(_rows(directory, "distance_stations.csv"), "node", stations)
    entries = _by_key(_rows(directory, "distance_6e_entrance.csv"), "node", entrances)
    counts = _by_key(_rows(directory, "11s_configuration_info.csv"), "node_idx", stations)
    population_rows = _rows(directory, "population_6e.csv")
    if len(population_rows) != 1 or set(population_rows[0]) != set(entrances):
        raise ValueError("population file must have exactly six entrance populations")
    population = {e: float(population_rows[0][e]) for e in entrances}
    if any(not np.isfinite(v) or v <= 0 for v in population.values()):
        raise ValueError("population must be finite and positive")
    hourly_prices = {}
    for label, name in (("electricity", "11s_elec_price_info.csv"), ("fee", "11s_service_fee_info.csv")):
        rows = _by_key(_rows(directory, name), "node_idx", stations)
        values = np.array([[float(rows[s][str(h)]) for h in range(24)] for s in stations])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("invalid hourly prices")
        hourly_prices[label] = values
    means = np.full((7, 11, 24), np.nan)
    for row in _rows(directory, "demand_info.csv"):
        i, w, h = stations.index(row["node_idx"]), int(row["weekday"]) - 1, int(row["time"])
        if not 0 <= w < 7 or not 0 <= h < 24 or not np.isnan(means[w, i, h]):
            raise ValueError("invalid or duplicate demand cell")
        if float(row["distance"]) != float(distances[stations[i]]["distance"]):
            raise ValueError("demand and network distances disagree")
        means[w, i, h] = float(row["demand"])
    if not np.isfinite(means).all() or (means < 0).any() or (means.sum(axis=(1, 2)) <= 0).any():
        raise ValueError("incomplete or invalid weekday demand grid")
    flow = _by_key(_rows(directory, "flowVolume.csv"), "time", [str(h) for h in range(24)])
    ods, raw_weights = [], []
    records = _by_key(_rows(directory, "subpaths_6e.csv"), "PATH", [f"p{i}" for i in range(1, 27)])
    for row in records.values():
        start, end = (float(entries[row[e]]["distance"]) for e in ("SOURCE", "ROOT"))
        ods.append(ODPairParameters(int(row["PATH"][1:]), start, end,
                                   [stations.index(s) for s in row["STATIONS"].split(",")]))
        raw_weights.append(population[row["SOURCE"]] * population[row["ROOT"]] / abs(end - start)**2)
    count_list = [int(counts[s]["battery number"]) for s in stations]
    forecast = means[0] * (200. / means[0].sum())
    electricity = np.repeat(hourly_prices["electricity"], 4, axis=1)
    sale = np.repeat(hourly_prices["electricity"] + hourly_prices["fee"], 4, axis=1)
    params = BusinessParameters(
        num_periods=96, interval_hours=.25, horizon=6, path_update_interval=2,
        station=StationParameters(num_stations=11, station_ids=list(range(11)),
            positions_km=[float(distances[s]["distance"]) for s in stations],
            num_slots=max(count_list), num_slots_by_station=count_list,
            initial_slot_soc=[[1.] * n for n in count_list], charging_efficiency=.95,
            slot_power_limits_kw=[[60.] * n for n in count_list], station_power_limits_kw=[960.] * 11),
        od_pairs=ods, vehicle_speed_kmh=75., range_km=300., battery_capacity_kwh=100.,
        min_exit_soc=.1, electricity_price=electricity.tolist(), swap_service_price=sale.tolist(),
        path_adjustment_penalty=10., reservation_failure_penalty=1000., min_swap_spacing_km=100.,
        max_wait_hours=.5, num_reservations=200, reservation_entry_window_hours=24.,
        reservation_entry_soc_range=[.5, 1.], report_entry_soc_range=[.5, 1.],
        random_arrival_rate_per_hour=forecast.mean(axis=1).tolist(), seed=SEED,
        solver=SolverParameters(threads=16, time_limit_sec=120., mip_gap=.0001),
        terminal_experiment=True, finish_pending_after_demand=True,
        od_sampling_weights=(np.array(raw_weights) / sum(raw_weights)).tolist(),
        reservation_hourly_weights=[float(flow[str(h)]["flow volume"]) for h in range(24)],
        random_hourly_means=forecast.tolist(), entry_time_error_hours=1/6, entry_soc_error=.05,
        travel_time_relative_error=.1, random_return_soc_range=[.1, .3], random_soc_prediction=.2,
        source_metadata={"dataset_version": DATASET_VERSION, "sources": hashes,
            "station_nodes": stations, "entrance_positions_km": {e: float(entries[e]["distance"]) for e in entrances},
            "poi": poi, "od_source_ids": list(records),
            "pricing": "sale unit price = electricity + service fee, per replenished kWh",
            "forecast": "original weekday observed means normalized to expected count 200",
            "random_daily_count": 200, "reservation_daily_count": 200})
    params.physical_node_positions_km = params.physical_nodes()
    params.validate()
    network = generate_candidate_network(params)
    if any(not get_feasible_arcs(network, i, .5) for i in range(len(ods))):
        raise ValueError("an input OD is not reachable at minimum entry SOC")
    return params, means, hashes


def make_day(base, means, poisson_counts, day_id):
    params = BusinessParameters.from_dict(base.to_dict())
    weekday = (day_id - 1) % 7
    forecast = means[weekday] * (200. / means[weekday].sum())
    params.random_hourly_means = forecast.tolist()
    params.random_arrival_rate_per_hour = forecast.mean(axis=1).tolist()
    params.source_metadata.update(day_id=day_id, weekday=weekday + 1,
                                  actual_demand_source="Poisson sampled day then fixed-total multinomial")
    reservations = _generate_terminal_scenario(params, SEED, day_id=day_id, include_random=False).reservations
    weights = np.asarray(poisson_counts, dtype=float)
    if weights.shape != (11, 24) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("invalid or zero-total sampled demand day")
    counts = rng(310, day_id).multinomial(200, (weights / weights.sum()).ravel()).reshape(11, 24)
    time_rng, soc_rng = rng(311, day_id), rng(312, day_id)
    requests = []
    for i in range(11):
        for h in range(24):
            for k in range(int(counts[i, h])):
                requests.append(dict(request_id=f"random:{day_id}:{i}:{h}:{k}", station=i,
                    arrival_time=h + float(time_rng.random()), return_soc=float(soc_rng.uniform(.1, .3))))
    scenario = SyntheticScenario(params, reservations, requests, SEED)
    plans = generate_dayahead_plan(params, generate_candidate_network(params), scenario.initial_reservations())
    return scenario, plans


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if fingerprint(json.loads(path.read_text(encoding="utf-8"))) != fingerprint(value):
            raise ValueError(f"immutable artifact differs: {path}")
    else:
        atomic_json(path, value)


def prepare_dataset(data_dir, output_dir):
    output_dir = Path(output_dir)
    base, means, hashes = load_inputs(data_dir)
    sampled = [rng(300, d).poisson(means[(d-1) % 7]) for d in range(1, 71)]
    split_rng = rng(301)
    test_days = [int(split_rng.choice(np.arange(w, 71, 7))) for w in range(1, 8)]
    artifacts = {}
    for day_id, counts in enumerate(sampled, 1):
        scenario, plans = make_day(base, means, counts, day_id)
        name = f"scenarios/day_{day_id:02d}.json"
        immutable_json(output_dir / name, scenario.to_dict())
        artifacts[name] = fingerprint(scenario.to_dict())
        immutable_json(output_dir / f"dayahead/day_{day_id:02d}.json", plans)
    dataset = {"version": DATASET_VERSION, "seed": SEED, "source_hashes": hashes,
        "description": "Poisson synthetic hourly counts based on observed weekday means",
        "axis_order": ["day", "station", "hour"], "station_nodes": [f"s{i}" for i in range(1, 12)],
        "days": [{"day_id": d, "week": (d-1)//7+1, "weekday": (d-1)%7+1,
                  "counts": sampled[d-1].tolist()} for d in range(1, 71)]}
    immutable_json(output_dir / "poisson_70days.json", dataset)
    immutable_json(output_dir / "base_config.json", base.to_dict())
    for name in SOURCE_FILES:
        contents = (Path(data_dir) / name).read_bytes()
        target = output_dir / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != contents:
            raise ValueError(f"source snapshot changed: {name}")
        if not target.exists():
            target.write_bytes(contents)
    manifest = {"version": DATASET_VERSION, "seed": SEED, "source_hashes": hashes,
        "poisson_dataset_hash": fingerprint(dataset), "scenario_hashes": artifacts,
        "test_day_ids_by_weekday": test_days,
        "training_validation_pool": [d for d in range(1, 71) if d not in test_days],
        "horizons": list(HORIZONS), "job_count": 35, "scenarios_per_horizon": 7,
        "counts_per_day": {"reservations": 200, "random": 200},
        "streams": {"poisson": 300, "test_split": 301, "random_allocation": 310,
                    "arrival_times": 311, "return_soc": 312,
                    "reservations": [110, 111, 112, 113, 120, 121, 122]},
        "generation": "NumPy SeedSequence([1, stream, day_id]); hourly counts, uniform within hour",
        "forecast": "source weekday means normalized to 200; no sampled-day future truth",
        "initial_inventory": "all 250 real batteries full; each day resets independently",
        "cleanup": "no new arrivals after hour 24; cyclic prices; no salvage or forced recharge",
        "audit": {"source_grid_complete": True, "od_reachable_at_half_soc": True,
                  "all_70_scenarios_validated": True, "all_dayahead_plans_feasible": True}}
    immutable_json(output_dir / "manifest.json", manifest)
    return manifest
