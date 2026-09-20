"""Persistent, paired experiments driven by a supervised one-day worker callback.

No solver is launched on import. A frozen plan is never reduced to fit a wall
clock deadline; incomplete work stays pending and resumes with the same inputs.
"""
from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np

from .experiment_control import (ExperimentPaused, atomic_json, check_deadline,
                                 fingerprint, seconds_remaining, utc_now)
from .experiment_metrics import mean_std
from .parameters import BusinessParameters
from .scenario import generate_synthetic_scenario
from .terminal_features import FeatureSpec
from .terminal_value import TerminalValueModel, terminal_configuration_fingerprint
from .value_training import fit_value_model, monte_carlo_samples

SCHEMA_VERSION = 1


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pilot_record(value):
    if isinstance(value, (str, Path)):
        value = _read(value)
    if not isinstance(value, dict):
        raise ValueError("pilot metrics must be dictionaries or JSON paths")
    if isinstance(value.get("metrics"), dict):
        return value["metrics"]
    if isinstance(value.get("worker_status"), dict):
        value = value["worker_status"]
    return value.get("metrics", value)


def _freeze_plan(params, pilot_metrics, deadline):
    pilots = [_pilot_record(value) for value in pilot_metrics]
    if len(pilots) != 3 or any(row.get("completed_periods") != params.num_periods for row in pilots):
        raise ValueError("three complete pilot days are required before freezing a formal plan")
    seconds, pilot_timing = [], []
    for index, row in enumerate(pilots):
        value = row.get("worker_wall_seconds")
        basis = "worker_wall_seconds"
        limitation = "recorded complete worker runtime; not a guarantee of later method runtime"
        if value is None:
            value = row.get("model_wall_seconds_total")
            basis = "model_wall_seconds_total"
            limitation = ("recorded modeling/solve time proxy only, not complete process wall time; "
                          "excludes other process overhead and any unrecorded attempt time")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("pilot metrics need finite positive worker runtime or an explicitly recorded modeling/solve proxy")
        seconds.append(float(value))
        pilot_timing.append({"pilot_index": index, "pilot_seed": 101 + index,
                             "timing_seconds": float(value), "timing_basis": basis,
                             "timing_limitation": limitation})
    quick = mean(seconds) < 30.
    budget = {"train_days_per_iteration": 6 if quick else 3,
              "validation_days": 4 if quick else 3,
              "test_days": 20 if quick else 10,
              "outer_iterations": 3 if quick else 2,
              "training_replicates": 2, "fit_epochs": 200}
    n, v, t, o, k = (budget[key] for key in ("train_days_per_iteration", "validation_days",
                                            "test_days", "outer_iterations", "training_replicates"))
    seeds = {"pilot": [101, 102, 103],
             "validation": list(range(21000, 21000 + v)),
             "test": list(range(41000, 41000 + t)),
             "training": [[[11000 + rep * 1000 + iteration * 100 + i for i in range(n)]
                           for iteration in range(o)] for rep in range(k)],
             "model_initialization": [51000, 52000]}
    flat = seeds["pilot"] + seeds["validation"] + seeds["test"]
    flat += [seed for rep in seeds["training"] for batch in rep for seed in batch]
    if len(flat) != len(set(flat)):
        raise ValueError("scenario splits overlap")
    # Core initial zero-policy trajectories can be shared across feature models.
    core_days = t + k * (2 * o * (n + v) + 2 * t - n)
    extra_joint_days = 3 * t
    inventory_days = k * (o * (n + v) + t - n)
    simple_days_upper = 5 * v + t
    horizon_days = 3 * (k + 1) * t
    other_sensitivity_days = 7 * (t + k * (o * (n + v) + t))
    upper_days = core_days + extra_joint_days + inventory_days + simple_days_upper + horizon_days + other_sensitivity_days
    remaining = seconds_remaining(deadline)
    return {"schema_version": SCHEMA_VERSION, "created_at": utc_now().isoformat(),
            "base_parameters_fingerprint": fingerprint(params.to_dict()),
            "pilot_metrics": pilots, "pilot_mean_day_seconds": mean(seconds),
            "pilot_timing": pilot_timing,
            "pilot_mean_day_seconds_basis": "mean of the per-pilot timing_seconds; see timing_basis and timing_limitation",
            "budget": budget, "seeds": seeds,
            "budget_rule": "mean selected pilot timing <30s: expanded; otherwise compact; never shrink at deadline",
            "estimated_core_trajectories": core_days,
            "estimated_all_trajectories_upper": upper_days,
            "estimated_core_seconds_at_zero_pilot_speed": core_days * mean(seconds),
            "estimated_all_seconds_at_zero_pilot_speed": upper_days * mean(seconds),
            "estimate_limitation": ("excludes training and neural-terminal overhead; timing estimate is not a completion promise; "
                                    "pilots without complete worker timing use the explicitly documented modeling/solve proxy"),
            "seconds_remaining_when_frozen": remaining if math.isfinite(remaining) else None,
            "feature_scaling": "fixed business scales only; no fitted test/validation statistics",
            "target_scale_yuan": 1000.,
            "policy_iteration": "fresh complete days from latest frozen policy; no replay of older-policy labels",
            "selection": "highest mean actual validation net profit among fitted outer candidates; test never selects",
            "reporting": "scenario-paired test differences for each training replicate; do not pool replicates as independent scenarios",
            "simple_inventory_coefficients_yuan_per_kwh": [0., .1, .25, .5, 1.],
            "sensitivity": {"horizon_periods": [24, 48, 72, 96],
                            "demand_multipliers": [.5, 1., 1.5],
                            "battery_slots": [14, 21, 28],
                            "station_power_multipliers": [.5, .75, 1., 1.25]},
            "sensitivity_training": "fixed base network for H; retrain full ReLU separately for each nonbaseline physical/demand configuration",
            "stopping": "any incomplete worker pauses the suite; same seed retained; deadline pauses without changing budget"}


class _Suite:
    def __init__(self, config_path, output_root, deadline, run_job, pilot_metrics):
        self.params = BusinessParameters.load_json(config_path)
        self.root = (Path(output_root) / "formal_suite").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = deadline.isoformat() if hasattr(deadline, "isoformat") else deadline
        self.deadline, self.run_job = deadline, run_job
        plan_path = self.root / "plan.json"
        if plan_path.exists():
            self.plan = _read(plan_path)
            if self.plan.get("schema_version") != SCHEMA_VERSION or self.plan["base_parameters_fingerprint"] != fingerprint(self.params.to_dict()):
                raise ValueError("existing suite plan does not match the supplied configuration")
        else:
            self.plan = _freeze_plan(self.params, pilot_metrics, deadline)
            atomic_json(plan_path, self.plan)
        state_path = self.root / "state.json"
        self.state = _read(state_path) if state_path.exists() else {"schema_version": SCHEMA_VERSION,
            "status": "pending", "completed_jobs": {}, "groups": {}, "phases": {}, "selected_models": {}}
        plan_digest = fingerprint(self.plan)
        if self.state.get("plan_fingerprint", plan_digest) != plan_digest:
            raise ValueError("refusing to resume with a modified frozen experiment plan")
        self.state.update(plan_fingerprint=plan_digest, last_deadline=deadline, updated_at=utc_now().isoformat())
        self._persist()

    def _persist(self):
        self.state["updated_at"] = utc_now().isoformat()
        atomic_json(self.root / "state.json", self.state)
        atomic_json(self.root / "report.json", {"schema_version": SCHEMA_VERSION,
            "status": self.state["status"], "phases": self.state["phases"],
            "budget": self.plan["budget"], "seeds": self.plan["seeds"],
            "groups": self.state["groups"], "selected_models": self.state["selected_models"],
            "reporting_rule": self.plan["reporting"]})

    def _checkpoint(self, operation):
        self.state["current_operation"] = operation
        self._persist()
        check_deadline(self.deadline)

    def _configuration(self, name, params):
        path = self.root / "configurations" / (name + ".json")
        if path.exists():
            if fingerprint(_read(path)) != fingerprint(params.to_dict()):
                raise ValueError("refusing to alter an existing sensitivity configuration")
        else:
            atomic_json(path, params.to_dict())
        return path

    def _scenario(self, configuration, params, seed):
        self._configuration(configuration, params)
        path = self.root / "scenarios" / configuration / (str(seed) + ".json")
        if path.exists():
            # Reading the parameter envelope suffices; the supervised worker
            # separately fingerprints the complete immutable scenario content.
            existing = _read(path)
            if existing["seed"] != seed or fingerprint(existing["params"]) != fingerprint(params.to_dict()):
                raise ValueError("saved scenario differs from the frozen suite inputs")
        else:
            self._checkpoint("generate " + configuration + " scenario " + str(seed))
            scenario = generate_synthetic_scenario(params, seed)
            atomic_json(path, scenario.to_dict())
        return path

    def _job(self, configuration, params, seed, method, *, model=None, variant="full",
             capture=False, route_mode="joint", charging_mode="joint", horizon=None):
        job_id = configuration + "/" + method + "/seed_" + str(seed)
        scenario_path = self._scenario(configuration, params, seed)
        output = self.root / "jobs" / job_id
        job = {"job_id": job_id, "scenario": str(scenario_path), "output_dir": str(output),
               "model": str(model) if model else None, "feature_variant": variant,
               "capture_features": bool(capture), "route_mode": route_mode, "charging_mode": charging_mode}
        if horizon is not None:
            job["horizon"] = int(horizon)
        identity = dict(job)
        identity["model_fingerprint"] = fingerprint(_read(model)) if model else None
        identity["parameter_fingerprint"] = fingerprint(params.to_dict())
        identity["scenario_fingerprint"] = fingerprint(_read(scenario_path))
        digest = fingerprint(identity)
        prior = self.state["completed_jobs"].get(job_id)
        if prior is not None:
            if prior["fingerprint"] != digest:
                raise ValueError("completed job inputs have changed: " + job_id)
            return prior
        self._checkpoint("run " + job_id)
        response = self.run_job(job)
        worker = response.get("worker_status", response)
        status = worker.get("state", worker.get("status")) if isinstance(worker, dict) else worker
        metrics = response.get("metrics") or (worker.get("metrics") if isinstance(worker, dict) else None)
        if status != "complete":
            self.state["last_incomplete_job"] = {"job": job, "worker_status": worker}
            self._persist()
            raise ExperimentPaused("suite paused at incomplete worker " + job_id + ": " + str(status))
        if not metrics or metrics.get("completed_periods") != params.num_periods:
            raise ValueError("complete worker must return reconciled full-day metrics")
        value = {"job_id": job_id, "fingerprint": digest, "scenario_seed": seed,
                 "output_dir": str(output), "feature_variant": variant, "metrics": metrics}
        self.state["completed_jobs"][job_id] = value
        self._persist()
        return value

    def _group(self, name, jobs, *, reference=None, metadata=None):
        values = [job["metrics"] for job in jobs]
        keys = sorted({key for row in values for key, value in row.items()
                       if value is None or isinstance(value, (int, float)) and not isinstance(value, bool)})
        statistics = {key: mean_std([row.get(key) for row in values]) for key in keys}
        result = {"status": "complete", "scenario_seeds": [job["scenario_seed"] for job in jobs],
                  "jobs": [job["job_id"] for job in jobs], "statistics": statistics,
                  "metadata": metadata or {}}
        if reference is not None:
            comparator = {job["scenario_seed"]: job["metrics"] for job in reference}
            if set(comparator) != set(result["scenario_seeds"]):
                raise ValueError("paired comparison requires the identical test scenario seeds")
            paired = {}
            for key in keys:
                differences = []
                for job in jobs:
                    left, right = job["metrics"].get(key), comparator[job["scenario_seed"]].get(key)
                    differences.append(None if left is None or right is None else left - right)
                paired[key] = mean_std(differences)
            result["paired_difference_from_zero"] = paired
        self.state["groups"][name] = result
        self._persist()
        return jobs

    def _test(self, configuration, params, label, model=None, variant="full", *, reference=None,
              route_mode="joint", charging_mode="joint", horizon=None, metadata=None):
        jobs = [self._job(configuration, params, seed, "test/" + label, model=model, variant=variant,
                          route_mode=route_mode, charging_mode=charging_mode, horizon=horizon)
                for seed in self.plan["seeds"]["test"]]
        return self._group(configuration + "/" + label, jobs, reference=reference, metadata=metadata)

    def _training_batch(self, configuration, params, label, replicate, iteration, model, spec):
        seeds = self.plan["seeds"]["training"][replicate][iteration]
        # Identical initial zero-policy trajectories are shared, not replayed
        # under a different policy. Inventory features are a fixed projection.
        method = "train/initial_zero" if model is None else "train/" + label + "/iteration_" + str(iteration)
        variant = "full" if model is None else spec.variant
        jobs = [self._job(configuration, params, seed, method, model=model, variant=variant, capture=True)
                for seed in seeds]
        source_spec = FeatureSpec(params, variant=variant)
        indices = [source_spec.index[name] for name in spec.names]
        features, targets = [], []
        for job in jobs:
            self._checkpoint("read complete MC labels " + job["job_id"])
            result_path = Path(job["output_dir"]) / "result.json.gz"
            manifest = _read(Path(job["output_dir"]) / "feature_manifest.json")
            if (manifest.get("variant") != variant or manifest.get("names") != source_spec.names
                    or manifest.get("dimension") != source_spec.dimension):
                raise ValueError("saved training feature manifest differs from the sampling schema")
            with gzip.open(result_path, "rt", encoding="utf-8") as stream:
                result = json.load(stream)
            expected_kind = TerminalValueModel.load(model).kind if model else "zero"
            method_record = result.get("method", {})
            if (fingerprint(result.get("parameter_snapshot")) != fingerprint(params.to_dict())
                    or method_record.get("terminal_kind") != expected_kind
                    or method_record.get("feature_variant") != variant
                    or method_record.get("route_mode") != "joint"
                    or method_record.get("charging_mode") != "joint"):
                raise ValueError("saved training trajectory differs from the current configuration or sampled policy")
            x, y = monte_carlo_samples(result)
            if x.shape[1] != source_spec.dimension:
                raise ValueError("saved training feature matrix has an incompatible fixed schema")
            features.append(x[:, indices])
            targets.append(y)
        return np.concatenate(features), np.concatenate(targets), [job["job_id"] for job in jobs]

    def _learn(self, configuration, params, kind, variant, replicate, reference):
        label = kind + "_" + variant + "_rep" + str(replicate)
        model_dir = self.root / "models" / configuration / label
        spec = FeatureSpec(params, variant=variant)
        spec_path = model_dir / "feature_spec.json"
        if spec_path.exists():
            if fingerprint(_read(spec_path)) != fingerprint(spec.to_dict()):
                raise ValueError("saved fixed feature specification differs from the current schema")
        else:
            atomic_json(spec_path, spec.to_dict())
        previous_model, previous_path = None, None
        candidates = []
        budget = self.plan["budget"]
        for iteration in range(budget["outer_iterations"]):
            model_path = model_dir / ("iteration_" + str(iteration) + ".json")
            fit_path = model_dir / ("iteration_" + str(iteration) + "_fit.json")
            if model_path.exists() and fit_path.exists():
                current = TerminalValueModel.load(model_path)
                fit = _read(fit_path)
                if (current.kind != kind or current.variant != variant
                        or current.configuration_fingerprint != terminal_configuration_fingerprint(params)
                        or current.feature_names != spec.names
                        or fit.get("sampled_policy_fingerprint") != (fingerprint(previous_model.to_dict()) if previous_model else None)):
                    raise ValueError("saved fit does not match model kind, configuration, policy or feature schema")
            else:
                x, y, training_jobs = self._training_batch(configuration, params, label, replicate, iteration, previous_path, spec)
                self._checkpoint("fit " + configuration + "/" + label + "/iteration_" + str(iteration))
                seed = self.plan["seeds"]["model_initialization"][replicate] + iteration
                current, diagnostics = fit_value_model(x, y, spec, kind=kind,
                    previous=previous_model if kind == "relu" else None, seed=seed,
                    epochs=budget["fit_epochs"], deadline=self.deadline)
                fit = {"diagnostics": diagnostics, "training_jobs": training_jobs,
                       "iteration": iteration, "training_replicate": replicate,
                       "sampled_policy_fingerprint": fingerprint(previous_model.to_dict()) if previous_model else None,
                       "only_current_policy_labels": True}
                atomic_json(model_path, current.to_dict())
                atomic_json(fit_path, fit)
            validation_jobs = [self._job(configuration, params, seed,
                "validation/" + label + "/iteration_" + str(iteration), model=model_path, variant=variant)
                for seed in self.plan["seeds"]["validation"]]
            score = mean(job["metrics"]["net_profit_yuan"] for job in validation_jobs)
            candidate = {"iteration": iteration, "model": str(model_path), "validation_net_profit_yuan": score,
                         "validation_jobs": [job["job_id"] for job in validation_jobs]}
            candidates.append(candidate)
            atomic_json(model_dir / ("iteration_" + str(iteration) + "_validation.json"), candidate)
            previous_model, previous_path = current, model_path
        # Stable tie breaking selects the earlier candidate; no test metrics enter.
        best = max(candidates, key=lambda row: row["validation_net_profit_yuan"])
        selection = {"kind": kind, "variant": variant, "training_replicate": replicate,
                     "candidates": candidates, "selected": best, "test_used_for_selection": False}
        atomic_json(model_dir / "selection.json", selection)
        self.state["selected_models"][configuration + "/" + label] = selection
        self._persist()
        jobs = self._test(configuration, params, label, Path(best["model"]), variant,
                          reference=reference, metadata={"training_replicate": replicate,
                          "selected_iteration": best["iteration"], "kind": kind, "variant": variant})
        return Path(best["model"]), jobs

    def core(self):
        zero = self._test("base", self.params, "zero")
        models = {}
        for kind in ("linear", "relu"):
            for replicate in range(self.plan["budget"]["training_replicates"]):
                model, _ = self._learn("base", self.params, kind, "full", replicate, zero)
                models[(kind, replicate)] = model
        self.state["phases"]["core"] = "complete"
        self._persist()
        return zero, models

    def _simple_inventory(self, zero):
        candidates = []
        for coefficient in self.plan["simple_inventory_coefficients_yuan_per_kwh"]:
            label = "simple_inventory_" + format(coefficient, "g").replace(".", "p")
            path = self.root / "models" / "base" / (label + ".json")
            model = TerminalValueModel(kind="simple_inventory", variant="simple_inventory",
                                       inventory_coefficient=coefficient,
                                       configuration_fingerprint=terminal_configuration_fingerprint(self.params))
            atomic_json(path, model.to_dict())
            jobs = [self._job("base", self.params, seed, "validation/" + label,
                             model=path if coefficient else None)
                    for seed in self.plan["seeds"]["validation"]]
            candidates.append({"coefficient_yuan_per_kwh": coefficient, "model": str(path),
                               "validation_net_profit_yuan": mean(job["metrics"]["net_profit_yuan"] for job in jobs),
                               "validation_jobs": [job["job_id"] for job in jobs]})
        best = max(candidates, key=lambda row: row["validation_net_profit_yuan"])
        selection = {"candidates": candidates, "selected": best, "test_used_for_selection": False}
        atomic_json(self.root / "models/base/simple_inventory_selection.json", selection)
        self.state["selected_models"]["base/simple_inventory"] = selection
        if best["coefficient_yuan_per_kwh"] == 0:
            self._group("base/simple_inventory", zero, reference=zero, metadata=best)
        else:
            self._test("base", self.params, "simple_inventory", best["model"], reference=zero, metadata=best)

    def _changed_configuration(self, factor, value):
        params = copy.deepcopy(self.params)
        if factor == "demand":
            count = params.num_reservations * value
            if not float(count).is_integer():
                raise ValueError("demand sensitivity must preserve an integer fixed reservation total")
            params.num_reservations = int(count)
            params.random_hourly_means = [[v * value for v in row] for row in params.random_hourly_means]
            params.random_arrival_rate_per_hour = [v * value for v in params.random_arrival_rate_per_hour]
            params.source_metadata["random_daily_expected_count"] = sum(map(sum, params.random_hourly_means))
            params.source_metadata["reservation_daily_count"] = params.num_reservations
        elif factor == "battery":
            params.station.num_slots = int(value)
            params.station.initial_slot_soc = [[1.] * int(value) for _ in params.station.station_ids]
            params.station.slot_power_limits_kw = [[row[0]] * int(value) for row in params.station.slot_power_limits_kw]
        elif factor == "power":
            params.station.station_power_limits_kw = [v * value for v in params.station.station_power_limits_kw]
        else:
            raise ValueError("unknown sensitivity factor")
        params.source_metadata["sensitivity"] = {"factor": factor, "value": value}
        params.validate()
        return params

    def paper(self, zero, core_models):
        for route in ("dayahead", "joint"):
            for charging in ("baseline", "joint"):
                label = "joint_ablation_" + route + "_" + charging
                if route == "joint" and charging == "joint":
                    self._group("base/" + label, zero, reference=zero)
                else:
                    self._test("base", self.params, label, reference=zero,
                               route_mode=route, charging_mode=charging)
        self.state["phases"]["joint_ablation"] = "complete"
        self._persist()
        self._simple_inventory(zero)
        for replicate in range(self.plan["budget"]["training_replicates"]):
            self._learn("base", self.params, "relu", "inventory_only", replicate, zero)
        self.state["phases"]["terminal_ablation"] = "complete"
        self._persist()
        for horizon in self.plan["sensitivity"]["horizon_periods"]:
            label = "horizon_" + str(horizon)
            zero_h = zero if horizon == self.params.horizon else self._test("base", self.params, label + "_zero", horizon=horizon)
            if horizon == self.params.horizon:
                self._group("sensitivity/" + label + "_zero", zero)
            for replicate in range(self.plan["budget"]["training_replicates"]):
                model = core_models[("relu", replicate)]
                if horizon == self.params.horizon:
                    source = self.state["groups"]["base/relu_full_rep" + str(replicate)]
                    self.state["groups"]["sensitivity/" + label + "_relu_rep" + str(replicate)] = copy.deepcopy(source)
                    self._persist()
                else:
                    self._test("base", self.params, label + "_relu_rep" + str(replicate), model,
                               reference=zero_h, horizon=horizon,
                               metadata={"training_replicate": replicate, "base_model_frozen": True})
        factors = (("demand", "demand_multipliers", 1.),
                   ("battery", "battery_slots", self.params.station.num_slots),
                   ("power", "station_power_multipliers", 1.))
        for factor, key, baseline in factors:
            for value in self.plan["sensitivity"][key]:
                name = factor + "_" + format(value, "g").replace(".", "p")
                if value == baseline:
                    self._group("sensitivity/" + name + "_zero", zero)
                    for replicate in range(self.plan["budget"]["training_replicates"]):
                        source = self.state["groups"]["base/relu_full_rep" + str(replicate)]
                        self.state["groups"]["sensitivity/" + name + "_relu_rep" + str(replicate)] = copy.deepcopy(source)
                    self._persist()
                    continue
                params = self._changed_configuration(factor, value)
                zero_changed = self._test(name, params, "zero")
                for replicate in range(self.plan["budget"]["training_replicates"]):
                    self._learn(name, params, "relu", "full", replicate, zero_changed)
        self.state["phases"]["sensitivity"] = "complete"
        self._persist()


def run_formal_and_paper(config_path, output_root, deadline, run_job, pilot_metrics, stage="all"):
    """Run/resume the frozen suite through a synchronous supervised callback.

    ``run_job(job)`` returns ``worker_status`` (dict with state, or a state
    string) and full-day ``metrics``. Any non-complete worker pauses immediately.
    Stage ``plan`` only freezes the plan; ``core`` stops after the three core
    methods; ``paper``/``all`` also run the section-5 comparisons. A new deadline
    resumes the identical pending job, seed splits and model schedule.
    """
    if stage not in {"plan", "core", "paper", "all"}:
        raise ValueError("stage must be plan, core, paper or all")
    suite = _Suite(config_path, output_root, deadline, run_job, pilot_metrics)
    if stage == "plan":
        return {"status": "planned", "plan": suite.plan, "suite_directory": str(suite.root)}
    suite.state.update(status="running", requested_stage=stage)
    suite._persist()
    try:
        zero, models = suite.core()
        if stage != "core":
            suite.paper(zero, models)
        suite.state["status"] = "complete" if stage != "core" else "core_complete_paper_pending"
        suite.state.pop("last_incomplete_job", None)
        suite._persist()
        return {"status": "complete", "completed_stage": stage,
                "suite_status": suite.state["status"], "suite_directory": str(suite.root),
                "report": str(suite.root / "report.json")}
    except ExperimentPaused as exc:
        suite.state.update(status="paused", pause_reason=str(exc))
        suite._persist()
        return {"status": "paused", "reason": str(exc), "suite_directory": str(suite.root),
                "current_operation": suite.state.get("current_operation"),
                "completed_jobs": len(suite.state["completed_jobs"])}
    except Exception as exc:
        suite.state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        suite._persist()
        raise
