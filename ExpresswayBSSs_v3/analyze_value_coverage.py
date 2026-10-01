"""Read-only empirical feature-range diagnostics on fixed business scales.

Training input: an NPZ with features[N,D] and Unicode feature_names[D], a
pilot-only prepare dataset directory/manifest.json, or *_fit.json / pilot-only
model JSON containing diagnostics.training_feature_range. An explicit
--training-feature-spec supplies names if an NPZ lacks them.

Queries: completed diagnostic_results.json (or its directory/one method's
solution.json), completed worker directory, or complete result.json[.gz].
A sibling completed journal is streamed instead of loading the full result.
Standalone full results require feature_names or --query-feature-spec.

This reports marginal empirical ranges only. Being inside every interval does
not establish joint-state coverage, sufficient sampling, reliable value error,
or improved operating profit. Query statistics never change any feature scale,
model weight, training sample, or selection decision. No solver/torch imports.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from src.experiment_control import atomic_json, fingerprint

LIMITATION = ("Marginal empirical training min/max only. Inside-range states are not evidence of "
              "sufficient or joint-state coverage, accurate value estimates, or improved profit. "
              "No query statistics are used for normalization, training, or model selection. "
              "Single-window and pilot-only results are diagnostics, not independent test-profit results.")
SCALE_KEYS = ("variant", "soc_thresholds", "time_limits_hours", "count_scale", "time_scale_hours",
              "inventory_count_scale", "network_length_km", "random_forecast_feature_extent")


def _json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _names(values):
    names = list(values)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature names must be nonempty strings")
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names are not allowed")
    return names


def _find_spec(directory, explicit=None):
    if explicit:
        spec = _json(explicit)
        _names(spec.get("feature_names", spec.get("names", [])))
        return spec
    for name in ("feature_spec.json", "feature_manifest.json"):
        path = Path(directory) / name
        if path.exists():
            return _json(path)
    return None


def _spec_names(spec):
    return None if spec is None else _names(spec.get("feature_names", spec.get("names", [])))


@dataclass
class TrainingRange:
    feature_names: list[str]
    minimum: np.ndarray
    maximum: np.ndarray
    samples: int
    metadata: dict
    feature_spec: dict | None = None

    def __post_init__(self):
        self.feature_names = _names(self.feature_names)
        self.minimum = np.asarray(self.minimum, dtype=float)
        self.maximum = np.asarray(self.maximum, dtype=float)
        shape = (len(self.feature_names),)
        if self.minimum.shape != shape or self.maximum.shape != shape:
            raise ValueError("training bounds do not match feature names")
        if (not np.isfinite(self.minimum).all() or not np.isfinite(self.maximum).all()
                or np.any(self.minimum > self.maximum)):
            raise ValueError("invalid/nonfinite training bounds")
        if isinstance(self.samples, bool) or int(self.samples) != self.samples or self.samples <= 0:
            raise ValueError("training range requires a positive integer sample count")
        if self.feature_spec is not None and _spec_names(self.feature_spec) != self.feature_names:
            raise ValueError("training feature specification order differs from the bound arrays")


def _range_from_matrix(matrix, names, metadata, spec=None):
    matrix = np.asarray(matrix)
    names = _names(names)
    if matrix.ndim != 2 or matrix.shape[1] != len(names) or not len(matrix):
        raise ValueError("training features must have shape [positive samples, feature names]")
    if not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all():
        raise ValueError("training features must be finite numeric values")
    return TrainingRange(names, matrix.min(axis=0), matrix.max(axis=0), len(matrix), metadata, spec)


def load_training_range(path, *, feature_spec_path=None):
    """Read only the current training batch or its saved marginal bounds."""
    path = Path(path).resolve()
    if path.is_dir() and (path / "dataset/manifest.json").exists():
        path = path / "dataset"
    if path.is_dir() or path.name == "manifest.json":
        directory = path if path.is_dir() else path.parent
        manifest = _json(directory / "manifest.json")
        if manifest.get("scope") != "pilot_only" or manifest.get("state") != "complete":
            raise ValueError("prepare input must be a complete pilot-only dataset")
        spec = _find_spec(directory, feature_spec_path)
        names = _spec_names(spec)
        if names is None:
            raise ValueError("prepared dataset lacks its fixed feature specification")
        features_path = directory / "features.npy"
        for name in ("features.npy", "feature_spec.json"):
            expected = manifest.get("artifact_sha256", {}).get(name)
            if expected is None or _sha(directory / name) != expected:
                raise ValueError("prepared training artifact checksum mismatch")
        x = np.load(features_path, mmap_mode="r", allow_pickle=False)
        try:
            result = _range_from_matrix(x, names, {"input": str(directory), "scope": "pilot_only",
                "kind": "prepared_training_matrix", "source": "current_training_batch_only",
                "generating_policy": manifest.get("generating_policy"),
                "source_manifest_sha256": _sha(directory / "manifest.json"),
                "configuration_fingerprint": manifest.get("configuration_fingerprint")}, spec)
            if result.samples != manifest["samples"] or len(names) != manifest["feature_dimension"]:
                raise ValueError("prepared manifest dimensions disagree with training features")
            return result
        finally:
            x._mmap.close()
    if path.suffix == ".npz":
        spec = _find_spec(path.parent, feature_spec_path)
        with np.load(path, allow_pickle=False) as archive:
            if "features" not in archive:
                raise ValueError("training NPZ requires a features[N,D] array")
            names = _names(archive["feature_names"].tolist()) if "feature_names" in archive else _spec_names(spec)
            if names is None:
                raise ValueError("NPZ requires Unicode feature_names or an explicit feature specification")
            scope = str(archive["scope"].item()) if "scope" in archive else "unspecified_training_batch"
            return _range_from_matrix(archive["features"], names,
                {"input": str(path), "kind": "training_npz", "scope": scope,
                 "source": "current_training_batch_only", "sha256": _sha(path)}, spec)
    value = _json(path)
    record = value.get("diagnostics", {}).get("training_feature_range")
    if record is None:
        raise ValueError("fit/model JSON requires diagnostics.training_feature_range")
    if record.get("source") != "current_training_batch_only" or record.get("scaling") != "fixed_business_scales":
        raise ValueError("saved bounds must describe the current training batch on fixed business scales")
    if record.get("usage") != "coverage_diagnostic_only":
        raise ValueError("saved training range lacks its coverage-only usage declaration")
    spec = _find_spec(path.parent, feature_spec_path)
    scope = value.get("scope", "unspecified_training_batch")
    result = TrainingRange(record["feature_names"], record["minimum"], record["maximum"], record["samples"],
        {"input": str(path), "kind": "saved_training_range", "source": record["source"], "scope": scope,
         "scaling": record["scaling"], "generating_policy": value.get("generating_policy"),
         "sha256": _sha(path), "configuration_fingerprint": value.get("configuration_fingerprint")}, spec)
    if "model" in value and value["model"].get("feature_names") != result.feature_names:
        raise ValueError("pilot-only model and recorded training range feature names disagree")
    return result


def _solution_record(solution, *, period, horizon, total_periods, terminal_kind, names, spec, source, scope):
    record = {"source": source, "period": period, "horizon": horizon,
              "terminal_period": period + horizon, "scope": scope, "method": terminal_kind}
    values = solution.get("terminal_features")
    if values is None or len(values) == 0:
        if terminal_kind in {"zero", "none", None}:
            record["skip_reason"] = "no_terminal_model"
        elif (0 <= period < total_periods and horizon > 0 and period + horizon == total_periods
                and "terminal_value" in solution and solution["terminal_value"] == 0.):
            record["skip_reason"] = "zero_value_operating_end_boundary"
        else:
            record["skip_reason"] = "missing_terminal_features_without_verified_zero_L"
    else:
        if names is None:
            raise ValueError("query feature names are required; dimension alone is not sufficient")
        record.update(terminal_features=values, feature_names=names, feature_spec=spec)
    return record


def _diagnostic_queries(path, explicit_spec=None, selected_method=None):
    directory = path.parent
    result = _json(path)
    if result.get("state") not in {"complete", "complete_with_diagnostic_failures"}:
        raise ValueError("query diagnostic has not completed")
    spec = _find_spec(directory, explicit_spec)
    names = _spec_names(spec)
    metadata = result.get("source", {})
    total = metadata.get("diagnostic_parameter_snapshot", metadata.get("original_parameter_snapshot", {})).get("num_periods")
    if total is None:
        raise ValueError("diagnostic must record the operating end period")
    period, horizon = result["period"], result["horizon"]
    found = False
    for method in result["methods"]:
        kind = method["method"]
        if selected_method is not None and kind != selected_method:
            continue
        found = True
        source = str(path) + "::" + kind
        scope = "pilot_only_diagnostic" if metadata.get("scope") == "pilot_only" else "single_window_diagnostic"
        if not method.get("feasible_incumbent"):
            yield {"source": source, "period": period, "scope": scope, "method": kind,
                   "skip_reason": "no_feasible_incumbent_diagnostic"}
            continue
        solution = _json(directory / kind / "solution.json")
        yield _solution_record(solution, period=period, horizon=horizon, total_periods=total,
            terminal_kind=kind, names=names, spec=spec, source=source, scope=scope)
    if selected_method is not None and not found:
        raise ValueError("selected method is absent from the completed diagnostic")


def _journal_queries(directory, explicit_spec=None):
    journal = directory / "journal"
    worker, status, header = (_json(directory / "worker_status.json"), _json(journal / "status.json"),
                              _json(journal / "initial.json"))
    if worker.get("state") != "complete" or status.get("state") != "complete":
        raise ValueError("query worker/journal is not a complete trajectory")
    identity = header["identity"]
    if header["fingerprint"] != fingerprint(identity) or status.get("fingerprint") != header["fingerprint"]:
        raise ValueError("query journal identity mismatch")
    total = identity["parameters"]["num_periods"]
    if status.get("completed_periods") != total:
        raise ValueError("query journal is missing operating periods")
    spec = _find_spec(directory, explicit_spec)
    names = _spec_names(spec)
    recorded_names = identity.get("feature_names")
    if names is None and recorded_names is not None:
        names = _names(recorded_names)
    elif names is not None and recorded_names is not None and names != recorded_names:
        raise ValueError("query journal and feature manifest order differ")
    model = identity.get("terminal_model")
    kind = model.get("kind") if isinstance(model, dict) else "zero" if model is None else "trained"
    scope = "pilot_only_trajectory" if "pilot_only" in str(directory) else "complete_trajectory_unspecified_split"
    count = 0
    with (journal / "rounds.jsonl").open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                raise ValueError("complete query journal contains an unfinished record")
            row = json.loads(raw)
            if count >= total or row["period"] != count:
                raise ValueError("query periods are duplicated, missing or out of order")
            yield _solution_record(row["solution"], period=count, horizon=row["horizon"], total_periods=total,
                terminal_kind=kind, names=names, spec=spec, source=str(journal / "rounds.jsonl"), scope=scope)
            count += 1
    if count != total:
        raise ValueError("complete query journal has too few records")


def iter_query_records(path, *, feature_spec_path=None):
    """Yield completed selected terminal states; adjacent journals are read once."""
    path = Path(path).resolve()
    if path.is_dir():
        if (path / "diagnostic_results.json").exists():
            yield from _diagnostic_queries(path / "diagnostic_results.json", feature_spec_path)
        elif (path / "journal/status.json").exists():
            yield from _journal_queries(path, feature_spec_path)
        else:
            raise ValueError("query directory must be a completed diagnostic or worker")
        return
    if path.name == "diagnostic_results.json":
        yield from _diagnostic_queries(path, feature_spec_path)
        return
    if path.name == "solution.json" and (path.parent.parent / "diagnostic_results.json").exists():
        yield from _diagnostic_queries(path.parent.parent / "diagnostic_results.json", feature_spec_path,
                                       selected_method=path.parent.name)
        return
    if path.name in {"result.json", "result.json.gz"} and (path.parent / "journal/status.json").exists():
        yield from _journal_queries(path.parent, feature_spec_path)
        return
    result = _json(path)  # Standalone exported full results only; worker paths use the streaming branch.
    total = result.get("parameter_snapshot", {}).get("num_periods")
    rows = result.get("rounds", [])
    if result.get("completed") is not True or total is None or len(rows) != total:
        raise ValueError("query full result must explicitly contain a complete operating trajectory")
    spec = _find_spec(path.parent, feature_spec_path)
    names = result.get("feature_names") or _spec_names(spec)
    if result.get("feature_names") is not None and spec is not None and names != _spec_names(spec):
        raise ValueError("query result and manifest feature order differ")
    for count, row in enumerate(rows):
        if row["period"] != count:
            raise ValueError("query full-result periods are not consecutive")
        yield _solution_record(row["solution"], period=count, horizon=row["horizon"], total_periods=total,
            terminal_kind=result.get("method", {}).get("terminal_kind", "trained"), names=names, spec=spec,
            source=str(path), scope=result.get("scope", "complete_trajectory_unspecified_split"))


def analyze_coverage(training, query_records, *, atol=1e-8):
    """One pass over named query states; no fitting, normalization or neighbors."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("absolute comparison tolerance must be finite and nonnegative")
    d = len(training.feature_names)
    below_count, above_count = np.zeros(d, dtype=int), np.zeros(d, dtype=int)
    below_max, above_max = np.zeros(d), np.zeros(d)
    constant = training.minimum == training.maximum
    skipped, scopes = Counter(), Counter()
    states, cache, groups = [], {}, {}
    seen_query_identifiers = set()
    query_count = outside_states = raw_outside_states = constant_changed_states = 0
    for record in query_records:
        if record.get("source") is not None:
            identifier = (record["source"], record.get("method"), record.get("period"), record.get("horizon"))
            if identifier in seen_query_identifiers:
                raise ValueError("duplicate selected terminal state supplied through overlapping query inputs")
            seen_query_identifiers.add(identifier)
        scope = record.get("scope", "unspecified_query")
        scopes[scope] += 1
        reason = record.get("skip_reason")
        if reason is not None:
            if reason not in {"zero_value_operating_end_boundary", "no_terminal_model", "no_feasible_incumbent_diagnostic",
                               "missing_terminal_features_without_verified_zero_L"}:
                raise ValueError("unknown query skip reason")
            skipped[reason] += 1
            continue
        key = tuple(record["feature_names"])
        if key not in cache:
            names = _names(key)
            if set(names) != set(training.feature_names):
                missing = sorted(set(training.feature_names) - set(names))
                extra = sorted(set(names) - set(training.feature_names))
                raise ValueError(f"query feature-name mismatch; missing={missing[:5]}, extra={extra[:5]}")
            index = {name: i for i, name in enumerate(names)}
            cache[key] = np.array([index[name] for name in training.feature_names])
        query_spec = record.get("feature_spec")
        if training.feature_spec is not None and query_spec is not None:
            for scale in SCALE_KEYS:
                if scale in training.feature_spec and scale in query_spec and training.feature_spec[scale] != query_spec[scale]:
                    raise ValueError(f"query fixed business scale differs: {scale}")
        vector = np.asarray(record["terminal_features"], dtype=float)
        if vector.shape != (d,) or not np.isfinite(vector).all():
            raise ValueError("query terminal features must be a finite vector matching its feature names")
        vector = vector[cache[key]]
        low = np.maximum(training.minimum - vector, 0.)
        high = np.maximum(vector - training.maximum, 0.)
        outside = (low > atol) | (high > atol)
        constant_changed = bool(np.any(outside & constant))
        below_count += low > atol
        above_count += high > atol
        below_max = np.maximum(below_max, low)
        above_max = np.maximum(above_max, high)
        query_count += 1
        outside_states += int(outside.any())
        raw_outside_states += int(np.any((low > 0.) | (high > 0.)))
        constant_changed_states += int(constant_changed)
        group_key = (record.get("source", "array"), record.get("method", "unspecified"))
        group = groups.setdefault(group_key, {"source": group_key[0], "method": group_key[1],
                                             "query_states": 0, "outside_states": 0})
        group["query_states"] += 1
        group["outside_states"] += int(outside.any())
        states.append({"source": group_key[0], "method": group_key[1], "scope": scope,
            "period": record.get("period"), "terminal_period": record.get("terminal_period"),
            "outside_training_range": bool(outside.any()), "outside_field_count": int(outside.sum()),
            "maximum_excess_fixed_scale": float(np.maximum(low, high).max()),
            "training_constant_field_changed": constant_changed})
    fields = []
    for i, name in enumerate(training.feature_names):
        outside = int(below_count[i] + above_count[i])
        fields.append({"feature_name": name, "training_minimum": float(training.minimum[i]),
            "training_maximum": float(training.maximum[i]), "training_constant": bool(constant[i]),
            "below_count": int(below_count[i]), "above_count": int(above_count[i]),
            "outside_count": outside, "outside_fraction": outside / query_count if query_count else None,
            "maximum_below_amount": float(below_max[i]), "maximum_above_amount": float(above_max[i]),
            "maximum_excess_amount": float(max(below_max[i], above_max[i])),
            "constant_changed_count": outside if constant[i] else 0})
    for group in groups.values():
        group["outside_fraction"] = group["outside_states"] / group["query_states"]
    return {"schema_version": 1, "analysis": "empirical_training_feature_range", "limitation": LIMITATION,
        "scope": "pilot_only" if training.metadata.get("scope") == "pilot_only" or any("pilot_only" in s for s in scopes)
                 else "coverage_diagnostic_only", "is_independent_test_profit_result": False,
        "training": dict(training.metadata, samples=training.samples, feature_dimension=d),
        "query_scopes": dict(scopes), "feature_names": training.feature_names,
        "feature_alignment": "exact name-set equality; explicit reorder to training order",
        "reordered_query_schemas": sum(list(key) != training.feature_names for key in cache),
        "fixed_scaling_unchanged": True, "comparison_absolute_tolerance": float(atol),
        "count_rule": "outside iff raw distance beyond training min/max exceeds absolute tolerance",
        "maximum_excess_rule": "raw excess in existing fixed feature units, including distances below tolerance",
        "query_states_with_features": query_count, "outside_training_range_states": outside_states,
        "outside_training_range_state_fraction": outside_states / query_count if query_count else None,
        "strict_zero_tolerance_outside_states": raw_outside_states,
        "training_constant_fields": int(constant.sum()),
        "training_constant_changed_fields": sum(row["constant_changed_count"] > 0 for row in fields),
        "states_with_changed_training_constant_field": constant_changed_states,
        "skipped": {key: skipped[key] for key in ("zero_value_operating_end_boundary", "no_terminal_model",
                                                   "no_feasible_incumbent_diagnostic",
                                                   "missing_terminal_features_without_verified_zero_L")},
        "per_feature": fields,
        "training_constant_but_query_changed": [row for row in fields if row["constant_changed_count"] > 0],
        "per_query_state": states, "per_source_and_method": list(groups.values())}


def analyze_files(training_path, query_paths, *, training_feature_spec=None, query_feature_spec=None, atol=1e-8):
    training = load_training_range(training_path, feature_spec_path=training_feature_spec)
    paths = [Path(path).resolve() for path in query_paths]
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("supply at least one distinct query input")
    def records():
        for path in paths:
            yield from iter_query_records(path, feature_spec_path=query_feature_spec)
    result = analyze_coverage(training, records(), atol=atol)
    result["query_inputs"] = [str(path) for path in paths]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--queries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-feature-spec", type=Path)
    parser.add_argument("--query-feature-spec", type=Path)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("coverage reports are immutable; choose a new output path")
    report = analyze_files(args.training, args.queries, training_feature_spec=args.training_feature_spec,
                           query_feature_spec=args.query_feature_spec, atol=args.atol)
    atomic_json(output, report)
    print(json.dumps({"output": str(output), "scope": report["scope"],
        "query_states_with_features": report["query_states_with_features"],
        "outside_training_range_state_fraction": report["outside_training_range_state_fraction"],
        "skipped": report["skipped"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
