"""Serializable fixed-feature terminal values, in yuan at the public boundary."""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TerminalValueModel:
    kind: str
    feature_names: list[str] = field(default_factory=list)
    variant: str = "full"
    linear_weights: Any = field(default_factory=list)
    bias: float = 0.0
    hidden_weights: Any = field(default_factory=list)
    hidden_bias: Any = field(default_factory=list)
    output_weights: Any = field(default_factory=list)
    output_scale: float = 1000.0
    inventory_coefficient: float = 0.0
    configuration_fingerprint: str | None = None

    def __post_init__(self):
        if self.kind not in {"linear", "relu", "simple_inventory"}:
            raise ValueError("unknown terminal value model")
        for name in ("linear_weights", "hidden_weights", "hidden_bias", "output_weights"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        if not np.isfinite(self.output_scale) or self.output_scale <= 0:
            raise ValueError("terminal output scale must be finite and positive")
        if self.kind == "simple_inventory":
            if not np.isfinite(self.inventory_coefficient):
                raise ValueError("inventory coefficient must be finite")
            return
        d = len(self.feature_names)
        if self.linear_weights.shape != (d,):
            raise ValueError("linear weights must match feature names")
        if len(set(self.feature_names)) != d:
            raise ValueError("feature names must be unique")
        if self.kind == "relu":
            h = self.hidden_bias.size
            if self.hidden_weights.shape != (h, d) or self.output_weights.shape != (h,):
                raise ValueError("invalid ReLU weight dimensions")
        if not np.isfinite(self.bias) or any(not np.all(np.isfinite(getattr(self, name)))
                for name in ("linear_weights", "hidden_weights", "hidden_bias", "output_weights")):
            raise ValueError("terminal parameters must be finite")

    @property
    def hidden_units(self):
        return int(self.hidden_bias.size) if self.kind == "relu" else 0

    @property
    def parameter_count(self):
        return 1 + self.linear_weights.size + self.hidden_weights.size + self.hidden_bias.size + self.output_weights.size

    def predict(self, features):
        x = np.asarray(features, dtype=float)
        if self.kind == "simple_inventory":
            result = self.inventory_coefficient * (x[..., 0] if x.ndim else x)
        else:
            if x.shape[-1] != len(self.feature_names):
                raise ValueError("terminal input dimension differs from trained feature schema")
            result = self.bias + x @ self.linear_weights
            if self.kind == "relu":
                result = result + np.maximum(0.0, x @ self.hidden_weights.T + self.hidden_bias) @ self.output_weights
            result = self.output_scale * result
        return float(result) if np.ndim(result) == 0 else result

    def to_dict(self):
        return {"schema_version": 1, "kind": self.kind, "variant": self.variant,
                "feature_names": list(self.feature_names), "bias": float(self.bias),
                "linear_weights": self.linear_weights.tolist(),
                "hidden_weights": self.hidden_weights.tolist(),
                "hidden_bias": self.hidden_bias.tolist(), "output_weights": self.output_weights.tolist(),
                "output_scale": float(self.output_scale), "inventory_coefficient": float(self.inventory_coefficient),
                "configuration_fingerprint": self.configuration_fingerprint}

    @classmethod
    def from_dict(cls, record):
        payload = dict(record)
        if payload.pop("schema_version", 1) != 1:
            raise ValueError("unsupported terminal model schema")
        return cls(**payload)

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def terminal_configuration_fingerprint(params):
    """Bind a value to its physical/demand configuration, allowing H/seed changes."""
    payload = dict(params.to_dict())
    for key in ("horizon", "solver", "seed", "source_metadata"):
        payload.pop(key, None)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def bounded_relu(model, cp, COPT, expression, lower, upper, name):
    """Exact graph over a valid finite input interval, including both signs."""
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError("ReLU needs valid finite input bounds")
    if upper <= 0:
        return 0.0
    if lower >= 0:
        return expression
    z = model.addVar(lb=0.0, ub=max(0.0, upper), name=name)
    active = model.addVar(vtype=COPT.BINARY, name=name + ":active")
    model.addConstr(z >= expression)
    model.addConstr(z <= expression - lower * (1 - active))
    model.addConstr(z <= upper * active)
    return z


def embed_value(model, cp, COPT, value_model, features, bounds, feature_names):
    """Embed a fixed neural/linear value in the same mixed-integer model."""
    if list(value_model.feature_names) != list(feature_names):
        raise ValueError("terminal feature schema differs from trained model")
    if len(features) != len(bounds):
        raise ValueError("missing feature bounds")
    affine = value_model.bias + cp.quicksum(float(w) * x for w, x in zip(value_model.linear_weights, features)
                                            if w != 0)
    hidden_values = []
    if value_model.kind == "relu":
        for index, (weights, bias, output) in enumerate(zip(value_model.hidden_weights,
                                                           value_model.hidden_bias,
                                                           value_model.output_weights)):
            expression = float(bias) + cp.quicksum(float(w) * x for w, x in zip(weights, features) if w != 0)
            lo, hi = float(bias), float(bias)
            for weight, (low, high) in zip(weights, bounds):
                lo += float(weight) * (low if weight >= 0 else high)
                hi += float(weight) * (high if weight >= 0 else low)
            z = bounded_relu(model, cp, COPT, expression, lo, hi, f"terminal_hidden[{index}]")
            hidden_values.append(z)
            affine += float(output) * z
    return value_model.output_scale * affine, hidden_values
