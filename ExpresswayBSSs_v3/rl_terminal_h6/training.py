"""Fresh-policy Monte Carlo regression with trainable hierarchical encoders."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import gzip
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from state_encoding import Normalizer
from learned_value import LearnedValue, digest
from src.parameters import validate_complete_result


class SetValueNetwork(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.slot_counts = spec.slot_counts
        self.battery = torch.nn.Linear(1, 4)
        self.request = torch.nn.Linear(len(spec.a_names), 8)
        self.user = torch.nn.Linear(8, 8)
        self.random = torch.nn.Linear(len(spec.r_names), 4)
        self.skip = torch.nn.Linear(spec.dimension, 1)
        self.hidden = torch.nn.Linear(spec.dimension, 16)
        self.output = torch.nn.Linear(16, 1, bias=False)

    def encode(self, batch):
        b = torch.relu(self.battery(batch["battery"]))
        station_parts = []
        offset = 0
        for count in self.slot_counts:
            station_parts.append(b[:, offset:offset+count].sum(dim=1))
            offset += count
        requests = torch.relu(self.request(batch["a"])) * batch["a_mask"][..., None]
        users = torch.relu(self.user(requests.sum(dim=2))) - torch.relu(self.user.bias)
        random = torch.relu(self.random(batch["r"])) * batch["r_mask"][..., None]
        return torch.cat([*station_parts, users.sum(dim=1), random.sum(dim=1), batch["x"]], dim=1)

    def forward(self, batch):
        phi = self.encode(batch)
        value = (self.skip(phi) + self.output(torch.relu(self.hidden(phi)))).squeeze(-1)
        return value * batch["continuation"]


def normalized_states(states, normalizer):
    return [{"battery": normalizer.transform("battery", sum(s["batteries"], [])).astype(np.float32),
             "a": [normalizer.transform("a", u).astype(np.float32) for u in s["users"]],
             "r": normalizer.transform("r", s["random"]).astype(np.float32),
             "x": normalizer.transform("x", [s["external"]])[0].astype(np.float32),
             "continuation": s["continuation"]} for s in states]


def collate(values, indices, spec, device):
    selected = [values[int(i)] for i in indices]
    b = len(selected)
    nu = max(1, max(len(s["a"]) for s in selected))
    nr = max(1, max((len(u) for s in selected for u in s["a"]), default=0))
    nrand = max(1, max(len(s["r"]) for s in selected))
    a = np.zeros((b, nu, nr, len(spec.a_names)), dtype=np.float32)
    am = np.zeros((b, nu, nr), dtype=np.float32)
    r = np.zeros((b, nrand, len(spec.r_names)), dtype=np.float32)
    rm = np.zeros((b, nrand), dtype=np.float32)
    for i, state in enumerate(selected):
        for j, user in enumerate(state["a"]):
            a[i,j,:len(user)] = user
            am[i,j,:len(user)] = 1.
        r[i,:len(state["r"])] = state["r"]
        rm[i,:len(state["r"])] = 1.
    arrays = {"a": a, "a_mask": am, "r": r, "r_mask": rm,
              "battery": np.stack([s["battery"] for s in selected]),
              "x": np.stack([s["x"] for s in selected]),
              "continuation": np.asarray([s["continuation"] for s in selected], dtype=np.float32)}
    return {k: torch.as_tensor(v, device=device) for k,v in arrays.items()}


def monte_carlo_states(result):
    validate_complete_result(result)
    rows = result["rounds"]
    rewards = np.asarray([r["reward"] for r in rows], dtype=float)
    returns = np.cumsum(rewards[::-1])[::-1]
    if not np.isfinite(returns).all() or not np.isclose(returns[0], result["summary"]["total_reward"], rtol=1e-8, atol=1e-6):
        raise ValueError("MC targets do not reconcile with the complete actual ledger")
    states = [r["state_input"] for r in rows]
    terminal = result["terminal_state_input"]
    if terminal["continuation"] != 0 or terminal["users"] or terminal["random"]:
        raise ValueError("complete trajectory must supply its absorbing terminal state")
    return [*states, terminal], np.r_[returns, 0.]


def fit(states, returns, spec, contract, *, previous=None, iteration=0, epochs=200, batch_size=256,
        device="cuda", progress=None):
    started = perf_counter()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for the frozen training protocol but unavailable")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1)
    normalizer = previous.normalizer if previous else Normalizer.fit(states, spec)
    values = normalized_states(states, normalizer)
    targets = np.asarray(returns, dtype=np.float32)/1000.
    if len(values) != len(targets) or not len(values) or not np.isfinite(targets).all():
        raise ValueError("invalid training data")
    net = SetValueNetwork(spec).to(device)
    if previous:
        if previous.record["state_schema"] != spec.schema() or previous.record["contract"] != contract:
            raise ValueError("warm-start schema or data contract differs")
        net.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32, device=device) for k,v in previous.weights.items()})
    before = {k: v.detach().cpu().numpy().copy() for k,v in net.state_dict().items()}
    optimizer = torch.optim.Adam(net.parameters(), lr=.001, weight_decay=0.)
    shuffle = np.random.default_rng(np.random.SeedSequence([1, 330, iteration]))
    history = []
    grad_max = {name: 0. for name in ("battery", "request", "user", "random", "skip", "hidden", "output")}
    for epoch in range(epochs):
        order = shuffle.permutation(len(values))
        loss_sum = 0.
        for offset in range(0, len(values), batch_size):
            indices = order[offset:offset+batch_size]
            batch = collate(values, indices, spec, device)
            target = torch.as_tensor(targets[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = net(batch)
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite neural regression loss")
            loss.backward()
            for name, parameter in net.named_parameters():
                if parameter.grad is not None:
                    group = name.split(".")[0]
                    grad_max[group] = max(grad_max[group], float(parameter.grad.detach().abs().max().item()))
            optimizer.step()
            loss_sum += float(loss.item())*len(indices)
        if epoch == 0 or (epoch+1)%10 == 0 or epoch+1 == epochs:
            record = {"epoch": epoch+1, "online_epoch_mse_scaled": loss_sum/len(values)}
            history.append(record)
            if progress:
                progress(record)
    weights = {k: v.detach().cpu().numpy().astype(float).tolist() for k,v in net.state_dict().items()}
    model = LearnedValue({"schema_version": 2, "kind": LearnedValue.kind, "variant": LearnedValue.variant,
        "architecture": {"battery": 4, "request": 8, "user": 8, "random": 4, "head": 16},
        "weights": weights, "normalizer": normalizer.record, "state_schema": spec.schema(),
        "contract": contract, "output_scale_yuan": 1000., "seed": 1, "iteration": iteration+1})
    predicted = []
    max_export_error = 0.
    with torch.no_grad():
        for offset in range(0, len(values), batch_size):
            indices = np.arange(offset, min(offset+batch_size, len(values)))
            batch_prediction = net(collate(values, indices, spec, device)).cpu().numpy()*1000.
            exported = np.asarray([model.predict(states[i]) for i in indices])
            max_export_error = max(max_export_error, float(np.max(np.abs(batch_prediction-exported))))
            if not np.allclose(batch_prediction, exported, rtol=2e-5, atol=.1):
                raise ValueError("PyTorch and exported inference disagree")
            predicted.extend(exported.tolist())
    change = {group: sum(float(np.linalg.norm(np.asarray(weights[k])-before[k])) for k in before if k.startswith(group+"."))
              for group in grad_max}
    if any(grad_max[g] == 0 or change[g] == 0 for g in grad_max):
        raise ValueError(f"a trainable layer received no update: gradients={grad_max}, changes={change}")
    diagnostics = {"seed": 1, "iteration": iteration+1, "samples": len(states), "epochs": epochs,
        "batch_size": batch_size, "optimizer_steps": epochs*math.ceil(len(states)/batch_size),
        "optimizer": "Adam", "learning_rate": .001, "weight_decay": 0., "warm_started": previous is not None,
        "optimizer_state_reset": True, "loss": "undiscounted complete realized-return MSE, targets divided by 1000",
        "normalizer_hash": digest(normalizer.record), "feature_dimension": spec.dimension,
        "parameter_count": model.parameter_count, "history": history, "gradient_max_by_layer": grad_max,
        "parameter_change_l2_by_layer": change, "training_rmse_yuan": float(np.sqrt(np.mean((np.asarray(predicted)-returns)**2))),
        "export_max_abs_error_yuan": max_export_error, "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "torch_version": torch.__version__, "wall_seconds": perf_counter()-started}
    return model, diagnostics
