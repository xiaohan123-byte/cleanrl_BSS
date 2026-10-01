"""Frozen-policy Monte Carlo value regression on fixed business features."""
from __future__ import annotations

from time import perf_counter
import numpy as np

from src.experiment_control import check_deadline


def monte_carlo_samples(result: dict) -> tuple[np.ndarray, np.ndarray]:
    periods = result["parameter_snapshot"]["num_periods"]
    rows = result["rounds"]
    if len(rows) != periods or result.get("completed", True) is not True:
        raise ValueError("only complete operating trajectories may supply MC labels")
    features = np.asarray([row["state_features"] for row in rows], dtype=np.float64)
    rewards = np.asarray([row["reward"] for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(rewards)):
        raise ValueError("nonfinite trajectory features or realised rewards")
    returns = np.cumsum(rewards[::-1])[::-1].copy()
    if not np.isclose(returns[0], result["summary"]["total_reward"], rtol=1e-8, atol=1e-6):
        raise ValueError("return labels do not reconcile with the actual ledger")
    return features, returns


def fit_value_model(features, returns_yuan, spec, *, kind="relu", previous=None,
                    seed=0, epochs=200, deadline=None):
    """Fit one *fresh* policy batch; output restores currency units.

    No feature scaling is fitted here. Ridge uses mean squared loss + lambda L2.
    ReLU uses Adam with fresh optimizer state; only layer weights receive decay.
    """
    from src.terminal_value import TerminalValueModel
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(returns_yuan, dtype=np.float64) / 1000.
    if x.ndim != 2 or x.shape[1] != spec.dimension or y.shape != (len(x),) or len(x) == 0:
        raise ValueError("incompatible feature/return batch")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("regression data must be finite")
    check_deadline(deadline)
    started = perf_counter()
    history = []
    if kind == "linear":
        # Centering leaves the intercept unpenalized; dual solve avoids D-by-D
        # matrices when there are more fixed features than observations.
        x_mean, y_mean = x.mean(0), float(y.mean())
        xc, yc = x - x_mean, y - y_mean
        penalty = len(x) * 1e-5
        if x.shape[1] <= len(x):
            gram = xc.T @ xc
            gram.flat[::len(gram) + 1] += penalty
            weights = np.linalg.solve(gram, xc.T @ yc)
        else:
            gram = xc @ xc.T
            gram.flat[::len(gram) + 1] += penalty
            weights = xc.T @ np.linalg.solve(gram, yc)
        bias = y_mean - float(x_mean @ weights)
        model = TerminalValueModel(kind="linear", feature_names=list(spec.names),
            variant=spec.variant, linear_weights=weights.tolist(), bias=bias,
            hidden_weights=[], hidden_bias=[], output_weights=[], output_scale=1000.)
        history.append({"epoch": 1, "mse_scaled": float(np.mean((x @ weights + bias - y)**2))})
    elif kind == "relu":
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(int(seed))
        generator = torch.Generator().manual_seed(int(seed) + 7919)
        class Regressor(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.skip = torch.nn.Linear(width, 1)
                self.hidden = torch.nn.Linear(width, 16)
                self.output = torch.nn.Linear(16, 1, bias=False)
            def forward(self, values):
                return (self.skip(values) + self.output(torch.relu(self.hidden(values)))).squeeze(1)
        network = Regressor(x.shape[1])
        if previous is not None:
            if previous.kind != "relu" or list(previous.feature_names) != list(spec.names):
                raise ValueError("warm-start model has incompatible fixed features")
            with torch.no_grad():
                network.skip.weight.copy_(torch.as_tensor(np.asarray(previous.linear_weights)[None, :], dtype=torch.float32))
                network.skip.bias.copy_(torch.tensor([previous.bias]))
                network.hidden.weight.copy_(torch.as_tensor(previous.hidden_weights, dtype=torch.float32))
                network.hidden.bias.copy_(torch.as_tensor(previous.hidden_bias, dtype=torch.float32))
                network.output.weight.copy_(torch.as_tensor(np.asarray(previous.output_weights)[None, :], dtype=torch.float32))
        weights = [p for name,p in network.named_parameters() if not name.endswith("bias")]
        biases = [p for name,p in network.named_parameters() if name.endswith("bias")]
        optimizer = torch.optim.Adam([{"params":weights,"weight_decay":1e-5},
                                      {"params":biases,"weight_decay":0.}], lr=1e-3)
        inputs, targets = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        for epoch in range(int(epochs)):
            check_deadline(deadline)
            order = torch.randperm(len(inputs), generator=generator)
            for offset in range(0,len(inputs),256):
                check_deadline(deadline)
                batch = order[offset:offset+256]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(network(inputs[batch]),targets[batch])
                loss.backward()
                optimizer.step()
            if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
                with torch.no_grad():
                    mse = torch.mean((network(inputs)-targets)**2).item()
                history.append({"epoch":epoch+1,"mse_scaled":float(mse)})
        model = TerminalValueModel(kind="relu", feature_names=list(spec.names),
            variant=spec.variant, linear_weights=network.skip.weight.detach().numpy()[0].tolist(),
            bias=float(network.skip.bias.item()),
            hidden_weights=network.hidden.weight.detach().numpy().tolist(),
            hidden_bias=network.hidden.bias.detach().numpy().tolist(),
            output_weights=network.output.weight.detach().numpy()[0].tolist(), output_scale=1000.)
    else:
        raise ValueError("fit kind must be linear or relu")
    if hasattr(spec, "params"):
        from src.terminal_value import terminal_configuration_fingerprint
        model.configuration_fingerprint = terminal_configuration_fingerprint(spec.params)
    check_deadline(deadline)
    prediction = model.predict(x)
    diagnostics = {"kind":kind,"samples":len(x),"feature_dimension":x.shape[1],
        "epochs":int(epochs) if kind=="relu" else 1,"target_scale_yuan":1000.,
        "seed":int(seed),"warm_started":previous is not None and kind=="relu",
        "optimizer_state_reset":True,"wall_seconds":perf_counter()-started,
        "training_rmse_yuan":float(np.sqrt(np.mean((prediction-np.asarray(returns_yuan))**2))),
        "history":history,"checkpoint_rule":"last epoch of each fit; outer models selected by validation net profit",
        "training_feature_range":{"feature_names":list(spec.names),"minimum":x.min(axis=0).tolist(),
            "maximum":x.max(axis=0).tolist(),"samples":len(x),"scaling":"fixed_business_scales",
            "source":"current_training_batch_only","usage":"coverage_diagnostic_only"}}
    return model, diagnostics
