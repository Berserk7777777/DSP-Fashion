"""Causal terminal-Personal forecasting and budget-neutral survivor selection.

Only per-particle score prefixes enter the predictor. Candidate IDs, user IDs,
GT metrics, other particles' future scores, and step-28 labels are not features.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = (
    "step_fraction", "sigma", "personal_current", "personal_initial",
    "prefix_mean", "prefix_std", "prefix_min", "prefix_max",
    "last_delta", "last3_delta_mean", "last7_delta_mean", "last3_delta_std",
    "last5_slope", "last10_slope", "acceleration", "gain_from_initial",
    "drop_from_peak", "quarter_prefix_value", "half_prefix_value",
    "three_quarter_prefix_value",
)


def prefix_features(personal, sigmas, total_steps=28):
    """Call with an already sliced prefix, never with a full trajectory + t."""
    p = np.asarray(personal, dtype=np.float64) / 100.0
    s = np.asarray(sigmas, dtype=np.float64)
    if not 1 <= len(p) <= total_steps or len(s) != len(p):
        raise ValueError("Expected matching, nonempty observed prefixes")
    if not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("Nonfinite observed score/sigma")
    d = np.diff(p)

    def slope(window):
        y = p[-window:]
        x = np.arange(len(y), dtype=np.float64)
        x -= x.mean()
        return float(x @ (y - y.mean()) / (x @ x)) if len(y) > 1 else 0.0

    result = [len(p) / total_steps, s[-1], p[-1], p[0], p.mean(), p.std(),
              p.min(), p.max(), d[-1] if len(d) else 0.0,
              d[-3:].mean() if len(d) else 0.0,
              d[-7:].mean() if len(d) else 0.0,
              d[-3:].std() if len(d) else 0.0,
              slope(5), slope(10), d[-1] - d[-2] if len(d) > 1 else 0.0,
              p[-1] - p[0], p[-1] - p.max()]
    result.extend(float(np.interp(fraction * (len(p) - 1), np.arange(len(p)), p))
                  for fraction in (0.25, 0.5, 0.75))
    return np.asarray(result, dtype=np.float32)


class TerminalPersonalMLP(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, 2))

    def forward(self, x):
        value = self.net(x)
        # Forecast the remaining reward change / 100. Variance is learned,
        # allowing different errors at different fidelities without monotonicity.
        return value[:, 0], value[:, 1].clamp(-12.0, 0.0)


class TerminalPersonalPredictor:
    def __init__(self, bundle, device="cpu"):
        if bundle["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Checkpoint feature schema mismatch")
        self.device = torch.device(device)
        self.mean = torch.as_tensor(bundle["feature_mean"], dtype=torch.float32, device=self.device)
        self.scale = torch.as_tensor(bundle["feature_scale"], dtype=torch.float32, device=self.device)
        self.models = []
        for state in bundle["member_states"]:
            model = TerminalPersonalMLP(bundle["hidden_dim"]).to(self.device)
            model.load_state_dict(state)
            self.models.append(model.eval())
        self.total_steps = bundle["total_steps"]
        self.q = bundle.get("calibration", {}).get("q", 1.0)

    @torch.inference_mode()
    def predict_features(self, features, current_personal, batch_size=4096):
        x = np.asarray(features, dtype=np.float32)
        means, deviations = [], []
        for start in range(0, len(x), batch_size):
            chunk = torch.as_tensor(x[start:start + batch_size], device=self.device)
            chunk = (chunk - self.mean) / self.scale
            values = [model(chunk) for model in self.models]
            mu = torch.stack([row[0] for row in values])
            variance = torch.stack([row[1].exp() for row in values])
            means.append((mu.mean(0) * 100).cpu().numpy())
            # Ensemble disagreement + learned residual variance. A 0.05-point
            # floor prevents numerical zero-width intervals during calibration.
            deviations.append(((variance.mean(0) + mu.var(0, unbiased=False)).sqrt()
                               * 100).clamp_min(0.05).cpu().numpy())
        return np.concatenate(means) + np.asarray(current_personal), np.concatenate(deviations)

    def predict(self, personal_prefixes, sigma_prefixes):
        if any(len(p) >= self.total_steps for p in personal_prefixes):
            raise ValueError("At the terminal step use the observed reward, not a forecast")
        x = [prefix_features(p, s, self.total_steps)
             for p, s in zip(personal_prefixes, sigma_prefixes)]
        return self.predict_features(x, [p[-1] for p in personal_prefixes])


def uncertainty_survivors(seeds, mean, std, q, keep):
    """Keep a mean-best anchor plus optimistic contenders, with full diagnostics.

    Intervals diagnose clearly dominated candidates. The fixed budget can
    force pruning contenders whose intervals overlap; such removals are
    explicitly counted and are NOT described as confidence-certified.
    """
    seeds = list(seeds)
    mean, std = np.asarray(mean), np.asarray(std)
    if not 1 <= keep <= len(seeds) or not np.isfinite(q) or q < 0:
        raise ValueError("Invalid survivor count or finite calibration multiplier")
    lower, upper = mean - q * std, mean + q * std
    threshold = float(lower.max())
    dominated = upper < threshold
    anchor = min(range(len(seeds)), key=lambda i: (-mean[i], seeds[i]))
    priority = sorted((i for i in range(len(seeds)) if i != anchor),
                      key=lambda i: (bool(dominated[i]), -upper[i], -mean[i], seeds[i]))
    chosen = [anchor, *priority[:keep - 1]]
    removed = set(range(len(seeds))) - set(chosen)
    trace = {"interval_q":float(q), "best_lower_bound":threshold,
             "mean_best_anchor":int(seeds[anchor]),
             "forced_prunes":sum(not bool(dominated[i]) for i in removed),
             "interval_dominated_prunes":sum(bool(dominated[i]) for i in removed),
             "candidates":[{"seed":int(seed), "mean":float(mean[i]), "std":float(std[i]),
                            "lower":float(lower[i]), "upper":float(upper[i]),
                            "interval_dominated":bool(dominated[i]), "keep":i in chosen}
                           for i, seed in enumerate(seeds)]}
    return [seeds[i] for i in chosen], trace
