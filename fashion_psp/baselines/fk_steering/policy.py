"""Pure FK potential and parent-index sampling, independent of torch/model code."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FKStep:
    raw_rewards: Tuple[float, ...]
    potentials: Tuple[float, ...]
    normalized_weights: Tuple[float, ...]
    ess: float
    parent_indices: Optional[Tuple[int, ...]]


def importance_weights(
    rewards: Sequence[float],
    previous_rewards: Sequence[float],
    *,
    potential_type: str,
    lmbda: float,
) -> Tuple[Tuple[float, ...], Tuple[float, ...], float]:
    """Return potential, stable normalized weights, and effective sample size."""
    if len(rewards) == 0 or len(rewards) != len(previous_rewards):
        raise ValueError("current and previous reward vectors must be non-empty and aligned")
    if lmbda <= 0 or any(not math.isfinite(float(x)) for x in tuple(rewards) + tuple(previous_rewards)):
        raise ValueError("rewards and lambda must be finite, with lambda > 0")
    if potential_type == "diff":
        potentials = [float(now) - float(old) for now, old in zip(rewards, previous_rewards)]
    elif potential_type == "max":
        potentials = [max(float(now), float(old)) for now, old in zip(rewards, previous_rewards)]
    elif potential_type == "add":
        potentials = [float(now) + float(old) for now, old in zip(rewards, previous_rewards)]
    elif potential_type == "rt":
        potentials = [float(now) for now in rewards]
    else:
        raise ValueError("potential_type must be diff, max, add, or rt")
    logits = [lmbda * value for value in potentials]
    offset = max(logits)
    weights = [math.exp(value - offset) for value in logits]
    total = sum(weights)
    normalized = [value / total for value in weights]
    ess = 1.0 / sum(value * value for value in normalized)
    return tuple(potentials), tuple(normalized), ess


def resample_parent_indices(
    weights: Sequence[float],
    *,
    count: int,
    method: str,
    rng: random.Random,
) -> Tuple[int, ...]:
    """Draw parent indices with replacement; adapter owns state duplication."""
    if count < 1 or not weights or any(weight < 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("invalid weight vector or output count")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weight sum must be positive")
    normalized = [float(weight) / total for weight in weights]
    cumulative: List[float] = []
    running = 0.0
    for weight in normalized:
        running += weight
        cumulative.append(running)

    def index_at(value: float) -> int:
        for index, boundary in enumerate(cumulative):
            if value <= boundary:
                return index
        return len(cumulative) - 1

    if method == "multinomial":
        return tuple(index_at(rng.random()) for _ in range(count))
    if method == "systematic":
        start = rng.random() / count
        return tuple(index_at(start + index / count) for index in range(count))
    raise ValueError("resampling method must be multinomial or systematic")
