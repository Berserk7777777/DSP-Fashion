"""Deterministic reward ranking shared by online PSP and offline replay."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import List, Sequence


def select_survivors(scores_by_seed: Mapping[int, float], keep: int) -> List[int]:
    """Higher reward wins; ascending integer seed breaks equal-score ties."""
    if not 1 <= keep <= len(scores_by_seed):
        raise ValueError("invalid keep count")
    if any(not math.isfinite(float(score)) for score in scores_by_seed.values()):
        raise ValueError("reward scores must be finite")
    return [seed for seed, _ in sorted(scores_by_seed.items(), key=lambda item: (-float(item[1]), item[0]))[:keep]]


def survivor_indices(active_seeds: Sequence[int], selected_seeds: Sequence[int]) -> List[int]:
    positions = {seed: index for index, seed in enumerate(active_seeds)}
    if len(positions) != len(active_seeds) or any(seed not in positions for seed in selected_seeds):
        raise ValueError("selected seeds are not a unique subset of active seeds")
    return [positions[seed] for seed in selected_seeds]
