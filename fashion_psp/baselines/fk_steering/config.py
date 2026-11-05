"""Frozen, serializable parameters for an FK-Steering experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple


@dataclass(frozen=True)
class FKSteeringConfig:
    num_particles: int
    resampling_steps: Tuple[int, ...]
    potential_type: Literal["diff", "max", "add", "rt"]
    lmbda: float
    resampling: Literal["multinomial", "systematic"]
    adaptive_resampling: bool
    ess_fraction: float
    terminal_selection: Literal["max_reward"] = "max_reward"

    def validate(self, total_steps: int) -> None:
        if self.num_particles < 2:
            raise ValueError("FK-Steering requires at least two particles")
        if not self.resampling_steps or tuple(sorted(set(self.resampling_steps))) != self.resampling_steps:
            raise ValueError("resampling steps must be unique and increasing")
        if any(step < 1 or step > total_steps for step in self.resampling_steps):
            raise ValueError("resampling step outside the native sampling range")
        if self.lmbda <= 0:
            raise ValueError("lambda must be positive")
        if not 0 < self.ess_fraction <= 1:
            raise ValueError("ESS fraction must be in (0, 1]")
