"""Schedule validation and transparent model-forward accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PruneNode:
    step: int
    keep: int


@dataclass(frozen=True)
class ComputeBudget:
    psp_model_calls: int
    bon_model_calls: int
    exact_match: bool


def psp_budget(*, total_steps: int, initial_particles: int, nodes: Sequence[PruneNode], bon_candidates: int) -> ComputeBudget:
    """Count native denoiser calls; reward/VAE calls are reported separately."""
    if not nodes or nodes[-1].step != total_steps or nodes[-1].keep != 1:
        raise ValueError("nodes must end with (total_steps, 1)")
    active = initial_particles
    previous = 0
    calls = 0
    for node in nodes:
        if node.step <= previous or not 1 <= node.keep <= active:
            raise ValueError("invalid PSP node sequence")
        calls += active * (node.step - previous)
        active = node.keep
        previous = node.step
    bon_calls = bon_candidates * total_steps
    return ComputeBudget(calls, bon_calls, calls == bon_calls)
