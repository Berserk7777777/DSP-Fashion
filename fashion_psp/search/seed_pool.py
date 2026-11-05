"""Task-local, collision-free seed assignments independent of run order."""

from __future__ import annotations

from typing import List


def candidate_seed(*, base: int, task_index: int, candidate_index: int, stride: int = 10_000) -> int:
    if task_index < 0 or candidate_index < 0 or candidate_index >= stride:
        raise ValueError("invalid task/candidate index for fixed-stride seed mapping")
    return base + task_index * stride + candidate_index


def make_seed_pool(*, base: int, task_index: int, count: int, stride: int = 10_000) -> List[int]:
    return [candidate_seed(base=base, task_index=task_index, candidate_index=i, stride=stride) for i in range(count)]


def validate_disjoint_pools(*pools: List[int]) -> None:
    flattened = [seed for pool in pools for seed in pool]
    if len(flattened) != len(set(flattened)):
        raise ValueError("seed pools overlap")
