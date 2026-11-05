"""Stable interfaces between native fashion generators and search policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Union


@dataclass(frozen=True)
class Particle:
    candidate_index: int
    seed: int
    task_id: str


@dataclass(frozen=True)
class RewardView:
    """A decoded image made available to an inference-time reward."""

    particle: Particle
    node: Union[int, str]
    image: Any
    image_path: Optional[Path] = None
    semantics: str = "predicted_clean"


@dataclass(frozen=True)
class FinalArtifact:
    particle: Particle
    image_path: Path
    scheduler_node: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class SamplingAdapter(Protocol):
    """Opaque native-sampler lifecycle required by Standard, BoN, and PSP."""

    name: str

    def prepare_task(self, task: Dict[str, Any]) -> Any: ...

    def initialize(self, task: Any, particles: Sequence[Particle]) -> Any: ...

    def step(self, state: Any) -> Tuple[Any, int]: ...

    def make_reward_views(self, state: Any, node: int) -> Sequence[RewardView]: ...

    def prune(self, state: Any, survivor_indices: Sequence[int]) -> Any: ...

    def finalize(self, state: Any, output_dir: Path) -> Sequence[FinalArtifact]: ...
