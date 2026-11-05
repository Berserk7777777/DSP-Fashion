"""Portable manifest records. JSON serialization belongs to the caller."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    backbone: str
    method: str
    reward: Optional[str]
    dataset: str
    split: str
    step_argument: int
    actual_loop_calls: int
    schedule: List[Dict[str, int]]
    model_forward_budget: Dict[str, int]
    checkpoint: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_index: int
    uid: int
    oid: int
    target_category: int
    status: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateRecord:
    task_id: str
    method: str
    candidate_index: int
    seed: int
    node: Union[int, str]
    image_path: str
    image_semantics: str
    reward: Optional[float] = None
    selected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
