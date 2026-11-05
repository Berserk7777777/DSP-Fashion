"""Backbone-agnostic orchestration primitives for FashionPSP."""

from .config import BACKBONE_SPECS, BackboneSpec, RunConfig
from .contracts import Particle, RewardView, SamplingAdapter

__all__ = [
    "BACKBONE_SPECS",
    "BackboneSpec",
    "Particle",
    "RewardView",
    "RunConfig",
    "SamplingAdapter",
]
