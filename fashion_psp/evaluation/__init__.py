"""Evaluation contract shared across backbone-specific metric runners."""

from .contract import POSTHOC_ONLY_METRICS, SELECTION_FORBIDDEN_INPUTS

__all__ = ["POSTHOC_ONLY_METRICS", "SELECTION_FORBIDDEN_INPUTS"]
