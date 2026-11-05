"""A small registry; reward implementation and calibration stay external."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping

RewardFn = Callable[[List[Any], Mapping[str, Any]], List[float]]


class RewardRegistry:
    def __init__(self) -> None:
        self._functions: Dict[str, RewardFn] = {}

    def register(self, name: str, fn: RewardFn) -> None:
        if not name or name in self._functions:
            raise ValueError(f"invalid or duplicate reward name: {name!r}")
        self._functions[name] = fn

    def get(self, name: str) -> RewardFn:
        try:
            return self._functions[name]
        except KeyError as exc:
            raise KeyError(f"unknown reward {name!r}; registered: {sorted(self._functions)}") from exc
