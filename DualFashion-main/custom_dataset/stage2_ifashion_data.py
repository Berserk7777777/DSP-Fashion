"""Attribute serialization helpers used by the public inference scripts.

The upstream repository imports these helpers from an unpublished
``custom_dataset`` package. PFITB and GOR inference only need deterministic
normalization of the structured attribute values, so keep the compatibility
surface deliberately small here.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _normalize_value(value: Any) -> str:
    """Convert an item attribute into the textual value consumed by T5."""
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        normalized = value.strip()
        return normalized if normalized else "Unknown"
    if isinstance(value, Mapping):
        return _safe_json_dumps(value)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = [_normalize_value(item) for item in value]
        values = [item for item in values if item != "Unknown"]
        return ", ".join(values) if values else "Unknown"
    return str(value)


def _safe_json_dumps(payload: Mapping[str, Any]) -> str:
    """Serialize structured attributes without changing their declared key order."""
    normalized = {str(key): _normalize_value(value) for key, value in payload.items()}
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ": "))
