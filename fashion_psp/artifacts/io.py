"""Small atomic writers used by long-running, task-resumable experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Commit a complete JSON file or leave the previous file untouched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write a deterministic snapshot; callers append only after task commit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))
