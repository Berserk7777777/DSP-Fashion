#!/usr/bin/env python
"""Aggregate separately frozen offline FITB group summaries without mixing groups.

Each input must be an ``official_eval/group_summary.csv`` emitted by
``offline_fitb_select_and_evaluate.py``.  The output retains every group row
and adds a clearly labelled across-group mean/std only for like-for-like
method/reward/metric rows.  It does not pool per-task samples across groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-summary", type=Path, action="append", required=True,
                        help="Repeat once for each group_summary.csv.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def finite(value: str | None) -> float | None:
    try:
        result = float(value) if value not in (None, "", "None") else None
    except ValueError:
        return None
    return result if result is not None and math.isfinite(result) else None


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    seen_rows: set[tuple[str, str, str, str]] = set()
    for path in args.group_summary:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if not source_rows:
            raise ValueError(f"No rows in {path}")
        for row in source_rows:
            key = (row.get("method", ""), row.get("reward", ""), row.get("group_id", ""), row.get("metric", ""))
            if key in seen_rows:
                raise ValueError(f"Duplicate method/reward/group/metric input: {key}")
            seen_rows.add(key)
            rows.append({**row, "row_kind": "per_group", "source_group_summary": str(path.resolve())})
    buckets: dict[tuple[str, str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("method", ""), row.get("reward", ""), row.get("metric", ""),
                 row.get("direction", ""), row.get("unit", ""))].append(row)
    aggregate = []
    for key, values in sorted(buckets.items()):
        means = [finite(row.get("mean")) for row in values]
        means = [value for value in means if value is not None]
        aggregate.append({
            "row_kind": "across_group_summary", "method": key[0], "reward": key[1], "metric": key[2],
            "direction": key[3], "unit": key[4], "group_count": len(values),
            "group_ids": json.dumps(sorted({row.get("group_id", "") for row in values})),
            "mean_of_group_means": float(np.mean(means)) if means else None,
            "std_sample_of_group_means": float(np.std(np.asarray(means), ddof=1)) if len(means) >= 2 else None,
            "groups_with_metric": len(means),
            "note": "Unweighted mean/std over group-level means; per-group rows are retained above.",
        })
    atomic_csv(args.output_dir / "per_group_and_aggregate_summary.csv", rows + aggregate)
    print(json.dumps({"group_rows": len(rows), "aggregate_rows": len(aggregate),
                      "output": str((args.output_dir / 'per_group_and_aggregate_summary.csv').resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
