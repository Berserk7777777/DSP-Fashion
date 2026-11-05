"""Merge the original and supplementary FITB candidate shards into a K=16 pool.

The extra shard is intentionally reward-only.  GT-derived columns are written
as NaN placeholders and are never consumed by Personal-reward schedule search.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


POSTHOC_COLUMNS = ("gt_sim", "gaa", "gaa_color", "gaa_pattern", "gaa_material", "gaa_style")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--extra-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--steps", default=None, help="Optional comma-separated step subset retained from both shards.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    requested_steps = None if args.steps is None else {int(part.strip()) for part in args.steps.split(",") if part.strip()}
    base = [row for row in read_csv(args.base_csv) if int(row["task_index"]) < args.max_tasks and (requested_steps is None or int(row["step"]) in requested_steps)]
    extra = [row for row in read_csv(args.extra_csv) if requested_steps is None or int(row["step"]) in requested_steps]
    fields = list(base[0])
    for row in extra:
        for field in fields:
            if field not in row:
                row[field] = "nan" if field in POSTHOC_COLUMNS else ""
    merged = base + extra
    keys = [(int(row["task_index"]), int(row["step"]), int(row["seed"])) for row in merged]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/step/seed rows between shards")
    coverage: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for task, step, seed in keys:
        coverage[task][step].add(seed)
    expected_tasks = set(range(args.max_tasks))
    expected_steps = requested_steps or set(range(1, 29))
    if set(coverage) != expected_tasks or any(set(coverage[task]) != expected_steps or any(len(seeds) != 16 for seeds in coverage[task].values()) for task in coverage):
        raise ValueError("merged pool must have 16 unique seeds for every step of every task")
    merged.sort(key=lambda row: (int(row["task_index"]), int(row["step"]), int(row["seed"])))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    print({"output": str(args.output_csv), "tasks": args.max_tasks, "steps": sorted(expected_steps),
           "candidates": len(merged), "seeds_per_task_step": 16})


if __name__ == "__main__":
    main()
