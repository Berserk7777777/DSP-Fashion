"""Summarize common-task regret for the four frozen Personal-reward GOR outputs.

The greedy runners did not persist unselected candidate images, so a true
within-greedy-pool oracle cannot be reconstructed post hoc.  This script uses
an observed task-level oracle: for each task and metric, the best final output
among Standard, trajectory BoN, greedy BoN, and greedy PSP.  It is therefore a
method-comparison regret, not an estimate of the missing greedy candidate-pool
regret.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("cs", "cis", "lpips", "compatibility", "personal")
DIRECTIONS = {"cs": "higher", "cis": "higher", "lpips": "lower", "compatibility": "higher", "personal": "higher"}
METHODS = {
    "standard": "trajectory_standard",
    "bon_full": "trajectory_bon_personal",
    "bon_greedy": "bon",
    "psp_greedy": "psp",
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-metrics", type=Path,
        default=repo / "DualFashion-main/output/ifashion_gor/native_metrics_personal/per_task_metrics.csv",
    )
    parser.add_argument(
        "--greedy-native-metrics", type=Path,
        default=repo / "DualFashion-main/output/ifashion_gor_greedy_personal/native_metrics/per_task_metrics.csv",
        help="Combined native-metric CSV with label=bon and label=psp.",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=repo / "DualFashion-main/output/ifashion_gor/personal_regret",
    )
    parser.add_argument("--expected-tasks", type=int, default=200)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_sources(args: argparse.Namespace) -> dict[str, dict[int, dict]]:
    trajectory = read_csv(args.trajectory_metrics)
    greedy_native = read_csv(args.greedy_native_metrics)
    sources = {
        "standard": [row for row in trajectory if row["label"] == "trajectory_standard"],
        "bon_full": [row for row in trajectory if row["label"] == "trajectory_bon_personal"],
        "bon_greedy": [row for row in greedy_native if row["label"] == "bon"],
        "psp_greedy": [row for row in greedy_native if row["label"] == "psp"],
    }
    result: dict[str, dict[int, dict]] = {}
    for method, rows in sources.items():
        if len(rows) != args.expected_tasks:
            raise RuntimeError(f"{method}: expected {args.expected_tasks} rows, found {len(rows)}")
        by_task = {}
        for row in rows:
            task = int(row["task_index"])
            if task in by_task:
                raise RuntimeError(f"{method}: duplicate task_index={task}")
            by_task[task] = row
        if set(by_task) != set(range(args.expected_tasks)):
            raise RuntimeError(f"{method}: task coverage is not 0..{args.expected_tasks - 1}")
        result[method] = by_task
    identities = {
        method: {(int(row["task_index"]), int(row["uid"]), int(row["oid"])) for row in by_task.values()}
        for method, by_task in result.items()
    }
    first = next(iter(identities.values()))
    if any(value != first for value in identities.values()):
        raise RuntimeError("the four GOR sources do not contain the same task identities")
    return result


def main() -> None:
    args = parse_args()
    if args.expected_tasks < 1:
        raise ValueError("--expected-tasks must be positive")
    sources = load_sources(args)
    per_task: list[dict] = []
    for task in range(args.expected_tasks):
        task_rows = {method: sources[method][task] for method in METHODS}
        oracle_method: dict[str, str] = {}
        oracle_value: dict[str, float] = {}
        for metric in METRICS:
            direction = DIRECTIONS[metric]
            ordered = sorted(
                METHODS,
                key=lambda method: (
                    -float(task_rows[method][metric]) if direction == "higher" else float(task_rows[method][metric]),
                    method,
                ),
            )
            oracle_method[metric] = ordered[0]
            oracle_value[metric] = float(task_rows[ordered[0]][metric])
        for method in METHODS:
            row = task_rows[method]
            record = {
                "method": method,
                "task_index": task,
                "uid": int(row["uid"]),
                "oid": int(row["oid"]),
            }
            for metric in METRICS:
                selected = float(row[metric])
                direction = DIRECTIONS[metric]
                regret = oracle_value[metric] - selected if direction == "higher" else selected - oracle_value[metric]
                record.update({
                    f"selected_{metric}": selected,
                    f"oracle_{metric}": oracle_value[metric],
                    f"{metric}_regret": regret,
                    f"{metric}_oracle_method": oracle_method[metric],
                    f"{metric}_oracle_hit": int(abs(regret) <= 1e-12),
                })
            per_task.append(record)

    summary: list[dict] = []
    for method in METHODS:
        rows = [row for row in per_task if row["method"] == method]
        record = {"method": method, "n_tasks": len(rows)}
        for metric in METRICS:
            regret = np.asarray([float(row[f"{metric}_regret"]) for row in rows], dtype=np.float64)
            selected = np.asarray([float(row[f"selected_{metric}"]) for row in rows], dtype=np.float64)
            oracle = np.asarray([float(row[f"oracle_{metric}"]) for row in rows], dtype=np.float64)
            record.update({
                f"selected_{metric}_mean": float(selected.mean()),
                f"oracle_{metric}_mean": float(oracle.mean()),
                f"{metric}_regret_mean": float(regret.mean()),
                f"{metric}_regret_median": float(np.median(regret)),
                f"{metric}_regret_p90": float(np.quantile(regret, .90)),
                f"{metric}_oracle_hit_rate": float(np.mean([int(row[f"{metric}_oracle_hit"]) for row in rows])),
                f"{metric}_direction": DIRECTIONS[metric],
            })
        summary.append(record)

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_root / "per_task_observed_oracle_regret.csv", per_task)
    write_csv(args.output_root / "observed_oracle_regret_summary.csv", summary)
    (args.output_root / "methodology.json").write_text(json.dumps({
        "task": "DualFashion iFashion GOR, 200 tasks",
        "reward_used_for_selection": "Personal",
        "methods": {
            "standard": "frozen first trajectory candidate",
            "bon_full": "frozen trajectory-level BoN Personal selection over the full 4-candidate outfit pool",
            "bon_greedy": "frozen online item-greedy BoN Personal selection",
            "psp_greedy": "frozen online item-greedy PSP Personal selection",
        },
        "metrics": list(METRICS),
        "regret": "For each task and metric, the oracle is the best of the four final outputs; higher metrics use oracle-selected, LPIPS uses selected-oracle.",
        "important_limit": "Greedy runners did not persist unselected candidate images, so this is observed cross-method regret, not within-greedy-candidate-pool regret.",
        "source_files": {
            "trajectory": str(args.trajectory_metrics.resolve()),
            "greedy_native": str(args.greedy_native_metrics.resolve()),
        },
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output_root), "methods": list(METHODS), "tasks": args.expected_tasks}, indent=2))


if __name__ == "__main__":
    main()
