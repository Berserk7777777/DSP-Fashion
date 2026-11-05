"""Create one comparable table for trajectory and item-greedy GOR metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED = {
    "trajectory_standard": ("Trajectory Standard", "trajectory-level offline"),
    "trajectory_bon_personal": ("Trajectory BoN (Personal)", "trajectory-level offline"),
    "bon": ("Item-Greedy BoN (Personal)", "online item-greedy"),
    "psp": ("Item-Greedy PSP (Personal)", "online item-greedy"),
}
METRICS = ("cs", "cis", "lpips", "compatibility", "personal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-summary", type=Path, required=True)
    parser.add_argument("--greedy-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_summary(path: Path) -> dict[str, dict[str, dict]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, dict[str, dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["label"], {})[row["metric"]] = row
    return result


def main() -> None:
    args = parse_args()
    collected = {}
    collected.update(read_summary(args.trajectory_summary))
    collected.update(read_summary(args.greedy_summary))
    if set(collected) != set(EXPECTED):
        raise ValueError(f"expected labels {sorted(EXPECTED)}, found {sorted(collected)}")
    rows = []
    for key, (method, protocol) in EXPECTED.items():
        values = collected[key]
        if set(values) != set(METRICS):
            raise ValueError(f"{key}: expected metrics {METRICS}, found {sorted(values)}")
        counts = {int(values[metric]["n_tasks"]) for metric in METRICS}
        if counts != {200}:
            raise ValueError(f"{key}: expected 200 tasks, found {counts}")
        row = {"method": method, "protocol": protocol, "n_tasks": 200}
        for metric in METRICS:
            row[metric] = float(values[metric]["mean"])
            row[f"{metric}_std_sample"] = float(values[metric]["std_sample"])
        rows.append(row)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "protocol.json").write_text(json.dumps({
        "task": "iFashion GOR, 200 tasks",
        "reward": "frozen Personal reward for both BoN selections",
        "metrics": {
            "cs": "mean item-to-target-category-prompt OpenCLIP ViT-H/14 cosine * 100",
            "cis": "post-hoc matched generated-to-GT OpenCLIP ViT-H/14 cosine * 100",
            "lpips": "post-hoc matched generated-to-GT LPIPS-VGG; lower is better",
            "compatibility": "sigmoid released FashionEvaluator on four raw OpenCLIP ViT-H/14 generated features",
            "personal": "mean generated-to-user-history category cosine * 100",
        },
        "selection_and_gt_policy": "All selections were frozen before CS/CIS/LPIPS evaluation. GT is post-hoc only.",
        "comparison_caveat": "Trajectory and item-greedy rows share metrics and task set but differ in search unit, item-order policy, and context evolution; this table is descriptive, not a controlled causal ablation.",
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
