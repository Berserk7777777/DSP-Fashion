"""Analyze how Personal reward at each timestep affects FITB PSP selection.

This consumes the completed 200-task offline analysis artifacts.  It does not
load CLIP, decode images, or use GT for selection.  GT-Sim/GAA are used only
post-hoc to measure predictive value, terminal oracle regret, and seed survival.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("personal", "compatibility", "gt_sim", "gaa")
GAA_ATTRIBUTES = ("gaa_color", "gaa_pattern", "gaa_material", "gaa_style")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx, ry = rankdata(np.asarray(x, dtype=np.float64)), rankdata(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    return float(np.dot(rx, ry) / denominator) if denominator > 0 else 0.0


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std_or_nan(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path(r"E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_timestep200_personal_analysis"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def load_rows(root: Path) -> list[dict]:
    rows = []
    for row in read_csv(root / "candidate_metrics_with_gt.csv"):
        converted = {key: value for key, value in row.items()}
        for key in ("task_index", "uid", "oid", "seed", "step"):
            converted[key] = int(converted[key])
        for key in METRICS + GAA_ATTRIBUTES:
            converted[key] = float(converted[key])
        rows.append(converted)
    if not rows:
        raise RuntimeError("candidate_metrics_with_gt.csv is empty")
    return rows


def step_analysis(rows: list[dict]) -> list[dict]:
    by_step: dict[int, list[dict]] = defaultdict(list)
    by_task_seed: dict[tuple[int, int], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_step[row["step"]].append(row)
        by_task_seed[(row["task_index"], row["seed"])][row["step"]] = row

    terminal = {key: values[28] for key, values in by_task_seed.items() if 28 in values}
    output = []
    for step in sorted(by_step):
        current = by_step[step]
        same_gt = [row["gt_sim"] for row in current]
        same_gaa = [row["gaa"] for row in current]
        same_personal = [row["personal"] for row in current]
        same_compat = [row["compatibility"] for row in current]
        terminal_gt = [terminal[(row["task_index"], row["seed"])] ["gt_sim"] for row in current]
        terminal_personal = [terminal[(row["task_index"], row["seed"])] ["personal"] for row in current]
        terminal_gaa = [terminal[(row["task_index"], row["seed"])] ["gaa"] for row in current]
        within_task_std = []
        for task in sorted({row["task_index"] for row in current}):
            values = [row["personal"] for row in current if row["task_index"] == task]
            within_task_std.append(float(np.std(values, ddof=1)))
        output.append({
            "step": step,
            "n_candidates": len(current),
            "personal_mean": mean_or_nan(same_personal),
            "personal_std": std_or_nan(same_personal),
            "personal_within_task_std_mean": mean_or_nan(within_task_std),
            "compatibility_mean": mean_or_nan(same_compat),
            "compatibility_std": std_or_nan(same_compat),
            "gt_sim_mean": mean_or_nan(same_gt),
            "gt_sim_std": std_or_nan(same_gt),
            "gaa_mean": mean_or_nan(same_gaa),
            "gaa_std": std_or_nan(same_gaa),
            **{f"{attribute}_mean": mean_or_nan([row[attribute] for row in current]) for attribute in GAA_ATTRIBUTES},
            **{f"personal_vs_{attribute}_spearman": spearman(same_personal, [row[attribute] for row in current]) for attribute in GAA_ATTRIBUTES},
            "personal_vs_same_step_gt_sim_spearman": spearman(same_personal, same_gt),
            "personal_vs_same_step_gaa_spearman": spearman(same_personal, same_gaa),
            "personal_vs_terminal_gt_sim_spearman": spearman(same_personal, terminal_gt),
            "personal_vs_terminal_gaa_spearman": spearman(same_personal, terminal_gaa),
            "personal_vs_terminal_personal_spearman": spearman(same_personal, terminal_personal),
        })
    return output


def terminal_oracles(rows: list[dict]) -> dict[tuple[int, int], dict]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["step"] == 28:
            grouped[(row["task_index"], row["seed"])].append(row)
    return {key: values[0] for key, values in grouped.items()}


def survival_analysis(root: Path, rows: list[dict]) -> list[dict]:
    traces = [json.loads(line) for line in (root / "selection_traces.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    terminal = terminal_oracles(rows)
    by_task: dict[int, list[dict]] = defaultdict(list)
    for (task, _seed), row in terminal.items():
        by_task[task].append(row)
    targets = {"terminal_personal_best": "personal", "terminal_gt_sim_best": "gt_sim", "terminal_gaa_best": "gaa"}
    target_seeds = {}
    for task, candidates in by_task.items():
        target_seeds[task] = {}
        for target_name, metric in targets.items():
            target_seeds[task][target_name] = max(candidates, key=lambda row: (row[metric], -row["seed"]))["seed"]

    output = []
    for trace in traces:
        task = int(trace["task_index"])
        schedule = trace["schedule"]
        for node in trace["trace"]:
            active_after = {int(seed) for seed in node["active_after"]}
            for target_name, seed in target_seeds[task].items():
                output.append({
                    "task_index": task,
                    "schedule": schedule,
                    "step": int(node["step"]),
                    "target": target_name,
                    "target_seed": seed,
                    "survived": int(seed in active_after),
                    "active_before": len(node["active_before"]),
                    "active_after": len(node["active_after"]),
                })
    return output


def schedule_analysis(root: Path, rows: list[dict]) -> list[dict]:
    selected = read_csv(root / "selected_metrics_with_gt.csv")
    terminal = terminal_oracles(rows)
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        grouped[row["schedule"]][row["method"]].append(row)
    output = []
    for schedule, methods in sorted(grouped.items()):
        psp = methods["psp"]
        for metric in ("personal", "compatibility", "gt_sim", "gaa"):
            values = [float(row[metric]) for row in psp]
            oracle_values = []
            for row in psp:
                candidates = [value for (task, _seed), value in terminal.items() if task == int(row["task_index"])]
                oracle_values.append(max(float(value[metric]) for value in candidates))
            output.append({
                "schedule": schedule,
                "metric": metric,
                "psp_mean": mean_or_nan(values),
                "terminal_oracle8_mean": mean_or_nan(oracle_values),
                "regret_mean": mean_or_nan([oracle - value for oracle, value in zip(oracle_values, values)]),
            })
    return output


def main() -> None:
    args = parse_args()
    root = args.analysis_root.resolve()
    output = (args.output_root or root / "mechanism_analysis").resolve()
    rows = load_rows(root)
    step_rows = step_analysis(rows)
    survival_rows = survival_analysis(root, rows)
    schedule_rows = schedule_analysis(root, rows)
    write_csv(output / "step_mechanism_summary.csv", step_rows)
    write_csv(output / "seed_survival.csv", survival_rows)
    write_csv(output / "schedule_oracle_regret.csv", schedule_rows)

    survival_summary = []
    for key in sorted({(row["schedule"], row["step"], row["target"]) for row in survival_rows}):
        schedule, step, target = key
        values = [row["survived"] for row in survival_rows if (row["schedule"], row["step"], row["target"]) == key]
        survival_summary.append({"schedule": schedule, "step": step, "target": target, "survival_rate": mean_or_nan(values), "n_tasks": len(values)})
    write_csv(output / "seed_survival_summary.csv", survival_summary)

    manifest = {
        "analysis_root": str(root),
        "output_root": str(output),
        "task_count": len({row["task_index"] for row in rows}),
        "candidate_count": len(rows),
        "steps": sorted({row["step"] for row in rows}),
        "interpretation": {
            "step_correlation": "Personal at step t is compared with same-candidate terminal GT-Sim/GAA/Personal; this is predictive analysis, not selection input.",
            "survival": "Terminal best seeds are defined post-hoc and survival is measured after each Personal-based pruning node.",
            "oracle_regret": "Terminal oracle8 minus the PSP-selected terminal metric, per task; higher is better for Personal/Compatibility/GT-Sim/GAA.",
            "lpips": "LPIPS is not present in the existing timestep cache analysis artifact and is not inferred from GT-Sim.",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "steps": len(step_rows), "survival_rows": len(survival_rows), "schedule_metric_rows": len(schedule_rows)}, indent=2))


if __name__ == "__main__":
    main()
