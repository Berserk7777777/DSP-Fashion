"""Summarize 8-seed terminal reward regret and post-hoc GT diagnostic gaps."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dualfashion-root", type=Path, required=True)
    parser.add_argument("--terminal-gt-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2207,
                        help="Prefix of FITB task indices to analyze; default is the complete test split.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing to write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile_rank(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values, key=lambda seed: (values[seed], seed))
    return {seed: rank / 7.0 for rank, seed in enumerate(ordered)}


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p90": float(np.quantile(array, .90)), "p95": float(np.quantile(array, .95)),
    }


def average_ranks(values: list[float]) -> np.ndarray:
    """Average ranks for ties, scaled to [0, 1] when possible."""
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks / max(len(array) - 1, 1)


def spearman(values: list[float], target: list[float]) -> float | None:
    left, right = average_ranks(values), average_ranks(target)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    root = args.dualfashion_root
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    score_paths = {
        "compatibility": root / "output/ifashion_fitb_comp/psp/psp_group0_compatibility/psp_compatibility_scores.csv",
        "personal": root / "output/ifashion_fitb_personal/psp/psp_group0_personal/psp_personal_scores.csv",
        "bottleneck": root / "output/ifashion_fitb_bottleneck/psp/psp_group0_bottleneck/psp_selection_summary.csv",
    }
    choice_paths = {
        "compatibility": root / "output/ifashion_fitb_comp/psp/psp_group0_compatibility/psp_selection_summary.csv",
        "personal": root / "output/ifashion_fitb_personal/psp/psp_group0_personal/psp_selection_summary.csv",
        "bottleneck": score_paths["bottleneck"],
    }
    for path in [*score_paths.values(), *choice_paths.values(), args.terminal_gt_metrics]:
        if not path.is_file():
            raise FileNotFoundError(path)

    compatibility: dict[int, dict[int, float]] = defaultdict(dict)
    for row in read_csv(score_paths["compatibility"]):
        if int(row["step"]) == 28:
            compatibility[int(row["task_index"])][int(row["seed"])] = float(row["compatibility_reward"])
    personal: dict[int, dict[int, float]] = defaultdict(dict)
    for row in read_csv(score_paths["personal"]):
        if int(row["step"]) == 28:
            personal[int(row["task_index"])][int(row["seed"])] = float(row["personal_reward"])
    gt: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    for row in read_csv(args.terminal_gt_metrics):
        gt[int(row["task_index"])][int(row["seed"])] = {
            "gt_sim": float(row["gt_sim"]), "gaa": float(row["gaa"]),
        }

    choices: dict[str, dict[int, dict[str, int]]] = {}
    for reward, path in choice_paths.items():
        by_task = {}
        for row in read_csv(path):
            task = int(row["task_index"])
            by_task[task] = {"seed": int(row["psp_seed"]), "uid": int(row["uid"]), "oid": int(row["oid"])}
        choices[reward] = by_task
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    task_ids = list(range(args.max_samples))
    if not set(task_ids).issubset(choices["compatibility"]):
        raise RuntimeError(f"selection summary does not contain task 0..{args.max_samples - 1}")

    per_task = []
    for task in task_ids:
        if set(compatibility[task]) != set(personal[task]) or set(compatibility[task]) != set(gt[task]) or len(gt[task]) != 8:
            raise RuntimeError(f"task {task}: terminal seed pools do not match")
        seeds = sorted(gt[task])
        b8 = {seed: min(percentile_rank(compatibility[task])[seed], percentile_rank(personal[task])[seed]) for seed in seeds}
        metrics = {
            "compatibility": compatibility[task], "personal": personal[task], "bottleneck_b8_rank": b8,
            "gt_sim": {seed: gt[task][seed]["gt_sim"] for seed in seeds},
            "gaa": {seed: gt[task][seed]["gaa"] for seed in seeds},
        }
        oracles = {name: max(values, key=lambda seed: (values[seed], -seed)) for name, values in metrics.items()}
        for selection_reward, selected_data in choices.items():
            chosen_data = selected_data[task]
            chosen = chosen_data["seed"]
            if chosen not in seeds:
                raise RuntimeError(f"task {task}: {selection_reward} chose seed outside terminal pool")
            row = {"task_index": task, "uid": chosen_data["uid"], "oid": chosen_data["oid"],
                   "selection_reward": selection_reward, "selected_seed": chosen}
            for name, values in metrics.items():
                oracle = oracles[name]
                row[f"selected_{name}"] = values[chosen]
                row[f"oracle_{name}"] = values[oracle]
                row[f"{name}_gap"] = values[oracle] - values[chosen]
                row[f"{name}_oracle_seed"] = oracle
                row[f"{name}_oracle_hit"] = int(chosen == oracle)
            per_task.append(row)
    write_csv(output / "per_task_terminal_regret.csv", per_task)

    # Reward alignment is rank-based within a task, avoiding incomparable
    # score scales across users/outfits.  It is still post-hoc test analysis.
    rank_alignment = []
    for reward_name, reward_values in (
        ("compatibility", compatibility),
        ("personal", personal),
    ):
        for target_name in ("gt_sim", "gaa"):
            correlations = []
            for task in task_ids:
                seeds = sorted(gt[task])
                target = [gt[task][seed][target_name] for seed in seeds]
                value = spearman([reward_values[task][seed] for seed in seeds], target)
                if value is not None:
                    correlations.append(value)
            record = {"reward_model": reward_name, "posthoc_target": target_name,
                      "n_valid_tasks": len(correlations)}
            for name, value in stats(correlations).items():
                record[f"within_task_spearman_{name}"] = value
            record["fraction_positive"] = float(np.mean(np.asarray(correlations) > 0))
            rank_alignment.append(record)
    for target_name in ("gt_sim", "gaa"):
        correlations = []
        for task in task_ids:
            seeds = sorted(gt[task])
            c_rank, p_rank = percentile_rank(compatibility[task]), percentile_rank(personal[task])
            b8 = [min(c_rank[seed], p_rank[seed]) for seed in seeds]
            target = [gt[task][seed][target_name] for seed in seeds]
            value = spearman(b8, target)
            if value is not None:
                correlations.append(value)
        record = {"reward_model": "bottleneck_b8_rank", "posthoc_target": target_name,
                  "n_valid_tasks": len(correlations)}
        for name, value in stats(correlations).items():
            record[f"within_task_spearman_{name}"] = value
        record["fraction_positive"] = float(np.mean(np.asarray(correlations) > 0))
        rank_alignment.append(record)
    write_csv(output / "reward_gt_rank_alignment.csv", rank_alignment)

    summary = []
    own_column = {"compatibility": "compatibility", "personal": "personal", "bottleneck": "bottleneck_b8_rank"}
    for reward in ("compatibility", "personal", "bottleneck"):
        rows = [row for row in per_task if row["selection_reward"] == reward]
        record = {"selection_reward": reward, "n_tasks": len(rows), "own_regret_metric": own_column[reward]}
        for metric in ("compatibility", "personal", "bottleneck_b8_rank", "gt_sim", "gaa"):
            gap = [float(row[f"{metric}_gap"]) for row in rows]
            metric_stats = stats(gap)
            for key, value in metric_stats.items():
                record[f"{metric}_gap_{key}"] = value
            record[f"{metric}_oracle_hit_rate"] = float(np.mean([int(row[f"{metric}_oracle_hit"]) for row in rows]))
            record[f"selected_{metric}_mean"] = float(np.mean([float(row[f"selected_{metric}"]) for row in rows]))
            record[f"oracle_{metric}_mean"] = float(np.mean([float(row[f"oracle_{metric}"]) for row in rows]))
        summary.append(record)
    write_csv(output / "terminal_regret_summary.csv", summary)
    (output / "methodology.json").write_text(json.dumps({
        "terminal_pool": "the same eight generated step-28 candidates for each FITB task",
        "optimization_regret": "max over terminal pool of the selection reward minus the selected terminal reward",
        "bottleneck": "B8=min(full-8 compatibility percentile, full-8 personal percentile), with ties sorted by seed; it is an ordinal rank regret",
        "gt_diagnostic_gap": "max over terminal pool of post-hoc GT-Sim/GAA minus the selected metric; not an optimization reward or tuning objective",
        "gt_warning": "GT identifies one held-out item, while FITB may admit multiple valid personalized items.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": summary, "rank_alignment": rank_alignment}, indent=2))


if __name__ == "__main__":
    main()
