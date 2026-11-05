"""Grid-search fair PSP pruning schedules on the DualFashion FITB cache.

The cache contains decoded x0 predictions for a nested seed pool at every step. This
script only changes the offline selection schedule; it never uses GT-Sim/GAA
to select a candidate.  Those fields are reported post-hoc for analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# Grid-search selection is reward-only.  GT metrics are post-hoc diagnostics
# and may be unavailable for supplementary particle shards.
METRICS = ("personal",)


def parse_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty integer list: {value!r}")
    return values


def parse_particle_counts(value: str) -> list[int]:
    value = value.strip()
    if "-" in value and "," not in value:
        start, end = (int(part) for part in value.split("-", 1))
        values = list(range(start, end + 1))
    else:
        values = parse_ints(value)
    if not values or any(item < 1 for item in values):
        raise ValueError("particle counts must be positive")
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-csv", type=Path, default=Path(r"E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_timestep200_personal_analysis\candidate_metrics_with_gt.csv"))
    p.add_argument("--cache-root", type=Path, default=Path(r"E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_timestep200_cache"), help="Optional cache root used for validation and provenance.")
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--reward", choices=("personal",), default="personal")
    p.add_argument("--initial-seeds-list", default="1,2,3,4,8,12,16", help="Nested particle-pool sizes, e.g. 1,2,3,4,8,12,16.")
    p.add_argument("--best-of-n-budget", type=int, default=4, help="Fair particle-step budget is this value times total steps.")
    p.add_argument("--total-steps", type=int, default=28)
    p.add_argument("--max-cutoffs", type=int, default=4, help="Maximum intermediate pruning stages; 0 permits all possible stages.")
    p.add_argument("--cutoff-times", default="2,4,6,7,8,10,12,14,16,18,20,22,24,26")
    p.add_argument("--remaining-seeds", default="16,12,8,4,3,2,1", help="Allowed survivor counts at pruning stages.")
    p.add_argument("--budget-mode", choices=("at-most", "exact"), default="at-most")
    p.add_argument("--top-k", type=int, default=50)
    return p.parse_args()


def load_rows(path: Path) -> dict[int, dict[int, dict[int, dict]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    tasks: dict[int, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            task, seed, step = int(row["task_index"]), int(row["seed"]), int(row["step"])
            row["task_index"], row["seed"], row["step"] = task, seed, step
            for metric in METRICS:
                row[metric] = float(row[metric])
            tasks[task][step][seed] = row
    if not tasks:
        raise RuntimeError(f"empty metrics CSV: {path}")
    return tasks


def validate(tasks: dict[int, dict[int, dict[int, dict]]], available_seeds: int, total_steps: int) -> None:
    errors = []
    reference_steps: list[int] | None = None
    for task, by_step in tasks.items():
        seeds = sorted(by_step.get(total_steps, {}))
        if len(seeds) != available_seeds:
            errors.append(f"task {task}: final seeds={len(seeds)} expected {available_seeds}")
        available_steps = sorted(by_step)
        if reference_steps is None:
            reference_steps = available_steps
        elif available_steps != reference_steps:
            errors.append(f"task {task}: cached steps differ from the reference task")
        for step in available_steps:
            if sorted(by_step.get(step, {})) != seeds:
                errors.append(f"task {task}: step {step} does not contain the same seed pool")
                break
    if errors:
        raise ValueError("Invalid timestep cache:\n" + "\n".join(errors[:10]))


def particle_steps(initial: int, cutoffs: tuple[int, ...], remaining: tuple[int, ...], total: int) -> int:
    value = initial * cutoffs[0]
    for index in range(len(cutoffs) - 1):
        value += remaining[index] * (cutoffs[index + 1] - cutoffs[index])
    value += remaining[-1] * (total - cutoffs[-1])
    return int(value)


def enumerate_schedules(initial: int, total: int, budget: int, budget_mode: str, max_cutoffs: int, cutoff_times: list[int], remaining: list[int]) -> list[dict]:
    valid_times = sorted({time for time in cutoff_times if 0 < time < total})
    out = []
    limit = min(max_cutoffs or initial - 1, initial - 1, len(valid_times))
    allowed_remaining = set(remaining or list(range(1, initial)))

    # Recursively enumerate legal survivor paths.  The stage limit keeps the
    # at-most-budget search tractable for K=16 while still covering richer
    # schedules than the two-stage default PSP policy.
    def visit(previous_time: int, active: int, spent: int, cuts: list[int], keeps: list[int]) -> None:
        terminal_cost = spent + active * (total - previous_time)
        is_valid_terminal = terminal_cost <= budget if budget_mode == "at-most" else terminal_cost == budget
        if is_valid_terminal:
            out.append({
                "initial_particles": initial,
                "cutoff_times": cuts[:],
                "remaining_seeds": keeps[:],
                "particle_steps": terminal_cost,
                "budget": budget,
            })
        if len(cuts) >= limit:
            return
        if budget_mode == "exact" and terminal_cost < budget:
            return
        for cutoff in valid_times:
            if cutoff <= previous_time:
                continue
            cost_at_cutoff = spent + active * (cutoff - previous_time)
            if cost_at_cutoff >= budget:
                break
            for keep in range(active - 1, 0, -1):
                if keep not in allowed_remaining:
                    continue
                visit(cutoff, keep, cost_at_cutoff, [*cuts, cutoff], [*keeps, keep])

    visit(0, initial, 0, [], [])
    return out


def percentile(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    if len(values) <= 1:
        return [1.0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank / (len(values) - 1)
    return ranks


def reward_scores(rows: list[dict], reward: str) -> dict[int, float]:
    return {row["seed"]: row["personal"] for row in rows}


def build_seed_windows(seeds: list[int], initial_particles: int) -> list[list[int]]:
    """Split the available pool into full, non-overlapping K-sized windows."""
    return [seeds[start : start + initial_particles] for start in range(0, len(seeds) - initial_particles + 1, initial_particles)]


def select_task(
    by_step: dict[int, dict[int, dict]], schedule: dict, reward: str,
    total_steps: int, seed_window: list[int],
) -> tuple[dict, list[int]]:
    active = list(seed_window)
    trace = []
    for cutoff, keep in zip(schedule["cutoff_times"], schedule["remaining_seeds"]):
        candidates = [by_step[cutoff][seed] for seed in active]
        scores = reward_scores(candidates, reward)
        active = sorted(active, key=lambda seed: (-scores[seed], seed))[:keep]
        trace.append({"step": cutoff, "keep": keep, "active_after": active[:]})
    final_candidates = [by_step[total_steps][seed] for seed in active]
    scores = reward_scores(final_candidates, reward)
    winner = sorted(active, key=lambda seed: (-scores[seed], seed))[0]
    return by_step[total_steps][winner], trace


def aggregate(selected: list[dict]) -> tuple[dict, dict]:
    values = {metric: np.asarray([row[metric] for row in selected], dtype=np.float64) for metric in METRICS}
    return (
        {metric: float(np.mean(value)) for metric, value in values.items()},
        {metric: float(np.std(value)) for metric, value in values.items()},
    )


def ranking_objective(result: dict, reward: str) -> float:
    return float(result["selected_mean"][reward])


def evaluate(
    tasks: dict[int, dict[int, dict[int, dict]]], schedule: dict, reward: str,
    total_steps: int, capture_traces: bool = False,
) -> dict:
    selected, traces = [], {}
    oracle = {metric: [] for metric in METRICS}
    seed_window_count: int | None = None
    for task, by_step in sorted(tasks.items()):
        seed_windows = build_seed_windows(sorted(by_step[total_steps]), schedule["initial_particles"])
        if seed_window_count is None:
            seed_window_count = len(seed_windows)
        elif len(seed_windows) != seed_window_count:
            raise ValueError(f"task {task} has inconsistent seed-window coverage")
        task_traces = []
        for window_index, seed_window in enumerate(seed_windows):
            row, trace = select_task(by_step, schedule, reward, total_steps, seed_window)
            selected.append(row)
            for metric in METRICS:
                oracle[metric].append(max(by_step[total_steps][seed][metric] for seed in seed_window))
            if capture_traces:
                task_traces.append({"window_index": window_index, "seed_window": seed_window, "selected_seed": row["seed"], "trace": trace})
        if capture_traces:
            traces[str(task)] = {"windows": task_traces}
    regrets = {}
    for metric in METRICS:
        oracle_values = oracle[metric]
        selected_values = [row[metric] for row in selected]
        regrets[metric] = float(np.mean(np.asarray(oracle_values) - np.asarray(selected_values)))
    selected_mean, selected_std = aggregate(selected)
    return {
        "selected_mean": selected_mean,
        "selected_std": selected_std,
        "oracle_window_mean": {metric: float(np.mean(values)) for metric, values in oracle.items()},
        "regret_mean": regrets,
        "seed_window_count": int(seed_window_count or 0),
        "selection_count": len(selected),
        "traces": traces,
    }


def main() -> None:
    args = parse_args()
    tasks = load_rows(args.metrics_csv.resolve())
    particle_counts = parse_particle_counts(args.initial_seeds_list)
    available_seeds = min(len(by_step[args.total_steps]) for by_step in tasks.values())
    if particle_counts[-1] > available_seeds:
        raise ValueError(f"requested {particle_counts[-1]} particles, but only {available_seeds} are available")
    validate(tasks, available_seeds, args.total_steps)
    reference_seeds = sorted(next(iter(tasks.values()))[args.total_steps])
    budget = args.best_of_n_budget * args.total_steps
    remaining = parse_ints(args.remaining_seeds) if args.remaining_seeds.strip() else []
    schedules = []
    for initial_particles in particle_counts:
        schedules.extend(enumerate_schedules(
            initial_particles, args.total_steps, budget, args.budget_mode, args.max_cutoffs,
            parse_ints(args.cutoff_times), remaining,
        ))
    if not schedules:
        raise ValueError("No valid schedules. Check budget, cutoff times, and remaining seeds.")
    print(f"Enumerated {len(schedules):,} {args.budget_mode}-budget schedules across K={particle_counts}.", flush=True)
    results = []
    for index, schedule in enumerate(schedules, start=1):
        evaluated = evaluate(tasks, schedule, args.reward, args.total_steps)
        # Keeping traces for every schedule would dominate memory for the
        # K=16 search. Recompute traces only for the final top-k schedules.
        results.append({
            "rank": index,
            "schedule": schedule,
            "selected_mean": evaluated["selected_mean"],
            "selected_std": evaluated["selected_std"],
            "oracle_window_mean": evaluated["oracle_window_mean"],
            "regret_mean": evaluated["regret_mean"],
            "seed_window_count": evaluated["seed_window_count"],
            "selection_count": evaluated["selection_count"],
        })
        if index % 1000 == 0 or index == len(schedules):
            print(f"Evaluated {index:,}/{len(schedules):,} schedules.", flush=True)
    results.sort(key=lambda item: (ranking_objective(item, args.reward), -item["schedule"]["particle_steps"]), reverse=True)
    for index, result in enumerate(results, start=1):
        result["rank"] = index
    for result in results[:args.top_k]:
        result["traces"] = evaluate(
            tasks, result["schedule"], args.reward, args.total_steps, capture_traces=True,
        )["traces"]
    output = (args.output_root or args.metrics_csv.parent / "grid_search" / args.reward).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "grid_search.json").write_text(json.dumps({"meta": {"metrics_csv": str(args.metrics_csv.resolve()), "cache_root": str(args.cache_root.resolve()), "reward": args.reward, "task_count": len(tasks), "initial_particles": particle_counts, "reference_task_seeds": reference_seeds, "seed_window_policy": "Per task, sorted seeds are partitioned into all complete non-overlapping K-sized windows; incomplete tails are ignored.", "seed_window_count_by_k": {str(k): len(build_seed_windows(reference_seeds, k)) for k in particle_counts}, "total_steps": args.total_steps, "budget_particle_steps": budget, "budget_mode": args.budget_mode, "max_cutoffs": args.max_cutoffs, "num_schedules": len(results), "selection_uses_gt": False}, "results": results[:args.top_k]}, indent=2), encoding="utf-8")
    summary_rows = []
    for result in results:
        row = {"rank": result["rank"], **result["schedule"]}
        for metric in METRICS:
            row[f"selected_{metric}"] = result["selected_mean"][metric]
            row[f"selected_{metric}_std"] = result["selected_std"][metric]
            row[f"oracle_window_{metric}"] = result["oracle_window_mean"][metric]
            row[f"regret_{metric}"] = result["regret_mean"][metric]
        row["seed_window_count"] = result["seed_window_count"]
        row["selection_count"] = result["selection_count"]
        summary_rows.append(row)
    with (output / "grid_search_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"output": str(output), "task_count": len(tasks), "num_schedules": len(results), "top": [{"rank": r["rank"], **r["schedule"], "objective": ranking_objective(r, args.reward), "selected_reward": r["selected_mean"].get(args.reward)} for r in results[:10]]}, indent=2))


if __name__ == "__main__":
    main()
