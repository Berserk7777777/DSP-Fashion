"""Summarize native metrics and six-seed Personal regret for FITB.

The selected metrics are read from the frozen six-seed native-metric output.
The regret oracle is post-hoc only: it is the largest step-28 Personal score
among slots 0..5 in the existing PSP offline cache score file.  No candidate
score from a future step is used by the online selector here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    dual = repo / "DualFashion-main"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, required=True,
                        help="Frozen six-seed payload directory containing 0057501/gen.npy")
    parser.add_argument("--native-metrics", type=Path, required=True,
                        help="Output directory from evaluate_dualfashion_fitb_native_metrics.py")
    parser.add_argument("--offline-score-csv", type=Path, default=
                        dual / "output/ifashion_fitb_personal/psp/psp_group0_personal/psp_personal_scores.csv")
    parser.add_argument("--offline-cache-root", type=Path, default=
                        dual / "output/ifashion_fitb_comp/psp_offline_cache")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=2207)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--seed-stride", type=int, default=10000)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
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


def main() -> None:
    args = parse_args()
    if args.max_tasks < 1 or args.base_seed < 0 or args.seed_stride < 1:
        raise ValueError("invalid task or seed arguments")
    gen_path = args.selection_root / "gen.npy"
    native_path = args.native_metrics / "per_task_metrics.csv"
    for path in (gen_path, native_path, args.offline_score_csv, args.offline_cache_root):
        if not path.is_file():
            if not path.is_dir():
                raise FileNotFoundError(path)

    generated = np.load(gen_path, allow_pickle=True).item()
    selected_by_identity: dict[tuple[int, int], dict] = {}
    for raw_uid, oid_map in generated.items():
        uid = int(raw_uid)
        for raw_oid, record in oid_map.items():
            oid = int(raw_oid)
            paths = list(record.get("image_paths", []))
            seeds = list(record.get("candidate_seeds", []))
            if len(paths) != 1 or len(seeds) != 1:
                raise RuntimeError(f"{uid}_{oid}: expected one selected image and seed")
            selected_by_identity[(uid, oid)] = {
                "selected_seed": int(seeds[0]),
                "selected_image_path": str(Path(paths[0]).resolve()),
            }
    native = [row for row in read_csv(native_path) if row.get("label") == "six_seed_7_21"]
    native_by_identity = {(int(row["uid"]), int(row["oid"])): row for row in native}
    if len(native_by_identity) != args.max_tasks or len(selected_by_identity) != args.max_tasks:
        raise RuntimeError(
            f"expected {args.max_tasks} tasks, native={len(native_by_identity)}, "
            f"selection={len(selected_by_identity)}"
        )

    offline_rows = read_csv(args.offline_score_csv)
    first6: dict[int, dict[int, dict]] = {}
    for row in offline_rows:
        if int(row["step"]) != 28:
            continue
        task = int(row["task_index"])
        if not 0 <= task < args.max_tasks:
            continue
        seed = int(row["seed"])
        expected = args.base_seed + task * args.seed_stride
        if expected <= seed < expected + 6:
            first6.setdefault(task, {})[seed] = row
    if set(first6) != set(range(args.max_tasks)) or any(len(first6[t]) != 6 for t in range(args.max_tasks)):
        bad = [t for t in range(args.max_tasks) if len(first6.get(t, {})) != 6]
        raise RuntimeError(f"offline first-six terminal pool incomplete; bad tasks: {bad[:10]}")

    rows: list[dict] = []
    for task in range(args.max_tasks):
        # The offline score file is the canonical task-index to identity map.
        task_candidates = first6[task]
        identity = (int(next(iter(task_candidates.values()))["uid"]),
                    int(next(iter(task_candidates.values()))["oid"]))
        chosen = selected_by_identity.get(identity)
        selected_metrics = native_by_identity.get(identity)
        if chosen is None or selected_metrics is None:
            raise RuntimeError(f"task {task}: six-seed/native identity mismatch for {identity}")
        expected_seed = args.base_seed + task * args.seed_stride
        if chosen["selected_seed"] not in range(expected_seed, expected_seed + 6):
            raise RuntimeError(
                f"task {task}: selected seed {chosen['selected_seed']} is outside six-seed pool "
                f"[{expected_seed}, {expected_seed + 5}]"
            )
        oracle_seed, oracle_row = max(
            task_candidates.items(), key=lambda item: (float(item[1]["personal_reward"]), -item[0])
        )
        oracle_path = Path(oracle_row["image_path"]).resolve()
        cache_root = args.offline_cache_root.resolve()
        if not oracle_path.is_file() or cache_root not in oracle_path.parents:
            raise RuntimeError(f"task {task}: oracle image is outside offline cache: {oracle_path}")
        selected_personal = float(selected_metrics["personal"])
        selected_offline_personal = float(task_candidates[chosen["selected_seed"]]["personal_reward"])
        oracle_personal = float(oracle_row["personal_reward"])
        record = {
            "task_index": task,
            "uid": identity[0],
            "oid": identity[1],
            "selected_seed": chosen["selected_seed"],
            "selected_image_path": chosen["selected_image_path"],
            "selected_personal": selected_personal,
            "selected_offline_personal": selected_offline_personal,
            "oracle_first6_seed": oracle_seed,
            "oracle_first6_image_path": str(oracle_path),
            "oracle_first6_personal": oracle_personal,
            "personal_regret_first6": oracle_personal - selected_personal,
            "personal_regret_first6_seed_pool": oracle_personal - selected_offline_personal,
            "first6_oracle_hit": int(chosen["selected_seed"] == oracle_seed),
        }
        for metric in ("cs", "cis", "lpips", "compatibility"):
            record[f"selected_{metric}"] = float(selected_metrics[metric])
        rows.append(record)

    write_csv(args.output_root / "per_task_metrics_with_first6_personal_regret.csv", rows)
    selected = np.asarray([row["selected_personal"] for row in rows], dtype=np.float64)
    oracle = np.asarray([row["oracle_first6_personal"] for row in rows], dtype=np.float64)
    regret = np.asarray([row["personal_regret_first6"] for row in rows], dtype=np.float64)
    pool_regret = np.asarray([row["personal_regret_first6_seed_pool"] for row in rows], dtype=np.float64)
    summary = {
        "method": "six_seed_7_21",
        "n_tasks": len(rows),
        "selected_metrics_mean": {
            metric: float(np.mean([row[f"selected_{metric}"] for row in rows]))
            for metric in ("cs", "cis", "lpips", "compatibility", "personal")
        },
        "personal_regret_first6": {
            "oracle_personal_mean": float(oracle.mean()),
            "selected_personal_mean": float(selected.mean()),
            "selected_offline_personal_mean": float(np.mean([row["selected_offline_personal"] for row in rows])),
            "mean": float(pool_regret.mean()),
            "median": float(np.median(pool_regret)),
            "p90": float(np.quantile(pool_regret, 0.90)),
            "max": float(pool_regret.max()),
            "oracle_hit_rate": float(np.mean([row["first6_oracle_hit"] for row in rows])),
            "oracle_hit_count": int(sum(row["first6_oracle_hit"] for row in rows)),
            "signed_png_mean": float(regret.mean()),
            "signed_png_median": float(np.median(regret)),
            "signed_png_p90": float(np.quantile(regret, 0.90)),
            "signed_png_negative_count": int(np.sum(regret < 0)),
        },
        "pool": {
            "candidate_count": 6,
            "steps": [28],
            "base_seed": args.base_seed,
            "seed_stride": args.seed_stride,
            "source": str(args.offline_score_csv.resolve()),
            "cache_root": str(args.offline_cache_root.resolve()),
        },
        "metric_sources": {
            "selected": str(native_path.resolve()),
            "offline_oracle": str(args.offline_score_csv.resolve()),
        },
        "warning": "The 2207 tasks are a development/online-replay set, not an untouched independent test claim.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_root / "report.md").write_text(
        "# six-seed 7/21 metrics and Personal regret\n\n"
        f"Tasks: {len(rows)}. First-6 oracle is the maximum step-28 Personal among slots 0..5.\n\n"
        f"- CS mean: {summary['selected_metrics_mean']['cs']:.6f}\n"
        f"- CIS mean: {summary['selected_metrics_mean']['cis']:.6f}\n"
        f"- LPIPS mean: {summary['selected_metrics_mean']['lpips']:.6f}\n"
        f"- Compatibility mean: {summary['selected_metrics_mean']['compatibility']:.6f}\n"
        f"- Personal mean: {summary['selected_metrics_mean']['personal']:.6f}\n"
        f"- First-6 oracle Personal mean: {summary['personal_regret_first6']['oracle_personal_mean']:.6f}\n"
        f"- First-6 Personal regret mean (same offline seed pool): {summary['personal_regret_first6']['mean']:.6f}\n"
        f"- First-6 oracle hit rate: {summary['personal_regret_first6']['oracle_hit_rate']:.6%} "
        f"({summary['personal_regret_first6']['oracle_hit_count']}/{len(rows)})\n"
        f"- Signed regret against selected online PNG mean: {summary['personal_regret_first6']['signed_png_mean']:.6f}\n\n"
        "These 2207 tasks were used for development/online replay and are not an untouched independent test set.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
