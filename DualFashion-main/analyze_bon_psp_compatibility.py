"""Aggregate the three-group BoN/PSP compatibility runs and PSP regret."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean, median, pstdev


def fmean(values):
    return float(mean(values)) if values else float("nan")


def fpstd(values):
    return float(pstdev(values)) if len(values) > 1 else 0.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/compatibility_bon_psp_analysis"))
    return parser.parse_args()


def load_bon(root: Path):
    rows = []
    pattern = re.compile(r"^bon4_group(\d+)_seed(\d+)_scores$")
    for path in sorted(root.glob("bon4_group*_seed*_scores/candidate_compatibility_scores.csv")):
        match = pattern.match(path.parent.name)
        if not match:
            continue
        group = int(match.group(1))
        with path.open(newline="", encoding="utf-8") as handle:
            candidates = list(csv.DictReader(handle))
        by_task = {}
        for row in candidates:
            key = (int(row["uid"]), int(row["oid"]))
            by_task.setdefault(key, []).append(row)
        for (uid, oid), task_rows in sorted(by_task.items()):
            task_rows.sort(key=lambda row: int(row["candidate_idx"]))
            scores = [float(row["compatibility_reward"]) for row in task_rows]
            winner_index = max(range(len(scores)), key=lambda idx: scores[idx])
            winner = task_rows[winner_index]
            rows.append(
                {
                    "group": group,
                    "uid": uid,
                    "oid": oid,
                    "bon_candidate_count": len(scores),
                    "bon_pool_mean": fmean(scores),
                    "bon_standard_reward": scores[0],
                    "bon_reward": scores[winner_index],
                    "bon_seed": int(winner["seed"]),
                    "bon_candidate_idx": winner_index,
                }
            )
    return rows


def load_psp(root: Path):
    rows = []
    pattern = re.compile(r"^psp_group(\d+)_task(\d+)_seed(\d+)$")
    for path in sorted(root.glob("psp_group*_task*_seed*/metadata.json")):
        match = pattern.match(path.parent.name)
        if not match:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        audit = list(payload.get("audit_candidates", []))
        final = list(payload.get("final", []))
        if not audit or not final:
            raise RuntimeError(f"Missing audit/final records: {path}")
        audit.sort(key=lambda row: float(row["compatibility_reward"]), reverse=True)
        psp_final = final[0]
        oracle = audit[0]
        final_by_seed = {int(row["seed"]): float(row["compatibility_reward"]) for row in audit}
        psp_seed = int(psp_final["seed"])
        psp_reward = float(psp_final["reward"])
        trace = {int(row["step"]): row for row in payload.get("trace", [])}
        oracle_seed = int(oracle["seed"])
        rows.append(
            {
                "group": int(match.group(1)),
                "task_index": int(match.group(2)),
                "uid": int(payload["uid"]),
                "oid": int(payload["oid"]),
                "psp_candidate_count": len(audit),
                "psp_reward": psp_reward,
                "psp_seed": psp_seed,
                "psp_rank_among_8": 1 + sum(float(row["compatibility_reward"]) > psp_reward for row in audit),
                "psp_oracle_reward": float(oracle["compatibility_reward"]),
                "psp_oracle_seed": oracle_seed,
                "psp_regret": float(oracle["compatibility_reward"]) - psp_reward,
                "psp_oracle_hit": psp_seed == oracle_seed,
                "psp_oracle_survived_step7": oracle_seed in set(trace.get(7, {}).get("survivor_seeds", [])),
                "psp_oracle_survived_step14": oracle_seed in set(trace.get(14, {}).get("survivor_seeds", [])),
                "psp_oracle_survived_step28": oracle_seed in set(trace.get(28, {}).get("survivor_seeds", [])),
                "psp_final_vs_audit_gap": psp_reward - final_by_seed.get(psp_seed, float("nan")),
                "psp_metadata": str(path.resolve()),
            }
        )
    return rows


def merge_rows(bon_rows, psp_rows):
    bon = {(row["group"], row["uid"], row["oid"]): row for row in bon_rows}
    psp = {(row["group"], row["uid"], row["oid"]): row for row in psp_rows}
    keys = sorted(set(bon) & set(psp))
    if len(keys) != len(bon) or len(keys) != len(psp):
        raise RuntimeError(f"BoN/PSP task mismatch: bon={len(bon)}, psp={len(psp)}, overlap={len(keys)}")
    rows = []
    for key in keys:
        row = dict(bon[key])
        row.update({key: value for key, value in psp[key].items() if key not in {"group", "uid", "oid"}})
        row["bon_minus_psp"] = row["bon_reward"] - row["psp_reward"]
        row["oracle_minus_bon"] = row["psp_oracle_reward"] - row["bon_reward"]
        rows.append(row)
    return rows


def summarize_group(rows):
    return {
        "group": rows[0]["group"],
        "task_count": len(rows),
        "bon_standard_mean": fmean([row["bon_standard_reward"] for row in rows]),
        "bon_pool_mean": fmean([row["bon_pool_mean"] for row in rows]),
        "bon_mean": fmean([row["bon_reward"] for row in rows]),
        "psp_mean": fmean([row["psp_reward"] for row in rows]),
        "psp_oracle_mean": fmean([row["psp_oracle_reward"] for row in rows]),
        "psp_regret_mean": fmean([row["psp_regret"] for row in rows]),
        "psp_regret_median": median([row["psp_regret"] for row in rows]),
        "psp_regret_max": max(row["psp_regret"] for row in rows),
        "psp_oracle_hit_count": sum(bool(row["psp_oracle_hit"]) for row in rows),
        "psp_oracle_hit_rate": fmean([float(row["psp_oracle_hit"]) for row in rows]),
        "psp_rank_mean": fmean([row["psp_rank_among_8"] for row in rows]),
        "oracle_survival_step7_rate": fmean([float(row["psp_oracle_survived_step7"]) for row in rows]),
        "oracle_survival_step14_rate": fmean([float(row["psp_oracle_survived_step14"]) for row in rows]),
        "oracle_survival_step28_rate": fmean([float(row["psp_oracle_survived_step28"]) for row in rows]),
    }


def summarize_task(rows):
    return {
        "task_index": rows[0]["task_index"],
        "uid": rows[0]["uid"],
        "oid": rows[0]["oid"],
        "group_count": len(rows),
        "bon_mean": fmean([row["bon_reward"] for row in rows]),
        "psp_mean": fmean([row["psp_reward"] for row in rows]),
        "psp_oracle_mean": fmean([row["psp_oracle_reward"] for row in rows]),
        "psp_regret_mean": fmean([row["psp_regret"] for row in rows]),
        "psp_regret_max": max(row["psp_regret"] for row in rows),
        "psp_oracle_hit_count": sum(bool(row["psp_oracle_hit"]) for row in rows),
        "psp_oracle_hit_rate": fmean([float(row["psp_oracle_hit"]) for row in rows]),
        "psp_rank_mean": fmean([row["psp_rank_among_8"] for row in rows]),
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bon_rows = load_bon(args.root)
    psp_rows = load_psp(args.root)
    merged = merge_rows(bon_rows, psp_rows)
    group_rows = [summarize_group([row for row in merged if row["group"] == group]) for group in sorted({row["group"] for row in merged})]
    task_rows = [summarize_task([row for row in merged if row["uid"] == uid and row["oid"] == oid]) for uid, oid in sorted({(row["uid"], row["oid"]) for row in merged})]
    task_rows.sort(key=lambda row: row["task_index"])
    group_mean_keys = [
        "bon_standard_mean", "bon_pool_mean", "bon_mean", "psp_mean", "psp_oracle_mean",
        "psp_regret_mean", "psp_regret_median", "psp_regret_max", "psp_oracle_hit_rate",
        "psp_rank_mean", "oracle_survival_step7_rate", "oracle_survival_step14_rate", "oracle_survival_step28_rate",
    ]
    overall = {"group_count": len(group_rows), "task_count_per_group": len(merged) // len(group_rows)}
    for key in group_mean_keys:
        overall[key] = fmean([row[key] for row in group_rows])
    overall["psp_oracle_hit_count_total"] = sum(row["psp_oracle_hit_count"] for row in group_rows)
    overall["psp_oracle_hit_count_possible"] = sum(row["task_count"] for row in group_rows)
    overall["psp_regret_std_across_group_means"] = fpstd([row["psp_regret_mean"] for row in group_rows])
    overall["psp_regret_all_task_mean"] = fmean([row["psp_regret"] for row in merged])
    overall["psp_regret_all_task_median"] = median([row["psp_regret"] for row in merged])
    overall["psp_regret_all_task_max"] = max(row["psp_regret"] for row in merged)

    write_csv(args.output_dir / "bon_psp_by_task.csv", merged)
    write_csv(args.output_dir / "bon_psp_by_group.csv", group_rows)
    write_csv(args.output_dir / "bon_psp_by_task_aggregate.csv", task_rows)
    report = {"overall": overall, "groups": group_rows, "task_aggregate": task_rows, "tasks": merged}
    (args.output_dir / "bon_psp_regret_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall": overall, "groups": group_rows}, indent=2))


if __name__ == "__main__":
    main()
