"""Compare frozen online PSP6 with the same first-six terminal Personal pool.

No model inference or selection is performed. Primary regret uses both the
oracle and selected seed score from one immutable offline table, avoiding
CPU/GPU or evaluator-batch rounding differences.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("cs", "cis", "lpips", "compatibility", "personal")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path, rows):
    if not rows:
        raise ValueError("No rows to report")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--candidate-scores", type=Path, required=True)
    p.add_argument("--native-metrics", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    config = json.loads((args.run_root / "run_config.json").read_text(encoding="utf-8"))
    if config["initial_particles"] != 6 or config["schedule"] != [[13, 4], [39, 2], [51, 1]]:
        raise ValueError("Expected the frozen native DiFashion 6-seed 13/39/final run")
    native_rows = read_csv(args.native_metrics)
    native = {row["task_id"]: row for row in native_rows}
    expected_count = int(config["total_dataset_tasks"])
    if len(native) != len(native_rows) or len(native) != expected_count:
        raise ValueError("Duplicate or missing native task metrics")
    pools, seen = {}, set()
    for row in read_csv(args.candidate_scores):
        if row["method"] != "psp" or row["node"] != "final" or int(row["group_id"]) != 0:
            continue
        task_id, slot = row["task_id"], int(row["candidate_index"])
        key = (task_id, slot)
        if key in seen:
            raise ValueError(f"Duplicate terminal candidate: {key}")
        seen.add(key)
        if row["personal_missing"].lower() != "false" or not math.isfinite(float(row["personal_sim"])):
            raise ValueError(f"Missing/invalid terminal Personal score: {key}")
        pools.setdefault(task_id, {})[slot] = row
    if set(pools) != set(native):
        raise ValueError("Offline PSP terminal tasks and online metrics do not match")
    rows, node_rows, invalid, selected_hash_mismatches = [], [], [], []
    used_seeds = set()
    source_hash = hashlib.sha256()
    for task_id in sorted(native, key=lambda t: int(native[t]["task_index"])):
        try:
            task = native[task_id]
            folder = args.run_root / task_id
            summary_path = folder / "task_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            source_hash.update(summary_path.read_bytes())
            if summary["task_id"] != task_id or summary["status"] != "complete":
                raise ValueError("Online task summary identity/status mismatch")
            pool = pools[task_id]
            if set(pool) != set(range(8)):
                raise ValueError("Expected eight original terminal slots; first six are slots 0..5")
            first6 = [pool[slot] for slot in range(6)]
            seeds = [int(row["seed"]) for row in first6]
            if seeds != summary["initial_seeds"] or len(set(seeds)) != 6 or used_seeds.intersection(seeds):
                raise ValueError("First-six original seed mapping differs or overlaps another task")
            used_seeds.update(seeds)
            for row in first6:
                for field in ("uid", "oid", "task_index", "target_category", "blank_index"):
                    if int(row[field]) != int(task[field]):
                        raise ValueError(f"Candidate identity mismatch: {field}")
                if json.loads(row["outfit"]) != json.loads(task["outfit"]):
                    raise ValueError("Candidate outfit condition differs")
            selected_seed = int(task["seed"])
            if selected_seed != summary["selected_seed"] or selected_seed not in seeds:
                raise ValueError("Selected seed mismatch or outside first-six pool")
            by_seed = {int(row["seed"]): row for row in first6}
            scores = {seed: float(row["personal_sim"]) for seed, row in by_seed.items()}
            ranking = sorted(seeds, key=lambda seed: (-scores[seed], seed))
            oracle_seed = ranking[0]
            selected_image = folder / "selected.png"
            cached_selected_image = Path(by_seed[selected_seed]["image_path"])
            image_hash = digest(selected_image)
            if image_hash != task["selected_image_sha256"]:
                raise ValueError("Selected image changed after native evaluation")
            if digest(cached_selected_image) != image_hash:
                selected_hash_mismatches.append({"task_id": task_id, "online_image": str(selected_image),
                                                   "offline_image": str(cached_selected_image)})
            if not Path(by_seed[oracle_seed]["image_path"]).is_file():
                raise FileNotFoundError("Missing terminal oracle image")
            regret = scores[oracle_seed] - scores[selected_seed]
            if regret < 0 or not math.isfinite(regret):
                raise ValueError("Invalid same-pool regret")
            row = {
                "task_id": task_id, "task_index": int(task["task_index"]), "uid": int(task["uid"]),
                "oid": int(task["oid"]), "group_id": 0, "pool_seeds": json.dumps(seeds),
                "selected_seed": selected_seed, "selected_terminal_personal_same_table": scores[selected_seed],
                "oracle_first6_seed": oracle_seed, "oracle_first6_personal": scores[oracle_seed],
                "personal_regret_first6": regret, "oracle_hit": int(selected_seed == oracle_seed),
                "selected_terminal_rank_first6": ranking.index(selected_seed) + 1,
                "selected_online_personal": float(task["online_personal"]),
                "native_minus_same_table_personal": float(task["personal"]) - scores[selected_seed],
                "selected_image": str(selected_image.resolve()), "selected_cached_image": str(cached_selected_image),
                "oracle_image": by_seed[oracle_seed]["image_path"],
            }
            for metric in METRICS:
                row[f"selected_native_{metric}"] = float(task[metric])
                if not math.isfinite(row[f"selected_native_{metric}"]):
                    raise ValueError("Nonfinite native metric")
            rows.append(row)
            alive = seeds
            if len(summary["trace"]) != 3:
                raise ValueError("Expected exactly three pruning nodes")
            for event, (step, keep) in zip(summary["trace"], ((13, 4), (39, 2), (51, 1))):
                if event["step"] != step or event["seeds_before"] != alive:
                    raise ValueError("Node step/seed mapping mismatch")
                before = alive
                alive = event["seeds_after"]
                if len(set(alive)) != keep or not set(alive) <= set(before):
                    raise ValueError("Invalid surviving seed pool")
                best_survivor = min(alive, key=lambda seed: (-scores[seed], seed))
                node_rows.append({"task_id": task_id, "step": step, "survivors": json.dumps(alive),
                                  "best_terminal_survivor_seed": best_survivor,
                                  "best_terminal_survivor_personal": scores[best_survivor],
                                  "oracle_first6_personal": scores[oracle_seed],
                                  "survivor_terminal_regret": scores[oracle_seed] - scores[best_survivor],
                                  "oracle_survives": int(oracle_seed in alive)})
        except Exception as exc:
            invalid.append({"task_id": task_id, "error": repr(exc)})
    args.output_root.mkdir(parents=True, exist_ok=True)
    integrity = {"complete": not invalid and not selected_hash_mismatches and len(rows) == expected_count,
                 "expected_tasks": expected_count, "valid_tasks": len(rows), "pool_size": 6,
                 "invalid_tasks": invalid, "selected_online_offline_image_hash_mismatches": selected_hash_mismatches}
    write_json(args.output_root / "integrity_report.json", integrity)
    if not integrity["complete"]:
        raise RuntimeError("Same-task/seed/image checks failed; see integrity_report.json")
    provenance = {
        "candidate_scores": str(args.candidate_scores.resolve()), "candidate_scores_sha256": digest(args.candidate_scores),
        "native_metrics": str(args.native_metrics.resolve()), "native_metrics_sha256": digest(args.native_metrics),
        "run_root": str(args.run_root.resolve()), "online_summaries_sha256": source_hash.hexdigest(),
        "script_sha256": digest(Path(__file__)),
        "oracle": "Per task: max Personal at method=psp,node=final,group_id=0,candidate_index in 0..5",
        "regret": "oracle Personal from the CSV minus selected seed Personal from the same CSV",
        "ties": "descending terminal Personal, ascending original seed",
        "GT": "not read; no generation, scoring, or reranking; native metrics are frozen existing results",
        "dataset_status": "exploratory test-set analysis", "pool_size": 6,
    }
    config_path = args.output_root / "report_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != provenance:
        raise ValueError("Output report inputs/config differ; choose a fresh directory")
    write_json(config_path, provenance)
    regrets = np.asarray([r["personal_regret_first6"] for r in rows])
    hits = sum(r["oracle_hit"] for r in rows)
    summary = {
        "n_tasks": len(rows), "reward": "personal", "method": "PSP@6 online 13/39/final", "group_id": 0,
        "pool_size": 6, "oracle_personal_mean": statistics.mean(r["oracle_first6_personal"] for r in rows),
        "selected_personal_same_table_mean": statistics.mean(r["selected_terminal_personal_same_table"] for r in rows),
        "mean_regret": float(regrets.mean()), "std_sample_regret": float(regrets.std(ddof=1)),
        "median_regret": float(np.median(regrets)), "p90_regret": float(np.quantile(regrets, .90)),
        "p95_regret": float(np.quantile(regrets, .95)), "max_regret": float(regrets.max()),
        "oracle_hit_count": hits, "oracle_hit_rate": hits / len(rows),
        "nonzero_regret_count": int((regrets > 0).sum()),
        "selected_native_metrics_mean": {m: statistics.mean(r[f"selected_native_{m}"] for r in rows) for m in METRICS},
        "max_abs_native_vs_table_personal": max(abs(r["native_minus_same_table_personal"]) for r in rows),
        "selected_rank_counts": {str(rank): sum(r["selected_terminal_rank_first6"] == rank for r in rows) for rank in range(1, 7)},
    }
    node_summary = []
    for step in (13, 39, 51):
        sub = [r for r in node_rows if r["step"] == step]
        node_summary.append({"step": step, "n_tasks": len(sub),
                             "survivor_terminal_regret_mean": statistics.mean(r["survivor_terminal_regret"] for r in sub),
                             "oracle_survival_count": sum(r["oracle_survives"] for r in sub),
                             "oracle_survival_rate": statistics.mean(r["oracle_survives"] for r in sub)})
    write_csv(args.output_root / "personal_regret_per_task.csv", rows)
    write_csv(args.output_root / "survivor_regret_per_node.csv", node_rows)
    write_csv(args.output_root / "node_summary.csv", node_summary)
    write_csv(args.output_root / "summary.csv", [{k: v for k, v in summary.items() if not isinstance(v, dict)}])
    write_json(args.output_root / "summary.json", {**summary, "nodes": node_summary})
    print(json.dumps({**summary, "nodes": node_summary}, indent=2))


if __name__ == "__main__":
    main()
