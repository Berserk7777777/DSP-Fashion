"""User-specified fixed6->4->2->1 schedule. Single offline experiment, no tuning."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import (
    CausalObserver, load_and_audit, posthoc, sha256, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"fashion_psp/artifacts/six_seed_7_21_user_specified"
RULE = {
    "version": "six_seed_7_21_user_specified_v1",
    "initial_pool": "Original metadata slots0..5, no replacement or remapping",
    "schedule": [[7, 4], [21, 2], [28, 1]],
    "selection": "Current live Personal descending, original seed ascending on ties",
    "budget": "6*7+4*(21-7)+2*(28-21)=112",
    "reward_views": "6+4+2=12 candidate images in3 scoring rounds",
    "dataset_role": "Previously inspected200 test subset; exploratory requested comparison, NOT independent confirmation",
    "parameter_search_runs": 0,
}


def replay(seeds, observe):
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError("Requires exactly6 original seeds")
    live = sorted(seeds)
    spent = views = rounds = 0
    trace = []
    schedule = dict(RULE["schedule"])
    for step in range(1, 29):
        before = live.copy()
        spent += len(before)
        scored = step in schedule
        scores = observe(step, before, scored)
        if set(scores) != (set(before) if scored else set()):
            raise ValueError("Unexpected score access")
        if scored:
            live = sorted(sorted(before, key=lambda s: (-scores[s], s))[:schedule[step]])
            views += len(before)
            rounds += 1
        trace.append({"step": step, "alive_before": before, "alive_after": live.copy(),
                      "count_before": len(before), "count_after": len(live),
                      "observed": scored, "personal_scores": scores,
                      "removed_reasons": {str(s): "terminal_selection" if step == 28 else "fixed_current_personal_topk"
                                          for s in before if s not in live},
                      "cumulative_particle_steps": spent, "cumulative_decode_images": views})
    assert spent == 112 and views == 12 and rounds == 3
    return {"selected_seed": live[0], "particle_steps": spent, "decode_images": views,
            "personal_evaluations": views, "scoring_rounds": rounds, "trace": trace}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {"rule": RULE, "script_sha256": sha256(__file__)}
    target = OUT/"frozen_rule.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Existing frozen experiment differs")
    write_json(target, frozen)  # save exact requested rule before opening data
    metrics = ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv"
    cache = ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_cache"
    input_hash = sha256(metrics)
    tasks, audit = load_and_audit(metrics, cache, 200)
    selections, traces, checks = [], [], []
    for task, entry in sorted(tasks.items()):
        seeds = entry["seeds"][:6]
        observer = CausalObserver(entry["personal"], seeds)
        result = replay(seeds, observer)
        used = set(observer.accesses)
        assert len(used) == result["personal_evaluations"] == 12
        poisoned = {key: value if key in used else float("nan")
                    for key, value in entry["personal"].items()}
        assert result == replay(seeds, CausalObserver(poisoned, seeds))
        rows = result.pop("trace")
        assert sum(r["count_before"] for r in rows) == 112
        traces.extend({"task_index": task, **row} for row in rows)
        selections.append({"task_index": task, "uid": entry["identity"][0], "oid": entry["identity"][1],
                           "method": "six_seed_7_21", "candidate_seeds": seeds, **result})
        checks.append({"task_index": task, "exact112": True, "reward_views12": True,
                       "causal_poisoning_pass": True, "original_metadata_slots0_to5": seeds})
    write_csv(OUT/"selection_pre_posthoc.csv", selections)
    with (OUT/"step_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    # Oracle labels and all terminal trajectories are used only AFTER choices are saved.
    evaluated = posthoc(metrics, tasks, selections)
    for row in evaluated:
        entry = tasks[row["task_index"]]
        pool6 = entry["seeds"][:6]
        oracle6 = max(entry["personal"][28, s] for s in pool6)
        row.update(oracle6_personal=oracle6,
                   optimal_seed_hit_within6=int(row["personal"] == oracle6),
                   personal_regret_within6=oracle6-row["personal"],
                   original8_oracle_available_in6=int(any(s in row["oracle_seeds"] for s in pool6)))
    summary = {
        "rule": RULE, "tasks": len(evaluated),
        "mean_personal": statistics.mean(r["personal"] for r in evaluated),
        "best_seed_hits_original8": sum(r["optimal_seed_hit"] for r in evaluated),
        "best_seed_hit_rate_original8": statistics.mean(r["optimal_seed_hit"] for r in evaluated),
        "best_seed_hits_within6": sum(r["optimal_seed_hit_within6"] for r in evaluated),
        "best_seed_hit_rate_within6": statistics.mean(r["optimal_seed_hit_within6"] for r in evaluated),
        "original8_oracle_available_in6_tasks": sum(r["original8_oracle_available_in6"] for r in evaluated),
        "mean_personal_regret_original8": statistics.mean(r["personal_regret"] for r in evaluated),
        "mean_personal_regret_within6": statistics.mean(r["personal_regret_within6"] for r in evaluated),
        "all_tasks_exact112": all(r["particle_steps"] == 112 for r in evaluated),
        "particle_steps_per_task": 112, "total_particle_steps": 22400,
        "candidate_reward_views_per_task": 12, "total_reward_views": 2400,
        "scoring_rounds_per_task": 3, "clip_padded4_batches_per_task": 4,
        "clip_padded4_slots_per_task": 16,
        "causal_future_unobserved_poisoning_checks_passed": len(checks),
        "parameter_search_runs": 0, "new_policy_candidates_evaluated": 1,
        "actual_neural_calls": 0,
    }
    # Read saved comparisons, do not run extra candidate policies on the test set.
    reference_dir = ROOT/"fashion_psp/artifacts/gap_budget_psp_v1"
    manifest = json.loads((reference_dir/"manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_sha256"] == input_hash
    with (reference_dir/"selections.csv").open(encoding="utf-8-sig", newline="") as f:
        reference = list(csv.DictReader(f))
    comparisons = []
    summary["comparisons_to_saved_results"] = {}
    for method in ("default", "adaptive", "bon4_first4"):
        ref = {int(r["task_index"]): r for r in reference if r["method"] == method}
        assert len(ref) == 200
        deltas = []
        for row in evaluated:
            b = ref[row["task_index"]]
            assert (int(b["uid"]), int(b["oid"])) == (row["uid"], row["oid"])
            delta = row["personal"]-float(b["personal"])
            deltas.append(delta)
            comparisons.append({"task_index": row["task_index"], "reference": method,
                                "personal_delta": delta, "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie"})
        summary["comparisons_to_saved_results"][method] = {
            "reference_mean_personal": statistics.mean(float(r["personal"]) for r in ref.values()),
            "reference_original8_hits": sum(int(r["optimal_seed_hit"]) for r in ref.values()),
            "mean_personal_delta": statistics.mean(deltas),
            "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas),
            "losses": sum(d < 0 for d in deltas),
        }
    write_json(OUT/"summary.json", summary)
    write_csv(OUT/"selections.csv", evaluated)
    write_csv(OUT/"candidate_pool_audit.csv", audit)
    write_csv(OUT/"budget_causality_checks.csv", checks)
    write_csv(OUT/"comparisons.csv", comparisons)
    write_csv(OUT/"pruning_events.csv", [r for r in traces if r["count_after"] < r["count_before"]])
    assert sha256(metrics) == input_hash
    write_json(OUT/"manifest.json", {"input_sha256": input_hash, **frozen,
                                   "selection_before_posthoc_sha256": sha256(OUT/"selection_pre_posthoc.csv"),
                                   "trace_sha256": sha256(OUT/"step_trace.jsonl")})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
