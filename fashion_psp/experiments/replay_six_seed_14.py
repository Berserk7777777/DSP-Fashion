"""User-requested single fixed6->2 at14, then terminal selection at28.

Original metadata slots0..5, no seed-subset or schedule search. This is an
exploratory replay on an already-inspected test subset, NOT independent testing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import (
    CausalObserver, load_and_audit, posthoc, sha256, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"fashion_psp/artifacts/six_seed_14_user_specified"
RULE = {
    "version": "six_seed_14_user_specified_v1",
    "initial_pool": "Original metadata slots0..5; no replacement/remapping",
    "schedule": [[14, 2], [28, 1]],
    "selection": "Current live Personal descending, original seed ascending on exact ties",
    "particle_steps": "6*14+2*(28-14)=112",
    "reward_views": "6+2=8 across2 rounds; step7 is NOT evaluated because it has no decision role",
    "dataset_role": "Already inspected200 test subset; exploratory requested comparison, NOT independent confirmation",
    "parameter_search_runs": 0,
}


def replay(seeds, observe):
    if len(seeds) != 6 or len(set(seeds)) != 6:
        raise ValueError("Requires6 distinct original seeds")
    live, spent, views, trace = sorted(seeds), 0, 0, []
    for step in range(1, 29):
        before = live.copy()
        spent += len(before)
        scored = step in (14, 28)
        scores = observe(step, before, scored)
        if set(scores) != (set(before) if scored else set()):
            raise ValueError("Unexpected score access")
        if scored:
            live = sorted(sorted(before, key=lambda s: (-scores[s], s))[:2 if step == 14 else 1])
            views += len(before)
        trace.append({"step": step, "alive_before": before, "alive_after": live.copy(),
                      "count_before": len(before), "count_after": len(live),
                      "observed": scored, "personal_scores": scores,
                      "removed_reasons": {str(s): "fixed_current_personal_top2" if step == 14 else "terminal_selection"
                                          for s in before if s not in live},
                      "cumulative_particle_steps": spent, "cumulative_decode_images": views})
    assert spent == 112 and views == 8
    return {"selected_seed": live[0], "particle_steps": spent, "decode_images": views,
            "personal_evaluations": views, "scoring_rounds": 2, "trace": trace}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {"rule": RULE, "script_sha256": sha256(__file__)}
    freeze = OUT/"frozen_rule.json"
    if freeze.exists() and json.loads(freeze.read_text(encoding="utf-8")) != frozen:
        raise ValueError("Existing frozen experiment differs")
    write_json(freeze, frozen)  # freeze requested schedule before loading trajectories
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
        assert len(used) == result["personal_evaluations"] == 8
        poisoned = {k: v if k in used else float("nan") for k, v in entry["personal"].items()}
        assert result == replay(seeds, CausalObserver(poisoned, seeds))
        rows = result.pop("trace")
        assert sum(r["count_before"] for r in rows) == 112
        traces.extend({"task_index": task, **r} for r in rows)
        selections.append({"method": "six_seed_14", "task_index": task,
                           "uid": entry["identity"][0], "oid": entry["identity"][1],
                           "candidate_seeds": seeds, **result})
        checks.append({"task_index": task, "exact112": True, "reward_views8": True,
                       "causal_poisoning_pass": True, "candidate_seeds": seeds})
    write_csv(OUT/"selection_pre_posthoc.csv", selections)
    with (OUT/"step_trace.jsonl").open("w", encoding="utf-8") as f:
        for r in traces:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n")
    # Terminal oracles, Compatibility and comparisons only AFTER saving choices.
    evaluated = posthoc(metrics, tasks, selections)
    for r in evaluated:
        e = tasks[r["task_index"]]
        oracle6 = max(e["personal"][28, s] for s in e["seeds"][:6])
        r.update(oracle6_personal=oracle6, optimal_seed_hit_within6=int(r["personal"] == oracle6),
                 personal_regret_within6=oracle6-r["personal"])
    summary = {
        "rule": RULE, "tasks": len(evaluated),
        "mean_personal": statistics.mean(r["personal"] for r in evaluated),
        "mean_compatibility": statistics.mean(r["compatibility"] for r in evaluated),
        "best_seed_hits_original8": sum(r["optimal_seed_hit"] for r in evaluated),
        "best_seed_hit_rate_original8": statistics.mean(r["optimal_seed_hit"] for r in evaluated),
        "best_seed_hits_within6": sum(r["optimal_seed_hit_within6"] for r in evaluated),
        "best_seed_hit_rate_within6": statistics.mean(r["optimal_seed_hit_within6"] for r in evaluated),
        "mean_personal_regret_original8": statistics.mean(r["personal_regret"] for r in evaluated),
        "mean_personal_regret_within6": statistics.mean(r["personal_regret_within6"] for r in evaluated),
        "all_tasks_exact112": all(r["particle_steps"] == 112 for r in evaluated),
        "particle_steps_per_task": 112, "total_particle_steps": 22400,
        "candidate_reward_views_per_task": 8, "total_reward_views": 1600,
        "scoring_rounds_per_task": 2, "clip_padded4_batches_per_task": 3,
        "clip_padded4_slots_per_task": 12,
        "poisoning_checks_passed": len(checks), "parameter_search_runs": 0,
        "new_policy_candidates_evaluated": 1, "actual_neural_calls": 0,
    }
    comparisons = []
    summary["comparisons_to_saved_results"] = {}
    references = (("gap_budget_psp_v1", ("default", "adaptive", "bon4_first4")),
                  ("six_seed_7_21_user_specified", ("six_seed_7_21",)))
    for folder, methods in references:
        base = ROOT/"fashion_psp/artifacts"/folder
        assert json.loads((base/"manifest.json").read_text(encoding="utf-8"))["input_sha256"] == input_hash
        with (base/"selections.csv").open(encoding="utf-8-sig", newline="") as f:
            saved = list(csv.DictReader(f))
        for method in methods:
            ref = {int(r["task_index"]): r for r in saved if r["method"] == method}
            assert len(ref) == 200
            deltas = []
            for r in evaluated:
                b = ref[r["task_index"]]
                assert (int(b["uid"]), int(b["oid"])) == (r["uid"], r["oid"])
                delta = r["personal"]-float(b["personal"])
                deltas.append(delta)
                comparisons.append({"task_index": r["task_index"], "reference": method,
                                    "personal_delta": delta,
                                    "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie"})
            summary["comparisons_to_saved_results"][method] = {
                "reference_mean_personal": statistics.mean(float(b["personal"]) for b in ref.values()),
                "reference_original8_hits": sum(int(b["optimal_seed_hit"]) for b in ref.values()),
                "mean_personal_delta": statistics.mean(deltas),
                "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas),
                "losses": sum(d < 0 for d in deltas)}
    case = next(r for r in evaluated if (r["uid"], r["oid"]) == (9248, 17906))
    case_events = [r for r in traces if r["task_index"] == case["task_index"] and r["observed"]]
    summary["case_9248_17906"] = {**case, "observation_events": case_events}
    assert case["selected_seed"] == 112029 and case["optimal_seed_hit"] == 1
    write_json(OUT/"summary.json", summary)
    write_csv(OUT/"selections.csv", evaluated)
    write_csv(OUT/"pruning_events.csv", [r for r in traces if r["count_after"] < r["count_before"]])
    write_csv(OUT/"budget_causality_checks.csv", checks)
    write_csv(OUT/"candidate_pool_audit.csv", audit)
    write_csv(OUT/"comparisons.csv", comparisons)
    assert sha256(metrics) == input_hash
    write_json(OUT/"manifest.json", {"input_sha256": input_hash, **frozen,
                                   "selection_before_posthoc_sha256": sha256(OUT/"selection_pre_posthoc.csv"),
                                   "trace_sha256": sha256(OUT/"step_trace.jsonl")})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
