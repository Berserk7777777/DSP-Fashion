"""Posthoc diagnosis of user-selected9248_17906; no new policy tuning/replay.

Full trajectories are intentionally used ONLY for diagnosis. Resource tables
contain no rewards. Existing policy events are read from saved replay outputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics

from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import (
    load_and_audit, sha256, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"fashion_psp/artifacts/late_personal_case_9248_17906"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv"
    cache = ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_cache"
    tasks, _ = load_and_audit(metrics, cache, 200)
    matches = [(t, e) for t, e in tasks.items() if e["identity"] == (9248, 17906)]
    if len(matches) != 1:
        raise ValueError("Ambiguous task identity")
    task, entry = matches[0]
    seeds, points = entry["seeds"], entry["personal"]
    terminal_best = max(points[28, s] for s in seeds)
    oracles = [s for s in seeds if points[28, s] == terminal_best]
    target = seeds[3]  # user shorthand2029 refers to original slot3; NOT literal seed2029
    assert target == 112029 and target in oracles
    alias = {s: 2026+i for i, s in enumerate(seeds)}
    rows, long_rows = [], []
    for step in range(1, 29):
        order = sorted(seeds, key=lambda s: (-points[step, s], s))
        order6 = [s for s in order if s in seeds[:6]]
        rows.append({"step": step, "target_actual_seed": target,
                     "target_personal": points[step, target],
                     "target_rank_original8": order.index(target)+1,
                     "target_rank_first6": order6.index(target)+1,
                     "current_leader_actual_seed": order[0],
                     "current_leader_personal": points[step, order[0]],
                     "leader_minus_target_gap": points[step, order[0]]-points[step, target],
                     "ranking_actual_seeds": order})
        long_rows.extend({"step": step, "seed": s, "display_alias_only": alias[s],
                          "original_slot": seeds.index(s), "personal": points[step, s],
                          "rank_original8": order.index(s)+1,
                          "image_path": entry["paths"][step, s]} for s in seeds)
    saved_results, saved_events, sources = [], [], []
    for folder, methods in (("gap_budget_psp_v1", {"default", "adaptive", "bon4_first4"}),
                            ("six_seed_7_21_user_specified", {"six_seed_7_21"})):
        base = ROOT/"fashion_psp/artifacts"/folder
        for name in ("selections.csv", "step_trace.jsonl"):
            sources.append({"path": str(base/name), "sha256": sha256(base/name)})
        with (base/"selections.csv").open(encoding="utf-8-sig", newline="") as f:
            saved_results.extend(r for r in csv.DictReader(f)
                                 if int(r["task_index"]) == task and r["method"] in methods)
        for line in (base/"step_trace.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            if row["task_index"] == task and row["count_before"] > row["count_after"]:
                saved_events.append({"method": row.get("method", "six_seed_7_21"), **row})
    # Pure arithmetic: unrestricted initial count, >=2 terminal finalists.
    deadlines = [{"initial_count": n, "minimum_terminal_finalists": 2,
                  "latest_first_reduction_step": (112-2*28)//(n-2),
                  "interpretation": "Necessary bound only; does not ensure exact112 without later pruning"}
                 for n in range(5, 17)]
    # Full two-node resource frontier <=14 candidate views; no case scores consulted.
    feasible = []
    for n in range(5, 11):
        for final in (2, 3):
            for keep in range(final+1, n):
                views = n+keep+final
                if views > 14:
                    continue
                for first in range(1, 28):
                    for second in range(first+1, 28):
                        cost = n*first+keep*(second-first)+final*(28-second)
                        if cost == 112:
                            feasible.append({"initial_count": n, "first_step": first,
                                             "first_keep": keep, "second_step": second,
                                             "terminal_finalists": final,
                                             "particle_steps": cost, "candidate_views": views})
    result = {
        "status": "POSTHOC_TEST_DIFFICULTY_DIAGNOSIS_NOT_POLICY_SELECTION",
        "task_index": task, "uid": 9248, "oid": 17906,
        "target_actual_seed": target, "display_alias_not_literal_seed": 2029,
        "target_terminal_personal": points[28, target],
        "target_first_rank1_step": next(r["step"] for r in rows if r["target_rank_original8"] == 1),
        "target_rank7_original8": rows[6]["target_rank_original8"],
        "target_rank7_first6": rows[6]["target_rank_first6"],
        "target_rank7_top4_cutoff_gap_original8": points[7, rows[6]["ranking_actual_seeds"][3]]-points[7, target],
        "default_selected_terminal_personal": next(float(r["personal"]) for r in saved_results if r["method"] == "default"),
        "target_terminal_image": entry["paths"][28, target],
        "saved_policy_results": saved_results,
        "new_policy_evaluations": 0, "parameter_search_runs": 0,
        "source_metrics_sha256": sha256(metrics), "saved_result_sources": sources,
        "resource_frontier_reward_access": False,
    }
    leader7 = rows[6]["current_leader_actual_seed"]
    gaps = [points[t, leader7]-points[t, target] for t in (5, 6, 7)]
    result["diagnostic_stability_counterexample_steps5_to7"] = {
        "reference_current_leader_actual_seed": leader7, "gaps": gaps,
        "mean_gap": statistics.mean(gaps), "sample_std_gap": statistics.stdev(gaps),
        "stable_loser_heuristic_flags_eventual_best": all(g > 0 for g in gaps)
        and statistics.mean(gaps) > 2*statistics.stdev(gaps),
        "interpretation": "These3 scores would have to be paid for online. Posthoc demonstration that short-term stability does not imply terminal inferiority.",
    }
    resource_designs = [
        {"name": "existing_six_seed_7_21", "count_segments": [[6, 7], [4, 14], [2, 7]],
         "scoring_populations": [6, 4, 2], "status": "existing_saved_replay"},
        {"name": "six_seed_defer_first_prune_to14_with_early_probe", "count_segments": [[6, 14], [2, 14]],
         "scoring_populations": [6, 6, 2], "status": "resource_design_NOT_new_200_task_performance_test",
         "observe_steps": [7, 14, 28], "prune_steps": [14, 28]},
        {"name": "six_seed_defer_first_prune_to14_without_early_probe", "count_segments": [[6, 14], [2, 14]],
         "scoring_populations": [6, 2], "status": "resource_design_NOT_new_200_task_performance_test",
         "observe_steps": [14, 28], "prune_steps": [14, 28]},
    ]
    for design in resource_designs:
        design["particle_steps"] = sum(n*h for n, h in design["count_segments"])
        design["reward_views"] = sum(design["scoring_populations"])
        design["clip_batches_padded4"] = sum((n+3)//4 for n in design["scoring_populations"])
        design["clip_slots_with_padding"] = 4*design["clip_batches_padded4"]
        assert design["particle_steps"] == 112 and design["reward_views"] <= 14
    result["resource_designs_not_performance_comparisons"] = resource_designs
    write_json(OUT/"summary.json", result)
    write_csv(OUT/"target_step_ranks.csv", rows)
    write_csv(OUT/"all_seed_scores.csv", long_rows)
    write_csv(OUT/"saved_pruning_events.csv", saved_events)
    write_csv(OUT/"resource_only_first_reduction_bounds.csv", deadlines)
    write_csv(OUT/"resource_only_two_node_frontier.csv", feasible)
    plot(entry, target, rows)
    print(json.dumps({k: v for k, v in result.items() if k not in ("saved_policy_results", "saved_result_sources")}, indent=2))
    print("Read existing results only; NO new policy evaluated. Saved:", OUT)


def plot(entry, target, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    colors = {target: "#d55e00", entry["seeds"][1]: "#0072b2"}
    for i, seed in enumerate(entry["seeds"]):
        highlight = seed in colors
        axes[0].plot(range(1, 29), [entry["personal"][t, seed] for t in range(1, 29)],
                     color=colors.get(seed, "#aaaaaa"), linewidth=2.4 if highlight else 1,
                     alpha=1 if highlight else 0.6,
                     label=f"seed {seed} (slot{i})" if highlight else None)
    axes[0].axvline(7, color="#666666", linestyle="--", label="Default first pruning")
    axes[0].axvline(21, color="#009e73", linestyle=":", label="6-seed second pruning")
    axes[0].set(ylabel="Personal", title="9248_17906: late improvement of terminal-best seed112029")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].plot(range(1, 29), [r["target_rank_original8"] for r in rows], "o-", color="#d55e00", label="Rank among original8")
    axes[1].plot(range(1, 29), [r["target_rank_first6"] for r in rows], "s--", color="#009e73", label="Rank among first6")
    axes[1].set(xlabel="Denoising step", ylabel="Target rank", yticks=range(1, 9), ylim=(8.5, 0.5), xlim=(1, 28))
    axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.text(0.02, 0.015, "Posthoc full-trajectory diagnosis only. Actual task seed112029 is not literal seed2029. No new policy tuning.", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT/"late_personal_diagnosis.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
