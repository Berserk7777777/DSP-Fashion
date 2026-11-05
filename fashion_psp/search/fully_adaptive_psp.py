"""Task-local adaptive observation/pruning controller; synthetic validation only.

No data loader, fitted table, terminal predictor, or test-derived parameters.
The three-CONSECUTIVE-step stability rule is deliberately preserved. Sparse
observations cannot be silently substituted for three consecutive steps.
"""
from __future__ import annotations

import math
import statistics

HORIZON, BUDGET, FINALISTS = 28, 112, 2
RULE = {
    "version": "fully_adaptive_psp_v1",
    "status": "research_prototype_synthetic_only_NOT_real_test_evaluated",
    "initial_particles": 8, "horizon": HORIZON, "particle_steps": BUDGET,
    "minimum_finalists": FINALISTS,
    "initial_probe": "ceil(initial latest feasible pruning deadline / 2)=5; initialization only, NOT mandatory pruning",
    "stability": "At least3 consecutive scored steps. Against the CURRENT leader's trajectory, all3 gaps>0 AND mean(gaps)>2*sample_std(gaps). Heuristic, NOT confidence interval.",
    "keep_count": "Propose removing every stable loser, clamp survivor count to exact denoising-feasible integers with >=2 finalists. Remove stable losers first; only then current lowest Personal if budget forces further removal.",
    "next_observation": "Incomplete window or no actual removal: next step. After removal: earliest of budget deadline minus2 and half the nearest linear gap-crossing time; at least next step. With2 finalists skip to terminal.",
    "deadline": "Latest d allowing the current population through d and at least2 particles through28; derived from remaining112 budget, no reward inputs.",
    "crossing": "Current leader fixed across the last two scored steps; positive closing speed only. A task-local scheduling heuristic, NOT a terminal-score forecast.",
    "budget_override": "Before accessing reward, force full-live observation when keeping current population would destroy exact112 feasibility.",
    "ties": "Descending current Personal, then ascending ORIGINAL seed; equal scores never establish stable inferiority.",
    "cost": "No reward-view cap. Every decoded candidate, scoring round and padded CLIP batch reported separately. >=24 views to build the first stability window; cannot promise near14.",
    "forbidden": "No dead/future/unobserved rewards, extra seeds, GT, Compatibility, CIS, LPIPS, old DP table or test-fitted predictor.",
    "design_choices": "Two finalists, half-deadline initialization and half-crossing trigger are declared unvalidated design choices, not proven optimal or calibrated.",
    "training": False, "parameter_search_runs": 0, "real_test_evaluations": 0,
}


def rank(scores):
    if not scores or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Expected finite current live Personal scores")
    return sorted(scores, key=lambda seed: (-scores[seed], seed))


def feasible_counts(step, spent, count, horizon=HORIZON, budget=BUDGET,
                    finalists=FINALISTS):
    """Exact future cost interval [m+f*(h-1), m*h], not a risk table.

    The next step must run all m retained particles; thereafter arbitrary
    monotone reductions down to f are possible. Every integer in this interval
    is reachable (verified against exhaustive schedules in synthetic tests).
    """
    h, remaining = horizon-step, budget-spent
    if h <= 0:
        return [1] if h == 0 and remaining == 0 else []
    return [m for m in range(finalists, count+1)
            if m+finalists*(h-1) <= remaining <= m*h]


def deadline(step, spent, count):
    """Latest next pruning step while keeping >=2 terminal competitors."""
    if count == FINALISTS:
        if spent+count*(HORIZON-step) != BUDGET:
            raise AssertionError("Two-finalist state must exactly exhaust budget")
        return HORIZON
    slack = BUDGET-spent-FINALISTS*(HORIZON-step)
    if slack < 0:
        raise AssertionError("Infeasible state")
    return min(HORIZON-1, step+slack//(count-FINALISTS))


def initial_control():
    d = deadline(0, 0, 8)
    return {"next_observation_step": (d+1)//2, "observations": []}


def plan(control, step, spent, count):
    """Call AFTER charging denoising but BEFORE reading this step's reward."""
    allowed = feasible_counts(step, spent, count)
    if not allowed:
        raise AssertionError("No exact-budget continuation")
    forced = step < HORIZON and count not in allowed
    observed = step == HORIZON or forced or step >= control["next_observation_step"]
    return {"observe": observed, "budget_requires_reduction": forced,
            "feasible_keep_counts": allowed,
            "observation_reason": ("terminal" if step == HORIZON else
                                   "budget_deadline" if forced else
                                   "scheduled_from_observed_history" if observed else "")}


def _window(control, live, scores, step):
    rows = control["observations"]
    consecutive = len(rows) == 3 and [r["step"] for r in rows] == list(range(step-2, step+1))
    evidence, stable = {}, []
    if consecutive:
        leader = rank(scores)[0]
        for seed in live:
            gaps = [r["scores"][str(leader)]-r["scores"][str(seed)] for r in rows]
            mean, std = statistics.mean(gaps), statistics.stdev(gaps)
            eligible = all(g > 0 for g in gaps) and mean > 2*std
            evidence[str(seed)] = {"gaps": gaps, "mean_gap": mean,
                                   "sample_std_gap": std, "stable_loser": eligible}
            if eligible:
                stable.append(seed)
    return consecutive, stable, evidence


def _next_observation(control, step, spent, survivors, scores, complete, removed):
    n = len(survivors)
    d = deadline(step, spent, n)
    if n == FINALISTS:
        return HORIZON, {"latest_feasible_pruning_step": d, "nearest_gap_crossing_steps": None}
    crossing = None
    if complete and removed:
        leader = rank({s: scores[s] for s in survivors})[0]
        previous = control["observations"][-2]["scores"]
        for seed in survivors:
            if seed == leader:
                continue
            gap = scores[leader]-scores[seed]
            closing = previous[str(leader)]-previous[str(seed)]-gap
            if closing > 0:
                time = gap/closing
                crossing = time if crossing is None else min(crossing, time)
        # Leave room for rebuilding a consecutive three-step window after a skip.
        target = max(step+1, d-2)
        if crossing is not None:
            # Clamp before floor: avoid overflow on extremely small closing rates.
            target = min(target, step+max(1, math.floor(min(crossing/2, HORIZON))))
    else:
        target = step+1
    return min(target, d), {"latest_feasible_pruning_step": d,
                           "nearest_gap_crossing_steps": crossing}


def decide(control, step, spent, live, scores, action=None):
    """Apply the previously announced observation; mutate serializable control."""
    live = sorted(live)
    expected = plan(control, step, spent, len(live))
    if action is not None and action != expected:
        raise AssertionError("Observation plan changed after reward access")
    wanted = set(live) if expected["observe"] else set()
    if set(scores) != wanted:
        raise ValueError("Expected exactly planned fresh live scores")
    detail = dict(expected)
    detail["observed"] = detail.pop("observe")
    removed = {}
    if not scores:
        survivors = live
    else:
        order = rank(scores)
        if step == HORIZON:
            survivors = order[:1]
            removed = {str(s): "terminal_selection" for s in live if s not in survivors}
        else:
            rows = control["observations"]
            rows.append({"step": step, "scores": {str(s): float(v) for s, v in scores.items()}})
            control["observations"] = rows[-3:]
            complete, stable, evidence = _window(control, live, scores, step)
            proposed = max(FINALISTS, len(live)-len(stable))
            allowed = expected["feasible_keep_counts"]
            keep = max(min(allowed), min(proposed, max(allowed)))
            stable_set = set(stable)
            removal_order = ([s for s in reversed(order) if s in stable_set]
                             + [s for s in reversed(order) if s not in stable_set])
            dropped = removal_order[:len(live)-keep]
            survivors = sorted(set(live)-set(dropped))
            removed = {str(s): "stability_heuristic" if s in stable_set else "budget_forced"
                       for s in dropped}
            next_step, timing = _next_observation(control, step, spent, survivors, scores,
                                                  complete, bool(dropped))
            control["next_observation_step"] = next_step
            # Delete dead history too; it is never needed for future decisions.
            for row in control["observations"]:
                row["scores"] = {str(s): row["scores"][str(s)] for s in survivors}
            detail.update(consecutive_window_complete=complete, stability_evidence=evidence,
                          proposed_keep_count=proposed, actual_keep_count=keep,
                          stable_loser_count=len(stable),
                          stable_losers_retained_for_exact_budget=len(stable_set-set(dropped)),
                          forced_removal_count=sum(v == "budget_forced" for v in removed.values()),
                          **timing)
    detail.update(removed_reasons=removed, next_observation_step=control["next_observation_step"])
    return survivors, detail


def replay(seeds, observe):
    """Callback(step, full live set, requested subset); no hidden score access."""
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("Requires the original8 distinct seeds")
    live, control, spent, views, rounds, trace = sorted(seeds), initial_control(), 0, 0, 0, []
    for step in range(1, HORIZON+1):
        before = live.copy()
        spent += len(before)
        action = plan(control, step, spent, len(before))
        requested = before if action["observe"] else []
        scores = observe(step, before, requested)
        live, detail = decide(control, step, spent, before, scores, action)
        views += len(requested)
        rounds += bool(requested)
        trace.append({"step": step, "alive_before": before, "alive_after": live.copy(),
                      "count_before": len(before), "count_after": len(live),
                      "evaluated_seeds": requested, "evaluation_count": len(requested),
                      "personal_scores": scores, "cumulative_particle_steps": spent,
                      "cumulative_decode_images": views, **detail})
    assert spent == BUDGET and len(live) == 1
    return {"selected_seed": live[0], "particle_steps": spent, "decode_images": views,
            "personal_evaluations": views, "scoring_rounds": rounds,
            "clip_batches_padded4": sum((r["evaluation_count"]+3)//4 for r in trace),
            "trace": trace}
