"""A single frozen, training-free, gap-adaptive PSP mechanism experiment.

Initial observation is the original PSP quarter-horizon node7. With two
internal pruning nodes, terminal population2, exact112 and <=14 views, choosing
keep3 or keep4 at7 fixes the next node to21 or14 respectively. No test-derived
margin, terminal forecast, calibrated confidence claim, or parameter search.
"""
from __future__ import annotations
import math


RULE = {
    "version":"gap_budget_psp_v1",
    "initial_particles":8,"horizon":28,"particle_steps":112,
    "first_step":7,"finalists":2,
    "first_decision":"Keep3 iff score(rank3)-score(rank4) > score(rank4)-score(rank5); otherwise keep4.",
    "second_step":"Derived by 8*7 + k*(t-7) + 2*(28-t)=112: k3->t21, k4->t14.",
    "second_decision":"Current live Personal top2; actual terminal live Personal top1.",
    "ties":"Equal boundary gaps keep4; score ties use smaller original seed.",
    "interpretation":"Raw adjacent-gap heuristic; NOT a confidence interval or a winning probability.",
    "scope":"Only first keep-count and the corresponding second time are adaptive; first time remains fixed7.",
    "training":False,"threshold_search":False,"policy_candidates":1,
    "observations":"All live scores at7, chosen second node, and28; no other score accesses.",
    "forbidden":"No future/dead scores, old DP tables, GT, Compatibility, CIS, LPIPS, learned test-derived predictors.",
    "comparison":"BoN4 uses the first4 ORIGINAL metadata seeds; not the historical independently seeded BoN4.",
    "status":"Frozen before this policy's first replay; already-inspected test subset, not independent confirmation.",
}


def rank(scores):
    if not scores or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Expected finite live Personal")
    return sorted(scores,key=lambda s:(-scores[s],s))


def first_decision(scores):
    if len(scores)!=8:
        raise ValueError("Requires original8 live scores")
    order=rank(scores)
    gap3=scores[order[2]]-scores[order[3]]
    gap4=scores[order[3]]-scores[order[4]]
    keep=3 if gap3>gap4 else 4
    # Rearranged budget identity, not an empirically selected node.
    numerator=112-8*7+keep*7-2*28
    assert numerator % (keep-2)==0
    second=numerator//(keep-2)
    return sorted(order[:keep]), second, {
        "gap_rank3_minus_rank4":gap3,"gap_rank4_minus_rank5":gap4,
        "keep_count":keep,"second_step":second,
    }


def replay(seeds,observe):
    if len(seeds)!=8 or len(set(seeds))!=8:
        raise ValueError("Requires8 distinct original seeds")
    active=sorted(seeds);spent=views=rounds=0;second=None;trace=[]
    for step in range(1,29):
        before=active.copy();spent+=len(before)
        reason=("initial_gap_decision" if step==7 else "budget_derived_second_node" if step==second
                else "terminal" if step==28 else "")
        scores=observe(step,before,bool(reason))
        detail={};removed={}
        if reason:
            if set(scores)!=set(before):
                raise ValueError("Expected exactly the live scores")
            order=rank(scores)
            views+=len(before);rounds+=1
            if step==7:
                active,second,detail=first_decision(scores)
            else:
                active=sorted(order[:1 if step==28 else 2])
            removed={str(s):"gap_partition" if step==7 else "current_personal_top2" if step==second
                     else "terminal_selection" for s in before if s not in active}
        elif scores:
            raise ValueError("Unrequested scores")
        trace.append({"step":step,"alive_before":before,"alive_after":active.copy(),
                      "count_before":len(before),"count_after":len(active),"observed":bool(reason),
                      "observation_reason":reason,"personal_scores":scores,"removed_reasons":removed,
                      "second_observation_step":second,"cumulative_particle_steps":spent,
                      "cumulative_decode_images":views,"cumulative_personal_evaluations":views,
                      "cumulative_scoring_rounds":rounds,**detail})
    assert spent==112 and views in (13,14) and rounds==3
    return {"selected_seed":active[0],"particle_steps":spent,"decode_images":views,
            "personal_evaluations":views,"scoring_rounds":rounds,"trace":trace}
