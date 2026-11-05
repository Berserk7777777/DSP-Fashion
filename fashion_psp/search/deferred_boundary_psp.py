"""Untested research extension: defer aggressive pruning for one boundary check.

Keeps the gap-v1 gate unchanged. On its aggressive branch, provisionally keep5
at7, observe ONLY the former ranks3..5 at8, keep the best of those alongside
the former top2, then full-live top2 at19. Terminal observes both finalists.
No task data, parameters, learned predictor or confidence claims in this file.
"""
from __future__ import annotations

from fashion_psp.search.gap_budget_psp import first_decision,rank


RULE = {
    "version":"deferred_boundary_psp_v1",
    "status":"research_prototype_NOT_evaluated_on_real_test_data",
    "initial_particles":8,"horizon":28,"particle_steps":112,
    "gate":"Unchanged gap-v1: g3=r3-r4 > g4=r4-r5 at original node7.",
    "aggressive_extension":"At7 provisionally keep top5; anchors=former top2; boundary=former ranks3..5. At8 observe only boundary3 and retain its current best plus anchors. At19 observe all3 and retain top2. At28 observe both and select1.",
    "conservative_branch":"Unchanged default: at7 keep4, at14 keep2, at28 select1.",
    "budget_identity":"Deferred:8*7+5*1+3*11+2*9=112; default:8*7+4*7+2*14=112.",
    "candidate_evaluations":"Deferred8+3+3+2=16 across4 rounds; default8+4+2=14 across3 rounds.",
    "partial_observation":"Former top2 are protected at8, not rescored or compared to fresh boundary scores. No stale score is treated as a current observation.",
    "ties":"Current score ties use smaller original seed; equal gate gaps use conservative branch.",
    "hypothesis":"One cheap boundary recheck may retain a late challenger that immediate top3 would discard; trades observation cost and later stages, can also hurt.",
    "limits":"Cannot recover former ranks6..8, and protecting former top2 can keep weak candidates. NOT a terminal confidence bound.",
    "training":False,"parameter_search":False,"real_test_evaluations":0,
    "evaluation_policy":"Synthetic validation only. Do not compare repeatedly on the already-inspected200 test tasks to select a winner.",
}


class PartialCausalObserver:
    """Replay guard for complete live sets and explicit scoring subsets.

    Every call advances exactly one step, with nonempty nonresurrected live
    particles. requested is a subset of live. Only requested values are read.
    No identity/terminal labels or old scores are returned to the policy.
    """
    def __init__(self,personal,seeds):
        self.__personal=personal
        self.__live=set(seeds)
        self.__step=0
        self.accesses=[]

    def __call__(self,step,live,requested):
        live=list(live);requested=list(requested)
        if (step!=self.__step+1 or not live or len(set(live))!=len(live)
                or not set(live)<=self.__live or len(set(requested))!=len(requested)
                or not set(requested)<=set(live)):
            raise AssertionError("Noncausal live/partial-observation request")
        self.__step=step;self.__live=set(live)
        self.accesses.extend((step,s) for s in requested)
        return {s:self.__personal[(step,s)] for s in requested}


def replay(seeds,observe):
    """observe(step, ALL_live, explicitly_requested_subset) -> fresh scores."""
    if len(seeds)!=8 or len(set(seeds))!=8:
        raise ValueError("Requires8 original distinct seeds")
    live=sorted(seeds);spent=views=rounds=0;second=None
    boundary=[];anchors=[];deferred=False;trace=[]
    for step in range(1,29):
        before=live.copy();spent+=len(before)
        reason=("initial_gap_gate" if step==7 else "partial_boundary_recheck" if step==8 and deferred
                else "second_full_comparison" if step==second else "terminal" if step==28 else "")
        requested=sorted(boundary if reason=="partial_boundary_recheck" else before if reason else [])
        scores=observe(step,before,requested)
        if set(scores)!=set(requested):raise ValueError("Expected exactly requested fresh scores")
        detail={};removed={}
        if requested:
            order=rank(scores);views+=len(requested);rounds+=1
            if step==7:
                proposed,_,gate=first_decision(scores)
                gate_evidence={k:gate[k] for k in ("gap_rank3_minus_rank4","gap_rank4_minus_rank5")}
                deferred=len(proposed)==3
                if deferred:
                    anchors=sorted(order[:2]);boundary=sorted(order[2:5])
                    live=sorted(order[:5]);second=19
                    detail={**gate_evidence,"proposed_keep_count":3,"actual_keep_count":5,
                            "anchors":anchors.copy(),"boundary_seeds":boundary.copy()}
                else:
                    live=sorted(order[:4]);second=14
                    detail={**gate_evidence,"proposed_keep_count":4,"actual_keep_count":4}
            elif step==8 and deferred:
                live=sorted(anchors+[order[0]])
                detail={"protected_without_rescore":anchors.copy(),"boundary_winner":order[0]}
            else:
                live=sorted(order[:1 if step==28 else 2])
            tag=("provisional_bottom3" if step==7 and deferred else "current_top4" if step==7
                 else "fresh_boundary_choice" if step==8 and deferred else "terminal_selection" if step==28
                 else "current_personal_top2")
            removed={str(s):tag for s in before if s not in live}
        trace.append({"step":step,"alive_before":before,"alive_after":live.copy(),
                      "count_before":len(before),"count_after":len(live),
                      "observation_reason":reason,"observed":bool(requested),
                      "evaluated_seeds":requested,"evaluation_count":len(requested),
                      "personal_scores":scores,"removed_reasons":removed,
                      "deferred_branch":deferred,"second_observation_step":second,
                      "cumulative_particle_steps":spent,"cumulative_decode_images":views,
                      "cumulative_personal_evaluations":views,"cumulative_scoring_rounds":rounds,**detail})
    assert spent==112 and views==(16 if deferred else 14) and rounds==(4 if deferred else 3)
    return {"selected_seed":live[0],"particle_steps":spent,"decode_images":views,
            "personal_evaluations":views,"scoring_rounds":rounds,"trace":trace}
