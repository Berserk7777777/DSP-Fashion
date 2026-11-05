"""Online partial-observation prototype; not benchmarked on real test tasks."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fashion_psp.search.gap_budget_psp import first_decision,rank


def online_search(policy,initial,advance,observe,checkpoint,resumed=None,notify=None):
    state=resumed or {"states":initial,"completed_step":0,"spent":0,"reward_views":0,
                      "scoring_rounds":0,"second_step":None,"deferred_branch":False,
                      "anchors":[],"boundary":[],"trace":[],"selected_slot":None,"selected_image":None}
    states=state["states"]
    for step in range(state["completed_step"]+1,29):
        before=sorted(states);spent=state["spent"]+len(before)
        reason=("initial_gap_gate" if step==7 else "partial_boundary_recheck" if step==8 and state["deferred_branch"]
                else "second_full_comparison" if step==state["second_step"] else "terminal" if step==28 else "")
        requested=sorted(state["boundary"] if reason=="partial_boundary_recheck" else before if reason else [])
        wanted=set(requested);images={}
        for slot in before:
            states[slot],image=advance(slot,states[slot],step,slot in wanted)
            if slot in wanted:images[slot]=image
        scores=observe(images) if requested else {}
        if set(scores)!=wanted:raise ValueError("Expected exactly requested fresh scores")
        survivors=before;removed={};detail={}
        if requested:
            order=rank(scores);state["reward_views"]+=len(requested);state["scoring_rounds"]+=1
            if step==7:
                proposed,_,gate=first_decision(scores)
                state["deferred_branch"]=len(proposed)==3
                gate_evidence={k:gate[k] for k in ("gap_rank3_minus_rank4","gap_rank4_minus_rank5")}
                if state["deferred_branch"]:
                    state["anchors"]=sorted(order[:2]);state["boundary"]=sorted(order[2:5])
                    survivors=sorted(order[:5]);state["second_step"]=19
                    detail={**gate_evidence,"proposed_keep_count":3,"actual_keep_count":5,
                            "anchors":state["anchors"].copy(),"boundary_seeds":state["boundary"].copy()}
                else:
                    survivors=sorted(order[:4]);state["second_step"]=14
                    detail={**gate_evidence,"proposed_keep_count":4,"actual_keep_count":4}
            elif step==8 and state["deferred_branch"]:
                survivors=sorted(state["anchors"]+[order[0]])
                detail={"protected_without_rescore":state["anchors"].copy(),"boundary_winner":order[0]}
            else:
                survivors=sorted(order[:1 if step==28 else 2])
            tag=("provisional_bottom3" if step==7 and state["deferred_branch"] else "current_top4" if step==7
                 else "fresh_boundary_choice" if step==8 and state["deferred_branch"] else "terminal_selection" if step==28
                 else "current_personal_top2")
            removed={str(s):tag for s in before if s not in survivors}
            if step==28:
                state["selected_slot"]=survivors[0];state["selected_image"]=images[survivors[0]]
        states={s:states[s] for s in survivors}
        state.update(states=states,completed_step=step,spent=spent)
        state["trace"].append({"step":step,"alive_before":before,"alive_after":survivors,
                               "count_before":len(before),"count_after":len(survivors),
                               "observed":bool(requested),"observation_reason":reason,
                               "evaluated_seeds":requested,"evaluation_count":len(requested),
                               "personal_scores":{str(s):float(v) for s,v in scores.items()},
                               "removed_reasons":removed,"particle_steps_spent":spent,
                               "reward_views_cumulative":state["reward_views"],
                               "scoring_rounds_cumulative":state["scoring_rounds"],
                               "deferred_branch":state["deferred_branch"],
                               "second_observation_step":state["second_step"],**detail})
        if step==28:
            assert spent==112 and state["reward_views"]==(16 if state["deferred_branch"] else 14)
        checkpoint(state)
        if notify:notify(step,len(survivors),spent)
    return state


def main():
    from fashion_psp.experiments.run_dualfashion_fitb_adaptive_personal import main as run,parse_args
    run(parse_args(default_policy=ROOT/"fashion_psp/artifacts/deferred_boundary_psp_v1/frozen_rule.json",
                   default_output=ROOT/"DualFashion-main/output/ifashion_fitb_deferred_boundary/personal_group0_seed2026"))


if __name__=="__main__":main()
