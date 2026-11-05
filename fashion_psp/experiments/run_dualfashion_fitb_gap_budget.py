"""Online callbacks for the single frozen gap-budget mechanism experiment."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fashion_psp.search.gap_budget_psp import first_decision,rank


def online_search(policy,initial,advance,observe,checkpoint,resumed=None,notify=None):
    state=resumed or {"states":initial,"completed_step":0,"spent":0,"reward_views":0,
                      "scoring_rounds":0,"second_step":None,"trace":[],
                      "selected_slot":None,"selected_image":None}
    states=state["states"]
    for step in range(state["completed_step"]+1,29):
        before=sorted(states);spent=state["spent"]+len(before)
        reason=("initial_gap_decision" if step==7 else "budget_derived_second_node" if step==state["second_step"]
                else "terminal" if step==28 else "")
        decoded={}
        for slot in before:
            states[slot],image=advance(slot,states[slot],step,bool(reason))
            if reason:decoded[slot]=image
        survivors=before;removed={};detail={}
        scores=observe(decoded) if reason else {}
        if reason:
            if set(scores)!=set(before):raise ValueError("Expected exactly live rewards")
            order=rank(scores)
            state["reward_views"]+=len(before);state["scoring_rounds"]+=1
            if step==7:
                survivors,state["second_step"],detail=first_decision(scores)
            else:
                survivors=sorted(order[:1 if step==28 else 2])
            removed={str(s):"gap_partition" if step==7 else "current_personal_top2" if step==state["second_step"]
                     else "terminal_selection" for s in before if s not in survivors}
            if step==28:
                state["selected_slot"]=survivors[0];state["selected_image"]=decoded[survivors[0]]
        states={s:states[s] for s in survivors}
        state.update(states=states,completed_step=step,spent=spent)
        state["trace"].append({"step":step,"alive_before":before,"alive_after":survivors,
                               "observed":bool(reason),"observation_reason":reason,
                               "personal_scores":{str(s):float(v) for s,v in scores.items()},
                               "removed_reasons":removed,"particle_steps_spent":spent,
                               "reward_views_cumulative":state["reward_views"],
                               "scoring_rounds_cumulative":state["scoring_rounds"],
                               "second_observation_step":state["second_step"],**detail})
        if step==28:
            assert spent==112 and state["reward_views"] in (13,14) and state["scoring_rounds"]==3
        checkpoint(state)
        if notify:notify(step,len(survivors),spent)
    return state


def main():
    from fashion_psp.experiments.run_dualfashion_fitb_adaptive_personal import main as run,parse_args
    run(parse_args(default_policy=ROOT/"fashion_psp/artifacts/gap_budget_psp_v1/frozen_rule.json",
                   default_output=ROOT/"DualFashion-main/output/ifashion_fitb_gap_budget/personal_group0_seed2026"))


if __name__=="__main__":main()
