"""Online exact112 controller with adaptive pruning times AND survivor counts."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fashion_psp.search.fully_adaptive_psp import (
    RULE, BUDGET, HORIZON, initial_control, plan, decide,
)


def online_search(policy, initial, advance, observe, checkpoint, resumed=None, notify=None):
    if policy != RULE:
        raise ValueError("Expected the frozen fully adaptive rule")
    if resumed is None and len(initial) != 8:
        raise ValueError("Requires the original8 seed slots")
    state = resumed or {"states": initial, "completed_step": 0, "spent": 0,
                        "reward_views": 0, "scoring_rounds": 0, "trace": [],
                        "control": initial_control(), "selected_slot": None, "selected_image": None}
    states = state["states"]
    for step in range(state["completed_step"]+1, HORIZON+1):
        before = sorted(states)
        spent = state["spent"]+len(before)
        action = plan(state["control"], step, spent, len(before))
        requested = before if action["observe"] else []
        images = {}
        for slot in before:
            states[slot], image = advance(slot, states[slot], step, action["observe"])
            if action["observe"]:
                images[slot] = image
        scores = observe(images) if requested else {}
        survivors, detail = decide(state["control"], step, spent, before, scores, action)
        state["reward_views"] += len(requested)
        state["scoring_rounds"] += bool(requested)
        if step == HORIZON:
            state["selected_slot"] = survivors[0]
            state["selected_image"] = images[survivors[0]]
        states = {s: states[s] for s in survivors}
        state.update(states=states, completed_step=step, spent=spent)
        state["trace"].append({"step": step, "alive_before": before, "alive_after": survivors,
                               "count_before": len(before), "count_after": len(survivors),
                               "evaluated_seeds": requested, "evaluation_count": len(requested),
                               "personal_scores": {str(s): float(v) for s, v in scores.items()},
                               "particle_steps_spent": spent,
                               "reward_views_cumulative": state["reward_views"],
                               "scoring_rounds_cumulative": state["scoring_rounds"], **detail})
        if step == HORIZON:
            assert spent == BUDGET
        checkpoint(state)
        if notify:
            notify(step, len(survivors), spent)
    return state


def main():
    from fashion_psp.experiments.run_dualfashion_fitb_adaptive_personal import main as run, parse_args
    run(parse_args(default_policy=ROOT/"fashion_psp/artifacts/fully_adaptive_psp_v1/frozen_rule.json",
                   default_output=ROOT/"DualFashion-main/output/ifashion_fitb_fully_adaptive/personal_group0_seed2026"))


if __name__ == "__main__":
    main()
