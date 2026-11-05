"""FITB online Personal search:6 seeds, top4 at7, top2 at21, winner at28."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fashion_psp.search.fixed_six_seed_psp import RULE, rank


def online_search(policy, initial, advance, observe, checkpoint, resumed=None, notify=None):
    if policy != RULE:
        raise ValueError("Expected frozen fixed6/7/21 rule")
    if resumed is None and sorted(initial) != list(range(6)):
        raise ValueError("Initialize exactly6 original slots0..5")
    state = resumed or {"states": initial, "completed_step": 0, "spent": 0,
                        "reward_views": 0, "scoring_rounds": 0, "trace": [],
                        "selected_slot": None, "selected_image": None}
    states = state["states"]
    schedule = dict(RULE["schedule"])
    for step in range(state["completed_step"]+1, 29):
        before = sorted(states)
        expected_count = 6 if step <= 7 else 4 if step <= 21 else 2
        if len(before) != expected_count or not set(before) <= set(range(6)):
            raise ValueError("Live slots do not match fixed6/7/21 schedule")
        spent = state["spent"]+len(before)
        scored = step in schedule
        images = {}
        for slot in before:
            states[slot], image = advance(slot, states[slot], step, scored)
            if scored:
                images[slot] = image
        scores = observe(images) if scored else {}
        if set(scores) != (set(before) if scored else set()):
            raise ValueError("Expected exactly current live Personal scores")
        survivors = sorted(rank(scores)[:schedule[step]]) if scored else before
        if scored:
            state["reward_views"] += len(before)
            state["scoring_rounds"] += 1
        if step == 28:
            state["selected_slot"] = survivors[0]
            state["selected_image"] = images[survivors[0]]
        states = {slot: states[slot] for slot in survivors}
        state.update(states=states, completed_step=step, spent=spent)
        state["trace"].append({"step": step, "alive_before": before, "alive_after": survivors,
                               "count_before": len(before), "count_after": len(survivors),
                               "observed": scored,
                               "observation_reason": "terminal" if step == 28 else "fixed_pruning" if scored else "",
                               "personal_scores": {str(s): float(v) for s, v in scores.items()},
                               "removed_reasons": {str(s): "terminal_selection" if step == 28 else "fixed_current_personal_topk"
                                                   for s in before if s not in survivors},
                               "particle_steps_spent": spent,
                               "reward_views_cumulative": state["reward_views"],
                               "scoring_rounds_cumulative": state["scoring_rounds"]})
        if step == 28:
            assert spent == 112 and state["reward_views"] == 12 and state["scoring_rounds"] == 3
        checkpoint(state)
        if notify:
            notify(step, len(survivors), spent)
    return state


def main():
    from fashion_psp.experiments.run_dualfashion_fitb_adaptive_personal import main as run, parse_args
    args = parse_args(default_policy=ROOT/"fashion_psp/artifacts/fixed_six_seed_7_21/frozen_rule.json",
                      default_output=ROOT/"DualFashion-main/output/ifashion_fitb_six_seed_7_21/personal_group0_seed2026")
    if args.seed_stride != 10000:
        raise ValueError("Keep original task/seed mapping: seed-stride must be10000")
    run(args)


if __name__ == "__main__":
    main()
