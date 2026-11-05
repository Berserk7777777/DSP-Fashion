"""Sparse Personal PSP online inference: skip VAE/CLIP unless observing.

Shares the existing neural implementation, task mapping and checkpoint format.
Policy decisions use only freshly observed live Personal, never cached scores.
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fashion_psp.experiments.search_sparse_personal import choose, observation_reason


def online_search(policy, initial, advance, observe, checkpoint, resumed=None, notify=None):
    state = resumed or {"states":initial, "completed_step":0, "spent":0,
                        "reward_views":0, "scoring_rounds":0, "trace":[],
                        "next_observation":policy.first,
                        "selected_slot":None, "selected_image":None}
    states = state["states"]
    for step in range(state["completed_step"] + 1, 29):
        before = sorted(states)
        spent = state["spent"] + len(before)
        reason = observation_reason(policy, step, spent, len(before), state["next_observation"])
        observe_now = bool(reason)
        images = {}
        for slot in before:
            states[slot], image = advance(slot, states[slot], step, observe_now)
            if observe_now:
                images[slot] = image
        scores = observe(images) if observe_now else {}
        survivors, removal_reasons, detail = before, {}, {}
        if observe_now:
            if set(scores) != set(before) or not all(np.isfinite(v) for v in scores.values()):
                raise ValueError("Reward must provide one finite score per live particle")
            state["reward_views"] += len(before)
            state["scoring_rounds"] += 1
            if step == 28:
                winner = min(before, key=lambda slot:(-scores[slot], slot))
                survivors = [winner]
                state["selected_slot"], state["selected_image"] = winner, images[winner]
                removal_reasons = {str(s):"terminal_selection" for s in before if s != winner}
            else:
                survivors, removal_reasons, state["next_observation"], detail = choose(policy, scores, step, spent)
        if step == 28 and spent != 112:
            raise AssertionError(f"Expected112 particle steps, got {spent}")
        states = {slot:states[slot] for slot in survivors}
        state.update(states=states, spent=spent, completed_step=step)
        state["trace"].append({"step":step, "alive_before":before, "alive_after":survivors,
                               "observed":observe_now, "observation_reason":reason,
                               "personal_scores":{str(s):float(v) for s,v in scores.items()},
                               "removed_reasons":removal_reasons,
                               "next_observation":state["next_observation"],
                               "particle_steps_spent":spent,
                               "reward_views_cumulative":state["reward_views"],
                               "scoring_rounds_cumulative":state["scoring_rounds"], **detail})
        checkpoint(state)
        if notify:
            notify(step, len(survivors), spent)
    return state


def main():
    from fashion_psp.experiments.run_dualfashion_fitb_adaptive_personal import main as run, parse_args
    run(parse_args(
        default_policy=ROOT / "fashion_psp/artifacts/sparse_personal_development_v3/selected_policy.json",
        default_output=ROOT / "DualFashion-main/output/ifashion_fitb_sparse_personal/personal_group0_seed2026"))


if __name__ == "__main__":
    main()
