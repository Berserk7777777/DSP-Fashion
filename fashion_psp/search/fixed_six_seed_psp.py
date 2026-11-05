"""User-specified fixed Personal PSP@6; no learned or task-derived policy."""
import math

RULE = {
    "version": "fixed_six_seed_7_21_v1",
    "initial_particles": 6,
    "horizon": 28,
    "particle_steps": 112,
    "schedule": [[7, 4], [21, 2], [28, 1]],
    "reward": "Personal only; score every live candidate at7,21,28 only",
    "seed_mapping": "base_seed + task_index*10000 + slot; original slots0..5",
    "ties": "Descending current Personal, ascending original seed (equivalently ascending slot)",
    "budget_identity": "6*7 + 4*(21-7) + 2*(28-21) = 112",
    "evaluation_cost": "6+4+2=12 candidate images,3 rounds; CLIP padded4:4 batches,16 slots",
    "method_type": "User-specified FIXED schedule, not adaptive pruning times",
    "data_boundary": "No cached scores, future/dead rewards, GT, Compatibility, CIS, LPIPS or DP table in decisions",
    "dataset_role": "Exploratory fixed schedule; previously inspected200 test tasks do not provide independent confirmation",
}


def rank(scores):
    if not scores or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Expected finite current Personal scores")
    return sorted(scores, key=lambda s: (-scores[s], s))
