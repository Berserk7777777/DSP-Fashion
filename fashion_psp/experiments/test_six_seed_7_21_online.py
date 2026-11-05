"""Verify implementation of the already requested fixed schedule; no tuning."""
import copy
import csv
import json
from pathlib import Path
import random
import unittest

from fashion_psp.experiments.run_dualfashion_fitb_six_seed_7_21 import online_search
from fashion_psp.experiments.replay_six_seed_7_21 import replay
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import CausalObserver, load_and_audit
from fashion_psp.search.fixed_six_seed_psp import RULE

ROOT = Path(__file__).resolve().parents[2]


class FixedSixTests(unittest.TestCase):
    def check_online(self, points, expected, snapshots_at=()):
        snapshots, accesses = [], []
        def advance(slot, latent, step, decode):
            self.assertEqual(latent, step-1)
            self.assertEqual(decode, step in (7, 21, 28))
            return step, (slot, step) if decode else None
        def observe(images):
            scores = {}
            for slot, (s, step) in images.items():
                self.assertEqual(s, slot)
                accesses.append((step, slot))
                scores[slot] = points[step, slot]
                images[slot] = ("rgb", slot, step)
            return scores
        def checkpoint(state):
            if state["completed_step"] in snapshots_at:
                snapshots.append(copy.deepcopy(state))
        result = online_search(RULE, dict.fromkeys(range(6), 0), advance, observe, checkpoint)
        self.assertEqual(result["selected_slot"], expected["selected_seed"])
        self.assertEqual(result["spent"], 112)
        self.assertEqual(result["reward_views"], 12)
        self.assertEqual(result["scoring_rounds"], 3)
        self.assertEqual(len(accesses), 12)
        self.assertEqual(result["selected_image"], ("rgb", result["selected_slot"], 28))
        for a, b in zip(result["trace"], expected["trace"]):
            for key in ("step", "alive_before", "alive_after", "observed", "removed_reasons"):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a["personal_scores"], {str(s): v for s, v in b["personal_scores"].items()})
            self.assertEqual(a["particle_steps_spent"], b["cumulative_particle_steps"])
        for snapshot in snapshots:
            resumed = online_search(RULE, {}, advance, observe, lambda state: None, copy.deepcopy(snapshot))
            self.assertEqual(result, resumed)
        return result

    def test_synthetic_parity_all_step_resume_ties_and_unobserved_poison(self):
        rng = random.Random(20261029)
        scenarios = [{(t, s): 1.0 for t in range(1, 29) for s in range(6)}]
        scenarios += [{(t, s): rng.gauss(0, 1) for t in range(1, 29) for s in range(6)}
                      for _ in range(9)]
        for p in scenarios:
            obs = CausalObserver(p, range(6))
            expected = replay(list(range(6)), obs)
            self.check_online(p, expected, range(1, 29))
            used = set(obs.accesses)
            poisoned = {k: v if k in used else float("nan") for k, v in p.items()}
            self.check_online(poisoned, expected)
        tied = replay(list(range(6)), CausalObserver(scenarios[0], range(6)))
        self.assertEqual(tied["selected_seed"], 0)

    def test_all200_match_existing_fixed_schedule_selections_and_resume(self):
        tasks, _ = load_and_audit(
            ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv",
            ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_cache", 200)
        with (ROOT/"fashion_psp/artifacts/six_seed_7_21_user_specified/selections.csv").open(encoding="utf-8-sig", newline="") as f:
            saved = {int(r["task_index"]): r for r in csv.DictReader(f)}
        for task, entry in tasks.items():
            points = {(t, slot): entry["personal"][t, seed]
                      for slot, seed in enumerate(entry["seeds"][:6]) for t in range(1, 29)}
            expected = replay(list(range(6)), CausalObserver(points, range(6)))
            result = self.check_online(points, expected, (6, 7, 8, 14, 20, 21, 22, 27, 28))
            self.assertEqual(entry["seeds"][result["selected_slot"]], int(saved[task]["selected_seed"]))
            self.assertEqual(entry["identity"], (int(saved[task]["uid"]), int(saved[task]["oid"])))

    def test_reject_eight_initial_slots_missing_and_nonfinite_reward(self):
        def advance(s, latent, t, decode):
            return t, (s, t) if decode else None
        with self.assertRaisesRegex(ValueError, "exactly6"):
            online_search(RULE, dict.fromkeys(range(8), 0), advance, lambda images: {}, lambda state: None)
        with self.assertRaisesRegex(ValueError, "exactly current live"):
            online_search(RULE, dict.fromkeys(range(6), 0), advance, lambda images: {}, lambda state: None)
        with self.assertRaisesRegex(ValueError, "finite"):
            online_search(RULE, dict.fromkeys(range(6), 0), advance,
                          lambda images: dict.fromkeys(images, float("nan")), lambda state: None)

    def test_backward_compatible_existing_gap_manifest(self):
        # Shared neural runner still prepares the completed8-seed run unchanged.
        root = ROOT/"DualFashion-main/output/ifashion_fitb_gap_budget/personal_group0_seed2026"
        config = json.loads((root/"run_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["particles"], 8)
        self.assertEqual(config["method"], "frozen_gap_budget_psp8")


if __name__ == "__main__":
    unittest.main()
