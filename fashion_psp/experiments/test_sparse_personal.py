"""Budget, causal-access, online/replay equivalence and checkpoint regression tests."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np

from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import CausalObserver, load_and_audit
from fashion_psp.experiments.search_sparse_personal import SparsePolicy, sparse_replay, batch_replay, observation_reason
from fashion_psp.experiments.run_dualfashion_fitb_sparse_personal import online_search

ROOT = Path(__file__).resolve().parents[2]


class SparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks, _ = load_and_audit(
            ROOT / "DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv",
            ROOT / "DualFashion-main/output/ifashion_fitb_timestep200_cache", 200)
        artifact = json.loads((ROOT / "fashion_psp/artifacts/sparse_personal_development_v3/selected_policy.json").read_text())
        cls.policy = SparsePolicy(**artifact["parameters"])

    def test_all200_online_parity_and_resume(self):
        for task, entry in sorted(self.tasks.items()):
            with self.subTest(task=task):
                points = {(step, slot):entry["personal"][(step, seed)]
                          for slot, seed in enumerate(entry["seeds"]) for step in range(1, 29)}
                expected = sparse_replay(self.policy, range(8), CausalObserver(points, range(8)))
                snaps, calls = {}, []
                def advance(slot, latent, step, decode):
                    self.assertEqual(latent, step-1)
                    self.assertEqual(decode, expected["trace"][step-1]["observed"])
                    return step, (slot, step) if decode else None
                def observe(images):
                    scores = {}
                    for slot, (s, step) in images.items():
                        self.assertEqual(s, slot)
                        calls.append((step, slot))
                        scores[slot] = points[(step, slot)]
                        images[slot] = ("decoded_rgb", slot, step)
                    return scores
                def checkpoint(state):
                    if state["completed_step"] in (3, 8, 9, 16, 27, 28):
                        snaps[state["completed_step"]] = copy.deepcopy(state)
                result = online_search(self.policy, dict.fromkeys(range(8), 0), advance, observe, checkpoint)
                self.assertEqual(result["spent"], 112)
                self.assertEqual(result["reward_views"], expected["decode_images"])
                self.assertEqual(result["scoring_rounds"], expected["scoring_rounds"])
                self.assertEqual(result["selected_slot"], expected["selected_seed"])
                self.assertEqual(result["selected_image"], ("decoded_rgb", result["selected_slot"], 28))
                expected_calls = [(r["step"],s) for r in expected["trace"] if r["observed"] for s in r["alive_before"]]
                self.assertEqual(calls, expected_calls)
                for a, b in zip(result["trace"], expected["trace"]):
                    for key in ("step", "alive_before", "alive_after", "removed_reasons", "observation_reason", "next_observation"):
                        self.assertEqual(a[key], b[key])
                    self.assertEqual(a["personal_scores"], {str(s):v for s,v in b["personal_scores"].items()})
                for saved in snaps.values():
                    resumed = online_search(self.policy, {}, advance, observe, lambda s:None, copy.deepcopy(saved))
                    self.assertEqual(result, resumed)

    def test_all200_scalar_batch_and_poison(self):
        data = np.array([[[e["personal"][(t,s)] for s in e["seeds"]] for t in range(1,29)]
                         for _,e in sorted(self.tasks.items())])
        winners, views, rounds = batch_replay(data, self.policy)
        for i, (_, e) in enumerate(sorted(self.tasks.items())):
            obs = CausalObserver(e["personal"], e["seeds"])
            result = sparse_replay(self.policy, e["seeds"], obs)
            self.assertEqual(result["selected_seed"], e["seeds"][winners[i]])
            self.assertEqual(result["decode_images"], views[i])
            self.assertEqual(result["scoring_rounds"], rounds[i])
            used = set(obs.accesses)
            poison = {k:v if k in used else float("nan") for k,v in e["personal"].items()}
            self.assertEqual(result, sparse_replay(self.policy, e["seeds"], CausalObserver(poison, e["seeds"])))

    def test_ties_budget_deadline_and_locked_skip(self):
        seeds = [80,20,70,10,60,30,50,40]
        points = {(t,s):1.0 for t in range(1,29) for s in seeds}
        policy = SparsePolicy(first=8, margin=1, near_interval=28, far_interval=28)
        result = sparse_replay(policy, seeds, CausalObserver(points,seeds))
        self.assertEqual(result["selected_seed"], 10)
        self.assertEqual(result["particle_steps"], 112)
        self.assertTrue(any(r["observation_reason"]=="budget_deadline" for r in result["trace"]))
        self.assertTrue(any("budget_forced" in r["removed_reasons"].values() for r in result["trace"]))
        self.assertEqual(observation_reason(policy,9,67,3,9), "")  # Remaining45/19 requires3.
        self.assertEqual(observation_reason(policy,28,112,2,100), "terminal")

    def test_reject_incomplete_reward(self):
        with self.assertRaisesRegex(ValueError,"finite score"):
            online_search(self.policy, dict.fromkeys(range(8),0),
                          lambda s,l,t,d:(t,None), lambda images:{}, lambda state:None)


if __name__ == "__main__":
    unittest.main()
