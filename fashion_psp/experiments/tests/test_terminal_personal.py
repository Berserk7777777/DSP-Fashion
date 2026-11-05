"""Regression checks for causal access, split isolation, budgets, and resuming."""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fashion_psp.experiments.train_dualfashion_terminal_personal import (
    load_trajectories, simulate_task, split_users,
)
from fashion_psp.search.terminal_personal import prefix_features, uncertainty_survivors


class VisiblePrefixPredictor:
    q = 1.0

    def __init__(self):
        self.seen_lengths = []

    def predict(self, scores, sigmas):
        self.seen_lengths.extend(map(len, scores))
        return np.asarray([row[-1] for row in scores]), np.ones(len(scores))


def make_task(task=0, uid=7):
    return {(task, 2026 + task * 10000 + slot):{
        "uid":uid, "oid":100 + task, "task_index":task,
        "seed":2026 + task * 10000 + slot,
        "personal":np.linspace(slot, 10 + slot, 28),
        "sigma":np.linspace(1, 0, 28), "image_path":["unused"] * 28,
    } for slot in range(8)}


class ForecastPolicyTests(unittest.TestCase):
    def test_future_change_cannot_change_visible_features(self):
        score, sigma = np.arange(28, dtype=float), np.linspace(1, 0, 28)
        expected = prefix_features(score[:7], sigma[:7])
        score[7:] += 9999
        sigma[7:] -= 9999
        np.testing.assert_array_equal(expected, prefix_features(score[:7], sigma[:7]))

    def test_whitelist_loader_ignores_gt_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scores.csv"
            fields = ["task_index", "uid", "oid", "seed", "step", "personal", "sigma", "image_path", "gt_sim"]
            rows = [{"task_index":0, "uid":7, "oid":100, "seed":seed, "step":step,
                     "personal":step + seed / 100, "sigma":1 / step, "image_path":"unused",
                     "gt_sim":"NOT A NUMBER; MUST NEVER BE PARSED"}
                    for seed in range(2026, 2034) for step in range(1, 29)]
            def save():
                with path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            save()
            loaded = load_trajectories(path, 1)
            self.assertEqual(len(loaded), 8)
            self.assertNotIn("gt_sim", loaded[(0, 2026)])
            rows.append(rows[0])
            save()
            with self.assertRaises(ValueError):
                load_trajectories(path, 1)

    def test_same_user_all_particles_and_tasks_stay_in_one_split(self):
        trajectories = {}
        for task in range(24):
            trajectories.update(make_task(task, uid=task // 2))
        splits = split_users(trajectories, 2026)
        assigned = {}
        for name, part in splits.items():
            for uid in part["uids"]:
                self.assertNotIn(uid, assigned)
                assigned[uid] = name
            for task in part["task_indices"]:
                self.assertIn(task // 2, part["uids"])
        self.assertEqual(len(assigned), 12)

    def test_replay_never_observes_future_at_pruning_and_spends_112(self):
        trajectories = make_task()
        predictor = VisiblePrefixPredictor()
        winner, cost, trace = simulate_task(trajectories, 0, predictor, "uncertainty_anchor_ucb")
        self.assertEqual(cost, 112)
        self.assertEqual(predictor.seen_lengths, [7] * 8 + [14] * 4)
        changed = copy.deepcopy(trajectories)
        for row in changed.values():
            row["personal"][14:] *= -10
        _, changed_cost, changed_trace = simulate_task(changed, 0, VisiblePrefixPredictor(), "uncertainty_anchor_ucb")
        self.assertEqual(cost, changed_cost)
        self.assertEqual(trace[:2], changed_trace[:2])
        self.assertEqual(winner, 2033)

    def test_overlapping_removals_are_explicitly_forced(self):
        alive, trace = uncertainty_survivors([1, 2, 3, 4], [8, 7, 6, 5], [10] * 4, 1, 2)
        self.assertIn(1, alive)
        self.assertEqual(trace["forced_prunes"], 2)
        self.assertEqual(trace["interval_dominated_prunes"], 0)
        alive, trace = uncertainty_survivors([1, 2, 3, 4], [8, 7, 1, 0], [0.1] * 4, 1, 2)
        self.assertEqual(alive, [1, 2])
        self.assertEqual(trace["forced_prunes"], 0)
        self.assertEqual(trace["interval_dominated_prunes"], 2)


if __name__ == "__main__":
    unittest.main()
