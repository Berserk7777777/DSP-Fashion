"""Causality and budget checks for offline, training-free Personal policies."""
import unittest

import numpy as np

from fashion_psp.evaluation.analyze_dualfashion_training_free_psp import (
    adaptive_evaluate, evaluate, reward_view_count, schedule_cost,
)


class TrainingFreeTests(unittest.TestCase):
    def test_every_rule_ignores_pruned_particles_future(self):
        rng = np.random.default_rng(7)
        p = rng.normal(size=(3, 28, 16)).cumsum(axis=1)
        rules = ["raw", "mean_3", "rank_mean_3", "rank_recent_5", "rank_trend_5",
                 "slope_5", "lcb_3", "ucb_3", "rescue_5"]
        for rule in rules:
            values, winner, trace = evaluate(p, 8, [(7, 4), (14, 2)], rule, capture=True)
            changed = p.copy()
            for task in range(3):
                removed = sorted(set(range(8)) - set(trace[0][1][task]))
                changed[task, 7:, removed] = 1e9
            other, other_winner, _ = evaluate(changed, 8, [(7, 4), (14, 2)], rule)
            np.testing.assert_array_equal(values, other)
            np.testing.assert_array_equal(winner, other_winner)

    def test_first_prune_does_not_read_future(self):
        p = np.random.default_rng(8).normal(size=(5, 28, 16))
        changed = p.copy()
        changed[:, 7:, :] *= -100
        for rule in ("raw", "rank_mean_3", "slope_5", "rescue_5"):
            _, _, a = evaluate(p, 8, [(7, 4), (14, 2)], rule, capture=True)
            _, _, b = evaluate(changed, 8, [(7, 4), (14, 2)], rule, capture=True)
            np.testing.assert_array_equal(a[0][1], b[0][1])

    def test_budget_and_reward_costs(self):
        self.assertEqual(schedule_cost(8, [(7, 4), (14, 2)]), 112)
        self.assertEqual(schedule_cost(12, [(1, 8), (5, 4), (10, 3), (22, 2)]), 112)
        self.assertEqual(reward_view_count(8, [(7, 4), (14, 2)]), 14)
        self.assertEqual(reward_view_count(8, [(7, 4), (14, 2)], "rank_mean_3"), 38)
        p = np.random.default_rng(9).normal(size=(20, 28, 16))
        for threshold in (0, 0.2, 0.8):
            _, _, times = adaptive_evaluate(p, threshold)
            for time in times:
                self.assertEqual(schedule_cost(8, [(int(time), 4), (28 - 2 * int(time), 2)]), 112)


if __name__ == "__main__":
    unittest.main()
