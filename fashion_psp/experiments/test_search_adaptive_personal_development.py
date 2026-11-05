"""Independent checks of causal adaptive development replay implementations."""
import unittest

import numpy as np

from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import CausalObserver, reachable_costs
from fashion_psp.experiments.search_adaptive_personal_development import (
    Policy, decide_scalar, feasible_counts, replay_batch, replay_scalar,
)


class DevelopmentSearchTests(unittest.TestCase):
    def test_closed_budget_interval_equals_recursive_reachability(self):
        for h in range(1, 28):
            for n in range(1, 9):
                self.assertEqual(reachable_costs(h, n), frozenset(range(n+h-1, n*h+1)))
        for step in range(1, 28):
            for spent in range(8, 113):
                for n in (1, 4, 8):
                    expected = [m for m in range(1, n+1) if 112-spent in reachable_costs(28-step, m)]
                    if expected:
                        self.assertEqual(feasible_counts(step, spent, n), expected)
                    else:
                        with self.assertRaises(AssertionError):
                            feasible_counts(step, spent, n)

    def test_batch_scalar_and_poisoning_on_independent_synthetic_paths(self):
        rng = np.random.default_rng(903)
        data = rng.normal(size=(6, 28, 8)).cumsum(axis=1) + rng.normal(size=(6, 1, 8))
        data[0] = 1.0  # Exact ties must keep the smaller integer seed.
        policies = [Policy(), Policy(min_step=3, protect_top=1, max_drop=1),
                    Policy(min_step=9, window=5, z=0.5, margin=1, protect_top=3, max_drop=2,
                           trend_horizon=2.0, mean_weight=0.25),
                    Policy(min_step=11, window=7, z=6, margin=6, protect_top=4, max_drop=8,
                           trend_horizon=4, mean_weight=1)]
        seeds = list(range(201, 209))
        for policy in policies:
            winners, masks = replay_batch(data, policy, True)
            poisoned = data.copy()
            for t, mask in enumerate(masks):
                poisoned[:, t, :][~mask] = 1e90
            winners2, masks2 = replay_batch(poisoned, policy, True)
            np.testing.assert_array_equal(winners, winners2)
            np.testing.assert_array_equal(masks, masks2)
            for task in range(6):
                points = {(t+1, seeds[s]): float(data[task, t, s]) for t in range(28) for s in range(8)}
                observer = CausalObserver(points, seeds)
                result = replay_scalar(policy, seeds, observer)
                self.assertEqual(result["selected_seed"], seeds[winners[task]])
                self.assertEqual(result["particle_steps"], 112)
                self.assertEqual(len(observer.accesses), 112)
                for t, r in enumerate(result["trace"]):
                    self.assertEqual(r["alive_before"], [seeds[s] for s in np.flatnonzero(masks[t][task])])

    def test_warmup_top_protection_and_pruning_cap(self):
        policy = Policy(min_step=5, window=3, z=1, margin=2, protect_top=2, max_drop=2)
        h = {s: [20-s]*4 for s in range(8)}
        alive, reasons, _ = decide_scalar(policy, h, 4, 32)
        self.assertEqual(alive, list(range(8)))
        self.assertFalse(reasons)
        h = {s: [20-s]*5 for s in range(8)}
        alive, reasons, _ = decide_scalar(policy, h, 5, 40)
        self.assertEqual(len(alive), 6)
        self.assertTrue({0, 1} <= set(alive))
        self.assertEqual(set(reasons.values()), {"stability_heuristic"})


if __name__ == "__main__":
    unittest.main()
