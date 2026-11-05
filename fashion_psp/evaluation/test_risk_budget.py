"""Budget, prefix-causality, and mathematical-bound checks."""
import unittest

import numpy as np

from fashion_psp.evaluation.research_dualfashion_risk_budget import (
    budget_dp, calibrate, choose_resume, replay_resume, replay_sync,
    select_keep,
)


class RiskBudgetTests(unittest.TestCase):
    def test_exact_budget_dp_matches_two_particle_enumeration(self):
        cost = np.zeros((29, 3, 3))
        cost[1:28, 2, 1] = np.arange(27) + 0.25
        # With two initial seeds and total31, the only cut is step3 ->1.
        value = budget_dp(cost, 2, budget=31)
        self.assertAlmostEqual(value[1, 2, 29], cost[3, 2, 1])
        self.assertTrue(np.isinf(value[1, 2, 0]))

    def test_exact112_and_bon4_equivalence(self):
        p = np.random.default_rng(1).normal(size=(5, 28, 16))
        cal = {"pair_q":np.full(28, 3.0), "absolute_q":np.r_[np.full(27, 3.0), 0],
               "center":np.zeros(28)}
        costs = np.zeros((29, 17, 17))
        value = budget_dp(costs, 16)
        for mode in ("risk_dp", "safe_budget", "empirical_dp"):
            for k in (4, 8, 16):
                result = replay_sync(p, k, cal, value, mode, empirical=costs)
                np.testing.assert_array_equal(result["spent"], 112)
                if k == 4:
                    np.testing.assert_allclose(result["personal"], p[:, 27, :4].max(1))
        for k in (4, 8, 16):
            result = replay_resume(p, k, cal)
            np.testing.assert_array_equal(result["spent"], 112)
            self.assertTrue((result["completed"] >= 1).all())
            if k == 4:
                np.testing.assert_allclose(result["personal"], p[:, 27, :4].max(1))

    def test_pairwise_loss_bound(self):
        rng = np.random.default_rng(2)
        current = rng.normal(size=(50, 16))
        residual = rng.normal(size=(50, 16))
        terminal = current + residual
        q = np.ptp(residual, axis=1).max()
        value = budget_dp(np.zeros((29, 17, 17)), 16)
        alive = np.ones((50, 16), bool)
        kept, bound = select_keep(current, alive, 1, np.full(50, 96), value, q, "risk_dp")
        loss = terminal.max(1) - np.where(kept, terminal, -np.inf).max(1)
        self.assertTrue((loss <= bound + 1e-10).all())

    def test_pruned_future_cannot_change_replay(self):
        p = np.random.default_rng(3).normal(size=(5, 28, 16))
        cal = {"pair_q":np.ones(28), "absolute_q":np.ones(28), "center":np.zeros(28)}
        costs = np.zeros((29, 17, 17))
        value = budget_dp(costs, 16)
        first = replay_sync(p, 16, cal, value, "risk_dp")
        changed = p.copy()
        for task, trace in enumerate(first["trace"]):
            previous = set(range(16))
            for cut in trace:
                remaining = set(cut["slots"])
                for removed in previous - remaining:
                    changed[task, cut["step"]:, removed] = 1e6
                previous = remaining
        second = replay_sync(changed, 16, cal, value, "risk_dp")
        np.testing.assert_array_equal(first["winner"], second["winner"])
        self.assertEqual(first["trace"], second["trace"])

    def test_resume_uses_observed_intervals_and_reserves_completion(self):
        levels = np.array([[5, 27], [10, 8]])
        lower = np.array([[0., 0.], [0., 0.]])
        upper = np.array([[100., 1.], [1., 100.]])
        chosen = choose_resume(levels, lower, upper, 1, exponent=0)
        np.testing.assert_array_equal(chosen, [1, 0])

    def test_group_calibration_covers_exact_common_drift(self):
        rng = np.random.default_rng(4)
        terminal = rng.normal(size=(20, 1, 16))
        drift = np.linspace(-5, 0, 28)[None, :, None]
        p = terminal + drift
        cal = calibrate(p, p, np.arange(20), alpha=0.1)
        np.testing.assert_allclose(cal["pair_q"], 0, atol=1e-10)
        np.testing.assert_allclose(cal["center"], -drift[0, :, 0], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
