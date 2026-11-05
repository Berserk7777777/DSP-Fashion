import unittest
import tempfile
from pathlib import Path
import numpy as np

from fashion_psp.evaluation.research_dualfashion_state_risk_dp import StateRiskDP, fit_model


class StateRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(91)
        cls.plan = rng.normal(size=(12, 28, 16)).cumsum(1)
        cls.model = fit_model(cls.plan, repeats=1, rounds=0)

    def test_budget_all_particle_counts_and_bon4(self):
        p = np.random.default_rng(92).normal(size=(4, 28, 16)).cumsum(1)
        for k in range(4, 17):
            result = self.model.replay(p, k)
            np.testing.assert_array_equal(result["spent"], 112)
            self.assertTrue((result["winner"] < k).all())
            if k == 4:
                np.testing.assert_allclose(result["personal"], p[:, 27, :4].max(1))

    def test_boundary_state_changes_current_action(self):
        shape = (29, 17, 17, 4, 4)
        core = np.zeros((29, 17, 17, 4))
        transition = np.zeros(shape + (4,))
        transition[..., 0] = 1
        core_transition = np.zeros(core.shape + (4,))
        core_transition[..., 0] = 1
        cost = np.full(shape, 100.)
        cost[7, 8, 5] = 1
        cost[7, 8, 4, :, 3] = 0
        cost[7, 8, 4, :, 1] = 200
        model = StateRiskDP(np.ones((29, 17)), core, core_transition, cost, transition, np.ones(shape))
        current = np.zeros((2, 16))
        current[0, :8] = [10, 9, 8, 7, 0, -1, -2, -3]
        current[1, :8] = [10, 9, 8, 7, 6.9, 6.8, 6.7, 6.6]
        alive = np.broadcast_to(np.arange(16) < 8, current.shape).copy()
        selected, _, _, _ = model.decide(current, alive, 7, np.full(2, 56))
        np.testing.assert_array_equal(selected.sum(1), [4, 5])

    def test_removed_future_has_no_effect(self):
        p = np.random.default_rng(93).normal(size=(5, 28, 16)).cumsum(1)
        first = self.model.replay(p, 8)
        changed = p.copy()
        for task, trace in enumerate(first["trace"]):
            alive = set(range(8))
            for cut in trace:
                new_alive = set(cut["slots"])
                for removed in alive - new_alive:
                    changed[task, cut["step"]:, removed] = 1e8
                alive = new_alive
        second = self.model.replay(changed, 8)
        self.assertEqual(first["trace"], second["trace"])
        np.testing.assert_array_equal(first["winner"], second["winner"])

    def test_statistics_are_nonnegative_and_normalized(self):
        self.assertTrue((self.model.loss >= -1e-12).all())
        for n in range(1, 17):
            for m in range(1, n + 1):
                np.testing.assert_allclose(self.model.transition[1:28, n, m].sum(-1), 1, atol=1e-10)
            np.testing.assert_allclose(self.model.loss[:, n, n], 0)

    def test_saved_policy_reproduces_actions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "planning.npz"
            self.model.save(path)
            restored = StateRiskDP.load(path)
        first = self.model.replay(self.plan[:3], 8)
        second = restored.replay(self.plan[:3], 8)
        np.testing.assert_array_equal(first["winner"], second["winner"])
        self.assertEqual(first["trace"], second["trace"])


if __name__ == "__main__":
    unittest.main()
