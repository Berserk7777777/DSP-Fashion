"""Online callback loop parity and persisted-state resume checks; no GPU needed."""
import copy
import unittest
import numpy as np
from fashion_psp.experiments.run_dualfashion_fitb_adaptive_personal import online_search
from fashion_psp.experiments.search_adaptive_personal_development import Policy, replay_scalar
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import CausalObserver


class OnlineTests(unittest.TestCase):
    def test_replay_parity_and_resume(self):
        policy = Policy(min_step=4, window=3, z=0, margin=0, protect_top=2,
                        max_drop=2, trend_horizon=0, mean_weight=0)
        rng = np.random.default_rng(531)
        for sample in range(5):
            data = rng.normal(size=(28, 8)).cumsum(0)
            if sample == 0:
                data[:] = 1
            points = {(step, s): float(data[step-1, s]) for step in range(1, 29) for s in range(8)}
            expected = replay_scalar(policy, list(range(8)), CausalObserver(points, range(8)))
            snapshots = {}
            def advance(slot, latent, step, decode):
                self.assertEqual(latent, step-1)
                self.assertTrue(decode)
                return step, (slot, step)
            def observe(images):
                return {s: points[(t, s)] for s, t in images.values()}
            def checkpoint(state):
                snapshots[state['completed_step']] = copy.deepcopy(state)
            result = online_search(policy, dict.fromkeys(range(8), 0), advance, observe, checkpoint)
            self.assertEqual(result['spent'], 112)
            self.assertEqual(result['reward_views'], 112)
            self.assertEqual(result['selected_slot'], expected['selected_seed'])
            for actual, reference in zip(result['trace'], expected['trace']):
                for key in ('step', 'alive_before', 'alive_after', 'removed_reasons'):
                    self.assertEqual(actual[key], reference[key])
            for stop in (3, 5, 14, 27, 28):
                resumed = online_search(policy, dict.fromkeys(range(8), 0), advance, observe,
                                        lambda state: None, copy.deepcopy(snapshots[stop]))
                self.assertEqual(result, resumed)


if __name__ == '__main__':
    unittest.main()
