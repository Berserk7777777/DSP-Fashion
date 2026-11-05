import copy
import unittest

import numpy as np

from fashion_psp.evaluation.research_dualfashion_state_risk_dp import fit_model
from fashion_psp.experiments.run_dualfashion_fitb_state_dp import online_search


class OnlineStateDPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fit_model(np.random.default_rng(19).normal(size=(12, 28, 16)).cumsum(1), repeats=1, rounds=0)

    def run_callbacks(self, p, checkpoint=lambda state:None, resumed=None):
        calls = []
        def advance(slot, latent, step, decode):
            calls.append((step, slot))
            return latent + 1, float(p[step - 1, slot]) if decode else None
        def observe(images):
            return dict(images)
        result = online_search(self.model, {i:0 for i in range(8)}, advance, observe, checkpoint, resumed)
        return result, calls

    def test_online_matches_frozen_trajectory_replay(self):
        p = np.random.default_rng(21).normal(size=(28, 16)).cumsum(0)
        expected = self.model.replay(p[None], 8)
        actual, calls = self.run_callbacks(p)
        self.assertEqual(len(calls), 112)
        self.assertEqual(actual['spent'], 112)
        self.assertEqual(actual['selected_slot'], expected['winner'][0])
        self.assertEqual(actual['selected_image'], expected['personal'][0])
        self.assertEqual(actual['reward_views'], expected['views'][0])
        actual_cuts = [(x['step'], x['alive_after']) for x in actual['trace'][:-1] if len(x['alive_after']) < len(x['alive_before'])]
        expected_cuts = [(x['step'], x['slots']) for x in expected['trace'][0]]
        self.assertEqual(actual_cuts, expected_cuts)

    def test_resume_repeats_no_completed_denoising(self):
        p = np.random.default_rng(22).normal(size=(28, 16)).cumsum(0)
        full, _ = self.run_callbacks(p)
        saved = {}
        def checkpoint(state):
            if state['completed_step'] == 9:
                saved.update(copy.deepcopy(state))
                raise InterruptedError('simulate interruption after atomic checkpoint')
        with self.assertRaises(InterruptedError):
            self.run_callbacks(p, checkpoint)
        spent_before = saved['spent']
        resumed, calls = self.run_callbacks(p, resumed=saved)
        self.assertTrue(all(step > 9 for step, _ in calls))
        self.assertEqual(len(calls) + spent_before, 112)
        self.assertEqual(full, resumed)

    def test_completed_step28_resume_needs_no_inference(self):
        p = np.random.default_rng(23).normal(size=(28, 16)).cumsum(0)
        state, _ = self.run_callbacks(p)
        resumed, calls = self.run_callbacks(p, resumed=copy.deepcopy(state))
        self.assertEqual(calls, [])
        self.assertEqual(state, resumed)


if __name__ == '__main__':
    unittest.main()
