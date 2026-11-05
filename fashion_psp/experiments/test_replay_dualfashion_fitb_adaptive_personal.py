"""Checks for exact-budget planning, frozen semantics and causal observations."""
import itertools
import unittest

from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import (
    BUDGET, CausalObserver, adaptive_decision, reachable_costs, replay,
)


class ReplayTests(unittest.TestCase):
    def test_reachability_matches_exhaustive_monotone_paths(self):
        for horizon in range(1, 6):
            for count in range(1, 6):
                exhaustive = {sum(path) for path in itertools.product(range(1, count + 1), repeat=horizon)
                              if path[0] == count and all(a >= b for a, b in zip(path, path[1:]))}
                self.assertEqual(reachable_costs(horizon, count), exhaustive)

    def test_stability_is_strict_and_sample_std(self):
        # For gaps [2,2,5], mean=3, sample std=sqrt(3); population std would mark it.
        histories = {0: [10, 10, 10], 1: [8, 8, 5], 2: [8, 8, 8], 3: [10, 10, 10],
                     4: [10, 9, 9], 5: [0, 0, 0], 6: [0, 0, 0], 7: [0, 0, 0]}
        _, _, evidence = adaptive_decision(histories, 3, 24)
        self.assertEqual(evidence["leader"], 0)
        self.assertEqual(set(evidence["eligible_seeds"]), {2, 5, 6, 7})

    def test_reference_is_fixed_current_leader(self):
        histories = {0: [0, 0, 10], 1: [9, 9, 9], **{s: [0, 0, 0] for s in range(2, 8)}}
        _, _, evidence = adaptive_decision(histories, 3, 24)
        self.assertEqual(evidence["leader"], 0)
        self.assertEqual(evidence["stability_evidence"]["1"]["gaps"], [-9, -9, 1])
        self.assertNotIn(1, evidence["eligible_seeds"])

    def test_exact_budget_ties_and_forced_pruning(self):
        seeds = list(range(8))
        data = {(step, seed): 1.0 for step in range(1, 29) for seed in seeds}
        for method, views, rounds in (("default", 14, 3), ("adaptive", 112, 28)):
            result = replay(seeds, method, CausalObserver(data, seeds))
            self.assertEqual(result["particle_steps"], BUDGET)
            self.assertEqual(result["selected_seed"], 0)
            self.assertEqual(result["decode_images"], views)
            self.assertEqual(result["scoring_rounds"], rounds)
            self.assertEqual(sum(r["count_before"] for r in result["trace"]), 112)
            if method == "adaptive":
                reasons = [reason for r in result["trace"] if r["step"] < 28
                           for reason in r["removed_reasons"].values()]
                self.assertTrue(reasons)
                self.assertEqual(set(reasons), {"budget_forced"})

    def test_stable_pruning_is_limited_by_budget_and_no_dead_future_access(self):
        seeds = list(range(8))
        data = {(step, seed): float(20 - seed) for step in range(1, 29) for seed in seeds}
        observer = CausalObserver(data, seeds)
        result = replay(seeds, "adaptive", observer)
        events = [r for r in result["trace"] if r["count_before"] > r["count_after"]]
        self.assertEqual(events[0]["step"], 3)
        self.assertGreater(events[0]["count_after"], 1)  # Prune-all-marked would underspend.
        self.assertTrue(all(v == "stability_heuristic" for v in events[0]["removed_reasons"].values()))
        accesses = set(observer.accesses)
        for event in events:
            dead = set(event["alive_before"]) - set(event["alive_after"])
            self.assertFalse(any((step, seed) in accesses for seed in dead for step in range(event["step"] + 1, 29)))
        # Poison every unobserved future: the complete policy output must stay identical.
        poisoned = {key: value if key in accesses else -1e99 for key, value in data.items()}
        self.assertEqual(result, replay(seeds, "adaptive", CausalObserver(poisoned, seeds)))

    def test_observer_rejects_time_travel_and_resurrection(self):
        data = {(t, s): 1.0 for t in range(1, 29) for s in (1, 2)}
        observer = CausalObserver(data, [1, 2])
        with self.assertRaises(AssertionError):
            observer(2, [1, 2], True)
        observer(1, [1, 2], True)
        observer(2, [1], True)
        with self.assertRaises(AssertionError):
            observer(3, [1, 2], True)


if __name__ == "__main__":
    unittest.main()
