"""Synthetic only. No real cache loader, parameter sweep or Personal benchmark."""
import copy
from functools import lru_cache
import json
import random
import unittest

from fashion_psp.search.deferred_boundary_psp import PartialCausalObserver
from fashion_psp.search.fully_adaptive_psp import (
    RULE, feasible_counts, replay, initial_control, plan, decide,
)
from fashion_psp.experiments.run_dualfashion_fitb_fully_adaptive import online_search


def points(top=3, delayed=False):
    return {(t, s): float(10 if s < top else 9 if delayed and t == 5 else 0)
            for t in range(1, 29) for s in range(8)}


def pruning(result):
    return [(r["step"], r["count_after"]) for r in result["trace"]
            if r["count_after"] < r["count_before"] and r["step"] < 28]


class FullyAdaptiveTests(unittest.TestCase):
    def run_replay(self, p, seeds=range(8)):
        observer = PartialCausalObserver(p, seeds)
        return replay(list(seeds), observer), observer

    def test_first_prune_time_and_counts_really_change(self):
        a, _ = self.run_replay(points(3))
        b, _ = self.run_replay(points(4))
        c, _ = self.run_replay(points(5))
        d, _ = self.run_replay(points(3, delayed=True))
        e, _ = self.run_replay({(t, s): 1.0 for t in range(1, 29) for s in range(8)})
        self.assertEqual([pruning(r)[0] for r in (a, b, c, d, e)],
                         [(7, 3), (7, 4), (7, 5), (8, 3), (9, 4)])
        for result in (a, b, c, d, e):
            self.assertEqual(result["particle_steps"], 112)
            self.assertEqual(result["trace"][4]["count_after"], 8)  # probe != prune

    def test_sparse_gap_trigger_changes_observation_time(self):
        # Same first prune (7,3); closing gap history brings the next observation forward.
        a = points(3)
        b = points(3)
        for t, gap in ((5, 6), (6, 4), (7, 2)):
            b[t, 2] = 10-gap
        # candidate2's noisy 3-point gap is NOT a stable loser; no terminal data involved.
        ra, _ = self.run_replay(a)
        rb, _ = self.run_replay(b)
        self.assertEqual(pruning(ra)[0], (7, 3))
        self.assertEqual(pruning(rb)[0], (7, 3))
        self.assertEqual(ra["trace"][6]["next_observation_step"], 19)
        self.assertEqual(rb["trace"][6]["next_observation_step"], 8)

    def test_ties_and_budget_forced_are_explicit(self):
        seeds = [80, 20, 70, 10, 60, 30, 50, 40]
        r, _ = self.run_replay({(t, s): 1.0 for t in range(1, 29) for s in seeds}, seeds)
        self.assertEqual(pruning(r), [(9, 4), (10, 2)])
        self.assertEqual(r["selected_seed"], 10)
        self.assertEqual(r["trace"][8]["alive_after"], [10, 20, 30, 40])
        self.assertEqual(r["decode_images"], 46)
        for row in r["trace"][:-1]:
            self.assertTrue(all(reason == "budget_forced" for reason in row["removed_reasons"].values()))

    def test_gap_in_observations_does_not_fake_consecutive_window(self):
        r, _ = self.run_replay(points(3))
        self.assertEqual(pruning(r), [(7, 3), (21, 2)])
        self.assertFalse(r["trace"][18]["consecutive_window_complete"])
        self.assertFalse(r["trace"][19]["consecutive_window_complete"])
        self.assertTrue(r["trace"][20]["consecutive_window_complete"])
        self.assertEqual(r["decode_images"], 35)

    def test_pruned_late_winner_is_not_recoverable(self):
        p = points(3)
        for t in range(8, 29):
            p[t, 7] = 100.0
        r, _ = self.run_replay(p)
        self.assertNotEqual(r["selected_seed"], 7)  # honest limitation, not a guarantee

    def test_exact_feasibility_matches_exhaustive_small_schedules(self):
        @lru_cache(None)
        def costs(h, n, f):
            if h == 1:
                return {n}
            return {n+c for m in range(f, n+1) for c in costs(h-1, m, f)}
        for f in (1, 2):
            for h in range(1, 6):
                for n in range(f, 7):
                    expected = {m: costs(h, m, f) for m in range(f, n+1)}
                    for remaining in range(h*f-1, h*n+2):
                        actual = feasible_counts(0, 0, n, horizon=h, budget=remaining, finalists=f)
                        self.assertEqual(actual, [m for m in expected if remaining in expected[m]])

    def test_random_causality_poisoning_and_exact_budget(self):
        rng = random.Random(20261025)
        for _ in range(100):
            p = {(t, s): rng.gauss(0, 10) for t in range(1, 29) for s in range(8)}
            r, obs = self.run_replay(p)
            self.assertEqual(sum(row["count_before"] for row in r["trace"]), 112)
            self.assertEqual(sum(row["evaluation_count"] for row in r["trace"]), len(obs.accesses))
            self.assertEqual(r["decode_images"], len(obs.accesses))
            used = set(obs.accesses)
            poisoned = {k: v if k in used else float("nan") for k, v in p.items()}
            same, _ = self.run_replay(poisoned)
            self.assertEqual(r, same)
            for row in r["trace"]:
                self.assertTrue(set(row["alive_after"]) <= set(row["alive_before"]))
                if row["step"] < 28:
                    self.assertGreaterEqual(row["count_after"], 2)

    def test_online_parity_decode_flags_and_every_step_resume(self):
        rng = random.Random(9182)
        scenarios = [points(3), points(5), points(3, True)]
        scenarios += [{(t, s): rng.gauss(0, 1) for t in range(1, 29) for s in range(8)}
                      for _ in range(7)]
        for p in scenarios:
            expected, _ = self.run_replay(p)
            snapshots = []
            def advance(slot, latent, step, decode):
                self.assertEqual(latent, step-1)
                self.assertEqual(decode, expected["trace"][step-1]["observed"])
                return step, (slot, step) if decode else None
            def observe(images):
                scores = {s: p[t, s] for s, (_, t) in images.items()}
                for s, (_, t) in list(images.items()):
                    images[s] = ("rgb", s, t)
                return scores
            def checkpoint(state):
                snapshot = copy.deepcopy(state)
                # Control survives JSON roundtrip (the neural checkpoint uses torch.save).
                snapshot["control"] = json.loads(json.dumps(snapshot["control"]))
                snapshots.append(snapshot)
            actual = online_search(RULE, dict.fromkeys(range(8), 0), advance, observe, checkpoint)
            self.assertEqual(actual["selected_slot"], expected["selected_seed"])
            self.assertEqual(actual["reward_views"], expected["decode_images"])
            self.assertEqual(actual["selected_image"], ("rgb", actual["selected_slot"], 28))
            for x, y in zip(actual["trace"], expected["trace"]):
                for key in ("step", "alive_before", "alive_after", "evaluated_seeds",
                            "removed_reasons", "next_observation_step", "stability_evidence"):
                    self.assertEqual(x.get(key), y.get(key))
            for snapshot in snapshots:
                resumed = online_search(RULE, {}, advance, observe, lambda s: None, copy.deepcopy(snapshot))
                self.assertEqual(actual, resumed)

    def test_reject_unplanned_or_nonfinite_scores(self):
        with self.assertRaisesRegex(ValueError, "planned"):
            replay(list(range(8)), lambda t, live, requested: {s: 1.0 for s in live})
        with self.assertRaisesRegex(ValueError, "finite"):
            replay(list(range(8)), lambda t, live, requested: {s: float("nan") for s in requested})
        control = initial_control()
        action = plan(control, 5, 40, 8)
        bad_action = {**action, "observe": False}
        with self.assertRaisesRegex(AssertionError, "plan changed"):
            decide(control, 5, 40, list(range(8)), dict.fromkeys(range(8), 1.0), bad_action)


if __name__ == "__main__":
    unittest.main()
