"""Synthetic-only mechanism checks; deliberately no test-cache loader."""
import random
import unittest

from fashion_psp.search.deferred_boundary_psp import PartialCausalObserver,replay
from fashion_psp.search.gap_budget_psp import replay as immediate
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import CausalObserver


def trajectories(first):
    return {(t,s):float(first[s]) for t in range(1,29) for s in range(8)}


class DeferredBoundaryTests(unittest.TestCase):
    def test_late_boundary_challenger_can_be_retained(self):
        p=trajectories([10,9,8,2,1,0,-1,-2])
        for t in range(8,29):p[(t,4)]=20
        obs=PartialCausalObserver(p,range(8));r=replay(list(range(8)),obs)
        base=immediate(list(range(8)),CausalObserver(p,range(8)))
        self.assertEqual(r["selected_seed"],4);self.assertNotEqual(base["selected_seed"],4)
        self.assertEqual(r["particle_steps"],112);self.assertEqual(r["decode_images"],16)
        self.assertEqual(r["scoring_rounds"],4)
        self.assertEqual(r["trace"][6]["alive_after"],[0,1,2,3,4])
        self.assertEqual(r["trace"][7]["evaluated_seeds"],[2,3,4])
        self.assertEqual(r["trace"][7]["alive_after"],[0,1,4])
        self.assertNotIn((8,0),obs.accesses);self.assertNotIn((8,1),obs.accesses)

    def test_early_false_challenger_can_also_hurt(self):
        p=trajectories([10,9,8,2,1,0,-1,-2])
        p[(8,4)]=30
        for t in range(9,29):p[(t,2)]=40
        r=replay(list(range(8)),PartialCausalObserver(p,range(8)))
        b=immediate(list(range(8)),CausalObserver(p,range(8)))
        self.assertEqual(b["selected_seed"],2);self.assertNotEqual(r["selected_seed"],2)

    def test_conservative_branch_is_identical_to_v1(self):
        p=trajectories([10,9,8,7,1,0,-1,-2])
        r=replay(list(range(8)),PartialCausalObserver(p,range(8)))
        b=immediate(list(range(8)),CausalObserver(p,range(8)))
        self.assertEqual(r["selected_seed"],b["selected_seed"])
        self.assertEqual(r["decode_images"],14);self.assertEqual(r["scoring_rounds"],3)
        self.assertEqual([(x["alive_before"],x["alive_after"],x["personal_scores"]) for x in r["trace"]],
                         [(x["alive_before"],x["alive_after"],x["personal_scores"]) for x in b["trace"]])

    def test_all_ties_reproducible_with_original_seeds(self):
        seeds=[80,20,70,10,60,30,50,40]
        p={(t,s):1.0 for t in range(1,29) for s in seeds}
        r=replay(seeds,PartialCausalObserver(p,seeds))
        self.assertEqual(r["selected_seed"],10);self.assertEqual(r["decode_images"],14)

    def test_random_paths_causality_counts_and_poisoning(self):
        rng=random.Random(20261025);branches=set()
        for _ in range(100):
            p={(t,s):rng.gauss(0,1) for t in range(1,29) for s in range(8)}
            obs=PartialCausalObserver(p,range(8));r=replay(list(range(8)),obs)
            branches.add(r["trace"][6]["deferred_branch"])
            self.assertEqual(r["particle_steps"],112)
            self.assertEqual(sum(row["count_before"] for row in r["trace"]),112)
            self.assertEqual(len(obs.accesses),r["decode_images"])
            self.assertEqual(sum(row["evaluation_count"] for row in r["trace"]),r["decode_images"])
            used=set(obs.accesses)
            poison={k:v if k in used else float("nan") for k,v in p.items()}
            self.assertEqual(r,replay(list(range(8)),PartialCausalObserver(poison,range(8))))
        self.assertEqual(branches,{False,True})

    def test_guard_rejects_resurrection_and_unlive_partial_access(self):
        obs=PartialCausalObserver({},range(8));obs(1,list(range(8)),[])
        obs(2,[0,1],[])
        with self.assertRaises(AssertionError):obs(3,[0,1],[2])
        with self.assertRaises(AssertionError):obs(3,[0,1,2],[])
        with self.assertRaises(AssertionError):obs(4,[0,1],[])

    def test_reject_extra_or_missing_scores(self):
        p=trajectories([10,9,8,2,1,0,-1,-2])
        def extra(step,live,requested):
            if step==8:return {s:p[(step,s)] for s in live}
            return {s:p[(step,s)] for s in requested}
        with self.assertRaisesRegex(ValueError,"exactly requested"):
            replay(list(range(8)),extra)
        with self.assertRaisesRegex(ValueError,"exactly requested"):
            replay(list(range(8)),lambda t,live,requested:{})


if __name__=="__main__":unittest.main()
