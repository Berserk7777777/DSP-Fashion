"""Synthetic branch, access-isolation and budget checks before test replay."""
import random
import unittest
from fashion_psp.search.gap_budget_psp import first_decision,replay
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import CausalObserver


class GapBudgetTests(unittest.TestCase):
    def test_both_gap_branches_and_ties(self):
        scores=dict(enumerate([10,9,8,2,1,0,-1,-2]))
        keep,second,_=first_decision(scores)
        self.assertEqual((keep,second),([0,1,2],21))
        scores=dict(enumerate([10,9,8,7,1,0,-1,-2]))
        keep,second,_=first_decision(scores)
        self.assertEqual((keep,second),([0,1,2,3],14))
        seeds=[80,20,70,10,60,30,50,40]
        keep,second,_=first_decision(dict.fromkeys(seeds,1))
        self.assertEqual((keep,second),([10,20,30,40],14))

    def test_budget_causality_and_seed_equivariance(self):
        rng=random.Random(601)
        branches=set()
        for _ in range(100):
            points={(t,s):rng.gauss(0,1) for t in range(1,29) for s in range(8)}
            obs=CausalObserver(points,range(8));r=replay(list(range(8)),obs)
            branches.add(r["trace"][6]["count_after"])
            self.assertEqual(r["particle_steps"],112)
            self.assertIn(r["decode_images"],(13,14))
            self.assertEqual(len(obs.accesses),r["decode_images"])
            used=set(obs.accesses)
            poison={k:v if k in used else float("nan") for k,v in points.items()}
            self.assertEqual(r,replay(list(range(8)),CausalObserver(poison,range(8))))
            mapped={(t,s+1000):v for (t,s),v in points.items()}
            shifted=replay(list(range(1000,1008)),CausalObserver(mapped,range(1000,1008)))
            self.assertEqual(shifted["selected_seed"],r["selected_seed"]+1000)
        self.assertEqual(branches,{3,4})

    def test_first_decision_positive_affine_invariance(self):
        scores=dict(enumerate([10,9,8,2,1,0,-1,-2]))
        a=first_decision(scores)
        b=first_decision({s:3*v+25 for s,v in scores.items()})
        self.assertEqual(a[:2],b[:2])


if __name__=="__main__":unittest.main()
