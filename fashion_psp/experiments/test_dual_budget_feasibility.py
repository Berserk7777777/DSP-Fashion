"""Exhaustive small-problem resource checks independent of all test rewards."""
import itertools
import unittest
from fashion_psp.search.dual_budget_feasibility import can_finish, feasible_actions


class ResourceTests(unittest.TestCase):
    def test_exhaustive_all_monotone_schedules(self):
        for h in range(1,5):
            for initial in range(1,5):
                schedules=[]
                for rest in itertools.product(range(1,initial+1),repeat=h-1):
                    ns=(initial,)+rest
                    if any(b>a for a,b in zip(ns,ns[1:])):
                        continue
                    # Score the population just BEFORE every strict drop and
                    # score the terminal population. Unscored steps cost0 views.
                    views=ns[-1]+sum(a for a,b in zip(ns,ns[1:]) if b<a)
                    schedules.append((sum(ns),views))
                for q in range(0,17):
                    expected={cost for cost,views in schedules if views<=q}
                    actual={b for b in range(h*initial+1) if can_finish(h,initial,b,q)}
                    self.assertEqual(actual,expected,(h,initial,q))

    def test_default_reserves_terminal_and_all_intermediate_views(self):
        self.assertTrue(can_finish(28,8,112,14))
        a=feasible_actions(7,56,8,0)
        self.assertIn(4,a["observe_keep"])
        b=feasible_actions(14,84,4,8)
        self.assertIn(2,b["observe_keep"])
        self.assertEqual(feasible_actions(28,112,2,12)["observe_keep"],[1])
        self.assertEqual(feasible_actions(28,112,2,13)["observe_keep"],[])

    def test_denoise_feasible_but_decode_infeasible(self):
        # Only6 views remain after scoring8 at step7. Keeping5 can complete
        # the denoising sum alone, but cannot finish within the14-view contract.
        self.assertNotIn(5,feasible_actions(7,56,8,0,14)["observe_keep"])
        self.assertIn(5,feasible_actions(7,56,8,0,16)["observe_keep"])
        # Paying for a second full8-particle observation exhausts16 views and
        # still requires terminal decoding, so it is not a feasible action.
        self.assertEqual(feasible_actions(8,64,8,8,16)["observe_keep"],[])


if __name__=="__main__":unittest.main()
