"""Synthetic checks for ties, loss attribution, and resource-only enumeration."""
import unittest
from fashion_psp.experiments.diagnose_personal_test_difficulty import rank_correlation, resource_schedules, task_diagnosis


class DiagnosisTests(unittest.TestCase):
    def test_correlation_ties_and_constant(self):
        self.assertAlmostEqual(rank_correlation({1:1,2:2,3:2},{1:1,2:2,3:2}),1)
        self.assertAlmostEqual(rank_correlation({1:1,2:2},{1:2,2:1}),-1)
        self.assertIsNone(rank_correlation({1:1,2:1},{1:2,2:1}))

    def test_tied_oracle_and_telescoping_losses(self):
        e={"seeds":list(range(8)),"identity":(1,2),"personal":{},"paths":{}}
        for t in range(1,29):
            for s in e["seeds"]:
                e["personal"][(t,s)]=float(7-s)
                e["paths"][(t,s)]="synthetic"
        # Seed7 is eliminated at7, but seed0 is an exactly tied terminal oracle.
        e["personal"][(28,7)]=e["personal"][(28,0)]=10.0
        info,_=task_diagnosis(0,e)
        self.assertEqual(info["oracle_loss_step"],0)
        self.assertEqual(info["default_regret"],0)
        e["personal"][(28,7)]=12.0
        info,_=task_diagnosis(0,e)
        self.assertEqual(info["oracle_loss_step"],7)
        self.assertEqual(info["default_regret"],2)
        self.assertEqual(info["stage7_terminal_best_loss"]+info["stage14_terminal_best_loss"],2)

    def test_exhaustive_two_node_resource_identity(self):
        actual={(r["first_step"],r["first_keep"],r["second_step"],r["second_keep"]) for r in resource_schedules()}
        # 8+k1+k2<=14, k1>k2>=2 leaves only (3,2) and (4,2).
        expected={(a,3,56-5*a,2) for a in range(1,28) if a<56-5*a<28}
        expected|={(a,4,28-2*a,2) for a in range(1,28) if a<28-2*a<28}
        self.assertEqual(actual,expected)


if __name__=="__main__":unittest.main()
