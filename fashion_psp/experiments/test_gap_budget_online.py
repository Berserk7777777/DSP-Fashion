"""Real cached-score callback parity and resume checks; no neural calls."""
import copy
from pathlib import Path
import unittest
from fashion_psp.experiments.run_dualfashion_fitb_gap_budget import online_search
from fashion_psp.search.gap_budget_psp import replay,RULE
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import load_and_audit,CausalObserver

ROOT=Path(__file__).resolve().parents[2]


class OnlineTests(unittest.TestCase):
    def test_all200_online_parity_and_resume(self):
        tasks,_=load_and_audit(ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv",
                               ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_cache",200)
        for t,e in sorted(tasks.items()):
            with self.subTest(task=t):
                points={(step,s):e["personal"][(step,seed)] for s,seed in enumerate(e["seeds"]) for step in range(1,29)}
                expected=replay(list(range(8)),CausalObserver(points,range(8)))
                snapshots={};accesses=[]
                def advance(slot,latent,step,decode):
                    self.assertEqual(latent,step-1)
                    self.assertEqual(decode,expected["trace"][step-1]["observed"])
                    return step,(slot,step) if decode else None
                def observe(images):
                    result={}
                    for slot,(s,step) in images.items():
                        self.assertEqual(s,slot);accesses.append((step,slot));result[slot]=points[(step,slot)]
                        images[slot]=("rgb",slot,step)
                    return result
                def checkpoint(state):
                    if state["completed_step"] in (6,7,13,14,20,21,27,28):
                        snapshots[state["completed_step"]]=copy.deepcopy(state)
                actual=online_search(RULE,dict.fromkeys(range(8),0),advance,observe,checkpoint)
                self.assertEqual(actual["selected_slot"],expected["selected_seed"])
                self.assertEqual(actual["spent"],112);self.assertEqual(actual["reward_views"],expected["decode_images"])
                self.assertEqual(actual["scoring_rounds"],3)
                self.assertEqual(actual["selected_image"],("rgb",actual["selected_slot"],28))
                self.assertEqual(accesses,[(r["step"],s) for r in expected["trace"] if r["observed"] for s in r["alive_before"]])
                for a,b in zip(actual["trace"],expected["trace"]):
                    for key in ("step","alive_before","alive_after","observation_reason","removed_reasons","second_observation_step"):
                        self.assertEqual(a[key],b[key])
                for saved in snapshots.values():
                    resumed=online_search(RULE,{},advance,observe,lambda state:None,copy.deepcopy(saved))
                    self.assertEqual(actual,resumed)


if __name__=="__main__":unittest.main()
