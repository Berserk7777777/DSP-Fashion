"""Synthetic-only partial decode flags, online parity, exact budget and resume."""
import copy
import random
import unittest

from fashion_psp.search.deferred_boundary_psp import PartialCausalObserver,RULE,replay
from fashion_psp.experiments.run_dualfashion_fitb_deferred_boundary import online_search


class OnlineTests(unittest.TestCase):
    def test_synthetic_partial_decode_and_resume(self):
        rng=random.Random(20261025);seen=set()
        for _ in range(20):
            points={(t,s):rng.gauss(0,1) for t in range(1,29) for s in range(8)}
            expected=replay(list(range(8)),PartialCausalObserver(points,range(8)))
            snapshots={};accesses=[]
            def advance(slot,latent,step,decode):
                self.assertEqual(latent,step-1)
                self.assertEqual(decode,slot in expected["trace"][step-1]["evaluated_seeds"])
                return step,(slot,step) if decode else None
            def observe(images):
                result={}
                for slot,(s,t) in images.items():
                    self.assertEqual(s,slot);accesses.append((t,slot));result[slot]=points[(t,slot)]
                    images[slot]=("rgb",slot,t)
                return result
            def checkpoint(state):
                if state["completed_step"] in (6,7,8,9,14,18,19,27,28):
                    snapshots[state["completed_step"]]=copy.deepcopy(state)
            actual=online_search(RULE,dict.fromkeys(range(8),0),advance,observe,checkpoint)
            seen.add(actual["deferred_branch"])
            self.assertEqual(actual["selected_slot"],expected["selected_seed"])
            self.assertEqual(actual["spent"],112);self.assertEqual(actual["reward_views"],expected["decode_images"])
            self.assertEqual(actual["selected_image"],("rgb",actual["selected_slot"],28))
            self.assertEqual(accesses,[(r["step"],s) for r in expected["trace"] for s in r["evaluated_seeds"]])
            for a,b in zip(actual["trace"],expected["trace"]):
                for k in ("step","alive_before","alive_after","evaluated_seeds","removed_reasons","second_observation_step"):
                    self.assertEqual(a[k],b[k])
                self.assertEqual(a["personal_scores"],{str(s):v for s,v in b["personal_scores"].items()})
            for snap in snapshots.values():
                resumed=online_search(RULE,{},advance,observe,lambda state:None,copy.deepcopy(snap))
                self.assertEqual(actual,resumed)
        self.assertEqual(seen,{False,True})

    def test_reject_full_live_reward_at_partial_step(self):
        first=[10,9,8,2,1,0,-1,-2]
        def advance(s,l,t,d):return t,(s,t) if d else None
        def observe(images):
            t=next(iter(images.values()))[1]
            return {s:float(first[s]) for s in (range(5) if t==8 else images)}
        with self.assertRaisesRegex(ValueError,"exactly requested"):
            online_search(RULE,dict.fromkeys(range(8),0),advance,observe,lambda state:None)


if __name__=="__main__":unittest.main()
