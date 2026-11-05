"""Integrity checks for partial pools, tie-breaking, relocation, and metric scope."""
import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from fashion_psp.evaluation.evaluate_fitb_offline_pools import (
    FIELDS, Scorer, export_results, open_database, relocated_image,
)


class OfflinePoolTests(unittest.TestCase):
    def test_partial_pool_excluded_from_oracle_and_ties_use_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = open_database(root / "metrics.sqlite")
            coverage = []
            for task, count in ((0, 2), (1, 1)):
                for slot in range(2):
                    seed = 2026 + task * 10000 + slot
                    coverage.append(dict(model="difashion", method="psp", group_id=0, task_index=task,
                                         uid=9, oid=18+task, candidate_index=slot, seed=seed,
                                         expected_steps=1, available_steps=int(slot < count),
                                         cache_complete=int(slot < count), evaluated_steps=0,
                                         status="ready" if slot < count else "missing"))
                    if slot < count:
                        r = dict(model="difashion", method="psp", group_id=0, task_index=task,
                                 uid=9, oid=18+task, candidate_index=slot, seed=seed, step=50,
                                 terminal_step=50, personal=70., compatibility=.6, cs=23., cis=55., lpips=.3,
                                 source_signature="same")
                        for table in ("scores", "current_inputs"):
                            db.execute(f"INSERT INTO {table} VALUES ({','.join('?' for _ in FIELDS)})", tuple(r.get(k) for k in FIELDS))
            db.execute("CREATE TEMP VIEW current_scores AS SELECT * FROM scores")
            report = export_results(SimpleNamespace(output_root=root, expected_candidates=2, no_excel=True,
                                                    groups=[0], methods=["psp"], model="difashion"), db, coverage)
            self.assertFalse(report["complete"])
            with (root / "task_oracle.csv").open(encoding="utf-8-sig") as f:
                oracle = list(csv.DictReader(f))
            self.assertEqual(len(oracle), 1)
            self.assertEqual(int(oracle[0]["seed"]), 2026)
            with (root / "terminal_candidates.csv").open(encoding="utf-8-sig") as f:
                terminals = list(csv.DictReader(f))
            self.assertEqual([r["personal_rank"] for r in terminals], ["1", "2", ""])
            db.close()

    def test_windows_metadata_resolves_inside_relocated_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "psp_offline_group0_task0_seed2026"
            image = folder / "offline_candidates" / "9_18" / "step_28" / "seed_2026_x0.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"for path resolution only")
            raw = r"E:\old\psp_offline_group0_task0_seed2026\offline_candidates\9_18\step_28\seed_2026_x0.png"
            self.assertEqual(relocated_image(raw, folder), image.resolve())
            with self.assertRaises(FileNotFoundError):
                relocated_image(raw.replace("2026_x0", "2027_x0"), folder)

    def test_intermediate_rewards_never_read_ground_truth(self):
        import numpy as np
        import torch
        class Clip:
            def encode_image(self, x):
                return torch.ones((len(x), 1024))
        class Data:
            tasks = [dict(uid=9, category=1, outfit=[1, 0, 2, 3], blank_index=1)]
            def gt(self, task):
                raise AssertionError("An intermediate step accessed GT")
        scorer = Scorer.__new__(Scorer)
        scorer.torch, scorer.np, scorer.device = torch, np, torch.device("cpu")
        scorer.data, scorer.clip = Data(), Clip()
        scorer.item_features = np.ones((4, 1024), dtype=np.float32)
        scorer.history = {9: {1: np.ones(1024)}}
        scorer.evaluator = lambda outfits: torch.zeros(len(outfits))
        scorer.pixels = lambda paths: torch.zeros((len(paths), 3, 2, 2))
        rows = scorer.score([dict(task_index=0, step=49, terminal_step=50, image_path="not_opened")])
        self.assertAlmostEqual(rows[0]["personal"], 100.)
        self.assertEqual(rows[0]["compatibility"], .5)
        self.assertTrue(all(rows[0][key] is None for key in ("cs", "cis", "lpips", "gt_image_path")))

    def test_changed_image_signature_excludes_old_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = open_database(Path(tmp) / "metrics.sqlite")
            row = dict(method="psp", group_id=0, task_index=0, seed=2026, step=1, source_signature="old")
            db.execute(f"INSERT INTO scores VALUES ({','.join('?' for _ in FIELDS)})", tuple(row.get(k) for k in FIELDS))
            row["source_signature"] = "new"
            db.execute(f"INSERT INTO current_inputs VALUES ({','.join('?' for _ in FIELDS)})", tuple(row.get(k) for k in FIELDS))
            count = db.execute("SELECT COUNT(*) FROM scores s JOIN current_inputs c USING(method,group_id,task_index,seed,step,source_signature)").fetchone()[0]
            self.assertEqual(count, 0)
            db.close()


if __name__ == "__main__":
    unittest.main()
