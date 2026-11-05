#!/usr/bin/env python
"""Append the fixed Standard baseline and post-selection GT Sim to one reward run.

Standard is the first member (candidate_index=0) of DiFashion's independent
BoN@4 seed pool. Compatibility and Personal Sim are reused from the existing
candidate score table. Only GT Sim requires model work: it is the frozen
DualFashion OpenCLIP image-image cosine between that fixed prediction and the
ground-truth target item. Ground truth is never used to select a seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--dualfashion-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    a = args()
    if a.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    index_path = a.run_root / "selected_images_index.csv"
    score_path = a.run_root / "candidate_scores.csv"
    summary_path = a.run_root / "official_eval" / "group_summary.csv"
    condition_path = a.data_root / "processed" / "new_fitb_test.npy"
    for path in (index_path, score_path, summary_path, condition_path, a.data_root / "test_grd.npy", a.data_root / "new_all_item_image_paths.npy"):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_tasks = len(np.load(condition_path, allow_pickle=True).item()["uids"])
    standard = [row for row in csv.DictReader(index_path.open(encoding="utf-8")) if row["method"] == "standard"]
    if len(standard) != expected_tasks or len({row["task_id"] for row in standard}) != len(standard):
        raise ValueError(f"Expected one Standard row for each of {expected_tasks} FITB tasks, got {len(standard)}")
    score_by_task = {}
    for row in csv.DictReader(score_path.open(encoding="utf-8")):
        if row["method"] == "bon" and row["candidate_index"] == "0" and row["node"] == "final":
            score_by_task[row["task_id"]] = row
    if set(score_by_task) != {row["task_id"] for row in standard}:
        raise ValueError("Standard candidate score coverage does not match the Standard selection index")

    ground = np.load(a.data_root / "test_grd.npy", allow_pickle=True).item()
    item_paths = np.load(a.data_root / "new_all_item_image_paths.npy", allow_pickle=True)
    if str(a.dualfashion_root.resolve()) not in sys.path:
        sys.path.insert(0, str(a.dualfashion_root.resolve()))
    # The released ViT-H/14 checkpoint is already present in the local HF
    # cache. Avoid an unnecessary network HEAD request before using it.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import open_clip

    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b-s32b-b79K")
    clip = clip.to(device).eval()
    records = []
    for start in tqdm(range(0, len(standard), a.batch_size), desc="Standard GT Sim", unit="image"):
        batch = standard[start:start + a.batch_size]
        predictions, targets, metadata = [], [], []
        for row in batch:
            task_score = score_by_task[row["task_id"]]
            outfit = json.loads(task_score["outfit"])
            blank = int(task_score["blank_index"])
            oid = int(row["oid"])
            gt = ground.get(oid, ground.get(str(oid)))
            full = list(map(int, gt["outfits"]))
            categories = list(map(int, gt["category"]))
            if any(full[i] != int(outfit[i]) for i in range(4) if i != blank):
                raise ValueError(f"GT context mismatch: {row['task_id']}")
            if categories[blank] != int(task_score["target_category"]):
                raise ValueError(f"GT target category mismatch: {row['task_id']}")
            gt_path = a.image_root / str(item_paths[full[blank]])
            with Image.open(row["final_image_path"]) as image:
                predictions.append(preprocess(image.convert("RGB")))
            with Image.open(gt_path) as image:
                targets.append(preprocess(image.convert("RGB")))
            metadata.append((task_score, full[blank], gt_path))
        with torch.inference_mode():
            prediction_features = clip.encode_image(torch.stack(predictions).to(device)).float()
            target_features = clip.encode_image(torch.stack(targets).to(device)).float()
            similarities = 100.0 * F.cosine_similarity(F.normalize(prediction_features, dim=-1), F.normalize(target_features, dim=-1))
        for row, (score, gt_item_id, gt_path), similarity in zip(batch, metadata, similarities):
            value = float(similarity.item())
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite GT Sim: {row['task_id']}")
            records.append({**row, "compatibility_sigmoid": float(score["compatibility_sigmoid"]),
                            "personal_sim": float(score["personal_sim"]), "gt_sim": value,
                            "gt_item_id": gt_item_id, "gt_image_path": str(gt_path.resolve()),
                            "gt_usage": "post_selection_evaluation_only"})
    del clip
    if device.type == "cuda":
        torch.cuda.empty_cache()
    records.sort(key=lambda row: int(row["task_index"]))
    target = a.run_root / "official_eval" / "standard_gt_sim_task_metrics.csv"
    write_csv(target, records)
    metric_rows = []
    for metric, direction, unit in (("compatibility_sigmoid", "higher", "probability"), ("personal_sim", "higher", "100*cosine"), ("gt_sim", "higher", "100*cosine")):
        values = np.asarray([float(row[metric]) for row in records])
        metric_rows.append({"method": "standard", "metric": metric, "direction": direction, "unit": unit,
                            "mean": float(values.mean()), "std_sample": float(values.std(ddof=1)),
                            "valid_tasks": len(values), "missing_tasks": 0})
    prior = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    core_names = {"compatibility_sigmoid", "personal_sim", "gt_clip_image_similarity"}
    comparison = metric_rows + [{"method": row["method"], "metric": "gt_sim" if row["metric"] == "gt_clip_image_similarity" else row["metric"],
                                 "direction": row["direction"], "unit": row["unit"], "mean": row["mean"],
                                 "std_sample": row["std_sample"], "valid_tasks": row["valid_tasks"], "missing_tasks": row["missing_tasks"]}
                                for row in prior if row["metric"] in core_names]
    write_csv(a.run_root / "official_eval" / "standard_bon_psp_core_metrics.csv", comparison)
    print(json.dumps({"standard_tasks": len(records), "standard_task_metrics": str(target),
                      "comparison": str(a.run_root / 'official_eval' / 'standard_bon_psp_core_metrics.csv')}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
