"""Native DiFashion metrics for frozen online FITB selections; no reranking."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "DiFashion-main/tools"))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import evaluate_fitb_offline_native as native
from collect_ifashion_fitb_candidates import task_rows
from fashion_psp.experiments.run_difashion_fitb_six_seed import verify_result

METRICS = ("cs", "cis", "lpips", "compatibility", "personal")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("D:/DiFashion/ifashion"))
    p.add_argument("--evaluation-root", type=Path, default=ROOT / "DiFashion-main/Evaluation")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    run = native.read_json(args.run_root / "run_config.json")
    config_hash = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    conditions = np.load(args.data_root / "processed/new_fitb_test.npy", allow_pickle=True).item()
    tasks = task_rows(conditions)
    selected, invalid, source_signatures = [], [], []
    seeds_used = set()
    for task in tqdm(tasks, desc="Verify frozen online selections", unit="task"):
        folder = args.run_root / task["task_id"]
        try:
            summary = verify_result(folder, task, config_hash)
            if summary is None:
                raise FileNotFoundError("Missing task_summary.json")
            if len(summary["initial_seeds"]) != 6 or set(summary["initial_seeds"]) & seeds_used:
                raise ValueError("Invalid or overlapping initial seed pool")
            seeds_used.update(summary["initial_seeds"])
            if summary["selected_seed"] not in summary["initial_seeds"]:
                raise ValueError("Selected seed outside the original seed pool")
            blank = int(np.flatnonzero(task["outfit"] == 0)[0])
            image_path = folder / "selected.png"
            online_score = next(r["personal"] for r in summary["trace"][-1]["candidates"]
                                if r["seed"] == summary["selected_seed"])
            selected.append({**task, "outfit": task["outfit"].tolist(),
                             "category": task["category"].tolist(), "input_ids": None,
                             "method": "psp6_online_25_75_100", "reward": "personal", "group_id": 0,
                             "blank_index": blank, "target_category": int(task["category"][blank]),
                             "seed": summary["selected_seed"], "image_path": str(image_path.resolve()),
                             "selected_image_sha256": digest(image_path), "online_personal": online_score})
            source_signatures.append({"task_id": task["task_id"], "selected_seed": summary["selected_seed"],
                                      "image_sha256": digest(image_path),
                                      "summary_sha256": digest(folder / "task_summary.json")})
        except Exception as exc:
            invalid.append({"task_id": task["task_id"], "error": repr(exc)})
    args.output_root.mkdir(parents=True, exist_ok=True)
    native.atomic_json(args.output_root / "input_integrity.json", {
        "complete": not invalid, "expected_tasks": len(tasks), "valid_tasks": len(selected),
        "missing_or_invalid_tasks": invalid,
    })
    if invalid:
        raise RuntimeError(f"{len(invalid)} invalid/missing online tasks; see input_integrity.json")
    contract = {
        "schema": 1, "run_root": str(args.run_root.resolve()), "run_config_sha256": digest(args.run_root / "run_config.json"),
        "selection_sha256": hashlib.sha256(json.dumps(source_signatures, sort_keys=True).encode()).hexdigest(),
        "data_root": str(args.data_root.resolve()), "evaluation_root": str(args.evaluation_root.resolve()),
        "task_count": len(tasks), "device": args.device, "batch_size": args.batch_size,
        "native_source_sha256": digest(Path(native.__file__)), "script_sha256": digest(Path(__file__)),
        "precision": "FP32 evaluator; no autocast", "gt_usage": "post-selection CIS and LPIPS only",
        "method": "online PSP6, 13:6->4,39:4->2,51:2->1; frozen Personal reward", "group_id": 0,
        "dataset_role": "exploratory test-set analysis",
    }
    config_path = args.output_root / "evaluation_config.json"
    if config_path.exists():
        if native.read_json(config_path) != contract:
            raise ValueError("Evaluation configuration/input hash mismatch; choose a fresh output directory")
        if not args.resume and not args.audit_only:
            raise FileExistsError("Evaluation output exists; use --resume")
    else:
        native.atomic_json(config_path, contract)
    evaluation_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    # All selections and hashes are frozen above, before the first GT read.
    gt = native.gt_records(args, selected)
    completed, pending = [], []
    for row in selected:
        saved = args.output_root / "per_task" / f"{row['task_id']}.json"
        if saved.exists():
            payload = native.read_json(saved)
            if payload["evaluation_sha256"] != evaluation_hash:
                raise ValueError(f"Saved evaluation hash mismatch: {saved}")
            metrics = payload["metrics"]
            for key in ("task_id", "seed", "selected_image_sha256", "image_path"):
                if metrics[key] != row[key]:
                    raise ValueError(f"Saved selected-image identity mismatch: {saved}, {key}")
            if not all(math.isfinite(metrics[k]) for k in METRICS):
                raise ValueError(f"NaN/Inf in saved metrics: {saved}")
            completed.append(metrics)
        else:
            pending.append(row)
    if pending and not args.audit_only:
        history = np.load(args.data_root / "processed/test_history_clipembs.npy", allow_pickle=True).item()
        item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
        categories = np.load(args.data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
        torch.manual_seed(2026)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
        device, clip, preprocess, tokenizer, evaluator, _ = native.load_models(args)
        import lpips
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
        with torch.inference_mode():
            for start in tqdm(range(0, len(pending), args.batch_size), desc="Native online FITB metrics", unit="batch"):
                batch = pending[start:start + args.batch_size]
                raw_features = clip.encode_image(native.load_image_batch(batch, preprocess, device)).float()
                raw_gt = clip.encode_image(native.load_image_batch(
                    [{"image_path": gt[r["task_id"]]["gt_image_path"]} for r in batch], preprocess, device)).float()
                features, gt_features = F.normalize(raw_features, dim=-1), F.normalize(raw_gt, dim=-1)
                text = tokenizer([native.category_prompt(r["target_category"], categories) for r in batch]).to(device)
                text_features = F.normalize(clip.encode_text(text).float(), dim=-1)
                contexts = native.load_contexts(batch, item_features, device)
                for index, row in enumerate(batch):
                    contexts[index, row["blank_index"], :] = raw_features[index]
                compatibility = torch.sigmoid(evaluator(contexts).float()).reshape(-1)
                distances = lpips_model(torch.stack([native.pil_lpips_tensor(r["image_path"], device) for r in batch]),
                                       torch.stack([native.pil_lpips_tensor(gt[r["task_id"]]["gt_image_path"], device)
                                                    for r in batch])).reshape(-1)
                for index, row in enumerate(batch):
                    hist = native.history_embedding(history, row["uid"], row["target_category"])
                    if hist is None or hist.size != features.shape[1]:
                        raise ValueError(f"Missing compatible Personal history: {row['task_id']}")
                    history_tensor = F.normalize(torch.as_tensor(hist, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
                    metrics = {k: v for k, v in row.items() if k != "input_ids"}
                    metrics.update({**gt[row["task_id"]],
                        "cs": float(100 * F.cosine_similarity(features[index:index+1], text_features[index:index+1]).item()),
                        "cis": float(100 * F.cosine_similarity(features[index:index+1], gt_features[index:index+1]).item()),
                        "lpips": float(distances[index].item()), "compatibility": float(compatibility[index].item()),
                        "personal": float(100 * F.cosine_similarity(features[index:index+1], history_tensor).item())})
                    if not all(math.isfinite(metrics[k]) for k in METRICS):
                        raise ValueError(f"NaN/Inf metrics: {row['task_id']}")
                    native.atomic_json(args.output_root / "per_task" / f"{row['task_id']}.json",
                                       {"evaluation_sha256": evaluation_hash, "metrics": metrics})
                    completed.append(metrics)
        del clip, evaluator, lpips_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    completed.sort(key=lambda row: row["task_index"])
    if len({r["task_id"] for r in completed}) != len(completed):
        raise ValueError("Duplicate evaluated task IDs")
    remaining = sorted({r["task_id"] for r in selected} - {r["task_id"] for r in completed})
    native.atomic_csv(args.output_root / "task_metrics.csv", completed)
    summary = []
    for metric in METRICS:
        values = [r[metric] for r in completed]
        summary.append({"method": "PSP@6 online 25/75/100", "reward": "personal", "group_id": 0,
                        "metric": metric, "direction": "lower" if metric == "lpips" else "higher",
                        "mean": float(np.mean(values)) if values else None,
                        "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "valid_tasks": len(values), "missing_tasks": len(remaining)})
    native.atomic_csv(args.output_root / "summary.csv", summary)
    native.atomic_json(args.output_root / "integrity_report.json", {
        "complete": not remaining, "expected_tasks": len(tasks), "evaluated_tasks": len(completed),
        "missing_tasks": remaining, "gt_post_selection_only": True,
        "max_abs_cpu_online_vs_gpu_evaluated_personal": max(
            (abs(r["personal"] - r["online_personal"]) for r in completed), default=None)})
    native.atomic_json(args.output_root / "metric_contract.json", {
        "CS": "100*cosine(OpenCLIP ViT-H/14 image, native category prompt); higher",
        "CIS": "100*cosine(OpenCLIP ViT-H/14 image, GT image); higher",
        "LPIPS": "LPIPS(net=vgg), native RGB [-1,1] inputs without resize; lower",
        "Compatibility": "sigmoid(DiFashion FashionEvaluator(fixed context raw CLIP features + generated raw CLIP feature)); higher",
        "Personal": "100*cosine(OpenCLIP ViT-H/14 generated image, frozen test user/category history); higher",
        "compatibility_checkpoint": str(args.evaluation_root / "compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth"),
        "selection": "frozen online winner; GT never used for reward or selection",
    })
    print(json.dumps({"complete": not remaining, "evaluated_tasks": len(completed), "summary": summary}, indent=2))
    return 0 if not remaining else 2


if __name__ == "__main__":
    raise SystemExit(main())
