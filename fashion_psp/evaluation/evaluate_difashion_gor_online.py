"""Evaluate frozen online DiFashion GOR winners, with item and outfit metrics.

CS, Personal and Compatibility reuse the shared GOR native evaluator. The
historical item-mean CIS/LPIPS are retained for comparison with prior tables.
Separate outfit CIS/LPIPS follow evaluate_gor.py's 2x2 merged-image protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from fashion_psp.evaluation import evaluate_difashion_gor_native_metrics as native
from fashion_psp.experiments.run_difashion_gor_six_seed import (
    BUDGET, SCHEDULE, build_tasks, rank_scores, seed_map, verify_result,
)

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("cs", "cis", "lpips", "compatibility", "personal", "cis_outfit", "lpips_outfit")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("D:/DiFashion/ifashion"))
    parser.add_argument("--evaluation-root", type=Path, default=ROOT / "DiFashion-main/Evaluation")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2, help="Four-item outfits per batch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def freeze_selections(args):
    run = native.read_json(args.run_root / "run_config.json")
    if (run.get("task") != "GOR" or run.get("native_inference_steps_argument") != 50
            or run.get("schedule") != [{"step": s, "keep": k} for s, k in SCHEDULE]
            or run.get("ground_truth_loaded_or_used") is not False):
        raise ValueError("Input is not the audited native-50 GOR online PSP6 run")
    config_hash = hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()
    data = np.load(args.data_root / "processed/new_fitb_test.npy", allow_pickle=True).item()
    all_tasks = build_tasks(data)
    indices = run["selected_task_indices"]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(all_tasks) for i in indices):
        raise ValueError("Invalid/duplicate selected task indices")
    pools = seed_map(len(all_tasks), run["seed_base"])
    selected, invalid, signatures = [], [], []
    for index in tqdm(indices, desc="Verify frozen GOR winners", unit="task"):
        task = all_tasks[index]
        folder = args.run_root / task["task_id"]
        try:
            summary = verify_result(folder, task, config_hash)
            if summary is None:
                raise FileNotFoundError("Missing task_summary.json")
            if summary["categories"] != task["category"].tolist() or summary["ground_truth_loaded_or_used"]:
                raise ValueError("Category/GT policy mismatch")
            if summary["initial_seeds"] != pools[index]:
                raise ValueError("Initial seeds disagree with run mapping")
            alive = pools[index]
            for node, (step, keep) in zip(summary["trace"], SCHEDULE):
                rows = node["candidates"]
                if node["seeds_before"] != alive or [r["seed"] for r in rows] != alive:
                    raise ValueError(f"Invalid candidate coverage at step {step}")
                for row in rows:
                    if (len(row["personal_items"]) != 4
                            or not all(math.isfinite(v) for v in row["personal_items"])
                            or abs(np.mean(row["personal_items"]) - row["personal"]) > 1e-4):
                        raise ValueError("Invalid saved item Personal scores")
                alive = rank_scores(rows, keep)
                if alive != node["seeds_after"]:
                    raise ValueError("Invalid survivor mapping")
            winner = next(r for r in summary["trace"][-1]["candidates"] if r["seed"] == alive[0])
            if summary["selected_seed"] != alive[0]:
                raise ValueError("Selected seed differs from final reward winner")
            paths, hashes = [], []
            for name in [f"item_{j}.png" for j in range(4)] + ["all.png"]:
                path = folder / "selected" / name
                source = folder / "step_51" / f"seed_{alive[0]}" / name
                if digest(path) != digest(source):
                    raise ValueError(f"Selected image differs from winner: {name}")
                hashes.append(digest(path))
                if name != "all.png":
                    paths.append(str(path.resolve()))
            selected.append({"task_index": index, "task_id": task["task_id"], "uid": task["uid"],
                             "oid": task["oid"], "categories": task["category"].tolist(),
                             "method": "psp6_online_25_75_100", "reward": "personal", "group_id": 0,
                             "seed": alive[0], "image_paths": paths, "image_sha256": hashes,
                             "online_personal": winner["personal"]})
            signatures.append({"task_id": task["task_id"], "seed": alive[0], "image_sha256": hashes,
                               "summary_sha256": digest(folder / "task_summary.json")})
        except Exception as exc:
            invalid.append({"task_id": task["task_id"], "error": repr(exc)})
    native.atomic_json(args.output_root / "input_integrity.json", {
        "complete": not invalid, "expected_tasks": len(indices), "valid_tasks": len(selected),
        "missing_or_invalid_tasks": invalid})
    if invalid:
        raise RuntimeError(f"{len(invalid)} invalid tasks; see input_integrity.json")
    return run, selected, signatures


def merged_image(paths):
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("GOR merge requires identical item image sizes")
    merged = Image.new("RGB", (width * 2, height * 2), (255, 255, 255))
    for index, image in enumerate(images):
        merged.paste(image, ((index % 2) * width, (index // 2) * height))
    return merged


def lpips_image(image, device):
    pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_pending(args, pending, evaluation_hash):
    # All winners/hashes have been frozen before the first GT read.
    gt = native.load_gt(pending, args.data_root)
    categories = np.load(args.data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed/test_history_clipembs.npy", allow_pickle=True).item()
    device, clip, preprocess, tokenizer, evaluator, lpips_model, _ = native.load_models(args)
    with torch.inference_mode():
        for start in tqdm(range(0, len(pending), args.batch_size), desc="Online GOR metrics", unit="batch"):
            batch = pending[start:start + args.batch_size]
            generated_paths = [p for row in batch for p in row["image_paths"]]
            ground_paths = [p for row in batch for p in gt[row["task_index"]]]
            raw = clip.encode_image(torch.stack([native.clip_tensor(p, preprocess, device) for p in generated_paths])).float()
            raw_gt = clip.encode_image(torch.stack([native.clip_tensor(p, preprocess, device) for p in ground_paths])).float()
            features, gt_features = F.normalize(raw, dim=-1), F.normalize(raw_gt, dim=-1)
            text = tokenizer([native.category_prompt(c, categories) for row in batch for c in row["categories"]]).to(device)
            text_features = F.normalize(clip.encode_text(text).float(), dim=-1)
            compatibility = torch.sigmoid(evaluator(raw.reshape(len(batch), 4, -1))).reshape(-1)
            for i, row in enumerate(batch):
                segment = slice(i * 4, (i + 1) * 4)
                personal_targets = torch.as_tensor(np.stack([
                    native.history_vector(history, row["uid"], c) for c in row["categories"]]), device=device)
                personal_targets = F.normalize(personal_targets.float(), dim=-1)
                item_distances = []
                for gen_path, gt_path in zip(row["image_paths"], gt[row["task_index"]]):
                    item_distances.append(float(lpips_model(native.lpips_tensor(gen_path, device).unsqueeze(0),
                                                           native.lpips_tensor(gt_path, device).unsqueeze(0)).item()))
                gen_outfit, gt_outfit = merged_image(row["image_paths"]), merged_image(gt[row["task_index"]])
                outfit_features = F.normalize(clip.encode_image(torch.stack([
                    preprocess(gen_outfit), preprocess(gt_outfit)]).to(device)).float(), dim=-1)
                outfit_distance = lpips_model(lpips_image(gen_outfit, device), lpips_image(gt_outfit, device)).item()
                metrics = {**row, "gt_image_paths": gt[row["task_index"]],
                           "cs": float((100 * (features[segment] * text_features[segment]).sum(-1)).mean().item()),
                           "cis": float((100 * (features[segment] * gt_features[segment]).sum(-1)).mean().item()),
                           "lpips": float(np.mean(item_distances)), "lpips_items": item_distances,
                           "compatibility": float(compatibility[i].item()),
                           "personal": float((100 * (features[segment] * personal_targets).sum(-1)).mean().item()),
                           "cis_outfit": float(100 * (outfit_features[0] * outfit_features[1]).sum().item()),
                           "lpips_outfit": float(outfit_distance)}
                if not all(math.isfinite(metrics[m]) for m in METRICS):
                    raise ValueError(f"Nonfinite metric for {row['task_id']}")
                native.atomic_json(args.output_root / "per_task" / f"{row['task_id']}.json",
                                   {"evaluation_sha256": evaluation_hash, "metrics": metrics})
    del clip, evaluator, lpips_model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not (args.resume or args.audit_only):
        raise FileExistsError("Output exists; use --resume or a fresh output root")
    run, rows, signatures = freeze_selections(args)
    checkpoint = args.evaluation_root / "compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth"
    contract = {"schema": 1, "task": "GOR", "method": "PSP@6 online 25/75/100", "reward": "personal",
                "group_id": 0, "task_count": len(rows), "run_root": str(args.run_root.resolve()),
                "run_config_sha256": digest(args.run_root / "run_config.json"),
                "selected_inputs_sha256": hashlib.sha256(json.dumps(signatures, sort_keys=True).encode()).hexdigest(),
                "data_root": str(args.data_root.resolve()), "evaluation_root": str(args.evaluation_root.resolve()),
                "sources": {str(p): digest(p) for p in [Path(__file__), Path(native.__file__),
                            args.evaluation_root / "evaluate_gor.py", args.evaluation_root / "eval_utils.py"]},
                "checkpoint_sha256": digest(checkpoint),
                "gt_source_sha256": digest(args.data_root / "test_grd.npy"),
                "history_source": str(args.data_root / "processed/test_history_clipembs.npy"),
                "precision": "FP32, no autocast", "device": args.device, "batch_size": args.batch_size,
                "gt_policy": "post-selection CIS/LPIPS only", "dataset_role": "exploratory test analysis",
                "cost": {"actual_online_forward_calls_per_task": BUDGET, "task_count": len(rows),
                         "offline_trajectory_collection": False}}
    config_path = args.output_root / "evaluation_config.json"
    if config_path.exists() and native.read_json(config_path) != contract:
        raise ValueError("Evaluation inputs/config changed; choose a new output root")
    native.atomic_json(config_path, contract)
    evaluation_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    pending = []
    for row in rows:
        path = args.output_root / "per_task" / f"{row['task_id']}.json"
        if path.is_file():
            payload = native.read_json(path)
            if (payload["evaluation_sha256"] != evaluation_hash
                    or any(payload["metrics"][k] != row[k] for k in ("task_id", "seed", "image_sha256"))
                    or not all(math.isfinite(payload["metrics"][k]) for k in METRICS)):
                raise ValueError(f"Saved metric identity/config mismatch: {path}")
        else:
            pending.append(row)
    torch.manual_seed(2026)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    if pending and not args.audit_only:
        evaluate_pending(args, pending, evaluation_hash)
    completed, missing = [], []
    for row in rows:
        path = args.output_root / "per_task" / f"{row['task_id']}.json"
        if path.is_file():
            completed.append(native.read_json(path)["metrics"])
        else:
            missing.append(row["task_id"])
    native.write_csv(args.output_root / "task_metrics.csv", completed)
    summary = []
    for metric in METRICS:
        values = [row[metric] for row in completed]
        summary.append({"method": "PSP@6 online 25/75/100", "reward": "personal", "group_id": 0,
                        "metric": metric, "direction": "lower" if metric.startswith("lpips") else "higher",
                        "mean": float(np.mean(values)) if values else None,
                        "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "valid_tasks": len(values), "missing_tasks": len(missing)})
    native.write_csv(args.output_root / "summary.csv", summary)
    native.atomic_json(args.output_root / "metric_contract.json", {
        "cs": "Native category prompt CS, 100*cosine, four-item mean; higher",
        "cis": "Historical shared-component CIS: four paired item CLIP cosines * 100, mean; higher",
        "lpips": "Historical shared-component LPIPS: four paired item VGG distances, mean; lower",
        "cis_outfit": "Original DiFashion evaluate_gor.py: 100*cosine(CLIP(2x2 generated), CLIP(2x2 GT)); higher",
        "lpips_outfit": "Original DiFashion evaluate_gor.py: VGG LPIPS on unresized 2x2 RGB [-1,1] outfits; lower",
        "compatibility": "sigmoid(DiFashion ifashion FashionEvaluator(raw four-item OpenCLIP features)); higher",
        "personal": "Mean of four 100*cosine(generated item, user/category history); higher"})
    report = {"complete": not missing, "expected_tasks": len(rows), "evaluated_tasks": len(completed),
              "missing_tasks": missing, "selected_images_match_terminal_winner": True,
              "gt_post_selection_only": True, "max_abs_online_vs_evaluated_personal": max(
                  (abs(r["personal"] - r["online_personal"]) for r in completed), default=None)}
    native.atomic_json(args.output_root / "integrity_report.json", report)
    print(json.dumps({**report, "summary": summary}, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
