"""Post-selection native metrics for immutable state-risk-DP selections.

Reuses the established ViT-H/14, FashionEvaluator and LPIPS-VGG definitions.
Deduplicates identical images across methods; resumable cache requires matching
image and GT bytes, plus a frozen evaluator signature. No metrics choose seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from fashion_psp.evaluation.analyze_dualfashion_training_free_psp import write_csv, write_json
from fashion_psp.evaluation.evaluate_dualfashion_fitb_native_metrics import (
    PERSONAL_CLIP_WEIGHTS, clip_tensor, history_vector, load_models, lpips_tensor,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def file_signature(path):
    stat = path.stat()
    return {"path":str(path.resolve()), "size":stat.st_size, "mtime_ns":stat.st_mtime_ns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = ROOT / "DualFashion-main/output/ifashion_fitb_state_risk_dp"
    parser.add_argument("--selection-csv", type=Path, default=default / "frozen_selections.csv")
    parser.add_argument("--output-dir", type=Path, default=default / "native_metrics")
    parser.add_argument("--dualfashion-root", type=Path, default=ROOT / "DualFashion-main")
    parser.add_argument("--data-root", type=Path, default=Path("C:/tmp/ifashion"))
    parser.add_argument("--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    selections = read_rows(args.selection_csv)
    frozen_hash = sha(args.selection_csv)
    protocol_path = args.selection_csv.parent / "selection_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["selections_sha256"] != frozen_hash:
        raise ValueError("Selection file changed after freezing")
    tasks_by_label = {}
    for row in selections:
        for field in ("task_index", "uid", "oid", "seed", "particle_steps", "reward_views"):
            row[field] = int(row[field])
        tasks_by_label.setdefault(row["label"], set()).add((row["uid"], row["oid"]))
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    if any(value != next(iter(tasks_by_label.values())) for value in tasks_by_label.values()):
        raise ValueError("Method task sets differ")
    if any(len(value) != 200 for value in tasks_by_label.values()):
        raise ValueError("Expected exactly200 tasks per method")
    # GT and all additional metrics become accessible only AFTER frozen
    # selections have been checked. The policy never imports this evaluator.
    ground = np.load(args.dualfashion_root / "output/ifashion_fitb_personal/psp/psp_group0_personal/0057501/grd.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed/test_history_clipembs.npy", allow_pickle=True).item()
    fitb = np.load(args.data_root / "fitb_test_dict.npy", allow_pickle=True).item()
    features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    unique = {}
    hashes = {}
    for row in selections:
        key = (row["uid"], row["oid"], row["seed"])
        gt = ground[row["uid"]][row["oid"]]
        gt_path = str(Path(gt["image_paths"][0]).resolve())
        path = str(Path(row["image_path"]).resolve())
        for candidate in (path, gt_path):
            if candidate not in hashes:
                hashes[candidate] = sha(Path(candidate))
        identifier = f"{key[0]}:{key[1]}:{key[2]}:{hashes[path]}:{hashes[gt_path]}"
        row["cache_key"] = identifier
        unique[identifier] = {"cache_key":identifier, "task_index":row["task_index"],
                              "uid":key[0], "oid":key[1], "seed":key[2], "image_path":path,
                              "gt_image_path":gt_path, "category":int(np.asarray(gt["cates"]).reshape(-1)[0]),
                              "image_sha256":hashes[path], "gt_sha256":hashes[gt_path]}
    signature = {"clip":file_signature(args.personal_clip_weights),
                 "compatibility":file_signature(args.dualfashion_root / "eval/compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth"),
                 "personal_history":file_signature(args.data_root / "processed/test_history_clipembs.npy"),
                 "context_features":file_signature(args.data_root / "cnn_features_clip.npy"),
                 "definitions":"native-ViT-H14-float32-personal-CIS-compatibility-LPIPS-VGG-v1"}
    signature_path = args.output_dir / "evaluator_signature.json"
    cache_path = args.output_dir / "candidate_native_metrics.csv"
    cached = {}
    if args.resume and cache_path.is_file():
        if not signature_path.is_file() or json.loads(signature_path.read_text(encoding="utf-8")) != signature:
            raise ValueError("Evaluator definition or checkpoint changed; use a fresh output directory")
        cached = {r["cache_key"]:r for r in read_rows(cache_path) if r["cache_key"] in unique}
    write_json(signature_path, signature)
    pending = [r for key, r in unique.items() if key not in cached]
    print(f"Frozen {len(selections)} selections; {len(unique)} unique images, {len(pending)} pending.", flush=True)
    if pending:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        clip, preprocess, _, evaluator, lpips_model, _ = load_models(args, device)
        gt_features = {}
        try:
            gt_paths = sorted({r["gt_image_path"] for r in pending})
            with torch.inference_mode():
                for start in tqdm(range(0, len(gt_paths), args.batch_size), desc="Frozen GT embeddings", unit="batch"):
                    chunk = gt_paths[start:start + args.batch_size]
                    pixels = torch.stack([clip_tensor(path, preprocess, device) for path in chunk])
                    embedded = F.normalize(clip.encode_image(pixels).float(), dim=-1)
                    for offset, path in enumerate(chunk):
                        gt_features[path] = embedded[offset].cpu()
            for start in tqdm(range(0, len(pending), args.batch_size), desc="Personal / Compatibility / CIS / LPIPS", unit="batch"):
                batch = pending[start:start + args.batch_size]
                generated = torch.stack([clip_tensor(r["image_path"], preprocess, device) for r in batch])
                with torch.inference_mode():
                    raw = clip.encode_image(generated).float()
                    normalized = F.normalize(raw, dim=-1)
                    outfits = []
                    for i, row in enumerate(batch):
                        item_ids = list(fitb[row["uid"]][row["oid"]])
                        blank = item_ids.index(0)
                        vectors = np.asarray([features[item] if item else np.zeros(1024, np.float32) for item in item_ids])
                        vectors[blank] = raw[i].cpu().numpy()
                        outfits.append(torch.as_tensor(vectors, dtype=torch.float32))
                    compatibility = torch.sigmoid(evaluator(torch.stack(outfits).to(device))).reshape(-1)
                    distances = lpips_model(torch.stack([lpips_tensor(r["image_path"], device) for r in batch]),
                                            torch.stack([lpips_tensor(r["gt_image_path"], device) for r in batch])).reshape(-1)
                    for i, row in enumerate(batch):
                        user = F.normalize(torch.as_tensor(history_vector(history, row["uid"], row["category"]), device=device), dim=0)
                        gt_vector = gt_features[row["gt_image_path"]].to(device)
                        cached[row["cache_key"]] = {**row,
                            "personal":float(100 * (normalized[i] * user).sum().item()),
                            "compatibility":float(compatibility[i].item()),
                            "cis":float(100 * (normalized[i] * gt_vector).sum().item()),
                            "lpips":float(distances[i].item())}
                if (start // args.batch_size + 1) % 25 == 0:
                    write_csv(cache_path, list(cached.values()))
            write_csv(cache_path, list(cached.values()))
        finally:
            del clip, evaluator, lpips_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    per_task = []
    for selection in selections:
        metrics = cached[selection["cache_key"]]
        per_task.append({**selection, **{key:float(metrics[key]) for key in ("personal", "compatibility", "cis", "lpips")}})
    write_csv(args.output_dir / "per_task_metrics.csv", per_task)
    summary = []
    for label in sorted(tasks_by_label):
        rows = [r for r in per_task if r["label"] == label]
        result = {"label":label, "tasks":len(rows)}
        for metric in ("personal", "compatibility", "cis", "lpips", "selection_personal", "particle_steps", "reward_views"):
            result[metric] = float(np.mean([float(r[metric]) for r in rows]))
        summary.append(result)
    write_csv(args.output_dir / "summary.csv", summary)
    paired = []
    baseline = {r["task_index"]:r for r in per_task if r["label"] == "default_psp8"}
    users = np.unique([r["uid"] for r in baseline.values()])
    rng = np.random.default_rng(729)
    indices = [np.concatenate([np.array([task for task, r in baseline.items() if r["uid"] == user])
                              for user in rng.choice(users, len(users), replace=True)]) for _ in range(1000)]
    for label in sorted(tasks_by_label):
        rows = {r["task_index"]:r for r in per_task if r["label"] == label}
        for metric in ("personal", "compatibility", "cis", "lpips"):
            delta = np.array([rows[t][metric] - baseline[t][metric] for t in range(200)])
            means = np.array([delta[ids].mean() for ids in indices])
            paired.append({"label":label, "metric":metric, "mean_delta_default":float(delta.mean()),
                           "cluster_bootstrap_p025":float(np.quantile(means, 0.025)),
                           "cluster_bootstrap_p975":float(np.quantile(means, 0.975)),
                           "selection_bias_corrected":False})
    write_csv(args.output_dir / "paired_deltas.csv", paired)
    write_json(args.output_dir / "protocol.json", {"selection_sha256":frozen_hash,
               "selection_file":str(args.selection_csv.resolve()), "gt_policy":"GT accessed after immutable selection validation",
               "personal":"100*cosine(OpenCLIP ViT-H14 generated image, category-history embedding)",
               "compatibility":"sigmoid(FashionEvaluator(fixed context plus generated target raw CLIP feature))",
               "cis":"100*cosine(OpenCLIP ViT-H14 generated image, GT image)", "lpips":"LPIPS-VGG on original RGB pixels in [-1,1]",
               "warning":"Exploratory200 tasks; clustered intervals do not correct repeated research or posthoc policy selection",
               "unique_images":len(unique), "evaluated_labels":sorted(tasks_by_label)})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
