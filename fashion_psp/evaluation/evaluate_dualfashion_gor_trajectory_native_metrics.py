"""Native metrics for frozen trajectory-level GOR Standard and Personal BoN.

The input is the result of ``evaluate_dualfashion_gor_offline.py``.  This
script consumes only already selected terminal image paths; it does not replay
offline pruning or inspect non-selected candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, required=True,
                        help=".../evaluation/bon/tasks/personal")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, default=repo / "DualFashion-main")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument(
        "--personal-clip-weights", type=Path,
        default=Path(
            r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
            r"\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"
        ),
    )
    return parser.parse_args()


def load_sources(root: Path, maximum: int) -> dict[str, list[dict]]:
    paths = sorted(root.glob("task_*.json"))[:maximum]
    if len(paths) != maximum:
        raise RuntimeError(f"expected {maximum} frozen task selections under {root}, found {len(paths)}")
    sources = {"trajectory_standard": [], "trajectory_bon_personal": []}
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        for label, key in (("trajectory_standard", "standard"), ("trajectory_bon_personal", "selected")):
            selected = record[key]
            images = [Path(value).resolve() for value in selected["image_paths"]]
            categories = [int(value) for value in record["ground_truth_category_ids"]]
            item_ids = [int(value) for value in record["ground_truth_item_ids"]]
            if len(images) != 4 or len(categories) != 4 or len(item_ids) != 4:
                raise ValueError(f"{path}: expected four canonical outfit items")
            sources[label].append({
                "label": label,
                "task_index": int(record["task_index"]), "uid": int(record["uid"]), "oid": int(record["oid"]),
                "image_paths": images, "category_ids": categories, "item_ids": item_ids,
                "selected_seed": int(selected["seed"]),
            })
    first = {(task["task_index"], task["uid"], task["oid"]) for task in sources["trajectory_standard"]}
    second = {(task["task_index"], task["uid"], task["oid"]) for task in sources["trajectory_bon_personal"]}
    if first != second:
        raise ValueError("standard and Personal BoN task coverage differs")
    return sources


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_samples < 1:
        raise ValueError("--batch-size and --max-samples must be positive")
    repo = Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from fashion_psp.evaluation.evaluate_dualfashion_gor_greedy_native_metrics import (
        batched, category_prompt, gt_image_path, history_vector, load_clip_image,
        load_lpips_image, load_models, write_csv as checked_write_csv,
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sources = load_sources(args.tasks_root, args.max_samples)
    # Validate every selected image before accessing GT images.
    from fashion_psp.evaluation.evaluate_dualfashion_gor_greedy_native_metrics import readable
    for tasks in sources.values():
        for task in tasks:
            for image_path in task["image_paths"]:
                readable(image_path)

    item_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    for tasks in sources.values():
        for task in tasks:
            task["gt_paths"] = [gt_image_path(item_id, item_paths, args.image_root) for item_id in task["item_ids"]]

    categories = np.load(args.data_root / "id_cate_dict.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    clip, preprocess, tokenizer, compatibility, lpips_model, evaluator_path = load_models(args, device)
    rows = []
    try:
        for label, tasks in sources.items():
            flat = [(task, position) for task in tasks for position in range(4)]
            by_task: dict[int, list[dict]] = {task["task_index"]: [] for task in tasks}
            for chunk in tqdm(list(batched(flat, args.batch_size)), desc=f"{label}: CS/CIS/LPIPS/Personal", unit="batch"):
                generated = torch.stack([
                    load_clip_image(task["image_paths"][position], preprocess, device) for task, position in chunk
                ])
                ground_truth = torch.stack([
                    load_clip_image(task["gt_paths"][position], preprocess, device) for task, position in chunk
                ])
                prompts = tokenizer([
                    category_prompt(task["category_ids"][position], categories) for task, position in chunk
                ]).to(device)
                with torch.inference_mode():
                    raw_generated = clip.encode_image(generated).float()
                    raw_ground_truth = clip.encode_image(ground_truth).float()
                    norm_generated = F.normalize(raw_generated, dim=-1)
                    norm_ground_truth = F.normalize(raw_ground_truth, dim=-1)
                    norm_text = F.normalize(clip.encode_text(prompts).float(), dim=-1)
                    lpips_scores = lpips_model(
                        torch.stack([load_lpips_image(task["image_paths"][position], device) for task, position in chunk]),
                        torch.stack([load_lpips_image(task["gt_paths"][position], device) for task, position in chunk]),
                    ).reshape(-1)
                for offset, (task, position) in enumerate(chunk):
                    personal_history = F.normalize(
                        torch.as_tensor(history_vector(history, task["uid"], task["category_ids"][position]), device=device), dim=0
                    )
                    by_task[task["task_index"]].append({
                        "raw_feature": raw_generated[offset].detach().cpu(),
                        "cs": float(100.0 * (norm_generated[offset] * norm_text[offset]).sum().item()),
                        "cis": float(100.0 * (norm_generated[offset] * norm_ground_truth[offset]).sum().item()),
                        "lpips": float(lpips_scores[offset].item()),
                        "personal": float(100.0 * (norm_generated[offset] * personal_history).sum().item()),
                    })
            for task in tasks:
                metrics = by_task[task["task_index"]]
                if len(metrics) != 4:
                    raise RuntimeError(f"task {task['task_index']}: expected four item metrics")
                raw_features = torch.stack([value["raw_feature"] for value in metrics]).to(device)
                with torch.inference_mode():
                    compatibility_score = float(torch.sigmoid(compatibility(raw_features.unsqueeze(0))).item())
                rows.append({
                    "label": label, "task_index": task["task_index"], "uid": task["uid"], "oid": task["oid"],
                    "cs": float(np.mean([value["cs"] for value in metrics])),
                    "cis": float(np.mean([value["cis"] for value in metrics])),
                    "lpips": float(np.mean([value["lpips"] for value in metrics])),
                    "compatibility": compatibility_score,
                    "personal": float(np.mean([value["personal"] for value in metrics])),
                })
    finally:
        del clip, compatibility, lpips_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (row["label"], row["task_index"]))
    checked_write_csv(args.output_root / "per_task_metrics.csv", rows)
    summary = []
    for label in sources:
        values = [row for row in rows if row["label"] == label]
        for metric, direction in (("cs", "higher"), ("cis", "higher"), ("lpips", "lower"), ("compatibility", "higher"), ("personal", "higher")):
            data = np.asarray([row[metric] for row in values], dtype=np.float64)
            summary.append({"label": label, "metric": metric, "direction": direction, "mean": float(data.mean()),
                            "std_sample": float(data.std(ddof=1)), "n_tasks": len(data)})
    write_csv(args.output_root / "summary.csv", summary)
    (args.output_root / "protocol.json").write_text(json.dumps({
        "task": "iFashion GOR", "selection": "frozen offline trajectory-level Standard and Personal BoN selections",
        "cs": "mean_i 100*cosine(OpenCLIP ViT-H/14(generated_i), target-category prompt_i)",
        "cis": "post-hoc mean_i 100*cosine(OpenCLIP ViT-H/14(generated_i), GT_i)",
        "lpips": "post-hoc mean_i LPIPS-VGG(generated_i[-1,1], GT_i[-1,1])",
        "compatibility": "sigmoid(FashionEvaluator([four raw generated OpenCLIP ViT-H/14 features]))",
        "personal": "mean_i 100*cosine(OpenCLIP ViT-H/14(generated_i), user_history(uid, category_i))",
        "gt_policy": "GT is loaded only after frozen selected images are validated; it never affects selection.",
        "compatibility_evaluator": str(evaluator_path.resolve()),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
