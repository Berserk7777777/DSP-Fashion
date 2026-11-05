"""Evaluate frozen online item-greedy DualFashion GOR selections.

This is a post-selection evaluator.  It never loads candidate/pruning images,
reruns generation, or changes selections.  GT images are used only for CIS and
LPIPS after all selected image paths have been validated.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


SPECIAL_CATEGORIES = {"shoes", "pants", "sneakers", "boots", "earrings", "slippers", "sandals"}
PERSONAL_CLIP_WEIGHTS = Path(
    r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    r"\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", action="append", required=True, metavar="LABEL=TASK_ROOT",
                        help="Repeatable final-selection root, e.g. bon=.../output/.../bon")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, default=repo / "DualFashion-main")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_inputs(entries: list[str]) -> dict[str, Path]:
    result = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--selection must be LABEL=TASK_ROOT, got {entry!r}")
        label, raw_root = entry.split("=", 1)
        root = Path(raw_root).resolve()
        if not label or label in result or not root.is_dir():
            raise ValueError(f"invalid or duplicate selection input: {entry!r}")
        result[label] = root
    return result


def readable(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()


def load_tasks(label: str, root: Path, maximum: int) -> list[dict]:
    paths = sorted(root.glob("task*/task_summary.json"))[:maximum]
    if len(paths) != maximum:
        raise RuntimeError(f"{label}: expected {maximum} completed tasks under {root}, found {len(paths)}")
    tasks = []
    for path in paths:
        task = json.loads(path.read_text(encoding="utf-8"))
        items = sorted(task.get("items", []), key=lambda item: int(item["original_outfit_position"]))
        if [int(item["original_outfit_position"]) for item in items] != list(range(4)):
            raise ValueError(f"{label}: invalid original outfit order in {path}")
        for item in items:
            readable(Path(item["selected_image_path"]))
        task["items"] = items
        task["label"] = label
        tasks.append(task)
    return tasks


def category_prompt(category_id: int, categories: dict) -> str:
    name = str(categories.get(category_id, categories.get(str(category_id))))
    article = "a pair of" if any(value in name for value in SPECIAL_CATEGORIES) else "a"
    return f"A photo of {article} {name}, on white background"


def history_vector(history: dict, uid: int, category_id: int) -> np.ndarray:
    user = history.get(uid, history.get(str(uid)))
    value = user.get(category_id, user.get(str(category_id))) if isinstance(user, dict) else None
    if value is None:
        raise KeyError(f"missing Personal history embedding for uid={uid}, category={category_id}")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def gt_image_path(item_id: int, item_paths: np.ndarray, image_root: Path) -> Path:
    path = image_root / str(item_paths[item_id])
    readable(path)
    return path.resolve()


def batched(values: list, batch_size: int):
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def load_models(args: argparse.Namespace, device: torch.device):
    import lpips
    import open_clip
    from safetensors.torch import load_file

    if not args.personal_clip_weights.is_file():
        raise FileNotFoundError(f"missing local ViT-H/14 weights: {args.personal_clip_weights}")
    dualfashion = str(args.dualfashion_root.resolve())
    if dualfashion not in sys.path:
        sys.path.insert(0, dualfashion)
    from eval.compatibility_evaluator.compatibility_net import FashionEvaluator

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    missing, unexpected = clip.load_state_dict(load_file(str(args.personal_clip_weights), device="cpu"), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ViT-H/14 checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    evaluator_path = args.dualfashion_root / "eval" / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    if not evaluator_path.is_file():
        raise FileNotFoundError(evaluator_path)
    compatibility = FashionEvaluator(cnn_feat_dim=1024)
    compatibility.load_state_dict(torch.load(evaluator_path, map_location="cpu"))
    return (
        clip.to(device).eval(), preprocess, open_clip.get_tokenizer("ViT-H-14"),
        compatibility.to(device).eval(), lpips.LPIPS(net="vgg").to(device).eval(), evaluator_path,
    )


def load_clip_image(path: Path, preprocess, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess(image.convert("RGB")).to(device)


def load_lpips_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1).to(device)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_samples < 1:
        raise ValueError("--batch-size and --max-samples must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sources = parse_inputs(args.selection)
    task_sets = {label: load_tasks(label, root, args.max_samples) for label, root in sources.items()}
    ids_by_label = {
        label: {(int(task["task_index"]), int(task["uid"]), int(task["oid"])) for task in tasks}
        for label, tasks in task_sets.items()
    }
    first_ids = next(iter(ids_by_label.values()))
    if not first_ids or any(ids != first_ids for ids in ids_by_label.values()):
        raise ValueError("all frozen selections must contain exactly the same tasks")

    # Selections are now frozen and validated. GT is accessible only for CIS/LPIPS.
    item_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    for tasks in task_sets.values():
        for task in tasks:
            for item in task["items"]:
                item["gt_image_path"] = str(gt_image_path(int(item["item_id"]), item_paths, args.image_root))

    categories = np.load(args.data_root / "id_cate_dict.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    clip, preprocess, tokenizer, compatibility, lpips_model, evaluator_path = load_models(args, device)
    rows = []
    try:
        for label, tasks in task_sets.items():
            flat = [(task, item) for task in tasks for item in task["items"]]
            by_task: dict[int, list[dict]] = {int(task["task_index"]): [] for task in tasks}
            for chunk in tqdm(list(batched(flat, args.batch_size)), desc=f"{label}: CS/CIS/LPIPS/Personal", unit="batch"):
                generated = torch.stack([load_clip_image(Path(item["selected_image_path"]), preprocess, device) for _, item in chunk])
                ground_truth = torch.stack([load_clip_image(Path(item["gt_image_path"]), preprocess, device) for _, item in chunk])
                prompts = tokenizer([category_prompt(int(item["category_id"]), categories) for _, item in chunk]).to(device)
                with torch.inference_mode():
                    raw_generated = clip.encode_image(generated).float()
                    raw_ground_truth = clip.encode_image(ground_truth).float()
                    norm_generated = F.normalize(raw_generated, dim=-1)
                    norm_ground_truth = F.normalize(raw_ground_truth, dim=-1)
                    norm_text = F.normalize(clip.encode_text(prompts).float(), dim=-1)
                    lpips_scores = lpips_model(
                        torch.stack([load_lpips_image(Path(item["selected_image_path"]), device) for _, item in chunk]),
                        torch.stack([load_lpips_image(Path(item["gt_image_path"]), device) for _, item in chunk]),
                    ).reshape(-1)
                for offset, (task, item) in enumerate(chunk):
                    personal_history = F.normalize(
                        torch.as_tensor(history_vector(history, int(task["uid"]), int(item["category_id"])), device=device), dim=0
                    )
                    by_task[int(task["task_index"])].append({
                        "raw_feature": raw_generated[offset].detach().cpu(),
                        "cs": float(100.0 * (norm_generated[offset] * norm_text[offset]).sum().item()),
                        "cis": float(100.0 * (norm_generated[offset] * norm_ground_truth[offset]).sum().item()),
                        "lpips": float(lpips_scores[offset].item()),
                        "personal": float(100.0 * (norm_generated[offset] * personal_history).sum().item()),
                    })
            for task in tasks:
                item_metrics = by_task[int(task["task_index"])]
                if len(item_metrics) != 4:
                    raise RuntimeError(f"{label} task {task['task_index']}: expected four item metrics")
                outfit_features = torch.stack([value["raw_feature"] for value in item_metrics]).to(device)
                with torch.inference_mode():
                    compatibility_score = float(torch.sigmoid(compatibility(outfit_features.unsqueeze(0))).item())
                rows.append({
                    "label": label, "task_index": int(task["task_index"]), "uid": int(task["uid"]), "oid": int(task["oid"]),
                    "cs": float(np.mean([value["cs"] for value in item_metrics])),
                    "cis": float(np.mean([value["cis"] for value in item_metrics])),
                    "lpips": float(np.mean([value["lpips"] for value in item_metrics])),
                    "compatibility": compatibility_score,
                    "personal": float(np.mean([value["personal"] for value in item_metrics])),
                    "transformer_forward_calls": int(sum(int(item["transformer_forward_calls"]) for item in task["items"])),
                })
    finally:
        del clip, compatibility, lpips_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda row: (row["label"], row["task_index"]))
    write_csv(args.output_root / "per_task_metrics.csv", rows)
    summary = []
    for label in sources:
        values = [row for row in rows if row["label"] == label]
        for metric, direction in (("cs", "higher"), ("cis", "higher"), ("lpips", "lower"), ("compatibility", "higher"), ("personal", "higher")):
            data = np.asarray([row[metric] for row in values], dtype=np.float64)
            summary.append({
                "label": label, "metric": metric, "direction": direction, "mean": float(data.mean()),
                "std_sample": float(data.std(ddof=1)), "n_tasks": len(data),
                "transformer_forward_calls_per_outfit": int(np.mean([row["transformer_forward_calls"] for row in values])),
            })
    write_csv(args.output_root / "summary.csv", summary)
    (args.output_root / "protocol.json").write_text(json.dumps({
        "task": "iFashion GOR", "selection": "frozen online item-greedy final selections",
        "selection_inputs": {label: str(root) for label, root in sources.items()},
        "cs": "mean_i 100*cosine(OpenCLIP ViT-H/14(generated_i), target-category prompt_i)",
        "cis": "post-hoc mean_i 100*cosine(OpenCLIP ViT-H/14(generated_i), GT_i)",
        "lpips": "post-hoc mean_i LPIPS-VGG(generated_i[-1,1], GT_i[-1,1])",
        "compatibility": "sigmoid(FashionEvaluator([four generated OpenCLIP ViT-H/14 features]))",
        "personal": "mean_i 100*cosine(OpenCLIP ViT-H/14(generated_i), user_history(uid, category_i))",
        "gt_policy": "GT is loaded only after final selected images are validated; it never affects selection or pruning.",
        "compatibility_evaluator": str(evaluator_path.resolve()),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
