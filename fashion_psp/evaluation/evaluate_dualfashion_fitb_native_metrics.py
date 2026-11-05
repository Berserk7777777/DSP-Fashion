"""Evaluate frozen DualFashion FITB selections with DiFashion CS/CIS/LPIPS.

Selection payloads are immutable inputs.  This evaluator never scores or ranks
candidates.  CIS and LPIPS load the test ground truth only after every selected
image path has been validated, so they remain post-selection diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


SPECIAL_CATEGORIES = {"shoes", "pants", "sneakers", "boots", "earrings", "slippers", "sandals"}
PERSONAL_CLIP_WEIGHTS = Path(
    r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", action="append", required=True, metavar="LABEL=CHECKPOINT_DIR",
                        help="Repeatable frozen selected payload, e.g. compatibility/bon=.../0057501")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, default=repo / "DualFashion-main")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=None, help="Bounded smoke test only.")
    parser.add_argument("--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_inputs(entries: list[str]) -> dict[str, Path]:
    result = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"--selection must be LABEL=CHECKPOINT_DIR, got {entry!r}")
        label, raw_path = entry.split("=", 1)
        path = Path(raw_path)
        if not label or label in result or not (path / "gen.npy").is_file() or not (path / "grd.npy").is_file():
            raise ValueError(f"invalid or duplicate selection input: {entry!r}")
        result[label] = path.resolve()
    return result


def category_prompt(category: int, categories: dict) -> str:
    name = str(categories.get(category, categories.get(str(category))))
    article = "a pair of" if any(value in name for value in SPECIAL_CATEGORIES) else "a"
    return f"A photo of {article} {name}, on white background"


def history_vector(history: dict, uid: int, category: int) -> np.ndarray:
    user = history.get(uid, history.get(str(uid)))
    value = user.get(category, user.get(str(category))) if isinstance(user, dict) else None
    if value is None:
        raise KeyError(f"missing Personal history embedding for uid={uid}, category={category}")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def readable(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image.verify()


def load_selection(label: str, root: Path) -> list[dict]:
    generated = np.load(root / "gen.npy", allow_pickle=True).item()
    ground = np.load(root / "grd.npy", allow_pickle=True).item()
    rows = []
    for raw_uid, oid_map in generated.items():
        uid = int(raw_uid)
        for raw_oid, record in oid_map.items():
            oid = int(raw_oid)
            paths = list(record.get("image_paths", []))
            categories = np.asarray(record.get("cates", []), dtype=np.int64).reshape(-1)
            if len(paths) != 1 or len(categories) != 1:
                raise ValueError(f"{label} {uid}_{oid}: expected one selected image and category")
            image_path = Path(paths[0]).resolve()
            readable(image_path)
            try:
                gt_path = Path(ground[raw_uid][raw_oid]["image_paths"][0]).resolve()
            except KeyError:
                gt_path = Path(ground[uid][oid]["image_paths"][0]).resolve()
            # GT is deliberately not opened until selection paths are all validated.
            rows.append({"label": label, "uid": uid, "oid": oid, "image_path": str(image_path),
                         "gt_image_path": str(gt_path), "category": int(categories[0]),
                         "selected_seed": int(record.get("candidate_seeds", [-1])[0])})
    return rows


def load_models(args: argparse.Namespace, device: torch.device):
    import open_clip
    import lpips

    if not args.personal_clip_weights.is_file():
        raise FileNotFoundError(f"missing local ViT-H/14 weights: {args.personal_clip_weights}")
    dualfashion = str(args.dualfashion_root.resolve())
    if dualfashion not in sys.path:
        sys.path.insert(0, dualfashion)
    from eval.compatibility_evaluator.compatibility_net import FashionEvaluator
    from safetensors.torch import load_file

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    missing, unexpected = clip.load_state_dict(load_file(str(args.personal_clip_weights), device="cpu"), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ViT-H/14 checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    evaluator_path = args.dualfashion_root / "eval" / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(evaluator_path, map_location="cpu"))
    return clip.to(device).eval(), preprocess, open_clip.get_tokenizer("ViT-H-14"), evaluator.to(device).eval(), lpips.LPIPS(net="vgg").to(device).eval(), evaluator_path


def clip_tensor(path: str, preprocess, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess(image.convert("RGB")).to(device)


def lpips_tensor(path: str, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1).to(device)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sources = parse_inputs(args.selection)
    rows = [row for label, root in sources.items() for row in load_selection(label, root)]
    source_tasks = {label: {(row["uid"], row["oid"]) for row in rows if row["label"] == label} for label in sources}
    first_tasks = next(iter(source_tasks.values()))
    if not first_tasks or any(tasks != first_tasks for tasks in source_tasks.values()):
        raise ValueError("all selections must contain exactly the same FITB tasks")
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise ValueError("--max-tasks must be positive")
        allowed = set(sorted(first_tasks)[: args.max_tasks])
        rows = [row for row in rows if (row["uid"], row["oid"]) in allowed]
    # GT becomes accessible only after frozen selection consistency has passed.
    for row in rows:
        readable(Path(row["gt_image_path"]))
    categories = np.load(args.data_root / "id_cate_dict.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    fitb = np.load(args.data_root / "fitb_test_dict.npy", allow_pickle=True).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    clip, preprocess, tokenizer, evaluator, lpips_model, evaluator_path = load_models(args, device)
    evaluated = []
    try:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="DualFashion FITB native metrics", unit="image"):
            batch = rows[start : start + args.batch_size]
            generated = torch.stack([clip_tensor(row["image_path"], preprocess, device) for row in batch])
            ground = torch.stack([clip_tensor(row["gt_image_path"], preprocess, device) for row in batch])
            prompts = tokenizer([category_prompt(row["category"], categories) for row in batch]).to(device)
            with torch.inference_mode():
                raw_generated = clip.encode_image(generated).float()
                raw_ground = clip.encode_image(ground).float()
                normal_generated = F.normalize(raw_generated, dim=-1)
                normal_ground = F.normalize(raw_ground, dim=-1)
                normal_text = F.normalize(clip.encode_text(prompts).float(), dim=-1)
                outfits = []
                for offset, row in enumerate(batch):
                    item_ids = list(fitb[row["uid"]][row["oid"]])
                    blank = item_ids.index(0)
                    vector = np.asarray([item_features[item_id] if item_id else np.zeros(1024, dtype=np.float32) for item_id in item_ids])
                    vector[blank] = raw_generated[offset].cpu().numpy()
                    outfits.append(torch.as_tensor(vector, dtype=torch.float32))
                compatibility = torch.sigmoid(evaluator(torch.stack(outfits).to(device))).reshape(-1)
                distances = lpips_model(
                    torch.stack([lpips_tensor(row["image_path"], device) for row in batch]),
                    torch.stack([lpips_tensor(row["gt_image_path"], device) for row in batch]),
                ).reshape(-1)
            for offset, row in enumerate(batch):
                history_embedding = F.normalize(torch.as_tensor(history_vector(history, row["uid"], row["category"]), device=device), dim=0)
                evaluated.append({**row,
                    "cs": float(100.0 * (normal_generated[offset] * normal_text[offset]).sum().item()),
                    "cis": float(100.0 * (normal_generated[offset] * normal_ground[offset]).sum().item()),
                    "lpips": float(distances[offset].item()),
                    "compatibility": float(compatibility[offset].item()),
                    "personal": float(100.0 * (normal_generated[offset] * history_embedding).sum().item()),
                })
    finally:
        del clip, evaluator, lpips_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    evaluated.sort(key=lambda row: (row["label"], row["uid"], row["oid"]))
    write_csv(args.output_root / "per_task_metrics.csv", evaluated)
    summary = []
    for label in sources:
        selected = [row for row in evaluated if row["label"] == label]
        for metric, direction in (("cs", "higher"), ("cis", "higher"), ("lpips", "lower"), ("compatibility", "higher"), ("personal", "higher")):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary.append({"label": label, "metric": metric, "direction": direction, "mean": float(values.mean()),
                            "std_sample": float(values.std(ddof=1)), "n_tasks": len(values)})
    write_csv(args.output_root / "summary.csv", summary)
    (args.output_root / "protocol.json").write_text(json.dumps({
        "task": "iFashion FITB", "selection_inputs": {label: str(path) for label, path in sources.items()},
        "cs": "100*cosine(OpenCLIP ViT-H/14 generated image, target-category prompt)",
        "cis": "100*cosine(OpenCLIP ViT-H/14 generated image, GT target image)",
        "lpips": "LPIPS-VGG(generated[-1,1], GT target[-1,1])",
        "compatibility": "sigmoid(FashionEvaluator([fixed observed item features plus generated target feature]))",
        "personal": "100*cosine(OpenCLIP ViT-H/14 generated image, user test-history target-category embedding)",
        "gt_policy": "GT is loaded only after selected payload validation; it does not affect selection.",
        "compatibility_evaluator": str(evaluator_path.resolve()),
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
