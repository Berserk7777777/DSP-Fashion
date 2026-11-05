#!/usr/bin/env python
"""Evaluate frozen DiFashion GOR Personal selections with native DiFashion metrics.

The input selection must already be frozen with Personal as its only reward.
This script never ranks candidates and never reads GT until it has validated
the selected Standard/BoN/PSP image paths.  GT is used only for post-selection
CIS and LPIPS.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


METHODS = ("standard", "bon", "psp")
SPECIAL_CATES = {"shoes", "pants", "sneakers", "boots", "earrings", "slippers", "sandals"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-metrics", type=Path, required=True,
                        help="Frozen Personal-selection selected_metrics.csv from fashion_psp.")
    parser.add_argument("--selection-manifest", type=Path, required=True,
                        help="Frozen selection run_manifest.json; must declare reward=personal.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True,
                        help="DiFashion-main/Evaluation")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of four-item outfits per GPU batch.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Smoke test only.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def readable(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:  # pragma: no cover - diagnostic only
        raise ValueError(f"Unreadable image {path}: {exc!r}") from exc


def parse_selections(args: argparse.Namespace) -> list[dict[str, Any]]:
    manifest = read_json(args.selection_manifest)
    if manifest.get("task") != "GOR" or manifest.get("backbone") != "DiFashion" or manifest.get("reward") != "personal":
        raise ValueError("Selection manifest is not the frozen DiFashion GOR Personal run")
    with args.selected_metrics.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for source in raw:
        if source.get("reward") != "personal" or source.get("method") not in METHODS:
            raise ValueError("selected_metrics.csv has a non-Personal reward or unsupported method")
        paths = [Path(value) for value in json.loads(source["image_paths"])]
        categories = [int(value) for value in json.loads(source["categories"])]
        if len(paths) != 4 or len(categories) != 4:
            raise ValueError(f"{source.get('task_id')}: expected four selected images/categories")
        for path in paths:
            readable(path)
        rows.append({"task_index": int(source["task_index"]), "task_id": source["task_id"], "uid": int(source["uid"]),
                     "oid": int(source["oid"]), "method": source["method"], "categories": categories,
                     "image_paths": [str(path.resolve()) for path in paths], "seed": int(source["seed"])})
    expected = {row["task_index"] for row in rows}
    for index in expected:
        methods = {row["method"] for row in rows if row["task_index"] == index}
        if methods != set(METHODS):
            raise ValueError(f"task_index={index} does not have exactly Standard/BoN/PSP selections")
    rows.sort(key=lambda row: (row["task_index"], METHODS.index(row["method"])))
    if args.max_tasks is not None:
        allowed = sorted(expected)[:args.max_tasks]
        rows = [row for row in rows if row["task_index"] in set(allowed)]
    return rows


def category_prompt(category: int, mapping: dict[Any, Any]) -> str:
    name = str(mapping.get(category, mapping.get(str(category))))
    article = "a pair of" if any(special in name for special in SPECIAL_CATES) else "a"
    return f"A photo of {article} {name}, on white background"


def history_vector(history: dict[Any, Any], uid: int, category: int) -> np.ndarray:
    user = history.get(uid, history.get(str(uid)))
    value = user.get(category, user.get(str(category))) if isinstance(user, dict) else None
    if value is None:
        raise KeyError(f"Missing history vector for uid={uid}, category={category}")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def load_gt(rows: list[dict[str, Any]], data_root: Path) -> dict[int, list[str]]:
    ground = np.load(data_root / "test_grd.npy", allow_pickle=True).item()
    item_paths = np.load(data_root / "new_all_item_image_paths.npy", allow_pickle=True)
    image_root = data_root / "images"
    result: dict[int, list[str]] = {}
    for row in rows:
        if row["task_index"] in result:
            continue
        record = ground.get(row["oid"], ground.get(str(row["oid"])))
        if record is None:
            raise KeyError(f"Missing GT outfit for oid={row['oid']}")
        categories = [int(value) for value in np.asarray(record["category"]).reshape(-1)]
        if categories != row["categories"]:
            raise ValueError(f"GT category mismatch for {row['task_id']}")
        paths = [str((image_root / str(item_paths[int(item)])).resolve()) for item in record["outfits"]]
        if len(paths) != 4:
            raise ValueError(f"GT outfit has unexpected size for {row['task_id']}")
        for path in paths:
            readable(Path(path))
        result[row["task_index"]] = paths
    return result


def load_models(args: argparse.Namespace):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    root = str(args.evaluation_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import open_clip
    import lpips
    from compatibility_evaluator.compatibility_net import FashionEvaluator

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable: {args.device}")
    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b-s32b-b79K")
    checkpoint = args.evaluation_root / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return device, clip.to(device).eval(), preprocess, open_clip.get_tokenizer("ViT-H-14"), evaluator.to(device).eval(), lpips.LPIPS(net="vgg").to(device).eval(), checkpoint


def clip_tensor(path: str, preprocess, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess(image.convert("RGB")).to(device)


def lpips_tensor(path: str, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1).to(device)


def evaluate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gt_paths = load_gt(rows, args.data_root)  # GT enters only after selection validation.
    categories = np.load(args.data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    device, clip, preprocess, tokenizer, evaluator, lpips_model, checkpoint = load_models(args)
    output: list[dict[str, Any]] = []
    try:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="Native DiFashion GOR metrics", unit="outfit"):
            batch = rows[start:start + args.batch_size]
            generated_paths = [path for row in batch for path in row["image_paths"]]
            ground_paths = [path for row in batch for path in gt_paths[row["task_index"]]]
            generated = torch.stack([clip_tensor(path, preprocess, device) for path in generated_paths])
            ground = torch.stack([clip_tensor(path, preprocess, device) for path in ground_paths])
            prompts = tokenizer([category_prompt(category, categories) for row in batch for category in row["categories"]]).to(device)
            with torch.inference_mode():
                raw_generated = clip.encode_image(generated).float()
                raw_ground = clip.encode_image(ground).float()
                normalized_generated = F.normalize(raw_generated, dim=-1)
                normalized_ground = F.normalize(raw_ground, dim=-1)
                normalized_text = F.normalize(clip.encode_text(prompts).float(), dim=-1)
                compatibility = torch.sigmoid(evaluator(raw_generated.reshape(len(batch), 4, -1))).reshape(-1)
                lpips_values = lpips_model(
                    torch.stack([lpips_tensor(path, device) for path in generated_paths]),
                    torch.stack([lpips_tensor(path, device) for path in ground_paths]),
                ).reshape(len(batch), 4)
            for offset, row in enumerate(batch):
                segment = slice(offset * 4, (offset + 1) * 4)
                histories = torch.as_tensor(np.stack([history_vector(history, row["uid"], category) for category in row["categories"]]),
                                            dtype=torch.float32, device=device)
                histories = F.normalize(histories, dim=-1)
                personal = 100.0 * (normalized_generated[segment] * histories).sum(dim=-1)
                output.append({**row, "gt_image_paths": gt_paths[row["task_index"]],
                               "cs": float((100.0 * (normalized_generated[segment] * normalized_text[segment]).sum(dim=-1)).mean().item()),
                               "cis": float((100.0 * (normalized_generated[segment] * normalized_ground[segment]).sum(dim=-1)).mean().item()),
                               "lpips": float(lpips_values[offset].mean().item()),
                               "compatibility": float(compatibility[offset].item()),
                               "personal": float(personal.mean().item())})
    finally:
        del clip, evaluator, lpips_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return output


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    rows = parse_selections(args)
    task_count = len({row["task_index"] for row in rows})
    if len(rows) != task_count * len(METHODS):
        raise ValueError("Selection rows are incomplete")
    config = {"schema": 1, "selection_source": str(args.selected_metrics.resolve()),
              "selection_manifest": str(args.selection_manifest.resolve()), "reward": "personal", "task": "GOR",
              "task_count": task_count, "methods": list(METHODS), "metrics": ["cs", "cis", "lpips", "compatibility", "personal"],
              "gt_policy": "GT is loaded only after frozen Personal selections have been validated.",
              "evaluation_source_sha256": hashlib.sha256((args.evaluation_root / "eval_utils.py").read_bytes()).hexdigest()}
    config_path = args.output_root / "run_config.json"
    if config_path.exists():
        if read_json(config_path) != config:
            raise ValueError("Existing output has a different configuration")
        if not args.resume:
            raise FileExistsError("Output exists; pass --resume")
    else:
        atomic_json(config_path, config)
    metrics = evaluate(args, rows)
    metrics.sort(key=lambda row: (row["method"], row["task_index"]))
    write_csv(args.output_root / "task_metrics.csv", metrics)
    summary = []
    for method in METHODS:
        selected = [row for row in metrics if row["method"] == method]
        for metric, direction, unit in (("cs", "higher", "100*cosine"), ("cis", "higher", "100*cosine"),
                                        ("lpips", "lower", "LPIPS-VGG distance"), ("compatibility", "higher", "probability"),
                                        ("personal", "higher", "100*cosine")):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary.append({"reward": "personal", "method": method, "metric": metric, "direction": direction, "unit": unit,
                            "mean": float(values.mean()), "std_sample": float(values.std(ddof=1)) if len(values) > 1 else None,
                            "valid_tasks": len(values), "missing_tasks": 0})
    write_csv(args.output_root / "summary.csv", summary)
    atomic_json(args.output_root / "integrity_report.json", {"task_count": task_count, "selected_rows": len(rows),
                                                                "metric_rows": len(metrics), "complete": len(metrics) == task_count * 3})


if __name__ == "__main__":
    main()
