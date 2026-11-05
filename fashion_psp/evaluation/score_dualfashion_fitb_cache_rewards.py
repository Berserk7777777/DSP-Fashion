"""Score every cached FITB x0 candidate with the frozen Personal reward.

This is a batched, selection-only pass.  It intentionally does not open GT
images or calculate GT-Sim/GAA, so the output is suitable for offline PSP
schedule search without leaking GT information into pruning.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from tqdm import tqdm


TASK_RE = re.compile(r"^psp_offline_group(?P<group>\d+)_task(?P<task>\d+)_seed(?P<seed>\d+)$")
PERSONAL_CLIP_WEIGHTS = Path(
    r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reward", choices=("personal",), default="personal")
    parser.add_argument(
        "--append-output-csv", action="store_true",
        help="Append non-overlapping scored rows to an existing output CSV.",
    )
    parser.add_argument(
        "--steps", default="1-28",
        help="Comma-separated cached step numbers or an inclusive range such as 1-28.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS)
    return parser.parse_args()


def load_metadata(root: Path, max_samples: int) -> list[dict]:
    indexed = []
    for path in root.glob("psp_offline_group*_task*_seed*/metadata.json"):
        match = TASK_RE.match(path.parent.name)
        if match:
            indexed.append((int(match.group("task")), path))
    indexed.sort()
    if [task for task, _ in indexed[:max_samples]] != list(range(max_samples)):
        raise RuntimeError(f"cache must contain exactly task indices 0..{max_samples - 1}")
    result = []
    for _, path in indexed[:max_samples]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cache = payload["offline_candidate_cache"]
        seeds = [int(seed) for seed in payload["candidate_seeds"]]
        steps = [int(step) for step in cache["steps"]]
        if len(seeds) != 8 or steps != list(range(1, 29)) or len(cache["images"]) != 224:
            raise ValueError(f"invalid 8-seed, 28-step cache: {path}")
        result.append(payload)
    return result


def parse_steps(value: str) -> set[int]:
    text = value.strip()
    if "-" in text and "," not in text:
        start, end = (int(part) for part in text.split("-", 1))
        steps = set(range(start, end + 1))
    else:
        steps = {int(part.strip()) for part in text.split(",") if part.strip()}
    if not steps or any(step < 1 or step > 28 for step in steps):
        raise ValueError("--steps must contain values in [1, 28]")
    return steps


def load_image(path: Path, preprocess) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_samples < 1:
        raise ValueError("--batch-size and --max-samples must be positive")
    if not args.personal_clip_weights.is_file():
        raise FileNotFoundError(args.personal_clip_weights)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fitb = np.load(args.data_root / "processed" / "fitb_test.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    requested_steps = parse_steps(args.steps)
    metadata = load_metadata(args.cache_root, args.max_samples)

    contexts: dict[int, tuple[torch.Tensor, int, torch.Tensor]] = {}
    entries: list[dict] = []
    for payload in metadata:
        task, uid = int(payload["task_index"]), int(payload["uid"])
        incomplete = [int(value) for value in fitb["outfits"][task].tolist()]
        categories = [int(value) for value in fitb["category"][task].tolist()]
        blank = incomplete.index(0)
        context = torch.zeros((len(incomplete), 1024), dtype=torch.float32)
        for index, item_id in enumerate(incomplete):
            if item_id:
                context[index] = torch.as_tensor(np.array(item_features[item_id], copy=True), dtype=torch.float32)
        history_vector = F.normalize(torch.as_tensor(history[uid][categories[blank]], dtype=torch.float32), dim=0)
        contexts[task] = (context, blank, history_vector)
        for cached in payload["offline_candidate_cache"]["images"]:
            if int(cached["step"]) not in requested_steps:
                continue
            path = Path(cached["image_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append({
                "task_index": task, "uid": uid, "oid": int(payload["oid"]),
                "seed": int(cached["seed"]), "step": int(cached["step"]),
                "scheduler_timestep": float(cached["scheduler_timestep"]), "sigma": float(cached["sigma"]),
                "image_path": str(path.resolve()),
            })
    entries.sort(key=lambda row: (row["task_index"], row["step"], row["seed"]))

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    missing, unexpected = clip.load_state_dict(load_file(str(args.personal_clip_weights), device="cpu"), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ViT-H/14 checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    clip = clip.to(device).eval()
    # PNG decode/resize dominates this offline pass; use CPU parallelism while
    # the GPU consumes the preceding image batch.
    with ThreadPoolExecutor(max_workers=32) as pool, torch.inference_mode():
        for offset in tqdm(range(0, len(entries), args.batch_size), desc="FITB cached reward scoring", unit="batch"):
            batch = entries[offset : offset + args.batch_size]
            pixels = torch.stack(list(pool.map(lambda row: load_image(Path(row["image_path"]), preprocess), batch))).to(
                device, dtype=next(clip.parameters()).dtype
            )
            feature = clip.encode_image(pixels).float()
            history_batch = torch.stack([contexts[row["task_index"]][2] for row in batch]).to(device)
            personal = 100.0 * (F.normalize(feature, dim=-1) * history_batch).sum(dim=-1)
            for index, row in enumerate(batch):
                row["personal"] = float(personal[index].cpu())
            del pixels, feature, history_batch, personal
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    append = args.append_output_csv and args.output_csv.is_file() and args.output_csv.stat().st_size > 0
    with args.output_csv.open("a" if append else "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(entries[0]))
        if not append:
            writer.writeheader()
        writer.writerows(entries)
    print(json.dumps({"output_csv": str(args.output_csv), "tasks": args.max_samples,
                      "steps": sorted(requested_steps), "candidates": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
