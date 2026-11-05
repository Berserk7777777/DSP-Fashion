"""Post-hoc GT diagnostics for every terminal candidate in a FITB PSP pool.

This scorer never reads or modifies a reward-selection trace.  It evaluates the
eight cached step-28 x0 images for each task only after generation, so GT-Sim
and GAA cannot leak into PSP pruning.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


# This experiment uses already cached, frozen evaluators.  Do not make an
# accidental network request during a long resumable evaluation.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


TASK_RE = re.compile(r"^psp_offline_group(?P<group>\d+)_task(?P<task>\d+)_seed(?P<seed>\d+)$")
ATTRIBUTE_BANKS = {
    "color": ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"],
    "pattern": ["solid", "striped", "plaid", "floral", "polka dot", "geometric", "animal print", "quilted", "textured"],
    "material": ["leather", "faux leather", "suede", "canvas", "denim", "knit", "cotton", "wool", "synthetic material"],
    "style": ["casual", "formal", "sporty", "elegant", "vintage", "minimalist", "streetwear"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\\tmp\\ifashion"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--personal-clip-weights", type=Path,
        default=Path(r"C:\\Users\\Administrator\\.cache\\huggingface\\hub\\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K\\snapshots\\1c2b8495b28150b8a4922ee1c8edee224c284c0c\\open_clip_model.safetensors"),
        help="Frozen local OpenCLIP ViT-H/14 weights used by the Personal/GT-Sim protocol.",
    )
    parser.add_argument("--clip-weights", type=Path, default=Path(r"C:\\Users\\Administrator\\.cache\\clip\\ViT-L-14.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=2207)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("refusing to write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata_paths(cache_root: Path, max_samples: int) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    for candidate in cache_root.glob("psp_offline_group*_task*_seed*/metadata.json"):
        match = TASK_RE.match(candidate.parent.name)
        if match:
            indexed.append((int(match.group("task")), candidate))
    indexed.sort()
    if len(indexed) < max_samples:
        raise RuntimeError(f"expected {max_samples} cache tasks, found {len(indexed)} under {cache_root}")
    paths = [path for _, path in indexed[:max_samples]]
    actual = [int(TASK_RE.match(path.parent.name).group("task")) for path in paths]
    if actual != list(range(max_samples)):
        raise RuntimeError("cache task indices are not contiguous from zero")
    return paths


def terminal_rows(metadata: dict, gt_path: Path) -> list[dict]:
    task = int(metadata["task_index"])
    uid, oid = int(metadata["uid"]), int(metadata["oid"])
    seeds = [int(seed) for seed in metadata["candidate_seeds"]]
    cache = metadata["offline_candidate_cache"]
    lookup = {(int(row["step"]), int(row["seed"])): row for row in cache["images"]}
    expected = {(28, seed) for seed in seeds}
    if not expected.issubset(lookup):
        raise RuntimeError(f"task {task}: missing terminal candidates")
    rows = []
    for seed in seeds:
        image = Path(lookup[(28, seed)]["image_path"])
        if not image.is_file():
            raise FileNotFoundError(image)
        rows.append({
            "task_index": task, "uid": uid, "oid": oid, "seed": seed,
            "image_path": str(image.resolve()), "gt_path": str(gt_path.resolve()),
        })
    return rows


def batches(values: list[Path], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


@torch.inference_mode()
def encode(model, preprocess, paths: list[Path], device: torch.device, batch_size: int) -> np.ndarray:
    def load(path: Path):
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"))

    encoded = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for chunk in batches(paths, batch_size):
            pixels = torch.stack(list(pool.map(load, chunk))).to(device, dtype=next(model.parameters()).dtype)
            features = model.encode_image(pixels)
            encoded.append(F.normalize(features.float(), dim=-1).cpu().numpy())
    return np.concatenate(encoded, axis=0)


def build_task_gt(data_root: Path, max_samples: int) -> dict[int, Path]:
    fitb = np.load(data_root / "processed" / "fitb_test.npy", allow_pickle=True).item()
    ground_truth = np.load(data_root / "test_grd.npy", allow_pickle=True).item()
    item_paths = np.load(data_root / "all_item_image_paths.npy", allow_pickle=True)
    result: dict[int, Path] = {}
    for task, (oid, outfit) in enumerate(zip(fitb["oids"], fitb["outfits"])):
        if task >= max_samples:
            break
        outfit = outfit.tolist() if hasattr(outfit, "tolist") else list(outfit)
        blank = outfit.index(0)
        gt_item = int(ground_truth[int(oid)]["outfits"][blank])
        result[task] = data_root / "semantic_category" / str(item_paths[gt_item])
    if len(result) != max_samples:
        raise RuntimeError(f"FITB test file has only {len(result)} tasks")
    missing = [path for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing GT image, first: {missing[0]}")
    return result


def valid_rows(path: Path, required: set[str]) -> list[dict] | None:
    if not path.is_file():
        return None
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != 8:
        return None
    if any(not required.issubset(row) for row in rows):
        return None
    return rows


def load_gaa_model(weights: Path, device: torch.device):
    if not weights.is_file():
        raise FileNotFoundError(f"missing GAA CLIP checkpoint: {weights}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=None, device=device, force_quick_gelu=True
    )
    try:
        open_clip.load_checkpoint(model, str(weights), strict=False, weights_only=False, device=device)
    except TypeError:
        open_clip.load_checkpoint(model, str(weights), strict=False)
    except RuntimeError as exc:
        if "TorchScript archive" not in str(exc):
            raise
        archive = torch.jit.load(str(weights), map_location=device)
        model.load_state_dict(archive.state_dict(), strict=False)
        del archive
    if device.type == "cuda":
        model = model.half()
    return model.eval(), preprocess


def load_personal_clip(weights: Path, device: torch.device):
    if not weights.is_file():
        raise FileNotFoundError(f"missing local Personal/GT-Sim OpenCLIP weights: {weights}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    if weights.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weights), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"ViT-H/14 checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    else:
        open_clip.load_checkpoint(model, str(weights), strict=False)
    model = model.to(device)
    if device.type == "cuda":
        model = model.half()
    return model.eval(), preprocess


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_samples < 1:
        raise ValueError("--batch-size and --max-samples must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    artifacts = args.output_root / "tasks"
    paths = metadata_paths(args.cache_root, args.max_samples)
    gt_paths = build_task_gt(args.data_root, args.max_samples)

    # First pass: exact GT-Sim protocol from evaluate_best_results_gt_similarity.py.
    model_h, preprocess_h = load_personal_clip(args.personal_clip_weights, device)
    for metadata_path in tqdm(paths, desc="Post-hoc terminal GT-Sim", unit="task"):
        metadata = dict(read_json(metadata_path))
        task = int(TASK_RE.match(metadata_path.parent.name).group("task"))
        metadata["task_index"] = task
        destination = artifacts / f"task_{task:04d}.json"
        cached = valid_rows(destination, {"gt_sim"}) if args.resume else None
        if cached is not None:
            continue
        rows = terminal_rows(metadata, gt_paths[task])
        embedding = encode(model_h, preprocess_h, [Path(row["image_path"]) for row in rows] + [gt_paths[task]], device, args.batch_size)
        gt_embedding = embedding[-1]
        for index, row in enumerate(rows):
            row["gt_sim"] = float(embedding[index] @ gt_embedding)
        atomic_json(destination, rows)
    del model_h
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Second pass: exact frozen GAA protocol from evaluate_gt_attribute_alignment.py.
    model_l, preprocess_l = load_gaa_model(args.clip_weights, device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    text_banks = {}
    with torch.inference_mode():
        for group, values in ATTRIBUTE_BANKS.items():
            prompts = [f"a product photo of a fashion item that is {value}" for value in values]
            text_banks[group] = F.normalize(model_l.encode_text(tokenizer(prompts).to(device)).float(), dim=-1).cpu().numpy()
    logit_scale = float(model_l.logit_scale.exp().detach().float().cpu())
    for metadata_path in tqdm(paths, desc="Post-hoc terminal GAA", unit="task"):
        metadata = dict(read_json(metadata_path))
        task = int(TASK_RE.match(metadata_path.parent.name).group("task"))
        metadata["task_index"] = task
        destination = artifacts / f"task_{task:04d}.json"
        cached = valid_rows(destination, {"gt_sim", "gaa"}) if args.resume else None
        if cached is not None:
            continue
        rows = valid_rows(destination, {"gt_sim"})
        if rows is None:
            raise RuntimeError(f"task {task}: GT-Sim artifact is missing or malformed")
        embedding = encode(model_l, preprocess_l, [Path(row["image_path"]) for row in rows] + [gt_paths[task]], device, args.batch_size)
        gt_embedding = embedding[-1]
        for index, row in enumerate(rows):
            agreements = []
            for group, text_embedding in text_banks.items():
                candidate_prob = torch.softmax(torch.tensor(embedding[index] @ text_embedding.T * logit_scale), dim=0).numpy()
                gt_prob = torch.softmax(torch.tensor(gt_embedding @ text_embedding.T * logit_scale), dim=0).numpy()
                value = float(np.minimum(candidate_prob, gt_prob).sum())
                row[f"gaa_{group}"] = value
                agreements.append(value)
            row["gaa"] = float(np.mean(agreements))
        atomic_json(destination, rows)
    del model_l

    final_rows = []
    for task in range(args.max_samples):
        rows = valid_rows(artifacts / f"task_{task:04d}.json", {"gt_sim", "gaa"})
        if rows is None:
            raise RuntimeError(f"task {task}: terminal metric artifact is incomplete")
        final_rows.extend(rows)
    write_csv(args.output_root / "terminal_gt_candidate_metrics.csv", final_rows)
    atomic_json(args.output_root / "manifest.json", {
        "task_count": args.max_samples,
        "candidate_count": len(final_rows),
        "cache_root": str(args.cache_root.resolve()),
        "gt_sim": "OpenCLIP ViT-H/14 cosine to held-out FITB GT image; post-hoc only",
        "gaa": "OpenAI CLIP ViT-L/14 fixed attribute-distribution overlap; post-hoc only",
        "gt_usage": "GT is never used for reward scoring, seed selection, or pruning.",
    })
    print(json.dumps({"output": str(args.output_root), "candidate_count": len(final_rows)}, indent=2))


if __name__ == "__main__":
    main()
