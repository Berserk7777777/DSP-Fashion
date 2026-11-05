"""Offline timestep analysis for the DualFashion iFashion FITB x0 cache.

Selection is strictly Personal Sim based.  GT-Sim and GAA are computed only
after the reward-based selections have been frozen to disk.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


TASK_RE = re.compile(r"^psp_offline_group(?P<group>\d+)_task(?P<task>\d+)_seed(?P<seed>\d+)$")
ATTRIBUTE_BANKS = {
    "color": ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"],
    "pattern": ["solid", "striped", "plaid", "floral", "polka dot", "geometric", "animal print", "quilted", "textured"],
    "material": ["leather", "faux leather", "suede", "canvas", "denim", "knit", "cotton", "wool", "synthetic material"],
    "style": ["casual", "formal", "sporty", "elegant", "vintage", "minimalist", "streetwear"],
}
PERSONAL_CLIP_WEIGHTS = Path(
    r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS,
        help="Local OpenCLIP ViT-H/14 safetensors checkpoint used for Personal/Compatibility/GT-Sim.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse complete per-task score artifacts.")
    parser.add_argument("--gaa-clip-weights", type=Path, default=Path(r"C:\Users\Administrator\.cache\clip\ViT-L-14.pt"))
    parser.add_argument(
        "--schedules",
        default="4:4,20:2,28:1;5:4,18:2,28:1;6:4,16:2,28:1;7:4,14:2,28:1;8:4,12:2,28:1;9:4,10:2,28:1",
        help="Semicolon-separated STEP:KEEP nodes. All nodes must terminate at 28:1.",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def task_metadata_paths(cache_root: Path, max_samples: int) -> list[Path]:
    indexed = []
    for path in cache_root.glob("psp_offline_group*_task*_seed*/metadata.json"):
        match = TASK_RE.match(path.parent.name)
        if match:
            indexed.append((int(match.group("task")), path))
    indexed.sort(key=lambda value: value[0])
    if len(indexed) < max_samples:
        raise RuntimeError(f"expected at least {max_samples} task caches under {cache_root}, found {len(indexed)}")
    paths = [path for _, path in indexed[:max_samples]]
    expected = list(range(max_samples))
    actual = [int(TASK_RE.match(path.parent.name).group("task")) for path in paths]
    if actual != expected:
        raise RuntimeError(f"cache task indices must be 0..{max_samples - 1}, found {actual[:8]}...")
    return paths


def validate_cache(metadata: dict) -> tuple[list[int], list[int]]:
    cache = metadata["offline_candidate_cache"]
    steps = [int(value) for value in cache["steps"]]
    seeds = [int(value) for value in metadata["candidate_seeds"]]
    if steps != list(range(1, 29)) or len(seeds) != 8:
        raise ValueError("this evaluator requires the complete 28-step, 8-seed FITB cache")
    rows = cache["images"]
    if len(rows) != 224 or int(cache["image_count"]) != 224:
        raise ValueError(f"incomplete cache for task {metadata['task_index']}")
    seen = {(int(row["step"]), int(row["seed"])) for row in rows}
    if seen != {(step, seed) for step in steps for seed in seeds}:
        raise ValueError(f"invalid step/seed coverage for task {metadata['task_index']}")
    if not all(Path(row["image_path"]).is_file() for row in rows):
        raise FileNotFoundError(f"missing cached x0 image for task {metadata['task_index']}")
    return steps, seeds


def parse_schedules(text: str) -> list[tuple[str, list[tuple[int, int]]]]:
    schedules = []
    for raw in (value.strip() for value in text.split(";") if value.strip()):
        nodes = []
        active, previous, calls = 8, 0, 0
        for token in raw.split(","):
            step_text, keep_text = token.strip().split(":", 1)
            step, keep = int(step_text), int(keep_text)
            if step <= previous or not 1 <= keep <= active:
                raise ValueError(f"invalid PSP node in schedule {raw}: {token}")
            calls += active * (step - previous)
            active, previous = keep, step
            nodes.append((step, keep))
        if not nodes or nodes[-1] != (28, 1) or calls != 112:
            raise ValueError(f"schedule {raw} must end at 28:1 and use exactly 112 denoiser calls; got {calls}")
        name = "s" + "_".join(f"{step}k{keep}" for step, keep in nodes)
        schedules.append((name, nodes))
    if not schedules:
        raise ValueError("--schedules cannot be empty")
    return schedules


def batched(values: list[Path], size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


@torch.inference_mode()
def encode_paths(model, preprocess, paths: list[Path], device: torch.device, batch_size: int) -> np.ndarray:
    def load_one(path: Path):
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"))

    encoded = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for chunk in batched(paths, batch_size):
            pixels = torch.stack(list(pool.map(load_one, chunk))).to(device, dtype=next(model.parameters()).dtype)
            encoded.append(model.encode_image(pixels).float().cpu())
    return torch.cat(encoded).numpy()


def build_context(incomplete: list[int], item_features: np.ndarray) -> tuple[torch.Tensor, int]:
    blank = incomplete.index(0)
    context = torch.zeros((len(incomplete), item_features.shape[1]), dtype=torch.float32)
    for index, item_id in enumerate(incomplete):
        if index != blank:
            context[index] = torch.from_numpy(np.asarray(item_features[item_id])).float()
    return context, blank


def load_reward_models(args: argparse.Namespace, device: torch.device):
    if str(args.dualfashion_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.dualfashion_root.resolve()))
    from eval.compatibility_evaluator.compatibility_net import FashionEvaluator

    if not args.personal_clip_weights.is_file():
        raise FileNotFoundError(f"missing local ViT-H/14 weights: {args.personal_clip_weights}")
    from safetensors.torch import load_file

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    missing, unexpected = clip.load_state_dict(load_file(str(args.personal_clip_weights), device="cpu"), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ViT-H/14 checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    clip = clip.to(device).eval()
    evaluator_path = args.dualfashion_root / "eval" / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(evaluator_path, map_location="cpu"))
    return clip, preprocess, evaluator.to(device).eval()


def score_task(
    metadata: dict,
    fitb: dict,
    history: dict,
    item_features: np.ndarray,
    clip,
    preprocess,
    evaluator,
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    task = int(metadata["task_index"])
    uid, oid = int(metadata["uid"]), int(metadata["oid"])
    incomplete = [int(value) for value in fitb["outfits"][task].tolist()]
    categories = [int(value) for value in fitb["category"][task].tolist()]
    context, blank = build_context(incomplete, item_features)
    category = categories[blank]
    history_vector = F.normalize(torch.as_tensor(history[uid][category], dtype=torch.float32), dim=0).to(device)
    lookup = {(int(row["step"]), int(row["seed"])): row for row in metadata["offline_candidate_cache"]["images"]}
    steps, seeds = validate_cache(metadata)
    rows = []
    for step in steps:
        paths = [Path(lookup[(step, seed)]["image_path"]) for seed in seeds]
        raw = torch.as_tensor(encode_paths(clip, preprocess, paths, device, batch_size), device=device)
        if raw.shape[1] != 1024:
            raise RuntimeError(f"OpenCLIP feature dim {raw.shape[1]} != 1024")
        with torch.inference_mode():
            outfits = context.unsqueeze(0).expand(len(seeds), -1, -1).clone().to(device)
            outfits[:, blank, :] = raw
            compatibility = torch.sigmoid(evaluator(outfits)).detach().cpu().numpy()
            personal = (100.0 * (F.normalize(raw, dim=-1) @ history_vector)).detach().cpu().numpy()
        for index, seed in enumerate(seeds):
            cache_row = lookup[(step, seed)]
            rows.append({
                "task_index": task, "uid": uid, "oid": oid, "seed": seed, "step": step,
                "scheduler_timestep": float(cache_row["scheduler_timestep"]), "sigma": float(cache_row["sigma"]),
                "image_path": str(paths[index].resolve()), "personal": float(personal[index]),
                "compatibility": float(compatibility[index]),
            })
        del raw, outfits
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def selection_rows(candidate_rows: list[dict], schedules: list[tuple[str, list[tuple[int, int]]]]) -> tuple[list[dict], list[dict]]:
    by_task_step: dict[int, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in candidate_rows:
        by_task_step[int(row["task_index"])][int(row["step"])][int(row["seed"])] = row
    selected, traces = [], []
    for task, by_step in sorted(by_task_step.items()):
        seeds = sorted(by_step[28])
        standard_seed = seeds[0]
        bon_seed = min(seeds[:4], key=lambda seed: (-by_step[28][seed]["personal"], seed))
        for name, nodes in schedules:
            active = list(seeds)
            trace = []
            for step, keep in nodes:
                ranked = sorted(active, key=lambda seed: (-by_step[step][seed]["personal"], seed))
                trace.append({"step": step, "active_before": active, "active_after": ranked[:keep],
                              "personal_scores": {str(seed): by_step[step][seed]["personal"] for seed in active}})
                active = ranked[:keep]
            psp_seed = active[0]
            traces.append({"task_index": task, "schedule": name, "nodes": nodes, "trace": trace})
            for method, seed in (("standard", standard_seed), ("bon4_paired", bon_seed), ("psp", psp_seed)):
                terminal = dict(by_step[28][seed])
                terminal.update({"schedule": name, "method": method, "selection_reward": float(by_step[28][seed]["personal"])})
                selected.append(terminal)
    return selected, traces


def load_gaa_model(weights: Path, device: torch.device):
    if not weights.is_file():
        raise FileNotFoundError(f"GAA weights unavailable: {weights}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None, device=device, force_quick_gelu=True)
    try:
        open_clip.load_checkpoint(model, str(weights), strict=False, weights_only=False, device=device)
    except (TypeError, RuntimeError):
        archive = torch.jit.load(str(weights), map_location=device)
        model.load_state_dict(archive.state_dict(), strict=False)
        del archive
    if device.type == "cuda":
        model = model.half()
    return model.eval(), preprocess


def gt_sim_task(
    task: int,
    rows: list[dict],
    fitb: dict,
    grd: dict,
    item_paths: np.ndarray,
    image_root: Path,
    clip_h,
    preprocess_h,
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    incomplete = [int(value) for value in fitb["outfits"][task].tolist()]
    blank = incomplete.index(0)
    gt_item = int(grd["outfits"][task][blank])
    gt_path = image_root / str(item_paths[gt_item])
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    paths = [Path(row["image_path"]) for row in rows]
    h = encode_paths(clip_h, preprocess_h, paths + [gt_path], device, batch_size)
    h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-8)
    gt_h = h[-1]
    out = []
    for index, base in enumerate(rows):
        row = dict(base)
        row["gt_sim"] = float(h[index] @ gt_h)
        out.append(row)
    return out


def gaa_task(
    task: int,
    rows: list[dict],
    fitb: dict,
    grd: dict,
    item_paths: np.ndarray,
    image_root: Path,
    clip_l,
    preprocess_l,
    text_banks: dict[str, np.ndarray],
    gaa_scale: float,
    device: torch.device,
    batch_size: int,
) -> list[dict]:
    incomplete = [int(value) for value in fitb["outfits"][task].tolist()]
    blank = incomplete.index(0)
    gt_item = int(grd["outfits"][task][blank])
    gt_path = image_root / str(item_paths[gt_item])
    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    paths = [Path(row["image_path"]) for row in rows]
    l = encode_paths(clip_l, preprocess_l, paths + [gt_path], device, batch_size)
    l = l / np.maximum(np.linalg.norm(l, axis=1, keepdims=True), 1e-8)
    gt_l = l[-1]
    out = []
    for index, base in enumerate(rows):
        row = dict(base)
        agreements = []
        for group, text in text_banks.items():
            generated = torch.softmax(torch.tensor(l[index] @ text.T * gaa_scale), dim=0).numpy()
            target = torch.softmax(torch.tensor(gt_l @ text.T * gaa_scale), dim=0).numpy()
            value = float(np.minimum(generated, target).sum())
            row[f"gaa_{group}"] = value
            agreements.append(value)
        row["gaa"] = float(np.mean(agreements))
        out.append(row)
    return out


def attach_posthoc(selected: list[dict], candidates: list[dict]) -> list[dict]:
    lookup = {(int(row["task_index"]), int(row["step"]), int(row["seed"])): row for row in candidates}
    result = []
    for row in selected:
        metrics = lookup[(int(row["task_index"]), 28, int(row["seed"]))]
        result.append({**row, **{key: metrics[key] for key in ("gt_sim", "gaa", "gaa_color", "gaa_pattern", "gaa_material", "gaa_style")}})
    return result


def summarize(selected: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        grouped[(row["schedule"], row["method"])].append(row)
    metrics = ("personal", "compatibility", "gt_sim", "gaa", "gaa_color", "gaa_pattern", "gaa_material", "gaa_style")
    summary = []
    for (schedule, method), rows in sorted(grouped.items()):
        record = {"schedule": schedule, "method": method, "n_tasks": len(rows)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in rows])
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(record)
    return summary


def main() -> None:
    args = parse_args()
    if args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("--max-samples and --batch-size must be positive")
    schedules = parse_schedules(args.schedules)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metadata_paths = task_metadata_paths(args.cache_root, args.max_samples)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pre_dir = args.output_root / "pre_gt_tasks"
    gt_sim_dir = args.output_root / "posthoc_gt_sim_tasks"
    gaa_dir = args.output_root / "posthoc_gaa_tasks"
    fitb = np.load(args.data_root / "processed" / "fitb_test.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")

    clip_h, preprocess_h, evaluator = load_reward_models(args, device)
    for metadata_path in tqdm(metadata_paths, desc="Personal/Compatibility scoring", unit="task"):
        metadata = read_json(metadata_path)
        task = int(metadata["task_index"])
        destination = pre_dir / f"task_{task:04d}.json"
        if args.resume and destination.is_file():
            cached = read_json(destination)
            if isinstance(cached, list) and len(cached) == 224:
                continue
        rows = score_task(metadata, fitb, history, item_features, clip_h, preprocess_h, evaluator, device, args.batch_size)
        atomic_json(destination, rows)
    del evaluator
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidates = []
    for task in range(args.max_samples):
        rows = read_json(pre_dir / f"task_{task:04d}.json")
        if not isinstance(rows, list) or len(rows) != 224:
            raise RuntimeError(f"invalid pre-GT artifact for task {task}")
        candidates.extend(rows)
    write_csv(args.output_root / "candidate_metrics_pre_gt.csv", candidates)
    selected_pre_gt, traces = selection_rows(candidates, schedules)
    write_csv(args.output_root / "selection_pre_gt.csv", selected_pre_gt)
    with (args.output_root / "selection_traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace) + "\n")

    # This is intentionally after selection_pre_gt.csv has been written.
    grd = np.load(args.data_root / "test_grd_dict.npy", allow_pickle=True).item()
    item_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    by_task = defaultdict(list)
    for row in candidates:
        by_task[int(row["task_index"])].append(row)
    with_gt_sim = []
    for task in tqdm(range(args.max_samples), desc="Post-hoc GT-Sim", unit="task"):
        destination = gt_sim_dir / f"task_{task:04d}.json"
        if args.resume and destination.is_file():
            cached = read_json(destination)
            if isinstance(cached, list) and len(cached) == 224:
                with_gt_sim.extend(cached)
                continue
        rows = gt_sim_task(task, by_task[task], fitb, grd, item_paths, args.image_root, clip_h, preprocess_h,
                           device, args.batch_size)
        atomic_json(destination, rows)
        with_gt_sim.extend(rows)
    del clip_h
    if device.type == "cuda":
        torch.cuda.empty_cache()

    clip_l, preprocess_l = load_gaa_model(args.gaa_clip_weights, device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    text_banks = {}
    with torch.inference_mode():
        for name, values in ATTRIBUTE_BANKS.items():
            prompts = [f"a product photo of a fashion item that is {value}" for value in values]
            z = clip_l.encode_text(tokenizer(prompts).to(device))
            text_banks[name] = F.normalize(z.float(), dim=-1).cpu().numpy()
    gaa_scale = float(clip_l.logit_scale.exp().float().cpu())
    by_task = defaultdict(list)
    for row in with_gt_sim:
        by_task[int(row["task_index"])].append(row)
    enriched = []
    for task in tqdm(range(args.max_samples), desc="Post-hoc GAA", unit="task"):
        destination = gaa_dir / f"task_{task:04d}.json"
        if args.resume and destination.is_file():
            cached = read_json(destination)
            if isinstance(cached, list) and len(cached) == 224:
                enriched.extend(cached)
                continue
        rows = gaa_task(task, by_task[task], fitb, grd, item_paths, args.image_root, clip_l, preprocess_l,
                        text_banks, gaa_scale, device, args.batch_size)
        atomic_json(destination, rows)
        enriched.extend(rows)
    write_csv(args.output_root / "candidate_metrics_with_gt.csv", enriched)
    selected = attach_posthoc(selected_pre_gt, enriched)
    write_csv(args.output_root / "selected_metrics_with_gt.csv", selected)
    summary = summarize(selected)
    write_csv(args.output_root / "schedule_summary.csv", summary)
    atomic_json(args.output_root / "run_manifest.json", {
        "reward": "Personal = 100*cosine(normalized OpenCLIP ViT-H/14(candidate), normalized user test-history embedding for target category)",
        "selection": "Personal only; GT-Sim and GAA are computed after selection_pre_gt.csv is frozen",
        "cache_root": str(args.cache_root.resolve()), "task_count": args.max_samples,
        "candidate_count": len(enriched), "schedules": [{"name": name, "nodes": nodes, "denoiser_calls": 112} for name, nodes in schedules],
        "gt_sim": "OpenCLIP ViT-H/14 cosine to FITB ground-truth item, post-hoc only",
        "gaa": "OpenAI CLIP ViT-L/14 attribute-distribution overlap, post-hoc only",
    })
    print(json.dumps({"output": str(args.output_root), "candidate_count": len(enriched), "schedules": [name for name, _ in schedules]}, indent=2))


if __name__ == "__main__":
    main()
