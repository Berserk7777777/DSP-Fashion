#!/usr/bin/env python
"""Post-hoc GT-oracle regret audit for frozen DiFashion FITB selections.

GT is read only after frozen BoN/PSP selections have been loaded.  It is never
used to generate candidates, score Personal Sim, rank candidates, or prune
PSP trajectories.  For each cached terminal candidate, this tool computes
official-style GT Sim plus GT Attribute Alignment (GAA), then reports:

* final selection regret against each method's own terminal candidate pool;
* PSP pruning regret after the frozen step-13 and step-26 decisions.

The audit is descriptive/test-set-oracle analysis, not an online reward.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


ATTRIBUTE_BANKS = {
    "color": ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"],
    "pattern": ["solid", "striped", "plaid", "floral", "polka dot", "geometric", "animal print", "quilted", "textured"],
    "material": ["leather", "faux leather", "suede", "canvas", "denim", "knit", "cotton", "wool", "synthetic material"],
    "style": ["casual", "formal", "sporty", "elegant", "vintage", "minimalist", "streetwear"],
}
PSP_NODES = ((13, "reward_step_13"), (26, "reward_step_26"), ("final", "final"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True,
                        help="Frozen reward-selection output containing selections/.")
    parser.add_argument("--expected-reward", required=True,
                        help="Exact frozen reward_name expected in selection records, e.g. personal or pareto.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="iFashion directory containing processed/ and test_grd.npy.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--personal-clip-weights", type=Path, required=True,
                        help="Local OpenCLIP ViT-H/14 LAION checkpoint used by GT Sim.")
    parser.add_argument("--gaa-clip-weights", type=Path,
                        default=Path(r"C:\Users\Administrator\.cache\clip\ViT-L-14.pt"))
    parser.add_argument("--gaa-text-embeddings", type=Path, required=True,
                        help="Frozen attribute text embeddings from evaluate_gt_attribute_alignment.py.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--personal-batch-size", type=int, default=8)
    parser.add_argument("--gaa-batch-size", type=int, default=24)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def image_ok(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception as exc:
        raise RuntimeError(f"Unreadable image: {path}") from exc


def build_tasks(data_root: Path, image_root: Path) -> list[dict[str, Any]]:
    fitb_path = data_root / "processed" / "new_fitb_test.npy"
    gt_path = data_root / "test_grd.npy"
    item_paths_path = data_root / "new_all_item_image_paths.npy"
    for path in (fitb_path, gt_path, item_paths_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    fitb = np.load(fitb_path, allow_pickle=True).item()
    ground = np.load(gt_path, allow_pickle=True).item()
    item_paths = np.load(item_paths_path, allow_pickle=True)
    tasks = []
    for index, (uid, oid, outfit, category) in enumerate(zip(
        fitb["uids"], fitb["oids"], fitb["outfits"], fitb["category"]
    )):
        uid, oid = int(uid), int(oid)
        outfit = np.asarray(outfit, dtype=np.int64).reshape(-1)
        category = np.asarray(category, dtype=np.int64).reshape(-1)
        blanks = np.flatnonzero(outfit == 0)
        if len(outfit) != 4 or len(category) != 4 or len(blanks) != 1:
            raise ValueError(f"Malformed FITB condition row {index}")
        blank = int(blanks[0])
        record = ground.get(oid, ground.get(str(oid)))
        if record is None:
            raise KeyError(f"Missing GT record for oid={oid}")
        gt_outfit = np.asarray(record["outfits"], dtype=np.int64).reshape(-1)
        gt_category = np.asarray(record["category"], dtype=np.int64).reshape(-1)
        if len(gt_outfit) != 4 or len(gt_category) != 4:
            raise ValueError(f"Malformed GT record for oid={oid}")
        if any(int(outfit[pos]) != int(gt_outfit[pos]) for pos in range(4) if pos != blank):
            raise ValueError(f"GT context mismatch at FITB row {index}")
        if int(category[blank]) != int(gt_category[blank]):
            raise ValueError(f"GT category mismatch at FITB row {index}")
        item_id = int(gt_outfit[blank])
        candidate = image_root / str(item_paths[item_id])
        image_ok(candidate)
        tasks.append({
            "task_index": index, "task_id": f"{index:05d}_{uid}_{oid}", "task_dir_name": f"{uid}_{oid}",
            "uid": uid, "oid": oid, "gt_item_id": item_id, "gt_image_path": str(candidate.resolve()),
        })
    return tasks


def train_overlap(data_root: Path, tasks: list[dict[str, Any]]) -> None:
    train_path = data_root / "processed" / "new_train.npy"
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    train = np.load(train_path, allow_pickle=True).item()
    train_ids = {int(item) for outfit in train["outfits"] for item in np.asarray(outfit).reshape(-1)}
    for task in tasks:
        task["gt_item_seen_in_train"] = task["gt_item_id"] in train_ids


def collect_candidates(cache_root: Path, selection_root: Path, tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    task_by_id = {task["task_id"]: task for task in tasks}
    selection_dir = selection_root / "selections"
    if not selection_dir.is_dir():
        raise FileNotFoundError(selection_dir)
    candidate_rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    for task in tqdm(tasks, desc="Inspecting frozen candidate cache", unit="task"):
        metadata_path = cache_root / "metadata" / f"{task['task_dir_name']}.json"
        selection_path = selection_dir / "psp" / f"{task['task_id']}.json"
        if not metadata_path.is_file() or not selection_path.is_file():
            raise FileNotFoundError(f"Missing metadata or selection for {task['task_id']}")
        metadata, selection = read_json(metadata_path), read_json(selection_path)
        for key in ("task_index", "task_id", "uid", "oid"):
            if metadata.get(key) != task[key] or selection.get(key) != task[key]:
                raise ValueError(f"Task identity mismatch in frozen artifacts: {task['task_id']}, {key}")
        if selection.get("reward_name") != args.expected_reward:
            raise ValueError(
                f"Selection reward mismatch for {task['task_id']}: "
                f"{selection.get('reward_name')!r} != {args.expected_reward!r}"
            )
        methods = metadata.get("methods", {})
        for method, expected in (("bon", 4), ("psp", 8)):
            candidates = methods.get(method, {}).get("candidates", [])
            if len(candidates) != expected:
                raise ValueError(f"{task['task_id']} has {len(candidates)} {method} candidates, expected {expected}")
            for candidate in candidates:
                seed = int(candidate["seed"])
                image_path = cache_root / candidate["files"]["final_image"]
                image_ok(image_path)
                candidate_rows.append({
                    **task, "method": method, "candidate_index": int(candidate["candidate_index"]), "seed": seed,
                    "final_image_path": str(image_path.resolve()),
                })
        selections[task["task_id"]] = selection
    return candidate_rows, selections


def batch(items: list[str], size: int):
    for begin in range(0, len(items), size):
        yield items[begin:begin + size]


@torch.inference_mode()
def encode_images(model, preprocess, paths: list[str], device: torch.device, batch_size: int, label: str) -> np.ndarray:
    def load(path: str):
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"))
    output = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for part in tqdm(list(batch(paths, batch_size)), desc=label, unit="batch"):
            pixels = torch.stack(list(pool.map(load, part))).to(device, dtype=next(model.parameters()).dtype)
            features = model.encode_image(pixels)
            features = F.normalize(features.float(), dim=-1)
            output.append(features.cpu().numpy())
    return np.concatenate(output, axis=0)


def cached_embeddings(cache_path: Path, index_path: Path, paths: list[str], loader) -> np.ndarray:
    if cache_path.is_file() and index_path.is_file():
        saved_paths = json.loads(index_path.read_text(encoding="utf-8"))
        if saved_paths != paths:
            raise RuntimeError(f"Embedding cache path list differs: {cache_path}; use another output directory")
        return np.load(cache_path)["embeddings"]
    values = loader()
    np.savez_compressed(cache_path, embeddings=values)
    index_path.write_text(json.dumps(paths), encoding="utf-8")
    return values


def load_gaa_model(weights: Path, device: torch.device):
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None, device=device, force_quick_gelu=True)
    try:
        open_clip.load_checkpoint(model, str(weights), strict=False, weights_only=False, device=device)
    except TypeError:
        try:
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


def compute_candidate_metrics(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = list(dict.fromkeys([row["final_image_path"] for row in rows] + [row["gt_image_path"] for row in rows]))
    path_index = {path: index for index, path in enumerate(paths)}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    def h_loader():
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=str(args.personal_clip_weights), device=device
        )
        # The cached Personal/GT-Sim model is inference-only.  Keeping it in
        # FP32 consumes most of a 16 GB GPU and makes this offline audit
        # needlessly slow; feature vectors are normalized in FP32 below.
        if device.type == "cuda":
            model = model.half()
        model.eval()
        values = encode_images(model, preprocess, paths, device, args.personal_batch_size, "GT Sim image embeddings")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return values
    h_embeddings = cached_embeddings(args.output_root / "vit_h14_embeddings.npz",
                                    args.output_root / "vit_h14_embedding_paths.json", paths, h_loader)
    gen_h = h_embeddings[[path_index[row["final_image_path"]] for row in rows]]
    gt_h = h_embeddings[[path_index[row["gt_image_path"]] for row in rows]]
    gt_sim = 100.0 * np.sum(gen_h * gt_h, axis=1)

    if not args.gaa_text_embeddings.is_file():
        raise FileNotFoundError(args.gaa_text_embeddings)
    text_embeddings = np.load(args.gaa_text_embeddings)
    if set(text_embeddings.files) != set(ATTRIBUTE_BANKS):
        raise ValueError("Frozen GAA text embedding groups do not match the documented attribute banks")
    def l_loader():
        model, preprocess = load_gaa_model(args.gaa_clip_weights, device)
        values = encode_images(model, preprocess, paths, device, args.gaa_batch_size, "GAA image embeddings")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return values
    l_embeddings = cached_embeddings(args.output_root / "vit_l14_embeddings.npz",
                                    args.output_root / "vit_l14_embedding_paths.json", paths, l_loader)
    model, _ = load_gaa_model(args.gaa_clip_weights, device)
    logit_scale = float(model.logit_scale.exp().detach().float().cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gen_l = l_embeddings[[path_index[row["final_image_path"]] for row in rows]]
    gt_l = l_embeddings[[path_index[row["gt_image_path"]] for row in rows]]
    for group in ATTRIBUTE_BANKS:
        text = text_embeddings[group]
        p_gen = torch.softmax(torch.from_numpy(gen_l @ text.T * logit_scale), dim=1).numpy()
        p_gt = torch.softmax(torch.from_numpy(gt_l @ text.T * logit_scale), dim=1).numpy()
        values = np.minimum(p_gen, p_gt).sum(axis=1)
        for row, value in zip(rows, values):
            row[f"gaa_{group}"] = float(value)
    for row, value in zip(rows, gt_sim):
        row["gt_sim"] = float(value)
        row["gaa"] = float(np.mean([row[f"gaa_{group}"] for group in ATTRIBUTE_BANKS]))
    return rows


def metric_names() -> list[str]:
    return ["gt_sim", "gaa", *[f"gaa_{group}" for group in ATTRIBUTE_BANKS]]


def regret_rows(rows: list[dict[str, Any]], selections: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_task_method: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row["task_id"], row["method"])
        if row["seed"] in by_task_method[key]:
            raise ValueError(f"Duplicate candidate metric for {key}, seed {row['seed']}")
        by_task_method[key][row["seed"]] = row
    result = []
    for task_id, selection in selections.items():
        for method, expected_count in (("bon", 4), ("psp", 8)):
            candidates = by_task_method[(task_id, method)]
            if len(candidates) != expected_count:
                raise ValueError(f"Incomplete candidate metrics for {task_id}/{method}")
        for metric in metric_names():
            bon = selection["bon"]
            bon_winner = int(bon["winner_seed"])
            bon_oracle_seed = max(by_task_method[(task_id, "bon")], key=lambda seed: by_task_method[(task_id, "bon")][seed][metric])
            bon_oracle = float(by_task_method[(task_id, "bon")][bon_oracle_seed][metric])
            bon_selected = float(by_task_method[(task_id, "bon")][bon_winner][metric])
            result.append({
                "task_id": task_id, "task_index": selection["task_index"], "uid": selection["uid"], "oid": selection["oid"],
                "gt_item_seen_in_train": by_task_method[(task_id, "bon")][bon_winner]["gt_item_seen_in_train"],
                "method": "bon", "node": "final", "metric": metric, "pool_size": 4,
                "oracle_seed": bon_oracle_seed, "oracle_value": bon_oracle, "selected_seed": bon_winner,
                "selected_value": bon_selected, "regret": bon_oracle - bon_selected,
                "oracle_hit": math.isclose(bon_oracle, bon_selected, abs_tol=1e-7),
                "definition": "terminal selection regret within the frozen BoN@4 seed pool",
            })
            psp = selection["psp"]
            psp_oracle_seed = max(by_task_method[(task_id, "psp")], key=lambda seed: by_task_method[(task_id, "psp")][seed][metric])
            psp_oracle = float(by_task_method[(task_id, "psp")][psp_oracle_seed][metric])
            for checkpoint, node in PSP_NODES:
                entry = next(item for item in psp["schedule"] if item["checkpoint"] == checkpoint)
                active = [int(seed) for seed in entry["active_after_seeds"]]
                if not active or any(seed not in by_task_method[(task_id, "psp")] for seed in active):
                    raise ValueError(f"Invalid PSP active set for {task_id}/{node}")
                surviving_seed = max(active, key=lambda seed: by_task_method[(task_id, "psp")][seed][metric])
                surviving_value = float(by_task_method[(task_id, "psp")][surviving_seed][metric])
                result.append({
                    "task_id": task_id, "task_index": selection["task_index"], "uid": selection["uid"], "oid": selection["oid"],
                    "gt_item_seen_in_train": by_task_method[(task_id, "psp")][psp_oracle_seed]["gt_item_seen_in_train"],
                    "method": "psp", "node": node, "metric": metric, "pool_size": 8,
                    "oracle_seed": psp_oracle_seed, "oracle_value": psp_oracle, "selected_seed": surviving_seed,
                    "selected_value": surviving_value, "regret": psp_oracle - surviving_value,
                    "oracle_hit": math.isclose(psp_oracle, surviving_value, abs_tol=1e-7),
                    "definition": "PSP pruning regret: terminal oracle over all 8 initial seeds minus best terminal oracle value among seeds surviving this node",
                })
    return result


def summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split, predicate in (("all", lambda row: True), ("gt_item_seen_train", lambda row: row["gt_item_seen_in_train"]),
                             ("gt_item_unseen_train", lambda row: not row["gt_item_seen_in_train"])):
        buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if predicate(row):
                buckets[(row["method"], row["node"], row["metric"])].append(row)
        for (method, node, metric), values in sorted(buckets.items()):
            regret = np.asarray([float(row["regret"]) for row in values])
            output.append({
                "split": split, "method": method, "node": node, "metric": metric, "n_tasks": len(values),
                "mean_regret": float(regret.mean()), "median_regret": float(np.median(regret)),
                "std_sample_regret": float(regret.std(ddof=1)) if len(regret) > 1 else None,
                "oracle_hit_rate": float(np.mean([bool(row["oracle_hit"]) for row in values])),
                "mean_oracle_value": float(np.mean([float(row["oracle_value"]) for row in values])),
                "mean_selected_value": float(np.mean([float(row["selected_value"]) for row in values])),
            })
    return output


def main() -> int:
    args = parse_args()
    for path in (args.cache_root, args.selection_root, args.data_root, args.image_root, args.personal_clip_weights, args.gaa_clip_weights):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.personal_batch_size < 1 or args.gaa_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "audit_config.json"
    config = {
        "schema": 2, "cache_root": str(args.cache_root.resolve()), "selection_root": str(args.selection_root.resolve()),
        "expected_reward": args.expected_reward,
        "data_root": str(args.data_root.resolve()), "image_root": str(args.image_root.resolve()),
        "personal_clip": "OpenCLIP ViT-H-14/laion2b-s32b-b79K; local checkpoint", "gaa_clip": "OpenAI CLIP ViT-L/14",
        "metrics": {"gt_sim": "100*cosine(OpenCLIP terminal candidate, OpenCLIP GT image)",
                    "gaa": "mean color/pattern/material/style overlap using frozen CLIP zero-shot attribute distributions"},
        "gt_usage": "post-hoc oracle evaluation only; never candidate generation, scoring, or selection",
        "regret": "oracle terminal metric over initial method pool minus best terminal metric remaining after frozen selection/pruning",
        "test_status": "exploratory; reward/policy must be frozen on validation before an unbiased test claim",
    }
    if config_path.is_file() and read_json(config_path) != config:
        raise ValueError("Existing audit configuration differs; choose a new output root")
    atomic_json(config_path, config)
    tasks = build_tasks(args.data_root, args.image_root)
    train_overlap(args.data_root, tasks)
    rows, selections = collect_candidates(args.cache_root, args.selection_root, tasks)
    candidate_path = args.output_root / "candidate_terminal_metrics.csv"
    if args.resume and candidate_path.is_file():
        with candidate_path.open(newline="", encoding="utf-8") as handle:
            saved = list(csv.DictReader(handle))
        expected = {(row["task_id"], row["method"], str(row["seed"])) for row in rows}
        actual = {(row["task_id"], row["method"], row["seed"]) for row in saved}
        if actual != expected:
            raise RuntimeError("Saved candidate table does not match current frozen cache")
        for row in saved:
            for key in ("gt_sim", "gaa", *[f"gaa_{group}" for group in ATTRIBUTE_BANKS]):
                row[key] = float(row[key])
            row["seed"] = int(row["seed"]); row["candidate_index"] = int(row["candidate_index"])
            row["task_index"] = int(row["task_index"]); row["uid"] = int(row["uid"]); row["oid"] = int(row["oid"])
            row["gt_item_id"] = int(row["gt_item_id"]); row["gt_item_seen_in_train"] = row["gt_item_seen_in_train"].lower() == "true"
        rows = saved
    else:
        rows = compute_candidate_metrics(args, rows)
        atomic_csv(candidate_path, rows)
    details = regret_rows(rows, selections)
    summary = summaries(details)
    atomic_csv(args.output_root / "per_task_oracle_regret.csv", details)
    atomic_csv(args.output_root / "oracle_regret_summary.csv", summary)
    atomic_json(args.output_root / "integrity_report.json", {
        "tasks": len(tasks), "candidate_terminal_images": len(rows), "expected_candidates": len(tasks) * 12,
        "regret_rows": len(details), "expected_regret_rows": len(tasks) * (1 + len(PSP_NODES)) * len(metric_names()),
        "missing": [], "complete": len(rows) == len(tasks) * 12,
    })
    print(json.dumps({"output": str(args.output_root.resolve()), "tasks": len(tasks), "candidate_terminal_images": len(rows),
                      "regret_rows": len(details), "summary_rows": len(summary)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
