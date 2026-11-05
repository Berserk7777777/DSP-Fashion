#!/usr/bin/env python
"""Reproducible offline BoN/PSP selection for the DiFashion iFashion FITB cache.

This is deliberately a *replay* tool.  It never loads ``test_grd.npy`` or a
retrieval-candidate file while computing candidate rewards or making a
selection.  PSP trajectories were collected in full beforehand.  At each
checkpoint this program scores the saved decoded ``x_prediction`` image, and
the final winner is the terminal image from that same seed's full trajectory.
It therefore simulates the PSP 8 -> 4 -> 2 -> 1 decision rule; it does not
measure the wall-clock cost of an online dynamically-pruned sampler.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


SCORE_SCHEMA = 4
SELECTION_SCHEMA = 5
# The first two PSP decisions score decoded x_predictions.  The terminal
# decision scores the actual final scheduler decode, which is also evaluated.
PSP_SCHEDULE = ((13, 4, "reward_step_13"), (26, 2, "reward_step_26"), ("final", 1, "final"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="iFashion directory, not its parent (contains processed/).")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="iFashion item-image root. Defaults to <data-root>/images; used only for post-selection GT metrics.")
    parser.add_argument("--dualfashion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reward", choices=("compatibility", "personal", "bottleneck", "pareto", "plugin"),
                        default="compatibility")
    parser.add_argument("--plugin", default=None,
                        help="For --reward plugin: importable module:function returning one finite scalar.")
    parser.add_argument("--group-id", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true", help="Reuse task-atomic score/selection files in this output root.")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Only process FITB rows at or after this index; completed earlier rows remain valid.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit processed rows for a smoke run.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Permit selection/evaluation of complete cached tasks when the full 2,207-task cache is incomplete.")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--check-repeat", action="store_true",
                        help="Recompute deterministic selections and fail if existing records disagree.")
    parser.add_argument("--skip-inception", action="store_true",
                        help="Skip the available frozen Inception metrics; compatibility/personal still run.")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows])
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=True) + "\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def image_is_readable(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:  # pragma: no cover - error path is the point
        return False, repr(exc)


def expected_task_rows(data_root: Path) -> list[dict[str, Any]]:
    data_path = data_root / "processed" / "new_fitb_test.npy"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing FITB condition file: {data_path}")
    data = np.load(data_path, allow_pickle=True).item()
    required = {"uids", "oids", "outfits", "category"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{data_path} is missing keys: {sorted(missing)}")
    rows = []
    for index, (uid, oid, outfit, category) in enumerate(zip(
        data["uids"], data["oids"], data["outfits"], data["category"]
    )):
        outfit = np.asarray(outfit, dtype=np.int64).reshape(-1)
        category = np.asarray(category, dtype=np.int64).reshape(-1)
        blanks = np.flatnonzero(outfit == 0)
        if len(outfit) != 4 or len(blanks) != 1 or len(category) != len(outfit):
            raise ValueError(f"FITB row {index} is not a four-item outfit with exactly one blank")
        rows.append({
            "task_index": index,
            "task_id": f"{index:05d}_{int(uid)}_{int(oid)}",
            "task_dir_name": f"{int(uid)}_{int(oid)}",
            "uid": int(uid), "oid": int(oid), "outfit": outfit.tolist(),
            "category": category.tolist(), "blank_index": int(blanks[0]),
            "target_category": int(category[int(blanks[0])]),
        })
    return rows


def cache_candidate_records(task: dict[str, Any], cache_root: Path, config: dict[str, Any], verify_images: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Build score records solely from cache metadata and FITB condition rows."""
    errors: list[str] = []
    metadata_path = cache_root / "metadata" / f"{task['task_dir_name']}.json"
    if not metadata_path.is_file():
        return [], ["missing task_metadata.json"]
    try:
        metadata = read_json(metadata_path)
    except Exception as exc:
        return [], [f"unreadable task metadata: {exc!r}"]
    if metadata.get("status") != "complete":
        errors.append(f"task status is {metadata.get('status')!r}")
    for field in ("task_index", "task_id", "uid", "oid"):
        if metadata.get(field) != task[field]:
            errors.append(f"metadata {field} does not match FITB condition row")
    base_map = {"bon": int(config["bon_seed_base"]), "psp": int(config["psp_seed_base"])}
    count_map = {"bon": int(config["bon_candidates"]), "psp": int(config["psp_candidates"])}
    stride = int(config["seed_stride"])
    records: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for method in ("bon", "psp"):
        bundle = metadata.get("methods", {}).get(method, {})
        candidates = bundle.get("candidates", [])
        if bundle.get("candidate_count") != count_map[method] or len(candidates) != count_map[method]:
            errors.append(f"{method}: expected {count_map[method]} candidates, found {len(candidates)}")
            continue
        indices = {candidate.get("candidate_index") for candidate in candidates}
        if indices != set(range(count_map[method])):
            errors.append(f"{method}: candidate indices are not 0..{count_map[method] - 1}")
        for candidate in candidates:
            candidate_index = candidate.get("candidate_index")
            seed = candidate.get("seed")
            if not isinstance(candidate_index, int) or not isinstance(seed, int):
                errors.append(f"{method}: invalid candidate index or seed")
                continue
            expected_seed = base_map[method] + task["task_index"] * stride + candidate_index
            if seed != expected_seed:
                errors.append(f"{method} candidate {candidate_index}: seed {seed} != expected {expected_seed}")
            if seed in seen_seeds:
                errors.append(f"seed-pool overlap at seed {seed}")
            seen_seeds.add(seed)
            files = candidate.get("files", {})
            if method == "psp":
                steps = files.get("steps", {})
                if set(steps) != {"13", "26", "51"}:
                    errors.append(f"psp seed {seed}: expected step metadata 13/26/51")
                targets = []
                for step in (13, 26):
                    step_data = steps.get(str(step), {})
                    if (step_data.get("step_number_one_based") != step
                            or step_data.get("loop_index_zero_based") != step - 1
                            or not isinstance(step_data.get("scheduler_timestep"), int)):
                        errors.append(f"psp seed {seed}: invalid saved step semantics at {step}")
                    targets.append((f"reward_step_{step:02d}", step_data.get("decoded_reward_image"), step_data))
                # At the terminal decision, use the final scheduler decode,
                # not the different x_prediction image captured at loop 51.
                targets.append(("final", files.get("final_image"), {
                    "loop_index_zero_based": 50,
                    "step_number_one_based": 51,
                    "scheduler_timestep": 1,
                    "x_prediction_saved_as_decoded_image": False,
                    "terminal_scheduler_decode": True,
                }))
            if method == "bon":
                targets = [("final", files.get("final_image"), {})]
            for node, relative, node_metadata in targets:
                if not relative:
                    errors.append(f"{method} seed {seed}: missing image path for {node}")
                    continue
                image_path = cache_root / relative
                if not image_path.is_file():
                    errors.append(f"{method} seed {seed}: missing image {relative}")
                    continue
                if verify_images:
                    good, reason = image_is_readable(image_path)
                    if not good:
                        errors.append(f"{method} seed {seed}: unreadable {relative}: {reason}")
                        continue
                records.append({
                    "task_index": task["task_index"], "task_id": task["task_id"], "uid": task["uid"], "oid": task["oid"],
                    "method": method, "group_id": 0, "candidate_index": candidate_index, "seed": seed,
                    "node": node, "image_path": str(image_path.resolve()), "outfit": task["outfit"],
                    "blank_index": task["blank_index"], "target_category": task["target_category"],
                    "loop_index_zero_based": node_metadata.get("loop_index_zero_based"),
                    "scheduler_timestep": node_metadata.get("scheduler_timestep"),
                    "x_prediction_saved_as_decoded_image": bool(node_metadata) or None,
                    "continuation_latent_available_in_cache": False if node_metadata else None,
                    "continuation_latent_used_for_replay": False if node_metadata else None,
                })
    if len(records) != count_map["bon"] + count_map["psp"] * 3:
        errors.append(f"expected {count_map['bon'] + count_map['psp'] * 3} score images, found {len(records)}")
    return records, errors


def inspect_cache(args: argparse.Namespace, verify_images: bool) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    config_path = args.cache_root / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing cache run config: {config_path}")
    config = read_json(config_path)
    required = {"bon_seed_base", "psp_seed_base", "seed_stride", "bon_candidates", "psp_candidates", "psp_prune_steps"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Cache config missing fields: {sorted(missing)}")
    if tuple(map(int, config["psp_prune_steps"])) != (13, 26, 51):
        raise ValueError(f"This replay contract requires cached PSP nodes [13, 26, 51], found {config['psp_prune_steps']}")
    expected = expected_task_rows(args.data_root)
    by_id = {row["task_id"]: row for row in expected}
    duplicate_dirs: list[str] = []
    seen_indices: set[int] = set()
    for path in (args.cache_root / "metadata").glob("*.json"):
        try:
            meta = read_json(path)
            index = int(meta["task_index"])
        except Exception:
            continue
        if index in seen_indices:
            duplicate_dirs.append(str(path))
        seen_indices.add(index)
    complete: dict[str, list[dict[str, Any]]] = {}
    problems: list[dict[str, Any]] = []
    for task in expected:
        records, errors = cache_candidate_records(task, args.cache_root, config, verify_images)
        if errors:
            problems.append({"task_index": task["task_index"], "task_id": task["task_id"], "errors": errors})
        else:
            complete[task["task_id"]] = records
    for extra in (args.cache_root / "metadata").glob("*.json"):
        try:
            extra_meta = read_json(extra)
        except Exception:
            continue
        if extra_meta.get("task_id") not in by_id:
            problems.append({"task_index": None, "task_id": extra.name, "errors": ["metadata file not in current FITB condition rows"]})
    if duplicate_dirs:
        problems.append({"task_index": None, "task_id": None, "errors": ["duplicate task_index metadata: " + "; ".join(duplicate_dirs)]})
    return expected, config, complete, problems


def output_config(args: argparse.Namespace, cache_config: dict[str, Any], task_count: int) -> dict[str, Any]:
    return {
        "schema": 2, "cache_root": str(args.cache_root.resolve()), "data_root": str(args.data_root.resolve()),
        "dualfashion_root": str(args.dualfashion_root.resolve()), "group_id": args.group_id,
        "reward": args.reward, "plugin": args.plugin, "task_count_from_condition": task_count,
        "candidate_score_schema": SCORE_SCHEMA, "selection_schema": SELECTION_SCHEMA,
        "selection_tie_break": "descending reward; then ascending integer seed; no selection RNG",
        "psp_schedule": [{"node": node, "checkpoint_step": step, "keep": keep} for step, keep, node in PSP_SCHEDULE],
        "candidate_collection_cost_unet_calls_per_task": cache_config.get("offline_cost"),
        "simulated_online_psp_budget_unet_calls_per_task": cache_config.get("online_simulation_budget"),
        "psp_replay_scope": "offline selection from full trajectories; 13/26 score decoded x_predictions and final scores the terminal scheduler decode. This is not online dynamic-pruning timing evidence.",
        "ground_truth_policy": "No GT item/image/retrieval candidate is read during candidate score or selection. Target category is from the FITB condition row; final official IS-Acc uses it only after selection.",
        "test_leakage": "exploratory_test_only=true. Reward choice, plugin, and combination parameters must be frozen on validation data before reporting an unbiased test comparison.",
        "exploratory_test_only": True,
    }


def prepare_output(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if args.output_root.exists() and any(args.output_root.iterdir()):
        config_path = args.output_root / "run_config.json"
        if not config_path.is_file():
            raise FileExistsError(f"Non-empty output root has no run_config.json: {args.output_root}")
        existing = read_json(config_path)
        if existing != config:
            # Schema 4 adds the no-selection Standard baseline to existing
            # reward runs. It changes no seed protocol or candidate score.
            legacy = dict(existing)
            legacy["selection_schema"] = config["selection_schema"]
            if args.resume and legacy == config:
                atomic_json(config_path, config)
            else:
                raise ValueError("Output run_config.json differs. Use a new output root for a different reward/configuration.")
        if not args.resume and not args.audit_only:
            raise FileExistsError("Output root already contains a matching run. Re-run with --resume or choose a new output root.")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_root / "run_config.json", config)


def load_score_file(path: Path, expected: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        payload = read_json(path)
        rows = payload["rows"]
        expected_keys = {(r["method"], r["candidate_index"], r["node"]) for r in expected}
        actual_keys = {(r["method"], r["candidate_index"], r["node"]) for r in rows}
        if payload.get("schema") != SCORE_SCHEMA or actual_keys != expected_keys:
            return None
        for row in rows:
            for key in ("compatibility_logit", "compatibility_sigmoid"):
                if row.get(key) is None or not math.isfinite(float(row[key])):
                    return None
            if row.get("personal_sim") is not None and not math.isfinite(float(row["personal_sim"])):
                return None
        return rows
    except Exception:
        return None


def history_embedding(history: dict[Any, Any], uid: int, category: int) -> np.ndarray | None:
    entry = history.get(uid, history.get(str(uid)))
    if entry is None:
        return None
    value = entry.get(category, entry.get(str(category))) if isinstance(entry, dict) else None
    if value is None:
        return None
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    return value if value.size else None


def load_reward_models(args: argparse.Namespace, evaluator_ckpt: Path):
    # The frozen OpenCLIP checkpoint is already cached locally. Avoid network
    # HEAD retries so a reproducible evaluation can run in an offline lab.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dual_root = str(args.dualfashion_root.resolve())
    if dual_root not in sys.path:
        sys.path.insert(0, dual_root)
    import open_clip
    from eval.compatibility_evaluator.compatibility_net import FashionEvaluator

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b-s32b-b79K")
    clip = clip.to(device).eval()
    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(evaluator_ckpt, map_location="cpu", weights_only=True))
    evaluator = evaluator.to(device).eval()
    return device, clip, preprocess, evaluator


def resolve_image_root(args: argparse.Namespace) -> Path:
    root = args.image_root or (args.data_root / "images")
    if not root.is_dir():
        raise FileNotFoundError(f"iFashion item-image root is unavailable: {root}")
    return root


def load_gt_items(args: argparse.Namespace, task_by_id: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Load GT only after selection and verify it against the fixed FITB condition."""
    ground_path = args.data_root / "test_grd.npy"
    item_paths_path = args.data_root / "new_all_item_image_paths.npy"
    image_root = resolve_image_root(args)
    for path in (ground_path, item_paths_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required post-selection GT asset is missing: {path}")
    ground = np.load(ground_path, allow_pickle=True).item()
    item_paths = np.load(item_paths_path, allow_pickle=True)
    resolved: dict[str, dict[str, Any]] = {}
    for task_id, task in task_by_id.items():
        oid = task["oid"]
        record = ground.get(oid, ground.get(str(oid)))
        if record is None:
            raise KeyError(f"No GT record for oid={oid}, task={task_id}")
        outfit = np.asarray(record.get("outfits"), dtype=np.int64).reshape(-1)
        categories = np.asarray(record.get("category"), dtype=np.int64).reshape(-1)
        if len(outfit) != 4 or len(categories) != 4:
            raise ValueError(f"GT outfit for task {task_id} is not length four")
        blank = task["blank_index"]
        if outfit[blank] <= 0:
            raise ValueError(f"GT target item is invalid for task {task_id}")
        for index, item_id in enumerate(task["outfit"]):
            if index != blank and int(outfit[index]) != int(item_id):
                raise ValueError(f"GT context mismatch for task {task_id}, oid={oid}, position={index}")
        if int(categories[blank]) != task["target_category"]:
            raise ValueError(f"GT target category mismatch for task {task_id}, oid={oid}")
        target_item = int(outfit[blank])
        if target_item >= len(item_paths):
            raise IndexError(f"GT item id {target_item} is outside new_all_item_image_paths.npy")
        path = image_root / str(item_paths[target_item])
        good, reason = image_is_readable(path)
        if not good:
            raise ValueError(f"GT image unreadable for task {task_id}: {path}: {reason}")
        resolved[task_id] = {"gt_item_id": target_item, "gt_image_path": str(path.resolve()),
                             "ground_truth_source": str(ground_path.resolve())}
    return resolved


def score_missing_tasks(args: argparse.Namespace, complete: dict[str, list[dict[str, Any]]], selected_tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Score every cached candidate image once, writing each task atomically."""
    evaluator_ckpt = args.dualfashion_root / "eval" / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    features_path = args.data_root / "cnn_features_clip.npy"
    history_path = args.data_root / "processed" / "test_history_clipembs.npy"
    for path in (evaluator_ckpt, features_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen evaluator asset is missing: {path}")
    score_dir = args.output_root / "score_tasks"
    cached: dict[str, list[dict[str, Any]]] = {}
    work: list[dict[str, Any]] = []
    selected_ids = {task["task_id"] for task in selected_tasks}
    for task in selected_tasks:
        specs = complete[task["task_id"]]
        saved = load_score_file(score_dir / f"{task['task_id']}.json", specs) if args.resume else None
        if saved is not None:
            cached[task["task_id"]] = saved
        else:
            work.extend(specs)
    # Include earlier already-scored tasks when compiling output tables.
    for task_id, specs in complete.items():
        if task_id not in selected_ids and args.resume:
            saved = load_score_file(score_dir / f"{task_id}.json", specs)
            if saved is not None:
                cached[task_id] = saved
    if not work:
        return cached
    item_features = np.load(features_path, mmap_mode="r")
    history = np.load(history_path, allow_pickle=True).item()
    device, clip, preprocess, evaluator = load_reward_models(args, evaluator_ckpt)
    pending: dict[str, list[dict[str, Any]]] = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    expected_by_id = {task_id: len(records) for task_id, records in complete.items()}
    for start in tqdm(range(0, len(work), args.batch_size), desc="Candidate reward scoring", unit="image"):
        batch_specs = work[start:start + args.batch_size]
        tensors = []
        for spec in batch_specs:
            with Image.open(spec["image_path"]) as image:
                tensors.append(preprocess(image.convert("RGB")))
        image_batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            candidate_features = clip.encode_image(image_batch).float()
        outfits = []
        histories = []
        history_valid = []
        for spec in batch_specs:
            features = np.zeros((4, item_features.shape[1]), dtype=np.float32)
            for index, item_id in enumerate(spec["outfit"]):
                if item_id != 0:
                    features[index] = item_features[int(item_id)]
            outfits.append(torch.from_numpy(features))
            hist = history_embedding(history, spec["uid"], spec["target_category"])
            histories.append(hist)
            history_valid.append(hist is not None and len(hist) == candidate_features.shape[1])
        outfit_tensor = torch.stack(outfits).to(device)
        for index, spec in enumerate(batch_specs):
            outfit_tensor[index, spec["blank_index"], :] = candidate_features[index]
        with torch.inference_mode():
            logits = evaluator(outfit_tensor).float()
            compatibilities = torch.sigmoid(logits)
        personal_values: list[float | None] = []
        for index, valid in enumerate(history_valid):
            if not valid:
                personal_values.append(None)
                continue
            hist = torch.as_tensor(histories[index], dtype=torch.float32, device=device)
            personal_values.append(float(100.0 * F.cosine_similarity(
                F.normalize(candidate_features[index:index + 1], dim=-1),
                F.normalize(hist.unsqueeze(0), dim=-1), dim=-1
            ).item()))
        for index, spec in enumerate(batch_specs):
            logit = float(logits[index].item())
            compatibility = float(compatibilities[index].item())
            if not math.isfinite(logit) or not math.isfinite(compatibility):
                raise FloatingPointError(f"Non-finite compatibility score for {spec['task_id']} seed {spec['seed']}")
            row = dict(spec)
            row.update({
                "schema": SCORE_SCHEMA, "compatibility_logit": logit,
                "compatibility_sigmoid": compatibility, "personal_sim": personal_values[index],
                "personal_missing": personal_values[index] is None,
                "score_models": {
                    "openclip": "ViT-H-14/laion2b-s32b-b79K",
                    "compatibility_checkpoint": str(evaluator_ckpt.resolve()),
                    "personal_history": str(history_path.resolve()),
                },
            })
            pending[spec["task_id"]].append(row)
            totals[spec["task_id"]] += 1
            if totals[spec["task_id"]] == expected_by_id[spec["task_id"]]:
                rows = sorted(pending[spec["task_id"]], key=lambda r: (r["method"], r["candidate_index"], r["node"]))
                atomic_json(score_dir / f"{spec['task_id']}.json", {"schema": SCORE_SCHEMA, "rows": rows})
                cached[spec["task_id"]] = rows
                del pending[spec["task_id"]]
    if pending:
        raise RuntimeError(f"Interrupted before task-atomic score files were complete: {sorted(pending)}")
    del clip, evaluator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cached


def average_percentiles(values: dict[int, float]) -> dict[int, float]:
    """Average tied ranks mapped to [0, 1], without injecting seed order into a reward."""
    ordered = sorted(values.items(), key=lambda item: item[1])
    if len(ordered) <= 1:
        return {seed: 1.0 for seed, _ in ordered}
    result: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for seed, _ in ordered[start:end]:
            result[seed] = average_rank / (len(ordered) - 1)
        start = end
    return result


def pareto_fronts(candidates: list[dict[str, Any]]) -> dict[int, int]:
    remaining = list(candidates)
    fronts: dict[int, int] = {}
    front = 0
    while remaining:
        current = []
        for candidate in remaining:
            dominated = any(
                other["compatibility_sigmoid"] >= candidate["compatibility_sigmoid"]
                and other["personal_sim"] >= candidate["personal_sim"]
                and (other["compatibility_sigmoid"] > candidate["compatibility_sigmoid"]
                     or other["personal_sim"] > candidate["personal_sim"])
                for other in remaining if other["seed"] != candidate["seed"]
            )
            if not dominated:
                current.append(candidate)
        if not current:
            raise RuntimeError("Pareto front construction did not progress")
        for candidate in current:
            fronts[candidate["seed"]] = front
        remaining = [candidate for candidate in remaining if candidate not in current]
        front += 1
    return fronts


def load_plugin(spec: str | None):
    if not spec or ":" not in spec:
        raise ValueError("--reward plugin requires --plugin module:function")
    module_name, function_name = spec.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Plugin is not callable: {spec}")
    return function


def rank_active(candidates: list[dict[str, Any]], reward: str, plugin) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if any(row.get("personal_sim") is None for row in candidates) and reward in {"personal", "bottleneck", "pareto"}:
        raise ValueError("Personal Sim is missing for an active candidate; inspect score tables/missing report")
    c_values = {row["seed"]: float(row["compatibility_sigmoid"]) for row in candidates}
    p_values = {row["seed"]: float(row["personal_sim"]) for row in candidates if row.get("personal_sim") is not None}
    c_pct = average_percentiles(c_values)
    p_pct = average_percentiles(p_values) if p_values else {}
    details: dict[str, Any] = {"formula": None, "scores": {}}
    if reward == "compatibility":
        details["formula"] = "Compatibility = sigmoid(FashionEvaluator(incomplete FITB context with candidate at blank))"
        for row in candidates:
            details["scores"][row["seed"]] = {"reward": c_values[row["seed"]]}
        ranked = sorted(candidates, key=lambda row: (-c_values[row["seed"]], row["seed"]))
    elif reward == "personal":
        details["formula"] = "Personal Sim = 100 * cosine(normalized OpenCLIP(candidate), normalized history[uid][target_category])"
        for row in candidates:
            details["scores"][row["seed"]] = {"reward": p_values[row["seed"]]}
        ranked = sorted(candidates, key=lambda row: (-p_values[row["seed"]], row["seed"]))
    elif reward == "bottleneck":
        details["formula"] = "min(average-tie percentile(Compatibility), average-tie percentile(Personal Sim)) within current active set"
        for row in candidates:
            seed = row["seed"]
            details["scores"][seed] = {"reward": min(c_pct[seed], p_pct[seed]), "compatibility_percentile": c_pct[seed], "personal_percentile": p_pct[seed]}
        ranked = sorted(candidates, key=lambda row: (-details["scores"][row["seed"]]["reward"], row["seed"]))
    elif reward == "pareto":
        fronts = pareto_fronts(candidates)
        details["formula"] = "non-dominated Pareto front on (Compatibility, Personal Sim); tie order: larger bottleneck percentile, Compatibility, Personal Sim, smaller seed"
        for row in candidates:
            seed = row["seed"]
            details["scores"][seed] = {"pareto_front": fronts[seed], "bottleneck_percentile": min(c_pct[seed], p_pct[seed]),
                                       "compatibility_percentile": c_pct[seed], "personal_percentile": p_pct[seed]}
        ranked = sorted(candidates, key=lambda row: (details["scores"][row["seed"]]["pareto_front"],
                         -details["scores"][row["seed"]]["bottleneck_percentile"], -c_values[row["seed"]],
                         -p_values[row["seed"]], row["seed"]))
    else:
        details["formula"] = "user plugin scalar; descending scalar then ascending seed"
        for row in candidates:
            value = plugin({"candidate": dict(row), "active_candidates": [dict(item) for item in candidates]})
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"Plugin returned non-finite score for seed {row['seed']}")
            details["scores"][row["seed"]] = {"reward": value}
        ranked = sorted(candidates, key=lambda row: (-details["scores"][row["seed"]]["reward"], row["seed"]))
    return ranked, details


def final_path_for(task: dict[str, Any], cache_root: Path, method: str, seed: int) -> Path:
    metadata = read_json(cache_root / "metadata" / f"{task['task_dir_name']}.json")
    for candidate in metadata["methods"][method]["candidates"]:
        if int(candidate["seed"]) == seed:
            return (cache_root / candidate["files"]["final_image"]).resolve()
    raise KeyError(f"No {method} candidate seed {seed} in {task['task_id']}")


def make_selection(task: dict[str, Any], rows: list[dict[str, Any]], args: argparse.Namespace, plugin) -> dict[str, Any]:
    by_key = {(row["method"], row["candidate_index"], row["node"]): row for row in rows}
    bon = [by_key[("bon", index, "final")] for index in range(4)]
    bon_ranked, bon_details = rank_active(bon, args.reward, plugin)
    standard = bon[0]
    bon_winner = bon_ranked[0]
    trace = []
    active = [by_key[("psp", index, "reward_step_13")] for index in range(8)]
    for step, keep, node in PSP_SCHEDULE:
        if node != "reward_step_13":
            active = [by_key[("psp", row["candidate_index"], node)] for row in active]
        before = list(active)
        ranked, details = rank_active(before, args.reward, plugin)
        active = ranked[:keep]
        trace.append({
            "checkpoint": step, "node": node, "keep": keep, "active_before_seeds": [row["seed"] for row in before],
            "ranked_seeds": [row["seed"] for row in ranked], "active_after_seeds": [row["seed"] for row in active],
            "reward": details, "selection_rng": "none; deterministic sort",
            "source": (
                "decoded x_prediction reward image recorded by DiFashion at this loop checkpoint"
                if node.startswith("reward_step_") else "terminal scheduler decode used for final selection and evaluation"
            ),
            "candidate_node_metadata": [{
                "seed": row["seed"], "candidate_index": row["candidate_index"],
                "loop_index_zero_based": row.get("loop_index_zero_based"),
                "scheduler_timestep": row.get("scheduler_timestep"),
                "x_prediction_saved_as_decoded_image": row.get("x_prediction_saved_as_decoded_image"),
                "continuation_latent_available_in_cache": row.get("continuation_latent_available_in_cache"),
                "continuation_latent_used_for_replay": False,
                "reward_image_path": row["image_path"],
            } for row in before],
        })
    psp_winner = active[0]
    psp_final = final_path_for(task, args.cache_root, "psp", int(psp_winner["seed"]))
    bon_final = final_path_for(task, args.cache_root, "bon", int(bon_winner["seed"]))
    standard_final = final_path_for(task, args.cache_root, "bon", int(standard["seed"]))
    for path in (psp_final, bon_final, standard_final):
        good, reason = image_is_readable(path)
        if not good:
            raise ValueError(f"Selected final image is unreadable: {path}: {reason}")
    return {
        "schema": SELECTION_SCHEMA, "task_index": task["task_index"], "task_id": task["task_id"], "uid": task["uid"], "oid": task["oid"],
        "group_id": args.group_id, "reward_name": args.reward, "plugin": args.plugin,
        "condition": {"outfit": task["outfit"], "blank_index": task["blank_index"], "target_category": task["target_category"],
                      "gt_item_used": False},
        "tie_break": "descending reward; then ascending seed", "selection_rng": "none; deterministic sort",
        "standard": {"definition": "BoN independent pool candidate_index=0; no reward selection",
                     "seed": standard["seed"], "candidate_index": standard["candidate_index"],
                     "final_image_path": str(standard_final)},
        "bon": {"candidate_count": 4, "active_seeds": [row["seed"] for row in bon], "ranked_seeds": [row["seed"] for row in bon_ranked],
                "reward": bon_details, "winner_seed": bon_winner["seed"], "winner_candidate_index": bon_winner["candidate_index"],
                "final_image_path": str(bon_final)},
        "psp": {"candidate_count": 8, "schedule": trace, "winner_seed": psp_winner["seed"],
            "winner_candidate_index": psp_winner["candidate_index"], "final_image_path": str(psp_final),
                "final_image_source": "terminal final scheduler decode, which was also the third PSP selection node; it is not step_51 x_prediction"},
        "budget": {"offline_candidate_collection_unet_calls": {"bon": 204, "psp": 408},
                   "simulated_online_unet_budget": {"bon": 204, "psp": 206}},
    }


def select_tasks(args: argparse.Namespace, complete: dict[str, list[dict[str, Any]]], tasks: list[dict[str, Any]], scores: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], dict[str, Any]]:
    plugin = load_plugin(args.plugin) if args.reward == "plugin" else None
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for task in tqdm(tasks, desc="Offline selection", unit="task"):
        rows = scores.get(task["task_id"])
        if rows is None:
            raise FileNotFoundError(f"No complete candidate scores for {task['task_id']}; run without --select-only first")
        generated = make_selection(task, rows, args, plugin)
        for method in ("standard", "bon", "psp"):
            path = args.output_root / "selections" / method / f"{task['task_id']}.json"
            if path.is_file() and args.resume:
                existing = read_json(path)
                if method == "standard":
                    same = existing.get("standard", {}).get("seed") == generated["standard"]["seed"] and existing.get("reward_name") == args.reward
                else:
                    same = existing.get(method, {}).get("winner_seed") == generated[method]["winner_seed"] and existing.get("reward_name") == args.reward
                if args.check_repeat and not same:
                    raise AssertionError(f"Non-reproducible {method} selection for {task['task_id']}")
                if same:
                    selected[(method, task["task_id"])] = existing
                    continue
            atomic_json(path, generated)
            selected[(method, task["task_id"])] = generated
    return selected


def compile_score_outputs(args: argparse.Namespace, complete: dict[str, list[dict[str, Any]]]) -> None:
    all_rows: list[dict[str, Any]] = []
    for task_id, specs in complete.items():
        rows = load_score_file(args.output_root / "score_tasks" / f"{task_id}.json", specs)
        if rows:
            all_rows.extend(rows)
    all_rows.sort(key=lambda row: (row["task_index"], row["method"], row["candidate_index"], row["node"]))
    atomic_csv(args.output_root / "candidate_scores.csv", all_rows)
    atomic_jsonl(args.output_root / "candidate_scores.jsonl", all_rows)


def compile_selection_index(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for method in ("standard", "bon", "psp"):
            path = args.output_root / "selections" / method / f"{task['task_id']}.json"
            if not path.is_file():
                continue
            selection = read_json(path)
            value = selection[method]
            final = Path(value["final_image_path"])
            good, reason = image_is_readable(final)
            rows.append({"task_index": task["task_index"], "task_id": task["task_id"], "uid": task["uid"], "oid": task["oid"],
                         "method": method, "reward": selection["reward_name"], "group_id": selection["group_id"],
                         "seed": value.get("winner_seed", value.get("seed")),
                         "candidate_index": value.get("winner_candidate_index", value.get("candidate_index")),
                         "final_image_path": str(final), "image_readable": good, "image_error": reason})
    rows.sort(key=lambda row: (row["method"], row["task_index"]))
    atomic_csv(args.output_root / "selected_images_index.csv", rows)
    return rows


def score_final_images(args: argparse.Namespace, selection_rows: list[dict[str, Any]], task_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Frozen C/P and GT OpenCLIP metrics on terminal selected images, after selection."""
    evaluator_ckpt = args.dualfashion_root / "eval" / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    gt_items = load_gt_items(args, task_by_id)
    device, clip, preprocess, evaluator = load_reward_models(args, evaluator_ckpt)
    results = []
    for start in tqdm(range(0, len(selection_rows), args.batch_size), desc="Final C/P evaluation", unit="image"):
        group = selection_rows[start:start + args.batch_size]
        tensors = []
        gt_tensors = []
        for row in group:
            with Image.open(row["final_image_path"]) as image:
                tensors.append(preprocess(image.convert("RGB")))
            with Image.open(gt_items[row["task_id"]]["gt_image_path"]) as image:
                gt_tensors.append(preprocess(image.convert("RGB")))
        with torch.inference_mode():
            features = clip.encode_image(torch.stack(tensors).to(device)).float()
            gt_features = clip.encode_image(torch.stack(gt_tensors).to(device)).float()
        contexts = []
        histories = []
        for row in group:
            task = task_by_id[row["task_id"]]
            context = np.zeros((4, item_features.shape[1]), dtype=np.float32)
            for index, item_id in enumerate(task["outfit"]):
                if item_id:
                    context[index] = item_features[int(item_id)]
            contexts.append(torch.from_numpy(context))
            histories.append(history_embedding(history, task["uid"], task["target_category"]))
        context_tensor = torch.stack(contexts).to(device)
        for index, row in enumerate(group):
            context_tensor[index, task_by_id[row["task_id"]]["blank_index"], :] = features[index]
        with torch.inference_mode():
            logits = evaluator(context_tensor).float()
            compatibility = torch.sigmoid(logits)
        for index, row in enumerate(group):
            hist = histories[index]
            personal = None
            if hist is not None and len(hist) == features.shape[1]:
                hist_tensor = torch.as_tensor(hist, dtype=torch.float32, device=device)
                personal = float(100.0 * F.cosine_similarity(F.normalize(features[index:index + 1], dim=-1),
                                   F.normalize(hist_tensor.unsqueeze(0), dim=-1), dim=-1).item())
            results.append({**row, "compatibility_logit": float(logits[index].item()),
                            "compatibility_sigmoid": float(compatibility[index].item()), "personal_sim": personal,
                            "personal_missing": personal is None,
                            "gt_sim": float(100.0 * F.cosine_similarity(
                                F.normalize(features[index:index + 1], dim=-1),
                                F.normalize(gt_features[index:index + 1], dim=-1), dim=-1).item()),
                            **gt_items[row["task_id"]]})
    del clip, evaluator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def gt_lpips_metrics(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], float], dict[str, Any]]:
    """Post-selection LPIPS-VGG against the GT item, matching DualFashion tensor scaling."""
    try:
        import lpips
        device = torch.device(args.device)
        model = lpips.LPIPS(net="vgg").to(device).eval()
    except Exception as exc:
        return {}, {"status": "not_run", "reason": f"LPIPS-VGG unavailable: {exc!r}"}
    values: dict[tuple[str, str], float] = {}
    try:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="GT LPIPS evaluation", unit="image"):
            group = rows[start:start + args.batch_size]
            generated, ground = [], []
            for row in group:
                with Image.open(row["final_image_path"]) as image:
                    generated.append(torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1))
                with Image.open(row["gt_image_path"]) as image:
                    ground.append(torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1))
            with torch.inference_mode():
                scores = model(torch.stack(generated).to(device), torch.stack(ground).to(device)).reshape(-1)
            for row, score in zip(group, scores):
                value = float(score.item())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Non-finite GT LPIPS for {row['method']} {row['task_id']}")
                values[(row["method"], row["task_id"])] = value
    except Exception as exc:
        return {}, {"status": "not_run", "reason": f"GT LPIPS evaluation failed: {exc!r}"}
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return values, {"status": "run", "implementation": "DualFashion eval_utils.calculate_lpips_given_data semantics",
                     "formula": "LPIPS-VGG(gen[-1,1], GT[-1,1])", "unit": "distance"}


def inception_metrics(args: argparse.Namespace, rows: list[dict[str, Any]], task_by_id: dict[str, dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    """Use the exact frozen Inception architecture/formula from DualFashion eval_utils."""
    model_path = args.dualfashion_root / "eval" / "finetuned-inception" / "Inception-finetune-epoch300.zip"
    labels_path = args.dualfashion_root / "eval" / "finetuned-inception" / "cid_to_label.npy"
    if args.skip_inception or not model_path.is_file() or not labels_path.is_file():
        return {}, {"status": "not_run", "reason": "--skip-inception or frozen Inception assets unavailable"}
    dual_root = str(args.dualfashion_root.resolve())
    if dual_root not in sys.path:
        sys.path.insert(0, dual_root)
    from eval_utils import InceptionV3
    device = torch.device(args.device)
    labels = np.load(labels_path, allow_pickle=True).item()
    model = InceptionV3(str(model_path), 50, map_location=torch.device("cpu")).to(device).eval()
    individual: dict[tuple[str, str], dict[str, Any]] = {}
    epsilon = 1e-16
    uniform = torch.ones(50, device=device) / 50.0
    for start in tqdm(range(0, len(rows), args.batch_size), desc="Frozen Inception evaluation", unit="image"):
        group = rows[start:start + args.batch_size]
        images = []
        targets = []
        for row in group:
            with Image.open(row["final_image_path"]) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
            category = task_by_id[row["task_id"]]["target_category"]
            targets.append(int(labels[category]))
        batch = torch.stack(images).to(device)
        batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        batch = 2 * batch - 1
        with torch.inference_mode():
            probabilities = model(batch)
        top3 = torch.topk(probabilities, k=3, dim=1).indices
        entropy = -(probabilities * torch.log(probabilities + epsilon)).sum(dim=1)
        kl_uniform = (probabilities * (torch.log(probabilities + epsilon) - torch.log(uniform.unsqueeze(0)))).sum(dim=1)
        for index, row in enumerate(group):
            individual[(row["method"], row["task_id"])] = {
                "is_acc_top3": float(int(targets[index] in top3[index].tolist())),
                "is_entropy": float(entropy[index].item()), "is_kl_uniform": float(kl_uniform[index].item()),
            }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return individual, {"status": "run", "model": str(model_path), "cid_to_label": str(labels_path),
                         "formula": "DualFashion calculate_inception_score_given_data: mean entropy and exp(mean KL(pred || uniform_50)); IS-Acc is top-3 category accuracy"}


def sample_std(values: list[float]) -> float | None:
    return float(np.std(np.asarray(values), ddof=1)) if len(values) >= 2 else None


def evaluate_and_summarize(args: argparse.Namespace, selection_rows: list[dict[str, Any]], tasks: list[dict[str, Any]]) -> None:
    task_by_id = {task["task_id"]: task for task in tasks}
    final_rows = score_final_images(args, selection_rows, task_by_id)
    inception, inception_contract = inception_metrics(args, final_rows, task_by_id)
    gt_lpips, gt_lpips_contract = gt_lpips_metrics(args, final_rows)
    task_rows = []
    for row in final_rows:
        extra = inception.get((row["method"], row["task_id"]), {})
        gt_lpips_value = gt_lpips.get((row["method"], row["task_id"]))
        task_rows.append({**row, **extra, "gt_lpips": gt_lpips_value, "gt_lpips_missing": gt_lpips_value is None})
    atomic_csv(args.output_root / "official_eval" / "task_metrics.csv", task_rows)
    summaries = []
    for method in ("standard", "bon", "psp"):
        rows = [row for row in task_rows if row["method"] == method]
        for metric, direction, unit in (("compatibility_sigmoid", "higher", "probability"), ("personal_sim", "higher", "100*cosine"),
                                        ("gt_sim", "higher", "100*cosine"), ("gt_lpips", "lower", "LPIPS distance"),
                                        ("is_acc_top3", "higher", "fraction"), ("is_entropy", "lower", "nats")):
            values = [float(row[metric]) for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
            summaries.append({"method": method, "reward": args.reward, "group_id": args.group_id, "metric": metric,
                              "direction": direction, "unit": unit, "mean": float(np.mean(values)) if values else None,
                              "std_sample": sample_std(values), "valid_tasks": len(values), "missing_tasks": len(rows) - len(values)})
        kl_values = [float(row["is_kl_uniform"]) for row in rows if row.get("is_kl_uniform") is not None]
        summaries.append({"method": method, "reward": args.reward, "group_id": args.group_id, "metric": "IS",
                          "direction": "higher", "unit": "exp(mean KL to uniform_50)", "mean": float(math.exp(np.mean(kl_values))) if kl_values else None,
                          "std_sample": None, "valid_tasks": len(kl_values), "missing_tasks": len(rows) - len(kl_values)})
        summaries.append({"method": method, "reward": args.reward, "group_id": args.group_id, "metric": "LPIP score",
                          "direction": "higher", "unit": "LPIPS distance", "mean": None, "std_sample": None,
                          "valid_tasks": 0, "missing_tasks": len(rows), "status": "not_applicable: one selected image per FITB task has no within-task pair"})
    atomic_csv(args.output_root / "official_eval" / "group_summary.csv", summaries)
    # A one-group aggregate is kept in the same schema; a multi-group aggregator can concatenate these rows later.
    atomic_csv(args.output_root / "official_eval" / "aggregate_summary.csv", summaries)
    metric_contract = {
        "Compatibility": {"implementation": "DualFashion eval.compatibility_evaluator.compatibility_net.FashionEvaluator",
                            "input": "OpenCLIP ViT-H/14 candidate + fixed incomplete FITB context", "formula": "sigmoid(logit)",
                            "unit": "probability", "direction": "higher", "missing": "hard error for candidate scoring"},
        "Personal Sim": {"implementation": "DualFashion eval_utils.evaluate_personalization_given_data_sim semantics",
                           "input": "OpenCLIP ViT-H/14 candidate and test_history_clipembs[uid][target_category]",
                           "formula": "100 * cosine(normalized embeddings)", "unit": "100*cosine", "direction": "higher",
                           "missing": "recorded per row; personal/bottleneck/Pareto selection refuses missing active rows"},
        "IS-Acc": {"implementation": "DualFashion eval_utils.calculate_inception_score_given_data", "input": "selected terminal image and target category after selection",
                      "formula": "top-3 frozen 50-class Inception accuracy", "unit": "fraction", "direction": "higher", "missing": inception_contract},
        "IS": {"implementation": "DualFashion eval_utils.calculate_inception_score_given_data", "input": "all selected terminal images",
               "formula": "exp(mean KL(prediction || uniform 50-class distribution))", "unit": "score", "direction": "higher", "missing": inception_contract},
        "IS-Entropy": {"implementation": "DualFashion eval_utils.calculate_inception_score_given_data", "input": "all selected terminal images",
                       "formula": "mean -sum(p log p)", "unit": "nats", "direction": "lower", "missing": inception_contract},
        "LPIP score": {"implementation": "DualFashion eval_utils.calculate_pairwise_lpips_given_groups", "input": "multiple samples inside one FITB task",
                         "formula": "mean pairwise LPIPS", "unit": "distance", "direction": "higher", "missing": "not applicable to selected-one-image-per-task output"},
        "GT Sim": {"implementation": "DualFashion eval_utils.calculate_clip_img_score_given_data semantics", "input": "selected terminal image and GT target-item image",
                                       "formula": "100 * cosine(normalized OpenCLIP ViT-H/14 embeddings)", "unit": "100*cosine", "direction": "higher",
                                       "usage": "post-selection evaluation only; GT is prohibited from candidate scoring and selection", "missing": "hard error after GT/context consistency checks"},
        "GT LPIPS": {"implementation": "DualFashion eval_utils.calculate_lpips_given_data semantics", "input": "selected terminal image and GT target-item image",
                      "formula": "LPIPS-VGG on RGB tensors scaled to [-1, 1]", "unit": "distance", "direction": "lower",
                      "usage": "post-selection evaluation only; GT is prohibited from candidate scoring and selection", "missing": gt_lpips_contract},
        "Excluded": ["GT Compatibility is reference-only and is not computed", "FID is a distribution statistic and is not added to the per-task GT-similarity table", "retrieval uses GT candidates and is not used here", "text and Gemini metrics require attributes/captions or external service"],
    }
    atomic_json(args.output_root / "official_eval" / "metric_contract.json", metric_contract)


def write_integrity_report(args: argparse.Namespace, expected: list[dict[str, Any]], complete: dict[str, list[dict[str, Any]]], problems: list[dict[str, Any]]) -> None:
    selected_index = args.output_root / "selected_images_index.csv"
    selected_rows = []
    if selected_index.is_file():
        with selected_index.open(newline="", encoding="utf-8") as handle:
            selected_rows = list(csv.DictReader(handle))
    selected_counts = {method: sum(1 for row in selected_rows if row.get("method") == method) for method in ("standard", "bon", "psp")}
    report = {
        "complete_cache_tasks": len(complete), "condition_task_count": len(expected), "cache_problem_count": len(problems),
        "cache_problems": problems, "selection_counts": selected_counts,
        "score_task_files": len(list((args.output_root / "score_tasks").glob("*.json"))),
        "candidate_score_rows_expected_per_complete_task": 28,
        "selection_image_index": str(selected_index),
        "cost_disclosure": {"offline_collection": "BoN@4=4*51=204, PSP full cache=8*51=408 UNet calls/task",
                             "simulated_online_budget": "BoN@4=204; PSP 8*13 + 4*(26-13) + 2*(51-26)=206 UNet calls/task",
                            "claim_limit": "Offline replay is not online dynamic-pruning timing evidence."},
        "exploratory_test_only": True,
    }
    atomic_json(args.output_root / "integrity_report.json", report)
    missing = []
    bad_ids = {row.get("task_id") for row in problems}
    for task in expected:
        if task["task_id"] in bad_ids:
            missing.append({"task_index": task["task_index"], "task_id": task["task_id"], "reason": "cache invalid or missing"})
    atomic_csv(args.output_root / "missing_tasks.csv", missing)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.resume_from < 0:
        raise ValueError("--batch-size must be >= 1 and --resume-from must be >= 0")
    if (args.score_only + args.select_only + args.evaluate_only) > 1:
        raise ValueError("Use at most one of --score-only, --select-only, --evaluate-only")
    expected, cache_config, complete, problems = inspect_cache(args, verify_images=True)
    config = output_config(args, cache_config, len(expected))
    prepare_output(args, config)
    write_integrity_report(args, expected, complete, problems)
    print(json.dumps({"condition_tasks": len(expected), "complete_cache_tasks": len(complete), "cache_problems": len(problems),
                      "output_root": str(args.output_root)}, indent=2))
    if args.audit_only:
        return 0 if not problems else 2
    if problems and not args.allow_partial:
        raise RuntimeError("Cache is incomplete/invalid. See integrity_report.json or use --allow-partial for an explicitly partial run.")
    all_complete_tasks = [task for task in expected if task["task_id"] in complete]
    tasks = [task for task in all_complete_tasks if task["task_index"] >= args.resume_from]
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    if not tasks:
        raise RuntimeError("No complete tasks selected")
    scores: dict[str, list[dict[str, Any]]] = {}
    if not args.select_only and not args.evaluate_only:
        scores = score_missing_tasks(args, complete, tasks)
        compile_score_outputs(args, complete)
    else:
        for task in tasks:
            rows = load_score_file(args.output_root / "score_tasks" / f"{task['task_id']}.json", complete[task["task_id"]])
            if rows is not None:
                scores[task["task_id"]] = rows
    if not args.score_only and not args.evaluate_only:
        select_tasks(args, complete, tasks, scores)
    # Rebuild the index from every completed selection in this run directory.
    # This preserves rows before --resume-from instead of replacing the final
    # report with only the resumed suffix.
    index = compile_selection_index(args, all_complete_tasks)
    if not args.score_only and not args.select_only:
        indexed_ids = {row["task_id"] for row in index}
        if len(index) != 3 * len(indexed_ids):
            raise RuntimeError("Selection index does not contain exactly one Standard, BoN, and PSP row per task")
        if any(str(row["image_readable"]).lower() != "true" for row in index):
            raise RuntimeError("At least one selected image is unreadable; see selected_images_index.csv")
        evaluation_tasks = [task for task in all_complete_tasks if task["task_id"] in indexed_ids]
        evaluate_and_summarize(args, index, evaluation_tasks)
    write_integrity_report(args, expected, complete, problems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
