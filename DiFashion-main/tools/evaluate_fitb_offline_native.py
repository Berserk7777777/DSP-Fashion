#!/usr/bin/env python
"""Offline iFashion FITB selection and evaluation with DiFashion's native evaluators.

This program consumes an immutable DiFashion candidate cache.  Candidate
scoring and BoN/PSP selection never load a GT item, retrieval candidates, or
``test_grd.npy``.  GT is opened only by the final evaluation phase for the
native DiFashion CS/CIS/LPIPS metrics.

The cache contract is deliberately narrow: BoN has four terminal images and
PSP has eight full trajectories with decoded x_predictions at steps 13/26 and
a terminal scheduler decode at step 51.  PSP replay chooses 8 -> 4 -> 2 -> 1;
it is an offline selection simulation, not a timing claim about online
dynamic pruning.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


SCORE_SCHEMA = 1
SELECTION_SCHEMA = 1
EVALUATION_SCHEMA = 1
PSP_NODES = (("reward_step_13", 4), ("reward_step_26", 2), ("final", 1))
SPECIAL_CATES = {"shoes", "pants", "sneakers", "boots", "earrings", "slippers", "sandals"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="iFashion root containing processed/, images/, and test_grd.npy")
    parser.add_argument("--evaluation-root", type=Path, required=True,
                        help="DiFashion/Evaluation directory")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, default=None,
                        help="Existing/shared candidate-score directory. Defaults to <output-root>/candidate_scores.")
    parser.add_argument("--reward", choices=("compatibility", "personal", "bottleneck"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("all", "score", "select", "evaluate", "audit"), default="all")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Only process condition rows whose zero-based index is at least this value.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Small, non-final smoke run limit.")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def readable_image(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:  # pragma: no cover - diagnostics
        return False, repr(exc)


def task_rows(data_root: Path) -> list[dict[str, Any]]:
    source = data_root / "processed" / "new_fitb_test.npy"
    if not source.is_file():
        raise FileNotFoundError(f"Missing FITB condition file: {source}")
    data = np.load(source, allow_pickle=True).item()
    required = {"uids", "oids", "outfits", "category"}
    absent = required.difference(data)
    if absent:
        raise ValueError(f"{source} is missing keys {sorted(absent)}")
    result = []
    seen_dirs: set[str] = set()
    for index, (uid, oid, outfit, category) in enumerate(zip(data["uids"], data["oids"], data["outfits"], data["category"])):
        outfit = np.asarray(outfit, dtype=np.int64).reshape(-1)
        category = np.asarray(category, dtype=np.int64).reshape(-1)
        blanks = np.flatnonzero(outfit == 0)
        if len(outfit) != 4 or len(category) != 4 or len(blanks) != 1:
            raise ValueError(f"FITB row {index} is not a four-item condition with exactly one blank")
        directory = f"{int(uid)}_{int(oid)}"
        if directory in seen_dirs:
            raise ValueError(f"Non-unique uid_oid task directory: {directory}")
        seen_dirs.add(directory)
        result.append({"task_index": index, "task_id": f"{index:05d}_{directory}", "task_dir": directory,
                       "uid": int(uid), "oid": int(oid), "outfit": outfit.tolist(), "category": category.tolist(),
                       "blank_index": int(blanks[0]), "target_category": int(category[int(blanks[0])])})
    return result


def image_path(cache_root: Path, relative: str | None, label: str) -> Path:
    if not relative:
        raise ValueError(f"Missing image path for {label}")
    path = cache_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached image for {label}: {path}")
    valid, error = readable_image(path)
    if not valid:
        raise ValueError(f"Unreadable cached image for {label}: {path}: {error}")
    return path.resolve()


def cached_records(task: dict[str, Any], cache_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_path = cache_root / "metadata" / f"{task['task_dir']}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    if metadata.get("status") != "complete":
        raise ValueError(f"{task['task_id']}: cache status is {metadata.get('status')!r}")
    for key in ("task_index", "task_id", "uid", "oid"):
        expected = task[key]
        actual = metadata.get(key)
        if key == "task_id":
            # Cache task_id uses its documented zero-padded condition index.
            actual = actual
        if actual != expected:
            raise ValueError(f"{task['task_id']}: metadata {key} mismatch: {actual!r} != {expected!r}")
    methods = metadata.get("methods", {})
    result: list[dict[str, Any]] = []
    for method, expected_count, expected_base in (("bon", 4, int(config["bon_seed_base"])), ("psp", 8, int(config["psp_seed_base"]))):
        candidates = methods.get(method, {}).get("candidates", [])
        if len(candidates) != expected_count:
            raise ValueError(f"{task['task_id']}: {method} has {len(candidates)}, expected {expected_count} candidates")
        for candidate in candidates:
            index = candidate.get("candidate_index")
            seed = candidate.get("seed")
            expected_seed = expected_base + task["task_index"] * int(config["seed_stride"]) + int(index)
            if seed != expected_seed:
                raise ValueError(f"{task['task_id']}: {method} seed {seed} != expected {expected_seed}")
            files = candidate.get("files", {})
            if method == "bon":
                result.append({**task, "method": method, "candidate_index": index, "seed": seed, "node": "final",
                               "image_path": str(image_path(cache_root, files.get("final_image"), f"{task['task_id']} bon {seed}"))})
                continue
            steps = files.get("steps", candidate.get("steps", {}))
            for step, node in ((13, "reward_step_13"), (26, "reward_step_26")):
                step_data = steps.get(str(step), {})
                if step_data.get("step_number_one_based") != step or step_data.get("loop_index_zero_based") != step - 1:
                    raise ValueError(f"{task['task_id']}: invalid PSP metadata at step {step}, seed {seed}")
                result.append({**task, "method": method, "candidate_index": index, "seed": seed, "node": node,
                               "image_path": str(image_path(cache_root, step_data.get("decoded_reward_image"), f"{task['task_id']} psp {step} {seed}")),
                               "scheduler_timestep": step_data.get("scheduler_timestep"),
                               "x_prediction_saved_as_decoded_image": True})
            # Terminal selection must use the scheduler final decode, not step_51 x_prediction.
            result.append({**task, "method": method, "candidate_index": index, "seed": seed, "node": "final",
                           "image_path": str(image_path(cache_root, files.get("final_image"), f"{task['task_id']} psp final {seed}")),
                           "scheduler_timestep": 1, "terminal_scheduler_decode": True})
    expected = 4 + 8 * 3
    if len(result) != expected:
        raise AssertionError(f"{task['task_id']}: found {len(result)} images, expected {expected}")
    return result


def history_embedding(history: dict[Any, Any], uid: int, category: int) -> np.ndarray | None:
    user = history.get(uid, history.get(str(uid)))
    if not isinstance(user, dict):
        return None
    value = user.get(category, user.get(str(category)))
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array if array.size else None


def percentile(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values.items(), key=lambda pair: pair[1])
    if len(ordered) <= 1:
        return {seed: 1.0 for seed, _ in ordered}
    output: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + end - 1) / 2.0 / (len(ordered) - 1)
        output.update({seed: rank for seed, _ in ordered[start:end]})
        start = end
    return output


def rank(candidates: list[dict[str, Any]], reward: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compatibility = {row["seed"]: float(row["compatibility"]) for row in candidates}
    personal = {row["seed"]: float(row["personal"]) for row in candidates}
    if not all(math.isfinite(value) for value in compatibility.values()) or not all(math.isfinite(value) for value in personal.values()):
        raise FloatingPointError("Non-finite candidate score")
    if reward == "compatibility":
        values = compatibility
        formula = "DiFashion Compatibility sigmoid"
    elif reward == "personal":
        values = personal
        formula = "DiFashion Personal Sim = 100 * cosine(OpenCLIP image, history embedding)"
    else:
        cp, pp = percentile(compatibility), percentile(personal)
        values = {seed: min(cp[seed], pp[seed]) for seed in compatibility}
        formula = "min(within-active-set average-tie percentile(Compatibility), percentile(Personal Sim))"
    ordered = sorted(candidates, key=lambda row: (-values[row["seed"]], int(row["seed"])))
    return ordered, {"formula": formula, "scores": {str(seed): float(value) for seed, value in values.items()},
                     "tie_break": "descending reward, then ascending integer seed"}


def load_models(args: argparse.Namespace):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    evaluation_root = str(args.evaluation_root.resolve())
    if evaluation_root not in sys.path:
        sys.path.insert(0, evaluation_root)
    import open_clip
    from compatibility_evaluator.compatibility_net import FashionEvaluator
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable for requested device {args.device}")
    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b-s32b-b79K")
    clip = clip.to(device).eval()
    evaluator_path = args.evaluation_root / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"Missing DiFashion compatibility evaluator: {evaluator_path}")
    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(evaluator_path, map_location="cpu"))
    evaluator = evaluator.to(device).eval()
    return device, clip, preprocess, open_clip.get_tokenizer("ViT-H-14"), evaluator, evaluator_path


def load_image_batch(rows: list[dict[str, Any]], preprocess, device: torch.device) -> torch.Tensor:
    images = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            images.append(preprocess(image.convert("RGB")))
    return torch.stack(images).to(device)


def score_directory(args: argparse.Namespace) -> Path:
    return args.score_root.resolve() if args.score_root is not None else args.output_root / "candidate_scores"


def load_contexts(rows: list[dict[str, Any]], item_features: np.ndarray, device: torch.device) -> torch.Tensor:
    contexts = []
    for row in rows:
        context = np.zeros((4, item_features.shape[1]), dtype=np.float32)
        for index, item_id in enumerate(row["outfit"]):
            if item_id:
                context[index] = item_features[int(item_id)]
        contexts.append(torch.from_numpy(context))
    return torch.stack(contexts).to(device)


def score_tasks(args: argparse.Namespace, tasks: list[dict[str, Any]], config: dict[str, Any]) -> None:
    history_path = args.data_root / "processed" / "test_history_clipembs.npy"
    feature_path = args.data_root / "cnn_features_clip.npy"
    for required in (history_path, feature_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing DiFashion evaluation asset: {required}")
    history = np.load(history_path, allow_pickle=True).item()
    item_features = np.load(feature_path, mmap_mode="r")
    device, clip, preprocess, _, evaluator, evaluator_path = load_models(args)
    score_root = score_directory(args)
    for task in tqdm(tasks, desc="Native candidate scoring", unit="task"):
        target = score_root / f"{task['task_id']}.json"
        if args.resume and target.is_file():
            try:
                saved = read_json(target)
                if saved.get("schema") == SCORE_SCHEMA and len(saved.get("rows", [])) == 28:
                    continue
            except Exception:
                pass
        records = cached_records(task, args.cache_root, config)
        scored: list[dict[str, Any]] = []
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            with torch.inference_mode():
                features = clip.encode_image(load_image_batch(batch, preprocess, device)).float()
                contexts = load_contexts(batch, item_features, device)
                for index, record in enumerate(batch):
                    contexts[index, record["blank_index"], :] = features[index]
                compatibilities = torch.sigmoid(evaluator(contexts).float())
            for index, record in enumerate(batch):
                history_value = history_embedding(history, record["uid"], record["target_category"])
                if history_value is None or history_value.size != features.shape[1]:
                    raise ValueError(f"No compatible history embedding for {record['task_id']} category {record['target_category']}")
                history_tensor = torch.as_tensor(history_value, dtype=torch.float32, device=device).unsqueeze(0)
                personal = 100.0 * F.cosine_similarity(F.normalize(features[index:index + 1], dim=-1),
                                                       F.normalize(history_tensor, dim=-1), dim=-1).item()
                scored.append({**record, "compatibility": float(compatibilities[index].item()), "personal": float(personal)})
        scored.sort(key=lambda row: (row["method"], row["candidate_index"], row["node"]))
        atomic_json(target, {"schema": SCORE_SCHEMA, "task_id": task["task_id"], "rows": scored,
                             "models": {"openclip": "ViT-H-14/laion2b-s32b-b79K",
                                        "compatibility_checkpoint": str(evaluator_path.resolve()),
                                        "history": str(history_path.resolve())}})
    del clip, evaluator
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_scores(score_root: Path, task: dict[str, Any]) -> list[dict[str, Any]]:
    path = score_root / f"{task['task_id']}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No candidate scores for {task['task_id']}: run --stage score first")
    payload = read_json(path)
    rows = payload.get("rows", [])
    if payload.get("schema") != SCORE_SCHEMA or len(rows) != 28:
        raise ValueError(f"Invalid candidate score record: {path}")
    return rows


def choose(task: dict[str, Any], scores: list[dict[str, Any]], reward: str) -> dict[str, Any]:
    bon = [row for row in scores if row["method"] == "bon" and row["node"] == "final"]
    psp = [row for row in scores if row["method"] == "psp"]
    if len(bon) != 4 or len(psp) != 24:
        raise ValueError(f"Invalid candidate score composition for {task['task_id']}")
    standard = next(row for row in bon if row["candidate_index"] == 0)
    bon_order, bon_trace = rank(bon, reward)
    active = sorted({row["seed"] for row in psp})
    trace = []
    winner = None
    for node, keep in PSP_NODES:
        candidates = [row for row in psp if row["node"] == node and row["seed"] in active]
        if len(candidates) != len(active):
            raise ValueError(f"{task['task_id']}: missing PSP scores at {node}")
        ordering, details = rank(candidates, reward)
        active = [row["seed"] for row in ordering[:keep]]
        trace.append({"node": node, "keep": keep, "ranked_seeds": [row["seed"] for row in ordering],
                      "kept_seeds": active, "reward": details})
        winner = ordering[0]
    assert winner is not None
    return {"schema": SELECTION_SCHEMA, "task_index": task["task_index"], "task_id": task["task_id"],
            "uid": task["uid"], "oid": task["oid"], "reward": reward,
            "selection_rng": "none", "condition": {"outfit": task["outfit"], "blank_index": task["blank_index"],
                                                        "target_category": task["target_category"], "gt_item_used": False},
            "standard": standard, "bon": {"winner": bon_order[0], "ranked_seeds": [row["seed"] for row in bon_order], "reward": bon_trace},
            "psp": {"winner": winner, "trace": trace,
                    "terminal_image_definition": "PSP final scheduler decode, not step_51 x_prediction"}}


def select_tasks(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
    root = args.output_root / "selections"
    for task in tqdm(tasks, desc="Offline BoN/PSP selection", unit="task"):
        path = root / f"{task['task_id']}.json"
        if args.resume and path.is_file():
            try:
                saved = read_json(path)
                if saved.get("schema") == SELECTION_SCHEMA and saved.get("reward") == args.reward:
                    continue
            except Exception:
                pass
        atomic_json(path, choose(task, load_scores(score_directory(args), task), args.reward))


def category_prompt(category: int, categories: dict[Any, Any]) -> str:
    name = str(categories.get(category, categories.get(str(category))))
    article = "a pair of" if any(special in name for special in SPECIAL_CATES) else "a"
    return f"A photo of {article} {name}, on white background"


def gt_records(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ground_path = args.data_root / "test_grd.npy"
    paths_path = args.data_root / "new_all_item_image_paths.npy"
    image_root = args.data_root / "images"
    for required in (ground_path, paths_path, image_root):
        if not required.exists():
            raise FileNotFoundError(f"Missing post-selection GT asset: {required}")
    ground = np.load(ground_path, allow_pickle=True).item()
    paths = np.load(paths_path, allow_pickle=True)
    output: dict[str, dict[str, Any]] = {}
    for task in tasks:
        entry = ground.get(task["oid"], ground.get(str(task["oid"])))
        if entry is None:
            raise KeyError(f"No GT record for {task['task_id']}")
        outfit = np.asarray(entry["outfits"], dtype=np.int64).reshape(-1)
        category = np.asarray(entry["category"], dtype=np.int64).reshape(-1)
        blank = task["blank_index"]
        if len(outfit) != 4 or len(category) != 4 or category[blank] != task["target_category"]:
            raise ValueError(f"GT layout/category mismatch for {task['task_id']}")
        for index, item_id in enumerate(task["outfit"]):
            if index != blank and outfit[index] != item_id:
                raise ValueError(f"GT context mismatch for {task['task_id']} at index {index}")
        gt_item = int(outfit[blank])
        gt_path = image_root / str(paths[gt_item])
        valid, error = readable_image(gt_path)
        if not valid:
            raise ValueError(f"Unreadable GT image {gt_path}: {error}")
        output[task["task_id"]] = {"gt_item_id": gt_item, "gt_image_path": str(gt_path.resolve())}
    return output


def pil_lpips_tensor(path: str, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def evaluate(args: argparse.Namespace, tasks: list[dict[str, Any]]) -> None:
    selections = []
    for task in tasks:
        path = args.output_root / "selections" / f"{task['task_id']}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing selection: {path}")
        selection = read_json(path)
        for method, source in (("standard", selection["standard"]), ("bon", selection["bon"]["winner"]),
                               ("psp", selection["psp"]["winner"])):
            selections.append({**task, "method": method, "reward": args.reward, "seed": source["seed"],
                               "candidate_index": source["candidate_index"], "image_path": source["image_path"]})
    gt = gt_records(args, tasks)  # Explicitly after every selection is frozen.
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    categories = np.load(args.data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
    device, clip, preprocess, tokenizer, evaluator, evaluator_path = load_models(args)
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("DiFashion's LPIPS dependency is required for native LPIPS evaluation") from exc
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
    rows: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(selections), args.batch_size), desc="Native CS/CIS/LPIPS evaluation", unit="image"):
        batch = selections[start:start + args.batch_size]
        generated = load_image_batch(batch, preprocess, device)
        gt_batch = load_image_batch([{"image_path": gt[row["task_id"]]["gt_image_path"]} for row in batch], preprocess, device)
        texts = tokenizer([category_prompt(row["target_category"], categories) for row in batch]).to(device)
        with torch.inference_mode():
            # Native CompatibilityEvaluator feeds raw OpenCLIP image features
            # into FashionEvaluator; only cosine-based metrics normalize.
            raw_features = clip.encode_image(generated).float()
            raw_gt_features = clip.encode_image(gt_batch).float()
            features = F.normalize(raw_features, dim=-1)
            gt_features = F.normalize(raw_gt_features, dim=-1)
            text_features = F.normalize(clip.encode_text(texts).float(), dim=-1)
            contexts = load_contexts(batch, item_features, device)
            for index, row in enumerate(batch):
                contexts[index, row["blank_index"], :] = raw_features[index]
            compatibility = torch.sigmoid(evaluator(contexts).float())
            lpips_values = lpips_model(torch.stack([pil_lpips_tensor(row["image_path"], device) for row in batch]),
                                       torch.stack([pil_lpips_tensor(gt[row["task_id"]]["gt_image_path"], device) for row in batch])).reshape(-1)
        for index, row in enumerate(batch):
            history_value = history_embedding(history, row["uid"], row["target_category"])
            if history_value is None or history_value.size != features.shape[1]:
                raise ValueError(f"No compatible history embedding for final evaluation {row['task_id']}")
            history_tensor = F.normalize(torch.as_tensor(history_value, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
            rows.append({**row, **gt[row["task_id"]], "cs": float(100.0 * F.cosine_similarity(features[index:index + 1], text_features[index:index + 1]).item()),
                         "cis": float(100.0 * F.cosine_similarity(features[index:index + 1], gt_features[index:index + 1]).item()),
                         "lpips": float(lpips_values[index].item()), "compatibility": float(compatibility[index].item()),
                         "personal": float(100.0 * F.cosine_similarity(features[index:index + 1], history_tensor).item())})
    del clip, evaluator, lpips_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rows.sort(key=lambda row: (row["method"], row["task_index"]))
    atomic_csv(args.output_root / "evaluation" / "task_metrics.csv", rows)
    summary: list[dict[str, Any]] = []
    for method in ("standard", "bon", "psp"):
        method_rows = [row for row in rows if row["method"] == method]
        for metric, direction, unit in (("cs", "higher", "100*cosine"), ("cis", "higher", "100*cosine"),
                                        ("lpips", "lower", "LPIPS-VGG distance"), ("compatibility", "higher", "probability"),
                                        ("personal", "higher", "100*cosine")):
            values = [float(row[metric]) for row in method_rows]
            summary.append({"method": method, "reward": args.reward, "metric": metric, "direction": direction, "unit": unit,
                            "mean": float(np.mean(values)), "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                            "valid_tasks": len(values), "missing_tasks": 0})
    atomic_csv(args.output_root / "evaluation" / "summary.csv", summary)
    atomic_json(args.output_root / "evaluation" / "metric_contract.json", {
        "CS": {"implementation": "DiFashion Evaluation/eval_utils.py calculate_clip_score_given_data formula", "formula": "100*cosine(OpenCLIP ViT-H/14 image, category prompt)", "direction": "higher", "uses_gt": False},
        "CIS": {"implementation": "DiFashion Evaluation/eval_utils.py calculate_clip_img_score_given_data formula", "formula": "100*cosine(OpenCLIP ViT-H/14 generated image, GT image)", "direction": "higher", "uses_gt": "post-selection only"},
        "LPIPS": {"implementation": "DiFashion Evaluation/eval_utils.py calculate_lpips_given_data formula", "formula": "LPIPS-VGG(gen[-1,1], GT[-1,1])", "direction": "lower", "uses_gt": "post-selection only"},
        "Compatibility": {"implementation": "DiFashion Evaluation/compatibility_evaluator FashionEvaluator", "checkpoint": str(evaluator_path.resolve()), "formula": "sigmoid(logit)", "direction": "higher", "uses_gt": False},
        "Personal": {"implementation": "DiFashion Evaluation/eval_utils.py evaluate_personalization_given_data_sim formula", "formula": "100*cosine(OpenCLIP generated image, test history embedding)", "direction": "higher", "uses_gt": False},
        "selection_policy": "GT/test_grd/retrieval candidates are not loaded during candidate scoring or selection. This cache is the test split, so comparison across reward choices is exploratory unless reward/schedule were frozen on validation first.",
        "psp_semantics": "offline replay from fully collected trajectories; final selection uses terminal scheduler decode rather than step_51 x_prediction; not an online compute-speed claim."
    })


def output_config(args: argparse.Namespace, cache_config: dict[str, Any], count: int) -> dict[str, Any]:
    source = args.evaluation_root / "eval_utils.py"
    return {"schema": 1, "cache_root": str(args.cache_root.resolve()), "data_root": str(args.data_root.resolve()),
            "evaluation_root": str(args.evaluation_root.resolve()), "output_reward": args.reward, "task_count_from_condition": count,
            "score_root": str(score_directory(args).resolve()),
            "cache_steps": cache_config.get("psp_prune_steps"), "selection": "BoN@4 terminal; PSP@8 13:8->4, 26:4->2, final(51):2->1",
            "cache_collection_cost": cache_config.get("offline_cost"), "simulated_online_budget": cache_config.get("online_simulation_budget"),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None,
            "test_result_status": "exploratory: reward and schedule are being compared on the test cache"}


def prepare_output(args: argparse.Namespace, config: dict[str, Any]) -> None:
    config_path = args.output_root / "run_config.json"
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not config_path.is_file():
            raise FileExistsError(f"Non-empty output root without run_config.json: {args.output_root}")
        if read_json(config_path) != config:
            raise ValueError("Output configuration differs. Choose a separate output root.")
        if not args.resume and args.stage != "audit":
            raise FileExistsError("Output exists; use --resume or choose a new output root")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, config)


def audit(args: argparse.Namespace, tasks: list[dict[str, Any]], config: dict[str, Any]) -> None:
    complete, failures = [], []
    for task in tqdm(tasks, desc="Cache integrity audit", unit="task"):
        try:
            records = cached_records(task, args.cache_root, config)
            seed_sets = {method: {row["seed"] for row in records if row["method"] == method} for method in ("bon", "psp")}
            if seed_sets["bon"].intersection(seed_sets["psp"]):
                raise ValueError("BoN and PSP seed pools overlap")
            complete.append(task["task_id"])
        except Exception as exc:
            failures.append({"task_index": task["task_index"], "task_id": task["task_id"], "error": repr(exc)})
    atomic_json(args.output_root / "integrity_report.json", {"total_tasks": len(tasks), "complete_tasks": len(complete),
                                                                "failed_tasks": len(failures), "failures": failures})
    atomic_csv(args.output_root / "missing_or_invalid_tasks.csv", failures)
    if failures:
        raise RuntimeError(f"Cache audit failed for {len(failures)} tasks; inspect missing_or_invalid_tasks.csv")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    config_path = args.cache_root / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing cache config: {config_path}")
    cache_config = read_json(config_path)
    if list(map(int, cache_config.get("psp_prune_steps", []))) != [13, 26, 51]:
        raise ValueError(f"Expected cache PSP steps [13, 26, 51], found {cache_config.get('psp_prune_steps')}")
    all_tasks = task_rows(args.data_root)
    selected = [task for task in all_tasks if task["task_index"] >= args.resume_from]
    if args.max_tasks is not None:
        selected = selected[:args.max_tasks]
    prepare_output(args, output_config(args, cache_config, len(all_tasks)))
    if args.stage in ("all", "audit"):
        audit(args, selected, cache_config)
    if args.stage == "audit":
        return
    if args.stage in ("all", "score"):
        score_tasks(args, selected, cache_config)
    if args.stage == "score":
        return
    if args.stage in ("all", "select"):
        select_tasks(args, selected)
    if args.stage == "select":
        return
    evaluate(args, selected)


if __name__ == "__main__":
    main()
