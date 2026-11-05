"""Unified offline BoN/PSP evaluation for DiFashion iFashion GOR caches.

The DiFashion collector caches whole four-item GOR trajectories: BoN saves four
terminal candidates and PSP saves eight candidates at steps 13, 26 and 52.
This tool scores those cached trajectories, freezes Standard/BoN/PSP choices,
then (and only then) loads GT for post-hoc GT-Sim and GAA diagnostics.
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
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from fashion_psp.search.psp import select_survivors
from fashion_psp.evaluation.evaluate_dualfashion_gor_offline import (
    ATTRIBUTE_BANKS,
    encode_images,
    load_gaa_model,
)


REWARD = "personal"
PSP_STEPS = ((13, 4), (26, 2), (52, 1))
METHODS = ("standard", "bon", "psp")
SCHEMA = 1


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="D:\\DiFashion\\ifashion (contains processed/ and test_grd.npy).")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="D:\\DiFashion\\ifashion\\images, used only after selection for GT metrics.")
    parser.add_argument("--dualfashion-root", type=Path, default=root / "DualFashion-main")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--with-gt-metrics", action="store_true")
    parser.add_argument("--skip-inception", action="store_true")
    parser.add_argument("--gaa-clip-weights", type=Path,
                        default=Path(r"C:\Users\Administrator\.cache\clip\ViT-L-14.pt"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def readable(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, repr(exc)


def load_conditions(data_root: Path) -> dict[int, dict[str, Any]]:
    source = data_root / "processed" / "new_fitb_test.npy"
    payload = np.load(source, allow_pickle=True).item()
    rows = {}
    for index, (uid, oid, category) in enumerate(zip(payload["uids"], payload["oids"], payload["category"])):
        categories = [int(value) for value in np.asarray(category).reshape(-1)]
        if len(categories) != 4:
            raise ValueError(f"condition row {index}: expected four categories")
        rows[index] = {"task_index": index, "task_id": f"{index:05d}_{int(uid)}_{int(oid)}",
                       "uid": int(uid), "oid": int(oid), "categories": categories}
    return rows


def candidate_paths(meta: dict[str, Any], method: str, seed: int, step: int | None) -> list[Path]:
    candidate = next((row for row in meta["methods"][method]["candidates"] if int(row["seed"]) == seed), None)
    if candidate is None:
        raise KeyError(f"missing {method} seed {seed}")
    files = candidate["files"]
    names = files["final_image_paths"] if step is None else files["steps"][str(step)]["image_paths"]
    paths = [Path(value) for value in names if Path(value).name != "all.png"]
    if len(paths) != 4:
        raise ValueError(f"{meta['task_id']} {method}/{seed}/{step}: expected four item images")
    return paths


def validate_cache(cache_root: Path, conditions: dict[int, dict[str, Any]], max_tasks: int | None) -> tuple[list[dict], list[dict]]:
    config = read_json(cache_root / "run_config.json")
    selected = [int(value) for value in config["selected_task_indices"]]
    if max_tasks is not None:
        selected = selected[:max_tasks]
    problems: list[dict] = []
    tasks: list[dict] = []
    if config.get("task") != "GOR" or config.get("bon_candidates") != 4 or config.get("psp_candidates") != 8:
        raise ValueError("cache is not the expected DiFashion GOR BoN@4/PSP@8 protocol")
    if config.get("psp_reward_steps") != [13, 26] or config.get("psp_terminal_step") != 52:
        raise ValueError("cache does not have the required 13/26/52 PSP protocol")
    for index in selected:
        condition = conditions.get(index)
        if condition is None:
            problems.append({"task_index": index, "error": "condition row missing"})
            continue
        path = cache_root / "metadata" / f"{condition['uid']}_{condition['oid']}.json"
        if not path.is_file():
            problems.append({"task_index": index, "task_id": condition["task_id"], "error": "metadata missing"})
            continue
        try:
            meta = read_json(path)
            if (meta.get("status") != "complete" or meta.get("task") != "GOR" or
                    any(meta.get(key) != condition[key] for key in ("task_index", "task_id", "uid", "oid")) or
                    [int(value) for value in meta["condition"]["category"]] != condition["categories"]):
                raise ValueError("metadata identity or condition mismatch")
            expected = {"bon": (4, None), "psp": (8, (13, 26, 52))}
            for method, (count, steps) in expected.items():
                rows = meta["methods"][method]["candidates"]
                if len(rows) != count:
                    raise ValueError(f"{method}: expected {count} candidates")
                actual = [int(row["seed"]) for row in rows]
                base = int(config[f"{method}_seed_base"])
                desired = [base + index * int(config["seed_stride"]) + position for position in range(count)]
                if actual != desired:
                    raise ValueError(f"{method}: seed mapping mismatch")
                for seed in actual:
                    saved_steps = (52,) if steps is None else steps
                    for step in saved_steps:
                        for image in candidate_paths(meta, method, seed, None if method == "bon" else step):
                            full = cache_root / image
                            good, reason = readable(full)
                            if not good:
                                raise ValueError(f"unreadable {full}: {reason}")
            tasks.append({"condition": condition, "metadata": meta, "metadata_path": str(path.resolve())})
        except Exception as exc:
            problems.append({"task_index": index, "task_id": condition["task_id"], "error": repr(exc)})
    return tasks, problems


def load_models(args: argparse.Namespace, device: torch.device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dual_root = str(args.dualfashion_root.resolve())
    if dual_root not in sys.path:
        sys.path.insert(0, dual_root)
    import open_clip
    from eval.compatibility_evaluator.compatibility_net import FashionEvaluator

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b-s32b-b79K")
    clip = clip.to(device).eval()
    checkpoint = args.dualfashion_root / "eval" / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return clip, preprocess, evaluator.to(device).eval(), open_clip


def history_vector(history: dict, uid: int, category: int) -> np.ndarray:
    user = history.get(uid, history.get(str(uid)))
    value = user.get(category, user.get(str(category))) if user else None
    if value is None:
        raise KeyError(f"missing history embedding uid={uid}, category={category}")
    return np.asarray(value, dtype=np.float32).reshape(-1)


def score_paths(paths_by_seed: dict[int, list[Path]], categories: list[int], uid: int, history: dict,
                clip, preprocess, evaluator, device: torch.device, batch_size: int) -> dict[int, dict]:
    seeds = list(paths_by_seed)
    flat = [path for seed in seeds for path in paths_by_seed[seed]]
    raw = torch.as_tensor(encode_images(clip, preprocess, flat, device, batch_size, normalize=False), device=device)
    candidates = F.normalize(raw.reshape(len(seeds), 4, -1), dim=-1)
    if candidates.shape[-1] != 1024:
        raise RuntimeError(f"OpenCLIP dimension {candidates.shape[-1]} is incompatible with FashionEvaluator")
    histories = torch.as_tensor(np.stack([history_vector(history, uid, category) for category in categories]), device=device)
    histories = F.normalize(histories, dim=-1)
    with torch.inference_mode():
        compatibility = torch.sigmoid(evaluator(candidates)).float().cpu().numpy()
        personal_by_item = 100.0 * (candidates * histories.unsqueeze(0)).sum(dim=-1)
    result = {}
    for offset, seed in enumerate(seeds):
        values = personal_by_item[offset].float().cpu().numpy()
        result[seed] = {"compatibility": float(compatibility[offset]), "personal": float(values.mean()),
                        "personal_by_item": [float(value) for value in values],
                        "image_paths": [str(path.resolve()) for path in paths_by_seed[seed]]}
    return result


def score_task(task: dict, cache_root: Path, history: dict, clip, preprocess, evaluator,
               device: torch.device, batch_size: int) -> dict:
    meta, condition = task["metadata"], task["condition"]
    scores = {"bon": {}, "psp": {}}
    bon_seeds = [int(row["seed"]) for row in meta["methods"]["bon"]["candidates"]]
    scores["bon"]["final"] = score_paths(
        {seed: [cache_root / path for path in candidate_paths(meta, "bon", seed, None)] for seed in bon_seeds},
        condition["categories"], condition["uid"], history, clip, preprocess, evaluator, device, batch_size)
    psp_seeds = [int(row["seed"]) for row in meta["methods"]["psp"]["candidates"]]
    for step, _ in PSP_STEPS:
        scores["psp"][str(step)] = score_paths(
            {seed: [cache_root / path for path in candidate_paths(meta, "psp", seed, step)] for seed in psp_seeds},
            condition["categories"], condition["uid"], history, clip, preprocess, evaluator, device, batch_size)
    return {"schema": SCHEMA, "task_index": condition["task_index"], "task_id": condition["task_id"],
            "uid": condition["uid"], "oid": condition["oid"], "categories": condition["categories"],
            "reward": REWARD, "scores": scores,
            "score_formula": "personal=mean_i 100*cosine(OpenCLIP ViT-H/14(gen_i), history(uid, category_i))",
            "gt_loaded_or_used": False}


def make_selection(score: dict) -> dict:
    bon = score["scores"]["bon"]["final"]
    standard_seed = min(bon)
    bon_seed = select_survivors({seed: item[REWARD] for seed, item in bon.items()}, 1)[0]
    active = sorted(score["scores"]["psp"]["13"])
    trace = []
    for step, keep in PSP_STEPS:
        values = score["scores"]["psp"][str(step)]
        before = list(active)
        reward_scores = {seed: float(values[seed][REWARD]) for seed in active}
        active = select_survivors(reward_scores, keep)
        trace.append({"step": step, "active_before_seeds": before, "active_after_seeds": active,
                      "reward_scores": reward_scores, "tie_break": "descending reward, ascending integer seed"})
    psp_seed = active[0]
    return {"schema": SCHEMA, "task_index": score["task_index"], "task_id": score["task_id"], "uid": score["uid"],
            "oid": score["oid"], "categories": score["categories"], "reward": REWARD,
            "ground_truth_loaded_or_used": False, "standard": {"seed": standard_seed, "metrics": bon[standard_seed]},
            "bon": {"seed": bon_seed, "metrics": bon[bon_seed], "candidate_count": 4},
            "psp": {"seed": psp_seed, "metrics": score["scores"]["psp"]["52"][psp_seed], "candidate_count": 8,
                    "schedule": trace, "terminal_image_source": "step_52 terminal scheduler decode"},
            "selection_rng": "none"}


def selections_to_rows(selections: list[dict]) -> tuple[list[dict], list[dict]]:
    selected, items = [], []
    for entry in selections:
        for method in METHODS:
            value = entry[method]
            metric = value["metrics"]
            row = {"task_index": entry["task_index"], "task_id": entry["task_id"], "uid": entry["uid"], "oid": entry["oid"],
                   "reward": entry["reward"], "method": method, "seed": value["seed"],
                   "categories": entry["categories"], "compatibility": metric["compatibility"], "personal": metric["personal"],
                   "image_paths": metric["image_paths"], "image_count": len(metric["image_paths"])}
            selected.append(row)
            for position, (category, path, personal) in enumerate(zip(entry["categories"], metric["image_paths"], metric["personal_by_item"])):
                items.append({**{key: row[key] for key in ("task_index", "task_id", "uid", "oid", "reward", "method", "seed")},
                              "item_position": position, "category": category, "image_path": path, "personal_item": personal})
    return selected, items


def load_gt(selected: list[dict], data_root: Path, image_root: Path) -> dict[int, list[Path]]:
    ground = np.load(data_root / "test_grd.npy", allow_pickle=True).item()
    item_paths = np.load(data_root / "new_all_item_image_paths.npy", allow_pickle=True)
    result = {}
    for row in selected:
        if row["task_index"] in result:
            continue
        record = ground.get(row["oid"], ground.get(str(row["oid"])))
        if record is None:
            raise KeyError(f"missing GT oid={row['oid']}")
        categories = [int(value) for value in record["category"]]
        if categories != row["categories"]:
            raise ValueError(f"GT category mismatch for {row['task_id']}")
        paths = [image_root / str(item_paths[int(item)]) for item in record["outfits"]]
        if len(paths) != 4 or not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"missing GT image for {row['task_id']}")
        result[row["task_index"]] = paths
    return result


def posthoc_gt_metrics(selected: list[dict], gt: dict[int, list[Path]], clip, preprocess, open_clip,
                       args: argparse.Namespace, device: torch.device) -> None:
    paths = []
    for row in selected:
        paths.extend(Path(value) for value in row["image_paths"])
        paths.extend(gt[row["task_index"]])
    unique = list(dict.fromkeys(paths))
    h14 = encode_images(clip, preprocess, unique, device, args.batch_size)
    h14_lookup = {str(path.resolve()): value for path, value in zip(unique, h14)}
    for row in selected:
        generated = np.stack([h14_lookup[str(Path(path).resolve())] for path in row["image_paths"]])
        reference = np.stack([h14_lookup[str(path.resolve())] for path in gt[row["task_index"]]])
        row["gt_sim"] = float(np.mean(np.sum(generated * reference, axis=1)))

    model, preprocess_l14 = load_gaa_model(args.gaa_clip_weights, device, open_clip)
    l14 = encode_images(model, preprocess_l14, unique, device, args.batch_size)
    lookup = {str(path.resolve()): value for path, value in zip(unique, l14)}
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    text = {}
    for group, values in ATTRIBUTE_BANKS.items():
        prompts = [f"a product photo of a fashion item that is {value}" for value in values]
        text[group] = F.normalize(model.encode_text(tokenizer(prompts).to(device)).float(), dim=-1).detach().cpu().numpy()
    scale = float(model.logit_scale.exp().float().detach().cpu())
    for row in selected:
        generated = np.stack([lookup[str(Path(path).resolve())] for path in row["image_paths"]])
        reference = np.stack([lookup[str(path.resolve())] for path in gt[row["task_index"]]])
        groups = []
        for group, embeddings in text.items():
            gen_prob = torch.softmax(torch.from_numpy(generated @ embeddings.T * scale), dim=1).numpy()
            gt_prob = torch.softmax(torch.from_numpy(reference @ embeddings.T * scale), dim=1).numpy()
            value = float(np.minimum(gen_prob, gt_prob).sum(axis=1).mean())
            row[f"gaa_{group}"] = value
            groups.append(value)
        row["gaa"] = float(np.mean(groups))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def inception_metrics(selected: list[dict], args: argparse.Namespace, device: torch.device) -> dict[str, dict]:
    if args.skip_inception:
        return {method: {"inception_status": "skipped"} for method in METHODS}
    root = str(args.dualfashion_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval_utils import calculate_inception_score_given_data
    from torch.utils.data import Dataset
    from torchvision.transforms import ToTensor

    class Values(Dataset):
        def __init__(self, values): self.values = values
        def __len__(self): return len(self.values)
        def __getitem__(self, index): return self.values[index]

    weights = args.dualfashion_root / "eval" / "finetuned-inception" / "Inception-finetune-epoch300.zip"
    labels = np.load(args.dualfashion_root / "eval" / "finetuned-inception" / "cid_to_label.npy", allow_pickle=True).item()
    if not weights.is_file():
        return {method: {"inception_status": "unavailable"} for method in METHODS}
    transform = ToTensor()
    result = {}
    for method in METHODS:
        rows = [row for row in selected if row["method"] == method]
        tensors, categories = [], []
        for row in rows:
            for path, category in zip(row["image_paths"], row["categories"]):
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
                categories.append(int(labels[int(category)]))
        acc, entropy, _, score, _ = calculate_inception_score_given_data(
            Values(tensors), Values([torch.tensor(value) for value in categories]), str(weights), 50,
            args.batch_size, device, num_workers=0)
        result[method] = {"inception_status": "complete", "inception_acc_top3": float(acc),
                          "inception_score": float(score), "inception_entropy": float(entropy), "n_images": len(tensors)}
    return result


def summarize(selected: list[dict], inception: dict[str, dict]) -> list[dict]:
    result = []
    for method in METHODS:
        rows = [row for row in selected if row["method"] == method]
        record = {"reward": REWARD, "method": method, "n_tasks": len(rows), "n_images": len(rows) * 4}
        for key in ("compatibility", "personal", "gt_sim", "gaa", "gaa_color", "gaa_pattern", "gaa_material", "gaa_style"):
            values = [float(row[key]) for row in rows if key in row and math.isfinite(float(row[key]))]
            if values:
                record[f"{key}_mean"] = float(np.mean(values))
                record[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        record.update(inception[method])
        result.append(record)
    return result


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    conditions = load_conditions(args.data_root)
    tasks, problems = validate_cache(args.cache_root, conditions, args.max_tasks)
    integrity = {"schema": SCHEMA, "cache_root": str(args.cache_root.resolve()), "expected_tasks": len(tasks) + len(problems),
                 "complete_tasks": len(tasks), "problems": problems, "candidate_protocol": "BoN@4 final; PSP@8 steps 13/26/52"}
    if args.audit_only:
        atomic_json(args.output_root / "integrity_report.json", integrity)
        print(json.dumps(integrity, indent=2))
        return 0 if not problems else 2
    if problems:
        atomic_json(args.output_root / "integrity_report.json", integrity)
        raise RuntimeError(f"cache audit failed for {len(problems)} tasks; see integrity_report.json")
    config = {"schema": SCHEMA, "backbone": "DiFashion", "task": "GOR", "reward": REWARD,
              "cache_root": str(args.cache_root.resolve()), "data_root": str(args.data_root.resolve()),
              "selection": "standard=BoN seed0; BoN=max terminal Personal; PSP=Personal 8->4 at 13, 4->2 at 26, 2->1 at terminal 52",
              "gt_policy": "GT not read during candidate scoring or selection; post-hoc only",
              "offline_collection_cost": "BoN=4*52, PSP=8*52 model calls/task",
              "simulated_online_budget": "BoN=4*52, PSP=8*13+4*(26-13)+2*(52-26)=208 model calls/task"}
    config_path = args.output_root / "run_manifest.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("existing output configuration differs; choose a new output root")
    if config_path.exists() and not args.resume:
        raise FileExistsError("output exists; use --resume or choose a new directory")
    atomic_json(config_path, config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    clip, preprocess, evaluator, open_clip = load_models(args, device)
    selections = []
    all_scores = []
    for task in tqdm(tasks, desc="DiFashion GOR Personal selection", unit="task"):
        task_id = task["condition"]["task_id"]
        score_path = args.output_root / "score_tasks" / f"{task_id}.json"
        selection_path = args.output_root / "selections" / f"{task_id}.json"
        if args.resume and score_path.is_file() and selection_path.is_file():
            score, selection = read_json(score_path), read_json(selection_path)
        else:
            score = score_task(task, args.cache_root, history, clip, preprocess, evaluator, device, args.batch_size)
            selection = make_selection(score)
            atomic_json(score_path, score)
            atomic_json(selection_path, selection)
        selections.append(selection)
        for method, nodes in score["scores"].items():
            for node, values in nodes.items():
                for seed, metrics in values.items():
                    all_scores.append({"task_index": score["task_index"], "task_id": score["task_id"], "uid": score["uid"], "oid": score["oid"],
                                       "reward": REWARD, "method": method, "node": node, "seed": seed,
                                       "compatibility": metrics["compatibility"], "personal": metrics["personal"],
                                       "personal_by_item": metrics["personal_by_item"], "image_paths": metrics["image_paths"]})
    selected, item_rows = selections_to_rows(selections)
    write_csv(args.output_root / "candidate_metrics.csv", all_scores)
    write_csv(args.output_root / "selected_metrics.csv", selected)
    write_csv(args.output_root / "selected_item_metrics.csv", item_rows)
    if args.with_gt_metrics:
        # This is intentionally after every frozen selection JSON is written.
        gt = load_gt(selected, args.data_root, args.image_root)
        posthoc_gt_metrics(selected, gt, clip, preprocess, open_clip, args, device)
        write_csv(args.output_root / "task_metrics.csv", selected)
    inception = inception_metrics(selected, args, device)
    summary = summarize(selected, inception)
    write_csv(args.output_root / "group_summary.csv", summary)
    integrity.update({"selection_count": len(selections), "selected_rows": len(selected), "candidate_rows": len(all_scores),
                      "selection_complete": len(selections) == len(tasks), "gt_metrics": bool(args.with_gt_metrics),
                      "all_personal_finite": all(math.isfinite(float(row["personal"])) for row in all_scores)})
    atomic_json(args.output_root / "integrity_report.json", integrity)
    print(json.dumps({"output": str(args.output_root), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
