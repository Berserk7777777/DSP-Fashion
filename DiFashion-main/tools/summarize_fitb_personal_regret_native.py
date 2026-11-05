#!/usr/bin/env python
"""Report DiFashion FITB Personal-selection regret with native oracle metrics.

The reward oracle is descriptive only: it finds the highest terminal Personal
Sim in BoN's four-seed pool or PSP's eight-seed pool.  Ground truth is opened
only after those seed identities are frozen, solely to report native CIS and
LPIPS for the PSP terminal-Personal oracle.  It is never used for selection.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-root", type=Path, required=True,
                        help="Shared native candidate_scores directory.")
    parser.add_argument("--selection-root", type=Path, required=True,
                        help="Frozen Personal selection directory containing selections/.")
    parser.add_argument("--native-metrics", type=Path, required=True,
                        help="Existing frozen Personal native task_metrics.csv.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = sorted({key for row in rows for key in row})
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def mean_std(values: list[float]) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else None


def source_module() -> Any:
    path = Path(__file__).with_name("evaluate_fitb_offline_native.py")
    spec = importlib.util.spec_from_file_location("native_fitb_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import native evaluator at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def choose_max(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select from empty seed pool")
    return sorted(rows, key=lambda row: (-float(row["personal"]), int(row["seed"])))[0]


def load_frozen_inputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    native_rows = []
    with args.native_metrics.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("task_index", "uid", "oid", "seed", "candidate_index", "blank_index", "target_category"):
                row[key] = int(row[key])
            for key in ("cs", "cis", "lpips", "compatibility", "personal"):
                row[key] = float(row[key])
            native_rows.append(row)
    if len(native_rows) != 2207 * 3:
        raise ValueError(f"Expected 6621 frozen native rows, found {len(native_rows)}")
    native = {(row["task_id"], row["method"]): row for row in native_rows}
    if len(native) != len(native_rows):
        raise ValueError("Duplicate task/method rows in native metrics")

    selections: dict[str, dict[str, Any]] = {}
    score_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(args.selection_root.joinpath("selections").glob("*.json")):
        selection = read_json(path)
        if selection.get("reward") != "personal":
            raise ValueError(f"Selection is not frozen Personal reward: {path}")
        task_id = selection["task_id"]
        if task_id in selections:
            raise ValueError(f"Duplicate frozen selection: {task_id}")
        score_path = args.score_root / f"{task_id}.json"
        score_payload = read_json(score_path)
        terminal = [row for row in score_payload.get("rows", []) if row.get("node") == "final"]
        bon = [row for row in terminal if row.get("method") == "bon"]
        psp = [row for row in terminal if row.get("method") == "psp"]
        if len(bon) != 4 or len(psp) != 8:
            raise ValueError(f"{task_id}: expected 4 BoN and 8 PSP terminal rows")
        for row in terminal:
            key = (task_id, str(row["method"]), int(row["seed"]))
            if key in score_index:
                raise ValueError(f"Duplicate terminal score {key}")
            score_index[key] = row
        selections[task_id] = selection
    if len(selections) != 2207:
        raise ValueError(f"Expected 2207 frozen selections, found {len(selections)}")
    return native_rows, selections, score_index


def score_psp_oracles(args: argparse.Namespace, module: Any, oracle_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate only the missing PSP terminal-Personal oracle images natively."""
    output = args.output_root / "psp_personal_terminal_oracle_native_metrics.csv"
    if args.resume and output.is_file():
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != len(oracle_rows):
            raise RuntimeError("Saved oracle-native table has an unexpected task count")
        for row in rows:
            for key in ("task_index", "uid", "oid", "seed", "candidate_index", "blank_index", "target_category"):
                row[key] = int(row[key])
            for key in ("cs", "cis", "lpips", "compatibility", "personal"):
                row[key] = float(row[key])
        return rows

    tasks = [dict(row["task"]) for row in oracle_rows]
    gt = module.gt_records(args, tasks)  # Post-selection only.
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    categories = np.load(args.data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
    device, clip, preprocess, tokenizer, evaluator, _ = module.load_models(args)
    try:
        import lpips
        lpips_model = lpips.LPIPS(net="vgg").to(device).eval()
        rows: list[dict[str, Any]] = []
        candidates = [dict(row["candidate"]) for row in oracle_rows]
        for start in tqdm(range(0, len(candidates), args.batch_size), desc="Native PSP Personal-oracle metrics", unit="image"):
            batch = candidates[start:start + args.batch_size]
            generated = module.load_image_batch(batch, preprocess, device)
            gt_batch = module.load_image_batch(
                [{"image_path": gt[row["task_id"]]["gt_image_path"]} for row in batch], preprocess, device
            )
            texts = tokenizer([module.category_prompt(row["target_category"], categories) for row in batch]).to(device)
            with torch.inference_mode():
                raw_features = clip.encode_image(generated).float()
                raw_gt_features = clip.encode_image(gt_batch).float()
                features = F.normalize(raw_features, dim=-1)
                gt_features = F.normalize(raw_gt_features, dim=-1)
                text_features = F.normalize(clip.encode_text(texts).float(), dim=-1)
                contexts = module.load_contexts(batch, item_features, device)
                for index, row in enumerate(batch):
                    contexts[index, row["blank_index"], :] = raw_features[index]
                compatibility = torch.sigmoid(evaluator(contexts).float())
                lpips_values = lpips_model(
                    torch.stack([module.pil_lpips_tensor(row["image_path"], device) for row in batch]),
                    torch.stack([module.pil_lpips_tensor(gt[row["task_id"]]["gt_image_path"], device) for row in batch]),
                ).reshape(-1)
            for index, row in enumerate(batch):
                history_value = module.history_embedding(history, row["uid"], row["target_category"])
                if history_value is None or history_value.size != features.shape[1]:
                    raise ValueError(f"Missing Personal history for {row['task_id']}")
                history_tensor = F.normalize(torch.as_tensor(history_value, dtype=torch.float32, device=device).unsqueeze(0), dim=-1)
                rows.append({**row, **gt[row["task_id"]],
                    "cs": float(100.0 * F.cosine_similarity(features[index:index + 1], text_features[index:index + 1]).item()),
                    "cis": float(100.0 * F.cosine_similarity(features[index:index + 1], gt_features[index:index + 1]).item()),
                    "lpips": float(lpips_values[index].item()),
                    "compatibility": float(compatibility[index].item()),
                    "personal": float(100.0 * F.cosine_similarity(features[index:index + 1], history_tensor).item()),
                })
    finally:
        del clip, evaluator, lpips_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    rows.sort(key=lambda row: row["task_index"])
    write_csv(output, rows)
    return rows


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    for path in (args.score_root, args.selection_root / "selections", args.native_metrics, args.data_root, args.evaluation_root):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema": 1, "reward": "personal", "score_root": str(args.score_root.resolve()),
        "selection_root": str(args.selection_root.resolve()), "native_metrics": str(args.native_metrics.resolve()),
        "pool_definition": {"bon": "terminal max Personal Sim over independent 4-seed BoN pool",
                            "psp": "terminal max Personal Sim over all independent 8-seed PSP trajectories"},
        "regret": "terminal Personal oracle minus frozen Personal selection / best terminal Personal among survivors",
        "gt_usage": "GT is read after oracle seed identity is fixed, only for CIS and LPIPS reporting.",
        "test_status": "exploratory test-cache analysis; not an unbiased final test claim.",
    }
    config_path = args.output_root / "report_config.json"
    if config_path.is_file() and read_json(config_path) != config:
        raise ValueError("Existing report configuration differs; choose a separate output directory")
    write_json(config_path, config)
    native_rows, selections, score_index = load_frozen_inputs(args)
    native_map = {(row["task_id"], row["method"]): row for row in native_rows}
    oracle_specs: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    for task_id, selection in sorted(selections.items()):
        bon_pool = [row for (task, method, _), row in score_index.items() if task == task_id and method == "bon"]
        psp_pool = [row for (task, method, _), row in score_index.items() if task == task_id and method == "psp"]
        bon_oracle, psp_oracle = choose_max(bon_pool), choose_max(psp_pool)
        task = {key: psp_oracle[key] for key in ("task_index", "task_id", "task_dir", "uid", "oid", "outfit", "category", "blank_index", "target_category")}
        oracle_specs.append({"task": task, "candidate": psp_oracle})
        bon_selected = selection["bon"]["winner"]
        psp_selected = selection["psp"]["winner"]
        if int(bon_selected["seed"]) != int(bon_oracle["seed"]):
            raise RuntimeError(f"{task_id}: frozen Personal BoN winner is not terminal Personal oracle")
        active = {row["seed"] for row in psp_pool}
        for trace in selection["psp"]["trace"]:
            active = {int(seed) for seed in trace["kept_seeds"]}
            best_remaining = choose_max([row for row in psp_pool if int(row["seed"]) in active])
            regret_rows.append({
                "task_id": task_id, "task_index": task["task_index"], "method": "psp", "node": trace["node"],
                "oracle_seed": int(psp_oracle["seed"]), "oracle_personal": float(psp_oracle["personal"]),
                "selected_seed": int(best_remaining["seed"]), "selected_personal": float(best_remaining["personal"]),
                "personal_regret": float(psp_oracle["personal"]) - float(best_remaining["personal"]),
                "oracle_hit": int(psp_oracle["seed"]) == int(best_remaining["seed"]),
                "definition": "all-8 terminal Personal oracle minus best terminal Personal among frozen-surviving PSP seeds",
            })
        regret_rows.append({
            "task_id": task_id, "task_index": task["task_index"], "method": "bon", "node": "final",
            "oracle_seed": int(bon_oracle["seed"]), "oracle_personal": float(bon_oracle["personal"]),
            "selected_seed": int(bon_selected["seed"]), "selected_personal": float(bon_selected["personal"]),
            "personal_regret": float(bon_oracle["personal"]) - float(bon_selected["personal"]),
            "oracle_hit": int(bon_oracle["seed"]) == int(bon_selected["seed"]),
            "definition": "all-4 terminal Personal oracle minus frozen BoN Personal selection",
        })
    module = source_module()
    oracle_native_rows = score_psp_oracles(args, module, oracle_specs)
    oracle_native = {row["task_id"]: row for row in oracle_native_rows}
    if len(oracle_native) != len(selections):
        raise RuntimeError("Incomplete native PSP oracle table")

    comparison: list[dict[str, Any]] = []
    for task_id, selection in sorted(selections.items()):
        for label, source in (("standard", native_map[(task_id, "standard")]),
                              ("bon_personal_oracle", native_map[(task_id, "bon")]),
                              ("psp_personal_selected", native_map[(task_id, "psp")]),
                              ("psp_personal_terminal_oracle", oracle_native[task_id])):
            comparison.append({"task_id": task_id, "task_index": source["task_index"], "method": label,
                               "seed": source["seed"], "candidate_index": source["candidate_index"],
                               **{metric: float(source[metric]) for metric in ("cs", "cis", "lpips", "compatibility", "personal")}})
    comparison.sort(key=lambda row: (row["task_index"], row["method"]))
    write_csv(args.output_root / "selection_and_personal_oracle_metrics.csv", comparison)
    write_csv(args.output_root / "personal_regret_per_task.csv", regret_rows)

    summary = []
    for method in ("standard", "bon_personal_oracle", "psp_personal_selected", "psp_personal_terminal_oracle"):
        rows = [row for row in comparison if row["method"] == method]
        for metric in ("cs", "cis", "lpips", "compatibility", "personal"):
            mean, std = mean_std([row[metric] for row in rows])
            summary.append({"table": "selection_metrics", "method": method, "metric": metric,
                            "direction": "lower" if metric == "lpips" else "higher", "mean": mean,
                            "std_sample": std, "n_tasks": len(rows)})
    for method, node in (("bon", "final"), ("psp", "reward_step_13"), ("psp", "reward_step_26"), ("psp", "final")):
        rows = [row for row in regret_rows if row["method"] == method and row["node"] == node]
        mean, std = mean_std([row["personal_regret"] for row in rows])
        summary.append({"table": "personal_regret", "method": method, "metric": "personal_regret", "node": node,
                        "direction": "lower", "mean": mean, "std_sample": std, "n_tasks": len(rows),
                        "oracle_hit_rate": float(np.mean([row["oracle_hit"] for row in rows])),
                        "mean_oracle_personal": float(np.mean([row["oracle_personal"] for row in rows])),
                        "mean_selected_personal": float(np.mean([row["selected_personal"] for row in rows]))})
    write_csv(args.output_root / "summary.csv", summary)
    write_json(args.output_root / "integrity_report.json", {
        "tasks": len(selections), "native_frozen_rows": len(native_rows), "oracle_native_rows": len(oracle_native_rows),
        "regret_rows": len(regret_rows), "complete": len(selections) == 2207 and len(oracle_native_rows) == 2207,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
