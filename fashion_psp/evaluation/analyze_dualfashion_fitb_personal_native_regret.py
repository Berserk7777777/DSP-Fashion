"""Measure native-metric terminal-pool regret for Personal-reward FITB runs.

The script scores frozen terminal candidates only.  It never uses GT until the
candidate paths and frozen method selections have been resolved.  Each method
is compared with the oracle from its own available seed pool: Standard/BoN use
the four seed-42 candidates, while PSP uses its eight seed-2026 candidates.
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

from evaluate_dualfashion_fitb_native_metrics import (
    PERSONAL_CLIP_WEIGHTS,
    category_prompt,
    clip_tensor,
    history_vector,
    load_models,
    lpips_tensor,
    readable,
)


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dualfashion-root", type=Path, default=repo / "DualFashion-main")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tasks", type=int, default=2207)
    parser.add_argument("--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def psp_image(cache_root: Path, task: int, uid: int, oid: int, seed: int) -> Path:
    path = cache_root / f"psp_offline_group0_task{task}_seed2026" / "offline_candidates" / f"{uid}_{oid}" / "step_28" / f"seed_{seed}_x0.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def candidate_rows(args: argparse.Namespace) -> tuple[list[dict], dict[tuple[str, int], int]]:
    root = args.dualfashion_root
    fitb_records = np.load(args.data_root / "fitb_test_dict.npy", allow_pickle=True).item()
    task_lookup = {
        (int(uid), int(oid)): index
        for index, (uid, oid) in enumerate(zip(
            np.load(args.data_root / "test_grd_dict.npy", allow_pickle=True).item()["uids"],
            np.load(args.data_root / "test_grd_dict.npy", allow_pickle=True).item()["oids"],
        ))
    }
    bon_scores = root / "output/ifashion_fitb_personal/bon/bon_group0_seed42_scores/candidate_personal_scores.csv"
    bon_summary = root / "output/ifashion_fitb_personal/bon/bon_group0_seed42_scores/bon_n4_summary.json"
    psp_summary = root / "output/ifashion_fitb_personal/psp/psp_group0_personal/psp_selection_summary.csv"
    cache_root = root / "output/ifashion_fitb_comp/psp_offline_cache"
    for path in (bon_scores, bon_summary, psp_summary):
        if not path.is_file():
            raise FileNotFoundError(path)

    bon_by_task: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(bon_scores):
        identity = (int(row["uid"]), int(row["oid"]))
        if identity not in task_lookup:
            raise RuntimeError(f"BoN candidate has unknown FITB task identity: {identity}")
        task = task_lookup[identity]
        if task < args.max_tasks:
            bon_by_task[task].append(row)
    if set(bon_by_task) != set(range(args.max_tasks)):
        raise RuntimeError("BoN candidate score file does not cover requested FITB prefix")

    psp_by_task = {int(row["task_index"]): row for row in read_csv(psp_summary) if int(row["task_index"]) < args.max_tasks}
    if set(psp_by_task) != set(range(args.max_tasks)):
        raise RuntimeError("PSP Personal selection summary does not cover requested FITB prefix")

    rows: list[dict] = []
    selected: dict[tuple[str, int], int] = {}
    for task in range(args.max_tasks):
        bon_candidates = sorted(bon_by_task[task], key=lambda row: int(row["candidate_idx"]))
        if len(bon_candidates) != 4:
            raise RuntimeError(f"task {task}: expected four BoN candidates")
        uid, oid = int(bon_candidates[0]["uid"]), int(bon_candidates[0]["oid"])
        for candidate in bon_candidates:
            image = Path(candidate["image_path"]).resolve()
            readable(image)
            rows.append({
                "pool": "bon4", "task_index": task, "uid": uid, "oid": oid,
                "seed": int(candidate["seed"]), "image_path": str(image),
            })
        standard = [row for row in bon_candidates if row["is_standard"].lower() == "true"]
        winner = [row for row in bon_candidates if row["is_bon_winner"].lower() == "true"]
        if len(standard) != 1 or len(winner) != 1:
            raise RuntimeError(f"task {task}: missing frozen Standard/BoN choice")
        selected[("standard", task)] = int(standard[0]["seed"])
        selected[("bon_personal", task)] = int(winner[0]["seed"])

        psp = psp_by_task[task]
        if (uid, oid) != (int(psp["uid"]), int(psp["oid"])):
            raise RuntimeError(f"task {task}: BoN/PSP task identity mismatch")
        seeds = [2026 + task * 10000 + slot for slot in range(8)]
        for seed in seeds:
            image = psp_image(cache_root, task, uid, oid, seed)
            readable(image)
            rows.append({"pool": "psp8", "task_index": task, "uid": uid, "oid": oid, "seed": seed, "image_path": str(image)})
        selected[("psp_personal", task)] = int(psp["psp_seed"])
        if selected[("psp_personal", task)] not in seeds:
            raise RuntimeError(f"task {task}: PSP selected seed outside its frozen terminal pool")
    return rows, selected


def loaded_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        {**row, "task_index": int(row["task_index"]), "uid": int(row["uid"]), "oid": int(row["oid"]), "seed": int(row["seed"])}
        for row in read_csv(path)
    ]


def score_candidates(args: argparse.Namespace, candidates: list[dict]) -> list[dict]:
    output_path = args.output_root / "candidate_native_metrics.csv"
    existing = loaded_rows(output_path) if args.resume else []
    done = {(row["pool"], row["task_index"], row["seed"]) for row in existing}
    pending = [row for row in candidates if (row["pool"], row["task_index"], row["seed"]) not in done]
    if not pending:
        return existing

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    categories = np.load(args.data_root / "id_cate_dict.npy", allow_pickle=True).item()
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    fitb = np.load(args.data_root / "fitb_test_dict.npy", allow_pickle=True).item()
    ground = np.load(
        args.dualfashion_root / "output/ifashion_fitb_personal/psp/psp_group0_personal/0057501/grd.npy",
        allow_pickle=True,
    ).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    clip, preprocess, tokenizer, evaluator, lpips_model, _ = load_models(args, device)
    scored = list(existing)
    try:
        for start in tqdm(range(0, len(pending), args.batch_size), desc="FITB terminal candidate native metrics", unit="image"):
            batch = pending[start:start + args.batch_size]
            # The GT path is resolved only after frozen candidate paths have been
            # validated.  It is never consulted by any selection mapping.
            prepared: list[tuple[list[int], int, dict, int, str]] = []
            for row in batch:
                item_ids = list(fitb[row["uid"]][row["oid"]])
                blank = item_ids.index(0)
                record = ground[row["uid"]][row["oid"]]
                category = int(np.asarray(record["cates"]).reshape(-1)[0])
                gt_path = str(Path(record["image_paths"][0]).resolve())
                readable(Path(gt_path))
                prepared.append((item_ids, blank, record, category, gt_path))

            generated = torch.stack([clip_tensor(row["image_path"], preprocess, device) for row in batch])
            gt_images = torch.stack([clip_tensor(item[4], preprocess, device) for item in prepared])
            prompts = tokenizer([category_prompt(item[3], categories) for item in prepared]).to(device)
            with torch.inference_mode():
                raw_generated = clip.encode_image(generated).float()
                raw_ground = clip.encode_image(gt_images).float()
                normal_generated = F.normalize(raw_generated, dim=-1)
                normal_ground = F.normalize(raw_ground, dim=-1)
                normal_text = F.normalize(clip.encode_text(prompts).float(), dim=-1)
                outfits = []
                for offset, (item_ids, blank, _, _, _) in enumerate(prepared):
                    vectors = np.asarray([
                        item_features[item_id] if item_id else np.zeros(1024, dtype=np.float32)
                        for item_id in item_ids
                    ])
                    vectors[blank] = raw_generated[offset].cpu().numpy()
                    outfits.append(torch.as_tensor(vectors, dtype=torch.float32))
                compatibility = torch.sigmoid(evaluator(torch.stack(outfits).to(device))).reshape(-1)
                distances = lpips_model(
                    torch.stack([lpips_tensor(row["image_path"], device) for row in batch]),
                    torch.stack([lpips_tensor(item[4], device) for item in prepared]),
                ).reshape(-1)
            for offset, row in enumerate(batch):
                category = prepared[offset][3]
                history_embedding = F.normalize(
                    torch.as_tensor(history_vector(history, row["uid"], category), device=device), dim=0
                )
                scored.append({
                    **row, "gt_image_path": prepared[offset][4], "category": category,
                    "cs": float(100.0 * (normal_generated[offset] * normal_text[offset]).sum().item()),
                    "cis": float(100.0 * (normal_generated[offset] * normal_ground[offset]).sum().item()),
                    "lpips": float(distances[offset].item()),
                    "compatibility": float(compatibility[offset].item()),
                    "personal": float(100.0 * (normal_generated[offset] * history_embedding).sum().item()),
                })
            if (start // args.batch_size + 1) % 50 == 0:
                write_csv(output_path, scored)
    finally:
        del clip, evaluator, lpips_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_csv(output_path, scored)
    return scored


def summarize_regret(args: argparse.Namespace, rows: list[dict], selected: dict[tuple[str, int], int]) -> None:
    by_pool_task: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for row in rows:
        by_pool_task[(row["pool"], int(row["task_index"]))][int(row["seed"])] = row
    methods = {"standard": "bon4", "bon_personal": "bon4", "psp_personal": "psp8"}
    directions = {"cs": "higher", "cis": "higher", "lpips": "lower", "compatibility": "higher", "personal": "higher"}
    per_task = []
    for method, pool in methods.items():
        for task in range(args.max_tasks):
            candidates = by_pool_task[(pool, task)]
            expected = 4 if pool == "bon4" else 8
            if len(candidates) != expected:
                raise RuntimeError(f"{pool} task {task}: expected {expected} scored terminal candidates, found {len(candidates)}")
            seed = selected[(method, task)]
            if seed not in candidates:
                raise RuntimeError(f"{method} task {task}: selected seed is absent from scored pool")
            chosen = candidates[seed]
            record = {
                "method": method, "pool": pool, "pool_size": expected, "task_index": task,
                "uid": chosen["uid"], "oid": chosen["oid"], "selected_seed": seed,
            }
            for metric, direction in directions.items():
                oracle_seed = min(
                    candidates,
                    key=lambda candidate_seed: (
                        -float(candidates[candidate_seed][metric]) if direction == "higher" else float(candidates[candidate_seed][metric]),
                        candidate_seed,
                    ),
                )
                selected_value = float(chosen[metric])
                oracle_value = float(candidates[oracle_seed][metric])
                regret = oracle_value - selected_value if direction == "higher" else selected_value - oracle_value
                record.update({
                    f"selected_{metric}": selected_value,
                    f"oracle_{metric}": oracle_value,
                    f"{metric}_regret": regret,
                    f"{metric}_oracle_seed": oracle_seed,
                    f"{metric}_oracle_hit": int(seed == oracle_seed),
                })
            # The Personal oracle is the candidate with the highest Personal
            # score.  Its CIS/LPIPS are diagnostic values for the same image;
            # they are deliberately kept separate from the CIS/LPIPS oracles,
            # which may select different candidates.
            personal_oracle_seed = int(record["personal_oracle_seed"])
            personal_oracle = candidates[personal_oracle_seed]
            record.update({
                "personal_oracle_cis": float(personal_oracle["cis"]),
                "personal_oracle_lpips": float(personal_oracle["lpips"]),
                "personal_oracle_compatibility": float(personal_oracle["compatibility"]),
            })
            per_task.append(record)
    write_csv(args.output_root / "per_task_native_regret.csv", per_task)

    summary = []
    for method, pool in methods.items():
        selected_rows = [row for row in per_task if row["method"] == method]
        result = {"method": method, "pool": pool, "pool_size": selected_rows[0]["pool_size"], "n_tasks": len(selected_rows)}
        for metric, direction in directions.items():
            regret = np.asarray([float(row[f"{metric}_regret"]) for row in selected_rows], dtype=np.float64)
            result[f"selected_{metric}_mean"] = float(np.mean([float(row[f"selected_{metric}"]) for row in selected_rows]))
            result[f"oracle_{metric}_mean"] = float(np.mean([float(row[f"oracle_{metric}"]) for row in selected_rows]))
            result[f"{metric}_regret_mean"] = float(regret.mean())
            result[f"{metric}_regret_median"] = float(np.median(regret))
            result[f"{metric}_regret_p90"] = float(np.quantile(regret, 0.90))
            result[f"{metric}_oracle_hit_rate"] = float(np.mean([int(row[f"{metric}_oracle_hit"]) for row in selected_rows]))
            result[f"{metric}_direction"] = direction
        result["personal_oracle_cis_mean"] = float(np.mean([
            float(row["personal_oracle_cis"]) for row in selected_rows
        ]))
        result["personal_oracle_lpips_mean"] = float(np.mean([
            float(row["personal_oracle_lpips"]) for row in selected_rows
        ]))
        result["personal_oracle_compatibility_mean"] = float(np.mean([
            float(row["personal_oracle_compatibility"]) for row in selected_rows
        ]))
        summary.append(result)
    write_csv(args.output_root / "native_regret_summary.csv", summary)

    # A compact table for the common question: what GT-oriented metrics does
    # the Personal-maximizing seed actually obtain?
    personal_oracle_rows = []
    for method, pool in methods.items():
        selected_rows = [row for row in per_task if row["method"] == method]
        personal_oracle_rows.append({
            "method": method,
            "pool": pool,
            "pool_size": selected_rows[0]["pool_size"],
            "n_tasks": len(selected_rows),
            "personal_oracle_personal_mean": float(np.mean([
                float(row["oracle_personal"]) for row in selected_rows
            ])),
            "personal_oracle_cis_mean": float(np.mean([
                float(row["personal_oracle_cis"]) for row in selected_rows
            ])),
            "personal_oracle_lpips_mean": float(np.mean([
                float(row["personal_oracle_lpips"]) for row in selected_rows
            ])),
            "personal_oracle_compatibility_mean": float(np.mean([
                float(row["personal_oracle_compatibility"]) for row in selected_rows
            ])),
            "personal_oracle_cis_median": float(np.median([
                float(row["personal_oracle_cis"]) for row in selected_rows
            ])),
            "personal_oracle_lpips_median": float(np.median([
                float(row["personal_oracle_lpips"]) for row in selected_rows
            ])),
        })
    write_csv(args.output_root / "personal_oracle_gt_metrics.csv", personal_oracle_rows)
    (args.output_root / "methodology.json").write_text(json.dumps({
        "experiment": "DualFashion iFashion FITB, frozen Personal-reward selections",
        "methods": {"standard": "first candidate in the frozen BoN-4 seed-42 pool", "bon_personal": "argmax Personal over the same BoN-4 pool", "psp_personal": "PSP final selection from its frozen PSP-8 seed-2026 pool"},
        "regret": {"higher_metrics": "oracle - selected", "lpips": "selected - oracle", "oracle_scope": "per task, within the method's frozen terminal candidate pool"},
        "metrics": {"cs": "100*cosine(OpenCLIP image, category prompt)", "cis": "100*cosine(OpenCLIP image, GT image)", "lpips": "LPIPS-VGG(image, GT image)", "compatibility": "sigmoid(FashionEvaluator(fixed context plus generated item))", "personal": "100*cosine(OpenCLIP image, user category-history embedding)"},
        "personal_oracle_diagnostics": {
            "personal_oracle_cis": "CIS of the candidate with the highest native Personal score within the method pool",
            "personal_oracle_lpips": "LPIPS of the candidate with the highest native Personal score within the method pool",
            "personal_oracle_compatibility": "Compatibility of the candidate with the highest native Personal score within the method pool",
            "warning": "These are not CIS/LPIPS oracle values; the Personal-maximizing candidate and GT-similarity-maximizing candidates may differ."
        },
        "gt_policy": "GT is loaded after candidate and method selection mappings are frozen; CIS/LPIPS are diagnostics only.",
        "warning": "BoN-4 and PSP-8 use different seed pools, so their oracle values are not a direct controlled comparison; compare regret rates within each pool.",
    }, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_tasks < 1:
        raise ValueError("--batch-size and --max-tasks must be positive")
    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates, selected = candidate_rows(args)
    rows = score_candidates(args, candidates)
    expected = args.max_tasks * 12
    if len(rows) != expected:
        raise RuntimeError(f"candidate metric file is incomplete: expected {expected}, found {len(rows)}")
    summarize_regret(args, rows, selected)
    print(json.dumps({"output": str(args.output_root), "candidates": len(rows), "tasks": args.max_tasks}, indent=2))


if __name__ == "__main__":
    main()
