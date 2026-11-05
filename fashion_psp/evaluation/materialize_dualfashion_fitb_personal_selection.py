"""Materialize frozen FITB selections from the timestep cache.

This utility is deliberately selection-only: it uses Personal scores from the
cache and never reads GT-Sim/GAA/CIS/LPIPS.  The resulting ``gen.npy`` and
``grd.npy`` payloads can be passed to the native DualFashion evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--grid-json", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=200)
    return parser.parse_args()


def load_rows(path: Path) -> dict[int, dict[int, dict[int, dict]]]:
    tasks: dict[int, dict[int, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            task, step, seed = int(row["task_index"]), int(row["step"]), int(row["seed"])
            row["task_index"], row["step"], row["seed"] = task, step, seed
            row["personal"] = float(row["personal"])
            tasks[task][step][seed] = row
    return tasks


def choose(by_step: dict[int, dict[int, dict]], cutoffs: list[int], keeps: list[int]) -> dict:
    active = sorted(by_step[max(by_step)])
    for step, keep in zip(cutoffs, keeps):
        candidates = [by_step[step][seed] for seed in active]
        ranked = sorted(candidates, key=lambda row: (-row["personal"], row["seed"]))
        active = [int(row["seed"]) for row in ranked[:keep]]
    # A schedule's last keep value is the number of survivors reaching the
    # terminal step.  The final winner is selected using the terminal score.
    final_candidates = [by_step[max(by_step)][seed] for seed in active]
    return sorted(final_candidates, key=lambda row: (-row["personal"], row["seed"]))[0]


def task_paths(fitb: dict, grd: dict, image_paths: np.ndarray, image_root: Path, task: int):
    uid, oid = int(fitb["uids"][task]), int(fitb["oids"][task])
    outfit = np.asarray(fitb["outfits"][task], dtype=np.int64).reshape(-1)
    categories = np.asarray(fitb["category"][task], dtype=np.int64).reshape(-1)
    blank = np.flatnonzero(outfit == 0)
    if len(blank) != 1:
        raise ValueError(f"task {task} ({uid}_{oid}) does not have one blank item")
    blank_idx = int(blank[0])
    context = [str((image_root / str(image_paths[int(item)])).resolve()) for item in outfit if int(item) != 0]
    target_item = int(grd[oid]["outfits"][blank_idx])
    gt_path = str((image_root / str(image_paths[target_item])).resolve())
    return uid, oid, outfit, categories, blank_idx, context, target_item, gt_path


def materialize(label: str, selected: dict[int, dict], fitb: dict, grd: dict,
                image_paths: np.ndarray, image_root: Path, destination: Path) -> None:
    gen, ground = {}, {}
    for task in sorted(selected):
        row = selected[task]
        uid, oid, outfit, categories, blank_idx, context, target_item, gt_path = task_paths(
            fitb, grd, image_paths, image_root, task
        )
        generated_path = Path(row["image_path"]).resolve()
        if not generated_path.is_file() or not Path(gt_path).is_file():
            raise FileNotFoundError(f"task {task}: generated={generated_path}, gt={gt_path}")
        full_outfit = outfit.copy()
        full_outfit[blank_idx] = target_item
        gen.setdefault(uid, {})[oid] = {
            "image_paths": [str(generated_path)],
            "cates": np.asarray([categories[blank_idx]], dtype=np.int64),
            "candidate_seeds": [int(row["seed"])],
            "incomplete_outfit": context,
            "outfits": full_outfit,
        }
        ground.setdefault(uid, {})[oid] = {
            "image_paths": [gt_path],
            "cates": np.asarray([categories[blank_idx]], dtype=np.int64),
            "incomplete_outfit": context,
            "outfits": full_outfit,
        }
    checkpoint = destination / "0057501"
    checkpoint.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint / "gen.npy", gen)
    np.save(checkpoint / "grd.npy", ground)
    (destination / "selection_manifest.json").write_text(
        json.dumps({"label": label, "n_tasks": len(selected), "selection_reward": "personal",
                    "uses_gt_for_selection": False}, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics_csv)
    task_ids = sorted(rows)[: args.max_tasks]
    if len(task_ids) != args.max_tasks:
        raise ValueError(f"cache has {len(task_ids)} tasks, requested {args.max_tasks}")
    fitb = np.load(args.data_root / "processed" / "fitb_test.npy", allow_pickle=True).item()
    # test_grd.npy is indexed by outfit id and contains the completed GT outfit.
    grd = np.load(args.data_root / "test_grd.npy", allow_pickle=True).item()
    image_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)

    grid_payload = json.loads(args.grid_json.read_text(encoding="utf-8"))
    schedule = grid_payload["results"][0]["schedule"]
    grid_cutoffs = [int(value) for value in schedule["cutoff_times"]]
    grid_keeps = [int(value) for value in schedule["remaining_seeds"]]
    default_cutoffs, default_keeps = [7, 14, 28], [4, 2, 1]
    choices = {
        "personal_grid_best": (grid_cutoffs, grid_keeps),
        "personal_default_7_14_28": (default_cutoffs, default_keeps),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"task_count": len(task_ids), "schedules": {}}
    for label, (cutoffs, keeps) in choices.items():
        selected = {task: choose(rows[task], cutoffs, keeps) for task in task_ids}
        destination = args.output_root / label
        materialize(label, selected, fitb, grd, image_paths, args.image_root, destination)
        manifest["schedules"][label] = {"cutoffs": cutoffs, "keeps": keeps,
                                         "particle_steps": int(8 * cutoffs[0] + sum(
                                             keeps[i] * (cutoffs[i + 1] - cutoffs[i])
                                             for i in range(len(cutoffs) - 1)) + keeps[-1] * (28 - cutoffs[-1]))}
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
