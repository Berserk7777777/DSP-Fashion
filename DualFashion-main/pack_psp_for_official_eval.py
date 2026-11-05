"""Pack PSP final survivors for DualFashion's official PFITB evaluator.

The PSP runner writes one task per directory and keeps its final survivor in
``metadata.json``.  ``evaluate_PFITB.py`` instead consumes the same packed
``gen.npy``/``grd.npy`` format as the regular inference runner.  This utility
only converts between those two formats; it does not score candidates or
perform regret/oracle analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


TASK_PATTERN = re.compile(r"^psp_group(?P<group>\d+)_task(?P<task>\d+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True, help="Directory containing psp_group*_task*_seed* runs.")
    parser.add_argument("--group", type=int, required=True, help="PSP seed group index, e.g. 0, 1, or 2.")
    parser.add_argument("--base-seed", type=int, required=True, help="Base seed used by this PSP group.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination containing 0057501/gen.npy and grd.npy.")
    parser.add_argument("--data-root", type=Path, required=True, help="iFashion root, e.g. C:\\tmp\\ifashion.")
    parser.add_argument("--processed-info-root", type=Path, required=True, help="DualFashion processed_info directory.")
    parser.add_argument("--image-root", type=Path, required=True, help="iFashion semantic_category directory.")
    parser.add_argument("--checkpoint", default="0057501")
    return parser.parse_args()


def load_fitb(data_root: Path) -> dict:
    path = data_root / "processed" / "fitb_test.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True).item()


def task_index_for(fitb: dict, uid: int, oid: int) -> int:
    for index, (candidate_uid, candidate_oid) in enumerate(zip(fitb["uids"], fitb["oids"])):
        if int(candidate_uid) == uid and int(candidate_oid) == oid:
            return index
    raise KeyError(f"Task {uid}_{oid} is not present in processed/fitb_test.npy")


def context_paths(outfit: np.ndarray, blank_idx: int, image_paths: np.ndarray, image_root: Path) -> list[str]:
    paths: list[str] = []
    for index, item_id in enumerate(outfit.tolist()):
        if index == blank_idx:
            continue
        relative = image_paths[int(item_id)]
        path = image_root / str(relative)
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(str(path.resolve()))
    return paths


def main() -> None:
    args = parse_args()
    fitb = load_fitb(args.data_root)
    image_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)

    pattern = f"psp_group{args.group}_task*_seed{args.base_seed}"
    run_dirs = []
    for run_dir in args.output_root.glob(pattern):
        match = TASK_PATTERN.match(run_dir.name)
        if match:
            run_dirs.append((int(match.group("task")), run_dir))
    run_dirs.sort(key=lambda pair: pair[0])
    if not run_dirs:
        raise FileNotFoundError(f"No PSP runs matched {args.output_root / pattern}")

    gen_data: dict[int, dict[int, dict]] = {}
    grd_data: dict[int, dict[int, dict]] = {}
    for task_index, run_dir in run_dirs:
        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        uid = int(metadata["uid"])
        oid = int(metadata["oid"])
        if task_index_for(fitb, uid, oid) != task_index:
            raise ValueError(f"{run_dir.name}: task index does not match FITB metadata for {uid}_{oid}")

        final_rows = metadata.get("final", [])
        if len(final_rows) != 1:
            raise ValueError(f"{metadata_path}: expected exactly one final survivor, found {len(final_rows)}")
        final = final_rows[0]
        generated_path = Path(final["image_path"])
        if not generated_path.exists():
            raise FileNotFoundError(generated_path)

        incomplete = np.asarray(fitb["outfits"][task_index], dtype=np.int64).copy()
        blank_idx = int(np.flatnonzero(incomplete == 0)[0])
        target_item_id = int(metadata["target_item_id"])
        target_category = int(fitb["category"][task_index][blank_idx])
        full_outfit = incomplete.copy()
        full_outfit[blank_idx] = target_item_id
        context = context_paths(incomplete, blank_idx, image_paths, args.image_root)

        ground_relative = image_paths[target_item_id]
        ground_path = args.image_root / str(ground_relative)
        if not ground_path.exists():
            raise FileNotFoundError(ground_path)

        gen_record = {
            "image_paths": [str(generated_path.resolve())],
            "cates": np.asarray([target_category], dtype=np.int64),
            "candidate_seeds": [int(final["seed"])],
            "incomplete_outfit": context,
            "outfits": full_outfit,
        }
        grd_record = {
            "image_paths": [str(ground_path.resolve())],
            "cates": np.asarray([target_category], dtype=np.int64),
            "incomplete_outfit": context,
            "outfits": full_outfit,
        }
        gen_data.setdefault(uid, {})[oid] = gen_record
        grd_data.setdefault(uid, {})[oid] = grd_record

    checkpoint_dir = args.output_dir / args.checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint_dir / "gen.npy", gen_data)
    np.save(checkpoint_dir / "grd.npy", grd_data)
    manifest = {
        "source_group": args.group,
        "source_base_seed": args.base_seed,
        "task_count": len(run_dirs),
        "checkpoint": args.checkpoint,
        "selection": "PSP final survivor from metadata.json",
        "regret_computed": False,
    }
    (args.output_dir / "pack_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Saved {checkpoint_dir / 'gen.npy'}")
    print(f"Saved {checkpoint_dir / 'grd.npy'}")


if __name__ == "__main__":
    main()
