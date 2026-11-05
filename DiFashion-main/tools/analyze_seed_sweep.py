#!/usr/bin/env python
"""Compare DiFashion inference outputs generated with different noise seeds."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Directories produced by inf4eval.py, one directory per seed.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the report artifacts.")
    parser.add_argument("--max-items", type=int, default=32, help="Maximum aligned samples to include.")
    return parser.parse_args()


def seed_from_name(path):
    match = re.search(r"(?:^|-)seed(\d+)(?:$|-)", path.name)
    if not match:
        raise ValueError(f"Cannot infer seed from directory name: {path}")
    return int(match.group(1))


def load_run(path):
    result_files = list(path.glob("*.npy"))
    if len(result_files) != 1:
        raise ValueError(f"Expected exactly one .npy result file in {path}, found {len(result_files)}")
    result = np.load(result_files[0], allow_pickle=True).item()
    samples = {}
    for uid, outfits in result.items():
        for oid, data in outfits.items():
            for index, image_path in enumerate(data["image_paths"]):
                samples[(str(uid), str(oid), index)] = Path(image_path)
    return samples


def image_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def make_grid(images, labels, output_path):
    width, height = images[0].size
    label_height = 24
    canvas = Image.new("RGB", (width * len(images), height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (image, label) in enumerate(zip(images, labels)):
        canvas.paste(image, (index * width, label_height))
        draw.text((index * width + 4, 4), label, fill="black")
    canvas.save(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    grids_dir = output_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    runs = {seed_from_name(Path(value)): load_run(Path(value)) for value in args.run_dirs}
    if len(runs) < 2:
        raise ValueError("At least two different seed directories are required.")

    common_keys = set.intersection(*(set(samples) for samples in runs.values()))
    if not common_keys:
        raise ValueError("The supplied runs have no aligned (uid, oid, item) outputs.")

    rows = []
    pairwise_mae = defaultdict(list)
    seeds = sorted(runs)
    for item_index, key in enumerate(sorted(common_keys)[: args.max_items]):
        arrays = {seed: image_array(runs[seed][key]) for seed in seeds}
        shapes = {array.shape for array in arrays.values()}
        if len(shapes) != 1:
            raise ValueError(f"Image dimensions differ for sample {key}: {shapes}")

        for left_index, left_seed in enumerate(seeds):
            for right_seed in seeds[left_index + 1 :]:
                mae = float(np.abs(arrays[left_seed] - arrays[right_seed]).mean())
                rmse = float(np.sqrt(np.square(arrays[left_seed] - arrays[right_seed]).mean()))
                pair = f"{left_seed}_vs_{right_seed}"
                pairwise_mae[pair].append(mae)
                rows.append(
                    {
                        "uid": key[0],
                        "oid": key[1],
                        "item_index": key[2],
                        "seed_a": left_seed,
                        "seed_b": right_seed,
                        "mae": mae,
                        "rmse": rmse,
                    }
                )

        images = []
        for seed in seeds:
            with Image.open(runs[seed][key]) as image:
                images.append(image.convert("RGB").copy())
        make_grid(images, [f"seed {seed}" for seed in seeds], grids_dir / f"{item_index:04d}_{key[0]}_{key[1]}_{key[2]}.jpg")

    with (output_dir / "per_item_pairwise.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["uid", "oid", "item_index", "seed_a", "seed_b", "mae", "rmse"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "seeds": seeds,
        "aligned_items": min(len(common_keys), args.max_items),
        "available_aligned_items": len(common_keys),
        "mean_pairwise_mae": {pair: float(np.mean(values)) for pair, values in pairwise_mae.items()},
        "nonzero_pairwise_mae_count": {
            pair: int(np.count_nonzero(np.asarray(values) > 0.0)) for pair, values in pairwise_mae.items()
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
