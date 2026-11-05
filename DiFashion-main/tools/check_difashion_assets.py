#!/usr/bin/env python
"""Check whether local DiFashion reproduction assets are in place."""

import argparse
from pathlib import Path


DATASET_FILES = [
    "new_id_cate_dict.npy",
    "new_all_item_image_paths.npy",
    "cnn_features_clip.npy",
    "fitb_valid_retrieval_candidates.npy",
    "fitb_test_retrieval_candidates.npy",
    "fitb_valid_dict.npy",
    "fitb_test_dict.npy",
    "valid_grd.npy",
    "test_grd.npy",
]

PROCESSED_FILES = [
    "new_train.npy",
    "new_fitb_valid.npy",
    "new_fitb_test.npy",
    "train_hist_latents.npy",
    "valid_hist_latents.npy",
    "test_hist_latents.npy",
    "valid_history_clipembs.npy",
    "test_history_clipembs.npy",
]

CHECKPOINT_DIRS = [
    "checkpoint-15000/unet",
    "checkpoint-15000/unet_ema",
    "checkpoint-15000/fashion_encoder",
    "checkpoint-15000/fashion_encoder_ema",
]


def existing_or_missing(root, rel_paths):
    rows = []
    for rel_path in rel_paths:
        path = root / rel_path
        rows.append((rel_path, path.exists(), path))
    return rows


def print_rows(title, rows):
    print(f"\n[{title}]")
    for rel_path, ok, path in rows:
        state = "OK" if ok else "MISSING"
        print(f"{state:7} {rel_path} -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True, help="Dataset directory, e.g. .../ifashion")
    parser.add_argument("--img-folder-path", required=True, help="Root folder containing item images")
    parser.add_argument("--output-dir", required=True, help="Released checkpoint root used by inf4eval.py")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    img_folder_path = Path(args.img_folder_path)
    output_dir = Path(args.output_dir)

    dataset_rows = existing_or_missing(data_path, DATASET_FILES)
    processed_rows = existing_or_missing(data_path / "processed", PROCESSED_FILES)
    checkpoint_rows = existing_or_missing(output_dir, CHECKPOINT_DIRS)
    image_rows = [("empty_image.png", (img_folder_path / "empty_image.png").exists(), img_folder_path / "empty_image.png")]

    print_rows("dataset", dataset_rows)
    print_rows("processed", processed_rows)
    print_rows("images", image_rows)
    print_rows("checkpoint", checkpoint_rows)

    all_rows = dataset_rows + processed_rows + image_rows + checkpoint_rows
    missing = [path for _, ok, path in all_rows if not ok]
    print(f"\nsummary: {len(all_rows) - len(missing)} OK, {len(missing)} missing")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
