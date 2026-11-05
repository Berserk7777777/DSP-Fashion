#!/usr/bin/env python
"""Export frozen DiFashion GOR GT/BoN/PSP example images for qualitative review."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
from pathlib import Path

from PIL import Image


DEFAULT_TASKS = ("00005_9242_17896", "00179_9410_18070", "00103_9334_17997", "00034_9263_17924")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", action="append", dest="task_ids", default=None,
                        help="Repeat to export specific tasks. Defaults to four curated PSP examples.")
    return parser.parse_args()


def paths(value: str) -> list[Path]:
    parsed = ast.literal_eval(value)
    result = [Path(path) for path in parsed]
    if len(result) != 4:
        raise ValueError(f"Expected four item images, found {len(result)}")
    for path in result:
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def make_grid(source: list[Path], target: Path) -> None:
    images = []
    try:
        for path in source:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        width = max(image.width for image in images)
        height = max(image.height for image in images)
        canvas = Image.new("RGB", (width * 2, height * 2), "white")
        for index, image in enumerate(images):
            copy = image.copy()
            copy.thumbnail((width, height), Image.Resampling.LANCZOS)
            x = (index % 2) * width + (width - copy.width) // 2
            y = (index // 2) * height + (height - copy.height) // 2
            canvas.paste(copy, (x, y))
        canvas.save(target)
    finally:
        for image in images:
            image.close()


def copy_group(name: str, source: list[Path], directory: Path, make_source_grid: bool) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    copied = []
    for index, path in enumerate(source):
        target = directory / f"item_{index}.png"
        shutil.copy2(path, target)
        copied.append(str(target.resolve()))
    all_path = directory / "all.png"
    source_all = source[0].parent / "all.png"
    if make_source_grid and source_all.is_file():
        shutil.copy2(source_all, all_path)
    else:
        make_grid(source, all_path)
    return {"source": [str(path.resolve()) for path in source], "exported": copied,
            "all": str(all_path.resolve()), "group": name}


def main() -> None:
    args = parse_args()
    requested = tuple(args.task_ids or DEFAULT_TASKS)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {args.output_root}")
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidate_rows = {row["task_id"]: row for row in csv.DictReader(handle)}
    missing = [task_id for task_id in requested if task_id not in candidate_rows]
    if missing:
        raise KeyError(f"Task IDs missing from candidate table: {missing}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = {"purpose": "qualitative export only; no GT/reward/selection changes", "task_ids": list(requested), "tasks": []}
    for task_id in requested:
        row = candidate_rows[task_id]
        task_root = args.output_root / task_id
        gt = copy_group("GT", paths(row["gt_image_paths"]), task_root / "GT", make_source_grid=False)
        bon = copy_group("BoN", paths(row["bon_image_paths"]), task_root / "bon", make_source_grid=True)
        psp = copy_group("PSP", paths(row["psp_image_paths"]), task_root / "psp", make_source_grid=True)
        metrics = {key: row[key] for key in row if key not in {"psp_image_paths", "bon_image_paths", "standard_image_paths", "gt_image_paths"}}
        record = {"task_id": task_id, "metrics": metrics, "images": {"gt": gt, "bon": bon, "psp": psp}}
        (task_root / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        manifest["tasks"].append(record)
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Exported {len(requested)} tasks to {args.output_root}")


if __name__ == "__main__":
    main()
