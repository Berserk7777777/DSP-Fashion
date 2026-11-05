"""Render GT / item-greedy BoN / item-greedy PSP comparisons for GOR tasks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="Comma-separated task indices")
    parser.add_argument("--input-root", type=Path, default=repo / "DualFashion-main" / "output" / "ifashion_gor_greedy_personal")
    parser.add_argument("--native-metrics", type=Path, default=None)
    parser.add_argument("--evaluation-root", type=Path, default=None)
    parser.add_argument("--processed-info-root", type=Path, default=repo / "DualFashion-main" / "processed_info")
    parser.add_argument("--image-root", type=Path, default=Path(r"C:\tmp\ifashion\semantic_category"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-ready-root", type=Path, default=None,
                        help="Optional root for GT/BoN/PSP copies organized by task and original outfit position.")
    parser.add_argument("--trajectory-standard-tasks-root", type=Path, default=None,
                        help="Optional .../evaluation/bon/tasks/personal root to add the frozen trajectory Standard column.")
    parser.add_argument("--tile-size", type=int, default=256)
    return parser.parse_args()


def metric_rows(path: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return (
        {int(row["task_index"]): row for row in rows if row["label"] == "bon"},
        {int(row["task_index"]): row for row in rows if row["label"] == "psp"},
    )


def evaluation_rows(root: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    def load(method: str) -> dict[int, dict]:
        with (root / method / "per_task_metrics.csv").open(newline="", encoding="utf-8") as handle:
            return {int(row["task_index"]): row for row in csv.DictReader(handle)}
    return load("bon"), load("psp")


def fit_image(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (size, size), "white")
    result.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return result


def main() -> None:
    args = parse_args()
    tasks = [int(value.strip()) for value in args.tasks.split(",") if value.strip()]
    if not tasks:
        raise ValueError("--tasks must contain at least one task index")
    native_path = args.native_metrics or args.input_root / "native_metrics" / "per_task_metrics.csv"
    eval_root = args.evaluation_root or args.input_root / "evaluation"
    bon_native, psp_native = metric_rows(native_path)
    bon_eval, psp_eval = evaluation_rows(eval_root)
    image_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    font = ImageFont.load_default()
    tile, margin, row_label, header = args.tile_size, 12, 160, 68
    args.output_root.mkdir(parents=True, exist_ok=True)

    for task_index in tasks:
        matches = sorted(args.input_root.joinpath("bon").glob(f"task{task_index:04d}_*/task_summary.json"))
        psp_matches = sorted(args.input_root.joinpath("psp").glob(f"task{task_index:04d}_*/task_summary.json"))
        if len(matches) != 1 or len(psp_matches) != 1:
            raise FileNotFoundError(f"task {task_index}: expected one BoN and PSP task summary")
        bon = json.loads(matches[0].read_text(encoding="utf-8"))
        psp = json.loads(psp_matches[0].read_text(encoding="utf-8"))
        if (bon["uid"], bon["oid"]) != (psp["uid"], psp["oid"]):
            raise ValueError(f"task {task_index}: BoN/PSP identity mismatch")
        bon_items = sorted(bon["items"], key=lambda item: int(item["original_outfit_position"]))
        psp_items = sorted(psp["items"], key=lambda item: int(item["original_outfit_position"]))
        standard_paths = None
        if args.trajectory_standard_tasks_root is not None:
            standard_path = args.trajectory_standard_tasks_root / f"task_{task_index:04d}.json"
            if not standard_path.is_file():
                raise FileNotFoundError(standard_path)
            standard = json.loads(standard_path.read_text(encoding="utf-8"))
            if (int(standard["uid"]), int(standard["oid"])) != (int(bon["uid"]), int(bon["oid"])):
                raise ValueError(f"task {task_index}: trajectory Standard identity mismatch")
            standard_paths = [Path(value) for value in standard["standard"]["image_paths"]]
            if len(standard_paths) != 4:
                raise ValueError(f"task {task_index}: expected four trajectory Standard images")
        columns = ("GT", "Trajectory Standard", "Item-Greedy BoN", "Item-Greedy PSP") if standard_paths else (
            "GT", "Item-Greedy BoN", "Item-Greedy PSP"
        )
        width = row_label + len(columns) * tile + (len(columns) + 1) * margin
        height = header + 4 * (tile + margin + 16) + margin
        canvas = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(canvas)
        b, p, be, pe = bon_native[task_index], psp_native[task_index], bon_eval[task_index], psp_eval[task_index]
        draw.text((margin, margin), f"Task {task_index}: uid {bon['uid']} / outfit {bon['oid']}", fill="black", font=font)
        draw.text((margin, margin + 14),
                  f"PSP vs BoN | CIS {float(p['cis']):.2f} ({float(p['cis'])-float(b['cis']):+.2f}) | "
                  f"LPIPS {float(p['lpips']):.3f} ({float(p['lpips'])-float(b['lpips']):+.3f}) | "
                  f"GT-Sim {float(pe['gt_sim']):.3f} ({float(pe['gt_sim'])-float(be['gt_sim']):+.3f}) | "
                  f"GAA {float(pe['gaa']):.3f} ({float(pe['gaa'])-float(be['gaa']):+.3f})",
                  fill="black", font=font)
        for column, name in enumerate(columns):
            x = row_label + margin + column * (tile + margin)
            draw.text((x, header - 16), name, fill="black", font=font)
        for position, (bon_item, psp_item) in enumerate(zip(bon_items, psp_items)):
            if int(bon_item["item_id"]) != int(psp_item["item_id"]):
                raise ValueError(f"task {task_index}: item order mismatch at {position}")
            y = header + position * (tile + margin + 16)
            item_id, category = int(psp_item["item_id"]), int(psp_item["category_id"])
            draw.text((margin, y), f"pos {position}\nitem {item_id}\ncategory {category}", fill="black", font=font)
            gt = args.image_root / str(image_paths[item_id])
            panel_paths = ([gt] + (standard_paths[position : position + 1] if standard_paths else []) + [
                Path(bon_item["selected_image_path"]), Path(psp_item["selected_image_path"])
            ])
            for column, path in enumerate(panel_paths):
                x = row_label + margin + column * (tile + margin)
                canvas.paste(fit_image(path, tile), (x, y))
        gallery_path = args.output_root / f"task{task_index:04d}_{bon['uid']}_{bon['oid']}_gt_bon_psp.png"
        canvas.save(gallery_path)
        if args.copy_ready_root is not None:
            task_root = args.copy_ready_root / f"task{task_index:04d}_{bon['uid']}_{bon['oid']}"
            for name in ("gt", "standard", "bon", "psp"):
                (task_root / name).mkdir(parents=True, exist_ok=True)
            for position, (bon_item, psp_item) in enumerate(zip(bon_items, psp_items)):
                item_id = int(psp_item["item_id"])
                gt = args.image_root / str(image_paths[item_id])
                stem = f"pos{position:02d}_item{item_id}"
                shutil.copy2(gt, task_root / "gt" / f"{stem}_GT.png")
                if standard_paths:
                    shutil.copy2(standard_paths[position], task_root / "standard" / f"{stem}_Standard.png")
                shutil.copy2(Path(bon_item["selected_image_path"]), task_root / "bon" / f"{stem}_BoN.png")
                shutil.copy2(Path(psp_item["selected_image_path"]), task_root / "psp" / f"{stem}_PSP.png")
            shutil.copy2(gallery_path, task_root / "comparison_GT_BoN_PSP.png")


if __name__ == "__main__":
    main()
