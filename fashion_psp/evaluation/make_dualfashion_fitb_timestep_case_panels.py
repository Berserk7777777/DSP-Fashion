"""Make paper-style FITB case panels for step 7/14/28.

Each panel keeps the same eight initial seeds in the same columns.  Rows are
the decoded x0 predictions at steps 7, 14, and 28.  Cell color and the score
label encode Personal; the image remains the primary evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


STEPS = (7, 14, 28)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root", type=Path, default=Path(r"E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_timestep200_personal_analysis"))
    p.add_argument("--cache-root", type=Path, default=Path(r"E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_timestep200_cache"))
    p.add_argument("--data-root", type=Path, default=Path(r"C:\tmp\ifashion"))
    p.add_argument("--processed-info-root", type=Path, default=Path(r"E:\difashion-PSP\DualFashion-main\processed_info"))
    p.add_argument("--image-root", type=Path, default=Path(r"C:\tmp\ifashion\semantic_category"))
    # Representative cases with a clear coarse-to-fine transition and a
    # change in the Personal-selected seed (also visually close to GT).
    p.add_argument("--task-indices", type=int, nargs="+", default=[98, 115, 33])
    p.add_argument("--output-root", type=Path, default=None)
    return p.parse_args()


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def score_color(value: float, low: float, high: float) -> tuple[int, int, int]:
    ratio = 0.5 if high <= low else float(np.clip((value - low) / (high - low), 0, 1))
    # Dark blue -> cyan -> yellow; readable against the white image frame.
    return int(30 + 225 * ratio), int(65 + 180 * ratio), int(160 - 115 * ratio)


def read_rows(path: Path) -> dict[tuple[int, int], list[dict[str, str]]]:
    out: dict[tuple[int, int], list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["task_index"]), int(row["step"]))
            out.setdefault(key, []).append(row)
    for values in out.values():
        values.sort(key=lambda row: int(row["seed"]))
    return out


def load_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (248, 248, 248))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def gt_path(task: int, data_root: Path, processed_info_root: Path, image_root: Path) -> tuple[int, int, Path]:
    fitb = np.load(data_root / "processed" / "fitb_test.npy", allow_pickle=True).item()
    grd = np.load(data_root / "test_grd_dict.npy", allow_pickle=True).item()
    item_paths = np.load(processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    uid, oid = int(fitb["uids"][task]), int(fitb["oids"][task])
    incomplete = [int(value) for value in fitb["outfits"][task]]
    blank = incomplete.index(0)
    item_id = int(grd["outfits"][task][blank])
    return uid, oid, image_root / str(item_paths[item_id])


def draw_panel(task: int, rows: dict[tuple[int, int], list[dict[str, str]]], data_root: Path, processed_info_root: Path, image_root: Path, output: Path) -> None:
    uid, oid, gt = gt_path(task, data_root, processed_info_root, image_root)
    cell = 132
    left = 150
    top = 82
    label_h = 34
    row_gap = 32
    width = left + 8 * cell + 24
    height = top + 3 * (cell + label_h) + 2 * row_gap + 38
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = font(24), font(15), font(13)
    draw.text((18, 18), f"FITB task {task:04d} | uid {uid} / oid {oid}", fill=(25, 25, 25), font=title_font)
    draw.text((18, 52), "GT", fill=(25, 25, 25), font=label_font)
    canvas.paste(load_image(gt, cell), (18, top))
    draw.rectangle((17, top - 1, 18 + cell, top + cell), outline=(40, 40, 40), width=2)
    draw.text((left, 52), "same columns = same initial seed; color encodes Personal", fill=(65, 65, 65), font=small_font)

    scores = [float(r["personal"]) for key in rows if key[0] == task for r in rows[key] if int(r["step"]) in STEPS]
    low, high = min(scores), max(scores)
    for row_index, step in enumerate(STEPS):
        values = rows[(task, step)]
        y = top + row_index * (cell + label_h + row_gap)
        draw.text((18, y + cell + 8), f"step {step}", fill=(25, 25, 25), font=label_font)
        best = max(values, key=lambda row: float(row["personal"]))
        for col, row in enumerate(values):
            x = left + col * cell
            image = load_image(Path(row["image_path"]), cell - 8)
            canvas.paste(image, (x + 4, y + 4))
            value = float(row["personal"])
            border = score_color(value, low, high)
            width_border = 5 if int(row["seed"]) == int(best["seed"]) else 2
            draw.rectangle((x + 1, y + 1, x + cell - 2, y + cell - 2), outline=border, width=width_border)
            draw.rectangle((x + 4, y + cell - 25, x + cell - 4, y + cell - 5), fill=border)
            draw.text((x + 9, y + cell - 23), f"s{col + 1}  {value:.1f}", fill=(255, 255, 255), font=small_font)
            draw.text((x + 6, y + cell + 8), "best" if int(row["seed"]) == int(best["seed"]) else f"s{col + 1}", fill=border if int(row["seed"]) == int(best["seed"]) else (65, 65, 65), font=small_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parsed = args()
    root = parsed.analysis_root.resolve()
    output_root = (parsed.output_root or root / "mechanism_analysis" / "case_panels_step7_14_28").resolve()
    rows = read_rows(root / "candidate_metrics_with_gt.csv")
    manifest = {"steps": list(STEPS), "task_indices": parsed.task_indices, "personal_is_visual_encoding_only": True}
    output_root.mkdir(parents=True, exist_ok=True)
    panel_paths = []
    for task in parsed.task_indices:
        for step in STEPS:
            if (task, step) not in rows or len(rows[(task, step)]) != 8:
                raise ValueError(f"Expected 8 candidates for task={task}, step={step}")
        panel_path = output_root / f"task_{task:04d}_step7_14_28.png"
        draw_panel(task, rows, parsed.data_root, parsed.processed_info_root, parsed.image_root, panel_path)
        panel_paths.append(panel_path)
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    if panels:
        combined = Image.new("RGB", (max(image.width for image in panels), sum(image.height for image in panels) + 24 * (len(panels) - 1)), (250, 250, 250))
        y = 0
        for index, image in enumerate(panels):
            combined.paste(image, (0, y))
            y += image.height
            if index + 1 < len(panels):
                y += 24
        combined.save(output_root / "selected_cases_step7_14_28.png")
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output_root)


if __name__ == "__main__":
    main()
