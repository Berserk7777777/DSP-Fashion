"""Plot step-7/14/28 heatmaps for the 200-task FITB timestep cache."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path(r"E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_timestep200_personal_analysis"),
    )
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def color(value: float, low: float, high: float) -> tuple[int, int, int]:
    if np.isnan(value):
        return (235, 235, 235)
    fraction = 0.0 if high <= low else float(np.clip((value - low) / (high - low), 0.0, 1.0))
    # A compact perceptual blue -> cyan -> yellow scale without a plotting dependency.
    r = int(30 + 225 * fraction)
    g = int(60 + 180 * fraction)
    b = int(150 - 120 * fraction)
    return r, g, b


def make_candidate_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    steps = [7, 14, 28]
    metrics = [
        ("personal", "Personal"),
        ("compatibility", "Compatibility"),
        ("gt_sim", "GT-Sim"),
        ("gaa", "GAA"),
    ]
    task_ids = sorted({int(row["task_index"]) for row in rows})
    # Seeds are intentionally different for each task.  Use the within-task
    # ordinal (seed 1..8) as the heatmap column instead of the raw seed value;
    # otherwise 200 tasks produce up to 1,600 sparse columns.
    seed_rank = {}
    for task_id in task_ids:
        task_seeds = sorted({int(row["seed"]) for row in rows if int(row["task_index"]) == task_id})
        seed_rank[task_id] = {seed: rank + 1 for rank, seed in enumerate(task_seeds)}
    seeds = list(range(1, max(len(ranks) for ranks in seed_rank.values()) + 1))
    matrices = {}
    ranges = {}
    for metric, _ in metrics:
        matrices[metric] = {}
        all_values = [float(row[metric]) for row in rows if int(row["step"]) in steps]
        ranges[metric] = (min(all_values), max(all_values))
        for step in steps:
            matrix = np.full((len(task_ids), len(seeds)), np.nan, dtype=np.float32)
            for item in rows:
                if int(item["step"]) == step:
                    task_id = int(item["task_index"])
                    rank = seed_rank[task_id][int(item["seed"])]
                    matrix[task_ids.index(task_id), rank - 1] = float(item[metric])
            matrices[metric][step] = matrix

    cell_w, cell_h = 16, 3
    left, top, gap_x, gap_y = 58, 48, 24, 44
    panel_w, panel_h = len(seeds) * cell_w, len(task_ids) * cell_h
    width, height = left + len(metrics) * (panel_w + gap_x) + 20, top + len(steps) * (panel_h + gap_y) + 30
    image = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    for row_index, step in enumerate(steps):
        for col_index, (metric, label) in enumerate(metrics):
            x0 = left + col_index * (panel_w + gap_x)
            y0 = top + row_index * (panel_h + gap_y)
            draw.text((x0, y0 - 24), f"step {step} | {label}", fill=(20, 20, 20))
            matrix = matrices[metric][step]
            low, high = ranges[metric]
            for task_index in range(len(task_ids)):
                for seed_index in range(len(seeds)):
                    fill = color(float(matrix[task_index, seed_index]), low, high)
                    draw.rectangle((x0 + seed_index * cell_w, y0 + task_index * cell_h, x0 + (seed_index + 1) * cell_w - 1, y0 + (task_index + 1) * cell_h - 1), fill=fill)
            for seed_index, seed in enumerate(seeds):
                draw.text((x0 + seed_index * cell_w - 2, y0 + panel_h + 3), str(seed), fill=(40, 40, 40))
            for task_index in range(0, len(task_ids), 25):
                draw.text((x0 - 32, y0 + task_index * cell_h - 4), str(task_ids[task_index]), fill=(40, 40, 40))
    draw.text((left, 14), "FITB candidate heatmaps: task x within-task seed rank", fill=(10, 10, 10))
    image.save(output)


def make_mean_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    steps = [7, 14, 28]
    metrics = [
        ("personal", "Personal"),
        ("compatibility", "Compatibility"),
        ("gt_sim", "GT-Sim"),
        ("gaa", "GAA"),
        ("gaa_color", "GAA-Color"),
        ("gaa_pattern", "GAA-Pattern"),
        ("gaa_material", "GAA-Material"),
        ("gaa_style", "GAA-Style"),
    ]
    matrix = np.asarray([[np.mean([float(row[metric]) for row in rows if int(row["step"]) == step]) for step in steps] for metric, _ in metrics], dtype=np.float32)
    cell_w, cell_h = 140, 34
    left, top = 150, 45
    image = Image.new("RGB", (left + len(steps) * cell_w + 20, top + len(metrics) * cell_h + 30), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.text((left, 14), "Mean metric evolution at pruning checkpoints", fill=(10, 10, 10))
    for column, step in enumerate(steps):
        draw.text((left + column * cell_w + 35, top - 25), f"step {step}", fill=(20, 20, 20))
    low, high = float(matrix.min()), float(matrix.max())
    for row_index, (_, label) in enumerate(metrics):
        draw.text((4, top + row_index * cell_h + 8), label, fill=(20, 20, 20))
        for column in range(len(steps)):
            x0, y0 = left + column * cell_w, top + row_index * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 2, y0 + cell_h - 2), fill=color(float(matrix[row_index, column]), low, high))
            draw.text((x0 + 44, y0 + 9), f"{matrix[row_index, column]:.3f}", fill=(255, 255, 255))
    image.save(output)


def make_survival_heatmap(rows: list[dict[str, str]], output: Path) -> None:
    schedules = sorted({row["schedule"] for row in rows})
    targets = ["terminal_personal_best", "terminal_gt_sim_best", "terminal_gaa_best"]
    labels = ["Personal-best", "GT-Sim-best", "GAA-best"]
    nodes = sorted({int(row["step"]) for row in rows})
    cell_w, cell_h = 48, 32
    left, top = 145, 42
    image = Image.new("RGB", (left + len(nodes) * cell_w + 30, top + len(schedules) * cell_h + 30), (248, 248, 248))
    draw = ImageDraw.Draw(image)
    draw.text((left, 14), "Terminal-best seed survival after Personal pruning", fill=(10, 10, 10))
    for column, node in enumerate(nodes):
        draw.text((left + column * cell_w + 14, top - 22), str(node), fill=(20, 20, 20))
    for row_index, schedule in enumerate(schedules):
        draw.text((4, top + row_index * cell_h + 8), schedule, fill=(20, 20, 20))
        for column, node in enumerate(nodes):
            values = [float(row["survival_rate"]) for row in rows if row["schedule"] == schedule and int(row["step"]) == node and row["target"] == "terminal_personal_best"]
            value = values[0] if values else float("nan")
            x0, y0 = left + column * cell_w, top + row_index * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 2, y0 + cell_h - 2), fill=color(value, 0.0, 1.0))
            if not np.isnan(value):
                draw.text((x0 + 12, y0 + 9), f"{value:.2f}", fill=(255, 255, 255))
    image.save(output)


def main() -> None:
    args = parse_args()
    root = args.analysis_root.resolve()
    output = (args.output_root or root / "mechanism_analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_rows = read_csv(root / "candidate_metrics_with_gt.csv")
    survival_rows = read_csv(output / "seed_survival_summary.csv")
    make_candidate_heatmap(candidate_rows, output / "candidate_heatmaps_step7_14_28.png")
    make_mean_heatmap(candidate_rows, output / "mean_metrics_heatmap_step7_14_28.png")
    make_survival_heatmap(survival_rows, output / "seed_survival_heatmap_step7_14_28.png")
    print(output)


if __name__ == "__main__":
    main()
