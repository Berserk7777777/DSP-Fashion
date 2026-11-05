"""Create a labelled contact sheet from score_bon_compatibility.py output."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=256)
    return parser.parse_args()


def load_font(size):
    for candidate in ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    args = parse_args()
    with (args.report_dir / "candidate_compatibility_scores.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    winners = [row for row in rows if row["is_bon_winner"] == "True"]
    winners.sort(key=lambda row: (int(row["group_id"]), int(row["uid"]), int(row["oid"])))

    tile = args.tile_size
    label_height = 44
    rows_count = (len(winners) + args.columns - 1) // args.columns
    canvas = Image.new("RGB", (args.columns * tile, rows_count * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = load_font(15)
    small_font = load_font(13)

    for index, record in enumerate(winners):
        image = Image.open(record["image_path"]).convert("RGB")
        image.thumbnail((tile, tile))
        x = (index % args.columns) * tile + (tile - image.width) // 2
        y = (index // args.columns) * (tile + label_height) + (tile - image.height) // 2
        canvas.paste(image, (x, y))
        label_y = (index // args.columns) * (tile + label_height) + tile + 3
        label = f"{record['uid']}_{record['oid']} G{record['group_id']}"
        detail = f"seed {record['seed']}  reward {float(record['compatibility_reward']):.4f}"
        draw.text(((index % args.columns) * tile + 4, label_y), label, fill="black", font=font)
        draw.text(((index % args.columns) * tile + 4, label_y + 17), detail, fill="black", font=small_font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
