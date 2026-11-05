#!/usr/bin/env python
"""Rank frozen Personal GOR selections for qualitative PSP-versus-BoN examples.

This is post-hoc analysis only.  It joins native DiFashion CIS/LPIPS with the
already-computed GT Sim diagnostic, and never affects candidate scoring,
rewarding, or selection.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-metrics", type=Path, required=True)
    parser.add_argument("--gt-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_csv(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["task_id"], row["method"]): row for row in csv.DictReader(handle)}


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    args = parse_args()
    native = load_csv(args.native_metrics)
    gt = load_csv(args.gt_metrics)
    task_ids = sorted({task_id for task_id, _ in native})
    rows = []
    for task_id in task_ids:
        psp, bon, standard = (native[(task_id, method)] for method in ("psp", "bon", "standard"))
        psp_gt, bon_gt, standard_gt = (gt[(task_id, method)] for method in ("psp", "bon", "standard"))
        deltas = {
            "psp_minus_bon_cis": value(psp, "cis") - value(bon, "cis"),
            "psp_minus_bon_lpips_improvement": value(bon, "lpips") - value(psp, "lpips"),
            "psp_minus_bon_gt_sim": value(psp_gt, "gt_sim") - value(bon_gt, "gt_sim"),
            "psp_minus_standard_cis": value(psp, "cis") - value(standard, "cis"),
            "psp_minus_standard_lpips_improvement": value(standard, "lpips") - value(psp, "lpips"),
            "psp_minus_standard_gt_sim": value(psp_gt, "gt_sim") - value(standard_gt, "gt_sim"),
        }
        strict = all(number > 0 for number in deltas.values())
        rows.append({
            "task_id": task_id, "task_index": psp["task_index"], "uid": psp["uid"], "oid": psp["oid"],
            "strict_psp_beats_bon_and_standard_on_cis_lpips_gt_sim": strict,
            "psp_seed": psp["seed"], "bon_seed": bon["seed"], "standard_seed": standard["seed"],
            "psp_cis": value(psp, "cis"), "bon_cis": value(bon, "cis"), "standard_cis": value(standard, "cis"),
            "psp_lpips": value(psp, "lpips"), "bon_lpips": value(bon, "lpips"), "standard_lpips": value(standard, "lpips"),
            "psp_gt_sim": value(psp_gt, "gt_sim"), "bon_gt_sim": value(bon_gt, "gt_sim"), "standard_gt_sim": value(standard_gt, "gt_sim"),
            "psp_personal": value(psp, "personal"), "bon_personal": value(bon, "personal"), "standard_personal": value(standard, "personal"),
            "psp_image_paths": psp["image_paths"], "bon_image_paths": bon["image_paths"],
            "standard_image_paths": standard["image_paths"], "gt_image_paths": psp["gt_image_paths"],
            **deltas,
        })
    # Normalize only for ranking examples. It remains an explicitly post-hoc
    # visualization heuristic, never a reward or reported model metric.
    for metric, higher in (("psp_cis", True), ("psp_lpips", False), ("psp_gt_sim", True),
                           ("psp_minus_bon_cis", True), ("psp_minus_bon_lpips_improvement", True),
                           ("psp_minus_bon_gt_sim", True), ("psp_minus_standard_cis", True),
                           ("psp_minus_standard_lpips_improvement", True), ("psp_minus_standard_gt_sim", True)):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        std = values.std(ddof=0)
        for row in rows:
            row[f"_rank_{metric}"] = ((row[metric] - values.mean()) / std if std else 0.0) * (1.0 if higher else -1.0)
    for row in rows:
        absolute = row["_rank_psp_cis"] + row["_rank_psp_lpips"] + row["_rank_psp_gt_sim"]
        improvement = sum(row[f"_rank_{metric}"] for metric in (
            "psp_minus_bon_cis", "psp_minus_bon_lpips_improvement", "psp_minus_bon_gt_sim",
            "psp_minus_standard_cis", "psp_minus_standard_lpips_improvement", "psp_minus_standard_gt_sim",
        ))
        row["posthoc_example_rank"] = absolute + improvement
        for key in [key for key in row if key.startswith("_rank_")]:
            del row[key]
    rows.sort(key=lambda row: (not row["strict_psp_beats_bon_and_standard_on_cis_lpips_gt_sim"], -row["posthoc_example_rank"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    strict_count = sum(row["strict_psp_beats_bon_and_standard_on_cis_lpips_gt_sim"] for row in rows)
    print(f"Wrote {len(rows)} tasks; strict PSP wins on CIS/LPIPS/GT Sim: {strict_count}")
    for row in rows[:10]:
        print(
            f"{row['task_id']} psp_seed={row['psp_seed']} "
            f"CIS {row['psp_cis']:.3f}/{row['bon_cis']:.3f}/{row['standard_cis']:.3f}; "
            f"LPIPS {row['psp_lpips']:.4f}/{row['bon_lpips']:.4f}/{row['standard_lpips']:.4f}; "
            f"GT {row['psp_gt_sim']:.4f}/{row['bon_gt_sim']:.4f}/{row['standard_gt_sim']:.4f}"
        )


if __name__ == "__main__":
    main()
