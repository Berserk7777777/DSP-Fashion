"""Join official FITB metrics with PCAR's GT-Sim and GAA diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


OFFICIAL_METRICS = ("IS-Acc", "IS", "Personal Sim", "Compatibility", "Grd Compatibility", "LPIP score")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def official_metrics(path: Path) -> dict[str, float]:
    payload = np.load(path, allow_pickle=True).item()
    records = [value for value in payload.values() if isinstance(value, dict)]
    if len(records) != 1:
        raise ValueError(f"cannot identify one official result in {path}")
    return {name: float(records[0][name]) for name in OFFICIAL_METRICS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gt-sim-summary", type=Path, required=True)
    parser.add_argument("--gaa-summary", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.output_root.resolve()
    paths = {
        "standard": root / "bon" / "bon_group0_seed42_scores" / "selected_for_official_eval" / "standard" / "group_1" / "eval_results.npy",
        "bon": root / "bon" / "bon_group0_seed42_scores" / "selected_for_official_eval" / "bon" / "group_1" / "eval_results.npy",
        "psp": root / "psp" / "psp_group0_pcar" / "eval_results.npy",
    }
    missing = [path for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    gt_sim = {(row["reward"], row["method"]): row for row in read_csv(args.gt_sim_summary)}
    gaa = {(row["reward"], row["method"]): row for row in read_csv(args.gaa_summary)}
    rows = []
    for method in ("standard", "bon", "psp"):
        row = {"reward": "pcar", "method": method, "official_result": str(paths[method])}
        row.update(official_metrics(paths[method]))
        gt_row = gt_sim[("pcar", method)]
        gaa_row = gaa[("pcar", method)]
        row.update({
            "gt_sim": float(gt_row["gt_similarity_mean"]),
            "gaa": float(gaa_row["gaa_mean"]),
            "gaa_color": float(gaa_row["gaa_color_mean"]),
            "gaa_pattern": float(gaa_row["gaa_pattern_mean"]),
            "gaa_material": float(gaa_row["gaa_material_mean"]),
            "gaa_style": float(gaa_row["gaa_style_mean"]),
        })
        rows.append(row)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_dir / "pcar_fitb_all_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown = [
        "# DualFashion iFashion FITB: PCAR", "",
        "One frozen offline group (BoN base seed 42; PSP base seed 2026). GT-Sim and GAA are post-selection diagnostics only.", "",
        "| Method | IS-Acc | IS | Personal | Compatibility | Grd Compatibility | LPIPS | GT-Sim | GAA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['method']} | {row['IS-Acc']:.4f} | {row['IS']:.4f} | {row['Personal Sim']:.4f} | "
            f"{row['Compatibility']:.4f} | {row['Grd Compatibility']:.4f} | {row['LPIP score']:.4f} | "
            f"{row['gt_sim']:.4f} | {row['gaa']:.4f} |"
        )
    (args.report_dir / "pcar_fitb_all_metrics.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(csv_path)


if __name__ == "__main__":
    main()
