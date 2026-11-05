"""Summarize the official DualFashion PFITB evaluation outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = ["IS-Acc", "IS", "Personal Sim", "Compatibility", "Grd Compatibility", "LPIP score"]
EXPERIMENTS = [
    ("Standard", 0, 42, Path("bon4_group0_seed42_scores/selected_for_official_eval/standard/group_1/eval_results.npy")),
    ("Standard", 1, 1000042, Path("bon4_group1_seed1000042_scores/selected_for_official_eval/standard/group_1/eval_results.npy")),
    ("Standard", 2, 2000042, Path("bon4_group2_seed2000042_scores/selected_for_official_eval/standard/group_1/eval_results.npy")),
    ("BoN", 0, 42, Path("bon4_group0_seed42_scores/selected_for_official_eval/bon/group_1/eval_results.npy")),
    ("BoN", 1, 1000042, Path("bon4_group1_seed1000042_scores/selected_for_official_eval/bon/group_1/eval_results.npy")),
    ("BoN", 2, 2000042, Path("bon4_group2_seed2000042_scores/selected_for_official_eval/bon/group_1/eval_results.npy")),
    ("PSP", 0, 2026, Path("psp_group0_official_eval/eval_results.npy")),
    ("PSP", 1, 1002026, Path("psp_group1_official_eval/eval_results.npy")),
    ("PSP", 2, 2002026, Path("psp_group2_official_eval/eval_results.npy")),
]


def build_experiments(output_root: Path):
    """Return experiment paths for either the legacy or final output layout."""
    if (output_root / "bon").is_dir() and (output_root / "psp").is_dir():
        bon_seeds = (42, 1000042, 2000042)
        psp_seeds = (2026, 1002026, 2002026)
        experiments = []
        for group, seed in enumerate(bon_seeds):
            base = output_root / "bon" / f"bon_group{group}_seed{seed}_scores" / "selected_for_official_eval"
            experiments.append(("Standard", group, seed, base / "standard" / "group_1" / "eval_results.npy"))
            experiments.append(("BoN", group, seed, base / "bon" / "group_1" / "eval_results.npy"))
        for group, seed in enumerate(psp_seeds):
            experiments.append(("PSP", group, seed, output_root / "psp" / f"psp_group{group}_official_eval" / "eval_results.npy"))
        return experiments

    return [(method, group, seed, output_root / relative_path) for method, group, seed, relative_path in EXPERIMENTS]


def load_metrics(path: Path) -> dict[str, float]:
    payload = np.load(path, allow_pickle=True).item()
    checkpoints = [key for key in payload if isinstance(payload[key], dict)]
    if "0057501" in payload:
        record = payload["0057501"]
    elif len(checkpoints) == 1:
        record = payload[checkpoints[0]]
    else:
        raise ValueError(f"Cannot identify one checkpoint in {path}")
    return {metric: float(record[metric]) for metric in METRICS}


def main() -> None:
    repo = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo / "output",
        help="Root containing either the legacy evaluation folders or ifashion_fitb_comp/bon and /psp.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for the generated CSV/Markdown reports (defaults next to the input root).",
    )
    args = parser.parse_args()

    output_root = args.output_root if args.output_root.is_absolute() else repo / args.output_root
    output_dir = args.report_dir
    if output_dir is None:
        output_dir = (
            output_root / "official_eval_summary"
            if output_root.name == "ifashion_fitb_comp"
            else repo / "output" / "compatibility_bon_psp_analysis"
        )
    elif not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    group_rows = []
    by_method: dict[str, list[dict[str, float]]] = {"Standard": [], "BoN": [], "PSP": []}
    experiments = build_experiments(output_root)
    for method, group, base_seed, path in experiments:
        if not path.exists():
            raise FileNotFoundError(path)
        values = load_metrics(path)
        row = {"method": method, "group": group, "base_seed": base_seed, "result_path": str(path.resolve())}
        row.update(values)
        group_rows.append(row)
        by_method[method].append(values)

    fields = ["method", "group", "base_seed", "result_path", *METRICS]
    csv_path = output_dir / "official_eval_bon_psp_by_group.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(group_rows)

    summary_rows = []
    for method in ("Standard", "BoN", "PSP"):
        matrix = np.asarray([[row[metric] for metric in METRICS] for row in by_method[method]], dtype=float)
        summary_rows.append(
            {
                "method": method,
                "group": "mean",
                "base_seed": "42/1000042/2000042" if method in {"Standard", "BoN"} else "2026/1002026/2002026",
                "result_path": "",
                **{metric: float(matrix[:, index].mean()) for index, metric in enumerate(METRICS)},
            }
        )
        summary_rows.append(
            {
                "method": method,
                "group": "std_population",
                "base_seed": "",
                "result_path": "",
                **{metric: float(matrix[:, index].std()) for index, metric in enumerate(METRICS)},
            }
        )

    summary_csv = output_dir / "official_eval_bon_psp_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    markdown_path = output_dir / "official_eval_bon_psp_table.md"
    lines = [
        "# DualFashion official PFITB evaluation",
        "",
        "Values are means over the ten evaluated tasks in each seed group; the final two rows per method are the three-group mean and population standard deviation.",
        "",
        "| Method | Group / base seed | IS-Acc | IS | Personal Sim | Compatibility | Grd Compatibility | LPIP score |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(
            f"| {row['method']} | {row['group']} / {row['base_seed']} | "
            f"{row['IS-Acc']:.4f} | {row['IS']:.4f} | {row['Personal Sim']:.4f} | "
            f"{row['Compatibility']:.4f} | {row['Grd Compatibility']:.4f} | {row['LPIP score']:.4f} |"
        )
    for row in summary_rows:
        label = "3-group mean" if row["group"] == "mean" else "3-group std (population)"
        lines.append(
            f"| {row['method']} | {label} | "
            f"{row['IS-Acc']:.4f} | {row['IS']:.4f} | {row['Personal Sim']:.4f} | "
            f"{row['Compatibility']:.4f} | {row['Grd Compatibility']:.4f} | {row['LPIP score']:.4f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(csv_path)
    print(summary_csv)
    print(markdown_path)


if __name__ == "__main__":
    main()
