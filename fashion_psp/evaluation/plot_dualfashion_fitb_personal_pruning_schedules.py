"""Render all fair DualFashion FITB Personal pruning schedules in one figure.

Each step profile is a legal schedule from the offline cache search.  A schedule
is included only when its particle-step cost is at most the BoN@4 budget
(``4 * total_steps``).  The plot deliberately uses one global color scale so
that schedules with different initial particle counts are comparable.
"""

from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
import numpy as np


def parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError(f"empty integer list: {value!r}")
    return values


def parse_list(value: str) -> tuple[int, ...]:
    """Read CSV list fields written by ``csv.DictWriter`` safely."""
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"expected a list, got {value!r}")
    return tuple(int(item) for item in parsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-name", default="pruning_schedules_personal_merged")
    parser.add_argument("--total-steps", type=int, default=28)
    parser.add_argument("--default-initial-particles", type=int, default=8)
    parser.add_argument("--default-cutoffs", default="7,14")
    parser.add_argument("--default-remaining", default="4,2")
    parser.add_argument("--max-plotted", type=int, default=0, help="0 plots every legal schedule.")
    parser.add_argument("--jitter", type=float, default=0.012)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["initial_particles"] = int(row["initial_particles"])
            row["cutoff_times"] = parse_list(row["cutoff_times"])
            row["remaining_seeds"] = parse_list(row["remaining_seeds"])
            row["particle_steps"] = int(row["particle_steps"])
            row["selected_personal"] = float(row["selected_personal"])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No strategies found in {path}")
    return rows


def signature(row: dict) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    return row["initial_particles"], row["cutoff_times"], row["remaining_seeds"]


def profile(row: dict, total_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the left-continuous particle-count staircase in normalized time."""
    x = [0.0]
    y = [float(row["initial_particles"])]
    active = row["initial_particles"]
    for cutoff, keep in zip(row["cutoff_times"], row["remaining_seeds"]):
        at = cutoff / total_steps
        x.extend((at, at))
        y.extend((float(active), float(keep)))
        active = keep
    x.append(1.0)
    y.append(float(active))
    return np.asarray(x), np.asarray(y)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv.resolve())
    budget = 4 * args.total_steps
    invalid = [row for row in rows if row["particle_steps"] > budget]
    if invalid:
        raise ValueError(f"summary contains {len(invalid)} schedules above the {budget} particle-step budget")

    # Keeping all schedules is the default.  An optional cap is useful only
    # when inspecting much denser searches interactively.
    if args.max_plotted > 0 and len(rows) > args.max_plotted:
        rows = sorted(rows, key=lambda row: row["selected_personal"], reverse=True)[: args.max_plotted]

    default_signature = (
        args.default_initial_particles,
        parse_ints(args.default_cutoffs),
        parse_ints(args.default_remaining),
    )
    default_row = next((row for row in rows if signature(row) == default_signature), None)
    best_row = max(rows, key=lambda row: (row["selected_personal"], -row["particle_steps"]))
    scores = np.asarray([row["selected_personal"] for row in rows], dtype=np.float64)
    lo, hi = float(scores.min()), float(scores.max())
    if np.isclose(lo, hi):
        hi = lo + 1e-8
    norm = colors.Normalize(vmin=lo, vmax=hi)
    cmap = plt.colormaps["viridis"]

    figure, axis = plt.subplots(figsize=(11.1, 6.6), constrained_layout=True)
    rng = np.random.default_rng(20261019)
    draw_rows = sorted(rows, key=lambda row: row["selected_personal"])
    for row in draw_rows:
        is_best = signature(row) == signature(best_row)
        is_default = signature(row) == default_signature
        x, y = profile(row, args.total_steps)
        if not is_best and not is_default:
            # Fixed per-line jitter exposes overlapping schedules without
            # changing their pruning order or their semantic interpretation.
            x = np.clip(x + rng.uniform(-args.jitter, args.jitter), 0.0, 1.0)
            y = np.maximum(y + rng.uniform(-0.055, 0.055), 1.0)
        if is_best:
            axis.step(x, y, where="post", color="#d62728", linewidth=2.5, alpha=1.0, zorder=5)
        elif is_default:
            axis.step(x, y, where="post", color="black", linewidth=1.9, linestyle=(0, (2.0, 2.0)), alpha=0.95, zorder=4)
        else:
            axis.step(x, y, where="post", color=cmap(norm(row["selected_personal"])), linewidth=0.72, alpha=0.5, zorder=2)

    best_k, best_cuts, best_keeps = signature(best_row)
    best_label = f"Global best: K={best_k}, ({','.join(map(str, best_cuts))},28)"
    handles = [plt.Line2D([], [], color="#d62728", linewidth=2.5, label=best_label)]
    if default_row is not None:
        handles.append(plt.Line2D([], [], color="black", linewidth=1.9, linestyle=(0, (2.0, 2.0)), label="Default PSP@8: (7,14,28)"))
    axis.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.94, fontsize=9)

    axis.set_title("DualFashion FITB: Personal-Reward Pruning Schedules", pad=10)
    axis.set_xlabel("Inference progress")
    axis.set_ylabel("Particles alive")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(0.85, 17.5)
    axis.set_yscale("log", base=2)
    axis.set_yticks([1, 2, 4, 8, 16])
    axis.set_yticklabels(["1", "2", "4", "8", "16"])
    axis.set_xticks(np.linspace(0, 1, 5))
    axis.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    axis.text(0.01, 0.02, f"{len(rows):,} legal schedules; particle steps <= {budget}", transform=axis.transAxes, fontsize=8.5, color="#444444")

    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.012, shrink=0.88)
    colorbar.set_label("Final Personal reward")

    output_dir = (args.output_dir or args.summary_csv.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{args.base_name}.{suffix}", dpi=args.dpi, bbox_inches="tight")
    print(
        f"Saved {len(rows):,} schedules to {output_dir / args.base_name}.{{png,pdf}}\n"
        f"Best: K={best_k}, cutoffs={best_cuts}, keeps={best_keeps}, "
        f"Personal={best_row['selected_personal']:.6f}, particle_steps={best_row['particle_steps']}"
    )


if __name__ == "__main__":
    main()
