"""Export pool CSVs as compact, typed Excel tables (artifact-tool, or server fallback).

The server fallback is used only when the Codex artifact runtime is unavailable.
Long step CSVs remain authoritative; Excel trajectories use wide seed rows.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

TEXT = {"model", "method", "population", "image_path", "image_kind", "image_sha256",
        "source_signature", "gt_image_path", "status", "metric", "definition", "direction"}


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        yield header
        for row in reader:
            values = []
            for name, v in zip(header, row):
                if not v:
                    values.append(None)
                elif name in TEXT:
                    values.append(v)
                else:
                    try:
                        values.append(float(v) if any(c in v.lower() for c in ".e") else int(v))
                    except ValueError:
                        values.append(v)
            yield values


def artifact_runtime():
    bundle = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    bundled_node = bundle / ("bin/node.exe" if os.name == "nt" else "bin/node")
    node = os.environ.get("FASHION_EXCEL_NODE") or (str(bundled_node) if bundled_node.is_file() else shutil.which("node"))
    if not node:
        return None
    env = os.environ.copy()
    if (bundle / "node_modules/@oai/artifact-tool").is_dir():
        env["NODE_PATH"] = str(bundle / "node_modules") + os.pathsep + env.get("NODE_PATH", "")
    check = subprocess.run([node, "-e", "require.resolve('@oai/artifact-tool')"], env=env, capture_output=True)
    return (node, env) if check.returncode == 0 else None


def export_excel(out, rows_per_part=5000):
    out = Path(out)
    excel = out / "excel"
    excel.mkdir(exist_ok=True)
    old_manifest = json.loads((excel / "manifest.json").read_text(encoding="utf-8")) if (excel / "manifest.json").exists() else []
    runtime = artifact_runtime()
    if runtime is None:
        try:
            import openpyxl  # Portable server fallback; artifact-tool is not installed there.
        except ImportError as exc:
            raise RuntimeError("Excel runtime unavailable. Install openpyxl with this Python, "
                               "or use --no-excel and export later with --export-only.") from exc
    manifest = []

    def save(name, tables, preview=False):
        target = excel / name
        if runtime:
            spec = excel / "_export_spec.json"
            spec.write_text(json.dumps(dict(output=str(target.resolve()), tables=tables, preview=preview),
                                       ensure_ascii=False, allow_nan=False), encoding="utf-8")
            subprocess.run([runtime[0], "--max-old-space-size=8192", str(Path(__file__).with_suffix(".mjs")), str(spec)],
                           env=runtime[1], check=True)
            spec.unlink()
        else:
            from openpyxl import Workbook
            from openpyxl.cell import WriteOnlyCell
            from openpyxl.styles import Font, PatternFill
            from openpyxl.utils import get_column_letter
            book = Workbook(write_only=True)
            for table in tables:
                sheet = book.create_sheet(table["name"])
                sheet.sheet_view.showGridLines = False
                sheet.freeze_panes = "I2" if table["name"] == "PSP_trajectories" else "A2"
                header = table["rows"][0]
                for i, field in enumerate(header, 1):
                    sheet.column_dimensions[get_column_letter(i)].width = max(18, min(34, len(field)+2)) if field not in TEXT else 36
                for ri, row in enumerate(table["rows"]):
                    cells = []
                    for value in row:
                        cell = WriteOnlyCell(sheet, value=value)
                        if isinstance(value, str):
                            cell.data_type = "s"  # File paths are literal strings, never formulas.
                        if ri == 0:
                            cell.fill = PatternFill("solid", fgColor="243746")
                            cell.font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
                        else:
                            cell.font = Font(name="Arial", size=10)
                            if isinstance(value, float):
                                cell.number_format = "0.000000"
                        cells.append(cell)
                    sheet.append(cells)
                sheet.auto_filter.ref = f"A1:{get_column_letter(len(header))}{len(table['rows'])}"
            temporary = target.with_suffix(".xlsx.tmp")
            book.save(temporary)
            os.replace(temporary, target)
        manifest.append(dict(file=name, backend="artifact-tool" if runtime else "openpyxl",
                             sheets=[dict(name=t["name"], records=len(t["rows"])-1) for t in tables]))

    tables = [dict(name=title, rows=list(read_table(out / filename))) for title, filename in (
        ("Summary", "summary.csv"), ("Coverage", "coverage_summary.csv"),
        ("PSP_steps", "step_summary.csv"), ("Seed_slots", "slot_summary.csv"))]
    tables.append(dict(name="Definitions", rows=[
        ["metric", "definition", "direction"],
        ["CS", "100*cosine(image, target-category prompt), ViT-H/14", "higher"],
        ["CIS", "100*cosine(image, GT target), ViT-H/14; terminal only", "higher"],
        ["LPIPS", "LPIPS-VGG, native image resolution; terminal only", "lower"],
        ["Compatibility", "sigmoid(native FashionEvaluator), raw CLIP image features", "higher"],
        ["Personal", "100*cosine(image, user history for target category)", "higher"],
        ["all_terminal_candidates", "Mean of all evaluated terminal candidates; not a selected policy", ""],
        ["full_pool_personal_oracle_posthoc", "Highest final Personal within complete pool; ties: lower seed. Not PSP/DSP", ""],
        ["Coverage", "Only committed complete trajectories are scored; missing images are never zero scores", ""],
        ["Groups", "group 0/1/2 have separate seed pools; summaries do not merge their candidates", ""],
        ["Data role", "Test-set offline analysis. No fitting or threshold selection", ""],
        ["Detailed sources", "terminal_candidates.csv; psp_steps_group*.csv; metrics.sqlite; protocol.json", ""],
    ]))
    save("summary.xlsx", tables, preview=True)
    for pattern, sheet_name, prefix, limit in (
        ("terminal_group*.csv", "Terminal", "terminal", max(rows_per_part, 50000)),
        ("psp_trajectories_group*.csv", "PSP_trajectories", "psp_trajectories", rows_per_part),
    ):
        for source in sorted(out.glob(pattern)):
            records = read_table(source)
            header = next(records)
            part = 0
            while True:
                chunk = list(itertools.islice(records, limit))
                if not chunk:
                    break
                part += 1
                save(f"{source.stem}_{part:03d}.xlsx", [dict(name=sheet_name, rows=[header, *chunk])])
    (excel / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    current = {r["file"] for r in manifest}
    for entry in old_manifest:
        name = entry["file"]
        if name not in current and Path(name).name == name and name.endswith(".xlsx"):
            (excel / name).unlink(missing_ok=True)
    print(f"Excel: {len(manifest)} files under {excel}", flush=True)


if __name__ == "__main__":
    export_excel(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 5000)
