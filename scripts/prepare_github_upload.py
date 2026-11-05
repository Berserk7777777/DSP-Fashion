"""Create a reviewed source snapshot without datasets, weights, or Git operations."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT_FILES = (".gitignore", ".gitattributes", "README.md", "pyproject.toml", "SERVER_DEPLOYMENT.md")
PROJECTS = ("DiFashion-main", "DualFashion-main", "fashion_psp", "FashionDPO-main")
TREES = (*PROJECTS, "environments", "scripts")
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".cache", "checkpoint",
             "checkpoints", "pretrained_models", "processed_info", "output", "outputs",
             "wandb", "tensorboard", "finetuned_inception", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"}
SKIP_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx",
                 ".pkl", ".pickle", ".h5", ".hdf5", ".parquet", ".arrow", ".db", ".sqlite",
                 ".zip", ".7z", ".tar", ".gz", ".tgz", ".mp4", ".pyc", ".pyo", ".log",
                 ".tmp", ".bak", ".pem", ".key"}
TEXT_SUFFIXES = {".py", ".ps1", ".sh", ".md", ".txt", ".json", ".jsonl", ".toml", ".yml",
                 ".yaml", ".ini", ".cfg", ".mjs", ".js", ".html", ".css", ".rst", ".bib"}
SECRET = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{28,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")


def skip_directory(relative):
    parts = relative.parts
    name = parts[-1]
    if name in SKIP_DIRS or name.endswith(".egg-info"):
        return True
    if len(parts) == 2 and parts[0] == "DiFashion-main":
        return name in {"datasets", "eval-test", "diagnostics"} or name.startswith(("offline", "online", "fitb", "gor"))
    if parts == ("DualFashion-main", "dataset"):
        return True
    if len(parts) > 2 and parts[:2] == ("fashion_psp", "artifacts"):
        return True
    return False


def source_files(root):
    for name in ROOT_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        yield path
    for name in TREES:
        if not (root / name).is_dir():
            raise FileNotFoundError(root / name)
        for folder, directories, filenames in os.walk(root / name, followlinks=False):
            parent = Path(folder)
            directories[:] = sorted(d for d in directories if not skip_directory((parent / d).relative_to(root)))
            for filename in sorted(filenames):
                path = parent / filename
                rel = path.relative_to(root)
                if path.is_symlink():
                    raise ValueError(f"Review symbolic link before upload: {rel}")
                if path.suffix.lower() in SKIP_SUFFIXES or ".sqlite" in filename:
                    continue
                if filename.startswith((".env", "id_rsa", "id_ed25519")) or filename in {"Thumbs.db", ".DS_Store"}:
                    continue
                if rel.parts[:2] == ("fashion_psp", "artifacts") and path.suffix != ".py":
                    continue
                yield path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-root", type=Path)
    args = p.parse_args()
    root = args.root.resolve()
    output = (args.output_root or root / "tmp/github-upload").resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "DSP-Fashion"
    # Never merge a newly generated manifest with stale files from an old snapshot.
    if snapshot.exists() and any(snapshot.iterdir()):
        raise FileExistsError(f"Snapshot exists; choose a new --output-root: {snapshot}")
    rows, syntax_errors, secret_hits = [], [], []
    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        if size >= 50 * 1024 * 1024:
            raise ValueError(f"Review large non-weight file before upload: {rel} ({size} bytes)")
        raw = path.read_bytes()
        if path.suffix.lower() in TEXT_SUFFIXES:
            content = raw.decode("utf-8-sig", errors="replace")
            for match in SECRET.finditer(content):
                secret_hits.append(dict(file=rel, line=content.count("\n", 0, match.start())+1))
            if path.suffix == ".py":
                try:
                    ast.parse(content, filename=rel)
                except SyntaxError as exc:
                    syntax_errors.append(dict(file=rel, line=exc.lineno, message=exc.msg))
        rows.append(dict(path=rel, bytes=size, sha256=hashlib.sha256(raw).hexdigest()))
    report = dict(projects=list(PROJECTS), files=len(rows), bytes=sum(r["bytes"] for r in rows),
                  by_tree=dict(Counter(r["path"].split("/")[0] for r in rows)),
                  syntax_errors=syntax_errors, credential_pattern_hits=secret_hits)
    (output / "review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if secret_hits or syntax_errors:
        raise ValueError(f"Source review needs attention; see {output / 'review.json'}")
    required = ["DiFashion-main/DiFashion/models/difashion.py",
                "DiFashion-main/tools/collect_ifashion_fitb_candidates.py",
                "DiFashion-main/Evaluation/compatibility_evaluator/compatibility_net.py",
                "DualFashion-main/eval/compatibility_evaluator/compatibility_net.py",
                "DualFashion-main/run_ifashion_offline_cache.py",
                "fashion_psp/artifacts/__init__.py", "fashion_psp/artifacts/io.py",
                "fashion_psp/evaluation/evaluate_fitb_offline_pools.py",
                "fashion_psp/evaluation/offline_pool_excel.mjs"]
    selected = {r["path"] for r in rows}
    if any(path not in selected for path in required):
        raise ValueError("Required source missing: " + str([x for x in required if x not in selected]))
    manifest = dict(schema=1, repository="Berserk7777777/DSP-Fashion", private=True,
                    files=sorted(rows, key=lambda r: r["path"]))
    manifest_text = json.dumps(manifest, indent=2)
    (root / "UPLOAD_MANIFEST.json").write_text(manifest_text, encoding="utf-8")
    archive = output / "DSP-Fashion-source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for row in manifest["files"]:
            rel = row["path"]
            target = snapshot / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, target)
            z.write(target, "DSP-Fashion/" + rel)
        (snapshot / "UPLOAD_MANIFEST.json").write_text(manifest_text, encoding="utf-8")
        z.writestr("DSP-Fashion/UPLOAD_MANIFEST.json", manifest_text)
    print(json.dumps({**report, "archive": str(archive), "snapshot": str(snapshot)}, indent=2))


if __name__ == "__main__":
    main()
