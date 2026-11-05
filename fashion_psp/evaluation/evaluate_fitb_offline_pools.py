"""Score immutable DiFashion/DualFashion FITB pools, with resumable image-level metrics.

BoN: terminal CS/CIS/LPIPS/Compatibility/Personal. PSP: reward trajectories,
plus terminal CS/CIS/LPIPS. No diffusion model, pruning, or parameter fitting.
SQLite and long CSVs are authoritative; Excel uses one row per seed trajectory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import sqlite3
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("cs", "cis", "lpips", "compatibility", "personal")
FIELDS = ("model", "method", "group_id", "task_index", "uid", "oid", "candidate_index",
          "seed", "step", "terminal_step", "scheduler_timestep", "sigma", "image_path",
          "image_kind", "image_sha256", "source_signature", *METRICS, "gt_image_path")
KEYS = ("method", "group_id", "task_index", "seed", "step")
DEFAULT_CLIP = Path(os.environ.get("USERPROFILE", str(Path.home()))) / (
    ".cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/"
    "1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def signature(path, content=False):
    path = Path(path).resolve()
    st = path.stat()
    return dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns,
                **({"sha256": sha(path)} if content else {}))


def lookup(mapping, key):
    return mapping[key] if key in mapping else mapping[str(key)]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=("difashion", "dualfashion"))
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--data-root", type=Path, help="Dataset directory containing processed/, not its parent.")
    p.add_argument("--image-root", type=Path)
    p.add_argument("--output-root", type=Path)
    p.add_argument("--groups", default="0,1,2")
    p.add_argument("--methods", default="bon,psp")
    p.add_argument("--cache-group", action="append", default=[], metavar="METHOD:GROUP=PATH",
                   help="Explicit cache directory override; does not change its task/seed mapping.")
    p.add_argument("--expected-candidates", type=int, default=16)
    p.add_argument("--max-tasks", type=int, help="First N original dataset rows; default all rows.")
    p.add_argument("--candidate-limit", type=int, help="Smoke run: score only this many prefix candidates.")
    p.add_argument("--personal-clip-weights", type=Path, default=DEFAULT_CLIP)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--verify-images", action="store_true", help="Also check declared DiFashion image SHA256.")
    p.add_argument("--audit-only", action="store_true", help="Inventory only; load no metric models.")
    p.add_argument("--export-only", action="store_true", help="Export the saved database without GPU scoring.")
    p.add_argument("--require-complete", action="store_true", help="Exit nonzero for an incomplete evaluated pool.")
    p.add_argument("--no-excel", action="store_true")
    p.add_argument("--excel-rows", type=int, default=5000, help="Max seed trajectories per Excel part.")
    args = p.parse_args(argv)
    dual = args.model == "dualfashion"
    args.cache_root = (args.cache_root or ROOT / (
        "DualFashion-main/output/ifashion_fitb_16seed_offline" if dual else
        "DiFashion-main/offline_fitb_strict50")).resolve()
    args.data_root = (args.data_root or Path("C:/tmp/ifashion" if dual else "D:/DiFashion/ifashion")).resolve()
    args.image_root = (args.image_root or args.data_root / ("semantic_category" if dual else "images")).resolve()
    args.output_root = (args.output_root or args.cache_root / "evaluation").resolve()
    args.groups = [int(s) for s in args.groups.split(",")]
    args.methods = args.methods.split(",")
    if (not args.groups or len(set(args.groups)) != len(args.groups) or set(args.groups) - {0, 1, 2}
            or not args.methods or len(set(args.methods)) != len(args.methods)
            or set(args.methods) - {"bon", "psp"}):
        p.error("groups must be unique 0/1/2; methods must be unique bon/psp")
    for name in ("batch_size", "expected_candidates", "excel_rows", "max_tasks", "candidate_limit"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if args.expected_candidates > 16 or args.candidate_limit and args.candidate_limit > args.expected_candidates:
        p.error("expected-candidates must be <=16; candidate-limit must be <= expected-candidates")
    if args.audit_only and args.export_only:
        p.error("audit-only and export-only are exclusive")
    return args


class Dataset:
    def __init__(self, args):
        import numpy as np
        self.np, self.args = np, args
        self.condition_path = args.data_root / ("processed/new_fitb_test.npy" if args.model == "difashion"
                                                 else "processed/fitb_test.npy")
        fitb = np.load(self.condition_path, allow_pickle=True).item()
        self.tasks = []
        for i, (uid, oid, outfit, cats) in enumerate(zip(fitb["uids"], fitb["oids"], fitb["outfits"], fitb["category"])):
            outfit, cats = np.asarray(outfit).reshape(-1).tolist(), np.asarray(cats).reshape(-1).tolist()
            if len(outfit) != 4 or len(cats) != 4 or outfit.count(0) != 1:
                raise ValueError(f"Invalid four-item FITB condition at original row {i}")
            blank = outfit.index(0)
            self.tasks.append(dict(task_index=i, task_id=f"{i:05d}_{int(uid)}_{int(oid)}",
                                   uid=int(uid), oid=int(oid), outfit=outfit, categories=cats,
                                   blank_index=blank, category=int(cats[blank])))
        self.total = len(self.tasks)
        self.tasks = self.tasks[:args.max_tasks]
        self.ground = None

    def gt(self, task):
        # Called only for terminal diagnostics. This object never enters reward computation.
        np, args = self.np, self.args
        if self.ground is None:
            self.ground = np.load(args.data_root / ("test_grd.npy" if args.model == "difashion"
                                                    else "test_grd_dict.npy"), allow_pickle=True).item()
            self.paths = np.load(args.data_root / ("new_all_item_image_paths.npy" if args.model == "difashion"
                                                   else "all_item_image_paths.npy"), allow_pickle=True)
        if args.model == "difashion":
            entry = lookup(self.ground, task["oid"])
            outfit, cats = entry["outfits"], entry["category"]
        else:
            i = task["task_index"]
            if int(self.ground["uids"][i]) != task["uid"] or int(self.ground["oids"][i]) != task["oid"]:
                raise ValueError("GT row identity mismatch")
            outfit, cats = self.ground["outfits"][i], self.ground["category"][i]
        outfit, cats = np.asarray(outfit).reshape(-1), np.asarray(cats).reshape(-1)
        blank = task["blank_index"]
        if len(outfit) != 4 or list(cats) != task["categories"] or any(
                int(outfit[i]) != value for i, value in enumerate(task["outfit"]) if i != blank):
            raise ValueError(f"GT/context mismatch: {task['task_id']}")
        raw = self.paths[int(outfit[blank])]
        relative = str(raw).replace("\\", "/")
        path = args.image_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path.resolve())


def group_roots(args):
    explicit = {}
    for text in args.cache_group:
        label, path = text.split("=", 1)
        method, group = label.split(":")
        key = method, int(group)
        if key in explicit or key[0] not in args.methods or key[1] not in args.groups:
            raise ValueError(f"Duplicate/unrequested cache-group: {label}")
        explicit[key] = Path(path).resolve()
    for method in args.methods:
        for group in args.groups:
            key = method, group
            if key in explicit:
                yield method, group, explicit[key]
                continue
            matches = sorted((args.cache_root / method).glob(f"group_{group}_seed*"))
            matches = [p for p in matches if p.is_dir()]
            if len(matches) > 1:
                raise ValueError(f"Ambiguous cache directories: {matches}; use --cache-group")
            yield method, group, matches[0] if matches else None


def relocated_image(raw, folder):
    # DualFashion metadata contains Windows absolute paths. The verified task
    # directory suffix remains valid when the whole cache is moved to Linux.
    parts = PureWindowsPath(str(raw)).parts if "\\" in str(raw) else Path(str(raw)).parts
    if folder.name in parts:
        candidate = folder.joinpath(*parts[parts.index(folder.name) + 1:])
    else:
        candidate = folder / str(raw)
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(candidate)


def inventory(args, data, db):
    terminal = 50 if args.model == "difashion" else 28
    coverage, ready_images, sources = [], 0, []
    for method, group, root in group_roots(args):
        config_name = "run_config.json" if args.model == "difashion" else "offline_cache_protocol.json"
        config_path = root / config_name if root else None
        config = read_json(config_path) if config_path and config_path.is_file() else None
        config_hash = digest(config)
        expected_steps = list(range(1, terminal + 1)) if method == "psp" else [terminal]
        if config is not None:
            if (config["method"] != method or config["group_id"] != group
                    or config["num_particles"] != args.expected_candidates):
                raise ValueError(f"Cache method/group/pool size mismatch: {config_path}")
            if args.model == "difashion":
                if (config["terminal_step"] != 50 or config["actual_unet_calls_per_candidate"] != 50
                        or config["image_steps"] != expected_steps):
                    raise ValueError("Requires the new strict-50 DiFashion cache")
                base = int(config["group_seed"])
            else:
                if config["num_inference_steps"] != 28:
                    raise ValueError("Requires the native 28-step DualFashion cache")
                base = int(config["group_seed"])
            sources.append(dict(method=method, group_id=group, config_sha256=config_hash, root=str(root)))
        else:
            base = (42 if method == "bon" else 2026) + group * 1000000
        for task in data.tasks:
            seed0 = base + task["task_index"] * 10000
            task_folder = (root / "tasks" / task["task_id"] if args.model == "difashion" else
                           root / f"psp_offline_group{group}_task{task['task_index']}_seed{base}") if root else None
            metadata = None
            if args.model == "dualfashion" and task_folder and (task_folder / "metadata.json").is_file():
                metadata = read_json(task_folder / "metadata.json")
                if (any(int(metadata[k]) != task[k] for k in ("task_index", "uid", "oid"))
                        or metadata["group_id"] != group or metadata["num_inference_steps"] != terminal
                        or metadata["candidate_seeds"] != list(range(seed0, seed0 + args.expected_candidates))):
                    raise ValueError(f"DualFashion task/seed identity mismatch: {task_folder}")
                cache = metadata["offline_candidate_cache"]
                if (cache["steps"] != expected_steps or cache["candidate_count"] != args.expected_candidates
                        or len(cache["images"]) != len(expected_steps) * args.expected_candidates):
                    raise ValueError(f"Invalid cache coverage: {task_folder}")
            for slot in range(args.expected_candidates):
                seed, images, error = seed0 + slot, [], ""
                folder = task_folder / f"seed_{seed}" if task_folder and args.model == "difashion" else task_folder
                path = folder / "candidate_metadata.json" if folder else None
                try:
                    if config is None:
                        raise FileNotFoundError("Group configuration not yet saved")
                    if args.model == "difashion":
                        if not path.is_file():
                            raise FileNotFoundError("Candidate not yet committed")
                        meta = read_json(path)
                        if (meta["config_sha256"] != config_hash or meta["task_id"] != task["task_id"]
                                or meta["task_index"] != task["task_index"] or meta["uid"] != task["uid"]
                                or meta["oid"] != task["oid"] or meta["group_id"] != group
                                or meta["method"] != method or meta["seed"] != seed
                                or meta["candidate_index"] != slot or meta["model_forward_calls"] != terminal):
                            raise ValueError(f"DiFashion candidate identity mismatch: {path}")
                        rows = meta["images"]
                        for r in rows:
                            image = folder / r["image"]
                            if not image.is_file() or image.stat().st_size != r["bytes"]:
                                raise FileNotFoundError(f"Missing/truncated image: {image}")
                            if args.verify_images and sha(image) != r["sha256"]:
                                raise ValueError(f"Image SHA256 mismatch: {image}")
                            images.append(dict(r, image_path=str(image.resolve())))
                    else:
                        if metadata is None:
                            raise FileNotFoundError("Task not yet committed")
                        rows = [r for r in metadata["offline_candidate_cache"]["images"] if int(r["seed"]) == seed]
                        images = [dict(r, image_path=str(relocated_image(r["image_path"], folder))) for r in rows]
                    if [int(r["step"]) for r in images] != expected_steps:
                        raise ValueError(f"Incomplete/duplicate step coverage: seed={seed}, {folder}")
                    if args.candidate_limit and slot >= args.candidate_limit:
                        error = "outside_candidate_limit"
                except FileNotFoundError as exc:
                    error = str(exc)
                valid = not error
                coverage.append(dict(model=args.model, method=method, group_id=group, **{
                    k: task[k] for k in ("task_index", "uid", "oid")}, candidate_index=slot,
                    seed=seed, expected_steps=len(expected_steps), available_steps=len(images),
                    cache_complete=int(valid), evaluated_steps=0, status="ready" if valid else error))
                if valid:
                    record = dict(model=args.model, method=method, group_id=group,
                                  **{k: task[k] for k in ("task_index", "uid", "oid")},
                                  candidate_index=slot, seed=seed, terminal_step=terminal)
                    records = []
                    for image in images:
                        source = dict(config_sha256=config_hash, image=signature(image["image_path"]),
                                      declared_sha256=image.get("sha256"))
                        records.append({**record, **{k: image.get(k) for k in (
                            "step", "scheduler_timestep", "sigma", "image_kind", "image_path")},
                            "image_sha256": image.get("sha256"), "source_signature": digest(source)})
                    db.executemany("INSERT INTO current_inputs (" + ",".join(FIELDS) + ") VALUES (" + ",".join("?" for _ in FIELDS) + ")",
                                   [tuple(r.get(k) for k in FIELDS) for r in records])
                    ready_images += len(records)
            if task["task_index"] % 100 == 0:
                print(f"Inventory {method} group={group} task={task['task_index']} ready_images={ready_images}", flush=True)
    db.commit()
    return coverage, ready_images, sources


def open_database(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA temp_store=FILE")
    columns = []
    for k in FIELDS:
        t = "INTEGER" if k in ("group_id", "task_index", "uid", "oid", "candidate_index", "seed", "step", "terminal_step") else (
            "REAL" if k in METRICS + ("scheduler_timestep", "sigma") else "TEXT")
        columns.append(f'"{k}" {t}')
    db.execute(f"CREATE TABLE IF NOT EXISTS scores ({','.join(columns)}, PRIMARY KEY ({','.join(KEYS)}))")
    db.execute("CREATE TABLE IF NOT EXISTS info (name TEXT PRIMARY KEY, payload TEXT)")
    db.execute("CREATE TEMP TABLE current_inputs AS SELECT * FROM scores WHERE 0")
    db.execute("CREATE UNIQUE INDEX input_keys ON current_inputs(method,group_id,task_index,seed,step)")
    return db


def install_contract(args, data, db):
    model_root = ROOT / ("DiFashion-main" if args.model == "difashion" else "DualFashion-main")
    evaluator = model_root / ("Evaluation" if args.model == "difashion" else "eval") / (
        "compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth")
    assets = [data.condition_path, args.data_root / "processed/test_history_clipembs.npy",
              args.data_root / "cnn_features_clip.npy", evaluator, args.personal_clip_weights,
              args.data_root / ("new_id_cate_dict.npy" if args.model == "difashion" else "id_cate_dict.npy"),
              args.data_root / ("test_grd.npy" if args.model == "difashion" else "test_grd_dict.npy"),
              args.data_root / ("new_all_item_image_paths.npy" if args.model == "difashion" else "all_item_image_paths.npy")]
    contract = dict(schema=1, model=args.model, cache_root=str(args.cache_root),
                    expected_candidates=args.expected_candidates, image_root=str(args.image_root),
                    precision="FP32; no autocast; TF32 disabled", batch_size=args.batch_size,
                    metrics={"cs": "100*cosine(ViT-H/14 image, native category prompt)",
                             "cis": "100*cosine(ViT-H/14 image, GT image)", "lpips": "LPIPS-VGG, native image resolution, [-1,1]",
                             "compatibility": "sigmoid(native FashionEvaluator), raw image features, fixed observed outfit",
                             "personal": "100*cosine(ViT-H/14 image, target-category user history)"},
                    versions={p: importlib.metadata.version(p) for p in ("torch", "torchvision", "open_clip_torch", "lpips", "numpy", "Pillow")},
                    assets=[signature(p, content=p != args.personal_clip_weights and p.suffix != ".npy") for p in assets],
                    source_sha256=sha(__file__), evaluator_source_sha256=sha(evaluator.parent.parent / "compatibility_net.py"),
                    uses_generation_model=False, gt_use="terminal CIS/LPIPS diagnostics only; never reward scoring",
                    dataset_role="test-set offline analysis; not a tuning or independent validation set")
    existing = db.execute("SELECT payload FROM info WHERE name='contract'").fetchone()
    if existing and json.loads(existing[0]) != contract:
        raise ValueError("Evaluation protocol/assets changed; use a new output-root to avoid mixing metrics")
    db.execute("INSERT OR REPLACE INTO info VALUES ('contract',?)", (json.dumps(contract),))
    db.commit()
    write_json(args.output_root / "protocol.json", contract)
    return evaluator


class Scorer:
    def __init__(self, args, data, evaluator_path):
        import numpy as np
        import torch
        import open_clip
        from safetensors.torch import load_file
        self.args, self.data, self.np, self.torch = args, data, np, torch
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
        torch.manual_seed(2026)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
        model.load_state_dict(load_file(str(args.personal_clip_weights), device="cpu"), strict=True)
        self.clip = model.float().to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        import importlib.util
        spec = importlib.util.spec_from_file_location("pool_compatibility", evaluator_path.parent.parent / "compatibility_net.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.evaluator = module.FashionEvaluator(cnn_feat_dim=1024)
        self.evaluator.load_state_dict(torch.load(evaluator_path, map_location="cpu", weights_only=True))
        self.evaluator = self.evaluator.float().to(self.device).eval()
        self.item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
        self.history = np.load(args.data_root / "processed/test_history_clipembs.npy", allow_pickle=True).item()
        self.categories = np.load(args.data_root / ("new_id_cate_dict.npy" if args.model == "difashion" else "id_cate_dict.npy"), allow_pickle=True).item()
        self.lpips, self.gt_cache, self.text_cache = None, {}, {}

    def pixels(self, paths, lpips=False):
        from PIL import Image
        torch, np = self.torch, self.np
        result = []
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                result.append(torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)
                              if lpips else self.preprocess(rgb))
        return torch.stack(result).to(self.device)

    def score(self, batch):
        torch, np = self.torch, self.np
        import torch.nn.functional as F
        with torch.inference_mode():
            raw = self.clip.encode_image(self.pixels([r["image_path"] for r in batch])).float()
            normalized = F.normalize(raw, dim=-1)
            contexts, histories = [], []
            for i, r in enumerate(batch):
                task = self.data.tasks[r["task_index"]]
                context = np.zeros((4, 1024), dtype=np.float32)
                for pos, item in enumerate(task["outfit"]):
                    if item:
                        context[pos] = self.item_features[int(item)]
                contexts.append(torch.as_tensor(context, device=self.device))
                contexts[-1][task["blank_index"]] = raw[i]
                h = np.asarray(lookup(lookup(self.history, task["uid"]), task["category"]), dtype=np.float32).reshape(-1)
                if h.shape != (1024,):
                    raise ValueError(f"Invalid Personal history: {task['task_id']}")
                histories.append(torch.as_tensor(h, device=self.device))
            compatibility = torch.sigmoid(self.evaluator(torch.stack(contexts))).reshape(-1)
            personal = 100 * (normalized * F.normalize(torch.stack(histories), dim=-1)).sum(dim=-1)
            results = [{**r, "personal": float(personal[i]), "compatibility": float(compatibility[i]),
                        "cs": None, "cis": None, "lpips": None, "gt_image_path": None} for i, r in enumerate(batch)]
            final = [i for i, r in enumerate(batch) if r["step"] == r["terminal_step"]]
            if final:
                if self.lpips is None:
                    import lpips
                    self.lpips = lpips.LPIPS(net="vgg").float().to(self.device).eval()
                for i in final:
                    task = self.data.tasks[batch[i]["task_index"]]
                    gt_path = self.data.gt(task)
                    if gt_path not in self.gt_cache:
                        # Bounded cache: tasks are scored consecutively, shared by all seeds/methods.
                        if len(self.gt_cache) >= 8:
                            self.gt_cache.clear()
                        self.gt_cache[gt_path] = F.normalize(self.clip.encode_image(self.pixels([gt_path])).float(), dim=-1)
                    category = task["category"]
                    if category not in self.text_cache:
                        name = str(lookup(self.categories, category))
                        article = "a pair of" if any(s in name for s in ("shoes", "pants", "sneakers", "boots", "earrings", "slippers", "sandals")) else "a"
                        tokens = self.tokenizer([f"A photo of {article} {name}, on white background"]).to(self.device)
                        self.text_cache[category] = F.normalize(self.clip.encode_text(tokens).float(), dim=-1)
                    distance = self.lpips(self.pixels([batch[i]["image_path"]], lpips=True), self.pixels([gt_path], lpips=True))
                    results[i].update(cs=float(100 * (normalized[i] * self.text_cache[category][0]).sum()),
                                      cis=float(100 * (normalized[i] * self.gt_cache[gt_path][0]).sum()),
                                      lpips=float(distance.item()), gt_image_path=gt_path)
        for r in results:
            required = METRICS if r["step"] == r["terminal_step"] else ("personal", "compatibility")
            if not all(math.isfinite(r[k]) for k in required):
                raise ValueError(f"Nonfinite metrics: {r}")
        return results


def csv_query(db, path, sql, params=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    cursor = db.execute(sql, params)
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([d[0] for d in cursor.description])
        w.writerows(cursor)
    os.replace(tmp, path)


def export_results(args, db, coverage):
    # Only currently verified inputs contribute; never silently keep stale or missing images.
    out = args.output_root
    for row in coverage:
        row["evaluated_steps"] = db.execute("SELECT COUNT(*) FROM current_scores WHERE method=? AND group_id=? AND task_index=? AND seed=?",
                                           tuple(row[k] for k in ("method", "group_id", "task_index", "seed"))).fetchone()[0]
        if row["cache_complete"]:
            row["status"] = "evaluated" if row["evaluated_steps"] == row["expected_steps"] else "pending_evaluation"
    db.execute("DROP TABLE IF EXISTS temp.coverage")
    if coverage:
        db.execute("CREATE TEMP TABLE coverage (" + ",".join(f'"{k}"' for k in coverage[0]) + ")")
        db.executemany("INSERT INTO coverage VALUES (" + ",".join("?" for _ in coverage[0]) + ")",
                       [tuple(r.values()) for r in coverage])
    db.execute("DROP VIEW IF EXISTS temp.complete_pools")
    db.execute(f"CREATE TEMP VIEW complete_pools AS SELECT method,group_id,task_index FROM coverage "
               f"GROUP BY method,group_id,task_index HAVING SUM(status='evaluated')={args.expected_candidates}")
    db.execute("DROP VIEW IF EXISTS temp.terminal")
    db.execute("CREATE TEMP VIEW terminal AS SELECT s.*, CASE WHEN c.task_index IS NOT NULL THEN 1 ELSE 0 END AS pool_complete, "
               "CASE WHEN c.task_index IS NOT NULL THEN ROW_NUMBER() OVER (PARTITION BY s.method,s.group_id,s.task_index ORDER BY personal DESC,seed) END AS personal_rank, "
               "CASE WHEN c.task_index IS NOT NULL THEN MAX(personal) OVER (PARTITION BY s.method,s.group_id,s.task_index)-personal END AS personal_regret "
               "FROM current_scores s LEFT JOIN complete_pools c USING(method,group_id,task_index) WHERE step=terminal_step")
    csv_query(db, out / "coverage.csv", "SELECT * FROM coverage ORDER BY method,group_id,task_index,candidate_index")
    sql = "SELECT model,method,group_id,COUNT(DISTINCT task_index) AS expected_tasks,COUNT(*) AS expected_candidates,SUM(cache_complete) AS cache_complete_candidates,SUM(status='evaluated') AS evaluated_candidates,SUM(expected_steps) AS expected_images,SUM(evaluated_steps) AS evaluated_images FROM coverage GROUP BY model,method,group_id"
    csv_query(db, out / "coverage_summary.csv", sql)
    select_metrics = ",".join(f"AVG({k}) AS {k}_mean,COUNT({k}) AS {k}_n" for k in METRICS)
    metrics_t = ",".join(f"AVG(t.{k}) AS {k}_mean,COUNT(t.{k}) AS {k}_n" for k in METRICS)
    summary_queries = []
    for population, condition in (("all_terminal_candidates", ""),
                                  ("full_pool_personal_oracle_posthoc", " AND t.personal_rank=1")):
        summary_queries.append(
            f"SELECT c.model,c.method,c.group_id,'{population}' AS population,COUNT(t.seed) AS n_images,"
            f"COUNT(DISTINCT t.task_index) AS n_tasks,{metrics_t} "
            "FROM (SELECT DISTINCT model,method,group_id FROM coverage) c LEFT JOIN terminal t "
            f"ON c.method=t.method AND c.group_id=t.group_id{condition} GROUP BY c.model,c.method,c.group_id")
    csv_query(db, out / "summary.csv", " UNION ALL ".join(summary_queries))
    csv_query(db, out / "step_summary.csv", "SELECT model,method,group_id,step,COUNT(*) AS n_images,COUNT(DISTINCT task_index) AS n_tasks,AVG(personal) AS personal_mean,AVG(compatibility) AS compatibility_mean FROM current_scores WHERE method='psp' GROUP BY model,method,group_id,step")
    csv_query(db, out / "slot_summary.csv", f"SELECT model,method,group_id,candidate_index,COUNT(*) AS n_tasks,{select_metrics} FROM terminal GROUP BY model,method,group_id,candidate_index")
    csv_query(db, out / "task_oracle.csv", "SELECT * FROM terminal WHERE personal_rank=1 ORDER BY method,group_id,task_index")
    csv_query(db, out / "terminal_candidates.csv", "SELECT * FROM terminal ORDER BY method,group_id,task_index,candidate_index")
    for group in args.groups:
        csv_query(db, out / f"psp_steps_group{group}.csv", "SELECT * FROM current_scores WHERE method='psp' AND group_id=? ORDER BY task_index,candidate_index,step", (group,))
        csv_query(db, out / f"terminal_group{group}.csv", "SELECT * FROM terminal WHERE group_id=? ORDER BY method,task_index,candidate_index", (group,))
        columns = ",".join(f"MAX(CASE WHEN step={step} THEN {metric} END) AS {metric}_s{step:02d}" for metric in ("personal", "compatibility") for step in range(1, (50 if args.model == "difashion" else 28) + 1))
        csv_query(db, out / f"psp_trajectories_group{group}.csv", f"SELECT model,method,group_id,task_index,uid,oid,candidate_index,seed,COUNT(*) AS evaluated_steps,{columns} FROM current_scores WHERE method='psp' AND group_id=? GROUP BY model,method,group_id,task_index,uid,oid,candidate_index,seed ORDER BY task_index,candidate_index", (group,))
    write_json(out / "coverage_snapshot.json", coverage)
    complete = bool(coverage) and all(r["status"] == "evaluated" for r in coverage)
    report = dict(model=args.model, requested_groups=args.groups, requested_methods=args.methods,
                  complete=complete, expected_candidates=len(coverage),
                  evaluated_candidates=sum(r["status"] == "evaluated" for r in coverage),
                  expected_images=sum(r["expected_steps"] for r in coverage),
                  evaluated_images=sum(r["evaluated_steps"] for r in coverage),
                  status_counts=dict(Counter(r["status"] for r in coverage)),
                  summary_note="Candidate means and posthoc full-pool oracles are separate; PSP cache is not a selected PSP/DSP policy.")
    write_json(out / "evaluation_status.json", report)
    if not args.no_excel:
        from fashion_psp.evaluation.offline_pool_excel import export_excel
        export_excel(out, args.excel_rows)
    return report


def main(argv=None):
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    data = Dataset(args)
    print(f"{args.model}: {len(data.tasks)}/{data.total} original task rows; groups={args.groups}; methods={args.methods}", flush=True)
    from tqdm import tqdm
    db = open_database(args.output_root / "metrics.sqlite")
    coverage, image_count, sources = inventory(args, data, db)
    write_json(args.output_root / "inventory.json", dict(expected_candidates=len(coverage),
               ready_candidates=sum(r["cache_complete"] for r in coverage), ready_images=image_count, groups=sources))
    if args.audit_only:
        import io
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
        (args.output_root / "inventory.csv").write_text(buffer.getvalue(), encoding="utf-8-sig")
        print(json.dumps(read_json(args.output_root / "inventory.json"), indent=2))
        db.close()
        if args.require_complete and not all(r["cache_complete"] for r in coverage):
            raise SystemExit(2)
        return
    try:
        evaluator_path = install_contract(args, data, db)
        db.execute("CREATE TEMP VIEW current_scores AS SELECT s.* FROM scores s JOIN current_inputs c USING(method,group_id,task_index,seed,step,source_signature)")
        db.execute("CREATE TEMP TABLE pending AS SELECT i.* FROM current_inputs i LEFT JOIN scores s USING(method,group_id,task_index,seed,step,source_signature) WHERE s.seed IS NULL")
        pending_count = db.execute("SELECT COUNT(*) FROM pending").fetchone()[0]
        print(f"Verified images={image_count}, reused={image_count-pending_count}, pending={pending_count}", flush=True)
        if pending_count and not args.export_only:
            scorer = Scorer(args, data, evaluator_path)
            sql = "INSERT OR REPLACE INTO scores (" + ",".join(FIELDS) + ") VALUES (" + ",".join("?" for _ in FIELDS) + ")"
            cursor = db.execute("SELECT * FROM pending ORDER BY method,group_id,task_index,candidate_index,step")
            for start in tqdm(range(0, pending_count, args.batch_size), desc="Offline pool metrics", unit="batch"):
                results = scorer.score([dict(r) for r in cursor.fetchmany(args.batch_size)])
                db.executemany(sql, [tuple(r.get(k) for k in FIELDS) for r in results])
                db.commit()  # A stopped run loses at most one batch, not the entire group.
            del scorer
        report = export_results(args, db, coverage)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        if args.require_complete and not report["complete"]:
            raise SystemExit(2)
    finally:
        db.close()


if __name__ == "__main__":
    main()
