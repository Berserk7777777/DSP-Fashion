#!/usr/bin/env python
"""Collect replayable Best-of-N and PSP candidates using DiFashion itself."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from diffusers import PNDMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from torchvision import transforms
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DiFashion"))
import data_utils  # noqa: E402
from models.difashion import DiFashion, MutualEncoder  # noqa: E402


BON_SEED_BASE = 42
PSP_SEED_BASE = 2026
SEED_STRIDE = 10000
BON_CANDIDATES = 4
PSP_CANDIDATES = 8
DEFAULT_INFERENCE_STEPS = 50
EXPECTED_DENOISING_LOOPS = 51
PSP_STEPS = (13, 26, 51)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True, help="Dataset root containing ifashion/")
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--checkpoint-root", type=Path, required=True, help="Contains checkpoint-15000/")
    p.add_argument("--base-model", type=Path, required=True, help="Local stable-diffusion-2-base directory")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--resume-from", type=int, default=0, help="FITB row index; completed tasks before/after it are still skipped")
    p.add_argument("--max-tasks", type=int, default=None, help="Optional smoke-test limit")
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=DEFAULT_INFERENCE_STEPS,
        help="DiFashion's original inference argument. This collector only accepts the default value 50.",
    )
    p.add_argument("--mixed-precision", choices=("fp16", "no"), default="fp16")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--preflight-only", action="store_true", help="Validate paths, tasks, seed map and output without loading models")
    p.add_argument("--audit-only", action="store_true", help="Re-verify existing task outputs and refresh missing_report.json without loading models")
    return p.parse_args()


def dump_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_manifest_index(path: Path):
    candidates, tasks = set(), set()
    if not path.is_file():
        return candidates, tasks
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("record_type") == "candidate_complete":
                candidates.add((row.get("task_id"), row.get("method"), row.get("candidate_index")))
            elif row.get("record_type") == "task_complete":
                tasks.add(row.get("task_id"))
    return candidates, tasks


def recover_manifest_records(metadata_path: Path, task, manifest: Path, candidate_index, completed_tasks):
    task_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    for method, bundle in task_meta["methods"].items():
        for candidate in bundle["candidates"]:
            key = (task["task_id"], method, candidate["candidate_index"])
            if key in candidate_index:
                continue
            append_jsonl(manifest, {
                "record_type": "candidate_complete", "task_index": task["task_index"], "task_id": task["task_id"],
                "uid": task["uid"], "oid": task["oid"], "method": method, "group_id": 0,
                "candidate_index": candidate["candidate_index"], "candidate_count": bundle["candidate_count"],
                "seed": candidate["seed"], "model": "DiFashion", "checkpoint": task_meta["checkpoint"],
                "steps": task_meta["actual_denoising_loop_calls"], "scheduler": task_meta["scheduler_class"],
                "timestep_records": candidate.get("steps", {}),
                "config": task_meta["guidance"], "files": candidate["files"],
                "task_output": str(metadata_path),
                "status": "complete", "error": None,
            })
            candidate_index.add(key)
    if task["task_id"] not in completed_tasks:
        append_jsonl(manifest, {"record_type": "task_complete", "task_index": task["task_index"], "task_id": task["task_id"], "status": "complete", "output": str(metadata_path)})
        completed_tasks.add(task["task_id"])


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_rows(data):
    rows = []
    for index, (uid, oid, outfit, category, input_ids) in enumerate(
        zip(data["uids"], data["oids"], data["outfits"], data["category"], data["input_ids"])
    ):
        outfit = np.asarray(outfit, dtype=np.int64).reshape(-1)
        if len(outfit) != 4 or np.count_nonzero(outfit == 0) != 1:
            raise ValueError(f"FITB row {index} must have exactly one blank item id 0; got {outfit.tolist()}")
        rows.append({
            "task_index": index,
            "task_id": f"{index:05d}_{int(uid)}_{int(oid)}",
            "task_dir_name": f"{int(uid)}_{int(oid)}",
            "uid": int(uid),
            "oid": int(oid),
            "outfit": outfit,
            "category": np.asarray(category, dtype=np.int64),
            "input_ids": np.asarray(input_ids, dtype=np.int64),
        })
    return rows


def build_seed_map(count):
    rows = []
    used = {}
    for task in range(count):
        pools = {
            "bon": [BON_SEED_BASE + task * SEED_STRIDE + i for i in range(BON_CANDIDATES)],
            "psp": [PSP_SEED_BASE + task * SEED_STRIDE + i for i in range(PSP_CANDIDATES)],
        }
        for method, seeds in pools.items():
            for candidate_index, seed in enumerate(seeds):
                if seed in used:
                    raise ValueError(f"Seed collision: {seed} maps to {used[seed]} and {(task, method, candidate_index)}")
                used[seed] = (task, method, candidate_index)
                rows.append({"task_index": task, "method": method, "candidate_index": candidate_index, "seed": seed})
    return rows


def preflight(args):
    required = [
        args.data_root / "ifashion/processed/new_fitb_test.npy",
        args.data_root / "ifashion/processed/test_hist_latents.npy",
        args.data_root / "ifashion/new_id_cate_dict.npy",
        args.data_root / "ifashion/new_all_item_image_paths.npy",
        args.image_root / "empty_image.png",
        args.checkpoint_root / "checkpoint-15000/unet_ema/diffusion_pytorch_model.bin",
        args.checkpoint_root / "checkpoint-15000/fashion_encoder_ema/diffusion_pytorch_model.bin",
        args.base_model / "scheduler/scheduler_config.json",
        args.base_model / "unet/config.json",
        args.base_model / "vae/config.json",
        args.base_model / "text_encoder/config.json",
        args.base_model / "tokenizer/tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    if args.num_inference_steps != DEFAULT_INFERENCE_STEPS:
        raise ValueError(
            f"This protocol fixes --num-inference-steps to DiFashion's default {DEFAULT_INFERENCE_STEPS}; "
            f"got {args.num_inference_steps}."
        )
    scheduler = PNDMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps, device="cpu")
    args.actual_denoising_loop_calls = len(scheduler.timesteps)
    if args.actual_denoising_loop_calls != EXPECTED_DENOISING_LOOPS:
        raise RuntimeError(
            "The installed PNDMScheduler no longer has the audited DiFashion schedule: "
            f"argument {args.num_inference_steps} produced {args.actual_denoising_loop_calls} loop calls, "
            f"expected {EXPECTED_DENOISING_LOOPS}."
        )
    if args.resume_from < 0:
        raise ValueError("--resume-from must be >= 0")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("--max-tasks must be >= 1")
    data = np.load(args.data_root / "ifashion/processed/new_fitb_test.npy", allow_pickle=True).item()
    tasks = task_rows(data)
    duplicate_task_dirs = {
        name for name in {task["task_dir_name"] for task in tasks}
        if sum(task["task_dir_name"] == name for task in tasks) > 1
    }
    if duplicate_task_dirs:
        raise ValueError(
            "uid_oid task directory names are not unique: " + ", ".join(sorted(duplicate_task_dirs)[:10])
        )
    if args.resume_from >= len(tasks):
        raise ValueError(f"--resume-from {args.resume_from} is past the {len(tasks)} FITB tasks")
    seed_map = build_seed_map(len(tasks))
    out = args.output_root
    if out.exists() and any(out.iterdir()):
        config_path = out / "run_config.json"
        if not config_path.is_file():
            raise FileExistsError(f"Output directory is non-empty but has no run_config.json: {out}")
    return data, tasks, seed_map


def make_model_args(args):
    return SimpleNamespace(
        pretrained_model_name_or_path=str(args.base_model), revision=None, non_ema_revision=None,
        enable_xformers_memory_efficient_attention=False, category_emb_size=64, hid_dim=256,
        use_history=True, use_mutual_guidance=True, eta=0.1, noise_offset=0.0, snr_gamma=None,
    )


def decode_latent(model, latent):
    decoded = model.vae.decode(latent / model.vae.config.scaling_factor, return_dict=False)[0]
    return model.image_processor.postprocess(decoded, output_type="pil", do_denormalize=[True])[0]


def candidate_seeds(task_index, method):
    base, count = (BON_SEED_BASE, BON_CANDIDATES) if method == "bon" else (PSP_SEED_BASE, PSP_CANDIDATES)
    return [base + task_index * SEED_STRIDE + i for i in range(count)]


def candidate_metadata(task, method, candidate_index, seed, args, scheduler_config, relpaths, timestep_schedule):
    return {
        "task_index": task["task_index"], "task_id": task["task_id"], "uid": task["uid"], "oid": task["oid"],
        "method": method, "group_id": 0, "candidate_index": candidate_index, "candidate_count": BON_CANDIDATES if method == "bon" else PSP_CANDIDATES,
        "seed": seed, "seed_base": BON_SEED_BASE if method == "bon" else PSP_SEED_BASE,
        "seed_stride": SEED_STRIDE, "checkpoint": str(args.checkpoint_root / "checkpoint-15000"),
        "base_model": str(args.base_model), "scheduler_num_inference_steps_argument": args.num_inference_steps,
        "actual_denoising_loop_calls": args.actual_denoising_loop_calls,
        "scheduler_class": "PNDMScheduler", "scheduler_config": scheduler_config,
        "guidance": {"category": 12.0, "mutual": 5.0, "history": 4.0, "eta": 0.1},
        "prediction_semantics": "x_prediction = DiFashion.pred_ori_sample_given_epsilon(t, x_t_before_step, guided_noise_pred), clamped to [-1,1]",
        "step_semantics": "step_number=i+1 (1-based); loop_index=i (0-based); scheduler_timestep is the actual PNDM timestep. x_prediction is decoded for reward scoring; continuation_latent is x_(t-1) after scheduler.step and is intentionally not written to disk.",
        "latent_artifacts_saved": False,
        "scheduler_timestep_schedule": timestep_schedule,
        "files": relpaths,
    }


def verify_task_dir(bon_dir: Path, psp_dir: Path, metadata_path: Path, output_root: Path, expected):
    errors = []
    if not metadata_path.is_file():
        return ["missing task_metadata.json"]
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        return [f"unreadable task_metadata.json: {e}"]
    for method, count in (("bon", BON_CANDIDATES), ("psp", PSP_CANDIDATES)):
        candidates = meta.get("methods", {}).get(method, {}).get("candidates", [])
        if len(candidates) != count:
            errors.append(f"{method}: expected {count} candidates, found {len(candidates)}")
        expected_seeds = set(candidate_seeds(expected["task_index"], method))
        if {candidate.get("seed") for candidate in candidates} != expected_seeds:
            errors.append(f"{method}: candidate seed set does not match the recorded seed protocol")
        for candidate in candidates:
            files = candidate.get("files", {})
            if method == "psp" and set(files.get("steps", {})) != {str(step) for step in PSP_STEPS}:
                errors.append(f"psp seed {candidate.get('seed')}: expected step snapshots {PSP_STEPS}")
            required_files = [files.get("final_image")]
            required_files.extend(files.get("decoded_reward_images", []))
            for item in filter(None, required_files):
                target = output_root / item
                if not target.is_file():
                    errors.append(f"missing {item}")
                elif target.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    try:
                        with Image.open(target) as im:
                            im.verify()
                    except Exception as e:
                        errors.append(f"unreadable image {item}: {e}")
    if meta.get("status") != "complete":
        errors.append(f"task status is {meta.get('status')!r}")
    if meta.get("task_index") != expected["task_index"] or meta.get("task_id") != expected["task_id"]:
        errors.append("task identity mismatch")
    return errors


def audit(args, tasks, started_at, failed):
    missing = []
    verified = 0
    for task in tasks:
        bon_dir = args.output_root / "bon" / task["task_dir_name"]
        psp_dir = args.output_root / "psp" / task["task_dir_name"]
        metadata_path = args.output_root / "metadata" / f"{task['task_dir_name']}.json"
        errors = verify_task_dir(bon_dir, psp_dir, metadata_path, args.output_root, task)
        if errors:
            missing.append({"task_index": task["task_index"], "task_id": task["task_id"], "errors": errors})
        else:
            verified += 1
    report = {
        "complete": not missing and not failed,
        "total_fitb_tasks": len(tasks), "verified_complete_tasks": verified,
        "missing_or_invalid_tasks": missing, "runtime_errors": failed,
        "offline_collection_cost_unet_calls_per_task": {"psp_full_trajectories": PSP_CANDIDATES * args.actual_denoising_loop_calls, "bon4_full_trajectories": BON_CANDIDATES * args.actual_denoising_loop_calls},
        "simulated_online_budget_unet_calls_per_task": {"psp_8_to_4_to_2_to_1": 8 * 13 + 4 * (26 - 13) + 2 * (args.actual_denoising_loop_calls - 26), "bon4": BON_CANDIDATES * args.actual_denoising_loop_calls},
        "wall_seconds": time.time() - started_at,
    }
    dump_json(args.output_root / "missing_report.json", report)
    return report


def main():
    args = parse_args()
    data, tasks, seed_map = preflight(args)
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    args.output_root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "dataset": "iFashion", "split": "test", "task": "FITB", "task_count_from_processed_rows": len(data["uids"]),
        "task_ids": "manifest key = row_index + uid + oid; output directory = uid_oid after uniqueness validation",
        "bon_seed_base": BON_SEED_BASE, "psp_seed_base": PSP_SEED_BASE, "seed_stride": SEED_STRIDE,
        "bon_candidates": BON_CANDIDATES, "psp_candidates": PSP_CANDIDATES, "psp_prune_steps": list(PSP_STEPS),
        "scheduler_num_inference_steps_argument": args.num_inference_steps,
        "actual_denoising_loop_calls": args.actual_denoising_loop_calls,
        "original_difashion_entry_default_argument": DEFAULT_INFERENCE_STEPS,
        "latent_artifacts_saved": False,
        "output_layout": {
            "bon": "bon/<uid>_<oid>/seed_<seed>.png",
            "psp_reward_predictions": "psp/<uid>_<oid>/step_<13|26|51>/seed_<seed>.png",
            "psp_terminal_images": "psp/<uid>_<oid>/final/seed_<seed>.png",
            "task_metadata": "metadata/<uid>_<oid>.json",
        },
        "scheduler": "PNDMScheduler loaded from local SD2 base; DiFashion default argument 50 produces 51 actual loop calls in installed diffusers 0.33.0",
        "checkpoint": str(args.checkpoint_root / "checkpoint-15000"), "base_model": str(args.base_model),
        "data_root": str(args.data_root), "image_root": str(args.image_root),
        "gt_access": "test_grd.npy and retrieval-candidate files are not loaded by this collector",
        "mixed_precision": args.mixed_precision, "device": args.device,
        "offline_cost": {"psp_calls_per_task": PSP_CANDIDATES * args.actual_denoising_loop_calls, "bon4_calls_per_task": BON_CANDIDATES * args.actual_denoising_loop_calls},
        "online_simulation_budget": {"psp_calls_per_task": 8 * 13 + 4 * (26 - 13) + 2 * (args.actual_denoising_loop_calls - 26), "bon4_calls_per_task": BON_CANDIDATES * args.actual_denoising_loop_calls},
    }
    cfg_path = args.output_root / "run_config.json"
    if cfg_path.exists():
        existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        if existing != cfg:
            raise ValueError("Existing run_config.json does not match this run; use a fresh output directory.")
    else:
        dump_json(cfg_path, cfg)
    if args.audit_only:
        report = audit(args, task_rows(data), time.time(), [])
        summary = {
            "complete": report["complete"], "verified_complete_tasks": report["verified_complete_tasks"],
            "total_fitb_tasks": report["total_fitb_tasks"],
            "missing_or_invalid_count": len(report["missing_or_invalid_tasks"]),
            "report_path": str(args.output_root / "missing_report.json"),
        }
        print(json.dumps(summary, indent=2))
        return 0 if report["complete"] else 2
    if args.preflight_only:
        print(json.dumps({"preflight": "ok", "task_count": len(data["uids"]), "unique_oids": len(set(map(int, data["oids"]))), "seed_slots_checked": len(seed_map), "output_root": str(args.output_root), "scheduler_argument": args.num_inference_steps, "actual_loop_calls": args.actual_denoising_loop_calls, "psp_steps": list(PSP_STEPS)}, indent=2))
        return 0

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("ifashion_candidate_collection")
    npy_root = args.data_root / "ifashion"
    test_history = np.load(npy_root / "processed/test_hist_latents.npy", allow_pickle=True).item()
    id_cate = np.load(npy_root / "new_id_cate_dict.npy", allow_pickle=True).item()
    image_paths = np.load(npy_root / "new_all_item_image_paths.npy", allow_pickle=True)
    test_data = np.load(npy_root / "processed/new_fitb_test.npy", allow_pickle=True).item()
    transforms_img = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), transforms.ToTensor(),
    ])
    img_dataset = data_utils.ImagePathDataset(str(args.image_root), image_paths, transforms_img, do_normalize=True)

    model_args = make_model_args(args)
    model = DiFashion(model_args, logger, len(id_cate), device).to(device)
    model.vae.enable_slicing()
    ckpt = args.checkpoint_root / "checkpoint-15000"
    ema_unet = EMAModel.from_pretrained(str(ckpt / "unet_ema"), UNet2DConditionModel)
    ema_encoder = EMAModel.from_pretrained(str(ckpt / "fashion_encoder_ema"), MutualEncoder)
    ema_unet.to(device)
    ema_encoder.to(device)
    ema_unet.copy_to(model.unet.parameters())
    ema_encoder.copy_to(model.fashion_encoder.parameters())
    model.eval()
    model.unet.eval()
    model.fashion_encoder.eval()
    scheduler_config = dict(model.noise_scheduler.config)
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
            null_img = img_dataset[0].unsqueeze(0).to(device)
            null_latent = model.vae.encode(null_img).latent_dist.mode()[0] * model.vae.config.scaling_factor
        null_latent = null_latent.detach()
        latent_cache = {}
        def get_latents(iids):
            new = [int(iid) for iid in iids if int(iid) not in latent_cache]
            if new:
                pixels = torch.stack([img_dataset[iid] for iid in new]).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                    z = model.vae.encode(pixels).latent_dist.mode() * model.vae.config.scaling_factor
                for iid, latent in zip(new, z):
                    latent_cache[iid] = latent.detach().cpu()
            return torch.stack([latent_cache[int(iid)] for iid in iids]).to(device)

        manifest = args.output_root / "manifest.jsonl"
        manifest_candidates, manifest_tasks = load_manifest_index(manifest)
        failures = []
        start_time = time.time()
        metadata_root = args.output_root / "metadata"
        metadata_root.mkdir(parents=True, exist_ok=True)
        selected_tasks = [t for t in tasks if t["task_index"] >= args.resume_from]
        bar = tqdm(selected_tasks, total=len(tasks), initial=args.resume_from, desc="FITB tasks", unit="task")
        for task in bar:
            final_bon_dir = args.output_root / "bon" / task["task_dir_name"]
            final_psp_dir = args.output_root / "psp" / task["task_dir_name"]
            final_metadata = metadata_root / f"{task['task_dir_name']}.json"
            final_errors = verify_task_dir(
                final_bon_dir, final_psp_dir, final_metadata, args.output_root, task
            )
            if not final_errors:
                recover_manifest_records(final_metadata, task, manifest, manifest_candidates, manifest_tasks)
                bar.set_postfix_str("skip complete")
                continue
            if final_bon_dir.exists() or final_psp_dir.exists() or final_metadata.exists():
                raise FileExistsError(
                    f"Incomplete direct-output task exists for {task['task_dir_name']}; preserving it. "
                    f"See missing_report.json after --audit-only before retrying."
                )
            stage = args.output_root / ".staging" / f"{task['task_id']}.{uuid.uuid4().hex}"
            stage.mkdir(parents=True, exist_ok=False)
            try:
                outfit_ids = task["outfit"].tolist()
                outfit_latents = get_latents(outfit_ids)
                uid_t = torch.tensor([task["uid"]], dtype=torch.long, device=device)
                oid_t = torch.tensor([task["oid"]], dtype=torch.long, device=device)
                outfit_t = torch.tensor([outfit_ids], dtype=torch.long, device=device)
                cat_t = torch.as_tensor(task["category"], dtype=torch.long, device=device).unsqueeze(0)
                ids_t = torch.as_tensor(task["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
                methods = {}
                candidate_manifest_rows = []
                for method in ("bon", "psp"):
                    method_stage_dir = stage / method / task["task_dir_name"]
                    method_stage_dir.mkdir(parents=True, exist_ok=False)
                    candidates = []
                    for candidate_index, seed in enumerate(candidate_seeds(task["task_index"], method)):
                        capture = {}
                        reward_images = []
                        timestep_schedule = []
                        def on_step(loop_index, timestep, latent_before, noise_pred, x_prediction, latent_after):
                            step_number = loop_index + 1
                            timestep_schedule.append({"loop_index_zero_based": int(loop_index), "step_number_one_based": step_number, "scheduler_timestep": int(timestep.item())})
                            if method != "psp" or step_number not in PSP_STEPS:
                                return
                            step_dir = method_stage_dir / f"step_{step_number:02d}"
                            step_dir.mkdir(parents=True, exist_ok=True)
                            reward_image = decode_latent(model, x_prediction)
                            image_name = f"seed_{seed}.png"
                            reward_image.save(step_dir / image_name)
                            reward_relpath = str(Path("psp") / task["task_dir_name"] / step_dir.name / image_name)
                            reward_images.append(reward_relpath)
                            capture[str(step_number)] = {
                                "loop_index_zero_based": int(loop_index), "step_number_one_based": step_number,
                                "scheduler_timestep": int(timestep.item()),
                                "x_prediction_saved_as_decoded_image": True,
                                "continuation_latent_saved": False,
                                "decoded_reward_image": reward_relpath,
                            }
                        gen = torch.Generator(device=device).manual_seed(seed)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                            result, init_latent = model.fashion_generation(
                                uid_t, oid_t, ids_t, outfit_t, None, cat_t, test_history,
                                num_inference_steps=args.num_inference_steps,
                                category_guidance_scale=12.0, hist_guidance_scale=4.0,
                                mutual_guidance_scale=5.0, null_img=null_img[0],
                                outfit_latents=outfit_latents, null_latent=null_latent,
                                generator=gen, output_type="pil", return_dict=False,
                                step_callback=on_step,
                            )
                        image = result[task["uid"]][task["oid"]]["images"][0]
                        if method == "bon":
                            final_image_name = f"seed_{seed}.png"
                            final_path = method_stage_dir / final_image_name
                            final_relpath = str(Path("bon") / task["task_dir_name"] / final_image_name)
                        else:
                            final_image_name = f"seed_{seed}.png"
                            final_dir = method_stage_dir / "final"
                            final_dir.mkdir(parents=True, exist_ok=True)
                            final_path = final_dir / final_image_name
                            final_relpath = str(Path("psp") / task["task_dir_name"] / "final" / final_image_name)
                        image.save(final_path)
                        if method == "bon":
                            paths = {"final_image": final_relpath, "latent_artifacts_saved": False}
                        else:
                            if set(capture) != {str(s) for s in PSP_STEPS}:
                                raise RuntimeError(f"seed {seed} missing PSP snapshots: got {sorted(capture)}")
                            paths = {"final_image": final_relpath, "latent_artifacts_saved": False, "decoded_reward_images": reward_images, "steps": capture}
                        if len(timestep_schedule) != args.actual_denoising_loop_calls:
                            raise RuntimeError(
                                f"seed {seed}: expected {args.actual_denoising_loop_calls} denoising-loop calls, "
                                f"observed {len(timestep_schedule)}"
                            )
                        candidates.append({"candidate_index": candidate_index, "seed": seed, "files": paths, "steps": capture})
                        candidate_manifest_rows.append({
                            "record_type": "candidate_complete", "task_index": task["task_index"], "task_id": task["task_id"],
                            "uid": task["uid"], "oid": task["oid"], "method": method, "group_id": 0,
                            "candidate_index": candidate_index, "candidate_count": len(candidate_seeds(task["task_index"], method)),
                            "seed": seed, "model": "DiFashion", "checkpoint": str(ckpt), "steps": args.actual_denoising_loop_calls,
                            "scheduler": "PNDMScheduler", "timestep_records": capture,
                            "scheduler_timestep_schedule": timestep_schedule,
                            "config": {"category_guidance_scale": 12.0, "mutual_guidance_scale": 5.0, "hist_guidance_scale": 4.0, "eta": 0.1},
                            "files": paths, "status": "complete", "error": None,
                        })
                        del result, image, init_latent
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    methods[method] = {"candidate_count": len(candidates), "candidates": candidates}
                task_meta = {
                    "status": "complete", "task_index": task["task_index"], "task_id": task["task_id"],
                    "task_dir_name": task["task_dir_name"],
                    "uid": task["uid"], "oid": task["oid"], "condition_source": "processed/new_fitb_test.npy; exactly one blank id 0",
                    "ground_truth_loaded_or_used": False, "latent_artifacts_saved": False,
                    "checkpoint": str(ckpt), "scheduler_class": "PNDMScheduler", "scheduler_config": scheduler_config,
                    "scheduler_num_inference_steps_argument": args.num_inference_steps,
                    "actual_denoising_loop_calls": args.actual_denoising_loop_calls,
                    "psp_prune_steps": list(PSP_STEPS),
                    "guidance": {"category_guidance_scale": 12.0, "mutual_guidance_scale": 5.0, "hist_guidance_scale": 4.0, "eta": 0.1},
                    "prediction_semantics": "PSP step images decode x_prediction = pred_ori_sample_given_epsilon(t, x_t_before_step, guided_noise_pred). PSP final images decode the terminal latent after the 51st scheduler step.",
                    "methods": methods,
                    "offline_collection_cost_unet_calls": {"psp": PSP_CANDIDATES * args.actual_denoising_loop_calls, "bon": BON_CANDIDATES * args.actual_denoising_loop_calls},
                    "simulated_online_budget_unet_calls": {"psp": 8 * 13 + 4 * (26 - 13) + 2 * (args.actual_denoising_loop_calls - 26), "bon": BON_CANDIDATES * args.actual_denoising_loop_calls},
                }
                dump_json(stage / "task_metadata.json", task_meta)
                check = verify_task_dir(
                    stage / "bon" / task["task_dir_name"],
                    stage / "psp" / task["task_dir_name"],
                    stage / "task_metadata.json",
                    stage,
                    task,
                )
                if check:
                    raise RuntimeError("task verification failed: " + "; ".join(check[:8]))
                final_bon_dir.parent.mkdir(parents=True, exist_ok=True)
                final_psp_dir.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / "bon" / task["task_dir_name"], final_bon_dir)
                os.replace(stage / "psp" / task["task_dir_name"], final_psp_dir)
                os.replace(stage / "task_metadata.json", final_metadata)
                (stage / "bon").rmdir()
                (stage / "psp").rmdir()
                stage.rmdir()
                for row in candidate_manifest_rows:
                    row["task_output"] = str(final_metadata)
                    append_jsonl(manifest, row)
                    manifest_candidates.add((row["task_id"], row["method"], row["candidate_index"]))
                append_jsonl(manifest, {"record_type": "task_complete", "task_index": task["task_index"], "task_id": task["task_id"], "status": "complete", "output": str(final_metadata)})
                manifest_tasks.add(task["task_id"])
                elapsed = max(time.time() - start_time, 1e-6)
                processed = max(bar.n - args.resume_from, 1)
                rate = processed / elapsed
                remaining = max(len(tasks) - bar.n, 0)
                bar.set_postfix_str(f"{rate:.3f} task/s, ETA {remaining / rate / 3600:.1f}h" if rate else "ETA pending")
            except Exception as exc:
                failures.append({"task_index": task["task_index"], "task_id": task["task_id"], "error": repr(exc)})
                append_jsonl(manifest, {"record_type": "task_error", "task_index": task["task_index"], "task_id": task["task_id"], "status": "error", "error": repr(exc), "staging_dir": str(stage)})
                logger.exception("Task failed: %s", task["task_id"])
                bar.set_postfix_str(f"failed={len(failures)}")
    report = audit(args, [task_rows(data)[i] for i in range(len(data["uids"]))], start_time if "start_time" in locals() else time.time(), failures if "failures" in locals() else [])
    logger.info("Run summary: %s", json.dumps({
        "complete": report["complete"], "total_fitb_tasks": report["total_fitb_tasks"],
        "verified_complete_tasks": report["verified_complete_tasks"],
        "missing_or_invalid_count": len(report["missing_or_invalid_tasks"]),
        "runtime_errors": report["runtime_errors"], "wall_seconds": report["wall_seconds"],
        "report_path": str(args.output_root / "missing_report.json"),
    }, ensure_ascii=True))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
