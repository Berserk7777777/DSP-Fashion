#!/usr/bin/env python
"""Collect replayable BoN@4 and PSP@8 GOR outfits from DiFashion.

The installed PNDM scheduler produces one more loop call than its
``num_inference_steps`` argument.  This protocol deliberately uses argument
51 so that the actual terminal scheduler decode is step 52.  PSP stores
decoded x-predictions at steps 13 and 26, and that true terminal decode at
step 52.  Ground-truth items are never loaded.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import PNDMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from PIL import Image
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
INFERENCE_ARGUMENT = 51
ACTUAL_LOOP_CALLS = 52
PSP_X_PREDICTION_STEPS = (13, 26)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing ifashion/.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Directory containing checkpoint-15000/.")
    parser.add_argument("--base-model", type=Path, required=True, help="Local stable-diffusion-2-base.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=int, default=0, help="First GOR task index to consider.")
    parser.add_argument("--max-tasks", type=int, default=None, help="Number of tasks after --resume-from.")
    parser.add_argument("--num-inference-steps", type=int, default=INFERENCE_ARGUMENT,
                        help="Must be 51: PNDM then has 52 actual loop calls.")
    parser.add_argument("--mixed-precision", choices=("fp16", "no"), default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def candidate_seeds(task_index: int, method: str) -> list[int]:
    base, count = (BON_SEED_BASE, BON_CANDIDATES) if method == "bon" else (PSP_SEED_BASE, PSP_CANDIDATES)
    return [base + task_index * SEED_STRIDE + candidate_index for candidate_index in range(count)]


def build_tasks(data: dict) -> list[dict]:
    tasks = []
    for index, (uid, oid, category, input_ids) in enumerate(zip(
        data["uids"], data["oids"], data["category"], data["input_ids"]
    )):
        category = np.asarray(category, dtype=np.int64).reshape(-1)
        if len(category) != 4:
            raise ValueError(f"GOR row {index} has {len(category)} categories, expected 4")
        tasks.append({
            "task_index": index,
            "task_id": f"{index:05d}_{int(uid)}_{int(oid)}",
            "task_dir_name": f"{int(uid)}_{int(oid)}",
            "uid": int(uid), "oid": int(oid),
            "category": category,
            "input_ids": np.asarray(input_ids, dtype=np.int64),
        })
    names = [task["task_dir_name"] for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError("uid_oid directories are not unique in the GOR condition rows")
    return tasks


def preflight(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    if args.num_inference_steps != INFERENCE_ARGUMENT:
        raise ValueError(
            f"This GOR protocol requires --num-inference-steps {INFERENCE_ARGUMENT}; "
            f"that yields actual terminal step {ACTUAL_LOOP_CALLS}, not argument {args.num_inference_steps}."
        )
    if args.resume_from < 0 or args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("--resume-from must be non-negative and --max-tasks must be positive")
    data_path = args.data_root / "ifashion" / "processed" / "new_fitb_test.npy"
    required = [
        data_path,
        args.data_root / "ifashion" / "processed" / "test_hist_latents.npy",
        args.data_root / "ifashion" / "new_id_cate_dict.npy",
        args.data_root / "ifashion" / "new_all_item_image_paths.npy",
        args.image_root / "empty_image.png",
        args.checkpoint_root / "checkpoint-15000" / "unet_ema" / "diffusion_pytorch_model.bin",
        args.checkpoint_root / "checkpoint-15000" / "fashion_encoder_ema" / "diffusion_pytorch_model.bin",
        args.base_model / "scheduler" / "scheduler_config.json",
        args.base_model / "unet" / "config.json",
        args.base_model / "vae" / "config.json",
        args.base_model / "text_encoder" / "config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    scheduler = PNDMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps, device="cpu")
    if len(scheduler.timesteps) != ACTUAL_LOOP_CALLS:
        raise RuntimeError(
            f"PNDM argument {args.num_inference_steps} produced {len(scheduler.timesteps)} loop calls; "
            f"expected {ACTUAL_LOOP_CALLS}."
        )
    data = np.load(data_path, allow_pickle=True).item()
    tasks = build_tasks(data)
    if args.resume_from >= len(tasks):
        raise ValueError(f"--resume-from {args.resume_from} is past {len(tasks)} available GOR tasks")
    return data, tasks


def make_model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_model_name_or_path=str(args.base_model), revision=None, non_ema_revision=None,
        enable_xformers_memory_efficient_attention=False, category_emb_size=64, hid_dim=256,
        use_history=True, use_mutual_guidance=True, eta=0.1, noise_offset=0.0, snr_gamma=None,
    )


def decode_latents(model: DiFashion, latents: torch.Tensor) -> list[Image.Image]:
    decoded = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
    return model.image_processor.postprocess(decoded, output_type="pil", do_denormalize=[True] * len(decoded))


def save_outfit(images: list[Image.Image], directory: Path) -> list[str]:
    if len(images) != 4:
        raise ValueError(f"GOR must produce four category images, got {len(images)}")
    directory.mkdir(parents=True, exist_ok=False)
    paths = []
    for item_index, image in enumerate(images):
        path = directory / f"item_{item_index}.png"
        image.save(path)
        paths.append(path.name)
    width, height = images[0].size
    merged = Image.new("RGB", (2 * width, 2 * height), color=(255, 255, 255))
    for item_index, image in enumerate(images):
        merged.paste(image.convert("RGB"), ((item_index % 2) * width, (item_index // 2) * height))
    merged.save(directory / "all.png")
    paths.append("all.png")
    return paths


def check_task(output_root: Path, task: dict) -> list[str]:
    metadata_path = output_root / "metadata" / f"{task['task_dir_name']}.json"
    if not metadata_path.is_file():
        return ["missing metadata"]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"invalid metadata: {exc!r}"]
    errors = []
    if metadata.get("status") != "complete" or metadata.get("task_id") != task["task_id"]:
        errors.append("metadata task identity/status mismatch")
    for method, count in (("bon", BON_CANDIDATES), ("psp", PSP_CANDIDATES)):
        candidates = metadata.get("methods", {}).get(method, {}).get("candidates", [])
        if len(candidates) != count:
            errors.append(f"{method}: expected {count} candidates, got {len(candidates)}")
            continue
        for candidate in candidates:
            paths = candidate.get("files", {}).get("final_image_paths", [])
            if len(paths) != 5:
                errors.append(f"{method} seed {candidate.get('seed')}: missing final outfit files")
            for relative in paths:
                path = output_root / relative
                if not path.is_file():
                    errors.append(f"missing {relative}")
            if method == "psp":
                steps = candidate.get("files", {}).get("steps", {})
                if set(steps) != {"13", "26", "52"}:
                    errors.append(f"psp seed {candidate.get('seed')}: expected steps 13,26,52")
                for step in ("13", "26", "52"):
                    for relative in steps.get(step, {}).get("image_paths", []):
                        if not (output_root / relative).is_file():
                            errors.append(f"missing {relative}")
    return errors


def make_run_config(args: argparse.Namespace, total_tasks: int, selected_tasks: list[dict]) -> dict:
    return {
        "schema": 1, "dataset": "iFashion", "split": "test", "task": "GOR",
        "condition_source": "processed/new_fitb_test.npy; GOR masks all four observed outfit item ids",
        "task_count_from_processed_rows": total_tasks,
        "selected_task_indices": [task["task_index"] for task in selected_tasks],
        "bon_seed_base": BON_SEED_BASE, "psp_seed_base": PSP_SEED_BASE, "seed_stride": SEED_STRIDE,
        "bon_candidates": BON_CANDIDATES, "psp_candidates": PSP_CANDIDATES,
        "scheduler_num_inference_steps_argument": args.num_inference_steps,
        "actual_denoising_loop_calls": ACTUAL_LOOP_CALLS,
        "psp_reward_steps": [13, 26], "psp_terminal_step": 52,
        "checkpoint": str((args.checkpoint_root / "checkpoint-15000").resolve()),
        "base_model": str(args.base_model.resolve()),
        "gt_access": "No ground-truth item/image/retrieval data is loaded.",
        "output_layout": {
            "bon": "bon/<uid>_<oid>/seed_<seed>/item_<0..3>.png and all.png",
            "psp": "psp/<uid>_<oid>/step_<13|26|52>/seed_<seed>/item_<0..3>.png and all.png",
        },
        "cost": {
            "offline_collection_unet_calls_per_task": {"bon": 4 * 52, "psp": 8 * 52},
            "simulated_online_unet_calls_per_task": {"bon": 4 * 52, "psp": 8 * 13 + 4 * 13 + 2 * (52 - 26)},
        },
    }


def main() -> int:
    args = parse_args()
    data, all_tasks = preflight(args)
    selected_tasks = all_tasks[args.resume_from:]
    if args.max_tasks is not None:
        selected_tasks = selected_tasks[:args.max_tasks]
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = make_run_config(args, len(all_tasks), selected_tasks)
    config_path = args.output_root / "run_config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing run_config.json differs; use a new output directory.")
    elif any(args.output_root.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {args.output_root}")
    else:
        write_json(config_path, config)
    if args.preflight_only:
        print(json.dumps({"preflight": "ok", "selected_tasks": len(selected_tasks), "run_config": str(config_path)}, indent=2))
        return 0
    if args.audit_only:
        missing = [{"task_id": task["task_id"], "errors": check_task(args.output_root, task)} for task in selected_tasks]
        missing = [row for row in missing if row["errors"]]
        write_json(args.output_root / "missing_report.json", {"selected_tasks": len(selected_tasks), "missing": missing})
        print(json.dumps({"complete": not missing, "missing": len(missing)}, indent=2))
        return 0 if not missing else 2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger = logging.getLogger("ifashion_gor_candidate_collection")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_root = args.data_root / "ifashion"
    history = np.load(data_root / "processed" / "test_hist_latents.npy", allow_pickle=True).item()
    id_categories = np.load(data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
    image_paths = np.load(data_root / "new_all_item_image_paths.npy", allow_pickle=True)
    transform = transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor()])
    image_dataset = data_utils.ImagePathDataset(str(args.image_root), image_paths, transform, do_normalize=True)
    model = DiFashion(make_model_args(args), logger, len(id_categories), device).to(device)
    model.vae.enable_slicing()
    checkpoint = args.checkpoint_root / "checkpoint-15000"
    ema_unet = EMAModel.from_pretrained(str(checkpoint / "unet_ema"), UNet2DConditionModel)
    ema_encoder = EMAModel.from_pretrained(str(checkpoint / "fashion_encoder_ema"), MutualEncoder)
    ema_unet.to(device)
    ema_encoder.to(device)
    ema_unet.copy_to(model.unet.parameters())
    ema_encoder.copy_to(model.fashion_encoder.parameters())
    model.eval(); model.unet.eval(); model.fashion_encoder.eval()
    manifest = args.output_root / "manifest.jsonl"
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
            null_image = image_dataset[0].unsqueeze(0).to(device)
            null_latent = model.vae.encode(null_image).latent_dist.mode()[0] * model.vae.config.scaling_factor
        null_latent = null_latent.detach()
        gor_outfit_latents = null_latent.repeat(4, 1, 1, 1)
        start = time.time()
        completed, failures = 0, []
        for task in tqdm(selected_tasks, desc="GOR tasks", unit="task"):
            existing_errors = check_task(args.output_root, task)
            if not existing_errors:
                completed += 1
                continue
            final_bon = args.output_root / "bon" / task["task_dir_name"]
            final_psp = args.output_root / "psp" / task["task_dir_name"]
            final_metadata = args.output_root / "metadata" / f"{task['task_dir_name']}.json"
            if final_bon.exists() or final_psp.exists() or final_metadata.exists():
                raise FileExistsError(f"Incomplete direct output for {task['task_id']}; inspect before retrying")
            stage = args.output_root / ".staging" / f"{task['task_id']}.{uuid.uuid4().hex}"
            stage.mkdir(parents=True, exist_ok=False)
            try:
                uid = torch.tensor([task["uid"]], device=device)
                oid = torch.tensor([task["oid"]], device=device)
                categories = torch.as_tensor(task["category"], device=device).unsqueeze(0)
                input_ids = torch.as_tensor(task["input_ids"], device=device).unsqueeze(0)
                empty_outfit = torch.zeros((1, 4), dtype=torch.long, device=device)
                methods = {}
                for method in ("bon", "psp"):
                    candidates = []
                    for candidate_index, seed in enumerate(candidate_seeds(task["task_index"], method)):
                        snapshots = {}
                        def on_step(loop_index, timestep, latent_before, noise_pred, x_prediction, latent_after):
                            step = loop_index + 1
                            if method != "psp" or step not in PSP_X_PREDICTION_STEPS:
                                return
                            directory = stage / "psp" / task["task_dir_name"] / f"step_{step}" / f"seed_{seed}"
                            images = decode_latents(model, x_prediction)
                            names = save_outfit(images, directory)
                            relpaths = [str(path.relative_to(stage)) for path in [directory / name for name in names]]
                            snapshots[str(step)] = {"loop_index_zero_based": loop_index, "step_number_one_based": step,
                                                    "scheduler_timestep": int(timestep.item()), "kind": "decoded_x_prediction",
                                                    "image_paths": relpaths}
                        generator = torch.Generator(device=device).manual_seed(seed)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                            result, _ = model.fashion_generation(
                                uid, oid, input_ids, empty_outfit, None, categories, history,
                                num_inference_steps=args.num_inference_steps, category_guidance_scale=12.0,
                                hist_guidance_scale=4.0, mutual_guidance_scale=5.0, null_img=null_image[0],
                                outfit_latents=gor_outfit_latents, null_latent=null_latent, generator=generator,
                                output_type="pil", return_dict=False, step_callback=on_step,
                            )
                        images = result[task["uid"]][task["oid"]]["images"]
                        step_dir = (stage / method / task["task_dir_name"] / ("step_52" if method == "psp" else f"seed_{seed}"))
                        if method == "psp":
                            step_dir = step_dir / f"seed_{seed}"
                        names = save_outfit(images, step_dir)
                        final_paths = [str(path.relative_to(stage)) for path in [step_dir / name for name in names]]
                        if method == "psp":
                            snapshots["52"] = {"loop_index_zero_based": 51, "step_number_one_based": 52,
                                               "scheduler_timestep": int(model.noise_scheduler.timesteps[-1].item()),
                                               "kind": "terminal_scheduler_decode", "image_paths": final_paths}
                            if set(snapshots) != {"13", "26", "52"}:
                                raise RuntimeError(f"PSP seed {seed} captured {sorted(snapshots)}, expected 13, 26, 52")
                        candidates.append({"candidate_index": candidate_index, "seed": seed,
                                           "files": {"final_image_paths": final_paths, "steps": snapshots}})
                        del result
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    methods[method] = {"candidate_count": len(candidates), "candidates": candidates}
                metadata = {
                    "status": "complete", "task": "GOR", "task_index": task["task_index"], "task_id": task["task_id"],
                    "uid": task["uid"], "oid": task["oid"], "condition": {"all_outfit_items_masked": True, "category": task["category"].tolist()},
                    "ground_truth_loaded_or_used": False, "checkpoint": str(checkpoint), "scheduler_class": "PNDMScheduler",
                    "scheduler_num_inference_steps_argument": INFERENCE_ARGUMENT, "actual_denoising_loop_calls": ACTUAL_LOOP_CALLS,
                    "psp_reward_steps": [13, 26], "psp_terminal_step": 52,
                    "prediction_semantics": "steps 13/26 decode x_prediction; step 52 decodes the terminal scheduler latent",
                    "methods": methods,
                }
                write_json(stage / "task_metadata.json", metadata)
                final_bon.parent.mkdir(parents=True, exist_ok=True)
                final_psp.parent.mkdir(parents=True, exist_ok=True)
                final_metadata.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / "bon" / task["task_dir_name"], final_bon)
                os.replace(stage / "psp" / task["task_dir_name"], final_psp)
                os.replace(stage / "task_metadata.json", final_metadata)
                (stage / "bon").rmdir(); (stage / "psp").rmdir(); stage.rmdir()
                append_jsonl(manifest, {"record_type": "task_complete", "task_id": task["task_id"], "task_index": task["task_index"], "status": "complete"})
                completed += 1
            except Exception as exc:
                failures.append({"task_id": task["task_id"], "task_index": task["task_index"], "error": repr(exc)})
                append_jsonl(manifest, {"record_type": "task_error", "task_id": task["task_id"], "error": repr(exc), "stage": str(stage)})
                logger.exception("GOR task failed: %s", task["task_id"])
        report = {"selected_tasks": len(selected_tasks), "completed_tasks": completed, "failures": failures,
                  "wall_seconds": time.time() - start,
                  "complete": completed == len(selected_tasks) and not failures}
        write_json(args.output_root / "missing_report.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
