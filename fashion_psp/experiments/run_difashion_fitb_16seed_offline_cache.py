"""Strict-50-call DiFashion FITB pools: BoN terminal only, PSP all 50 images.

No pruning, reward models or GT. PNDM argument49 gives exactly50 UNet calls.
The default pool is16; --num-particles can select a smaller prefix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

from fashion_psp.experiments.offline_cache_workers import run_workers, task_indices

ROOT = Path(__file__).resolve().parents[2]
TOTAL_STEPS = 50
SCHEDULER_ARGUMENT = 49
SEED_STRIDE = 10000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("bon", "psp"), required=True)
    p.add_argument("--group-id", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--seed-base", type=int, default=None,
                   help="Method base before adding group_id*1000000; default42/2026.")
    p.add_argument("--num-particles", type=int, default=16)
    p.add_argument("--data-root", type=Path, default=Path("D:/DiFashion"))
    p.add_argument("--image-root", type=Path, default=None)
    p.add_argument("--checkpoint-root", type=Path, default=Path("D:/DiFashion"))
    p.add_argument("--base-model", type=Path,
                   default=ROOT / "DiFashion-main/pretrained_models/stable-diffusion-2-base")
    p.add_argument("--output-root", type=Path, required=True, help="One method/group output directory.")
    p.add_argument("--max-tasks", type=int, default=None, help="First N original task rows; omit for all.")
    p.add_argument("--resume-from", type=int, default=0, help="Skip completed candidates; start at this task row.")
    p.add_argument("--concurrent-tasks", type=int, default=1)
    p.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mixed-precision", choices=("fp16", "no"), default="fp16")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def candidate_seeds(task_index, seed_base, group_id, count):
    base = seed_base + group_id * 1000000 + task_index * SEED_STRIDE
    return [base + slot for slot in range(count)]


def capture_steps(method):
    return list(range(1, TOTAL_STEPS + 1)) if method == "psp" else [TOTAL_STEPS]


def candidate_folder(root, task, seed):
    return root / "tasks" / task["task_id"] / f"seed_{seed}"


def verify_candidate(folder, task, seed, config_hash, steps, thorough=False):
    path = folder / "candidate_metadata.json"
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (record.get("config_sha256") != config_hash or record.get("seed") != seed
            or record.get("task_index") != task["task_index"]
            or record.get("task_id") != task["task_id"]):
        raise ValueError(f"Existing candidate identity/config mismatch: {path}")
    images = record.get("images", [])
    if (record.get("model_forward_calls") != TOTAL_STEPS
            or [row.get("step") for row in images] != steps):
        return None
    for row in images:
        image = folder / row["image"]
        if not image.is_file() or image.stat().st_size != row["bytes"]:
            return None
        if thorough and hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            return None
    return record


def preflight(args):
    import numpy as np
    from diffusers import PNDMScheduler
    sys.path.insert(0, str(ROOT / "DiFashion-main/tools"))
    from collect_ifashion_fitb_candidates import task_rows

    if not 1 <= args.num_particles <= 16 or not 1 <= args.concurrent_tasks <= 32:
        raise ValueError("num-particles must be1..16; concurrent-tasks must be1..32")
    task_indices(0, args.worker_index, args.worker_count)
    if args.concurrent_tasks > 1 and args.worker_count != 1:
        raise ValueError("Worker cannot create nested workers")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("max-tasks must be positive")
    for key in ("data_root", "checkpoint_root", "base_model", "output_root"):
        setattr(args, key, getattr(args, key).resolve())
    args.image_root = (args.image_root or args.data_root / "ifashion/images").resolve()
    if args.seed_base is None:
        args.seed_base = 42 if args.method == "bon" else 2026
    checkpoint = args.checkpoint_root / "checkpoint-15000"
    required = [args.data_root / "ifashion/processed/new_fitb_test.npy",
                args.data_root / "ifashion/processed/test_hist_latents.npy",
                args.data_root / "ifashion/new_id_cate_dict.npy",
                args.data_root / "ifashion/new_all_item_image_paths.npy",
                args.image_root / "empty_image.png",
                checkpoint / "unet_ema/diffusion_pytorch_model.bin",
                checkpoint / "fashion_encoder_ema/diffusion_pytorch_model.bin"]
    required += [args.base_model / part for part in (
        "scheduler/scheduler_config.json", "unet/config.json", "vae/config.json",
        "text_encoder/config.json", "tokenizer/tokenizer_config.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    data = np.load(required[0], allow_pickle=True).item()
    all_tasks = task_rows(data)
    tasks = all_tasks[:args.max_tasks] if args.max_tasks is not None else all_tasks
    if not 0 <= args.resume_from < len(tasks):
        raise ValueError("resume-from must be within selected task range")
    if args.seed_base < 0 or candidate_seeds(len(all_tasks)-1, args.seed_base,
                                           args.group_id, args.num_particles)[-1] >= 2**63:
        raise ValueError("Unsupported seed range")
    scheduler = PNDMScheduler.from_pretrained(str(args.base_model), subfolder="scheduler")
    scheduler.set_timesteps(SCHEDULER_ARGUMENT, device="cpu")
    if len(scheduler.timesteps) != TOTAL_STEPS:
        raise RuntimeError(f"PNDM argument49 must yield50 actual calls, got{len(scheduler.timesteps)}")
    config = dict(
        version="difashion_fitb_full_pool_strict50_v1", method=args.method, group_id=args.group_id,
        seed_base=args.seed_base, group_seed=args.seed_base+args.group_id*1000000,
        seed_stride=SEED_STRIDE, num_particles=args.num_particles,
        seed_mapping="seed_base + group_id*1000000 + original_task_index*10000 + slot",
        terminal_step=TOTAL_STEPS, scheduler_num_inference_steps_argument=SCHEDULER_ARGUMENT,
        actual_unet_calls_per_candidate=TOTAL_STEPS, scheduler_class="PNDMScheduler",
        scheduler_timesteps=scheduler.timesteps.tolist(), image_steps=capture_steps(args.method),
        image_semantics="Intermediate: native x_prediction before update. Step50: decoded terminal scheduler latent.",
        precision=args.mixed_precision, guidance=dict(category=12., mutual=5., history=4., eta=.1),
        checkpoint=str(checkpoint), base_model=str(args.base_model), image_root=str(args.image_root),
        total_dataset_tasks=len(all_tasks), pruning=False, reward_evaluated=False, ground_truth_used=False,
        offline_particle_steps_per_task=args.num_particles*TOTAL_STEPS,
        offline_decode_views_per_task=args.num_particles*len(capture_steps(args.method)),
        assets=[dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
                for path in required],
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), ROOT / "DiFashion-main/DiFashion/models/difashion.py")})
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return tasks, config, digest


def install_config(args, config):
    path = args.output_root / "run_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing cache configuration differs; use a new output directory")
    elif args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("Nonempty cache directory has no run_config.json")
    else:
        dump_json(path, config)


def audit(args, tasks, digest):
    completed, missing = [], []
    steps = capture_steps(args.method)
    for task in tasks:
        for seed in candidate_seeds(task["task_index"], args.seed_base, args.group_id, args.num_particles):
            record = verify_candidate(candidate_folder(args.output_root, task, seed), task,
                                      seed, digest, steps, thorough=True)
            if record is None:
                missing.append(dict(task_index=task["task_index"], seed=seed))
            else:
                completed.append(record)
    report = dict(complete=not missing, tasks=len(tasks), expected_candidates=len(tasks)*args.num_particles,
                  complete_candidates=len(completed), missing=missing,
                  saved_images=sum(len(row["images"]) for row in completed),
                  model_forward_calls=sum(row["model_forward_calls"] for row in completed))
    dump_json(args.output_root / "audit.json", report)
    return report


def child_command(args, index, count):
    command = [sys.executable, "-m", "fashion_psp.experiments.run_difashion_fitb_16seed_offline_cache"]
    for key, value in vars(args).items():
        if key in {"worker_index", "worker_count", "concurrent_tasks"} or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command + ["--concurrent-tasks", "1", "--worker-index", str(index),
                      "--worker-count", str(count)]


def pending_tasks(args, tasks, digest):
    steps = capture_steps(args.method)
    selected = []
    for index in task_indices(len(tasks), args.worker_index, args.worker_count):
        task = tasks[index]
        if task["task_index"] < args.resume_from:
            continue
        if any(verify_candidate(candidate_folder(args.output_root, task, seed), task, seed, digest,
                                steps) is None for seed in candidate_seeds(
                                    task["task_index"], args.seed_base, args.group_id, args.num_particles)):
            selected.append(task)
    return selected


def generate(args, tasks, digest):
    import numpy as np
    import torch
    from diffusers import UNet2DConditionModel
    from diffusers.training_utils import EMAModel
    from torchvision import transforms
    from tqdm.auto import tqdm
    from collect_ifashion_fitb_candidates import (
        DiFashion, MutualEncoder, data_utils, decode_latent, make_model_args)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    logging.basicConfig(level=logging.INFO)
    category_ids = np.load(args.data_root / "ifashion/new_id_cate_dict.npy", allow_pickle=True).item()
    model = DiFashion(make_model_args(args), logging.getLogger("difashion_offline50"),
                      len(category_ids), device).to(device)
    model.vae.enable_slicing()
    checkpoint = args.checkpoint_root / "checkpoint-15000"
    for subfolder, cls, target in (("unet_ema", UNet2DConditionModel, model.unet),
                                    ("fashion_encoder_ema", MutualEncoder, model.fashion_encoder)):
        ema = EMAModel.from_pretrained(str(checkpoint / subfolder), cls)
        ema.copy_to(target.parameters())
        del ema
    model.eval()
    paths = np.load(args.data_root / "ifashion/new_all_item_image_paths.npy", allow_pickle=True)
    dataset = data_utils.ImagePathDataset(str(args.image_root), paths, transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), transforms.ToTensor(),
    ]), do_normalize=True)
    history = np.load(args.data_root / "ifashion/processed/test_hist_latents.npy", allow_pickle=True).item()
    steps = capture_steps(args.method)
    def autocast():
        return torch.autocast(device.type, dtype=torch.float16,
                              enabled=device.type == "cuda" and args.mixed_precision == "fp16")
    with torch.inference_mode():
        for task in tqdm(tasks, desc=f"{args.method} group{args.group_id} worker{args.worker_index}", unit="task"):
            pixels = torch.stack([dataset[int(i)] for i in task["outfit"]]).to(device)
            with autocast():
                outfit_latents = model.vae.encode(pixels).latent_dist.mode() * model.vae.config.scaling_factor
            blank = int(np.flatnonzero(task["outfit"] == 0)[0])
            null_latent = outfit_latents[blank]
            uid = torch.tensor([task["uid"]], device=device)
            oid = torch.tensor([task["oid"]], device=device)
            outfit = torch.as_tensor(task["outfit"], device=device).unsqueeze(0)
            categories = torch.as_tensor(task["category"], device=device).unsqueeze(0)
            ids = torch.as_tensor(task["input_ids"], device=device).unsqueeze(0)
            summaries = []
            for slot, seed in enumerate(candidate_seeds(task["task_index"], args.seed_base,
                                                        args.group_id, args.num_particles)):
                folder = candidate_folder(args.output_root, task, seed)
                record = verify_candidate(folder, task, seed, digest, steps)
                if record is not None:
                    summaries.append(record)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                images, observed = [], []
                calls = [0]
                def counter(*unused):
                    calls[0] += 1
                def save_snapshot(step, timestep, image, kind):
                    path = folder / f"step_{step:02d}.png"
                    image.save(path)
                    images.append(dict(step=step, scheduler_timestep=timestep, image=path.name,
                                       image_kind=kind, bytes=path.stat().st_size,
                                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                def on_step(index, timestep, before, noise, x_prediction, after):
                    step = index + 1
                    observed.append(int(timestep.item()))
                    if args.method == "psp" and step < TOTAL_STEPS:
                        save_snapshot(step, int(timestep.item()), decode_latent(model, x_prediction),
                                      "native_pre_update_x_prediction")
                hook = model.unet.register_forward_hook(counter)
                began = time.perf_counter()
                try:
                    with autocast():
                        result, initial = model.fashion_generation(
                            uid, oid, ids, outfit, None, categories, history,
                            num_inference_steps=SCHEDULER_ARGUMENT,
                            category_guidance_scale=12., hist_guidance_scale=4., mutual_guidance_scale=5.,
                            outfit_latents=outfit_latents, null_latent=null_latent,
                            generator=torch.Generator(device=device).manual_seed(seed),
                            output_type="pil", return_dict=False, step_callback=on_step)
                    if calls[0] != TOTAL_STEPS or len(observed) != TOTAL_STEPS:
                        raise RuntimeError(f"Expected50 actual calls, observed UNet={calls[0]}, callback={len(observed)}")
                    save_snapshot(TOTAL_STEPS, observed[-1], result[task["uid"]][task["oid"]]["images"][0],
                                  "terminal_scheduler_latent")
                    if [row["step"] for row in images] != steps:
                        raise RuntimeError("Saved image steps do not match declared cache protocol")
                    record = dict(config_sha256=digest, task_id=task["task_id"], task_index=task["task_index"],
                                  uid=task["uid"], oid=task["oid"], method=args.method, group_id=args.group_id,
                                  candidate_index=slot, seed=seed, model_forward_calls=calls[0],
                                  scheduler_num_inference_steps_argument=SCHEDULER_ARGUMENT,
                                  scheduler_timesteps=observed, images=images, reward_evaluated=False,
                                  elapsed_seconds=time.perf_counter()-began)
                    dump_json(folder / "candidate_metadata.json", record)
                    summaries.append(record)
                    del result, initial
                finally:
                    hook.remove()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            dump_json(args.output_root / "tasks" / task["task_id"] / "task_summary.json", dict(
                config_sha256=digest, task_index=task["task_index"], task_id=task["task_id"],
                uid=task["uid"], oid=task["oid"], method=args.method, group_id=args.group_id,
                num_particles=args.num_particles, model_forward_calls=sum(r["model_forward_calls"] for r in summaries),
                images_saved=sum(len(r["images"]) for r in summaries),
                candidate_seeds=[r["seed"] for r in summaries]))


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = parse_args()
    tasks, config, digest = preflight(args)
    if args.preflight_only:
        print(json.dumps(dict(preflight="ok", selected_tasks=len(tasks), config=config,
                              concurrent_tasks=args.concurrent_tasks, output=str(args.output_root)), indent=2))
        return 0
    install_config(args, config)
    if args.audit_only:
        report = audit(args, tasks, digest)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 2
    if args.concurrent_tasks > 1:
        count = min(args.concurrent_tasks, len(tasks))
        commands = []
        original_index, original_count = args.worker_index, args.worker_count
        for index in range(count):
            args.worker_index, args.worker_count = index, count
            if pending_tasks(args, tasks, digest):
                commands.append((index, child_command(args, index, count)))
        args.worker_index, args.worker_count = original_index, original_count
        run_workers(commands, args.output_root, ROOT)
        report = audit(args, tasks, digest)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 2
    pending = pending_tasks(args, tasks, digest)
    if pending:
        generate(args, pending, digest)
    if args.worker_count == 1:
        report = audit(args, tasks, digest)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
