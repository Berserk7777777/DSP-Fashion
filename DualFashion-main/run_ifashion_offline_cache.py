"""Create resumable fixed-pool x0 image caches for offline reward studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms as T
from torchvision.utils import save_image

from Inference_ifashion_PFITB import (
    MATCHING_LAMBDA,
    build_preference_text,
    compute_stage2_context_offset,
    encode_text,
    ensure_stage2_modules,
    load_image_tensor,
    load_numpy_dict,
    prepend_pad_segment,
    restore_stage2_parameters,
)
from run_psp_ifashion_smoke import decode_latent, estimate_x0_from_flowmatch, parse_step_list
from sd3_modules.stage2_pipeline import DualDiffSD3Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--model-name", default="0057501")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fitb-file", default="processed/fitb_test.npy")
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--max-samples", type=int, default=2207)
    parser.add_argument("--concurrent-tasks", type=int, default=1,
                        help="Independent task workers; each loads its own model. Default: serial.")
    parser.add_argument("--task-worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--task-worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--num-particles", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--offline-cache-steps", default="7,14,28")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--text-guidance-scale", type=float, default=8.0)
    parser.add_argument("--context-guidance-scale", type=float, default=7.0)
    parser.add_argument("--preference-guidance-scale", type=float, default=8.0)
    parser.add_argument(
        "--negative-prompt",
        default="low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--deterministic-context", action="store_true")
    parser.add_argument("--resume-from", default="auto", help="auto, none, or a non-negative task index")
    return parser.parse_args()


def parse_resume_start(value: str) -> int | None:
    text = value.strip().lower()
    if text == "none":
        return None
    if text in {"", "auto"}:
        return 0
    try:
        index = int(text)
    except ValueError as exc:
        raise ValueError("--resume-from must be auto, none, or a non-negative task index") from exc
    if index < 0:
        raise ValueError("--resume-from task index must be non-negative")
    return index


def task_dir(output_root: Path, group: int, task_index: int, seed: int) -> Path:
    return output_root / f"psp_offline_group{group}_task{task_index}_seed{seed}"


def task_complete(path: Path, expected_steps: list[int], expected_particles: int) -> bool:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache = payload["offline_candidate_cache"]
        images = cache["images"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if not cache.get("enabled") or list(cache.get("steps", [])) != expected_steps:
        return False
    if cache.get("candidate_count") != expected_particles:
        return False
    if len(images) != len(expected_steps) * expected_particles:
        return False
    seeds = payload.get("candidate_seeds", [])
    if len(seeds) != expected_particles or len(set(seeds)) != expected_particles:
        return False
    if {(row.get("step"), row.get("seed")) for row in images} != {
        (step, seed) for step in expected_steps for seed in seeds
    }:
        return False
    return all(Path(row.get("image_path", "")).exists() for row in images)


def save_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def task_indices(total: int, worker_index: int, worker_count: int) -> range:
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("Invalid task worker index/count")
    return range(worker_index, total, worker_count)


def worker_command(args: argparse.Namespace, index: int, count: int) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve())]
    for key, value in vars(args).items():
        if key in {"concurrent_tasks", "task_worker_index", "task_worker_count"}:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command + ["--concurrent-tasks", "1", "--task-worker-index", str(index),
                      "--task-worker-count", str(count)]


def run_task_workers(args: argparse.Namespace, total: int, cache_steps: list[int],
                     resume_start: int | None) -> None:
    """Dispatch disjoint original task indices; no model is loaded by this parent."""
    count = min(args.concurrent_tasks, total)
    log_root = args.output_root / "worker_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    processes = []
    handles = []
    pending = set()
    try:
        for index in range(count):
            assigned = task_indices(total, index, count)
            incomplete = [task for task in assigned
                          if (resume_start is None or task >= resume_start) and
                          (resume_start is None or not task_complete(
                              task_dir(args.output_root, args.group, task, args.seed),
                              cache_steps, args.num_particles))]
            if not incomplete:
                print(f"[worker {index+1}/{count}] all assigned tasks already complete", flush=True)
                continue
            log_path = log_root / f"{run_id}_worker_{index:02d}.log"
            handle = log_path.open("w", encoding="utf-8")
            handles.append(handle)
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            environment["PYTHONIOENCODING"] = "utf-8"
            process = subprocess.Popen(
                worker_command(args, index, count), stdout=handle, stderr=subprocess.STDOUT,
                env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            processes.append((index, process, log_path))
            pending.add(index)
            print(f"[worker {index+1}/{count}] pending_tasks={len(incomplete)} "
                  f"pid={process.pid} log={log_path}", flush=True)
        while pending:
            for index, process, log_path in processes:
                if index not in pending:
                    continue
                code = process.poll()
                if code is None:
                    continue
                pending.remove(index)
                if code:
                    raise RuntimeError(f"Task worker {index} failed with exit code {code}; "
                                       f"see {log_path}. Resume with --resume-from auto.")
                print(f"[worker {index+1}/{count}] completed", flush=True)
            if pending:
                time.sleep(.5)
    finally:
        # On failure or Ctrl+C, stop all children before returning to the caller.
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()


def main() -> None:
    args = parse_args()
    if args.num_particles not in (8, 16):
        raise ValueError("This offline cache protocol supports 8 or 16 initial seeds.")
    if args.seed_stride < args.num_particles:
        raise ValueError("--seed-stride must be at least --num-particles.")
    if not 1 <= args.concurrent_tasks <= 32:
        raise ValueError("--concurrent-tasks must be between 1 and 32")
    task_indices(0, args.task_worker_index, args.task_worker_count)
    if args.concurrent_tasks > 1 and args.task_worker_count != 1:
        raise ValueError("Concurrent parent cannot itself be a task worker")
    if args.concurrent_tasks > 1 and not args.deterministic_context:
        raise ValueError("Concurrent tasks require --deterministic-context to preserve context mapping")

    cache_steps = parse_step_list(args.offline_cache_steps, args.num_inference_steps)
    if args.num_inference_steps != 28:
        raise ValueError("This DualFashion offline-cache protocol is fixed at --num-inference-steps 28.")
    if not cache_steps or cache_steps[-1] != args.num_inference_steps:
        raise ValueError("--offline-cache-steps must be non-empty and include the terminal step 28.")
    expected_cache_image_count = len(cache_steps) * args.num_particles
    resume_start = parse_resume_start(args.resume_from)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    fitb = load_numpy_dict(str(args.data_root / args.fitb_file))
    ground_truth = load_numpy_dict(str(args.data_root / "test_grd_dict.npy"))
    id_to_category = load_numpy_dict(str(args.data_root / "id_cate_dict.npy"))
    preferences = load_numpy_dict(str(args.processed_info_root / "stage2_ifashion_test" / "test_preference.npy"))
    image_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    total_samples = min(len(fitb["uids"]), args.max_samples)
    if total_samples < 1:
        raise ValueError("--max-samples must be positive")
    if resume_start is not None and resume_start >= total_samples:
        raise ValueError("--resume-from must be smaller than the selected number of tasks")
    if args.concurrent_tasks > 1:
        run_task_workers(args, total_samples, cache_steps, resume_start)
        return
    assigned_tasks = task_indices(total_samples, args.task_worker_index, args.task_worker_count)
    if not any((resume_start is None or task >= resume_start) and
               (resume_start is None or not task_complete(
                   task_dir(args.output_root, args.group, task, args.seed),
                   cache_steps, args.num_particles)) for task in assigned_tasks):
        print("All assigned tasks already complete; no model load needed", flush=True)
        return

    model_path = args.models_root / args.model_name
    from transformers import T5TokenizerFast

    tokenizer = T5TokenizerFast(tokenizer_file=str(model_path / "tokenizer" / "tokenizer.json"), extra_ids=100)
    pipe = DualDiffSD3Pipeline.from_pretrained(
        str(model_path), tokenizer=tokenizer, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    ensure_stage2_modules(pipe.transformer, pipe.vae.config.latent_channels)
    restore_stage2_parameters(pipe.transformer, str(model_path))
    pipe.transformer.eval()
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.set_sampling_mode("t2i")
    if args.cpu_offload and device.type == "cuda":
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)

    transform = T.Compose(
        [
            T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(512),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )
    scheduler = pipe.scheduler
    mask_keys = ["Color", "Material", "Design features", "Clothing Fashion Style"]
    progress = tqdm(assigned_tasks, desc=f"{args.model_name} | FITB offline cache "
                    f"worker {args.task_worker_index+1}/{args.task_worker_count}", unit="task")

    for task_index in progress:
        if resume_start is not None and task_index < resume_start:
            progress.set_postfix_str("resume skip")
            continue

        output_dir = task_dir(args.output_root, args.group, task_index, args.seed)
        if resume_start is not None and task_complete(output_dir, cache_steps, args.num_particles):
            progress.set_postfix_str("cached")
            continue

        task_started = time.perf_counter()
        uid = int(fitb["uids"][task_index])
        oid = int(fitb["oids"][task_index])
        incomplete = fitb["outfits"][task_index].tolist()
        full_outfit = ground_truth["outfits"][task_index]
        categories = fitb["category"][task_index].tolist()
        blank_idx = incomplete.index(0)
        target_item_id = int(full_outfit[blank_idx])
        target_category_id = int(categories[blank_idx])
        context_ids = [int(item) for index, item in enumerate(full_outfit) if index != blank_idx]

        preference_text = build_preference_text(preferences, uid, target_category_id, mask_keys)
        category_name = id_to_category.get(target_category_id, "item")
        sentence_text = f"Recommend a fashion {category_name} item, on white background."
        with torch.no_grad():
            preference_emb = encode_text(pipe.tokenizer, pipe.text_encoder, preference_text, args.max_length, device).to(
                device=device, dtype=pipe.transformer.dtype
            )
            sentence_emb = encode_text(pipe.tokenizer, pipe.text_encoder, sentence_text, args.max_length, device).to(
                device=device, dtype=pipe.transformer.dtype
            )
            negative_segment = encode_text(
                pipe.tokenizer, pipe.text_encoder, args.negative_prompt, args.max_length, device
            ).to(device=device, dtype=pipe.transformer.dtype)
        text_base = prepend_pad_segment(sentence_emb, negative_segment, preference_emb.shape[1] + sentence_emb.shape[1])
        preference_base = torch.cat([preference_emb, sentence_emb], dim=1).contiguous()
        uncond_base = torch.cat([negative_segment, negative_segment], dim=1).contiguous()

        context_images = torch.stack(
            [load_image_tensor(item, image_paths, str(args.image_root), transform) for item in context_ids], dim=0
        ).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=pipe.transformer.dtype
        ):
            context_flat = context_images.reshape(-1, *context_images.shape[2:])
            context_dist = pipe.vae.encode(context_flat).latent_dist
            context_latents = context_dist.mode() if args.deterministic_context else context_dist.sample()
            context_latents = (context_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
        context_latents = context_latents.reshape(1, context_images.size(1), *context_latents.shape[1:]).to(
            device=device, dtype=pipe.transformer.dtype
        )
        context_offset = compute_stage2_context_offset(pipe.transformer, context_latents)
        latent_shape = tuple(context_latents.shape[2:])

        scheduler.set_timesteps(args.num_inference_steps, device=device)
        timesteps = scheduler.timesteps.to(device)
        candidate_seeds = [args.seed + task_index * args.seed_stride + index for index in range(args.num_particles)]
        active = [
            {
                "seed": seed,
                "latent": torch.randn(
                    (1, *latent_shape),
                    generator=torch.Generator(device=device).manual_seed(seed),
                    device=device,
                    dtype=pipe.transformer.dtype,
                ),
            }
            for seed in candidate_seeds
        ]
        initial_latent_hashes = {
            str(particle["seed"]): hashlib.sha256(
                particle["latent"].detach().float().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            for particle in active
        }
        cache_rows: list[dict] = []
        trace: list[dict] = []
        transformer_forward_calls = 0
        vae_decode_calls = 0

        for step_idx, timestep in enumerate(timesteps, start=1):
            for particle in active:
                latent = particle["latent"]
                context_base = context_offset.expand(1, *context_offset.shape[1:]).contiguous()
                context_zero = torch.zeros_like(context_base)
                latent_model_input = torch.cat([latent, latent, latent, latent], dim=0)
                latent_model_input = latent_model_input + MATCHING_LAMBDA * torch.cat(
                    [context_zero, context_zero, context_base, context_base], dim=0
                )
                prompt_embeds = torch.cat([uncond_base, text_base, text_base, preference_base], dim=0)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type, enabled=device.type == "cuda", dtype=pipe.transformer.dtype
                ):
                    prediction = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep.expand(4),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=None,
                        return_dict=False,
                    )[0]
                transformer_forward_calls += 1
                uncond, text, matching, preference = prediction.chunk(4)
                prediction = (
                    uncond
                    + args.text_guidance_scale * (text - uncond)
                    + args.context_guidance_scale * (matching - text)
                    + args.preference_guidance_scale * (preference - matching)
                )
                before = latent
                particle["x0_prediction"] = estimate_x0_from_flowmatch(
                    scheduler, prediction, before, step_idx - 1
                )
                scheduler._step_index = step_idx - 1
                particle["latent"] = scheduler.step(prediction, timestep, latent, return_dict=False)[0]
                particle["before"] = before

            if step_idx in cache_steps:
                x0_batch = [particle["x0_prediction"] for particle in active]
                decoded_x0 = torch.cat([decode_latent(pipe, latent) for latent in x0_batch], dim=0)
                vae_decode_calls += len(decoded_x0)
                step_rows = []
                for particle, image in zip(active, decoded_x0):
                    seed = int(particle["seed"])
                    image_path = output_dir / "offline_candidates" / f"{uid}_{oid}" / f"step_{step_idx:02d}" / f"seed_{seed}_x0.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(((image + 1.0) / 2.0).clamp(0, 1), str(image_path))
                    row = {
                        "step": step_idx,
                        "seed": seed,
                        "image_path": str(image_path.resolve()),
                        "image_kind": "flowmatch_x0_prediction",
                        "scheduler_timestep": float(timestep.detach().float().cpu()),
                        "sigma": float(scheduler.sigmas[step_idx - 1].detach().float().cpu()),
                    }
                    cache_rows.append(row)
                    step_rows.append(row)
                trace.append({"step": step_idx, "batch_before": args.num_particles,
                              "batch_after": args.num_particles, "offline_cache_images": step_rows})
            if device.type == "cuda":
                torch.cuda.empty_cache()

        metadata = {
            "uid": uid,
            "oid": oid,
            "task_index": task_index,
            "group_id": args.group,
            "base_seed": args.seed,
            "seed_stride": args.seed_stride,
            "target_item_id": target_item_id,
            "candidate_seeds": candidate_seeds,
            "initial_latent_hashes": initial_latent_hashes,
            "num_inference_steps": args.num_inference_steps,
            "checkpoint": str(model_path.resolve()),
            "precision": str(pipe.transformer.dtype),
            "deterministic_context": args.deterministic_context,
            "guidance_scales": {
                "text": args.text_guidance_scale,
                "context": args.context_guidance_scale,
                "preference": args.preference_guidance_scale,
            },
            "reward": "not evaluated; offline candidate cache only",
            "transformer_forward_calls": transformer_forward_calls,
            "vae_encode_calls": 1,
            "vae_decode_calls": vae_decode_calls,
            "reward_score_events": 0,
            "reward_model_calls": 0,
            "offline_candidate_cache": {
                "enabled": True,
                "steps": cache_steps,
                "candidate_count": args.num_particles,
                "expected_image_count": len(cache_steps) * args.num_particles,
                "image_count": len(cache_rows),
                "image_kind": "flowmatch_x0_prediction",
                "images": cache_rows,
            },
            "trace": trace,
            "elapsed_s": time.perf_counter() - task_started,
        }
        if len(cache_rows) != expected_cache_image_count:
            raise RuntimeError(
                f"Task {task_index} saved {len(cache_rows)} cache images instead of {expected_cache_image_count}"
            )
        save_metadata(output_dir / "metadata.json", metadata)
        progress.set_postfix_str(
            f"task={task_index} cache={expected_cache_image_count} elapsed={metadata['elapsed_s']:.0f}s"
        )

    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
