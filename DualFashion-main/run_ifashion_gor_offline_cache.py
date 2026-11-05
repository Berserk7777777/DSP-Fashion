"""Create resumable GOR image-only candidate caches for offline reward research.

Each candidate is a complete sequential outfit trajectory.  The saved images are
the flow-matching x0 predictions, not noisy intermediate latents.  This makes
the cache usable for offline reward studies without repeatedly loading SD3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torchvision.utils import save_image

from Inference_ifashion_PFITB import (
    MATCHING_LAMBDA,
    build_preference_text,
    compute_stage2_context_offset,
    encode_text,
    ensure_stage2_modules,
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
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=2207)
    parser.add_argument("--num-particles", type=int, choices=(4, 8), required=True)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--offline-cache-steps", required=True, help="One-based steps, e.g. 28 or 7,14,28.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--text-guidance-scale", type=float, default=8.0)
    parser.add_argument("--context-guidance-scale", type=float, default=7.0)
    parser.add_argument("--preference-guidance-scale", type=float, default=8.0)
    parser.add_argument(
        "--negative-prompt",
        default="low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--resume-from", default="auto", help="auto, none, or a non-negative GOR task index")
    return parser.parse_args()


def parse_resume_start(value: str) -> int | None:
    value = value.strip().lower()
    if value == "none":
        return None
    if value in {"", "auto"}:
        return 0
    try:
        start = int(value)
    except ValueError as exc:
        raise ValueError("--resume-from must be auto, none, or a non-negative task index") from exc
    if start < 0:
        raise ValueError("--resume-from task index must be non-negative")
    return start


def task_dir(output_root: Path, task_index: int, uid: int, oid: int) -> Path:
    return output_root / f"task{task_index:04d}_{uid}_{oid}"


def task_complete(path: Path, expected_images: int, expected_steps: list[int], expected_particles: int) -> bool:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        cache = json.loads(metadata_path.read_text(encoding="utf-8"))["offline_candidate_cache"]
        images = cache["images"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        cache.get("enabled") is True
        and list(cache.get("steps", [])) == expected_steps
        and cache.get("candidate_count") == expected_particles
        and len(images) == expected_images
        and all(Path(row.get("image_path", "")).exists() for row in images)
    )


def save_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def make_prompt_config(pipe, uid: int, category_id: int, id_to_category: dict, preferences: dict, args, device):
    mask_keys = ["Color", "Material", "Design features", "Clothing Fashion Style"]
    preference_text = build_preference_text(preferences, uid, category_id, mask_keys)
    category_name = id_to_category.get(category_id, "item")
    sentence_text = f"Recommend a fashion {category_name} item, on white background."
    with torch.no_grad():
        preference = encode_text(pipe.tokenizer, pipe.text_encoder, preference_text, args.max_length, device)
        sentence = encode_text(pipe.tokenizer, pipe.text_encoder, sentence_text, args.max_length, device)
        negative = encode_text(pipe.tokenizer, pipe.text_encoder, args.negative_prompt, args.max_length, device)
    preference = preference.to(device=device, dtype=pipe.transformer.dtype)
    sentence = sentence.to(device=device, dtype=pipe.transformer.dtype)
    negative = negative.to(device=device, dtype=pipe.transformer.dtype)
    return {
        "preference_text": preference_text,
        "sentence_text": sentence_text,
        "text": prepend_pad_segment(sentence, negative, preference.shape[1] + sentence.shape[1]),
        "preference": torch.cat([preference, sentence], dim=1).contiguous(),
        "uncond": torch.cat([negative, negative], dim=1).contiguous(),
    }


def denoise_one_item(
    pipe,
    scheduler,
    prompt: dict,
    context_latents: torch.Tensor,
    generator: torch.Generator,
    latent_channels: int,
    args: argparse.Namespace,
    cache_steps: list[int],
    candidate_seed: int,
    generation_position: int,
    original_position: int,
    item_id: int,
    category_id: int,
    output_dir: Path,
) -> tuple[torch.Tensor, list[dict], int]:
    device = context_latents.device
    context_offset = compute_stage2_context_offset(pipe.transformer, context_latents)
    latent = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=latent_channels,
        height=args.resolution,
        width=args.resolution,
        dtype=pipe.transformer.dtype,
        device=device,
        generator=generator,
    )
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    image_rows: list[dict] = []
    forward_calls = 0
    for step_idx, timestep in enumerate(scheduler.timesteps, start=1):
        base = context_offset.expand(1, *context_offset.shape[1:]).contiguous()
        zero = torch.zeros_like(base)
        model_input = torch.cat([latent, latent, latent, latent], dim=0)
        model_input = model_input + MATCHING_LAMBDA * torch.cat([zero, zero, base, base], dim=0)
        prompt_embeds = torch.cat([prompt["uncond"], prompt["text"], prompt["text"], prompt["preference"]], dim=0)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, enabled=device.type == "cuda", dtype=pipe.transformer.dtype
        ):
            prediction = pipe.transformer(
                hidden_states=model_input,
                timestep=timestep.expand(4),
                encoder_hidden_states=prompt_embeds,
                pooled_projections=None,
                return_dict=False,
            )[0]
        forward_calls += 1
        uncond, text, matching, preference = prediction.chunk(4)
        velocity = (
            uncond
            + args.text_guidance_scale * (text - uncond)
            + args.context_guidance_scale * (matching - text)
            + args.preference_guidance_scale * (preference - matching)
        )
        x0_prediction = estimate_x0_from_flowmatch(scheduler, velocity, latent, step_idx - 1)
        scheduler._step_index = step_idx - 1
        latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0]
        if step_idx in cache_steps:
            image = decode_latent(pipe, x0_prediction)[0]
            image_path = (
                output_dir
                / "offline_candidates"
                / f"candidate_{candidate_seed}"
                / f"position_{generation_position:02d}_item_{item_id}"
                / f"step_{step_idx:02d}_x0.png"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            save_image(((image + 1.0) / 2.0).clamp(0, 1), str(image_path))
            image_rows.append(
                {
                    "step": step_idx,
                    "candidate_seed": candidate_seed,
                    "generation_position": generation_position,
                    "original_outfit_position": original_position,
                    "item_id": item_id,
                    "category_id": category_id,
                    "image_kind": "flowmatch_x0_prediction",
                    "image_path": str(image_path.resolve()),
                }
            )
    return latent.detach(), image_rows, forward_calls


def main() -> None:
    args = parse_args()
    if args.seed_stride < args.num_particles:
        raise ValueError("--seed-stride must be at least --num-particles")
    cache_steps = parse_step_list(args.offline_cache_steps, args.num_inference_steps)
    if not cache_steps or cache_steps[-1] != args.num_inference_steps:
        raise ValueError("--offline-cache-steps must include the final inference step")
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

    ground_truth = load_numpy_dict(str(args.data_root / "test_grd_dict.npy"))
    id_to_category = load_numpy_dict(str(args.data_root / "id_cate_dict.npy"))
    preferences = load_numpy_dict(str(args.processed_info_root / "stage2_ifashion_test" / "test_preference.npy"))
    total_samples = min(len(ground_truth["uids"]), args.max_samples)
    if total_samples < 1:
        raise ValueError("--max-samples must be positive")
    if resume_start is not None and resume_start >= total_samples:
        raise ValueError("--resume-from must be smaller than the selected number of tasks")

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

    latent_channels = pipe.vae.config.latent_channels
    latent_height = args.resolution // pipe.vae_scale_factor
    latent_width = args.resolution // pipe.vae_scale_factor
    zero_context = torch.zeros(
        1, 1, latent_channels, latent_height, latent_width, device=device, dtype=pipe.transformer.dtype
    )
    scheduler = pipe.scheduler
    progress = tqdm(range(total_samples), desc=f"{args.model_name} | GOR offline cache", unit="task")
    for task_index in progress:
        if resume_start is not None and task_index < resume_start:
            progress.set_postfix_str("resume skip")
            continue
        uid = int(ground_truth["uids"][task_index])
        oid = int(ground_truth["oids"][task_index])
        item_ids = [int(item) for item in ground_truth["outfits"][task_index]]
        category_ids = [int(category) for category in ground_truth["category"][task_index]]
        output_dir = task_dir(args.output_root, task_index, uid, oid)
        expected_images = len(item_ids) * args.num_particles * len(cache_steps)
        if resume_start is not None and task_complete(output_dir, expected_images, cache_steps, args.num_particles):
            progress.set_postfix_str("cached")
            continue
        task_started = time.perf_counter()
        prompts = [
            make_prompt_config(pipe, uid, category, id_to_category, preferences, args, device)
            for category in category_ids
        ]
        candidate_seeds = [args.seed + task_index * args.seed_stride + slot for slot in range(args.num_particles)]
        all_rows: list[dict] = []
        candidate_records: list[dict] = []
        total_forward_calls = 0
        for candidate_seed in candidate_seeds:
            gpu_generator = torch.Generator(device=device).manual_seed(candidate_seed)
            cpu_generator = torch.Generator(device="cpu").manual_seed(candidate_seed)
            order = torch.randperm(len(item_ids), generator=cpu_generator).tolist()
            previous_latents: list[torch.Tensor] = []
            trajectory_rows: list[dict] = []
            initial_hashes: list[str] = []
            for generation_position, original_position in enumerate(order):
                context = zero_context if not previous_latents else torch.stack(previous_latents, dim=1)
                final_latent, rows, forward_calls = denoise_one_item(
                    pipe=pipe,
                    scheduler=scheduler,
                    prompt=prompts[original_position],
                    context_latents=context,
                    generator=gpu_generator,
                    latent_channels=latent_channels,
                    args=args,
                    cache_steps=cache_steps,
                    candidate_seed=candidate_seed,
                    generation_position=generation_position,
                    original_position=original_position,
                    item_id=item_ids[original_position],
                    category_id=category_ids[original_position],
                    output_dir=output_dir,
                )
                total_forward_calls += forward_calls
                all_rows.extend(rows)
                trajectory_rows.extend(rows)
                initial_hashes.append(hashlib.sha256(final_latent.float().cpu().contiguous().numpy().tobytes()).hexdigest())
                previous_latents.append(final_latent)
                if len(previous_latents) > 3:
                    previous_latents = previous_latents[-3:]
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            candidate_records.append(
                {
                    "candidate_seed": candidate_seed,
                    "generation_order_original_positions": order,
                    "generation_order_item_ids": [item_ids[index] for index in order],
                    "terminal_latent_hashes": initial_hashes,
                    "images": trajectory_rows,
                }
            )
        if len(all_rows) != expected_images:
            raise RuntimeError(f"Task {task_index} saved {len(all_rows)} images, expected {expected_images}")
        metadata = {
            "uid": uid,
            "oid": oid,
            "task_index": task_index,
            "ground_truth_item_ids": item_ids,
            "ground_truth_category_ids": category_ids,
            "candidate_seeds": candidate_seeds,
            "candidate_trajectories": candidate_records,
            "num_inference_steps": args.num_inference_steps,
            "checkpoint": str(model_path.resolve()),
            "precision": str(pipe.transformer.dtype),
            "guidance_scales": {
                "text": args.text_guidance_scale,
                "context": args.context_guidance_scale,
                "preference": args.preference_guidance_scale,
            },
            "reward": "not evaluated; image-only offline candidate cache",
            "reward_score_events": 0,
            "reward_model_calls": 0,
            "transformer_forward_calls": total_forward_calls,
            "vae_decode_calls": len(all_rows),
            "offline_candidate_cache": {
                "enabled": True,
                "task_type": "GOR",
                "selection_unit": "complete sequential outfit trajectory",
                "steps": cache_steps,
                "candidate_count": args.num_particles,
                "items_per_candidate": len(item_ids),
                "expected_image_count": expected_images,
                "image_count": len(all_rows),
                "image_kind": "flowmatch_x0_prediction",
                "images": all_rows,
            },
            "elapsed_s": time.perf_counter() - task_started,
        }
        save_metadata(output_dir / "metadata.json", metadata)
        progress.set_postfix_str(f"task={task_index} cache={len(all_rows)} elapsed={metadata['elapsed_s']:.0f}s")

    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
