"""Benchmark genuinely particle-parallel DualFashion search on FITB and GOR.

This is intentionally separate from the offline-cache and greedy runners.  Those
prototypes call the denoiser once per seed to fit a small GPU.  Here all active
particles form one tensor batch: BoN@4 uses B=4 throughout, PSP@8 uses
B=8 -> 4 -> 2, and BoN@8 uses B=8 throughout.  No images are written.

The default Personal reward is the same frozen OpenCLIP reward used by the
online greedy GOR experiment.  Therefore ``runtime_s`` is an end-to-end search
measurement after the DualFashion pipeline has loaded, while ``denoise_s``
isolates the flow-matching denoiser portion.  ``--reward none`` is available
only for generator microbenchmarks; it must not be reported as full PSP time.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms as T


ROOT = Path(__file__).resolve().parents[2]
DUALFASHION_ROOT = ROOT / "DualFashion-main"
if str(DUALFASHION_ROOT) not in sys.path:
    sys.path.insert(0, str(DUALFASHION_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Inference_ifashion_PFITB import (  # noqa: E402
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
from run_ifashion_gor_greedy_personal import (  # noqa: E402
    PERSONAL_CLIP_WEIGHTS,
    PersonalReward,
    load_personal_clip,
)
from run_psp_ifashion_smoke import decode_latent, estimate_x0_from_flowmatch  # noqa: E402
from sd3_modules.stage2_pipeline import DualDiffSD3Pipeline  # noqa: E402


NEGATIVE_PROMPT = (
    "low quality, blurry, low resolution, backlit, cartoon, animated, deformed, "
    "oversaturated, undersaturated, out of frame"
)
MASK_KEYS = ["Color", "Material", "Design features", "Clothing Fashion Style"]


@dataclass(frozen=True)
class Method:
    name: str
    particles: int
    schedule: tuple[tuple[int, int], ...]

    @property
    def particle_steps(self) -> int:
        previous, total = 0, 0
        active = self.particles
        for step, keep in self.schedule:
            total += (step - previous) * active
            previous, active = step, keep
        return total

    @property
    def batch_schedule(self) -> str:
        active = self.particles
        values = []
        for _, keep in self.schedule:
            values.append(str(active))
            active = keep
        return " -> ".join(values)


METHODS = (
    Method("bon4", 4, ((28, 1),)),
    Method("psp8", 8, ((7, 4), (14, 2), (28, 1))),
    Method("bon8", 8, ((28, 1),)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("fitb", "gor", "both"), default="both")
    parser.add_argument("--models-root", type=Path, default=DUALFASHION_ROOT / "checkpoint")
    parser.add_argument("--model-name", default="0057501")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-info-root", type=Path, default=DUALFASHION_ROOT / "processed_info")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--fitb-file", default="processed/fitb_test.npy")
    parser.add_argument("--task-indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--item-seed-stride", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--text-guidance-scale", type=float, default=8.0)
    parser.add_argument("--context-guidance-scale", type=float, default=7.0)
    parser.add_argument("--preference-guidance-scale", type=float, default=8.0)
    parser.add_argument("--reward", choices=("personal", "none"), default="personal")
    parser.add_argument("--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_indices(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or min(values) < 0:
        raise ValueError("--task-indices must be a non-empty comma-separated list of non-negative integers")
    return values


def prompt_for(pipe, uid: int, category: int, category_names: dict, preferences: dict, args, device):
    preference_text = build_preference_text(preferences, uid, category, MASK_KEYS)
    sentence_text = f"Recommend a fashion {category_names.get(category, 'item')} item, on white background."
    with torch.inference_mode():
        preference = encode_text(pipe.tokenizer, pipe.text_encoder, preference_text, args.max_length, device)
        sentence = encode_text(pipe.tokenizer, pipe.text_encoder, sentence_text, args.max_length, device)
        negative = encode_text(pipe.tokenizer, pipe.text_encoder, NEGATIVE_PROMPT, args.max_length, device)
    preference = preference.to(device=device, dtype=pipe.transformer.dtype)
    sentence = sentence.to(device=device, dtype=pipe.transformer.dtype)
    negative = negative.to(device=device, dtype=pipe.transformer.dtype)
    return {
        "uncond": torch.cat([negative, negative], dim=1).contiguous(),
        "text": prepend_pad_segment(sentence, negative, preference.shape[1] + sentence.shape[1]),
        "preference": torch.cat([preference, sentence], dim=1).contiguous(),
    }


def batched_velocity(pipe, latents: torch.Tensor, timestep: torch.Tensor, prompt: dict, context_offset: torch.Tensor, args):
    """One Transformer launch for every currently active particle."""
    batch = latents.shape[0]
    context = context_offset.expand(batch, *context_offset.shape[1:]).contiguous()
    zeros = torch.zeros_like(context)
    model_input = torch.cat([latents, latents, latents, latents], dim=0)
    model_input = model_input + MATCHING_LAMBDA * torch.cat([zeros, zeros, context, context], dim=0)
    prompt_embeds = torch.cat(
        [
            prompt["uncond"].repeat(batch, 1, 1),
            prompt["text"].repeat(batch, 1, 1),
            prompt["text"].repeat(batch, 1, 1),
            prompt["preference"].repeat(batch, 1, 1),
        ],
        dim=0,
    ).contiguous()
    with torch.inference_mode(), torch.autocast(
        device_type=latents.device.type, enabled=latents.device.type == "cuda", dtype=pipe.transformer.dtype
    ):
        prediction = pipe.transformer(
            hidden_states=model_input,
            timestep=timestep.expand(batch * 4),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=None,
            return_dict=False,
        )[0]
    uncond, text, matching, preference = prediction.chunk(4)
    return (
        uncond
        + args.text_guidance_scale * (text - uncond)
        + args.context_guidance_scale * (matching - text)
        + args.preference_guidance_scale * (preference - matching)
    )


def latents_from_seeds(pipe, seeds: list[int], shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.stack(
        [
            torch.randn(
                shape,
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device,
                dtype=pipe.transformer.dtype,
            )
            for seed in seeds
        ],
        dim=0,
    )


def score_images(decoded: torch.Tensor, seeds: list[int], uid: int, category: int, reward) -> torch.Tensor:
    if reward is None:
        # The value only controls which tensor rows survive. It is deliberately
        # deterministic and has no data-dependent compute, for microbenchmarks.
        return torch.arange(len(seeds), 0, -1, dtype=torch.float32)
    scores = reward.score({seed: decoded[index] for index, seed in enumerate(seeds)}, uid, category)
    return torch.tensor([scores[seed] for seed in seeds], dtype=torch.float32)


def particle_search(pipe, method: Method, prompt: dict, context_latents: torch.Tensor, uid: int, category: int,
                    seeds: list[int], reward, args) -> tuple[torch.Tensor, dict]:
    """Run one FITB item or one sequential GOR item and return its selected latent."""
    device = context_latents.device
    context_offset = compute_stage2_context_offset(pipe.transformer, context_latents)
    latent_shape = tuple(context_latents.shape[2:])
    active_seeds = list(seeds)
    latents = latents_from_seeds(pipe, active_seeds, latent_shape, device)
    nodes = dict(method.schedule)
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    denoise_s = 0.0
    reward_s = 0.0
    launches = 0
    final = None
    for step, timestep in enumerate(pipe.scheduler.timesteps, start=1):
        started = time.perf_counter()
        velocity = batched_velocity(pipe, latents, timestep, prompt, context_offset, args)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        denoise_s += time.perf_counter() - started
        x0 = estimate_x0_from_flowmatch(pipe.scheduler, velocity, latents, step - 1)
        pipe.scheduler._step_index = step - 1
        latents = pipe.scheduler.step(velocity, timestep, latents, return_dict=False)[0].detach()
        launches += 1
        if step not in nodes:
            continue
        # One batched VAE decode followed by one batched Personal reward call.
        decoded = decode_latent(pipe, x0)
        started = time.perf_counter()
        scores = score_images(decoded, active_seeds, uid, category, reward)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        reward_s += time.perf_counter() - started
        keep = nodes[step]
        rank = sorted(range(len(active_seeds)), key=lambda index: (-float(scores[index]), active_seeds[index]))[:keep]
        active_seeds = [active_seeds[index] for index in rank]
        latents = latents[rank].detach()
        if step == args.num_inference_steps:
            final = latents[:1]
    if final is None:
        raise RuntimeError("terminal selection was not reached")
    return final, {"denoise_s": denoise_s, "reward_s": reward_s, "transformer_launches": launches}


def fitb_context(pipe, task: int, fitb: dict, ground_truth: dict, category_names: dict, preferences: dict,
                 image_paths, args, device, transform):
    uid = int(fitb["uids"][task])
    full = [int(value) for value in ground_truth["outfits"][task]]
    incomplete = [int(value) for value in fitb["outfits"][task].tolist()]
    categories = [int(value) for value in fitb["category"][task].tolist()]
    blank = incomplete.index(0)
    context_ids = [item for index, item in enumerate(full) if index != blank]
    context_images = torch.stack(
        [load_image_tensor(item, image_paths, str(args.image_root), transform) for item in context_ids], dim=0
    ).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, enabled=device.type == "cuda", dtype=pipe.transformer.dtype
    ):
        flattened = context_images.reshape(-1, *context_images.shape[2:])
        latents = pipe.vae.encode(flattened).latent_dist.mode()
        latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    context = latents.reshape(1, context_images.shape[1], *latents.shape[1:]).to(dtype=pipe.transformer.dtype)
    return uid, categories[blank], prompt_for(pipe, uid, categories[blank], category_names, preferences, args, device), context


def run_fitb(pipe, method: Method, task: int, resources: dict, reward, args, device, transform) -> dict:
    started = time.perf_counter()
    uid, category, prompt, context = fitb_context(pipe, task, **resources, args=args, device=device, transform=transform)
    seeds = [args.seed + task * args.seed_stride + index for index in range(method.particles)]
    _, phases = particle_search(pipe, method, prompt, context, uid, category, seeds, reward, args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"runtime_s": time.perf_counter() - started, **phases, "generated_items": 1}


def run_gor(pipe, method: Method, task: int, resources: dict, reward, args, device) -> dict:
    started = time.perf_counter()
    ground_truth, category_names, preferences = resources["ground_truth"], resources["category_names"], resources["preferences"]
    uid = int(ground_truth["uids"][task])
    item_categories = [int(value) for value in ground_truth["category"][task]]
    channels = pipe.vae.config.latent_channels
    height = args.resolution // pipe.vae_scale_factor
    width = args.resolution // pipe.vae_scale_factor
    context = torch.zeros(1, 1, channels, height, width, device=device, dtype=pipe.transformer.dtype)
    selected = []
    denoise_s = reward_s = 0.0
    launches = 0
    # The standard order fixes GOR's context cardinality as 0 -> 1 -> 2 -> 3.
    # Timing is invariant to the category permutation, so the canonical order is enough.
    for position, category in enumerate(item_categories):
        prompt = prompt_for(pipe, uid, category, category_names, preferences, args, device)
        seeds = [args.seed + task * args.seed_stride + position * args.item_seed_stride + index for index in range(method.particles)]
        final, phases = particle_search(pipe, method, prompt, context, uid, category, seeds, reward, args)
        selected.append(final)
        selected = selected[-3:]
        context = torch.stack(selected, dim=1)
        denoise_s += phases["denoise_s"]
        reward_s += phases["reward_s"]
        launches += phases["transformer_launches"]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"runtime_s": time.perf_counter() - started, "denoise_s": denoise_s, "reward_s": reward_s,
            "transformer_launches": launches, "generated_items": len(item_categories)}


def gpu_stats(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return float("nan"), float("nan")
    return torch.cuda.max_memory_allocated(device) / 2**30, torch.cuda.max_memory_reserved(device) / 2**30


def clean(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def write_outputs(output: Path, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "per_task.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["status"] == "ok":
            groups.setdefault((row["task_type"], row["method"]), []).append(row)
    summary = []
    for (task_type, method), group in groups.items():
        result = {
            "task_type": task_type, "method": method, "n": len(group),
            "runtime_s_mean": float(np.mean([float(row["runtime_s"]) for row in group])),
            "runtime_s_median": float(np.median([float(row["runtime_s"]) for row in group])),
            "peak_allocated_gib_max": float(np.max([float(row["peak_allocated_gib"]) for row in group])),
            "peak_reserved_gib_max": float(np.max([float(row["peak_reserved_gib"]) for row in group])),
            "denoise_s_mean": float(np.mean([float(row["denoise_s"]) for row in group])),
            "reward_s_mean": float(np.mean([float(row["reward_s"]) for row in group])),
            "particle_steps": group[0]["particle_steps"], "transformer_launches": group[0]["transformer_launches"],
            "batch_schedule": group[0]["batch_schedule"], "reward": group[0]["reward"],
        }
        summary.append(result)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else ["task_type"])
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    args = parse_args()
    if args.num_inference_steps != 28:
        raise ValueError("This benchmark is defined for DualFashion's released 28-step scheduler.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    indices = parse_indices(args.task_indices)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA device is required for Runtime/Peak VRAM benchmarking.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model_path = args.models_root / args.model_name
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast(tokenizer_file=str(model_path / "tokenizer" / "tokenizer.json"), extra_ids=100)
    pipe = DualDiffSD3Pipeline.from_pretrained(str(model_path), tokenizer=tokenizer, torch_dtype=torch.bfloat16)
    ensure_stage2_modules(pipe.transformer, pipe.vae.config.latent_channels)
    restore_stage2_parameters(pipe.transformer, str(model_path))
    pipe.transformer.eval()
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.set_sampling_mode("t2i")
    pipe = pipe.to(device)
    reward = None
    if args.reward == "personal":
        history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
        clip, preprocess = load_personal_clip(args.personal_clip_weights, device)
        reward = PersonalReward(history, clip, preprocess, device, cpu_offload=False)
    ground_truth = load_numpy_dict(str(args.data_root / "test_grd_dict.npy"))
    category_names = load_numpy_dict(str(args.data_root / "id_cate_dict.npy"))
    preferences = load_numpy_dict(str(args.processed_info_root / "stage2_ifashion_test" / "test_preference.npy"))
    fitb = load_numpy_dict(str(args.data_root / args.fitb_file))
    image_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    if max(indices) >= len(ground_truth["uids"]) or max(indices) >= len(fitb["uids"]):
        raise IndexError("--task-indices exceeds the selected test split")
    fitb_resources = {"fitb": fitb, "ground_truth": ground_truth, "category_names": category_names,
                      "preferences": preferences, "image_paths": image_paths}
    gor_resources = {"ground_truth": ground_truth, "category_names": category_names, "preferences": preferences}
    transform = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(512), T.ToTensor(), T.Normalize([0.5], [0.5])])
    kinds = ("fitb", "gor") if args.task == "both" else (args.task,)
    rows: list[dict] = []
    for task_type in kinds:
        runner = run_fitb if task_type == "fitb" else run_gor
        resources = fitb_resources if task_type == "fitb" else gor_resources
        for method in METHODS:
            # Warmups compile/cache kernels but are never part of the reported samples.
            for warmup_index in range(args.warmup):
                clean(device)
                runner(pipe, method, indices[warmup_index % len(indices)], resources, reward, args, device, transform) if task_type == "fitb" else runner(pipe, method, indices[warmup_index % len(indices)], resources, reward, args, device)
            for task in indices:
                clean(device)
                torch.cuda.reset_peak_memory_stats(device)
                try:
                    result = runner(pipe, method, task, resources, reward, args, device, transform) if task_type == "fitb" else runner(pipe, method, task, resources, reward, args, device)
                    allocated, reserved = gpu_stats(device)
                    row = {"status": "ok", "task_type": task_type, "task_index": task, "method": method.name,
                           "particles": method.particles, "batch_schedule": method.batch_schedule,
                           "particle_steps": method.particle_steps * result["generated_items"],
                           "transformer_launches": result["transformer_launches"], "reward": args.reward,
                           "precision": str(pipe.transformer.dtype), "device": torch.cuda.get_device_name(device),
                           "cpu_offload": False, "peak_allocated_gib": allocated, "peak_reserved_gib": reserved, **result}
                except torch.cuda.OutOfMemoryError as error:
                    clean(device)
                    row = {"status": "oom", "task_type": task_type, "task_index": task, "method": method.name,
                           "particles": method.particles, "batch_schedule": method.batch_schedule,
                           "particle_steps": method.particle_steps * (1 if task_type == "fitb" else 4),
                           "transformer_launches": "", "reward": args.reward, "precision": str(pipe.transformer.dtype),
                           "device": torch.cuda.get_device_name(device), "cpu_offload": False, "error": str(error).split("\n")[0]}
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    write_outputs(args.output_dir, rows)
    print(f"Saved {len(rows)} measurements to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
