"""Online item-greedy GOR search with frozen Personal reward.

Unlike the earlier GOR cache, a candidate is an image for the *current* item.
After selecting that item, its final latent becomes shared context for the next
item.  The item order is frozen from the first (standard) BoN trajectory so
BoN and PSP vary only through their current-item particle pools.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from tqdm import tqdm

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
from run_psp_ifashion_smoke import decode_latent, estimate_x0_from_flowmatch
from sd3_modules.stage2_pipeline import DualDiffSD3Pipeline
from fashion_psp.baselines.fk_steering import importance_weights, resample_parent_indices
from fashion_psp.search.fixed_six_seed_psp import RULE as SIX_SEED_RULE


PERSONAL_CLIP_WEIGHTS = Path(
    r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    r"\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bon", "psp", "fk", "dsp"), required=True,
                        help="dsp: fixed six-seed Personal pruning, 7:4,21:2,28:1 (112 steps/item).")
    parser.add_argument("--models-root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--model-name", default="0057501")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--order-reference-root", type=Path, required=True,
                        help="Existing BoN offline cache; its first trajectory fixes each task's item order.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument("--item-seed-stride", type=int, default=64)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument(
        "--fk-resample-steps", default="6,11,17,22",
        help="Intermediate one-based FK nodes. For 28 steps, 6/11/17/22 are the nearest 20/40/60/80 percent nodes.",
    )
    parser.add_argument("--fk-potential", choices=("diff", "max", "add", "rt"), default="diff")
    parser.add_argument("--fk-lambda", type=float, default=0.2)
    parser.add_argument("--fk-resampling", choices=("multinomial", "systematic"), default="systematic")
    parser.add_argument("--fk-resampling-seed", type=int, default=9090)
    parser.add_argument("--fk-stochastic-seed", type=int, default=8191)
    parser.add_argument("--fk-gamma", type=float, default=0.005)
    parser.add_argument("--fk-adaptive-resampling", action="store_true")
    parser.add_argument("--fk-ess-fraction", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--text-guidance-scale", type=float, default=8.0)
    parser.add_argument("--context-guidance-scale", type=float, default=7.0)
    parser.add_argument("--preference-guidance-scale", type=float, default=8.0)
    parser.add_argument("--negative-prompt", default=(
        "low quality, blurry, low resolution, backlit, cartoon, animated, deformed, "
        "oversaturated, undersaturated, out of frame"
    ))
    parser.add_argument("--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--reward-cpu-offload", action="store_true",
                        help="Move Personal CLIP to GPU only while scoring a prune node; avoids contention with SD3 offload.")
    parser.add_argument("--save-prune-candidates", action="store_true",
                        help="Save all x0 images at the chosen method's reward nodes. Disabled by default to limit disk use.")
    parser.add_argument("--resume-from", default="auto", help="auto, none, or a non-negative task index")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Validate all requested task identities, generation orders and seed pools, then exit before loading models.")
    return parser.parse_args()


def parse_resume(value: str) -> int | None:
    text = value.strip().lower()
    if text == "none":
        return None
    if text in {"", "auto"}:
        return 0
    try:
        result = int(text)
    except ValueError as exc:
        raise ValueError("--resume-from must be auto, none, or a non-negative task index") from exc
    if result < 0:
        raise ValueError("--resume-from must be non-negative")
    return result


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def task_dir(root: Path, task: int, uid: int, oid: int) -> Path:
    return root / f"task{task:04d}_{uid}_{oid}"


def task_complete(path: Path, expected_item_count: int) -> bool:
    summary = path / "task_summary.json"
    if not summary.is_file():
        return False
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
        items = data["items"]
        return len(items) == expected_item_count and all(Path(item["selected_image_path"]).is_file() for item in items)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


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


def load_personal_clip(weights: Path, device: torch.device):
    import open_clip

    if not weights.is_file():
        raise FileNotFoundError(f"missing frozen Personal OpenCLIP weights: {weights}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    if weights.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weights), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Personal CLIP checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    else:
        open_clip.load_checkpoint(model, str(weights), strict=False)
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    return model, preprocess


class PersonalReward:
    """Frozen current-item Personal Sim; no GT, outfit, or future-item signal."""

    def __init__(self, history: dict, clip, preprocess, device: torch.device, cpu_offload: bool):
        self.history = history
        self.clip = clip
        self.preprocess = preprocess
        self.device = device
        self.cpu_offload = cpu_offload
        if self.cpu_offload and self.device.type == "cuda":
            self.clip.to("cpu")
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def score(self, decoded_images: dict[int, torch.Tensor], uid: int, category: int) -> dict[int, float]:
        seeds = sorted(decoded_images)
        # decode_latent returns an RGB tensor in [-1, 1].  Convert through the
        # frozen OpenCLIP preprocessing path used by the existing evaluator.
        if self.cpu_offload and self.device.type == "cuda":
            self.clip.to(self.device)
        try:
            pixels = torch.stack([
                self.preprocess(to_pil_image(((decoded_images[seed] + 1.0) / 2.0).clamp(0, 1).float().cpu()))
                for seed in seeds
            ]).to(self.device, dtype=next(self.clip.parameters()).dtype)
            image_features = F.normalize(self.clip.encode_image(pixels).float(), dim=-1)
            user_history = self.history[int(uid)]
            target = user_history.get(int(category), user_history.get(str(int(category))))
            if target is None:
                raise KeyError(f"missing Personal history vector for uid={uid}, category={category}")
            history = F.normalize(torch.as_tensor(target, dtype=torch.float32, device=self.device), dim=0)
            values = 100.0 * (image_features @ history)
            return {seed: float(values[index].detach().cpu()) for index, seed in enumerate(seeds)}
        finally:
            if self.cpu_offload and self.device.type == "cuda":
                self.clip.to("cpu")
                torch.cuda.empty_cache()


def fixed_order(order_root: Path, task: int, uid: int, oid: int, item_ids: list[int]) -> list[int]:
    metadata_path = task_dir(order_root, task, uid, oid) / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"BoN standard order metadata is unavailable: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trajectories = metadata.get("candidate_trajectories", [])
    if not trajectories:
        raise ValueError(f"no reference trajectory in {metadata_path}")
    order = [int(value) for value in trajectories[0]["generation_order_original_positions"]]
    if sorted(order) != list(range(len(item_ids))):
        raise ValueError(f"invalid BoN standard order for task {task}: {order}")
    return order


def parse_fk_steps(value: str, num_steps: int) -> tuple[int, ...]:
    steps = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if not steps or any(step < 1 or step >= num_steps for step in steps):
        raise ValueError("--fk-resample-steps must be unique intermediate nodes in [1, num_inference_steps - 1]")
    return steps


def method_spec(method: str, fk_steps: tuple[int, ...], num_steps: int) -> tuple[int, list[tuple[int, int]]]:
    if method == "bon":
        return 4, [(28, 1)]
    if method == "psp":
        return 8, [(7, 4), (14, 2), (28, 1)]
    if method == "dsp":
        if num_steps != SIX_SEED_RULE["horizon"]:
            raise ValueError("The frozen six-seed 7/21 policy requires exactly 28 steps")
        return SIX_SEED_RULE["initial_particles"], [tuple(node) for node in SIX_SEED_RULE["schedule"]]
    return 4, [(step, 4) for step in fk_steps] + [(num_steps, 1)]


def particle_seeds(args, task: int, generation_position: int, particle_count: int) -> list[int]:
    base = args.seed + task * args.seed_stride + generation_position * args.item_seed_stride
    return [base + slot for slot in range(particle_count)]


def predict_velocity(pipe, latent: torch.Tensor, timestep: torch.Tensor, prompt: dict, context_offset: torch.Tensor, args):
    base = context_offset.expand(1, *context_offset.shape[1:]).contiguous()
    zero = torch.zeros_like(base)
    model_input = torch.cat([latent, latent, latent, latent], dim=0)
    model_input = model_input + MATCHING_LAMBDA * torch.cat([zero, zero, base, base], dim=0)
    prompt_embeds = torch.cat([prompt["uncond"], prompt["text"], prompt["text"], prompt["preference"]], dim=0)
    with torch.no_grad(), torch.autocast(
        device_type=latent.device.type, enabled=latent.device.type == "cuda", dtype=pipe.transformer.dtype
    ):
        prediction = pipe.transformer(
            hidden_states=model_input,
            timestep=timestep.expand(4),
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


def save_candidate_image(image: torch.Tensor, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(((image + 1.0) / 2.0).clamp(0, 1), str(path))
    return str(path.resolve())


def generate_and_select_item(
    *, pipe, scheduler, prompt: dict, context_latents: torch.Tensor, scorer: PersonalReward,
    uid: int, category: int, item_id: int, task: int, generation_position: int,
    seeds: list[int], schedule: list[tuple[int, int]], args, output_dir: Path,
) -> tuple[torch.Tensor, dict]:
    device = context_latents.device
    latent_channels = pipe.vae.config.latent_channels
    context_offset = compute_stage2_context_offset(pipe.transformer, context_latents)
    states: dict[int, torch.Tensor] = {}
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        states[seed] = pipe.prepare_latents(
            batch_size=1, num_channels_latents=latent_channels, height=args.resolution, width=args.resolution,
            dtype=pipe.transformer.dtype, device=device, generator=generator,
        )
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    nodes = {step: keep for step, keep in schedule}
    trace = []
    selected_image_path = None
    selected_seed = None
    final_decoded: dict[int, torch.Tensor] = {}
    forward_calls = 0
    for step, timestep in enumerate(scheduler.timesteps, start=1):
        decoded: dict[int, torch.Tensor] = {}
        for seed in list(states):
            latent = states[seed]
            velocity = predict_velocity(pipe, latent, timestep, prompt, context_offset, args)
            x0_prediction = estimate_x0_from_flowmatch(scheduler, velocity, latent, step - 1)
            scheduler._step_index = step - 1
            states[seed] = scheduler.step(velocity, timestep, latent, return_dict=False)[0].detach()
            forward_calls += 1
            if step in nodes:
                decoded[seed] = decode_latent(pipe, x0_prediction)[0].detach()
            del velocity, x0_prediction
        if step not in nodes:
            continue
        rewards = scorer.score(decoded, uid, category)
        active_before = sorted(states)
        ranked = sorted(active_before, key=lambda seed: (-rewards[seed], seed))
        survivors = ranked[:nodes[step]]
        candidate_paths = {}
        if args.save_prune_candidates:
            for seed in active_before:
                candidate_paths[str(seed)] = save_candidate_image(
                    decoded[seed], output_dir / "prune_candidates" / f"item_{generation_position:02d}_{item_id}" /
                    f"step_{step:02d}" / f"seed_{seed}_x0.png"
                )
        trace.append({
            "step": step, "active_before": active_before, "active_after": survivors,
            "personal_scores": {str(seed): rewards[seed] for seed in active_before},
            "candidate_image_paths": candidate_paths,
            "particle_steps_spent": forward_calls,
            "trigger_reason": "terminal_selection" if step == args.num_inference_steps else "fixed_current_personal_topk",
        })
        if step == args.num_inference_steps:
            selected_seed = survivors[0]
            final_decoded = decoded
        states = {seed: states[seed] for seed in survivors}
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if selected_seed is None or selected_seed not in states:
        raise RuntimeError("final greedy selection failed")
    selected_image_path = save_candidate_image(
        final_decoded[selected_seed], output_dir / "selected" / f"item_{generation_position:02d}_{item_id}.png"
    )
    reward_views = sum(len(node["active_before"]) for node in trace)
    if args.method == "dsp":
        if forward_calls != SIX_SEED_RULE["particle_steps"] or reward_views != 12:
            raise RuntimeError(f"Six-seed budget mismatch: {forward_calls} denoiser calls, {reward_views} rewards")
    record = {
        "generation_position": generation_position, "original_outfit_position": None,
        "item_id": item_id, "category_id": category, "candidate_seeds": seeds,
        "selected_seed": selected_seed, "selected_personal_reward": trace[-1]["personal_scores"][str(selected_seed)],
        "selected_image_path": selected_image_path, "trace": trace, "transformer_forward_calls": forward_calls,
        "particle_steps": forward_calls, "reward_views": reward_views,
        "vae_decode_calls": reward_views, "reward_score_events": len(trace),
    }
    return states[selected_seed].detach(), record


def generate_and_select_item_fk(
    *, pipe, scheduler, prompt: dict, context_latents: torch.Tensor, scorer: PersonalReward,
    uid: int, category: int, item_id: int, task: int, generation_position: int,
    seeds: list[int], fk_steps: tuple[int, ...], args, output_dir: Path,
) -> tuple[torch.Tensor, dict]:
    """Run one GOR item with online FK-Steering over a constant four-particle pool.

    A selected terminal latent becomes the context for the next item.  Intermediate
    resampling only affects the current item, so no trajectory can leak future
    garment information into an earlier decision.
    """
    device = context_latents.device
    latent_channels = pipe.vae.config.latent_channels
    context_offset = compute_stage2_context_offset(pipe.transformer, context_latents)
    active = []
    for slot, seed in enumerate(seeds):
        generator = torch.Generator(device=device).manual_seed(seed)
        active.append({
            "slot": slot,
            "seed": seed,
            "root_seed": seed,
            "lineage": [seed],
            "previous_reward": 0.0,
            "latent": pipe.prepare_latents(
                batch_size=1, num_channels_latents=latent_channels, height=args.resolution, width=args.resolution,
                dtype=pipe.transformer.dtype, device=device, generator=generator,
            ),
        })

    scheduler.set_timesteps(args.num_inference_steps, device=device)
    resample_rng = random.Random(
        args.fk_resampling_seed + task * args.seed_stride + generation_position * args.item_seed_stride
    )
    trace = []
    forward_calls = 0
    terminal_decoded: list[torch.Tensor] = []
    terminal_scores: list[float] = []

    for step, timestep in enumerate(scheduler.timesteps, start=1):
        decoded: dict[int, torch.Tensor] = {}
        for particle in active:
            latent = particle["latent"]
            velocity = predict_velocity(pipe, latent, timestep, prompt, context_offset, args)
            x0_prediction = estimate_x0_from_flowmatch(scheduler, velocity, latent, step - 1)
            scheduler._step_index = step - 1
            after = scheduler.step(velocity, timestep, latent, return_dict=False)[0].detach()

            # The native Flow Matching ODE is deterministic.  This controlled
            # stochastic Euler perturbation is what gives duplicated FK parents
            # distinct valid continuations after a resampling node.
            if step < args.num_inference_steps:
                sigma = float(scheduler.sigmas[step - 1].detach().float().cpu().item())
                noise_scale = sigma * float(np.sqrt((1.0 + args.fk_gamma) ** 2 - 1.0))
                noise_seed = (
                    args.fk_stochastic_seed + task * 1_000_000 + generation_position * 100_000
                    + step * 1_000 + int(particle["slot"])
                )
                noise = torch.randn(
                    after.shape,
                    generator=torch.Generator(device=after.device).manual_seed(noise_seed),
                    device=after.device,
                    dtype=after.dtype,
                )
                particle["latent"] = after + noise_scale * noise
                particle.setdefault("controlled_noise_history", []).append({
                    "step": step, "noise_seed": noise_seed, "sigma": sigma, "noise_scale": noise_scale,
                })
            else:
                # The selected terminal image and latent must agree before the
                # latent is appended to the next greedy GOR context.
                particle["latent"] = after
            forward_calls += 1
            if step in fk_steps or step == args.num_inference_steps:
                decoded[int(particle["slot"])] = decode_latent(pipe, x0_prediction)[0].detach()
            del velocity, x0_prediction

        if step not in fk_steps and step != args.num_inference_steps:
            continue

        scores_by_slot = scorer.score(decoded, uid, category)
        candidate_paths = {}
        if args.save_prune_candidates:
            for particle in active:
                slot = int(particle["slot"])
                candidate_paths[str(slot)] = save_candidate_image(
                    decoded[slot], output_dir / "prune_candidates" / f"item_{generation_position:02d}_{item_id}" /
                    f"step_{step:02d}" / f"slot_{slot}_root_{particle['root_seed']}_x0.png"
                )

        if step == args.num_inference_steps:
            winner = min(
                range(len(active)),
                key=lambda index: (-scores_by_slot[int(active[index]["slot"])], int(active[index]["root_seed"]), index),
            )
            terminal_decoded = [decoded[int(particle["slot"])] for particle in active]
            terminal_scores = [scores_by_slot[int(particle["slot"])] for particle in active]
            trace.append({
                "step": step, "node_kind": "terminal_select", "active_before": [int(p["slot"]) for p in active],
                "winner_slot": int(active[winner]["slot"]),
                "candidate_scores": [
                    {"slot": int(p["slot"]), "seed": int(p["seed"]), "root_seed": int(p["root_seed"]),
                     "personal_reward": scores_by_slot[int(p["slot"])]}
                    for p in active
                ],
                "candidate_image_paths": candidate_paths,
            })
            selected = active[winner]
            selected_image_path = save_candidate_image(
                terminal_decoded[winner], output_dir / "selected" / f"item_{generation_position:02d}_{item_id}.png"
            )
            record = {
                "generation_position": generation_position, "original_outfit_position": None,
                "item_id": item_id, "category_id": category, "candidate_seeds": seeds,
                "selected_slot": int(selected["slot"]), "selected_seed": int(selected["seed"]),
                "selected_root_seed": int(selected["root_seed"]), "selected_personal_reward": terminal_scores[winner],
                "selected_image_path": selected_image_path, "trace": trace,
                "terminal_particles": [
                    {"slot": int(p["slot"]), "seed": int(p["seed"]), "root_seed": int(p["root_seed"]),
                     "personal_reward": terminal_scores[index], "controlled_noise_history": p["controlled_noise_history"]}
                    for index, p in enumerate(active)
                ],
                "transformer_forward_calls": forward_calls,
            }
            return selected["latent"].detach(), record

        rewards = [scores_by_slot[int(particle["slot"])] for particle in active]
        previous_rewards = [float(particle["previous_reward"]) for particle in active]
        potentials, weights, ess = importance_weights(
            rewards, previous_rewards, potential_type=args.fk_potential, lmbda=args.fk_lambda,
        )
        threshold = args.fk_ess_fraction * len(active)
        should_resample = (not args.fk_adaptive_resampling) or ess < threshold
        parents = (
            resample_parent_indices(weights, count=len(active), method=args.fk_resampling, rng=resample_rng)
            if should_resample else tuple(range(len(active)))
        )
        trace_node = {
            "step": step, "node_kind": "intermediate_resample", "active_before": [int(p["slot"]) for p in active],
            "potential_type": args.fk_potential, "lambda": args.fk_lambda, "ess": float(ess),
            "ess_threshold": float(threshold), "resampled": bool(should_resample), "parents": list(parents),
            "candidate_scores": [
                {"slot": int(p["slot"]), "seed": int(p["seed"]), "root_seed": int(p["root_seed"]),
                 "personal_reward": rewards[index], "previous_reward": previous_rewards[index],
                 "potential": float(potentials[index]), "normalized_weight": float(weights[index])}
                for index, p in enumerate(active)
            ],
            "candidate_image_paths": candidate_paths,
        }
        trace.append(trace_node)
        for particle, reward in zip(active, rewards):
            particle["previous_reward"] = reward
        if should_resample:
            parents_active = active
            active = [
                {
                    "slot": child_slot,
                    "seed": int(parents_active[parent_slot]["seed"]),
                    "root_seed": int(parents_active[parent_slot]["root_seed"]),
                    "lineage": list(parents_active[parent_slot]["lineage"]) + [int(parents_active[parent_slot]["root_seed"])],
                    "previous_reward": float(parents_active[parent_slot]["previous_reward"]),
                    "latent": parents_active[parent_slot]["latent"].clone(),
                    "controlled_noise_history": list(parents_active[parent_slot]["controlled_noise_history"]),
                }
                for child_slot, parent_slot in enumerate(parents)
            ]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raise RuntimeError("FK greedy item generation reached no terminal selection")


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.fk_gamma <= float(np.sqrt(2.0) - 1.0):
        raise ValueError("--fk-gamma must be in [0, sqrt(2)-1]")
    fk_steps = parse_fk_steps(args.fk_resample_steps, args.num_inference_steps)
    particle_count, schedule = method_spec(args.method, fk_steps, args.num_inference_steps)
    if args.max_samples < 1 or args.item_seed_stride < particle_count:
        raise ValueError("--max-samples must be positive and --item-seed-stride must fit every particle")
    if args.seed_stride < 4 * args.item_seed_stride:
        raise ValueError("--seed-stride must reserve four non-overlapping item seed pools")
    resume = parse_resume(args.resume_from)
    if resume is not None and resume >= args.max_samples:
        raise ValueError("--resume-from must be smaller than --max-samples")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    ground_truth = load_numpy_dict(str(args.data_root / "test_grd_dict.npy"))
    id_to_category = load_numpy_dict(str(args.data_root / "id_cate_dict.npy"))
    preferences = load_numpy_dict(str(args.processed_info_root / "stage2_ifashion_test" / "test_preference.npy"))
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    total = min(args.max_samples, len(ground_truth["uids"]))
    model_path = args.models_root / args.model_name
    if args.method == "dsp" and (args.seed_stride != 10000 or args.item_seed_stride != 64):
        raise ValueError("Keep existing GOR task/item seed mapping: stride10000, item stride64")
    if args.prepare_only:
        for task in range(total):
            uid, oid = int(ground_truth["uids"][task]), int(ground_truth["oids"][task])
            items = [int(value) for value in ground_truth["outfits"][task]]
            order = fixed_order(args.order_reference_root, task, uid, oid, items)
            pools = [particle_seeds(args, task, position, particle_count) for position in range(len(order))]
            if len({seed for pool in pools for seed in pool}) != len(order)*particle_count:
                raise ValueError(f"Overlapping GOR item seed pools at task {task}")
            for category in ground_truth["category"][task]:
                user_history = history[uid]
                if int(category) not in user_history and str(int(category)) not in user_history:
                    raise KeyError(f"Missing Personal history: uid={uid}, category={category}")
        print(json.dumps({"prepare_only": True, "method": args.method, "validated_tasks": total,
                          "particle_count_per_item": particle_count, "schedule": schedule,
                          "denoiser_calls_per_item": sum((step-previous)*active for (previous,active),(step,_) in
                                                        zip([(0,particle_count),*schedule[:-1]],schedule)),
                          "reward_views_per_item": particle_count+sum(keep for _,keep in schedule[:-1]),
                          "seed_mapping": "base_seed + task_index*10000 + generation_position*64 + slot",
                          "output_root": str(args.output_root.resolve())}, indent=2))
        return

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
    clip, preprocess = load_personal_clip(args.personal_clip_weights, device)
    scorer = PersonalReward(history, clip, preprocess, device, args.reward_cpu_offload)

    latent_channels = pipe.vae.config.latent_channels
    latent_height, latent_width = args.resolution // pipe.vae_scale_factor, args.resolution // pipe.vae_scale_factor
    zero_context = torch.zeros(1, 1, latent_channels, latent_height, latent_width,
                               device=device, dtype=pipe.transformer.dtype)
    progress = tqdm(range(total), desc=f"{args.model_name} | greedy-{args.method}", unit="task")
    for task in progress:
        if resume is not None and task < resume:
            progress.set_postfix_str("resume skip")
            continue
        uid, oid = int(ground_truth["uids"][task]), int(ground_truth["oids"][task])
        item_ids = [int(value) for value in ground_truth["outfits"][task]]
        categories = [int(value) for value in ground_truth["category"][task]]
        output_dir = task_dir(args.output_root, task, uid, oid)
        if resume is not None and task_complete(output_dir, len(item_ids)):
            if args.method == "dsp":
                cached = json.loads((output_dir / "task_summary.json").read_text(encoding="utf-8"))
                if (cached.get("method") != "dsp" or cached.get("base_seed") != args.seed
                        or cached.get("schedule") != [list(node) for node in schedule]
                        or any(item.get("candidate_seeds") != particle_seeds(args,task,position,particle_count)
                               or item.get("particle_steps") != 112 or item.get("reward_views") != 12
                               for position,item in enumerate(cached["items"]))):
                    raise ValueError(f"Resume configuration mismatch in {output_dir}; select a separate output root")
            progress.set_postfix_str("cached")
            continue
        started = time.perf_counter()
        order = fixed_order(args.order_reference_root, task, uid, oid, item_ids)
        prompts = [make_prompt_config(pipe, uid, category, id_to_category, preferences, args, device) for category in categories]
        selected_latents: list[torch.Tensor] = []
        records = []
        for generation_position, original_position in enumerate(order):
            context = zero_context if not selected_latents else torch.stack(selected_latents, dim=1)
            common_kwargs = {
                "pipe": pipe, "scheduler": pipe.scheduler, "prompt": prompts[original_position],
                "context_latents": context, "scorer": scorer, "uid": uid,
                "category": categories[original_position], "item_id": item_ids[original_position],
                "task": task, "generation_position": generation_position,
                "seeds": particle_seeds(args, task, generation_position, particle_count),
                "args": args, "output_dir": output_dir,
            }
            if args.method == "fk":
                latent, record = generate_and_select_item_fk(**common_kwargs, fk_steps=fk_steps)
            else:
                latent, record = generate_and_select_item(**common_kwargs, schedule=schedule)
            record["original_outfit_position"] = original_position
            records.append(record)
            selected_latents.append(latent)
            if len(selected_latents) > 3:
                selected_latents = selected_latents[-3:]
        atomic_json(output_dir / "task_summary.json", {
            "task_index": task, "uid": uid, "oid": oid, "method": args.method,
            "selection_unit": "current item; selected latent is appended to the next item context",
            "reward": "100*cosine(OpenCLIP ViT-H/14(current x0), user test-history(uid, current category))",
            "order_source": str((task_dir(args.order_reference_root, task, uid, oid) / "metadata.json").resolve()),
            "fixed_generation_order_original_positions": order,
            "schedule": schedule, "particle_count_per_item": particle_count,
            "base_seed": args.seed, "seed_stride": args.seed_stride, "item_seed_stride": args.item_seed_stride,
            "frozen_rule_version": SIX_SEED_RULE["version"] if args.method == "dsp" else None,
            "total_particle_steps": sum(record["transformer_forward_calls"] for record in records),
            "total_reward_views": sum(record.get("reward_views", sum(len(node.get("candidate_scores", []))
                                      for node in record["trace"])) for record in records),
            "fk_config": ({
                "resampling_steps": list(fk_steps), "potential_type": args.fk_potential,
                "lambda": args.fk_lambda, "resampling": args.fk_resampling,
                "adaptive_resampling": bool(args.fk_adaptive_resampling),
                "ess_fraction": args.fk_ess_fraction,
                "resampling_seed_base": args.fk_resampling_seed,
                "stochastic_seed_base": args.fk_stochastic_seed,
                "controlled_stochastic": True, "controlled_stochastic_gamma": args.fk_gamma,
                "controlled_stochastic_rule": "x_next=x_det+sigma_i*sqrt((1+gamma)^2-1)*Normal(0,I) after each nonterminal Euler step",
            } if args.method == "fk" else None),
            "denoiser_calls_per_item": sum((step - previous) * active for (previous, active), (step, _) in zip([(0, particle_count), *schedule[:-1]], schedule)),
            "items": records, "elapsed_s": time.perf_counter() - started,
        })
        progress.set_postfix_str(f"task={task} elapsed={time.perf_counter() - started:.0f}s")
    del clip, pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
