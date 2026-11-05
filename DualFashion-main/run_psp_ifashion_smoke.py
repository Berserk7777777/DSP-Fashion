"""Online PSP smoke runner for the DualFashion iFashion image branch.

This runner intentionally keeps the transformer/VAE batch size at one.  It is
therefore suitable for the current single-GPU setup while preserving the PSP
semantics: all particles are initialized from a fixed task-candidate seed
block, intermediate x0 predictions are scored by a selected frozen reward, and only the
requested survivors continue denoising.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

from eval.compatibility_evaluator.compatibility_net import FashionEvaluator
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
from sd3_modules.stage2_pipeline import DualDiffSD3Pipeline

# The shared FK policy is dependency-light and lives beside the backbone
# repositories.  Keep model/scheduler code local to DualFashion.
FASHION_PSP_ROOT = Path(__file__).resolve().parents[1]
if str(FASHION_PSP_ROOT) not in sys.path:
    sys.path.insert(0, str(FASHION_PSP_ROOT))
from fashion_psp.baselines.fk_steering import (  # noqa: E402
    FKSteeringConfig,
    importance_weights,
    resample_parent_indices,
)


def parse_schedule(value: str, num_steps: int, num_particles: int) -> list[tuple[int, int]]:
    """Parse ``step:keep,step:keep`` and always finalize to one particle."""
    entries: list[tuple[int, int]] = []
    if value.strip():
        for raw in value.split(","):
            step_text, keep_text = raw.split(":", 1)
            step, keep = int(step_text), int(keep_text)
            if not 1 <= step <= num_steps:
                raise ValueError(f"prune step {step} must be in [1, {num_steps}]")
            if not 1 <= keep <= num_particles:
                raise ValueError(f"keep count {keep} must be in [1, {num_particles}]")
            entries.append((step, keep))
    if not entries or entries[-1][0] != num_steps:
        entries.append((num_steps, 1))
    entries.sort()
    if any(keep > num_particles for _, keep in entries):
        raise ValueError("keep count cannot exceed the initial particle count")
    return entries


def parse_step_list(value: str, num_steps: int) -> list[int]:
    """Parse a comma-separated list of one-based denoising steps."""
    if not value.strip():
        return []
    steps = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not steps:
        return []
    if any(step < 1 or step > num_steps for step in steps):
        raise ValueError(f"cache steps must be in [1, {num_steps}]")
    return steps


def parse_fk_steps(value: str, num_steps: int) -> tuple[int, ...]:
    """FK resampling nodes; terminal selection is always handled separately."""
    steps = tuple(parse_step_list(value, num_steps))
    if not steps:
        raise ValueError("--fk-resample-steps must contain at least one intermediate step")
    if steps[-1] == num_steps:
        raise ValueError("final FK selection is deterministic; do not include --num-inference-steps")
    return steps


def estimate_x0_from_flowmatch(scheduler, velocity, sample, sigma_idx: int):
    """Predict x0 directly from the guided flow velocity before stepping.

    This remains valid when a controlled stochastic wrapper perturbs the
    post-Euler latent.  Inferring velocity from ``after - before`` would fold
    that random increment into the reward prediction.
    """
    sigma = scheduler.sigmas[sigma_idx].to(device=sample.device, dtype=sample.dtype)
    while sigma.ndim < sample.ndim:
        sigma = sigma.view(*sigma.shape, 1)
    return (sample - sigma * velocity).detach()


class CompatibilityReward:
    def __init__(self, data_root: Path, evaluator_ckpt: Path, device: torch.device, clip_weights: Path | None = None):
        self.device = device
        self.item_features = np.load(data_root / "cnn_features_clip.npy", mmap_mode="r")
        import open_clip

        if clip_weights is not None and clip_weights.suffix.lower() == ".safetensors":
            # open_clip 3.2 calls torch.load even for a safetensors path, which
            # fails with PyTorch >=2.6 (and is unnecessary for a trusted local
            # cache).  Instantiate the known architecture and load safely.
            from safetensors.torch import load_file

            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
            state = load_file(str(clip_weights), device="cpu")
            missing, unexpected = self.clip_model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"OpenCLIP local checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        else:
            pretrained = str(clip_weights) if clip_weights is not None else "laion2b-s32b-b79K"
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=pretrained)
        self.clip_model = self.clip_model.to(device).eval()
        self.evaluator = FashionEvaluator(cnn_feat_dim=1024)
        state = torch.load(evaluator_ckpt, map_location="cpu")
        self.evaluator.load_state_dict(state)
        self.evaluator = self.evaluator.to(device).eval()

    @torch.inference_mode()
    def score(self, image_tensor: torch.Tensor, outfit: list[int], blank_idx: int) -> list[float]:
        context = torch.from_numpy(np.asarray([self.item_features[item_id] for item_id in outfit])).float()
        scores: list[float] = []
        # Keep reward inference serial as well: ViT-H/14 is large enough that a
        # reward batch of four can exceed the available GPU memory.
        for image in image_tensor.detach().float().cpu():
            image = ((image.clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
            pil_image = Image.fromarray(image, mode="RGB")
            batch = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            candidate_feature = self.clip_model.encode_image(batch).float().cpu()
            outfits = context.unsqueeze(0)
            outfits[:, blank_idx, :] = candidate_feature
            logits = self.evaluator(outfits.to(self.device))
            scores.append(float(torch.sigmoid(logits).item()))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return scores


class PersonalReward:
    """OpenCLIP cosine similarity to a user's category-specific history."""

    def __init__(
        self,
        history_path: Path,
        uid: int,
        target_category: int,
        device: torch.device,
        clip_weights: Path | None = None,
    ):
        self.device = device
        history = np.load(history_path, allow_pickle=True).item()
        try:
            embedding = history[uid][target_category]
        except KeyError as exc:
            raise KeyError(
                f"Missing test history embedding for uid={uid}, category={target_category} in {history_path}"
            ) from exc
        self.history_embedding = F.normalize(
            torch.as_tensor(embedding, dtype=torch.float32, device=device), dim=-1
        )

        import open_clip

        if clip_weights is not None and clip_weights.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file

            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", pretrained=None
            )
            state = load_file(str(clip_weights), device="cpu")
            missing, unexpected = self.clip_model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"OpenCLIP local checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
                )
        else:
            pretrained = str(clip_weights) if clip_weights is not None else "laion2b-s32b-b79K"
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14", pretrained=pretrained
            )
        self.clip_model = self.clip_model.to(device).eval()

    @torch.inference_mode()
    def score(self, image_tensor: torch.Tensor, outfit: list[int], blank_idx: int) -> list[float]:
        # Keep the signature shared with CompatibilityReward so PSP/FK logic
        # does not need reward-specific branches.
        scores: list[float] = []
        for image in image_tensor.detach().float().cpu():
            image = ((image.clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
            batch = self.preprocess(Image.fromarray(image, mode="RGB")).unsqueeze(0).to(self.device)
            candidate = F.normalize(self.clip_model.encode_image(batch).float(), dim=-1)
            scores.append(float((100.0 * (candidate @ self.history_embedding.unsqueeze(1)).squeeze()).item()))
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--model-name", default="0057501")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\tmp\ifashion"))
    parser.add_argument("--fitb-file", default="processed/fitb_test.npy")
    parser.add_argument("--processed-info-root", type=Path, default=Path("processed_info"))
    parser.add_argument("--image-root", type=Path, default=Path(r"C:\tmp\ifashion\semantic_category"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument(
        "--search-method", choices=("psp", "fk"), default="psp",
        help="PSP hard-prunes particles; FK keeps a constant population and resamples parents.",
    )
    parser.add_argument("--num-particles", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--prune-schedule", default="14:2", help="Comma-separated step:keep pairs; final step keeps one.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--seed-stride", type=int, default=10000)
    parser.add_argument(
        "--fk-resample-steps", default="6,11,17,22",
        help=(
            "Comma-separated intermediate one-based nodes for FK-Steering; ignored by PSP. "
            "For the default 28-step Flow Matching schedule these are the nearest "
            "integer nodes to 20/40/60/80 percent."
        ),
    )
    parser.add_argument("--fk-potential", choices=("diff", "max", "add", "rt"), default="diff")
    parser.add_argument("--fk-lambda", type=float, default=20.0)
    parser.add_argument("--fk-resampling", choices=("multinomial", "systematic"), default="systematic")
    parser.add_argument(
        "--fk-adaptive-resampling", action="store_true",
        help="Resample only when ESS falls below --fk-ess-fraction times K.",
    )
    parser.add_argument("--fk-ess-fraction", type=float, default=0.5)
    parser.add_argument(
        "--fk-resampling-seed", type=int, default=9_090,
        help="Independent RNG base for FK parent sampling; task index is added with --seed-stride.",
    )
    parser.add_argument(
        "--fk-controlled-stochastic", action="store_true",
        help=(
            "Use the PSP paper's controlled stochastic Euler wrapper at every FK step. "
            "Required for meaningful FK branching with DualFashion's deterministic flow-matching scheduler."
        ),
    )
    parser.add_argument(
        "--fk-gamma", type=float, default=0.005,
        help="Controlled-flow stochasticity gamma; must be in [0, sqrt(2)-1]. Tune on validation data.",
    )
    parser.add_argument(
        "--fk-stochastic-seed", type=int, default=8_191,
        help="Independent base seed for reproducible particle-step noise fields.",
    )
    parser.add_argument(
        "--fk-stochastic-rejuvenation", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--text-guidance-scale", type=float, default=8.0)
    parser.add_argument("--context-guidance-scale", type=float, default=7.0)
    parser.add_argument("--preference-guidance-scale", type=float, default=8.0)
    parser.add_argument("--negative-prompt", default="low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reward-device", default=None)
    parser.add_argument(
        "--reward", choices=("compatibility", "personal"), default="compatibility",
        help="Online score used by PSP/FK pruning and final selection.",
    )
    parser.add_argument(
        "--history-path", type=Path, default=None,
        help="test_history_clipembs.npy; defaults to <data-root>/processed/test_history_clipembs.npy for personal reward.",
    )
    parser.add_argument("--evaluator-ckpt", type=Path, default=Path("eval/compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth"))
    parser.add_argument(
        "--clip-weights",
        type=Path,
        default=None,
        help="Local OpenCLIP ViT-H/14 weights. If omitted, discover the first local HF cache snapshot.",
    )
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--deterministic-context", action="store_true")
    parser.add_argument(
        "--audit-all-candidates",
        action="store_true",
        help="After PSP, fully denoise and score every original seed for inspection.",
    )
    parser.add_argument(
        "--save-intermediate-predictions",
        action="store_true",
        help="Save x_prediction latent tensors, latent PNG projections, and decoded reward PNGs at each prune step.",
    )
    parser.add_argument(
        "--offline-cache-steps",
        default="",
        help=(
            "Comma-separated denoising steps at which to decode every original particle's x0 prediction. "
            "When provided, disables online reward scoring/pruning and saves an offline candidate cache."
        ),
    )
    return parser.parse_args()


def decode_latent(pipe: DualDiffSD3Pipeline, latent: torch.Tensor) -> torch.Tensor:
    """Decode one latent and return a CPU tensor in [-1, 1]."""
    with torch.no_grad(), torch.autocast(
        device_type=latent.device.type,
        enabled=latent.device.type == "cuda",
        dtype=pipe.transformer.dtype,
    ):
        scaled = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(scaled, return_dict=False)[0]
    return image.float().clamp(-1, 1).cpu()


def save_latent_visualization(latent: torch.Tensor, path: Path) -> None:
    """Save a diagnostic RGB projection of a latent tensor as PNG."""
    value = latent.detach().float().cpu()[0]
    if value.shape[0] < 3:
        value = value.repeat(3, 1, 1)[:3]
    else:
        value = value[:3]
    value = value - value.amin(dim=(1, 2), keepdim=True)
    value = value / value.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(value, str(path))


def main() -> None:
    args = parse_args()
    if args.num_particles < 1:
        raise ValueError("--num-particles must be positive")
    if not 0.0 <= args.fk_gamma <= float(np.sqrt(2.0) - 1.0):
        raise ValueError("--fk-gamma must be in [0, sqrt(2)-1]")
    if args.fk_stochastic_rejuvenation:
        raise ValueError(
            "--fk-stochastic-rejuvenation was an invalid post-resample heuristic and has been retired. "
            "Use --fk-controlled-stochastic with --fk-gamma instead."
        )
    schedule = parse_schedule(args.prune_schedule, args.num_inference_steps, args.num_particles)
    fk_config = None
    if args.search_method == "fk":
        fk_config = FKSteeringConfig(
            num_particles=args.num_particles,
            resampling_steps=parse_fk_steps(args.fk_resample_steps, args.num_inference_steps),
            potential_type=args.fk_potential,
            lmbda=args.fk_lambda,
            resampling=args.fk_resampling,
            adaptive_resampling=args.fk_adaptive_resampling,
            ess_fraction=args.fk_ess_fraction,
        )
        fk_config.validate(args.num_inference_steps)
    offline_cache_steps = parse_step_list(args.offline_cache_steps, args.num_inference_steps)
    offline_cache_mode = bool(offline_cache_steps)
    if offline_cache_mode and (args.audit_all_candidates or args.save_intermediate_predictions):
        raise ValueError(
            "--offline-cache-steps cannot be combined with --audit-all-candidates "
            "or --save-intermediate-predictions. The cache already stores every requested seed/step image."
        )
    if args.search_method == "fk" and offline_cache_mode:
        raise ValueError("FK-Steering is an online baseline; --offline-cache-steps is unsupported in FK mode")
    if args.search_method == "fk" and args.audit_all_candidates:
        raise ValueError("--audit-all-candidates is a PSP trajectory audit and is unsupported in FK mode")
    if args.fk_controlled_stochastic and args.search_method != "fk":
        raise ValueError("--fk-controlled-stochastic is only valid with --search-method fk")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    reward_device = torch.device(args.reward_device or str(device))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    data_root = args.data_root
    fitb = load_numpy_dict(str(data_root / args.fitb_file))
    grd = load_numpy_dict(str(data_root / "test_grd_dict.npy"))
    id_to_category = load_numpy_dict(str(data_root / "id_cate_dict.npy"))
    preferences = load_numpy_dict(str(args.processed_info_root / "stage2_ifashion_test" / "test_preference.npy"))
    image_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    if not 0 <= args.task_index < len(fitb["uids"]):
        raise IndexError(f"--task-index must be in [0, {len(fitb['uids']) - 1}]")

    model_path = args.models_root / args.model_name
    tokenizer_path = model_path / "tokenizer" / "tokenizer.json"
    from transformers import T5TokenizerFast

    tokenizer = T5TokenizerFast(tokenizer_file=str(tokenizer_path), extra_ids=100)
    pipe = DualDiffSD3Pipeline.from_pretrained(
        str(model_path), tokenizer=tokenizer,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
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

    uid = int(fitb["uids"][args.task_index])
    oid = int(fitb["oids"][args.task_index])
    incomplete = fitb["outfits"][args.task_index].tolist()
    full = grd["outfits"][args.task_index]
    categories = fitb["category"][args.task_index].tolist()
    blank_idx = incomplete.index(0)
    target_item_id = int(full[blank_idx])
    outfit = [int(item) for item in full]
    context_ids = [item for index, item in enumerate(outfit) if index != blank_idx]
    mask_keys = ["Color", "Material", "Design features", "Clothing Fashion Style"]
    preference_text = build_preference_text(preferences, uid, int(categories[blank_idx]), mask_keys)
    category_name = id_to_category.get(int(categories[blank_idx]), "item")
    sentence_text = f"Recommend a fashion {category_name} item, on white background."
    with torch.no_grad():
        preference_emb = encode_text(pipe.tokenizer, pipe.text_encoder, preference_text, args.max_length, device).to(device=device, dtype=pipe.transformer.dtype)
        sentence_emb = encode_text(pipe.tokenizer, pipe.text_encoder, sentence_text, args.max_length, device).to(device=device, dtype=pipe.transformer.dtype)
        negative_segment = encode_text(pipe.tokenizer, pipe.text_encoder, args.negative_prompt, args.max_length, device).to(device=device, dtype=pipe.transformer.dtype)
    text_base = torch.cat([sentence_emb], dim=1).contiguous()
    preference_base = torch.cat([preference_emb, sentence_emb], dim=1).contiguous()
    uncond_base = torch.cat([negative_segment, negative_segment], dim=1).contiguous()
    text_base = prepend_pad_segment(text_base, negative_segment, preference_base.shape[1])

    transform = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(512), T.ToTensor(), T.Normalize([0.5], [0.5])])
    context_images = torch.stack([load_image_tensor(item, image_paths, str(args.image_root), transform) for item in context_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=pipe.transformer.dtype):
        context_flat = context_images.reshape(-1, *context_images.shape[2:])
        context_dist = pipe.vae.encode(context_flat).latent_dist
        context_latents = context_dist.mode() if args.deterministic_context else context_dist.sample()
        context_latents = (context_latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    context_latents = context_latents.reshape(1, context_images.size(1), *context_latents.shape[1:]).to(device=device, dtype=pipe.transformer.dtype)
    context_offset = compute_stage2_context_offset(pipe.transformer, context_latents)
    latent_shape = tuple(context_latents.shape[2:])

    reward = None
    if not offline_cache_mode:
        clip_weights = args.clip_weights
        if clip_weights is None:
            cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            candidates = sorted(
                cache_root.glob("hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/snapshots/*/open_clip_model.safetensors")
            )
            clip_weights = candidates[0] if candidates else None
        elif not clip_weights.exists():
            clip_weights = None
        if clip_weights is None:
            raise FileNotFoundError(
                f"OpenCLIP weights not found at {args.clip_weights}. Pass --clip-weights with the local .bin/.safetensors file."
            )
        if args.reward == "compatibility":
            reward = CompatibilityReward(data_root, args.evaluator_ckpt, reward_device, clip_weights)
        else:
            history_path = args.history_path or (data_root / "processed" / "test_history_clipembs.npy")
            if not history_path.exists():
                raise FileNotFoundError(f"Personal reward history embeddings not found: {history_path}")
            reward = PersonalReward(
                history_path,
                uid,
                int(categories[blank_idx]),
                reward_device,
                clip_weights,
            )
    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = scheduler.timesteps.to(device)
    keep_by_step = dict(schedule)
    reported_schedule = (
        schedule
        if args.search_method == "psp"
        else [(step, args.num_particles) for step in fk_config.resampling_steps]
        + [(args.num_inference_steps, 1)]
    )
    candidate_seeds = [args.seed + args.task_index * args.seed_stride + i for i in range(args.num_particles)]
    active = [
        {
            "slot": index,
            "seed": seed,
            "root_seed": seed,
            "lineage": [seed],
            "previous_reward": 0.0,
            "latent": torch.randn(
                (1, *latent_shape),
                generator=torch.Generator(device=device).manual_seed(seed),
                device=device,
                dtype=pipe.transformer.dtype,
            ),
        }
        for index, seed in enumerate(candidate_seeds)
    ]
    initial_latent_hashes = {
        str(particle["seed"]): hashlib.sha256(
            particle["latent"].detach().float().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        for particle in active
    }
    trace: list[dict] = []
    transformer_forward_calls = 0
    vae_decode_calls = 0
    reward_score_events = 0
    reward_model_calls = 0
    fk_rng = random.Random(args.fk_resampling_seed + args.task_index * args.seed_stride)
    fk_score_rows: list[dict] = []
    fk_controlled_stochastic_events: list[dict] = []
    intermediate_prediction_rows: list[dict] = []
    offline_cache_rows: list[dict] = []
    offline_cache_root = args.output_dir / "offline_candidates" / f"{uid}_{oid}"
    if offline_cache_mode:
        offline_cache_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    text_scale = args.text_guidance_scale
    context_scale = args.context_guidance_scale
    preference_scale = args.preference_guidance_scale

    for step_idx, timestep in enumerate(timesteps, start=1):
        step_noise_rows: list[dict] = []
        for particle in active:
            latent = particle["latent"]
            latent_model_input = torch.cat([latent, latent, latent, latent], dim=0)
            context_base = context_offset.expand(1, *context_offset.shape[1:]).contiguous()
            context_zero = torch.zeros_like(context_base)
            latent_model_input = latent_model_input + MATCHING_LAMBDA * torch.cat([context_zero, context_zero, context_base, context_base], dim=0)
            prompt_embeds = torch.cat([uncond_base, text_base, text_base, preference_base], dim=0)
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=pipe.transformer.dtype):
                noise_pred = pipe.transformer(hidden_states=latent_model_input, timestep=timestep.expand(4), encoder_hidden_states=prompt_embeds, pooled_projections=None, return_dict=False)[0]
            transformer_forward_calls += 1
            uncond, text, matching, pref = noise_pred.chunk(4)
            noise_pred = uncond + text_scale * (text - uncond) + context_scale * (matching - text) + preference_scale * (pref - matching)
            before = latent
            particle["x0_prediction"] = estimate_x0_from_flowmatch(
                scheduler, noise_pred, before, step_idx - 1
            )
            # FlowMatchEulerDiscreteScheduler advances a mutable global
            # step_index on every call.  Reset it because particles are
            # intentionally denoised one at a time at the same timestep.
            scheduler._step_index = step_idx - 1
            after = scheduler.step(noise_pred, timestep, latent, return_dict=False)[0]
            if args.search_method == "fk" and args.fk_controlled_stochastic:
                sigma = float(scheduler.sigmas[step_idx - 1].detach().float().cpu().item())
                noise_scale = sigma * float(np.sqrt((1.0 + args.fk_gamma) ** 2 - 1.0))
                noise_seed = (
                    args.fk_stochastic_seed
                    + args.task_index * 1_000_000
                    + step_idx * 1_000
                    + int(particle["slot"])
                )
                generator = torch.Generator(device=after.device).manual_seed(noise_seed)
                noise = torch.randn(
                    after.shape,
                    generator=generator,
                    device=after.device,
                    dtype=after.dtype,
                )
                after = after + noise_scale * noise
                noise_event = {
                    "step": step_idx,
                    "slot": int(particle["slot"]),
                    "seed": int(particle["seed"]),
                    "noise_seed": noise_seed,
                    "sigma": sigma,
                    "noise_scale": noise_scale,
                }
                particle.setdefault("controlled_noise_history", []).append(noise_event)
                step_noise_rows.append(noise_event)
            particle["before"] = before
            particle["latent"] = after

        if step_noise_rows:
            fk_controlled_stochastic_events.append(
                {
                    "step": step_idx,
                    "gamma": args.fk_gamma,
                    "particles": step_noise_rows,
                }
            )

        if offline_cache_mode and step_idx in offline_cache_steps:
            x0_batch = [particle["x0_prediction"] for particle in active]
            decoded_x0 = torch.cat([decode_latent(pipe, latent) for latent in x0_batch], dim=0)
            vae_decode_calls += len(x0_batch)
            step_rows = []
            for particle, image in zip(active, decoded_x0):
                seed = int(particle["seed"])
                image_path = offline_cache_root / f"step_{step_idx:02d}" / f"seed_{seed}_x0.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                save_image(((image + 1.0) / 2.0).clamp(0, 1), str(image_path))
                row = {
                    "step": step_idx,
                    "seed": seed,
                    "image_path": str(image_path.resolve()),
                    "image_kind": "flowmatch_x0_prediction",
                }
                offline_cache_rows.append(row)
                step_rows.append(row)
            trace.append(
                {
                    "step": step_idx,
                    "batch_before": len(active),
                    "batch_after": len(active),
                    "offline_cache_images": step_rows,
                }
            )
        elif args.search_method == "psp" and step_idx in keep_by_step:
            x0_batch = [particle["x0_prediction"] for particle in active]
            decoded_x0 = torch.cat([decode_latent(pipe, latent) for latent in x0_batch], dim=0)
            vae_decode_calls += len(x0_batch)
            assert reward is not None
            scores = reward.score(decoded_x0, outfit, blank_idx)
            reward_score_events += 1
            reward_model_calls += len(active)
            keep = min(keep_by_step[step_idx], len(active))
            order = sorted(range(len(active)), key=lambda index: scores[index], reverse=True)
            selected = order[:keep]
            prediction_artifacts = []
            if args.save_intermediate_predictions:
                prediction_root = args.output_dir / "intermediate_predictions" / f"{uid}_{oid}" / f"step_{step_idx:02d}"
                prediction_root.mkdir(parents=True, exist_ok=True)
                survivor_seed_set = {int(active[i]["seed"]) for i in selected}
                for candidate_idx, particle in enumerate(active):
                    seed = int(particle["seed"])
                    latent_path = prediction_root / f"seed_{seed}_x_prediction.pt"
                    latent_png_path = prediction_root / f"seed_{seed}_x_prediction_latent.png"
                    decoded_path = prediction_root / f"seed_{seed}_x_prediction_decoded_reward.png"
                    torch.save({"uid": uid, "oid": oid, "step": step_idx, "seed": seed, "latent": x0_batch[candidate_idx].detach().float().cpu()}, latent_path)
                    save_latent_visualization(x0_batch[candidate_idx], latent_png_path)
                    save_image(((decoded_x0[candidate_idx] + 1.0) / 2.0).clamp(0, 1), str(decoded_path))
                    row = {"step": step_idx, "seed": seed, "reward": float(scores[candidate_idx]), "survivor": seed in survivor_seed_set, "latent_path": str(latent_path.resolve()), "latent_png_path": str(latent_png_path.resolve()), "decoded_reward_path": str(decoded_path.resolve())}
                    prediction_artifacts.append(row)
                    intermediate_prediction_rows.append(row)
            trace.append({"step": step_idx, "batch_before": len(active), "batch_after": keep, "scores": [{"seed": active[i]["seed"], "reward": float(scores[i])} for i in range(len(active))], "survivor_seeds": [active[i]["seed"] for i in selected], "prediction_artifacts": prediction_artifacts})
            active = [active[i] for i in selected]
        elif args.search_method == "fk" and step_idx in fk_config.resampling_steps:
            # FK keeps K slots. Parent indices may repeat, so copy every
            # particle-owned field rather than treating this like PSP pruning.
            x0_batch = [particle["x0_prediction"] for particle in active]
            decoded_x0 = torch.cat([decode_latent(pipe, latent) for latent in x0_batch], dim=0)
            vae_decode_calls += len(x0_batch)
            assert reward is not None
            scores = reward.score(decoded_x0, outfit, blank_idx)
            reward_score_events += 1
            reward_model_calls += len(active)
            previous_rewards = [float(particle["previous_reward"]) for particle in active]
            potentials, weights, ess = importance_weights(
                scores,
                previous_rewards,
                potential_type=fk_config.potential_type,
                lmbda=fk_config.lmbda,
            )
            threshold = fk_config.ess_fraction * len(active)
            should_resample = (not fk_config.adaptive_resampling) or ess < threshold
            parent_indices = (
                resample_parent_indices(weights, count=len(active), method=fk_config.resampling, rng=fk_rng)
                if should_resample
                else tuple(range(len(active)))
            )
            node_rows = []
            for slot, particle in enumerate(active):
                row = {
                    "step": step_idx,
                    "slot": slot,
                    "seed": int(particle["seed"]),
                    "root_seed": int(particle["root_seed"]),
                    "reward": float(scores[slot]),
                    "previous_reward": previous_rewards[slot],
                    "potential": float(potentials[slot]),
                    "normalized_weight": float(weights[slot]),
                    "ess": float(ess),
                }
                node_rows.append(row)
            trace_node = {
                "step": step_idx,
                "search_method": "fk",
                "batch_before": len(active),
                "batch_after": len(active),
                "potential_type": fk_config.potential_type,
                "lambda": fk_config.lmbda,
                "ess": float(ess),
                "ess_threshold": float(threshold),
                "resampled": bool(should_resample),
                "parents": list(parent_indices),
                "candidate_scores": node_rows,
                "controlled_stochastic_step": (
                    {"gamma": args.fk_gamma, "particles": step_noise_rows}
                    if step_noise_rows else None
                ),
            }
            trace.append(trace_node)
            for particle, score in zip(active, scores):
                particle["previous_reward"] = float(score)
            child_rows = [
                {
                    "step": step_idx,
                    "child_slot": child_slot,
                    "parent_slot": int(parent_slot),
                    "parent_seed": int(active[parent_slot]["seed"]),
                    "parent_root_seed": int(active[parent_slot]["root_seed"]),
                    "parent_reward": float(scores[parent_slot]),
                    "parent_potential": float(potentials[parent_slot]),
                    "parent_normalized_weight": float(weights[parent_slot]),
                    "ess": float(ess),
                    "resampled": bool(should_resample),
                }
                for child_slot, parent_slot in enumerate(parent_indices)
            ]
            trace_node["children"] = child_rows
            fk_score_rows.extend(child_rows)
            if should_resample:
                parent_active = active
                active = []
                for child_slot, parent_slot in enumerate(parent_indices):
                    parent = parent_active[parent_slot]
                    child = {
                        "slot": child_slot,
                        "seed": int(parent["seed"]),
                        "root_seed": int(parent["root_seed"]),
                        "lineage": list(parent["lineage"]) + [int(parent["root_seed"])],
                        "previous_reward": float(parent["previous_reward"]),
                        "latent": parent["latent"].clone(),
                        "controlled_noise_history": list(parent.get("controlled_noise_history", [])),
                    }
                    active.append(child)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_rows = []
    if offline_cache_mode:
        final_paths = {row["seed"]: row["image_path"] for row in offline_cache_rows if row["step"] == args.num_inference_steps}
        if len(final_paths) != args.num_particles:
            raise RuntimeError("Offline cache did not save every final-step candidate.")
        final_rows = [
            {"index": index, "seed": int(particle["seed"]), "image_path": final_paths[int(particle["seed"])]}
            for index, particle in enumerate(active)
        ]
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        final_images = torch.cat([decode_latent(pipe, particle["latent"]) for particle in active], dim=0)
        vae_decode_calls += len(active)
        assert reward is not None
        final_scores = reward.score(final_images, outfit, blank_idx)
        reward_score_events += 1
        reward_model_calls += len(active)
        for index, (particle, score) in enumerate(zip(active, final_scores)):
            path = args.output_dir / f"{uid}_{oid}" / f"{target_item_id}_{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            save_image(((final_images[index] + 1.0) / 2.0).clamp(0, 1), str(path))
            terminal_hash = hashlib.sha256(
                final_images[index].detach().float().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            final_rows.append(
                {
                    "index": index,
                    "slot": int(particle["slot"]),
                    "seed": int(particle["seed"]),
                    "root_seed": int(particle["root_seed"]),
                    "lineage": list(particle["lineage"]),
                    "controlled_noise_history": list(particle.get("controlled_noise_history", [])),
                    "reward": float(score),
                    "terminal_image_sha256": terminal_hash,
                    "image_path": str(path.resolve()),
                }
            )

    final_selection = None
    if not offline_cache_mode:
        winner_index = min(
            range(len(final_rows)),
            key=lambda index: (-float(final_rows[index]["reward"]), int(final_rows[index]["root_seed"]), index),
        )
        for index, row in enumerate(final_rows):
            row["selected"] = index == winner_index
        final_selection = final_rows[winner_index]

    audit_rows: list[dict] = []
    audit_initial_hashes: dict[str, str] = {}
    audit_transformer_forward_calls = 0
    audit_vae_decode_calls = 0
    audit_reward_model_calls = 0
    if args.audit_all_candidates:
        audit_root = args.output_dir / "all_candidates" / f"{uid}_{oid}"
        audit_root.mkdir(parents=True, exist_ok=True)
        # Re-run every original seed from its untouched initial latent.  This
        # is a regular full trajectory audit, not a continuation of survivors.
        scheduler.set_timesteps(args.num_inference_steps, device=device)
        audit_timesteps = scheduler.timesteps.to(device)
        audit_latents: list[torch.Tensor] = []
        for candidate_seed in candidate_seeds:
            latent = torch.randn(
                (1, *latent_shape),
                generator=torch.Generator(device=device).manual_seed(candidate_seed),
                device=device,
                dtype=pipe.transformer.dtype,
            )
            audit_initial_hashes[str(candidate_seed)] = hashlib.sha256(
                latent.detach().float().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            scheduler._step_index = None
            for timestep in audit_timesteps:
                latent_model_input = torch.cat([latent, latent, latent, latent], dim=0)
                context_base = context_offset.expand(1, *context_offset.shape[1:]).contiguous()
                context_zero = torch.zeros_like(context_base)
                latent_model_input = latent_model_input + MATCHING_LAMBDA * torch.cat(
                    [context_zero, context_zero, context_base, context_base], dim=0
                )
                prompt_embeds = torch.cat([uncond_base, text_base, text_base, preference_base], dim=0)
                with torch.no_grad(), torch.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=pipe.transformer.dtype,
                ):
                    noise_pred = pipe.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep.expand(4),
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=None,
                        return_dict=False,
                    )[0]
                audit_transformer_forward_calls += 1
                uncond, text, matching, pref = noise_pred.chunk(4)
                noise_pred = (
                    uncond
                    + text_scale * (text - uncond)
                    + context_scale * (matching - text)
                    + preference_scale * (pref - matching)
                )
                latent = scheduler.step(noise_pred, timestep, latent, return_dict=False)[0]
            audit_latents.append(latent)

        audit_images = torch.cat([decode_latent(pipe, latent) for latent in audit_latents], dim=0)
        audit_vae_decode_calls += len(audit_latents)
        audit_scores = reward.score(audit_images, outfit, blank_idx)
        audit_reward_model_calls += len(audit_latents)
        for candidate_idx, (candidate_seed, score) in enumerate(zip(candidate_seeds, audit_scores)):
            image_path = audit_root / f"{target_item_id}_{candidate_idx}_seed{candidate_seed}.png"
            save_image(((audit_images[candidate_idx] + 1.0) / 2.0).clamp(0, 1), str(image_path))
            audit_rows.append(
                {
                    "candidate_idx": candidate_idx,
                    "seed": candidate_seed,
                    "initial_latent_hash": audit_initial_hashes[str(candidate_seed)],
                    "psp_initial_latent_hash": initial_latent_hashes[str(candidate_seed)],
                    "initial_hash_match": audit_initial_hashes[str(candidate_seed)] == initial_latent_hashes[str(candidate_seed)],
                    "compatibility_reward": float(score),
                    "image_path": str(image_path.resolve()),
                }
            )
    peak_vram_gb = None
    if device.type == "cuda":
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    scheduler_config = json.loads(json.dumps(dict(pipe.scheduler.config), default=str))
    payload = {
        "uid": uid,
        "oid": oid,
        "target_item_id": target_item_id,
        "task_index": args.task_index,
        "search_method": args.search_method,
        "candidate_seeds": candidate_seeds,
        "initial_latent_hashes": initial_latent_hashes,
        "schedule": reported_schedule,
        "fk_config": (
            {
                "num_particles": fk_config.num_particles,
                "resampling_steps": list(fk_config.resampling_steps),
                "potential_type": fk_config.potential_type,
                "lambda": fk_config.lmbda,
                "resampling": fk_config.resampling,
                "adaptive_resampling": fk_config.adaptive_resampling,
                "ess_fraction": fk_config.ess_fraction,
                "resampling_rng_seed": args.fk_resampling_seed + args.task_index * args.seed_stride,
                "controlled_stochastic": args.fk_controlled_stochastic,
                "controlled_stochastic_gamma": args.fk_gamma,
                "controlled_stochastic_seed_base": args.fk_stochastic_seed,
                "controlled_stochastic_rule": (
                    "x_next=x_det+sigma_i*sqrt((1+gamma)^2-1)*Normal(0,I) after every Euler step"
                    if args.fk_controlled_stochastic else None
                ),
            }
            if fk_config is not None else None
        ),
        "num_inference_steps": args.num_inference_steps,
        "scheduler_class": pipe.scheduler.__class__.__name__,
        "scheduler_config": scheduler_config,
        "checkpoint": str(model_path.resolve()),
        "precision": str(pipe.transformer.dtype),
        "deterministic_context": args.deterministic_context,
        "guidance_scales": {
            "text": args.text_guidance_scale,
            "context": args.context_guidance_scale,
            "preference": args.preference_guidance_scale,
        },
        "reward": (
            "not evaluated; offline candidate cache only"
            if offline_cache_mode
            else (
                "sigmoid(FashionEvaluator(OpenCLIP_ViT-H-14(candidate), fixed_iFashion_context))"
                if args.reward == "compatibility"
                else "100*cosine(normalized OpenCLIP_ViT-H-14(candidate), normalized test_history_clipembs[uid][target_category])"
            )
        ),
        "reward_name": args.reward,
        "transformer_forward_calls": transformer_forward_calls,
        "model_forward_budget": {
            "actual_transformer_calls": transformer_forward_calls,
            "expected_fk_or_bon_calls": args.num_particles * args.num_inference_steps,
            "note": "FK keeps K particles at every denoising step; VAE/reward calls are reported separately.",
        } if args.search_method == "fk" else None,
        "vae_encode_calls": 1,
        "vae_decode_calls": vae_decode_calls,
        "reward_score_events": reward_score_events,
        "reward_model_calls": reward_model_calls,
        "fk_controlled_stochastic_events": fk_controlled_stochastic_events,
        "audit_all_candidates": bool(args.audit_all_candidates),
        "audit_initial_latent_hashes": audit_initial_hashes,
        "audit_initial_hashes_match_psp": bool(
            args.audit_all_candidates
            and audit_initial_hashes
            and all(
                audit_initial_hashes[str(seed)] == initial_latent_hashes[str(seed)]
                for seed in candidate_seeds
            )
        ) if args.audit_all_candidates else None,
        "audit_transformer_forward_calls": audit_transformer_forward_calls,
        "audit_vae_decode_calls": audit_vae_decode_calls,
        "audit_reward_model_calls": audit_reward_model_calls,
        "audit_candidates": audit_rows,
        "save_intermediate_predictions": bool(args.save_intermediate_predictions),
        "intermediate_prediction_artifacts": intermediate_prediction_rows,
        "offline_candidate_cache": {
            "enabled": offline_cache_mode,
            "steps": offline_cache_steps,
            "candidate_count": args.num_particles,
            "expected_image_count": len(offline_cache_steps) * args.num_particles,
            "image_count": len(offline_cache_rows),
            "image_kind": "flowmatch_x0_prediction",
            "images": offline_cache_rows,
        },
        "peak_vram_gb": peak_vram_gb,
        "trace": trace,
        "final": final_rows,
        "final_selection": final_selection,
        "terminal_unique_image_count": len({row.get("terminal_image_sha256") for row in final_rows}),
        "terminal_duplicate_count": len(final_rows) - len({row.get("terminal_image_sha256") for row in final_rows}),
        "elapsed_s": time.perf_counter() - start,
        "command": sys.argv,
    }
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    if not offline_cache_mode and args.search_method == "psp":
        with (args.output_dir / "intermediate_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "seed", "reward", "survivor"])
            for row in trace:
                survivors = set(row["survivor_seeds"])
                for item in row["scores"]:
                    writer.writerow([row["step"], item["seed"], item["reward"], item["seed"] in survivors])
    if fk_score_rows:
        with (args.output_dir / "fk_resampling_trace.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fk_score_rows[0]))
            writer.writeheader()
            writer.writerows(fk_score_rows)
    if intermediate_prediction_rows:
        with (args.output_dir / "intermediate_prediction_artifacts.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(intermediate_prediction_rows[0]))
            writer.writeheader()
            writer.writerows(intermediate_prediction_rows)
    if audit_rows:
        with (args.output_dir / "all_candidates_scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
            writer.writeheader()
            writer.writerows(audit_rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
