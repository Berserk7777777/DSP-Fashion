"""Run one online FK-Steering diagnostic for a DiFashion iFashion GOR row.

GOR has four generated items per outfit.  FK resampling therefore operates on
four *outfit* particles, while DiFashion's denoising batch contains sixteen
item latents.  Each selected parent outfit is expanded to its four contiguous
item-latent indices before the pipeline reindexes PNDM state.

This program intentionally does not load an item ground truth, retrieval
candidate, or post-hoc GT metric.  It is an exploratory single-task run until
the FK hyperparameters have been frozen on a validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from diffusers import PNDMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from PIL import Image
from torchvision import transforms

WORKSPACE = Path(__file__).resolve().parents[3]
DIFASHION_ROOT = WORKSPACE / "DiFashion-main"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(DIFASHION_ROOT / "DiFashion"))

import data_utils  # noqa: E402
from fashion_psp.baselines.fk_steering import importance_weights, resample_parent_indices  # noqa: E402
from models.difashion import DiFashion, MutualEncoder  # noqa: E402


# The GOR cache protocol deliberately uses a 51-step scheduler argument.  Its
# installed PNDM scheduler exposes 52 actual model-loop calls.
INFERENCE_ARGUMENT = 51
EXPECTED_LOOPS = 52
DEFAULT_PARTICLES = 4
ITEMS_PER_OUTFIT = 4
PERCENT_NODES = (10, 21, 31, 42)  # round(52 * [0.2, 0.4, 0.6, 0.8])
PERSONAL_CLIP_WEIGHTS = Path(
    r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K"
    r"\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Directory containing ifashion/.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, required=True,
                        help="Contains cached OpenCLIP assets for Personal Sim.")
    parser.add_argument("--personal-clip-weights", type=Path, default=PERSONAL_CLIP_WEIGHTS,
                        help="Local frozen ViT-H/14 safetensors checkpoint; never downloaded at runtime.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=103)
    parser.add_argument("--expected-task-id", default="00103_9334_17997")
    parser.add_argument("--seed-base", type=int, default=4042,
                        help="Particle i seed = seed_base + task_index * 10000 + i.")
    parser.add_argument("--num-particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--resampling-steps", type=int, nargs="+", default=list(PERCENT_NODES),
                        help="One-based actual loop nodes. Default: 20/40/60/80%% of 52 calls.")
    parser.add_argument("--potential-type", choices=("diff", "max", "add", "rt"), default="max")
    parser.add_argument("--lmbda", type=float, required=True,
                        help="FK temperature for Personal Sim's 0--100 scale; freeze on validation.")
    parser.add_argument("--resampling", choices=("systematic", "multinomial"), default="systematic")
    parser.add_argument("--adaptive-resampling", action="store_true",
                        help="Only resample an intermediate node if ESS < K/2. Terminal is always resampled.")
    parser.add_argument("--resampling-seed", type=int, default=20261016)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=("fp16", "no"), default="fp16")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_args(base_model: Path) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_model_name_or_path=str(base_model), revision=None, non_ema_revision=None,
        enable_xformers_memory_efficient_attention=False, category_emb_size=64, hid_dim=256,
        use_history=True, use_mutual_guidance=True, eta=0.1, noise_offset=0.0, snr_gamma=None,
    )


def decode_batch(model: DiFashion, latents: torch.Tensor) -> list[Image.Image]:
    decoded = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
    return model.image_processor.postprocess(
        decoded, output_type="pil", do_denormalize=[True] * decoded.shape[0]
    )


def save_outfit(images: list[Image.Image], directory: Path) -> list[Path]:
    if len(images) != ITEMS_PER_OUTFIT:
        raise ValueError(f"GOR requires {ITEMS_PER_OUTFIT} images per particle, got {len(images)}")
    directory.mkdir(parents=True, exist_ok=False)
    saved: list[Path] = []
    for item_index, image in enumerate(images):
        path = directory / f"item_{item_index}.png"
        image.save(path)
        saved.append(path)
    width, height = images[0].size
    combined = Image.new("RGB", (2 * width, 2 * height), color=(255, 255, 255))
    for item_index, image in enumerate(images):
        combined.paste(image.convert("RGB"), ((item_index % 2) * width, (item_index // 2) * height))
    all_path = directory / "all.png"
    combined.save(all_path)
    return saved + [all_path]


class GORPersonalReward:
    """Mean of four frozen Personal Sim item scores for a generated outfit."""

    def __init__(self, data_root: Path, dualfashion_root: Path, weights: Path,
                 uid: int, categories: np.ndarray, device: torch.device):
        history_path = data_root / "ifashion" / "processed" / "test_history_clipembs.npy"
        if not history_path.is_file():
            raise FileNotFoundError(f"Missing Personal Sim history: {history_path}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        sys.path.insert(0, str(dualfashion_root.resolve()))
        import open_clip

        history = np.load(history_path, allow_pickle=True).item()
        user_history = history.get(uid, history.get(str(uid)))
        if user_history is None:
            raise KeyError(f"No test history for uid={uid}")
        targets = []
        for category in categories.tolist():
            target = user_history.get(int(category), user_history.get(str(int(category))))
            if target is None:
                raise KeyError(f"No history embedding for uid={uid}, category={category}")
            targets.append(np.asarray(target, dtype=np.float32).reshape(-1))
        self.targets = F.normalize(torch.as_tensor(np.stack(targets), device=device), dim=-1)
        if not weights.is_file():
            raise FileNotFoundError(f"Missing frozen Personal OpenCLIP weights: {weights}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
        from safetensors.torch import load_file
        missing, unexpected = self.model.load_state_dict(load_file(str(weights), device="cpu"), strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Personal OpenCLIP checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        self.model = self.model.to(device).eval()
        self.device = device
        self.provenance = {
            "formula": "mean_j(100 * cosine(OpenCLIP(generated_item_j), test_history_clipembs[uid][category_j]))",
            "openclip": "ViT-H-14/laion2b-s32b-b79K",
            "openclip_weights": str(weights.resolve()),
            "personal_history": str(history_path.resolve()),
            "uses_ground_truth": False,
        }

    def score(self, images: list[Image.Image]) -> tuple[list[float], list[list[float]]]:
        if len(images) % ITEMS_PER_OUTFIT:
            raise ValueError("Decoded GOR batch is not divisible into four-item outfits")
        pixels = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
        with torch.inference_mode():
            features = F.normalize(self.model.encode_image(pixels).float(), dim=-1)
            feature_blocks = features.reshape(-1, ITEMS_PER_OUTFIT, features.shape[-1])
            item_values = 100.0 * (feature_blocks * self.targets.unsqueeze(0)).sum(dim=-1)
            outfit_values = item_values.mean(dim=1)
        return (
            [float(value) for value in outfit_values.cpu().tolist()],
            [[float(value) for value in row] for row in item_values.cpu().tolist()],
        )


def stable_weights_from_logits(logits: list[float]) -> tuple[list[float], float]:
    offset = max(logits)
    values = [math.exp(value - offset) for value in logits]
    total = sum(values)
    weights = [value / total for value in values]
    return weights, 1.0 / sum(value * value for value in weights)


def check_inputs(args: argparse.Namespace) -> None:
    data_root = args.data_root / "ifashion"
    required = [
        data_root / "processed" / "new_fitb_test.npy",
        data_root / "processed" / "test_hist_latents.npy",
        data_root / "processed" / "test_history_clipembs.npy",
        args.personal_clip_weights,
        data_root / "new_id_cate_dict.npy",
        data_root / "new_all_item_image_paths.npy",
        args.image_root / "empty_image.png",
        args.checkpoint_root / "checkpoint-15000" / "unet_ema" / "diffusion_pytorch_model.bin",
        args.checkpoint_root / "checkpoint-15000" / "fashion_encoder_ema" / "diffusion_pytorch_model.bin",
        args.base_model / "scheduler" / "scheduler_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    if args.num_particles != DEFAULT_PARTICLES:
        raise ValueError("This controlled FK protocol fixes GOR to four online outfit particles")
    if not math.isfinite(args.lmbda) or args.lmbda <= 0:
        raise ValueError("--lmbda must be finite and positive")
    if tuple(sorted(set(args.resampling_steps))) != tuple(args.resampling_steps):
        raise ValueError("--resampling-steps must be unique and strictly increasing")
    if any(step < 1 or step >= EXPECTED_LOOPS for step in args.resampling_steps):
        raise ValueError(f"Intermediate nodes must lie in [1, {EXPECTED_LOOPS - 1}]")
    scheduler = PNDMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    scheduler.set_timesteps(INFERENCE_ARGUMENT, device="cpu")
    if len(scheduler.timesteps) != EXPECTED_LOOPS:
        raise RuntimeError(f"PNDM argument {INFERENCE_ARGUMENT} produced {len(scheduler.timesteps)} loops")


def main() -> int:
    args = parse_args()
    check_inputs(args)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False

    data_root = args.data_root / "ifashion"
    data = np.load(data_root / "processed" / "new_fitb_test.npy", allow_pickle=True).item()
    if not 0 <= args.task_index < len(data["uids"]):
        raise IndexError(f"task index outside 0..{len(data['uids']) - 1}: {args.task_index}")
    uid, oid = int(data["uids"][args.task_index]), int(data["oids"][args.task_index])
    task_id = f"{args.task_index:05d}_{uid}_{oid}"
    if args.expected_task_id and task_id != args.expected_task_id:
        raise ValueError(f"Task identity mismatch: expected {args.expected_task_id}, resolved {task_id}")
    categories = np.asarray(data["category"][args.task_index], dtype=np.int64).reshape(-1)
    input_ids = np.asarray(data["input_ids"][args.task_index], dtype=np.int64)
    if len(categories) != ITEMS_PER_OUTFIT or input_ids.shape[0] != ITEMS_PER_OUTFIT:
        raise ValueError("Selected GOR condition does not contain four categories and prompts")
    particle_seeds = [args.seed_base + args.task_index * 10000 + index for index in range(args.num_particles)]
    if len(set(particle_seeds)) != len(particle_seeds):
        raise RuntimeError("FK particle seed collision")

    output = args.output_root / task_id
    result_path = output / "run.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if args.resume:
            print(json.dumps(existing, indent=2))
            return 0
        raise FileExistsError(f"Existing complete output: {result_path}; use --resume or a new output root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete output preserved at {output}; choose a new output root")
    output.mkdir(parents=True, exist_ok=False)

    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("difashion_gor_fk")
    image_paths = np.load(data_root / "new_all_item_image_paths.npy", allow_pickle=True)
    history_latents = np.load(data_root / "processed" / "test_hist_latents.npy", allow_pickle=True).item()
    id_categories = np.load(data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), transforms.ToTensor(),
    ])
    dataset = data_utils.ImagePathDataset(str(args.image_root), image_paths, transform, do_normalize=True)
    model = DiFashion(model_args(args.base_model), logger, len(id_categories), device).to(device)
    model.vae.enable_slicing()
    checkpoint = args.checkpoint_root / "checkpoint-15000"
    ema_unet = EMAModel.from_pretrained(str(checkpoint / "unet_ema"), UNet2DConditionModel)
    ema_encoder = EMAModel.from_pretrained(str(checkpoint / "fashion_encoder_ema"), MutualEncoder)
    ema_unet.to(device); ema_encoder.to(device)
    ema_unet.copy_to(model.unet.parameters()); ema_encoder.copy_to(model.fashion_encoder.parameters())
    model.eval(); model.unet.eval(); model.fashion_encoder.eval()
    scorer = GORPersonalReward(args.data_root, args.dualfashion_root, args.personal_clip_weights, uid, categories, device)

    particles = [{"slot": index, "initial_seed": seed, "ancestor_seed": seed, "lineage": [seed]}
                 for index, seed in enumerate(particle_seeds)]
    previous_rewards = [0.0] * args.num_particles
    log_products = [0.0] * args.num_particles
    resample_rng = random.Random(args.resampling_seed)
    trace: list[dict[str, Any]] = []
    schedules: list[dict[str, int]] = []
    terminal_selection: dict[str, Any] = {}
    node_set = set(args.resampling_steps) | {EXPECTED_LOOPS}

    def on_resample(loop_index, timestep, _latent_before, _noise_pred, x_prediction, _latent_after):
        step = loop_index + 1
        schedules.append({"loop_index_zero_based": int(loop_index), "step_number_one_based": step,
                          "scheduler_timestep": int(timestep.item())})
        if step not in node_set:
            return None
        images = decode_batch(model, x_prediction)
        scores, item_scores = scorer.score(images)
        if len(scores) != args.num_particles:
            raise RuntimeError(f"Expected {args.num_particles} GOR particles, decoded {len(scores)}")
        candidates = []
        for slot, (particle, score, individual) in enumerate(zip(particles, scores, item_scores)):
            start = slot * ITEMS_PER_OUTFIT
            node_dir = output / "nodes" / f"step_{step:02d}" / f"slot_{particle['slot']}_ancestor_{particle['ancestor_seed']}"
            paths = save_outfit(images[start:start + ITEMS_PER_OUTFIT], node_dir)
            candidates.append({**particle, "personal_sim": score, "item_personal_sim": individual,
                               "reward_image_paths": [str(path.resolve()) for path in paths]})

        potentials, normalized, ess = importance_weights(
            scores, previous_rewards, potential_type=args.potential_type, lmbda=args.lmbda
        )
        if step == EXPECTED_LOOPS and args.potential_type in {"max", "add"}:
            normalized, ess = stable_weights_from_logits([
                args.lmbda * potential - log_product
                for potential, log_product in zip(potentials, log_products)
            ])
            weight_formula = "terminal softmax(lambda * potential - log_product_of_prior_selected_potentials)"
        else:
            weight_formula = "softmax(lambda * potential)"
        # There is no next PNDM transition after the terminal node.  FK's
        # terminal operation is consequently one categorical sample, rather
        # than a K-to-K state copy which would create unused duplicates.
        if step == EXPECTED_LOOPS:
            rng_state = repr(resample_rng.getstate())
            selected_slot = int(resample_parent_indices(
                normalized, count=1, method=args.resampling, rng=resample_rng
            )[0])
            terminal_selection.update({
                "selection": "one categorical sample from terminal FK weights",
                "resampling": args.resampling,
                "resampling_rng_state_before": rng_state,
                "selected_slot_before_terminal": selected_slot,
                "selected_particle_before_terminal": dict(particles[selected_slot]),
                "selected_terminal_weight": float(normalized[selected_slot]),
                "selected_terminal_personal_sim_x_prediction": float(scores[selected_slot]),
            })
            trace.append({
                "step_number_one_based": step, "loop_index_zero_based": int(loop_index),
                "scheduler_timestep": int(timestep.item()), "node_kind": "terminal_single_sample",
                "reward": "personal", "reward_provenance": scorer.provenance,
                "prediction_semantics": "decoded x_prediction (predicted clean x0 before scheduler update)",
                "potential_type": args.potential_type, "lmbda": args.lmbda, "weight_formula": weight_formula,
                "particles_before": candidates, "previous_rewards": list(previous_rewards),
                "potentials": list(potentials), "normalized_weights": list(normalized), "ess": ess,
                "did_resample": False, "parent_indices_outfit": None, "parent_indices_item_latent": None,
                "terminal_selection": dict(terminal_selection),
            })
            return None

        should_resample = not args.adaptive_resampling or ess < args.num_particles / 2.0
        parents = list(range(args.num_particles))
        rng_state = None
        if should_resample:
            rng_state = repr(resample_rng.getstate())
            parents = list(resample_parent_indices(
                normalized, count=args.num_particles, method=args.resampling, rng=resample_rng
            ))
        item_parents = [parent * ITEMS_PER_OUTFIT + item
                        for parent in parents for item in range(ITEMS_PER_OUTFIT)]
        trace.append({
            "step_number_one_based": step, "loop_index_zero_based": int(loop_index),
            "scheduler_timestep": int(timestep.item()),
            "node_kind": "intermediate_resample",
            "reward": "personal", "reward_provenance": scorer.provenance,
            "prediction_semantics": "decoded x_prediction (predicted clean x0 before scheduler update)",
            "potential_type": args.potential_type, "lmbda": args.lmbda, "weight_formula": weight_formula,
            "particles_before": candidates, "previous_rewards": list(previous_rewards),
            "potentials": list(potentials), "normalized_weights": list(normalized), "ess": ess,
            "adaptive_resampling": bool(args.adaptive_resampling), "did_resample": should_resample,
            "resampling": args.resampling if should_resample else None,
            "resampling_rng_state_before": rng_state, "parent_indices_outfit": parents,
            "parent_indices_item_latent": item_parents,
        })
        if not should_resample:
            previous_rewards[:] = scores
            return None
        if args.potential_type in {"max", "add"}:
            next_log_products = [log_products[parent] + args.lmbda * potentials[parent] for parent in parents]
        else:
            next_log_products = [log_products[parent] for parent in parents]
        previous_rewards[:] = [scores[parent] for parent in parents]
        log_products[:] = next_log_products
        particles[:] = [{
            "slot": child_slot,
            "initial_seed": particles[parent]["initial_seed"],
            "ancestor_seed": particles[parent]["ancestor_seed"],
            "lineage": list(particles[parent]["lineage"]) + [step],
            "parent_slot_at_previous_node": parent,
        } for child_slot, parent in enumerate(parents)]
        # The core sampler expects an index for every active item latent.  It
        # reindexes current, previous, and PNDM multistep tensors with this map.
        return item_parents

    try:
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                null_image = dataset[0].unsqueeze(0).to(device)
                null_latent = model.vae.encode(null_image).latent_dist.mode()[0] * model.vae.config.scaling_factor
            uid_t = torch.full((args.num_particles,), uid, dtype=torch.long, device=device)
            oid_t = torch.full((args.num_particles,), oid, dtype=torch.long, device=device)
            empty_outfit = torch.zeros((args.num_particles, ITEMS_PER_OUTFIT), dtype=torch.long, device=device)
            category_t = torch.as_tensor(categories, dtype=torch.long, device=device).unsqueeze(0).repeat(args.num_particles, 1)
            input_t = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(args.num_particles, 1, 1)
            # ``fashion_generation`` has sixteen generated item latents here,
            # but an FK seed identifies one four-item outfit particle.  Draw
            # four sequential item noises from each particle generator first;
            # this is equivalent to a standalone GOR call for that seed and
            # avoids pretending a four-generator list matches sixteen latents.
            initial_latents = torch.cat([
                model.prepare_latents(
                    ITEMS_PER_OUTFIT, model.vae.config.latent_channels, 512, 512,
                    model.text_encoder.dtype, device, torch.Generator(device=device).manual_seed(seed),
                )
                for seed in particle_seeds
            ], dim=0)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                result, _ = model.fashion_generation(
                    uid_t, oid_t, input_t, empty_outfit, None, category_t, history_latents,
                    num_inference_steps=INFERENCE_ARGUMENT,
                    category_guidance_scale=12.0, hist_guidance_scale=4.0, mutual_guidance_scale=5.0,
                    null_img=null_image[0], outfit_latents=null_latent.repeat(args.num_particles * ITEMS_PER_OUTFIT, 1, 1, 1),
                    null_latent=null_latent, init_latents=initial_latents, output_type="pil", return_dict=False,
                    resample_callback=on_resample,
                )
        if len(schedules) != EXPECTED_LOOPS:
            raise RuntimeError(f"Expected {EXPECTED_LOOPS} loop calls, observed {len(schedules)}")
        if len(trace) != len(args.resampling_steps) + 1:
            raise RuntimeError(f"Expected {len(args.resampling_steps) + 1} FK nodes, observed {len(trace)}")
        final_images = result[uid][oid]["images"]
        if len(final_images) != args.num_particles * ITEMS_PER_OUTFIT:
            raise RuntimeError(f"Expected {args.num_particles * ITEMS_PER_OUTFIT} terminal GOR images, got {len(final_images)}")
        if not terminal_selection:
            raise RuntimeError("Terminal FK selection was not recorded")
        selected_slot = int(terminal_selection["selected_slot_before_terminal"])
        selected_particle = particles[selected_slot]
        start = selected_slot * ITEMS_PER_OUTFIT
        directory = output / "final" / "selected"
        paths = save_outfit(final_images[start:start + ITEMS_PER_OUTFIT], directory)
        selected_final = {**selected_particle,
                          "terminal_image_paths": [str(path.resolve()) for path in paths],
                          "sha256": [sha256(path) for path in paths],
                          "terminal_decode_semantics": "scheduler terminal latent after its final update"}
        if selected_final["ancestor_seed"] != terminal_selection["selected_particle_before_terminal"]["ancestor_seed"]:
            raise RuntimeError("Terminal selection lineage changed after final model output")
        duplicate_checks = []
        for node in trace[:-1]:
            parents = node["parent_indices_outfit"]
            for parent in sorted({value for value in parents if parents.count(value) > 1}):
                children = [slot for slot, value in enumerate(parents) if value == parent]
                duplicate_checks.append({"resample_step": node["step_number_one_based"], "parent_slot": parent,
                                         "children": children,
                                         "not_checked": "Only the selected terminal result is saved in this single-output run."})
        payload = {
            "schema": 1, "status": "complete", "task": "GOR", "task_id": task_id,
            "task_index": args.task_index, "uid": uid, "oid": oid,
            "condition": {"all_outfit_items_masked": True, "category": categories.tolist()},
            "ground_truth_loaded_or_used": False,
            "method": "fk_steering_online_diagnostic", "reward": "personal",
            "test_status": "exploratory_single_task; FK parameters require validation freeze before a test claim",
            "particles": args.num_particles, "particle_seed_mapping": particle_seeds,
            "scheduler": {"class": "PNDMScheduler", "native_num_inference_steps_argument": INFERENCE_ARGUMENT,
                          "actual_loop_calls": EXPECTED_LOOPS, "resampling_steps_one_based": args.resampling_steps,
                          "terminal_single_sample_step_one_based": EXPECTED_LOOPS},
            "percent_node_mapping": {"20%": 10, "40%": 21, "60%": 31, "80%": 42, "100%": 52},
            "fk": {"potential_type": args.potential_type, "lmbda": args.lmbda,
                   "resampling": args.resampling, "adaptive_resampling": bool(args.adaptive_resampling),
                   "resampling_seed": args.resampling_seed,
                   "reference_semantics": "K=4 outfit particles; with-replacement resampling; parent blocks expanded to four GOR item latents"},
            "checkpoint": str(checkpoint.resolve()), "base_model": str(args.base_model.resolve()),
            "guidance": {"category": 12.0, "mutual": 5.0, "history": 4.0, "eta": 0.1},
            "online_unet_forward_calls": args.num_particles * EXPECTED_LOOPS,
            "trace": trace, "scheduler_timestep_schedule": schedules,
            "final_selection": {**terminal_selection, "selected_final": selected_final},
            "deterministic_duplicate_gate_local": {"duplicate_descendant_checks": duplicate_checks,
                "warning": "PNDM is deterministic after initial noise. Duplicated parents can remain pixel-identical without a validated rejuvenation kernel."},
        }
        atomic_json(result_path, payload)
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        del model, ema_unet, ema_encoder, scorer
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
