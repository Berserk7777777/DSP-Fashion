"""Run one online FK-Steering diagnostic on a DiFashion iFashion FITB task.

This runner keeps K particles alive for the complete native DiFashion PNDM
trajectory.  At the requested online nodes it scores decoded x0 predictions,
then resamples parent particles with replacement.  It is intentionally a
single-task diagnostic: parameters must be frozen on validation data before
using it for a test-set claim.

No ground-truth or retrieval candidate asset is opened by this program.
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

# This must be set before CUDA is initialized for deterministic CuBLAS GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
from diffusers import PNDMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from torchvision import transforms

WORKSPACE = Path(__file__).resolve().parents[3]
DIFASHION_ROOT = WORKSPACE / "DiFashion-main"
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(DIFASHION_ROOT / "DiFashion"))

import data_utils  # noqa: E402
from fashion_psp.baselines.fk_steering import importance_weights, resample_parent_indices  # noqa: E402
from models.difashion import DiFashion, MutualEncoder  # noqa: E402


DEFAULT_STEPS = 50
EXPECTED_LOOPS = 51
DEFAULT_PARTICLES = 4
PERCENT_NODES = (10, 20, 31, 41)  # round(51 * [0.2, 0.4, 0.6, 0.8])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="iFashion directory containing processed/.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, required=True,
                        help="Contains the cached OpenCLIP dependency used by Personal Sim.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=874)
    parser.add_argument("--expected-task-id", default="00874_10461_18722")
    parser.add_argument("--seed-base", type=int, default=4042,
                        help="Per-task particle i uses seed_base + task_index * 10000 + i.")
    parser.add_argument("--num-particles", type=int, default=DEFAULT_PARTICLES)
    parser.add_argument("--resampling-steps", type=int, nargs="+", default=list(PERCENT_NODES),
                        help="One-based actual PNDM loop nodes. Default maps 20/40/60/80%% of 51 calls.")
    parser.add_argument("--potential-type", choices=("diff", "max", "add", "rt"), default="max")
    parser.add_argument("--lmbda", type=float, required=True,
                        help="Frozen FK temperature for this reward scale. Do not transfer ImageReward's value blindly.")
    parser.add_argument("--resampling", choices=("systematic", "multinomial"), default="systematic")
    parser.add_argument("--adaptive-resampling", action="store_true",
                        help="Match FKD's optional ESS < K/2 gate. Default resamples at every configured node.")
    parser.add_argument("--resampling-seed", type=int, default=20261013,
                        help="CPU RNG seed used only for parent sampling.")
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


def decode_batch(model: DiFashion, latents: torch.Tensor):
    decoded = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False)[0]
    return model.image_processor.postprocess(
        decoded, output_type="pil", do_denormalize=[True] * decoded.shape[0]
    )


class PersonalReward:
    """Frozen Personal Sim scorer with a fixed user/category history vector."""

    def __init__(self, data_root: Path, dualfashion_root: Path, uid: int, category: int, device: torch.device):
        history_path = data_root / "processed" / "test_history_clipembs.npy"
        if not history_path.is_file():
            raise FileNotFoundError(f"Missing Personal Sim history: {history_path}")
        # The frozen evaluator has already been cached locally. A network
        # probe makes this experiment less reproducible and can hang in a lab.
        os.environ["HF_HUB_OFFLINE"] = "1"
        sys.path.insert(0, str(dualfashion_root.resolve()))
        import open_clip

        history = np.load(history_path, allow_pickle=True).item()
        user_history = history.get(uid, history.get(str(uid)))
        if user_history is None:
            raise KeyError(f"No test history for uid={uid}")
        target = user_history.get(category, user_history.get(str(category)))
        if target is None:
            raise KeyError(f"No target-category history for uid={uid}, category={category}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b-s32b-b79K"
        )
        self.model = self.model.to(device).eval()
        self.target = F.normalize(
            torch.as_tensor(np.asarray(target, dtype=np.float32).reshape(1, -1), device=device), dim=-1
        )
        self.device = device
        self.provenance = {
            "formula": "100 * cosine(normalized OpenCLIP(candidate), normalized test_history_clipembs[uid][target_category])",
            "openclip": "ViT-H-14/laion2b-s32b-b79K",
            "personal_history": str(history_path.resolve()),
        }

    def score(self, images) -> list[float]:
        pixels = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
        with torch.inference_mode():
            features = F.normalize(self.model.encode_image(pixels).float(), dim=-1)
            values = 100.0 * (features * self.target).sum(dim=-1)
        return [float(value) for value in values.cpu().tolist()]


def stable_weights_from_logits(logits: list[float]) -> tuple[list[float], float]:
    offset = max(logits)
    values = [math.exp(value - offset) for value in logits]
    total = sum(values)
    weights = [value / total for value in values]
    return weights, 1.0 / sum(value * value for value in weights)


def check_inputs(args: argparse.Namespace) -> None:
    required = [
        args.data_root / "processed" / "new_fitb_test.npy",
        args.data_root / "processed" / "test_hist_latents.npy",
        args.data_root / "new_id_cate_dict.npy",
        args.data_root / "new_all_item_image_paths.npy",
        args.image_root / "empty_image.png",
        args.checkpoint_root / "checkpoint-15000" / "unet_ema" / "diffusion_pytorch_model.bin",
        args.checkpoint_root / "checkpoint-15000" / "fashion_encoder_ema" / "diffusion_pytorch_model.bin",
        args.base_model / "scheduler" / "scheduler_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    if args.num_particles != DEFAULT_PARTICLES:
        raise ValueError("This controlled protocol fixes FK-Steering to four online particles")
    if not math.isfinite(args.lmbda) or args.lmbda <= 0:
        raise ValueError("--lmbda must be finite and positive")
    if tuple(sorted(set(args.resampling_steps))) != tuple(args.resampling_steps):
        raise ValueError("--resampling-steps must be unique and strictly increasing")
    if any(step < 1 or step >= EXPECTED_LOOPS for step in args.resampling_steps):
        raise ValueError("Intermediate FK resampling nodes must lie in [1, 50]; terminal node 51 is automatic")
    scheduler = PNDMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    scheduler.set_timesteps(DEFAULT_STEPS, device="cpu")
    if len(scheduler.timesteps) != EXPECTED_LOOPS:
        raise RuntimeError(f"DiFashion default 50 steps produced {len(scheduler.timesteps)} loops, expected 51")


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

    data = np.load(args.data_root / "processed" / "new_fitb_test.npy", allow_pickle=True).item()
    if not 0 <= args.task_index < len(data["uids"]):
        raise IndexError(f"task index outside 0..{len(data['uids']) - 1}: {args.task_index}")
    uid, oid = int(data["uids"][args.task_index]), int(data["oids"][args.task_index])
    task_id = f"{args.task_index:05d}_{uid}_{oid}"
    if args.expected_task_id and task_id != args.expected_task_id:
        raise ValueError(f"Task identity mismatch: expected {args.expected_task_id}, resolved {task_id}")
    outfit = np.asarray(data["outfits"][args.task_index], dtype=np.int64).reshape(-1)
    category = np.asarray(data["category"][args.task_index], dtype=np.int64).reshape(-1)
    input_ids = np.asarray(data["input_ids"][args.task_index], dtype=np.int64)
    blank = np.flatnonzero(outfit == 0)
    if len(outfit) != 4 or len(category) != 4 or len(blank) != 1:
        raise ValueError("Selected FITB condition is malformed")
    target_category = int(category[int(blank[0])])
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
    logger = logging.getLogger("difashion_fk")
    image_paths = np.load(args.data_root / "new_all_item_image_paths.npy", allow_pickle=True)
    history_latents = np.load(args.data_root / "processed" / "test_hist_latents.npy", allow_pickle=True).item()
    id_categories = np.load(args.data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
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
    scorer = PersonalReward(args.data_root, args.dualfashion_root, uid, target_category, device)

    particles = [{"slot": index, "initial_seed": seed, "ancestor_seed": seed, "lineage": [seed]}
                 for index, seed in enumerate(particle_seeds)]
    previous_rewards = [0.0] * args.num_particles
    log_products = [0.0] * args.num_particles
    resample_rng = random.Random(args.resampling_seed)
    trace: list[dict[str, Any]] = []
    schedules: list[dict[str, int]] = []

    def on_resample(loop_index, timestep, _latent_before, _noise_pred, x_prediction, _latent_after):
        step = loop_index + 1
        schedules.append({"loop_index_zero_based": int(loop_index), "step_number_one_based": step,
                          "scheduler_timestep": int(timestep.item())})
        if step not in set(args.resampling_steps) | {EXPECTED_LOOPS}:
            return None
        images = decode_batch(model, x_prediction)
        scores = scorer.score(images)
        node_dir = output / "nodes" / f"step_{step:02d}"
        node_dir.mkdir(parents=True, exist_ok=True)
        candidates = []
        for particle, image, score in zip(particles, images, scores):
            name = f"slot_{particle['slot']}_ancestor_{particle['ancestor_seed']}.png"
            image_path = node_dir / name
            image.save(image_path)
            candidates.append({**particle, "personal_sim": score, "reward_image": str(image_path.resolve())})

        potentials, normalized, ess = importance_weights(
            scores, previous_rewards, potential_type=args.potential_type, lmbda=args.lmbda
        )
        # The reference FKD corrects MAX/ADD terminal weights by the product
        # of earlier selected potentials. RT/DIFF use the usual local weights.
        if step == EXPECTED_LOOPS and args.potential_type in {"max", "add"}:
            normalized, ess = stable_weights_from_logits([
                args.lmbda * potential - log_product
                for potential, log_product in zip(potentials, log_products)
            ])
            weight_formula = "terminal softmax(lambda * potential - log_product_of_prior_selected_potentials)"
        else:
            weight_formula = "softmax(lambda * potential)"
        should_resample = step == EXPECTED_LOOPS or not args.adaptive_resampling or ess < args.num_particles / 2.0
        parents = list(range(args.num_particles))
        rng_state = None
        if should_resample:
            rng_state = repr(resample_rng.getstate())
            parents = list(resample_parent_indices(
                normalized, count=args.num_particles, method=args.resampling, rng=resample_rng
            ))
        trace.append({
            "step_number_one_based": step, "loop_index_zero_based": int(loop_index),
            "scheduler_timestep": int(timestep.item()), "node_kind": "terminal_forced_resample" if step == EXPECTED_LOOPS else "intermediate_resample",
            "reward": "personal", "reward_provenance": scorer.provenance,
            "potential_type": args.potential_type, "lmbda": args.lmbda, "weight_formula": weight_formula,
            "particles_before": candidates, "previous_rewards": list(previous_rewards),
            "potentials": list(potentials), "normalized_weights": list(normalized), "ess": ess,
            "adaptive_resampling": bool(args.adaptive_resampling), "did_resample": should_resample,
            "resampling": args.resampling if should_resample else None, "resampling_rng_state_before": rng_state,
            "parent_indices": parents,
        })
        if not should_resample:
            previous_rewards[:] = scores
            return None
        if args.potential_type in {"max", "add"} and step != EXPECTED_LOOPS:
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
        return parents

    try:
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                null_image = dataset[0].unsqueeze(0).to(device)
                null_latent = model.vae.encode(null_image).latent_dist.mode()[0] * model.vae.config.scaling_factor
                single_outfit_latents = model.vae.encode(
                    torch.stack([dataset[int(item)] for item in outfit]).to(device)
                ).latent_dist.mode() * model.vae.config.scaling_factor
            uid_t = torch.full((args.num_particles,), uid, dtype=torch.long, device=device)
            oid_t = torch.full((args.num_particles,), oid, dtype=torch.long, device=device)
            outfit_t = torch.as_tensor(outfit, dtype=torch.long, device=device).unsqueeze(0).repeat(args.num_particles, 1)
            category_t = torch.as_tensor(category, dtype=torch.long, device=device).unsqueeze(0).repeat(args.num_particles, 1)
            input_t = torch.as_tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0).repeat(args.num_particles, 1, 1)
            generators = [torch.Generator(device=device).manual_seed(seed) for seed in particle_seeds]
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                pipe_output, *_ = model.fashion_generation(
                    uid_t, oid_t, input_t, outfit_t, None, category_t, history_latents,
                    num_inference_steps=DEFAULT_STEPS,
                    category_guidance_scale=12.0, hist_guidance_scale=4.0, mutual_guidance_scale=5.0,
                    null_img=null_image[0], outfit_latents=single_outfit_latents.repeat(args.num_particles, 1, 1, 1),
                    null_latent=null_latent, generator=generators, output_type="pil", return_dict=True,
                    resample_callback=on_resample,
                )
        if len(schedules) != EXPECTED_LOOPS:
            raise RuntimeError(f"Expected {EXPECTED_LOOPS} loop calls, observed {len(schedules)}")
        if len(trace) != len(args.resampling_steps) + 1:
            raise RuntimeError(f"Expected {len(args.resampling_steps) + 1} FK nodes, observed {len(trace)}")
        final_dir = output / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        final_particles = []
        for particle, image in zip(particles, pipe_output.images):
            image_path = final_dir / f"slot_{particle['slot']}_ancestor_{particle['ancestor_seed']}.png"
            image.save(image_path)
            final_particles.append({**particle, "terminal_image": str(image_path.resolve()), "sha256": sha256(image_path)})
        intermediate_duplicates = []
        for node in trace[:-1]:
            for parent in sorted({index for index in node["parent_indices"] if node["parent_indices"].count(index) > 1}):
                children = [index for index, value in enumerate(node["parent_indices"]) if value == parent]
                child_hashes = [final_particles[index]["sha256"] for index in children]
                intermediate_duplicates.append({"resample_step": node["step_number_one_based"], "parent_slot": parent,
                                                "children": children, "terminal_hashes": child_hashes,
                                                "pixel_identical_terminal": len(set(child_hashes)) == 1})
        payload = {
            "schema": 1, "status": "complete", "task_id": task_id, "task_index": args.task_index,
            "uid": uid, "oid": oid, "condition": {"outfit": outfit.tolist(), "blank_index": int(blank[0]),
                                                       "target_category": target_category},
            "ground_truth_loaded_or_used": False,
            "method": "fk_steering_online_diagnostic", "reward": "personal",
            "test_status": "exploratory_single_task; FK parameters require validation freeze before a test claim",
            "particles": args.num_particles, "particle_seed_mapping": particle_seeds,
            "scheduler": {"class": "PNDMScheduler", "native_num_inference_steps_argument": DEFAULT_STEPS,
                          "actual_loop_calls": EXPECTED_LOOPS, "resampling_steps_one_based": args.resampling_steps,
                          "terminal_forced_resample_step_one_based": EXPECTED_LOOPS},
            "percent_node_mapping": {"20%": 10, "40%": 20, "60%": 31, "80%": 41, "100%": 51},
            "fk": {"potential_type": args.potential_type, "lmbda": args.lmbda, "resampling": args.resampling,
                   "adaptive_resampling": bool(args.adaptive_resampling), "resampling_seed": args.resampling_seed,
                   "reference_semantics": "FKD resamples K=4 with replacement at intermediate nodes and force-resamples at terminal node"},
            "checkpoint": str(checkpoint.resolve()), "base_model": str(args.base_model.resolve()),
            "guidance": {"category": 12.0, "mutual": 5.0, "history": 4.0, "eta": 0.1},
            "online_unet_forward_calls": args.num_particles * EXPECTED_LOOPS,
            "trace": trace, "scheduler_timestep_schedule": schedules, "final_particles": final_particles,
            "deterministic_duplicate_gate_local": {"duplicate_descendant_checks": intermediate_duplicates,
                "warning": "A duplicated parent with identical terminal hashes demonstrates deterministic continuation; one task is not the required validation-set gate."},
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
