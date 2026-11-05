#!/usr/bin/env python
"""Replay one DiFashion FITB seed and score its PSP decision nodes.

This diagnostic tool is deliberately separate from the immutable BoN/PSP
candidate cache.  It generates one fixed seed on one FITB condition, saves
the step-13 and step-26 decoded x_predictions plus the terminal scheduler
decode, and computes the frozen Personal Sim score for each image.  It never
loads test_grd.npy, retrieval candidates, or any ground-truth image.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import PNDMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from PIL import Image
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DiFashion"))
import data_utils  # noqa: E402
from models.difashion import DiFashion, MutualEncoder  # noqa: E402


NODE_STEPS = (13, 26)
DEFAULT_STEPS = 50
EXPECTED_LOOPS = 51


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="iFashion directory containing processed/.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=("fp16", "no"), default="fp16")
    parser.add_argument("--num-inference-steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def model_args(base_model: Path) -> SimpleNamespace:
    return SimpleNamespace(
        pretrained_model_name_or_path=str(base_model), revision=None, non_ema_revision=None,
        enable_xformers_memory_efficient_attention=False, category_emb_size=64, hid_dim=256,
        use_history=True, use_mutual_guidance=True, eta=0.1, noise_offset=0.0, snr_gamma=None,
    )


def decode_latent(model: DiFashion, latent: torch.Tensor) -> Image.Image:
    decoded = model.vae.decode(latent / model.vae.config.scaling_factor, return_dict=False)[0]
    return model.image_processor.postprocess(decoded, output_type="pil", do_denormalize=[True])[0]


def personal_score(image_path: Path, uid: int, category: int, data_root: Path,
                   dualfashion_root: Path, device: torch.device) -> tuple[float, dict]:
    history_path = data_root / "processed" / "test_history_clipembs.npy"
    if not history_path.is_file():
        raise FileNotFoundError(f"Missing Personal Sim history: {history_path}")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(dualfashion_root.resolve()))
    import open_clip

    history = np.load(history_path, allow_pickle=True).item()
    user_history = history.get(uid, history.get(str(uid)))
    if user_history is None:
        raise KeyError(f"No history for uid={uid}")
    target = user_history.get(category, user_history.get(str(category)))
    if target is None:
        raise KeyError(f"No target-category history for uid={uid}, category={category}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b-s32b-b79K")
    model = model.to(device).eval()
    with Image.open(image_path) as image:
        pixels = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        candidate = model.encode_image(pixels).float()
        history_embedding = torch.as_tensor(np.asarray(target, dtype=np.float32).reshape(-1), device=device).unsqueeze(0)
        score = 100.0 * F.cosine_similarity(
            F.normalize(candidate, dim=-1), F.normalize(history_embedding, dim=-1), dim=-1
        ).item()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(score), {
        "formula": "100 * cosine(normalized OpenCLIP(candidate), normalized test_history_clipembs[uid][target_category])",
        "openclip": "ViT-H-14/laion2b-s32b-b79K",
        "personal_history": str(history_path.resolve()),
    }


def main() -> int:
    args = parse_args()
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
    if args.num_inference_steps != DEFAULT_STEPS:
        raise ValueError(f"Only DiFashion's default {DEFAULT_STEPS} inference steps are supported")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    scheduler = PNDMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    scheduler.set_timesteps(args.num_inference_steps, device="cpu")
    if len(scheduler.timesteps) != EXPECTED_LOOPS:
        raise RuntimeError(f"Expected {EXPECTED_LOOPS} PNDM loop calls, got {len(scheduler.timesteps)}")

    data = np.load(args.data_root / "processed" / "new_fitb_test.npy", allow_pickle=True).item()
    if not 0 <= args.task_index < len(data["uids"]):
        raise IndexError(f"task index {args.task_index} is outside 0..{len(data['uids']) - 1}")
    uid, oid = int(data["uids"][args.task_index]), int(data["oids"][args.task_index])
    outfit = np.asarray(data["outfits"][args.task_index], dtype=np.int64).reshape(-1)
    category = np.asarray(data["category"][args.task_index], dtype=np.int64).reshape(-1)
    input_ids = np.asarray(data["input_ids"][args.task_index], dtype=np.int64)
    blank = np.flatnonzero(outfit == 0)
    if len(outfit) != 4 or len(category) != 4 or len(blank) != 1:
        raise ValueError("Selected FITB condition is malformed")
    task_id = f"{args.task_index:05d}_{uid}_{oid}"
    target_category = int(category[int(blank[0])])
    output = args.output_root / task_id / f"seed_{args.seed}"
    result_path = output / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if args.resume:
            print(json.dumps(existing, indent=2))
            return 0
        raise FileExistsError(f"Existing replay result: {result_path}; use --resume or another --output-root")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Incomplete replay output exists: {output}")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("single_seed_replay")
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
    # EMAModel.to() mutates in place and returns None in this diffusers release.
    ema_unet.to(device)
    ema_encoder.to(device)
    ema_unet.copy_to(model.unet.parameters())
    ema_encoder.copy_to(model.fashion_encoder.parameters())
    model.eval(); model.unet.eval(); model.fashion_encoder.eval()

    output.mkdir(parents=True, exist_ok=False)
    captures: dict[str, dict] = {}
    schedules: list[dict] = []
    try:
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                null_image = dataset[0].unsqueeze(0).to(device)
                null_latent = model.vae.encode(null_image).latent_dist.mode()[0] * model.vae.config.scaling_factor
                outfit_images = torch.stack([dataset[int(item)] for item in outfit]).to(device)
                outfit_latents = model.vae.encode(outfit_images).latent_dist.mode() * model.vae.config.scaling_factor
            uid_t = torch.tensor([uid], dtype=torch.long, device=device)
            oid_t = torch.tensor([oid], dtype=torch.long, device=device)
            outfit_t = torch.tensor([outfit.tolist()], dtype=torch.long, device=device)
            category_t = torch.tensor(category, dtype=torch.long, device=device).unsqueeze(0)
            input_t = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

            def on_step(loop_index, timestep, latent_before, noise_pred, x_prediction, latent_after):
                step = loop_index + 1
                schedules.append({"loop_index_zero_based": int(loop_index), "step_number_one_based": step,
                                  "scheduler_timestep": int(timestep.item())})
                if step not in NODE_STEPS:
                    return
                image_path = output / f"step_{step:02d}_x_prediction.png"
                decode_latent(model, x_prediction).save(image_path)
                captures[str(step)] = {
                    "node": f"reward_step_{step:02d}", "image_path": str(image_path.resolve()),
                    "loop_index_zero_based": int(loop_index), "scheduler_timestep": int(timestep.item()),
                    "x_prediction_saved_as_decoded_image": True,
                }

            generator = torch.Generator(device=device).manual_seed(args.seed)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda" and args.mixed_precision == "fp16"):
                result, _ = model.fashion_generation(
                    uid_t, oid_t, input_t, outfit_t, None, category_t, history_latents,
                    num_inference_steps=args.num_inference_steps,
                    category_guidance_scale=12.0, hist_guidance_scale=4.0, mutual_guidance_scale=5.0,
                    null_img=null_image[0], outfit_latents=outfit_latents, null_latent=null_latent,
                    generator=generator, output_type="pil", return_dict=False, step_callback=on_step,
                )
            if len(schedules) != EXPECTED_LOOPS or set(captures) != {str(step) for step in NODE_STEPS}:
                raise RuntimeError(f"Unexpected replay trace: loops={len(schedules)}, captured={sorted(captures)}")
            final_path = output / "final_scheduler_decode.png"
            result[uid][oid]["images"][0].save(final_path)
    except Exception:
        # Preserve the output for debugging without ever overwriting it on retry.
        raise
    finally:
        del model, ema_unet, ema_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    nodes = [captures[str(step)] for step in NODE_STEPS]
    nodes.append({"node": "final", "image_path": str(final_path.resolve()),
                  "loop_index_zero_based": EXPECTED_LOOPS - 1, "scheduler_timestep": 1,
                  "x_prediction_saved_as_decoded_image": False, "terminal_scheduler_decode": True})
    for node in nodes:
        score, provenance = personal_score(Path(node["image_path"]), uid, target_category,
                                           args.data_root, args.dualfashion_root, device)
        node["personal_sim"] = score
        node["score_provenance"] = provenance
    payload = {
        "schema": 1, "status": "complete", "task_index": args.task_index, "task_id": task_id,
        "uid": uid, "oid": oid, "seed": args.seed, "condition_source": str((args.data_root / "processed" / "new_fitb_test.npy").resolve()),
        "condition": {"outfit": outfit.tolist(), "blank_index": int(blank[0]), "target_category": target_category},
        "ground_truth_loaded_or_used": False,
        "checkpoint": str(checkpoint.resolve()), "base_model": str(args.base_model.resolve()),
        "scheduler": {"class": "PNDMScheduler", "num_inference_steps_argument": DEFAULT_STEPS,
                      "actual_denoising_loop_calls": EXPECTED_LOOPS, "captured_nodes": [13, 26, "final"]},
        "guidance": {"category_guidance_scale": 12.0, "mutual_guidance_scale": 5.0, "hist_guidance_scale": 4.0, "eta": 0.1},
        "nodes": nodes, "scheduler_timestep_schedule": schedules,
    }
    write_json(result_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
