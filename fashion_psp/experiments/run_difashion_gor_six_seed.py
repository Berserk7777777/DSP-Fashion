"""Serial online Personal PSP@6 for DiFashion iFashion GOR.

Each particle is one complete four-item outfit.  The native DiFashion
inference argument is fixed at 50; its audited PNDM scheduler performs 51
model-loop calls.  Particles are pruned at loop 13 (6 -> 4), loop 39
(4 -> 2), and loop 51 (2 -> 1).  Ground truth is never loaded by this
runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DiFashion-main/tools"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import PNDMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from collect_ifashion_fitb_candidates import (  # noqa: E402
    DiFashion,
    MutualEncoder,
    data_utils,
)
from fashion_psp.baselines.fk_steering.run_difashion_gor_fk import (  # noqa: E402
    GORPersonalReward,
)

SCHEDULE = ((13, 4), (39, 2), (51, 1))
NATIVE_INFERENCE_STEPS = 50
EXPECTED_LOOP_CALLS = 51
INITIAL_PARTICLES = 6
ITEMS_PER_OUTFIT = 4
SEED_STRIDE = 10000
BUDGET = 6 * 13 + 4 * (39 - 13) + 2 * (51 - 39)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("D:/DiFashion"),
                        help="Parent directory containing ifashion/.")
    parser.add_argument("--image-root", type=Path, default=Path("D:/DiFashion/ifashion/images"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("D:/DiFashion"))
    parser.add_argument("--base-model", type=Path,
                        default=ROOT / "DiFashion-main/pretrained_models/stable-diffusion-2-base")
    parser.add_argument("--dualfashion-root", type=Path, default=ROOT / "DualFashion-main",
                        help="Local OpenCLIP package root used by Personal reward.")
    parser.add_argument("--personal-clip-weights", type=Path, default=Path(
        r"C:/Users/Administrator/.cache/huggingface/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/"
        r"snapshots/1c2b8495b28150b8a4922ee1c8edee224c284c0c/open_clip_model.safetensors"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=2026)
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Zero-based task index. Completed tasks are skipped.")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Use the first N GOR rows, for example 200.")
    parser.add_argument("--num-inference-steps", type=int, default=NATIVE_INFERENCE_STEPS,
                        help="Must remain DiFashion's native default 50.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reward-device", default="cpu")
    parser.add_argument("--mixed-precision", choices=("fp16", "no"), default="fp16")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tree_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_cpu(item) for item in value]
    return value


def save_state(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_tasks(data: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = []
    for index, (uid, oid, category, input_ids) in enumerate(
        zip(data["uids"], data["oids"], data["category"], data["input_ids"])
    ):
        categories = np.asarray(category, dtype=np.int64).reshape(-1)
        ids = np.asarray(input_ids, dtype=np.int64)
        if len(categories) != ITEMS_PER_OUTFIT or ids.shape[0] != ITEMS_PER_OUTFIT:
            raise ValueError(f"GOR row {index} must contain four categories and four prompts")
        tasks.append({
            "task_index": index,
            "task_id": f"{index:05d}_{int(uid)}_{int(oid)}",
            "task_dir_name": f"{int(uid)}_{int(oid)}",
            "uid": int(uid),
            "oid": int(oid),
            "category": categories,
            "input_ids": ids,
        })
    names = [task["task_dir_name"] for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError("GOR uid_oid output directory names are not unique")
    return tasks


def seed_map(task_count: int, seed_base: int) -> dict[int, list[int]]:
    used: dict[int, tuple[int, int]] = {}
    mapping = {}
    for task_index in range(task_count):
        seeds = [seed_base + task_index * SEED_STRIDE + slot for slot in range(INITIAL_PARTICLES)]
        for slot, seed in enumerate(seeds):
            if seed < 0 or seed >= 2**63 or seed in used:
                raise ValueError(f"Invalid or colliding GOR seed {seed}: {used.get(seed)} vs {(task_index, slot)}")
            used[seed] = (task_index, slot)
        mapping[task_index] = seeds
    return mapping


def preflight(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, list[int]]]:
    if args.num_inference_steps != NATIVE_INFERENCE_STEPS:
        raise ValueError(f"DiFashion GOR runner fixes --num-inference-steps to {NATIVE_INFERENCE_STEPS}")
    if args.resume_from < 0:
        raise ValueError("--resume-from must be non-negative")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError("--max-tasks must be positive")
    if args.reward_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Reward CUDA device unavailable: {args.reward_device}")
    data_root = args.data_root / "ifashion"
    checkpoint = args.checkpoint_root / "checkpoint-15000"
    required = [
        data_root / "processed/new_fitb_test.npy",
        data_root / "processed/test_hist_latents.npy",
        data_root / "processed/test_history_clipembs.npy",
        data_root / "new_id_cate_dict.npy",
        data_root / "new_all_item_image_paths.npy",
        args.image_root / "empty_image.png",
        args.personal_clip_weights,
        checkpoint / "unet_ema/diffusion_pytorch_model.bin",
        checkpoint / "fashion_encoder_ema/diffusion_pytorch_model.bin",
        args.base_model / "scheduler/scheduler_config.json",
        args.base_model / "unet/config.json",
        args.base_model / "vae/config.json",
        args.base_model / "text_encoder/config.json",
        args.base_model / "tokenizer/tokenizer_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))
    scheduler = PNDMScheduler.from_pretrained(args.base_model, subfolder="scheduler")
    scheduler.set_timesteps(NATIVE_INFERENCE_STEPS, device="cpu")
    actual_calls = len(scheduler.timesteps)
    if actual_calls != EXPECTED_LOOP_CALLS:
        raise RuntimeError(f"Native DiFashion argument 50 produced {actual_calls} loops, expected {EXPECTED_LOOP_CALLS}")
    data = np.load(data_root / "processed/new_fitb_test.npy", allow_pickle=True).item()
    tasks = build_tasks(data)
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    if not tasks:
        raise ValueError("No GOR tasks selected")
    if args.resume_from > len(tasks):
        raise ValueError(f"--resume-from {args.resume_from} is past selected task count {len(tasks)}")
    mapping = seed_map(len(data["uids"]), args.seed_base)
    return data, tasks, mapping


def make_model_args(base_model: Path) -> SimpleNamespace:
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


def save_outfit(images: list[Image.Image], directory: Path) -> list[str]:
    if len(images) != ITEMS_PER_OUTFIT:
        raise ValueError(f"Expected four GOR images, got {len(images)}")
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite candidate directory: {directory}")
    directory.mkdir(parents=True, exist_ok=False)
    names = []
    for index, image in enumerate(images):
        name = f"item_{index}.png"
        image.save(directory / name)
        names.append(name)
    width, height = images[0].size
    merged = Image.new("RGB", (2 * width, 2 * height), color=(255, 255, 255))
    for index, image in enumerate(images):
        merged.paste(image.convert("RGB"), ((index % 2) * width, (index // 2) * height))
    merged.save(directory / "all.png")
    names.append("all.png")
    return names


def rank_scores(rows: list[dict[str, Any]], keep: int) -> list[int]:
    if len(rows) < keep or not all(math.isfinite(float(row["personal"])) for row in rows):
        raise ValueError("Personal reward must be finite for every live GOR particle")
    ordered = sorted(rows, key=lambda row: (-float(row["personal"]), int(row["seed"])))
    return [int(row["seed"]) for row in ordered[:keep]]


def set_personal_targets(scorer: GORPersonalReward, history: dict[Any, Any], task: dict[str, Any]) -> None:
    user = history.get(task["uid"], history.get(str(task["uid"])))
    if user is None:
        raise KeyError(f"Missing Personal history for uid={task['uid']}")
    values = []
    for category in task["category"].tolist():
        value = user.get(int(category), user.get(str(int(category))))
        if value is None:
            raise KeyError(f"Missing Personal history for {task['task_id']}, category={category}")
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
            raise ValueError(f"Invalid Personal history vector for {task['task_id']}, category={category}")
        values.append(vector)
    targets = F.normalize(torch.as_tensor(np.stack(values), dtype=torch.float32, device=scorer.device), dim=-1)
    scorer.targets = targets
    scorer.provenance = {
        **scorer.provenance,
        "uid": task["uid"],
        "categories": [int(value) for value in task["category"]],
    }


def verify_image(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()


def verify_result(folder: Path, task: dict[str, Any], config_hash: str) -> dict[str, Any] | None:
    summary_path = folder / "task_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("task_id") != task["task_id"]:
        raise ValueError(f"Invalid GOR task summary identity/status: {summary_path}")
    if summary.get("config_sha256") != config_hash or summary.get("model_forward_calls") != BUDGET:
        raise ValueError(f"GOR task summary config/budget mismatch: {summary_path}")
    trace = summary.get("trace", [])
    if len(trace) != len(SCHEDULE):
        raise ValueError(f"GOR task has {len(trace)} nodes, expected {len(SCHEDULE)}: {folder}")
    for node, (step, keep) in zip(trace, SCHEDULE):
        if node.get("step") != step or len(node.get("seeds_after", [])) != keep:
            raise ValueError(f"GOR node mismatch at step {step}: {folder}")
        if rank_scores(node["candidates"], keep) != node["seeds_after"]:
            raise ValueError(f"GOR saved ranking mismatch at step {step}: {folder}")
        for candidate in node["candidates"]:
            for relative in candidate["image_paths"]:
                verify_image(folder / relative)
    selected = folder / "selected"
    for index in range(ITEMS_PER_OUTFIT):
        verify_image(selected / f"item_{index}.png")
    verify_image(selected / "all.png")
    return summary


def make_config(args: argparse.Namespace, all_task_count: int, tasks: list[dict[str, Any]],
                assets: list[Path]) -> dict[str, Any]:
    source_files = [Path(__file__), ROOT / "DiFashion-main/DiFashion/models/difashion.py",
                    ROOT / "fashion_psp/baselines/fk_steering/run_difashion_gor_fk.py"]
    return {
        "schema": 1,
        "method": "difashion_serial_online_personal_gor_psp6_25_75_100_v1",
        "dataset": "iFashion",
        "split": "test",
        "task": "GOR",
        "task_count_from_processed_rows": all_task_count,
        "selected_task_indices": [task["task_index"] for task in tasks],
        "native_inference_steps_argument": NATIVE_INFERENCE_STEPS,
        "actual_denoising_loop_calls": EXPECTED_LOOP_CALLS,
        "schedule": [{"step": step, "keep": keep} for step, keep in SCHEDULE],
        "percentage_interpretation": "13=25% and 39=75% of the 51 actual PNDM loops; 51=terminal",
        "initial_particles": INITIAL_PARTICLES,
        "particle_definition": "one seed owns one four-item outfit; Personal is the mean of four item scores",
        "online_particle_step_budget_per_task": BUDGET,
        "item_unet_forward_equivalent_per_task": BUDGET * ITEMS_PER_OUTFIT,
        "seed_mapping": "seed_base + original_task_index*10000 + slot[0..5]",
        "seed_base": args.seed_base,
        "checkpoint": str((args.checkpoint_root / "checkpoint-15000").resolve()),
        "base_model": str(args.base_model.resolve()),
        "data_root": str(args.data_root.resolve()),
        "image_root": str(args.image_root.resolve()),
        "dualfashion_dependency": str(args.dualfashion_root.resolve()),
        "personal_reward_weights": str(args.personal_clip_weights.resolve()),
        "assets": [signature(path) for path in assets],
        "reward": "mean_j(100*cosine(OpenCLIP ViT-H-14 generated item_j, test_history_clipembs[uid][category_j]))",
        "reward_device": args.reward_device,
        "tie_break": "descending four-item mean Personal, then ascending seed",
        "node_image_semantics": {
            "13": "decoded x_prediction before scheduler update at loop 13; score only",
            "39": "decoded x_prediction before scheduler update at loop 39; score only",
            "51": "decoded terminal scheduler latent after loop 51; final score",
        },
        "continuation": "per-seed latents, prev_latents, initial_latents, PNDM ets/cur_sample/cur_model_output/counter",
        "ground_truth_loaded_or_used": False,
        "output_layout": "<task_id>/step_13|step_39|step_51/seed_<seed>/{item_0..3,all}.png plus selected/",
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_files},
        "experiment_status": "exploratory test run; reward and pruning schedule fixed before this run",
    }


def audit(output_root: Path, tasks: list[dict[str, Any]], config_hash: str) -> dict[str, Any]:
    missing = []
    completed = []
    for task in tasks:
        try:
            row = verify_result(output_root / task["task_id"], task, config_hash)
            if row is None:
                raise FileNotFoundError("missing task_summary.json")
            completed.append(row)
        except Exception as exc:
            missing.append({"task_id": task["task_id"], "error": repr(exc)})
    report = {
        "complete": not missing,
        "expected_tasks": len(tasks),
        "completed_tasks": len(completed),
        "missing_or_invalid_tasks": missing,
        "online_particle_step_budget_per_task": BUDGET,
        "completed_online_particle_step_budget": sum(row["model_forward_calls"] for row in completed),
    }
    atomic_json(output_root / "missing_report.json", report)
    atomic_json(output_root / "manifest.json", {"config_sha256": config_hash, "tasks": completed})
    return report


def main() -> int:
    args = parse_args()
    data, tasks, seeds_by_task = preflight(args)
    all_task_count = len(data["uids"])
    checkpoint = args.checkpoint_root / "checkpoint-15000"
    assets = [
        args.data_root / "ifashion/processed/new_fitb_test.npy",
        args.data_root / "ifashion/processed/test_hist_latents.npy",
        args.data_root / "ifashion/processed/test_history_clipembs.npy",
        args.data_root / "ifashion/new_id_cate_dict.npy",
        args.data_root / "ifashion/new_all_item_image_paths.npy",
        checkpoint / "unet_ema/diffusion_pytorch_model.bin",
        checkpoint / "fashion_encoder_ema/diffusion_pytorch_model.bin",
        args.base_model / "scheduler/scheduler_config.json",
        args.personal_clip_weights,
    ]
    config = make_config(args, all_task_count, tasks, assets)
    config_text = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_text.encode("utf-8")).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "run_config.json"
    if config_path.is_file():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Existing run_config.json differs; use a new output directory or the same parameters")
    elif any(args.output_root.iterdir()):
        raise FileExistsError(f"Non-empty output directory has no run_config.json: {args.output_root}")
    else:
        atomic_json(config_path, config)

    if args.preflight_only:
        print(json.dumps({
            "preflight": "ok",
            "total_dataset_tasks": all_task_count,
            "selected_tasks": len(tasks),
            "selected_task_indices": [task["task_index"] for task in tasks],
            "native_inference_steps": NATIVE_INFERENCE_STEPS,
            "actual_loop_calls": EXPECTED_LOOP_CALLS,
            "schedule": SCHEDULE,
            "online_particle_step_budget_per_task": BUDGET,
            "output_root": str(args.output_root.resolve()),
        }, indent=2))
        return 0
    if args.audit_only:
        report = audit(args.output_root, tasks, config_hash)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 2

    pending = [task for task in tasks
               if task["task_index"] >= args.resume_from
               and verify_result(args.output_root / task["task_id"], task, config_hash) is None]
    if not pending:
        report = audit(args.output_root, tasks, config_hash)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("difashion_online_gor_psp6")
    data_root = args.data_root / "ifashion"
    history_latents = np.load(data_root / "processed/test_hist_latents.npy", allow_pickle=True).item()
    personal_history = np.load(data_root / "processed/test_history_clipembs.npy", allow_pickle=True).item()
    id_categories = np.load(data_root / "new_id_cate_dict.npy", allow_pickle=True).item()
    image_paths = np.load(data_root / "new_all_item_image_paths.npy", allow_pickle=True)
    image_dataset = data_utils.ImagePathDataset(
        str(args.image_root), image_paths,
        transforms.Compose([transforms.Resize(512), transforms.CenterCrop(512), transforms.ToTensor()]),
        do_normalize=True,
    )

    model = DiFashion(make_model_args(args.base_model), logger, len(id_categories), device).to(device)
    model.vae.enable_slicing()
    for subfolder, cls, target in (("unet_ema", UNet2DConditionModel, model.unet),
                                   ("fashion_encoder_ema", MutualEncoder, model.fashion_encoder)):
        ema = EMAModel.from_pretrained(str(checkpoint / subfolder), cls)
        ema.copy_to(target.parameters())
        del ema
    model.eval()
    model.unet.eval()
    model.fashion_encoder.eval()

    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16,
        enabled=device.type == "cuda" and args.mixed_precision == "fp16",
    ):
        null_image = image_dataset[0].unsqueeze(0).to(device)
        null_latent = model.vae.encode(null_image).latent_dist.mode()[0] * model.vae.config.scaling_factor
    null_latent = null_latent.detach()
    gor_outfit_latents = null_latent.repeat(ITEMS_PER_OUTFIT, 1, 1, 1)

    first = pending[0]
    scorer = GORPersonalReward(
        args.data_root, args.dualfashion_root, args.personal_clip_weights,
        first["uid"], first["category"], torch.device(args.reward_device),
    )
    progress = tqdm(tasks, desc="GOR online PSP6", unit="task")
    with torch.inference_mode():
        for task in progress:
            folder = args.output_root / task["task_id"]
            if task["task_index"] < args.resume_from:
                continue
            try:
                if verify_result(folder, task, config_hash) is not None:
                    continue
            except Exception:
                raise
            folder.mkdir(parents=True, exist_ok=True)
            state_path = folder / "resume_state.pt"
            if state_path.is_file():
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                if state.get("config_sha256") != config_hash or state.get("task_id") != task["task_id"]:
                    raise ValueError(f"Resume state identity/config mismatch: {state_path}")
            elif any(path.name not in {"error.json"} for path in folder.iterdir()):
                raise FileExistsError(f"Incomplete GOR task has no resume_state.pt: {folder}")
            else:
                state = {
                    "config_sha256": config_hash,
                    "task_id": task["task_id"],
                    "node_index": 0,
                    "alive": seeds_by_task[task["task_index"]],
                    "particles": {},
                    "trace": [],
                    "calls": 0,
                    "runtime_seconds": 0.0,
                }
            set_personal_targets(scorer, personal_history, task)
            uid = torch.tensor([task["uid"]], device=device)
            oid = torch.tensor([task["oid"]], device=device)
            categories = torch.as_tensor(task["category"], device=device).unsqueeze(0)
            input_ids = torch.as_tensor(task["input_ids"], device=device).unsqueeze(0)
            empty_outfit = torch.zeros((1, ITEMS_PER_OUTFIT), dtype=torch.long, device=device)
            actual_calls = [0]
            hook = model.unet.register_forward_hook(lambda *_: actual_calls.__setitem__(0, actual_calls[0] + 1))
            try:
                while state["node_index"] < len(SCHEDULE):
                    end, keep = SCHEDULE[state["node_index"]]
                    for seed in list(state["alive"]):
                        previous = state["particles"].get(seed)
                        candidate_dir = folder / f"step_{end}" / f"seed_{seed}"
                        if previous is not None and previous["completed_step"] == end:
                            if not candidate_dir.is_dir():
                                raise FileNotFoundError(f"Missing saved node images for resumed seed: {candidate_dir}")
                            continue
                        started = time.perf_counter()
                        before_calls = actual_calls[0]
                        with torch.autocast(
                            device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda" and args.mixed_precision == "fp16",
                        ):
                            result = model.fashion_generation(
                                uid, oid, input_ids, empty_outfit, None, categories, history_latents,
                                num_inference_steps=NATIVE_INFERENCE_STEPS,
                                category_guidance_scale=12.0, hist_guidance_scale=4.0,
                                mutual_guidance_scale=5.0, outfit_latents=gor_outfit_latents,
                                null_latent=null_latent,
                                generator=torch.Generator(device=device).manual_seed(int(seed)),
                                output_type="sampling_state", sampling_state=previous,
                                stop_after_step=end,
                            )
                            score_latents = result["latents"] if end == EXPECTED_LOOP_CALLS else result["x_prediction"]
                            images = decode_batch(model, score_latents)
                        observed = actual_calls[0] - before_calls
                        expected = end - (previous["completed_step"] if previous else 0)
                        if observed != expected:
                            raise AssertionError(f"Forward count mismatch for {task['task_id']} seed={seed}: {observed} != {expected}")
                        save_outfit(images, candidate_dir)
                        state["particles"][seed] = tree_cpu(result)
                        state["calls"] += observed
                        state["runtime_seconds"] += time.perf_counter() - started
                        save_state(state_path, state)
                        del result, images, score_latents
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        progress.set_postfix(task=task["task_index"], node=end, seed=seed, calls=state["calls"])

                    scoring_started = time.perf_counter()
                    rows = []
                    for seed in state["alive"]:
                        candidate_dir = folder / f"step_{end}" / f"seed_{seed}"
                        images = []
                        for index in range(ITEMS_PER_OUTFIT):
                            with Image.open(candidate_dir / f"item_{index}.png") as image:
                                images.append(image.convert("RGB").copy())
                        outfit_scores, item_scores = scorer.score(images)
                        particle = state["particles"][seed]
                        rows.append({
                            "seed": int(seed),
                            "personal": float(outfit_scores[0]),
                            "personal_items": [float(value) for value in item_scores[0]],
                            "image_dir": str(candidate_dir.relative_to(folder)),
                            "image_paths": [str((candidate_dir / f"item_{index}.png").relative_to(folder)) for index in range(ITEMS_PER_OUTFIT)]
                            + [str((candidate_dir / "all.png").relative_to(folder))],
                            "scheduler_timestep": int(particle["scheduler_timestep"]),
                            "loop_index_zero_based": end - 1,
                            "completed_step": end,
                            "prediction_kind": "terminal_latent" if end == EXPECTED_LOOP_CALLS else "x_prediction",
                        })
                    survivors = rank_scores(rows, keep)
                    state["runtime_seconds"] += time.perf_counter() - scoring_started
                    state["trace"].append({
                        "step": end,
                        "seeds_before": [int(seed) for seed in state["alive"]],
                        "seeds_after": survivors,
                        "candidates": rows,
                        "tie_break": "personal descending, seed ascending",
                        "scoring_uses_ground_truth": False,
                        "cumulative_particle_step_calls": state["calls"],
                    })
                    state["alive"] = survivors
                    state["particles"] = {int(seed): state["particles"][seed] for seed in survivors}
                    state["node_index"] += 1
                    save_state(state_path, state)

                if state["calls"] != BUDGET:
                    raise AssertionError(f"Expected {BUDGET} particle-step calls, got {state['calls']}")
                selected_seed = int(state["alive"][0])
                selected_source = folder / "step_51" / f"seed_{selected_seed}"
                selected_dir = folder / "selected"
                selected_names = ("item_0.png", "item_1.png", "item_2.png", "item_3.png", "all.png")
                if selected_dir.exists():
                    missing_selected = [name for name in selected_names if not (selected_dir / name).is_file()]
                    if missing_selected:
                        raise FileExistsError(
                            f"Partial selected directory cannot be repaired automatically: {selected_dir}; "
                            f"missing={missing_selected}"
                        )
                else:
                    selected_dir.mkdir(exist_ok=False)
                    for name in selected_names:
                        shutil.copy2(selected_source / name, selected_dir / name)
                summary = {
                    "status": "complete",
                    "task": "GOR",
                    "task_index": task["task_index"],
                    "task_id": task["task_id"],
                    "uid": task["uid"],
                    "oid": task["oid"],
                    "categories": [int(value) for value in task["category"]],
                    "config_sha256": config_hash,
                    "initial_seeds": seeds_by_task[task["task_index"]],
                    "selected_seed": selected_seed,
                    "trace": state["trace"],
                    "model_forward_calls": state["calls"],
                    "item_unet_forward_equivalent": state["calls"] * ITEMS_PER_OUTFIT,
                    "runtime_sampling_and_scoring_seconds": state["runtime_seconds"],
                    "reward_provenance": scorer.provenance,
                    "ground_truth_loaded_or_used": False,
                }
                atomic_json(folder / "task_summary.json", summary)
                verify_result(folder, task, config_hash)
                state_path.unlink(missing_ok=True)
                (folder / "error.json").unlink(missing_ok=True)
                print(f"TASK_COMPLETE {task['task_id']} selected_seed={selected_seed} particle_step_calls={state['calls']}", flush=True)
            except Exception as exc:
                atomic_json(folder / "error.json", {"task_id": task["task_id"], "error": repr(exc)})
                raise
            finally:
                hook.remove()

    report = audit(args.output_root, tasks, config_hash)
    print("RUN_COMPLETE" if report["complete"] else "RUN_INCOMPLETE", json.dumps(report, ensure_ascii=True), flush=True)
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
