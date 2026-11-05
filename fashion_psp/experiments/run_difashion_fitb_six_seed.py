"""Serial online Personal PSP@6 on native DiFashion: 13:6->4,39:4->2,51:2->1."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DiFashion-main/tools"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import UNet2DConditionModel
from diffusers.training_utils import EMAModel
from torchvision import transforms
from tqdm.auto import tqdm

from collect_ifashion_fitb_candidates import (
    DiFashion, MutualEncoder, data_utils, decode_latent, dump_json,
    make_model_args, preflight,
)
from fashion_psp.baselines.fk_steering.run_difashion_fitb_fk import PersonalReward

SCHEDULE = ((13, 4), (39, 2), (51, 1))
BUDGET = 6 * 13 + 4 * (39 - 13) + 2 * (51 - 39)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("D:/DiFashion"),
                   help="Parent directory containing ifashion/.")
    p.add_argument("--image-root", type=Path, default=Path("D:/DiFashion/ifashion/images"))
    p.add_argument("--checkpoint-root", type=Path, default=Path("D:/DiFashion"))
    p.add_argument("--base-model", type=Path,
                   default=ROOT / "DiFashion-main/pretrained_models/stable-diffusion-2-base")
    p.add_argument("--dualfashion-root", type=Path, default=ROOT / "DualFashion-main",
                   help="Local OpenCLIP dependency only; generator remains DiFashion.")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed-base", type=int, default=2026)
    p.add_argument("--resume-from", type=int, default=0, help="Zero-based FITB task index; skips completed tasks.")
    p.add_argument("--max-tasks", type=int, default=None, help="Process only the first N rows; omit for all tasks.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reward-device", default="cpu", help="CPU conserves VRAM; cuda:0 enables GPU scoring.")
    p.add_argument("--mixed-precision", choices=("fp16", "no"), default="fp16")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--audit-only", action="store_true")
    return p.parse_args()


def tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_cpu(v) for v in value]
    return value


def save_state(path, state):
    temp = path.with_suffix(".pt.tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def rank_scores(scores, keep):
    if not scores or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError("Reward must contain finite Personal scores for every live seed")
    return sorted(sorted(scores, key=lambda seed: (-scores[seed], seed))[:keep])


def target_for(history, task):
    uid = task["uid"]
    blank = int(np.flatnonzero(task["outfit"] == 0)[0])
    category = int(task["category"][blank])
    user = history.get(uid, history.get(str(uid), {}))
    value = user.get(category, user.get(str(category)))
    if value is None:
        raise ValueError(f"Missing Personal history: {task['task_id']}, category={category}")
    value = np.asarray(value, dtype=np.float32).reshape(1, -1)
    if value.shape != (1, 1024) or not np.isfinite(value).all() or np.linalg.norm(value) == 0:
        raise ValueError(f"Invalid Personal history vector: {task['task_id']}")
    return value, category


def signature(path):
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def resume_config_compatible(existing, current):
    if existing == current:
        return True
    # This exact legacy revision only compared order-sensitive private PNDM
    # metadata. Preserve its run/task hashes; record the repair separately.
    legacy_sources = {
        str(Path("fashion_psp/experiments/run_difashion_fitb_six_seed.py")):
            "52483baa31e87ca935084e1102570e3b1df4c0ffff21a2530ff80e78fe4c1d40",
        str(Path("DiFashion-main/DiFashion/models/difashion.py")):
            "49523355241dc24dc2420b0e8fad57d2672249bf85dd7110c0b47bffaff681bd",
    }
    model_key = str(Path("DiFashion-main/DiFashion/models/difashion.py"))
    repaired_model_hash = "ff322d2ad0cc50f0a4d33cf3e1eb96fd6022019f79f00d78fed167ae7eaf77cf"
    return (existing.get("source_sha256") == legacy_sources
            and set(current.get("source_sha256", {})) == set(legacy_sources)
            and current["source_sha256"][model_key] == repaired_model_hash
            and {k: v for k, v in existing.items() if k != "source_sha256"}
            == {k: v for k, v in current.items() if k != "source_sha256"})


def verify_result(folder, task, config_hash):
    summary_path = folder / "task_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary["task_id"] != task["task_id"] or summary["config_sha256"] != config_hash
            or summary["model_forward_calls"] != BUDGET):
        raise ValueError(f"Completed task identity/config/budget mismatch: {folder}")
    if (summary["selected_seed"] != summary["trace"][-1]["seeds_after"][0]
            or len(summary["trace"]) != len(SCHEDULE)):
        raise ValueError(f"Invalid completed task trace: {folder}")
    for node, (step, keep) in zip(summary["trace"], SCHEDULE):
        if node["step"] != step or len(node["seeds_after"]) != keep:
            raise ValueError(f"Invalid completed node: {folder}, step={step}")
        scores = {r["seed"]: r["personal"] for r in node["candidates"]}
        if rank_scores(scores, keep) != node["seeds_after"]:
            raise ValueError(f"Saved score/selection mismatch: {folder}, step={step}")
        for row in node["candidates"]:
            with Image.open(folder / row["image"]) as image:
                image.verify()
    with Image.open(folder / "selected.png") as image:
        image.verify()
    final = folder / summary["trace"][-1]["candidates"][0]["image"]
    for row in summary["trace"][-1]["candidates"]:
        if row["seed"] == summary["selected_seed"]:
            final = folder / row["image"]
    if (folder / "selected.png").read_bytes() != final.read_bytes():
        raise ValueError(f"Selected image does not match winning terminal image: {folder}")
    return summary


def audit(output_root, tasks, config_hash):
    missing, completed = [], []
    for task in tasks:
        try:
            row = verify_result(output_root / task["task_id"], task, config_hash)
            if row is None:
                raise FileNotFoundError("Missing task_summary.json")
            completed.append(row)
        except Exception as exc:
            missing.append({"task_id": task["task_id"], "error": repr(exc)})
    report = {"complete": not missing, "expected_tasks": len(tasks), "completed_tasks": len(completed),
              "missing_or_invalid_tasks": missing, "actual_online_particle_steps_per_task": BUDGET,
              "model_forward_calls_completed_tasks": sum(r["model_forward_calls"] for r in completed)}
    dump_json(output_root / "missing_report.json", report)
    dump_json(output_root / "manifest.json", {"config_sha256": config_hash, "tasks": completed})
    return report


def main():
    args = parse_args()
    args.num_inference_steps = 50  # Native default; never substitute the loop count 51 here.
    data, all_tasks, _ = preflight(args)
    tasks = all_tasks[:args.max_tasks] if args.max_tasks is not None else all_tasks
    if args.resume_from >= len(tasks):
        raise ValueError("--resume-from must be within the selected task range")
    if args.seed_base < 0 or args.seed_base + (len(all_tasks) - 1) * 10000 + 5 >= 2**63:
        raise ValueError("Seed mapping outside supported nonnegative int64 range")
    history_path = args.data_root / "ifashion/processed/test_history_clipembs.npy"
    personal_history = np.load(history_path, allow_pickle=True).item()
    for task in tasks:
        target_for(personal_history, task)
    checkpoint = args.checkpoint_root / "checkpoint-15000"
    asset_paths = [args.data_root / "ifashion/processed/new_fitb_test.npy", history_path,
                   args.data_root / "ifashion/processed/test_hist_latents.npy",
                   args.data_root / "ifashion/new_all_item_image_paths.npy",
                   args.data_root / "ifashion/new_id_cate_dict.npy",
                   checkpoint / "unet_ema/diffusion_pytorch_model.bin",
                   checkpoint / "fashion_encoder_ema/diffusion_pytorch_model.bin"]
    config = {
        "method": "difashion_serial_online_personal_psp6_quarter_threequarter_v1",
        "native_inference_argument": 50, "actual_loop_calls": args.actual_denoising_loop_calls,
        "percentage_mapping": "ceil(actual_PNDM_loop_count * fraction); fractions=0.25,0.75,1.0",
        "schedule": SCHEDULE, "initial_particles": 6, "online_forward_calls_per_task": BUDGET,
        "seed_mapping": f"{args.seed_base} + original_task_index*10000 + slot[0..5]",
        "total_dataset_tasks": len(data["uids"]), "base_model": str(args.base_model.resolve()),
        "data_root": str(args.data_root.resolve()), "image_root": str(args.image_root.resolve()),
        "dualfashion_dependency": str(args.dualfashion_root.resolve()),
        "checkpoint": str(checkpoint.resolve()), "assets": [signature(p) for p in asset_paths],
        "device": args.device, "reward_device": args.reward_device, "precision": args.mixed_precision,
        "reward": "100*cosine(OpenCLIP ViT-H-14/laion2b-s32b-b79K image, category user history)",
        "tie_break": "descending current Personal, ascending original seed",
        "node_images": "13/39: pre-update x_prediction; 51: terminal scheduler latent decode",
        "continuation": "per-seed x_t, prev_latents, PNDM ets/cur_sample/cur_model_output/counter",
        "gt_access": False, "experiment_status": "exploratory test run; schedule specified by user",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "DiFashion-main/DiFashion/models/difashion.py")},
        "python": sys.executable, "torch": torch.__version__,
    }
    config = json.loads(json.dumps(config))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if not resume_config_compatible(existing, config):
            raise ValueError("Existing run configuration differs; use a new output directory")
        if existing != config:
            dump_json(args.output_root / "resume_compatibility_fix.json", {
                "reason": "Ignore private Diffusers scheduler metadata ordering; public parameters and timestep schedule remain strict",
                "original_sources": existing["source_sha256"], "current_sources": config["source_sha256"],
                "original_run_config_preserved": True,
            })
            config_hash = hashlib.sha256(json.dumps(existing, sort_keys=True).encode()).hexdigest()
            print("RESUME_COMPATIBILITY_FIX: accepted audited scheduler metadata repair; preserved existing task hashes", flush=True)
    elif any(args.output_root.iterdir()):
        raise FileExistsError("Nonempty output directory has no run_config.json")
    else:
        dump_json(config_path, config)
    if args.preflight_only:
        print(json.dumps({"preflight": "ok", "dataset_tasks": len(all_tasks), "selected_tasks": len(tasks),
                          "native_argument": 50, "loop_calls": 51, "schedule": SCHEDULE,
                          "forward_calls_per_task": BUDGET, "output": str(args.output_root)}, indent=2))
        return 0
    if args.audit_only:
        report = audit(args.output_root, tasks, config_hash)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 2
    pending = [task for task in tasks if task["task_index"] >= args.resume_from
               and verify_result(args.output_root / task["task_id"], task, config_hash) is None]
    if not pending:
        report = audit(args.output_root, tasks, config_hash)
        print(json.dumps(report, indent=2))
        return 0 if report["complete"] else 2
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    logging.basicConfig(level=logging.INFO)
    model = DiFashion(make_model_args(args), logging.getLogger("difashion_online_psp6"),
                      len(np.load(args.data_root / "ifashion/new_id_cate_dict.npy", allow_pickle=True).item()),
                      device).to(device)
    model.vae.enable_slicing()
    for subfolder, cls, target in (("unet_ema", UNet2DConditionModel, model.unet),
                                    ("fashion_encoder_ema", MutualEncoder, model.fashion_encoder)):
        ema = EMAModel.from_pretrained(str(checkpoint / subfolder), cls)
        ema.copy_to(target.parameters())
        del ema
    model.eval()
    first_target, first_category = target_for(personal_history, pending[0])
    scorer = PersonalReward(args.data_root / "ifashion", args.dualfashion_root,
                            pending[0]["uid"], first_category, torch.device(args.reward_device))
    image_paths = np.load(args.data_root / "ifashion/new_all_item_image_paths.npy", allow_pickle=True)
    dataset = data_utils.ImagePathDataset(str(args.image_root), image_paths, transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512), transforms.ToTensor(),
    ]), do_normalize=True)
    latent_history = np.load(args.data_root / "ifashion/processed/test_hist_latents.npy", allow_pickle=True).item()
    progress = tqdm(tasks, desc="FITB online PSP6", unit="task")
    with torch.inference_mode():
        for task in progress:
            folder = args.output_root / task["task_id"]
            if task["task_index"] < args.resume_from or verify_result(folder, task, config_hash) is not None:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            state_path = folder / "resume_state.pt"
            seeds = [args.seed_base + task["task_index"] * 10000 + slot for slot in range(6)]
            state = {"config_sha256": config_hash, "task_id": task["task_id"], "node_index": 0,
                     "alive": seeds, "particles": {}, "trace": [], "calls": 0, "runtime_seconds": 0.0}
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                if state["config_sha256"] != config_hash or state["task_id"] != task["task_id"]:
                    raise ValueError(f"Resume state identity/config mismatch: {state_path}")
            target, _ = target_for(personal_history, task)
            scorer.target = F.normalize(torch.as_tensor(target, device=scorer.device), dim=-1)
            uid = torch.tensor([task["uid"]], device=device)
            oid = torch.tensor([task["oid"]], device=device)
            outfit = torch.tensor(task["outfit"], device=device).unsqueeze(0)
            categories = torch.tensor(task["category"], device=device).unsqueeze(0)
            ids = torch.tensor(task["input_ids"], device=device).unsqueeze(0)
            pixels = torch.stack([dataset[int(i)] for i in task["outfit"]]).to(device)
            def autocast():
                return torch.autocast(device.type, dtype=torch.float16,
                                      enabled=device.type == "cuda" and args.mixed_precision == "fp16")
            with autocast():
                outfit_latents = model.vae.encode(pixels).latent_dist.mode() * model.vae.config.scaling_factor
            null_latent = outfit_latents[int(np.flatnonzero(task["outfit"] == 0)[0])]
            actual_calls = [0]
            hook = model.unet.register_forward_hook(lambda *_: actual_calls.__setitem__(0, actual_calls[0] + 1))
            try:
                while state["node_index"] < len(SCHEDULE):
                    end, keep = SCHEDULE[state["node_index"]]
                    node = folder / f"step_{end}"
                    node.mkdir(exist_ok=True)
                    for seed in state["alive"]:
                        previous = state["particles"].get(seed)
                        if previous is not None and previous["completed_step"] == end:
                            continue
                        began = time.perf_counter()
                        before_calls = actual_calls[0]
                        with autocast():
                            result = model.fashion_generation(
                                uid, oid, ids, outfit, None, categories, latent_history,
                                category_guidance_scale=12.0, hist_guidance_scale=4.0, mutual_guidance_scale=5.0,
                                outfit_latents=outfit_latents, null_latent=null_latent,
                                generator=torch.Generator(device=device).manual_seed(seed),
                                output_type="sampling_state", sampling_state=previous, stop_after_step=end,
                            )
                            image = decode_latent(model, result["latents"] if end == 51 else result["x_prediction"])
                        expected = end - (previous["completed_step"] if previous else 0)
                        observed = actual_calls[0] - before_calls
                        if observed != expected:
                            raise AssertionError(f"Forward counter mismatch: {observed} != {expected}")
                        image.save(node / f"seed_{seed}.png")
                        state["particles"][seed] = tree_cpu(result)
                        state["calls"] += observed
                        state["runtime_seconds"] += time.perf_counter() - began
                        save_state(state_path, state)
                        del result, image
                        progress.set_postfix(task=task["task_index"], node=end, seed=seed, calls=state["calls"])
                    rows = []
                    began = time.perf_counter()
                    for seed in state["alive"]:
                        relative = f"step_{end}/seed_{seed}.png"
                        with Image.open(folder / relative) as image:
                            score = scorer.score([image])[0]
                        particle = state["particles"][seed]
                        rows.append({"seed": seed, "personal": score, "image": relative,
                                     "scheduler_timestep": particle["scheduler_timestep"],
                                     "loop_index": end - 1})
                    survivors = rank_scores({r["seed"]: r["personal"] for r in rows}, keep)
                    state["runtime_seconds"] += time.perf_counter() - began
                    state["trace"].append({"step": end, "seeds_before": state["alive"],
                                           "seeds_after": survivors, "candidates": rows,
                                           "scoring_image": "terminal_latent" if end == 51 else "x_prediction",
                                           "cumulative_forward_calls": state["calls"]})
                    state["alive"] = survivors
                    state["particles"] = {s: state["particles"][s] for s in survivors}
                    state["node_index"] += 1
                    save_state(state_path, state)
                if state["calls"] != BUDGET:
                    raise AssertionError(f"Expected {BUDGET} actual forward calls, got {state['calls']}")
                selected_seed = state["alive"][0]
                (folder / "selected.png").write_bytes((folder / "step_51" / f"seed_{selected_seed}.png").read_bytes())
                summary = {"task_id": task["task_id"], "task_index": task["task_index"],
                           "uid": task["uid"], "oid": task["oid"], "config_sha256": config_hash,
                           "initial_seeds": seeds, "selected_seed": selected_seed, "trace": state["trace"],
                           "model_forward_calls": state["calls"], "runtime_sampling_and_scoring_seconds": state["runtime_seconds"],
                           "status": "complete", "gt_used": False}
                dump_json(folder / "task_summary.json", summary)
                verify_result(folder, task, config_hash)
                state_path.unlink()
                print(f"TASK_COMPLETE {task['task_id']} seed={selected_seed} actual_forward_calls={state['calls']}", flush=True)
            except Exception as exc:
                dump_json(folder / "error.json", {"task_id": task["task_id"], "error": repr(exc)})
                raise
            finally:
                hook.remove()
    report = audit(args.output_root, tasks, config_hash)
    print("RUN_COMPLETE" if report["complete"] else "RUN_INCOMPLETE", json.dumps(report))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
