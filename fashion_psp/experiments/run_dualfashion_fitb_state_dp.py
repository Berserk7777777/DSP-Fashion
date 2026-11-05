"""Online Personal state-risk DP@8, serial denoising, exact112, step resume.

Uses a frozen planning table; never looks up cached candidate scores/images.
Only final selected PNGs and compact decision traces are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fashion_psp.evaluation.research_dualfashion_state_risk_dp import StateRiskDP, B, H


def needs_scores(policy, count, step, remaining):
    feasible = [m for m in range(1, count + 1) if remaining >= m
                and np.isfinite(policy.value[step + 1, m, remaining - m]).all()]
    if not feasible:
        raise ValueError("No exact-budget continuation")
    return feasible != [count]


def online_search(policy, initial, advance, observe, checkpoint, resumed=None, notify=None):
    """Callbacks keep neural inference out of the planning and resume logic.

    advance(slot, latent, step, decode) -> next_latent, prediction_or_None.
    observe(predictions) -> {slot: Personal}; may replace predictions with RGB.
    Checkpoint is after a complete step.
    """
    state = resumed or {"states":initial, "completed_step":0, "spent":0,
                        "reward_views":0, "trace":[], "selected_slot":None, "selected_image":None}
    states = state["states"]
    width = max(initial) + 1
    for step in range(state["completed_step"] + 1, H + 1):
        before = sorted(states)
        spent = state["spent"] + len(before)
        observe_now = step == H or needs_scores(policy, len(before), step, B - spent)
        images = {}
        for slot in before:
            states[slot], image = advance(slot, states[slot], step, observe_now)
            if observe_now:
                images[slot] = image
        scores = observe(images) if observe_now else {slot:0.0 for slot in before}
        if set(scores) != set(before) or not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("Reward must provide one finite score per live particle")
        state["reward_views"] += len(before) if observe_now else 0
        if step == H:
            if spent != B:
                raise AssertionError(f"Expected112 particle steps, got {spent}")
            winner = min(before, key=lambda slot:(-scores[slot], slot))
            survivors, loss, support = [winner], 0.0, 0.0
            state["selected_slot"], state["selected_image"] = winner, images[winner]
        else:
            current = np.zeros((1, width))
            mask = np.zeros((1, width), bool)
            current[0, before] = [scores[slot] for slot in before]
            mask[0, before] = True
            new_mask, loss_array, support_array, _ = policy.decide(current, mask, step, np.array([B - spent]))
            survivors = np.flatnonzero(new_mask[0]).tolist()
            loss, support = float(loss_array[0]), float(support_array[0])
        states = {slot:states[slot] for slot in survivors}
        state.update(states=states, spent=spent, completed_step=step)
        state["trace"].append({"step":step, "alive_before":before, "alive_after":survivors,
                               "observed":observe_now,
                               "personal_scores":{str(slot):float(scores[slot]) for slot in before} if observe_now else {},
                               "estimated_pruning_loss":loss, "weighted_support":support,
                               "particle_steps_spent":spent})
        checkpoint(state)
        if notify:
            notify(step, len(survivors), spent)
    return state


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(path):
    stat = path.stat()
    return {"path":str(path.resolve()), "bytes":stat.st_size, "mtime_ns":stat.st_mtime_ns}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    dual = ROOT / "DualFashion-main"
    p.add_argument("--dualfashion-root", type=Path, default=dual)
    p.add_argument("--data-root", type=Path, default=Path("C:/tmp/ifashion"))
    p.add_argument("--policy", type=Path, default=dual / "output/ifashion_fitb_state_risk_dp/planning_statistics_all_tasks.npz")
    p.add_argument("--output-root", type=Path, default=dual / "output/ifashion_fitb_state_dp/personal_group0_seed2026")
    p.add_argument("--model-name", default="0057501")
    p.add_argument("--max-samples", type=int, default=2207)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--seed-stride", type=int, default=10000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reward-device", default="cuda:0", help="Reward model is offloaded to CPU between scoring events.")
    p.add_argument("--clip-weights", type=Path, default=None)
    p.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume-from", default="auto", help="auto, none, or zero-based task index; resumes saved completed steps")
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.max_samples < 1 or args.seed_stride < 8:
        raise ValueError("max-samples must be positive; seed-stride must reserve8 seeds")
    resume = args.resume_from.strip().lower()
    start = 0 if resume in ("auto", "none") else int(resume)
    if start < 0:
        raise ValueError("resume-from must be nonnegative")
    sys.path.insert(0, str(args.dualfashion_root))
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm
    from torchvision import transforms as T
    from Inference_ifashion_PFITB import (
        compute_stage2_context_offset, ensure_stage2_modules, load_image_tensor,
        load_numpy_dict, restore_stage2_parameters, save_inference_payloads,
    )
    from run_ifashion_gor_greedy_personal import make_prompt_config, predict_velocity, PERSONAL_CLIP_WEIGHTS
    from run_psp_ifashion_smoke import decode_latent, estimate_x0_from_flowmatch

    policy = StateRiskDP.load(args.policy)
    if policy.nmax < 8:
        raise ValueError("Planning table must support8 particles")
    fitb_path = args.data_root / "processed/fitb_test.npy"
    fitb = load_numpy_dict(str(fitb_path))
    total = min(args.max_samples, len(fitb["uids"]))
    if start >= total:
        raise ValueError("resume-from outside requested task range")
    info = args.dualfashion_root / "processed_info"
    paths = np.load(info / "all_item_image_paths.npy", allow_pickle=True)
    image_root = args.data_root / "semantic_category"
    weights = args.clip_weights or PERSONAL_CLIP_WEIGHTS
    model_path = args.dualfashion_root / "checkpoint" / args.model_name
    history_path = args.data_root / "processed/test_history_clipembs.npy"
    manifest = {"version":2, "method":"state_risk_dp8", "policy_sha256":digest(args.policy),
                "policy_path":str(args.policy.resolve()), "seed":args.seed, "seed_stride":args.seed_stride,
                "particles":8, "steps":H, "budget":B, "deterministic_context":True,
                "guidance":{"text":8.0,"context":7.0,"preference":8.0},
                "device":args.device, "reward_device":args.reward_device, "cpu_offload":args.cpu_offload,
                "image_root":str(image_root.resolve()), "fitb":signature(fitb_path),
                "clip":signature(weights), "history":signature(history_path),
                "preferences":signature(info / "stage2_ifashion_test/test_preference.npy"),
                "model":str(model_path.resolve()), "precision":"bfloat16 on CUDA, float32 on CPU",
                "reward_pixels":"round-to-nearest uint8, same as cached torchvision PNGs",
                "reward_precision":"float32", "reward_batch_size":4, "reward_cudnn_tf32":True,
                "reward_padding":"repeat last sample to4; match original8-particle calibration",
                "text_generation":False}
    output = args.output_root.resolve()
    manifest_path = output / "run_config.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Output configuration differs; choose a new output-root")
        if resume == "none" and any(output.glob("tasks/*/task_summary.json")):
            raise FileExistsError("Completed outputs exist; use --resume-from auto or a new directory")
    atomic_json(manifest_path, manifest)
    # Validate every observed context source before a long generation run.
    for task in range(start, total):
        for item in fitb["outfits"][task]:
            if int(item) and not (image_root / str(paths[int(item)])).is_file():
                raise FileNotFoundError(image_root / str(paths[int(item)]))
    print(f"Prepared {total} FITB tasks; seed={args.seed}; K=8; steps=28; budget=112.\nPolicy: {args.policy}\nOutput: {output}", flush=True)
    if args.prepare_only:
        return

    device, reward_device = torch.device(args.device), torch.device(args.reward_device)
    if (device.type == "cuda" or reward_device.type == "cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this Python environment")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    ground = load_numpy_dict(str(args.data_root / "test_grd_dict.npy"))
    categories = load_numpy_dict(str(args.data_root / "id_cate_dict.npy"))
    preferences = load_numpy_dict(str(info / "stage2_ifashion_test/test_preference.npy"))
    history = load_numpy_dict(str(history_path))
    args.max_length = 256
    args.negative_prompt = "low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame"
    args.text_guidance_scale, args.context_guidance_scale, args.preference_guidance_scale = 8.0, 7.0, 8.0
    # Imports of neural models are delayed until after argument/data validation.
    from transformers import T5TokenizerFast
    from sd3_modules.stage2_pipeline import DualDiffSD3Pipeline
    from safetensors.torch import load_file
    import open_clip

    tokenizer = T5TokenizerFast(tokenizer_file=str(model_path / "tokenizer/tokenizer.json"), extra_ids=100)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    pipe = DualDiffSD3Pipeline.from_pretrained(str(model_path), tokenizer=tokenizer, torch_dtype=dtype)
    ensure_stage2_modules(pipe.transformer, pipe.vae.config.latent_channels)
    restore_stage2_parameters(pipe.transformer, str(model_path))
    pipe.transformer.eval()
    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.set_sampling_mode("t2i")
    if args.cpu_offload and device.type == "cuda":
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    clip.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    clip.eval()  # Keep float32, matching the risk calibration reward.

    def offload_generator():
        for hook in getattr(pipe, "_all_hooks", []):
            hook.offload()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def pixels(image):
        rgb = ((image.detach().float().cpu().clamp(-1, 1) + 1) * 127.5 + 0.5).clamp(0, 255).byte()
        return Image.fromarray(rgb.permute(1, 2, 0).numpy())

    checkpoint_root = output / args.model_name
    gen, grd = {}, {}
    if resume != "none" and (checkpoint_root / "gen.npy").is_file():
        gen = load_numpy_dict(str(checkpoint_root / "gen.npy"))
        grd = load_numpy_dict(str(checkpoint_root / "grd.npy"))
    transform = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(512), T.ToTensor(), T.Normalize([0.5], [0.5])])

    def task_folder(task):
        return output / "tasks" / f"task{task:04d}_{int(fitb['uids'][task])}_{int(fitb['oids'][task])}"

    def finished(task):
        path = task_folder(task) / "task_summary.json"
        if not path.is_file():
            return False
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["task_index"] != task or record["policy_sha256"] != manifest["policy_sha256"]:
            raise ValueError("Completed task identity/policy mismatch")
        return Path(record["selected_image_path"]).is_file()

    completed = {task for task in range(total) if resume != "none" and finished(task)}

    def pack(record):
        task, uid, oid = record["task_index"], record["uid"], record["oid"]
        incomplete = [int(x) for x in fitb["outfits"][task]]
        blank = incomplete.index(0)
        full = np.asarray(ground["outfits"][task], dtype=np.int64)
        gt_path = image_root / str(paths[int(full[blank])])
        if not gt_path.is_file():
            raise FileNotFoundError(gt_path)
        context_paths = [str((image_root / str(paths[x])).resolve()) for x in incomplete if x]
        common = {"cates":np.array([record["category"]]), "incomplete_outfit":context_paths, "outfits":full}
        gen.setdefault(uid, {})[oid] = {**common, "image_paths":[record["selected_image_path"]],
                                      "candidate_seeds":[record["selected_seed"]]}
        grd.setdefault(uid, {})[oid] = {**common, "image_paths":[str(gt_path.resolve())]}

    # Rebuild packed records if interrupted after committing a task summary.
    for task in completed:
        pack(json.loads((task_folder(task) / "task_summary.json").read_text(encoding="utf-8")))
    save_inference_payloads(str(checkpoint_root), gen, grd)
    atomic_json(output / "resume_state.json", {"completed_indices":sorted(completed), "completed_count":len(completed), "total_tasks":total})
    with tqdm(total=total, initial=len(completed), desc="FITB state-DP@8", unit="task", smoothing=0.05) as progress:
        for task in range(start, total):
            if task in completed:
                continue
            started = time.perf_counter()
            uid, oid = int(fitb["uids"][task]), int(fitb["oids"][task])
            incomplete = [int(x) for x in fitb["outfits"][task]]
            blank = incomplete.index(0)
            category = int(fitb["category"][task][blank])
            prompt = make_prompt_config(pipe, uid, category, categories, preferences, args, device)
            with torch.inference_mode():
                images = torch.stack([load_image_tensor(x, paths, str(image_root), transform) for x in incomplete if x]).to(device)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=dtype):
                    dist = pipe.vae.encode(images).latent_dist
                    context = ((dist.mode() - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor).unsqueeze(0).to(dtype=dtype)
                    offset = compute_stage2_context_offset(pipe.transformer, context)
                seeds = [args.seed + task * args.seed_stride + slot for slot in range(8)]
                initial = {slot:torch.randn((1, *context.shape[2:]), generator=torch.Generator(device=device).manual_seed(seed), device=device, dtype=dtype)
                           for slot, seed in enumerate(seeds)}
            folder = task_folder(task)
            folder.mkdir(parents=True, exist_ok=True)
            state_path = folder / "resume_state.pt"
            identity = {"config":manifest, "task":task, "uid":uid, "oid":oid,
                        "conditioning_sha256":hashlib.sha256(b"".join(x.detach().float().cpu().contiguous().numpy().tobytes() for x in [offset, prompt['uncond'], prompt['text'], prompt['preference']])).hexdigest()}
            resumed = None
            elapsed_before = 0.0
            if resume != "none" and state_path.is_file():
                saved = torch.load(state_path, map_location="cpu", weights_only=False)
                if saved["identity"] != identity:
                    raise ValueError("Step resume conditioning/configuration mismatch")
                resumed = saved["state"]
                resumed["states"] = {int(slot):value.to(device=device, dtype=dtype) for slot, value in resumed["states"].items()}
                elapsed_before = saved["elapsed_s"]
                tqdm.write(f"Resume task {task}: step {resumed['completed_step']}, alive={len(resumed['states'])}")
            scheduler = pipe.scheduler
            scheduler.set_timesteps(H, device=device)
            target = history[uid].get(category, history[uid].get(str(category)))
            if target is None:
                raise KeyError(f"Missing Personal history: {uid}/{category}")
            target = F.normalize(torch.as_tensor(target, dtype=torch.float32, device=reward_device), dim=0)

            @torch.inference_mode()
            def advance(slot, latent, step, decode):
                timestep = scheduler.timesteps[step - 1].to(device)
                velocity = predict_velocity(pipe, latent, timestep, prompt, offset, args)
                prediction = estimate_x0_from_flowmatch(scheduler, velocity, latent, step - 1) if decode else None
                scheduler._step_index = step - 1
                after = scheduler.step(velocity, timestep, latent, return_dict=False)[0].detach()
                return after, prediction

            @torch.inference_mode()
            def observe(decoded):
                # Finish all denoiser calls before serial VAE decoding so CPU
                # offload does not reload the Transformer for every image.
                for slot, prediction in list(decoded.items()):
                    decoded[slot] = decode_latent(pipe, prediction)[0]
                offload_generator()
                clip.to(reward_device)
                old_tf32 = torch.backends.cudnn.allow_tf32
                torch.backends.cudnn.allow_tf32 = True
                try:
                    scores = {}
                    slots = sorted(decoded)
                    for start_at in range(0, len(slots), 4):
                        group = slots[start_at:start_at + 4]
                        padded = group + [group[-1]] * (4 - len(group))
                        # Original timestep scores used float32 batches of4.
                        # On this GPU batch1 selects a different convolution
                        # precision path, even for identical PNG pixels.
                        batch = torch.stack([preprocess(pixels(decoded[slot])) for slot in padded]).to(reward_device)
                        feature = F.normalize(clip.encode_image(batch).float(), dim=-1)
                        values = 100 * (feature @ target)
                        for index, slot in enumerate(group):
                            scores[slot] = float(values[index].item())
                    return scores
                finally:
                    torch.backends.cudnn.allow_tf32 = old_tf32
                    clip.to("cpu")
                    if reward_device.type == "cuda":
                        torch.cuda.empty_cache()

            def checkpoint(state):
                cpu_state = {**state, "states":{slot:value.detach().cpu() for slot, value in state["states"].items()}}
                temp = state_path.with_suffix(".tmp.pt")
                torch.save({"identity":identity, "state":cpu_state, "elapsed_s":elapsed_before + time.perf_counter() - started}, temp)
                temp.replace(state_path)

            result = online_search(policy, initial, advance, observe, checkpoint, resumed,
                                   lambda step, alive, spent:progress.set_postfix(task=task, step=step, alive=alive, budget=f"{spent}/112", refresh=True))
            selected = folder / "selected.png"
            pixels(result["selected_image"]).save(selected)
            winner = result["selected_slot"]
            record = {"task_index":task, "uid":uid, "oid":oid, "category":category,
                      "candidate_seeds":seeds, "selected_seed":seeds[winner],
                      "selected_personal":result["trace"][-1]["personal_scores"][str(winner)],
                      "selected_image_path":str(selected.resolve()), "particle_steps":result["spent"],
                      "reward_views":result["reward_views"], "policy_sha256":manifest["policy_sha256"],
                      "elapsed_s":elapsed_before + time.perf_counter() - started, "trace":result["trace"]}
            atomic_json(folder / "task_summary.json", record)
            pack(record)
            save_inference_payloads(str(checkpoint_root), gen, grd)
            completed.add(task)
            atomic_json(output / "resume_state.json", {"completed_indices":sorted(completed), "completed_count":len(completed), "total_tasks":total})
            state_path.unlink(missing_ok=True)
            offload_generator()
            progress.update(1)
    print(f"Saved final selections and evaluation payloads: {checkpoint_root}", flush=True)


if __name__ == "__main__":
    main()
