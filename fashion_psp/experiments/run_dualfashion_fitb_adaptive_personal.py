"""Online development-tuned adaptive Personal PSP@8, serial denoising, exact112.

Uses the saved global heuristic parameters; no cached candidate scores/images.
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
from fashion_psp.experiments.search_adaptive_personal_development import Policy, decide_scalar

H, B = 28, 112


def online_search(policy, initial, advance, observe, checkpoint, resumed=None, notify=None):
    """Callbacks keep neural inference out of the planning and resume logic.

    advance(slot, latent, step, decode) -> next_latent, prediction_or_None.
    observe(predictions) -> {slot: Personal}; may replace predictions with RGB.
    Checkpoint is after a complete step.
    """
    state = resumed or {"states":initial, "completed_step":0, "spent":0,
                        "reward_views":0, "trace":[], "history":{},
                        "selected_slot":None, "selected_image":None}
    states = state["states"]
    history = {int(slot):list(values) for slot, values in state.get("history", {}).items()}
    for step in range(state["completed_step"] + 1, H + 1):
        before = sorted(states)
        spent = state["spent"] + len(before)
        # The selected policy evaluates every live candidate at every reached step.
        observe_now = True
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
            survivors = [winner]
            state["selected_slot"], state["selected_image"] = winner, images[winner]
        else:
            for slot in before:
                history.setdefault(slot, []).append(float(scores[slot]))
            survivors, removal_reasons, detail = decide_scalar(policy, history, step, spent)
        states = {slot:states[slot] for slot in survivors}
        history = {slot:history[slot] for slot in survivors if slot in history}
        state.update(states=states, history=history, spent=spent, completed_step=step)
        state["trace"].append({"step":step, "alive_before":before, "alive_after":survivors,
                               "observed":observe_now,
                               "personal_scores":{str(slot):float(scores[slot]) for slot in before} if observe_now else {},
                               "removed_reasons":({str(slot):"terminal_selection" for slot in before if slot not in survivors}
                                                  if step == H else removal_reasons),
                               "particle_steps_spent":spent,
                               "reward_views_cumulative":state["reward_views"],
                               **({} if step == H else detail)})
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


def parse_args(default_policy=None, default_output=None):
    p = argparse.ArgumentParser(description=__doc__)
    dual = ROOT / "DualFashion-main"
    p.add_argument("--dualfashion-root", type=Path, default=dual)
    p.add_argument("--data-root", type=Path, default=Path("C:/tmp/ifashion"))
    p.add_argument("--adaptive-policy", type=Path, default=default_policy or ROOT / "fashion_psp/artifacts/adaptive_personal_development_search_v2/selected_policy.json")
    p.add_argument("--output-root", type=Path, default=default_output or dual / "output/ifashion_fitb_adaptive_personal/personal_group0_seed2026")
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


def main(args=None):
    args = args or parse_args()
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

    selected_policy = json.loads(args.adaptive_policy.read_text(encoding="utf-8"))
    sparse = selected_policy.get("version") == "sparse_personal_v3"
    gap_budget = selected_policy.get("rule", {}).get("version") == "gap_budget_psp_v1"
    deferred_boundary = selected_policy.get("rule", {}).get("version") == "deferred_boundary_psp_v1"
    fully_adaptive = selected_policy.get("rule", {}).get("version") == "fully_adaptive_psp_v1"
    fixed_six = selected_policy.get("rule", {}).get("version") == "fixed_six_seed_7_21_v1"
    if gap_budget or deferred_boundary or fully_adaptive or fixed_six:
        if fixed_six:
            from fashion_psp.search.fixed_six_seed_psp import RULE
            from fashion_psp.experiments.run_dualfashion_fitb_six_seed_7_21 import online_search as search_impl
            rule_source = ROOT / "fashion_psp/search/fixed_six_seed_psp.py"
        elif fully_adaptive:
            from fashion_psp.search.fully_adaptive_psp import RULE
            from fashion_psp.experiments.run_dualfashion_fitb_fully_adaptive import online_search as search_impl
            rule_source = ROOT / "fashion_psp/search/fully_adaptive_psp.py"
        elif deferred_boundary:
            from fashion_psp.search.deferred_boundary_psp import RULE
            from fashion_psp.experiments.run_dualfashion_fitb_deferred_boundary import online_search as search_impl
            rule_source = ROOT / "fashion_psp/search/deferred_boundary_psp.py"
        else:
            from fashion_psp.search.gap_budget_psp import RULE
            from fashion_psp.experiments.run_dualfashion_fitb_gap_budget import online_search as search_impl
            rule_source = ROOT / "fashion_psp/search/gap_budget_psp.py"
        if selected_policy["rule"] != RULE or selected_policy.get("rule_source_sha256") != digest(rule_source):
            raise ValueError("Frozen research rule/source mismatch")
        if deferred_boundary and selected_policy.get("gap_gate_source_sha256") != digest(ROOT / "fashion_psp/search/gap_budget_psp.py"):
            raise ValueError("Inherited gate source mismatch")
        if fully_adaptive and selected_policy.get("online_source_sha256") != digest(ROOT / "fashion_psp/experiments/run_dualfashion_fitb_fully_adaptive.py"):
            raise ValueError("Frozen fully adaptive online source mismatch")
        if fixed_six and selected_policy.get("online_source_sha256") != digest(ROOT / "fashion_psp/experiments/run_dualfashion_fitb_six_seed_7_21.py"):
            raise ValueError("Frozen fixed6/7/21 online source mismatch")
        policy = RULE
        policy_parameters = RULE
    elif sparse:
        from fashion_psp.experiments.search_sparse_personal import SparsePolicy
        from fashion_psp.experiments.run_dualfashion_fitb_sparse_personal import online_search as search_impl
        policy = SparsePolicy(**selected_policy["parameters"])
        if not selected_policy.get("search", {}).get("all200_tuning"):
            raise ValueError("Unexpected sparse policy provenance")
        if policy != SparsePolicy(first=8, margin=1, protect=2, max_drop=8,
                                  near_interval=2, far_interval=2, uncertainty_gap=2):
            raise ValueError("Sparse policy differs from the reviewed online rule")
    elif selected_policy.get("version") == "development_tuned_v2":
        search_impl = online_search
        if selected_policy.get("dataset_role") != "all200_development_tuned_NOT_independent_test":
            raise ValueError("Unexpected adaptive policy provenance")
        policy = Policy(**selected_policy["parameters"])
        if policy != Policy(min_step=4, window=3, z=0, margin=0, protect_top=2,
                            max_drop=2, trend_horizon=0, mean_weight=0):
            raise ValueError("Policy differs from the reviewed online rule; use a new run configuration")
    else:
        raise ValueError("Unsupported adaptive policy artifact")
    if not (gap_budget or deferred_boundary or fully_adaptive or fixed_six):
        policy_parameters = selected_policy["parameters"]
    initial_particles = 6 if fixed_six else 8
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
    manifest = {"version":1, "method":("fixed_six_seed_7_21_personal_psp6" if fixed_six else "prototype_fully_adaptive_psp8" if fully_adaptive else "prototype_deferred_boundary_psp8" if deferred_boundary else "frozen_gap_budget_psp8" if gap_budget else "development_tuned_sparse_personal_psp8" if sparse else "development_tuned_adaptive_personal_psp8"), "policy_sha256":digest(args.adaptive_policy),
                "policy_path":str(args.adaptive_policy.resolve()), "policy_parameters":policy_parameters,
                "dataset_role":("user_specified_fixed_schedule_online_inference_exploratory_not_independent_test" if fixed_six else "research_prototype_not_real_test_evaluated" if deferred_boundary or fully_adaptive else "single_frozen_rule_previously_inspected_test_subset_not_independent_test" if gap_budget else "development_set_tuned_policy_online_inference_not_independent_test"),
                "seed":args.seed, "seed_stride":args.seed_stride,
                "particles":initial_particles, "steps":H, "budget":B, "deterministic_context":True,
                "guidance":{"text":8.0,"context":7.0,"preference":8.0},
                "device":args.device, "reward_device":args.reward_device, "cpu_offload":args.cpu_offload,
                "image_root":str(image_root.resolve()), "fitb":signature(fitb_path),
                "clip":signature(weights), "history":signature(history_path),
                "preferences":signature(info / "stage2_ifashion_test/test_preference.npy"),
                "model":str(model_path.resolve()), "precision":"bfloat16 on CUDA, float32 on CPU",
                "reward_pixels":"round-to-nearest uint8, same as cached torchvision PNGs",
                "reward_precision":"float32", "reward_batch_size":4, "reward_cudnn_tf32":True,
                "reward_padding":"repeat last sample to4; match original8-particle cache scoring",
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
    print(f"Prepared {total} FITB tasks; seed={args.seed}; K={initial_particles}; steps=28; budget=112.\nPolicy: {args.adaptive_policy}\nOutput: {output}", flush=True)
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
    clip.eval()  # Keep float32, matching the original cache reward scoring.

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
    with tqdm(total=total, initial=len(completed), desc=f"FITB Personal PSP@{initial_particles}", unit="task", smoothing=0.05) as progress:
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
                seeds = [args.seed + task * args.seed_stride + slot for slot in range(initial_particles)]
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

            result = search_impl(policy, initial, advance, observe, checkpoint, resumed,
                                   lambda step, alive, spent:progress.set_postfix(task=task, step=step, alive=alive, budget=f"{spent}/112", refresh=True))
            selected = folder / "selected.png"
            pixels(result["selected_image"]).save(selected)
            winner = result["selected_slot"]
            record = {"task_index":task, "uid":uid, "oid":oid, "category":category,
                      "candidate_seeds":seeds, "selected_seed":seeds[winner],
                      "selected_personal":result["trace"][-1]["personal_scores"][str(winner)],
                      "selected_image_path":str(selected.resolve()), "particle_steps":result["spent"],
                      "reward_views":result["reward_views"], "policy_sha256":manifest["policy_sha256"],
                      "scoring_rounds":sum(int(r["observed"]) for r in result["trace"]),
                      "clip_batches_padded4":sum((len(r["personal_scores"])+3)//4 for r in result["trace"] if r["observed"]),
                      "clip_image_slots_including_padding":sum(4*((len(r["personal_scores"])+3)//4) for r in result["trace"] if r["observed"]),
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
