"""Single-task Rewrite-PSP@8: shared noise, immutable preferences, Personal reward.

By default task 395 uses noise seed 2026 + 395*10000, matching its original
cache's first particle. Choose field-order controls or coherent history modes.
Selection uses fixed history Personal. Reports do not compare against GT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fashion_psp.conditioning.preference_rewrite import ATTRIBUTE_KEYS, build_order_rewrites, build_history_rewrites


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dualfashion-root", type=Path, default=ROOT / "DualFashion-main")
    p.add_argument("--data-root", type=Path, default=Path(r"C:\tmp\ifashion"))
    p.add_argument("--task-index", type=int, default=395)
    p.add_argument("--seed", type=int, default=2026, help="Base seed; task index times stride is added.")
    p.add_argument("--seed-stride", type=int, default=10000)
    p.add_argument("--noise-seed", type=int, default=None, help="Override the resolved shared noise seed.")
    p.add_argument("--output-dir", type=Path, default=ROOT / "DualFashion-main/output/ifashion_fitb_rewrite/task395_9674_18277")
    p.add_argument("--reference-cache", type=Path, default=None)
    p.add_argument("--rewrite-mode", choices=("order", "history"), default="order")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--reward-device", default="cpu", help="CPU keeps frozen ViT-H/14 out of inference VRAM.")
    p.add_argument("--clip-weights", type=Path, default=None)
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Validate attributes/tokens and save manifest without model inference.")
    p.add_argument("--audit-all-rewrites", action="store_true", help="After online PSP, independently generate all eight full trajectories; audit cost is separate.")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted step or return completed metadata.")
    return p.parse_args()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def latent_hash(value):
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def save_reward_image(value, path):
    """Save the exact uint8 pixels used by the native Personal scorer.

    torchvision.save_image rounds while the native scorer truncates. Keeping
    that one-level difference out of our artifacts makes PNG rescoring agree.
    """
    from PIL import Image
    pixels = ((value.detach().float().cpu().clamp(-1,1)+1.0)*127.5).byte().permute(1,2,0).numpy()
    Image.fromarray(pixels,mode="RGB").save(path)


def gallery(manifest, metrics, output):
    from PIL import Image, ImageDraw
    size, label_h = 220, 50
    canvas = Image.new("RGB", (size * 8, (size + label_h) * 4 + 36), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{manifest['uid']}_{manifest['oid']} | {manifest['rewrite_kind']} | shared noise {manifest['noise_seed']}", fill="black")
    for row, step in enumerate((7, 14, 28)):
        for column in range(8):
            entries = [x for x in metrics if x["rewrite_id"] == column and x["step"] == step and x["phase"] == "psp"]
            entry = entries[0] if entries else None
            x, y = column * size, 36 + row * (size + label_h)
            if entry:
                with Image.open(entry["image_path"]) as im:
                    canvas.paste(im.convert("RGB").resize((size, size)), (x, y))
                color = "green" if entry["survivor"] else "red"
                draw.rectangle((x+1, y+1, x+size-2, y+size-2), outline=color, width=3)
                text = f"rewrite {column} | step {step}\nPersonal {entry['personal']:.3f} | {'keep' if entry['survivor'] else 'prune'}"
            else:
                text = f"rewrite {column} | step {step}\nPruned previously"
            draw.text((x+5, y+size+4), text, fill="black")
    canonical_id = manifest.get("canonical_audit_id", 0)
    extras = [("Canonical rerun, same noise", str(output / f"audit/step_28/rewrite_{canonical_id:02d}_x0.png")), ("Rewrite-PSP selected", str(output / "selected.png")), ("Historical cache reference", manifest.get("reference_canonical_image"))]
    for column, (label, path) in enumerate(extras):
        x, y = column*size, 36+3*(size+label_h)
        if path and Path(path).is_file():
            with Image.open(path) as im:
                canvas.paste(im.convert("RGB").resize((size, size)), (x, y))
        draw.text((x+5, y+size+4), label, fill="black")
    canvas.save(output / "rewrite_psp_gallery.png")
    compact = Image.new("RGB",(size*2,size+90),"white")
    pen = ImageDraw.Draw(compact)
    posthoc_path = output/"personal_comparison.json"
    posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))["methods"] if posthoc_path.is_file() else {}
    for col,(label,path) in enumerate(extras[:2]):
        if path and Path(path).is_file():
            with Image.open(path) as im:
                compact.paste(im.convert("RGB").resize((size,size)),(col*size,0))
        pen.text((col*size+5,size+5),label,fill="black")
        key = {0:"canonical",1:"rewrite_psp"}.get(col)
        if key in posthoc:
            values = posthoc[key]
            pen.text((col*size+5,size+25),f"Personal: {values['personal_from_png']:.3f}",fill="black")
    compact.save(output/"canonical_rewrite_psp.png")


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from transformers import T5TokenizerFast

    args.dualfashion_root = args.dualfashion_root.resolve()
    sys.path.insert(0, str(args.dualfashion_root))
    from Inference_ifashion_PFITB import build_preference_text, load_numpy_dict
    info = args.dualfashion_root / "processed_info"
    fitb = load_numpy_dict(str(args.data_root / "processed/fitb_test.npy"))
    grd = load_numpy_dict(str(args.data_root / "test_grd_dict.npy"))
    if not 0 <= args.task_index < len(fitb["uids"]):
        raise IndexError("Task index out of range")
    uid, oid = int(fitb["uids"][args.task_index]), int(fitb["oids"][args.task_index])
    blank = list(fitb["outfits"][args.task_index]).index(0)
    category = int(fitb["category"][args.task_index][blank])
    categories = load_numpy_dict(str(args.data_root / "id_cate_dict.npy"))
    outfit = [int(item) for item in grd["outfits"][args.task_index]]
    context = [item for i, item in enumerate(outfit) if i != blank]
    prefs = load_numpy_dict(str(info / "stage2_ifashion_test/test_preference.npy"))
    canonical = build_preference_text(prefs, uid, category, list(ATTRIBUTE_KEYS))
    rewrites = (build_history_rewrites(prefs[uid][category], load_numpy_dict(str(info / "item_info.npy")))
                if args.rewrite_mode == "history" else build_order_rewrites(canonical))
    model_path = args.dualfashion_root / "checkpoint/0057501"
    tokenizer = T5TokenizerFast(tokenizer_file=str(model_path / "tokenizer/tokenizer.json"), extra_ids=100)
    tokenizer.pad_token = tokenizer.eos_token
    for row in rewrites:
        ids = tokenizer(row["preference_text"], truncation=False).input_ids
        if len(ids) > 256:
            raise ValueError(f"rewrite {row['rewrite_id']} exceeds 256 tokens; truncation changes semantics")
        row["token_count"] = len(ids)
        row["token_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    if len({row["token_sha256"] for row in rewrites}) != 8:
        raise ValueError("Rewrites collapse to identical token sequences")
    noise_seed = args.noise_seed if args.noise_seed is not None else args.seed + args.task_index * args.seed_stride
    reference = args.reference_cache or args.dualfashion_root / f"output/ifashion_fitb_comp/psp_offline_cache/psp_offline_group0_task{args.task_index}_seed{args.seed}"
    reference_metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8")) if (reference / "metadata.json").is_file() else None
    if reference_metadata and (reference_metadata["uid"], reference_metadata["oid"], reference_metadata["task_index"]) != (uid, oid, args.task_index):
        raise ValueError("Reference cache is for a different task")
    paths = np.load(info / "all_item_image_paths.npy", allow_pickle=True)
    manifest = {
        "method": "Rewrite-PSP@8", "rewrite_kind": "balanced_json_key_order_only", "task_index": args.task_index,
        "uid": uid, "oid": oid, "target_category": category, "target_item_id": outfit[blank],
        "base_seed": args.seed, "noise_seed": noise_seed, "shared_noise": True,
        "canonical_preference": canonical, "sentence_text": f"Recommend a fashion {categories.get(category, 'item')} item, on white background.",
        "rewrites": rewrites, "schedule": [[7,4],[14,2],[28,1]], "num_inference_steps": 28,
        "guidance_scales": {"text":8.0,"context":7.0,"preference":8.0},
        "reference_cache": str(reference.resolve()),
        "reference_initial_latent_hash": reference_metadata.get("initial_latent_hashes", {}).get(str(noise_seed)) if reference_metadata else None,
        "reference_canonical_image": str(reference / f"offline_candidates/{uid}_{oid}/step_28/seed_{noise_seed}_x0.png"),
        "gt_image_path": str(args.data_root / "semantic_category" / str(paths[outfit[blank]])),
        "selection_uses_gt": False, "reward": "100*cosine(OpenCLIP ViT-H/14(candidate), fixed original category-history)",
        "budget_particle_steps": 112, "checkpoint": str(model_path.resolve()),
        "device": args.device, "reward_device": args.reward_device, "cpu_offload": args.cpu_offload,
        "audit_all_rewrites": args.audit_all_rewrites,
        "data_root": str(args.data_root.resolve()),
    }
    if args.rewrite_mode == "history":
        manifest["rewrite_kind"] = "fixed_history_item_attribute_modes"
        manifest["canonical_audit_id"] = 8  # audit only; not an online particle
        manifest.pop("gt_image_path", None)
    output = args.output_dir.resolve()
    manifest_path = output / "rewrite_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output manifest differs from this configuration; choose a new output directory")
    write_json(manifest_path, manifest)
    print(f"Task {args.task_index}: {uid}_{oid}, category={category}, shared noise={noise_seed}", flush=True)
    print(f"8 fixed {args.rewrite_mode} rewrites; token counts={[r['token_count'] for r in rewrites]}; online budget=112", flush=True)
    if args.prepare_only:
        print(f"Prepared {manifest_path}")
        return
    metadata_path = output / "metadata.json"
    if metadata_path.is_file() and args.resume and ((output / "personal_comparison.json").is_file() or (output / "posthoc_comparison.json").is_file()) and json.loads(metadata_path.read_text(encoding="utf-8")).get("artifact_quantizer") == "native_floor_uint8":
        print(f"Already complete: {metadata_path}")
        return
    if metadata_path.is_file() and not args.resume:
        raise FileExistsError(f"Completed run exists: {output}; use --resume or a new output directory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in the selected Python environment")
    from torchvision import transforms as T
    from tqdm import tqdm
    from Inference_ifashion_PFITB import MATCHING_LAMBDA, compute_stage2_context_offset, encode_text, ensure_stage2_modules, load_image_tensor, prepend_pad_segment, restore_stage2_parameters
    from run_psp_ifashion_smoke import PersonalReward, decode_latent, estimate_x0_from_flowmatch, save_latent_visualization
    from run_ifashion_gor_greedy_personal import PERSONAL_CLIP_WEIGHTS
    from sd3_modules.stage2_pipeline import DualDiffSD3Pipeline

    weights = args.clip_weights or PERSONAL_CLIP_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"Missing frozen Personal weights: {weights}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats(device)
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
    negative = "low quality, blurry, low resolution, backlit, cartoon, animated, deformed, oversaturated, undersaturated, out of frame"
    with torch.inference_mode():
        sentence = encode_text(tokenizer, pipe.text_encoder, manifest["sentence_text"], 256, device).to(device=device, dtype=dtype)
        neg = encode_text(tokenizer, pipe.text_encoder, negative, 256, device).to(device=device, dtype=dtype)
        text = prepend_pad_segment(sentence, neg, 512)
        uncond = torch.cat([neg, neg], dim=1)
        prompts = {}
        canonical_id = manifest.get("canonical_audit_id", 0)
        conditioning_rows = rewrites + ([{"rewrite_id":canonical_id, "preference_text":canonical}] if canonical_id == 8 else [])
        for row in conditioning_rows:
            pref = encode_text(tokenizer, pipe.text_encoder, row["preference_text"], 256, device).to(device=device, dtype=dtype)
            prompts[row["rewrite_id"]] = torch.cat([uncond,text,text,torch.cat([pref,sentence],dim=1)],dim=0)
        transform = T.Compose([T.Resize(512, interpolation=T.InterpolationMode.BICUBIC),T.CenterCrop(512),T.ToTensor(),T.Normalize([0.5],[0.5])])
        images = torch.stack([load_image_tensor(item, paths, str(args.data_root / "semantic_category"), transform) for item in context]).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type=="cuda", dtype=dtype):
            dist = pipe.vae.encode(images).latent_dist
            context_latents = ((dist.mode()-pipe.vae.config.shift_factor)*pipe.vae.config.scaling_factor).unsqueeze(0).to(dtype=dtype)
            offset = compute_stage2_context_offset(pipe.transformer, context_latents)
        shape = (1,*context_latents.shape[2:])
        shared = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(noise_seed),device=device,dtype=dtype)
        shared_hash = latent_hash(shared)
        if manifest["reference_initial_latent_hash"] and shared_hash != manifest["reference_initial_latent_hash"]:
            raise ValueError("Initial noise differs from reference cache despite using its seed")
    reward = PersonalReward(args.data_root / "processed/test_history_clipembs.npy", uid, category, torch.device(args.reward_device), weights)
    scheduler = pipe.scheduler
    scheduler.set_timesteps(28, device=device)
    timesteps = scheduler.timesteps.to(device)
    active = [{"rewrite_id":i,"latent":shared.clone()} for i in range(8)]
    metrics, trace = [], []
    forward_calls, start_step, elapsed_before = 0, 1, 0.0
    state_path = output / "resume_state.pt"
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state["manifest"] != manifest or state["shared_hash"] != shared_hash:
            raise ValueError("Resume state configuration or noise differs")
        active = [{"rewrite_id":row["rewrite_id"],"latent":row["latent"].to(device=device,dtype=dtype)} for row in state["active"]]
        metrics, trace = state["metrics"], state["trace"]
        forward_calls, start_step, elapsed_before = state["forward_calls"], state["completed_step"]+1, state["elapsed_s"]
        print(f"Resuming at step {start_step}, alive={len(active)}",flush=True)
    started = time.perf_counter()

    @torch.inference_mode()
    def advance(particle, step):
        latent = particle["latent"]
        zero = torch.zeros_like(offset)
        model_input = torch.cat([latent]*4,dim=0) + MATCHING_LAMBDA*torch.cat([zero,zero,offset,offset],dim=0)
        with torch.autocast(device_type=device.type,enabled=device.type=="cuda",dtype=dtype):
            prediction = pipe.transformer(hidden_states=model_input,timestep=timesteps[step-1].expand(4),encoder_hidden_states=prompts[particle["rewrite_id"]],pooled_projections=None,return_dict=False)[0]
        u,t,c,p = prediction.chunk(4)
        prediction = u+8.0*(t-u)+7.0*(c-t)+8.0*(p-c)
        x0 = estimate_x0_from_flowmatch(scheduler,prediction,latent,step-1)
        scheduler._step_index = step-1
        particle["latent"] = scheduler.step(prediction,timesteps[step-1],latent,return_dict=False)[0]
        return x0

    def save_state(step):
        state = {"manifest":manifest,"shared_hash":shared_hash,"completed_step":step,"active":[{"rewrite_id":p["rewrite_id"],"latent":p["latent"].detach().cpu()} for p in active],"metrics":metrics,"trace":trace,"forward_calls":forward_calls,"elapsed_s":elapsed_before+time.perf_counter()-started}
        temporary = output / "resume_state.tmp.pt"
        torch.save(state,temporary)
        temporary.replace(state_path)

    keep_at = {7:4,14:2,28:1}
    with tqdm(total=112,initial=forward_calls,desc=f"Rewrite-PSP {uid}_{oid}",unit="particle-step") as progress:
        for step in range(start_step,29):
            predictions = []
            for particle in active:
                predictions.append(advance(particle,step))
                forward_calls += 1
                progress.update(1)
            if step in keep_at:
                decoded = torch.cat([decode_latent(pipe,x0) for x0 in predictions])
                scores = reward.score(decoded,outfit,blank)
                order = sorted(range(len(active)),key=lambda i:(-scores[i],active[i]["rewrite_id"]))
                selected = order[:keep_at[step]]
                survivors = [active[i]["rewrite_id"] for i in selected]
                for i,particle in enumerate(active):
                    rid = particle["rewrite_id"]
                    folder = output / f"predictions/step_{step:02d}"
                    folder.mkdir(parents=True,exist_ok=True)
                    image_path = folder / f"rewrite_{rid:02d}_noise{noise_seed}_x0.png"
                    save_reward_image(decoded[i],image_path)
                    torch.save(predictions[i].detach().float().cpu(),folder/f"rewrite_{rid:02d}_x0.pt")
                    save_latent_visualization(predictions[i],folder/f"rewrite_{rid:02d}_latent.png")
                    metrics.append({"phase":"psp","step":step,"rewrite_id":rid,"noise_seed":noise_seed,"personal":float(scores[i]),"survivor":rid in survivors,"image_path":str(image_path)})
                trace.append({"step":step,"alive_before":[p["rewrite_id"] for p in active],"survivor_rewrites":survivors,"scores":{str(p["rewrite_id"]):float(scores[i]) for i,p in enumerate(active)}})
                active = [active[i] for i in selected]
                print(f"\nstep {step}: survivors={survivors}; Personal={[round(scores[i],4) for i in selected]}",flush=True)
            save_state(step)
            progress.set_postfix(step=step,alive=len(active))
    online_elapsed = elapsed_before+time.perf_counter()-started
    if start_step > 1:
        # Restore exact saved reward views too when resuming older artifacts.
        for row in metrics:
            if row["phase"] != "psp":
                continue
            rid, step = row["rewrite_id"], row["step"]
            latent_path = output/f"predictions/step_{step:02d}/rewrite_{rid:02d}_x0.pt"
            x0 = torch.load(latent_path,map_location=device,weights_only=True).to(dtype=dtype)
            save_reward_image(decode_latent(pipe,x0)[0],row["image_path"])
    winner = active[0]
    terminal = decode_latent(pipe,winner["latent"])
    terminal_score = reward.score(terminal,outfit,blank)[0]
    save_reward_image(terminal[0],output/"selected.png")
    # Freeze the online decision before any post-selection text/image diagnostics.
    write_json(output/"online_selection.json", {"selected_rewrite_id":winner["rewrite_id"],"personal_terminal":terminal_score,"forward_calls":forward_calls,"online_elapsed_s":online_elapsed,"trace":trace})
    audit_calls = 0
    audit_ids = sorted(set(range(8)) | {canonical_id}) if args.audit_all_rewrites else sorted({canonical_id,winner["rewrite_id"]})
    metrics = [row for row in metrics if row["phase"] == "psp"]
    with tqdm(total=28*len(audit_ids),desc="Separate canonical/replay audit",unit="particle-step") as progress:
            for rid in audit_ids:
                particle = {"rewrite_id":rid,"latent":shared.clone()}
                scheduler.set_timesteps(28,device=device)
                for step in range(1,29):
                    x0 = advance(particle,step)
                    audit_calls += 1
                    progress.update(1)
                    if step in keep_at:
                        decoded = decode_latent(pipe,x0)
                        score = reward.score(decoded,outfit,blank)[0]
                        folder = output / f"audit/step_{step:02d}"
                        folder.mkdir(parents=True,exist_ok=True)
                        path = folder/f"rewrite_{rid:02d}_x0.png"
                        save_reward_image(decoded[0],path)
                        metrics.append({"phase":"audit","step":step,"rewrite_id":rid,"noise_seed":noise_seed,"personal":score,"survivor":None,"image_path":str(path)})
    with (output/"scores.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.DictWriter(f,fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    write_json(metadata_path,{"manifest":manifest,"artifact_quantizer":"native_floor_uint8","initial_latent_hashes":{str(i):shared_hash for i in range(8)},"all_initial_latents_identical":True,"matches_reference_initial_noise":shared_hash==manifest["reference_initial_latent_hash"] if manifest["reference_initial_latent_hash"] else None,"transformer_forward_calls":forward_calls,"audit_forward_calls":audit_calls,"online_elapsed_s":online_elapsed,"selected_rewrite_id":winner["rewrite_id"],"selected_preference_text":rewrites[winner["rewrite_id"]]["preference_text"],"selected_personal_terminal":terminal_score,"selected_image":str(output/"selected.png"),"trace":trace,"peak_vram_gib":torch.cuda.max_memory_allocated(device)/2**30 if device.type=="cuda" else None})
    from PIL import Image
    comparison = {"selection_reward":"fixed original history Personal only", "methods":{}}
    @torch.inference_mode()
    def feature(path):
        with Image.open(path) as im:
            pixels = reward.preprocess(im.convert("RGB")).unsqueeze(0).to(reward.device)
        return torch.nn.functional.normalize(reward.clip_model.encode_image(pixels).float(),dim=-1)
    for label,path in (("canonical",output/f"audit/step_28/rewrite_{canonical_id:02d}_x0.png"),("rewrite_psp",output/"selected.png")):
        embedding = feature(path)
        comparison["methods"][label] = {"personal_from_png":float((100.0*embedding@reward.history_embedding.unsqueeze(1)).item()),"image_path":str(path)}
    verification_path = output/f"audit/step_28/rewrite_{winner['rewrite_id']:02d}_x0.png"
    original = np.asarray(Image.open(output/"selected.png")).astype(float)
    replay = np.asarray(Image.open(verification_path)).astype(float)
    comparison["winner_replay_exact_png_pixels"] = bool(np.array_equal(original,replay))
    comparison["winner_replay_pixel_mae"] = float(np.abs(original-replay).mean())
    comparison["selected_png_reward_matches_online"] = abs(comparison["methods"]["rewrite_psp"]["personal_from_png"]-terminal_score) < 1e-4
    if not comparison["selected_png_reward_matches_online"]:
        raise RuntimeError("Saved reward pixels do not reproduce the online Personal score")
    write_json(output/"personal_comparison.json",comparison)
    write_json(output/"selected_preference.json", {
        "source_user_preference": json.loads(canonical),
        "conditioning_preference": json.loads(rewrites[winner["rewrite_id"]]["preference_text"]),
        "preference_origin": "fixed before generation; never captioned from selected image",
        "source_history_item_id": rewrites[winner["rewrite_id"]].get("source_history_item_id"),
        "selected_image": str(output/"selected.png"), "personal":terminal_score,
        "alignment_status": "not verified; Personal is not a text-image consistency score",
    })
    gallery(manifest,metrics,output)
    print(f"Complete: rewrite {winner['rewrite_id']}, terminal Personal={terminal_score:.4f}, online calls={forward_calls}, audit calls={audit_calls}\n{output}",flush=True)


if __name__ == "__main__":
    main()
