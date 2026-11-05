"""Offline diagnostics for PSP intermediate x0 predictions.

GT similarity and attribute agreement are diagnostic only. They must not be
fed into a reward, selector, or candidate pruning decision.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image


COLORS = ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"]
PATTERNS = ["solid", "striped", "plaid", "floral", "polka dot", "geometric", "animal print", "textured"]
SILHOUETTES = ["slim fitting", "fitted", "regular fit", "oversized", "loose fitting", "cropped", "long line", "structured"]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


@torch.inference_mode()
def encode_images(model, preprocess, paths, device, batch_size):
    out = []
    good = []
    def load_one(p):
        try:
            return preprocess(Image.open(p).convert("RGB")), p
        except Exception:
            return None, p
    with ThreadPoolExecutor(max_workers=8) as pool:
      for batch in batched(paths, batch_size):
        loaded = list(pool.map(load_one, batch))
        ims = [x for x, _ in loaded if x is not None]
        keep = [p for x, p in loaded if x is not None]
        if not ims:
            continue
        z = model.encode_image(torch.stack(ims).to(device, dtype=next(model.parameters()).dtype))
        z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(z.float().cpu().numpy())
        good.extend(keep)
    return np.concatenate(out, axis=0), good


@torch.inference_mode()
def encode_text(model, tokenizer, labels, device):
    prompts = [f"a fashion item that is {x}" for x in labels]
    z = model.encode_text(tokenizer(prompts).to(device))
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return z.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).parent)
    ap.add_argument("--data-root", type=Path, default=Path(r"C:\tmp\ifashion"))
    ap.add_argument("--cache-root", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    outdir = args.output or args.repo / "output" / "ifashion_fitb_semantics"
    outdir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_root or args.repo / "output" / "ifashion_fitb_comp" / "psp_offline_cache"
    personal_path = args.repo / "output/ifashion_fitb_personal/psp/psp_group0_personal/psp_personal_scores.csv"
    comp_path = args.repo / "output/ifashion_fitb_comp/psp/psp_group0_compatibility/psp_compatibility_scores.csv"
    personal = read_csv(personal_path)
    comp = read_csv(comp_path)
    comp_by_key = {(r["task_index"], r["seed"], r["step"]): r for r in comp}
    fit = np.load(args.data_root / "processed/fitb_test.npy", allow_pickle=True).item()
    grd = np.load(args.data_root / "test_grd.npy", allow_pickle=True).item()
    all_paths = np.load(args.data_root / "all_item_image_paths.npy")
    rows = []
    gt_by_task = {}
    for task, (uid, oid, outfits) in enumerate(zip(fit["uids"], fit["oids"], fit["outfits"])):
        outfits = outfits.tolist() if hasattr(outfits, "tolist") else list(outfits)
        missing = outfits.index(0)
        gt_iid = grd[oid]["outfits"][missing]
        gt_path = args.data_root / "semantic_category" / str(all_paths[gt_iid])
        gt_by_task[str(task)] = str(gt_path)
    for r in personal:
        task = r["task_index"]
        c = comp_by_key.get((task, r["seed"], r["step"]))
        if c is None:
            raise RuntimeError(f"missing compatibility row for {task}/{r['seed']}/{r['step']}")
        rows.append({"task_index": int(task), "uid": int(r["uid"]), "oid": int(r["oid"]), "step": int(r["step"]),
                     "seed": int(r["seed"]), "personal": float(r["personal_reward"]),
                     "compatibility": float(c["compatibility_reward"]), "image_path": r["image_path"],
                     "gt_path": gt_by_task[task]})
    print(f"rows={len(rows)} tasks={len(gt_by_task)}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79K", device=device)
    if device.type == "cuda":
        model = model.half()
    model.eval()
    image_paths = [Path(r["image_path"]) for r in rows]
    gt_paths = [Path(gt_by_task[str(i)]) for i in range(len(gt_by_task))]
    emb_cache = outdir / "clip_embeddings.npz"
    if emb_cache.exists():
        z = np.load(emb_cache)
        image_emb, gt_emb = z["image"], z["gt"]
    else:
        emb, good = encode_images(model, preprocess, image_paths + gt_paths, device, args.batch_size)
        if len(good) != len(image_paths) + len(gt_paths):
            raise RuntimeError(f"failed to encode {len(image_paths)+len(gt_paths)-len(good)} images")
        image_emb = emb[: len(image_paths)]
        gt_emb = emb[len(image_paths):]
        np.savez_compressed(emb_cache, image=image_emb, gt=gt_emb)
    gt_index = {str(i): i for i in range(len(gt_paths))}
    for i, r in enumerate(rows):
        r["gt_similarity"] = float(image_emb[i] @ gt_emb[gt_index[str(r["task_index"])]])
    # Zero-shot attribute distributions. Agreement is cosine between the two
    # distributions; it is a proxy, not a ground-truth attribute annotation.
    banks = {"color": COLORS, "pattern": PATTERNS, "silhouette": SILHOUETTES}
    attr_scores = {}
    for name, labels in banks.items():
        text_emb = encode_text(model, open_clip.get_tokenizer("ViT-H-14"), labels, device)
        logits = image_emb @ text_emb.T
        gt_logits = gt_emb @ text_emb.T
        probs = torch.softmax(torch.tensor(logits) * 20.0, dim=1).numpy()
        gt_probs = torch.softmax(torch.tensor(gt_logits) * 20.0, dim=1).numpy()
        row_gt_probs = gt_probs[[gt_index[str(r["task_index"])] for r in rows]]
        attr_scores[name] = (probs * row_gt_probs).sum(axis=1) / (np.linalg.norm(probs, axis=1) * np.linalg.norm(row_gt_probs, axis=1) + 1e-8)
    for i, r in enumerate(rows):
        for name, vals in attr_scores.items(): r[name + "_agreement"] = float(vals[i])
    with (outdir / "step_candidate_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    # Aggregate by step and by task. Task means are the unit for uncertainty.
    metrics = ["personal", "compatibility", "gt_similarity", "color_agreement", "pattern_agreement", "silhouette_agreement"]
    summary = []
    by_step = defaultdict(list)
    for r in rows: by_step[r["step"]].append(r)
    for step, rs in sorted(by_step.items()):
        rec = {"step": step, "n_candidates": len(rs), "n_tasks": len({r["task_index"] for r in rs})}
        for m in metrics:
            vals = np.array([r[m] for r in rs], dtype=float); rec[m+"_candidate_mean"] = float(vals.mean()); rec[m+"_candidate_std"] = float(vals.std())
            taskvals = np.array([np.mean([r[m] for r in rs if r["task_index"] == t]) for t in sorted({r["task_index"] for r in rs})])
            rec[m+"_task_mean"] = float(taskvals.mean()); rec[m+"_task_std"] = float(taskvals.std(ddof=1))
        summary.append(rec)
    with (outdir / "step_summary.json").open("w", encoding="utf-8") as f: json.dump(summary, f, indent=2)
    # Per-task rank correlations and winner overlap.
    diag = []
    for task in sorted({r["task_index"] for r in rows}):
        rr = [r for r in rows if r["task_index"] == task]
        bys = {s: sorted([r for r in rr if r["step"] == s], key=lambda x: x["seed"]) for s in [7,14,28]}
        x7 = {r["seed"]: r for r in bys[7]}; x14 = {r["seed"]: r for r in bys[14]}; x28 = {r["seed"]: r for r in bys[28]}
        def rho(a, b, key):
            common = sorted(set(a) & set(b)); va=np.array([a[k][key] for k in common]); vb=np.array([b[k][key] for k in common]);
            ra=va.argsort().argsort(); rb=vb.argsort().argsort(); return float(np.corrcoef(ra,rb)[0,1]) if len(common)>2 else float("nan")
        d={"task_index":task}
        for m in metrics: d[m+"_rho_7_28"] = rho(x7,x28,m); d[m+"_rho_14_28"] = rho(x14,x28,m)
        for s in [7,14,28]:
            d[f"{s}_winner_personal"] = max(bys[s], key=lambda x:x["personal"])["seed"]
            d[f"{s}_winner_gt"] = max(bys[s], key=lambda x:x["gt_similarity"])["seed"]
        diag.append(d)
    with (outdir / "task_rank_diagnostics.json").open("w", encoding="utf-8") as f: json.dump(diag, f)
    print(json.dumps({"output": str(outdir), "summary": summary}, indent=2))


if __name__ == "__main__": main()
