"""Post-hoc GT similarity for Standard, BoN, and PSP reward selections.

GT images are used only as evaluation references. This script does not change
reward scores, candidate ranking, or PSP pruning.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image


REWARD_FILES = {
    "compatibility": {
        "bon": "output/ifashion_fitb_comp/bon/bon_group0_seed42_scores/candidate_compatibility_scores.csv",
        "psp_scores": "output/ifashion_fitb_comp/psp/psp_group0_compatibility/psp_compatibility_scores.csv",
        "psp_selection": "output/ifashion_fitb_comp/psp/psp_group0_compatibility/psp_selection_summary.csv",
    },
    "personal": {
        "bon": "output/ifashion_fitb_personal/bon/bon_group0_seed42_scores/candidate_personal_scores.csv",
        "psp_scores": "output/ifashion_fitb_personal/psp/psp_group0_personal/psp_personal_scores.csv",
        "psp_selection": "output/ifashion_fitb_personal/psp/psp_group0_personal/psp_selection_summary.csv",
    },
    "bottleneck": {
        "bon": "output/ifashion_fitb_bottleneck/bon/bon_group0_seed42_scores/candidate_bottleneck_scores.csv",
        "psp_scores": "output/ifashion_fitb_bottleneck/psp/psp_group0_bottleneck/psp_bottleneck_scores.csv",
        "psp_selection": "output/ifashion_fitb_bottleneck/psp/psp_group0_bottleneck/psp_selection_summary.csv",
    },
    "pcar": {
        "bon": "output/ifashion_fitb_pcar/bon/bon_group0_seed42_scores/candidate_pcar_scores.csv",
        "psp_scores": "output/ifashion_fitb_pcar/psp/psp_group0_pcar/psp_pcar_scores.csv",
        "psp_selection": "output/ifashion_fitb_pcar/psp/psp_group0_pcar/psp_selection_summary.csv",
    },
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


@torch.inference_mode()
def encode_images(model, preprocess, paths, device, batch_size):
    embeddings = []
    encoded_paths = []

    def load_one(path):
        try:
            with Image.open(path) as image:
                return preprocess(image.convert("RGB")), path
        except Exception as exc:
            raise RuntimeError(f"failed to read image: {path}") from exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        for batch in batched(paths, batch_size):
            loaded = list(pool.map(load_one, batch))
            pixels = torch.stack([item[0] for item in loaded])
            pixels = pixels.to(device, dtype=next(model.parameters()).dtype)
            z = model.encode_image(pixels)
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            embeddings.append(z.float().cpu().numpy())
            encoded_paths.extend(item[1] for item in loaded)
    return np.concatenate(embeddings), encoded_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).parent)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\tmp\ifashion"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    repo = args.repo
    output = args.output or repo / "output/ifashion_fitb_best_gt_similarity"
    output.mkdir(parents=True, exist_ok=True)

    fitb = np.load(args.data_root / "processed/fitb_test.npy", allow_pickle=True).item()
    grd = np.load(args.data_root / "test_grd.npy", allow_pickle=True).item()
    item_paths = np.load(args.data_root / "all_item_image_paths.npy")
    task_gt = {}
    task_uid_oid = {}
    for i, (uid, oid, outfit) in enumerate(zip(fitb["uids"], fitb["oids"], fitb["outfits"])):
        outfit = outfit.tolist() if hasattr(outfit, "tolist") else list(outfit)
        missing_position = outfit.index(0)
        gt_item_id = grd[oid]["outfits"][missing_position]
        task_gt[i] = args.data_root / "semantic_category" / str(item_paths[gt_item_id])
        task_uid_oid[i] = (int(uid), int(oid))

    selected = []
    for reward, files in REWARD_FILES.items():
        bon_rows = read_csv(repo / files["bon"])
        grouped = defaultdict(list)
        for row in bon_rows:
            grouped[(int(row["uid"]), int(row["oid"]))].append(row)
        for task, (uid, oid) in task_uid_oid.items():
            candidates = grouped[(uid, oid)]
            standard = next(r for r in candidates if r["is_standard"].lower() == "true")
            bon = next(r for r in candidates if r["is_bon_winner"].lower() == "true")
            selected.extend([
                {"task_index": task, "uid": uid, "oid": oid, "method": "standard", "reward": reward,
                 "seed": int(standard["seed"]), "image_path": Path(standard["image_path"]), "gt_path": task_gt[task]},
                {"task_index": task, "uid": uid, "oid": oid, "method": "bon", "reward": reward,
                 "seed": int(bon["seed"]), "image_path": Path(bon["image_path"]), "gt_path": task_gt[task]},
            ])

        psp_scores = read_csv(repo / files["psp_scores"])
        psp_by_key = {(int(r["task_index"]), int(r["seed"]), int(r["step"])): r for r in psp_scores}
        psp_selection = read_csv(repo / files["psp_selection"])
        for row in psp_selection:
            task = int(row["task_index"])
            seed = int(row["psp_seed"])
            chosen = psp_by_key[(task, seed, 28)]
            selected.append({"task_index": task, "uid": int(row["uid"]), "oid": int(row["oid"]),
                             "method": "psp", "reward": reward, "seed": seed,
                             "image_path": Path(chosen["image_path"]), "gt_path": task_gt[task]})

    # Deduplicate repeated Standard and GT paths before model inference.
    all_paths = list(dict.fromkeys(str(r["image_path"]) for r in selected) |
                      dict.fromkeys(str(p) for p in task_gt.values()))
    missing = [p for p in all_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected/reference images are missing; first: {missing[0]}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79K", device=device)
    if device.type == "cuda":
        model = model.half()
    model.eval()
    embedding_cache = output / "selected_and_gt_embeddings.npz"
    path_cache = output / "selected_and_gt_embedding_paths.json"
    if embedding_cache.exists() and path_cache.exists():
        encoded_paths = json.loads(path_cache.read_text(encoding="utf-8"))
        embeddings = np.load(embedding_cache)["embeddings"]
        if encoded_paths != all_paths:
            raise RuntimeError("embedding cache paths do not match current selected image paths")
    else:
        embeddings, encoded = encode_images(model, preprocess, [Path(p) for p in all_paths], device, args.batch_size)
        encoded_paths = [str(p) for p in encoded]
        np.savez_compressed(embedding_cache, embeddings=embeddings)
        path_cache.write_text(json.dumps(encoded_paths), encoding="utf-8")
    emb_by_path = {str(p): embeddings[i] for i, p in enumerate(encoded_paths)}

    per_task = []
    for row in selected:
        gt_embedding = emb_by_path[str(row["gt_path"])]
        image_embedding = emb_by_path[str(row["image_path"])]
        per_task.append({"task_index": row["task_index"], "uid": row["uid"], "oid": row["oid"],
                         "method": row["method"], "reward": row["reward"], "seed": row["seed"],
                         "gt_similarity": float(image_embedding @ gt_embedding),
                         "image_path": str(row["image_path"]), "gt_path": str(row["gt_path"])})
    with (output / "per_task_gt_similarity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_task[0].keys()))
        writer.writeheader()
        writer.writerows(per_task)

    rng = np.random.default_rng(42)
    summary = []
    for reward in REWARD_FILES:
        for method in ["standard", "bon", "psp"]:
            rows = sorted([r for r in per_task if r["reward"] == reward and r["method"] == method], key=lambda r: r["task_index"])
            values = np.array([r["gt_similarity"] for r in rows], dtype=float)
            samples = rng.integers(0, len(values), size=(args.bootstrap_samples, len(values)))
            lower, upper = np.quantile(values[samples].mean(axis=1), [0.025, 0.975])
            summary.append({"reward": reward, "method": method, "n_tasks": len(values),
                            "gt_similarity_mean": float(values.mean()), "gt_similarity_std": float(values.std(ddof=1)),
                            "bootstrap_95ci_lower": float(lower), "bootstrap_95ci_upper": float(upper)})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    paired = []
    lookup = {(r["reward"], r["method"], r["task_index"]): r["gt_similarity"] for r in per_task}
    for reward in REWARD_FILES:
        for left, right in [("bon", "standard"), ("psp", "standard"), ("psp", "bon")]:
            delta = np.array([lookup[(reward, left, task)] - lookup[(reward, right, task)] for task in range(len(task_uid_oid))])
            samples = rng.integers(0, len(delta), size=(args.bootstrap_samples, len(delta)))
            lower, upper = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
            paired.append({"reward": reward, "comparison": f"{left}-{right}", "n_tasks": len(delta),
                           "mean_delta": float(delta.mean()), "paired_bootstrap_95ci_lower": float(lower),
                           "paired_bootstrap_95ci_upper": float(upper), "fraction_positive_tasks": float((delta > 0).mean())})
    with (output / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        writer.writeheader()
        writer.writerows(paired)
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paired_comparisons": paired}, f, indent=2)
    print(json.dumps({"output": str(output), "unique_images_encoded": len(encoded_paths),
                      "summary": summary, "paired_comparisons": paired}, indent=2))


if __name__ == "__main__":
    main()
