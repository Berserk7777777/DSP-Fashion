"""Score DiFashion PFITB folders with the frozen iFashion compatibility model."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import open_clip
from eval.compatibility_evaluator.compatibility_net import FashionEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluator-ckpt", type=Path, default=Path(__file__).resolve().parent / "eval/compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["9239/17894", "9239/17895", "9242/17897", "9242/17899"],
        help="Task pairs in uid/oid form.",
    )
    return parser.parse_args()


def extract_features(model, transform, paths, device, batch_size):
    result = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), batch_size), desc="OpenCLIP features"):
            images = []
            for path in paths[start : start + batch_size]:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            result.append(model.encode_image(torch.stack(images).to(device)).float().cpu())
    return torch.cat(result, dim=0)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fitb = np.load(args.data_root / "fitb_test_dict.npy", allow_pickle=True).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")

    requested_tasks = {tuple(int(part) for part in value.split("/")) for value in args.tasks}
    task_dirs = []
    image_paths = []
    # DiFashion exports are nested as <eval>/.../images/<uid>/<oid>.
    for images_root in [args.input_root] + [p for p in args.input_root.rglob("images") if p.is_dir()]:
        for uid, oid in sorted(requested_tasks):
            task_dir = images_root / str(uid) / str(oid)
            if not task_dir.is_dir():
                continue
            candidates = sorted(
                [p for p in task_dir.glob("*.jpg") if p.stem.isdigit()],
                key=lambda p: int(p.stem),
            )
            if not candidates:
                continue
            if uid not in fitb or oid not in fitb[uid]:
                raise KeyError(f"Missing FITB metadata for {uid}_{oid}")
            outfit = list(fitb[uid][oid])
            if outfit.count(0) != 1:
                raise ValueError(f"{uid}_{oid} does not contain one blank: {outfit}")
            task_dirs.append((uid, oid, task_dir, candidates, outfit))
            image_paths.extend(candidates)

    found_keys = {(uid, oid) for uid, oid, *_ in task_dirs}
    if found_keys != requested_tasks:
        raise ValueError(f"Requested tasks not found. requested={sorted(requested_tasks)}, found={sorted(found_keys)}")

    print(f"Scoring {len(image_paths)} DiFashion candidates on {device}.")
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b-s32b-b79K")
    clip_model = clip_model.to(device).eval()
    candidate_features = extract_features(clip_model, preprocess, image_paths, device, args.batch_size)
    del clip_model
    torch.cuda.empty_cache()

    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(args.evaluator_ckpt, map_location="cpu"))
    evaluator = evaluator.to(device).eval()

    rows = []
    task_summaries = []
    offset = 0
    for uid, oid, task_dir, candidates, outfit in task_dirs:
        blank_idx = outfit.index(0)
        context = torch.from_numpy(np.asarray([item_features[item_id] for item_id in outfit])).float()
        features = candidate_features[offset : offset + len(candidates)]
        offset += len(candidates)
        batch_outfits = context.unsqueeze(0).repeat(len(candidates), 1, 1)
        batch_outfits[:, blank_idx, :] = features
        with torch.inference_mode():
            rewards = torch.sigmoid(evaluator(batch_outfits.to(device))).cpu().numpy().tolist()
        winner_idx = int(np.argmax(rewards))
        for idx, (path, reward) in enumerate(zip(candidates, rewards)):
            rows.append({
                "uid": uid,
                "oid": oid,
                "candidate_idx": idx,
                "compatibility_reward": float(reward),
                "is_winner": idx == winner_idx,
                "image_path": str(path.resolve()),
            })
        ground_truth_paths = list(task_dir.glob("*_grd.png")) + list(task_dir.glob("grd.jpg")) + list(task_dir.glob("grd.png"))
        task_summaries.append({
            "uid": uid,
            "oid": oid,
            "candidate_count": len(candidates),
            "winner_idx": winner_idx,
            "winner_reward": float(rewards[winner_idx]),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "winner_image": str(candidates[winner_idx].resolve()),
            "ground_truth_reference": str(ground_truth_paths[0].resolve()) if ground_truth_paths else None,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "difashion_compatibility_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    winners_dir = args.output_dir / "winners"
    winners_dir.mkdir(parents=True, exist_ok=True)
    for summary in task_summaries:
        destination = winners_dir / f"{summary['uid']}_{summary['oid']}_winner.png"
        shutil.copy2(summary["winner_image"], destination)
        summary["winner_image"] = str(destination.resolve())

    report = {
        "reward": "sigmoid(FashionEvaluator(OpenCLIP_ViT-H-14(candidate), fixed_iFashion_context))",
        "task_count": len(task_summaries),
        "candidate_count": len(rows),
        "mean_candidate_reward": float(np.mean([row["compatibility_reward"] for row in rows])),
        "mean_winner_reward": float(np.mean([item["winner_reward"] for item in task_summaries])),
        "tasks": task_summaries,
    }
    with (args.output_dir / "difashion_compatibility_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
