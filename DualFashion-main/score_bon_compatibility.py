"""Score PFITB candidates with the frozen iFashion compatibility evaluator.

The script keeps the test ground-truth image strictly out of candidate
selection. It only supplies it as a separately labelled visual reference.
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import open_clip

from eval.compatibility_evaluator.compatibility_net import FashionEvaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing checkpoint subdirectories and generated gen.npy/grd.npy.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="iFashion directory containing cnn_features_clip.npy and fitb_test_dict.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to receive scores, selected .npy payloads, and copied images.",
    )
    parser.add_argument("--checkpoint", default="0057501")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reward",
        choices=("compatibility", "personal", "bottleneck", "pcar"),
        default="compatibility",
        help="Candidate-selection reward. Bottleneck uses min of per-group Compatibility/Personal percentiles.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help="test_history_clipembs.npy; required for --reward personal.",
    )
    parser.add_argument("--pcar-checkpoint", type=Path, default=None, help="Frozen PCAR best.pt; required for --reward pcar.")
    parser.add_argument(
        "--evaluator-ckpt",
        type=Path,
        default=Path(__file__).resolve().parent
        / "eval/compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth",
    )
    return parser.parse_args()


def load_candidate_features(model, transform, image_paths, batch_size, device):
    features = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="OpenCLIP features"):
            batch_paths = image_paths[start : start + batch_size]
            images = []
            for image_path in batch_paths:
                with Image.open(image_path) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images).to(device)
            features.append(model.encode_image(batch).float().cpu())
    return torch.cat(features, dim=0)


def copy_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination.resolve())


def load_pcar(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(path)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from fashion_psp.rewards.train_pcar import PCAR

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    category_ids = [int(value) for value in payload["category_ids"]]
    model = PCAR(len(category_ids), config["hidden_dim"], config["embedding_dim"], config["category_dim"])
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), {category: index for index, category in enumerate(category_ids)}


@torch.inference_mode()
def score_pcar(model, category_lookup, candidate_features, history_embedding, context_embedding, category: int, device):
    count = len(candidate_features)
    candidates = candidate_features.to(device).unsqueeze(1)
    history = history_embedding.to(device).reshape(1, -1).expand(count, -1)
    context = context_embedding.to(device).reshape(1, -1).expand(count, -1)
    fallback = category not in category_lookup
    if fallback:
        # Never alias an unseen category to an arbitrary trained category.
        category_embedding = model.category_embedding.weight.mean(dim=0, keepdim=True).expand(count, -1)
        scores = model.forward_with_category_embedding(history, context, category_embedding, candidates)
    else:
        categories = torch.full((count,), category_lookup[category], dtype=torch.long, device=device)
        scores = model(history, context, categories, candidates)
    return scores.squeeze(1).float().cpu().numpy(), fallback


def save_selected_payload(output_dir, checkpoint, grouped_records, grd_data, kind):
    """Write one official-evaluation payload per independent N-candidate group."""
    records_by_group = defaultdict(list)
    for record in grouped_records:
        records_by_group[record["group_id"]].append(record)

    for group_id, group_records in records_by_group.items():
        payload = defaultdict(dict)
        ground_truth = defaultdict(dict)
        for record in group_records:
            uid, oid = record["uid"], record["oid"]
            source = record[kind]
            source_record = record["source_record"]
            payload[uid][oid] = {
                "image_paths": [source["image_path"]],
                "cates": np.asarray([source_record["cates"][source["candidate_idx"]]], dtype=np.int64),
                "candidate_seeds": [source["seed"]],
                "incomplete_outfit": source_record["incomplete_outfit"],
                "outfits": source_record["outfits"],
            }
            ground_truth[uid][oid] = grd_data[uid][oid]

        target = output_dir / "selected_for_official_eval" / kind / f"group_{group_id}" / checkpoint
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "gen.npy", {uid: dict(items) for uid, items in payload.items()})
        np.save(target / "grd.npy", {uid: dict(items) for uid, items in ground_truth.items()})


def main():
    args = parse_args()
    if args.group_size < 2:
        raise ValueError("--group-size must be at least 2 for Best-of-N.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    input_ckpt_dir = args.input_dir / args.checkpoint
    gen_data = np.load(input_ckpt_dir / "gen.npy", allow_pickle=True).item()
    grd_data = np.load(input_ckpt_dir / "grd.npy", allow_pickle=True).item()
    fitb_dict = np.load(args.data_root / "fitb_test_dict.npy", allow_pickle=True).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    history = None
    if args.reward in {"personal", "bottleneck", "pcar"}:
        history_path = args.history_path or (args.data_root / "processed" / "test_history_clipembs.npy")
        history = np.load(history_path, allow_pickle=True).item()
    if args.reward == "pcar":
        if args.pcar_checkpoint is None:
            raise ValueError("--pcar-checkpoint is required for --reward pcar")
        pcar, pcar_categories = load_pcar(args.pcar_checkpoint, device)
    else:
        pcar = pcar_categories = None

    all_image_paths = []
    task_records = []
    for uid, oid_map in gen_data.items():
        for oid, record in oid_map.items():
            paths = list(record["image_paths"])
            seeds = list(record.get("candidate_seeds", []))
            if len(paths) != len(seeds):
                raise ValueError(f"{uid}_{oid}: image/seed count mismatch ({len(paths)} vs {len(seeds)}).")
            if len(paths) % args.group_size:
                raise ValueError(f"{uid}_{oid}: {len(paths)} candidates cannot be split into N={args.group_size} groups.")
            outfit = list(fitb_dict[int(uid)][int(oid)])
            if outfit.count(0) != 1:
                raise ValueError(f"{uid}_{oid}: PFITB outfit must contain exactly one blank: {outfit}")
            task_records.append((int(uid), int(oid), record, outfit, len(all_image_paths)))
            all_image_paths.extend(paths)

    print(f"Scoring {len(all_image_paths)} candidates on {device}.")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b-s32b-b79K"
    )
    clip_model = clip_model.to(device).eval()
    candidate_features = load_candidate_features(
        clip_model, preprocess, all_image_paths, args.batch_size, device
    )
    del clip_model
    torch.cuda.empty_cache()

    evaluator = None
    if args.reward in {"compatibility", "bottleneck"}:
        evaluator = FashionEvaluator(cnn_feat_dim=1024)
        state_dict = torch.load(args.evaluator_ckpt, map_location="cpu")
        evaluator.load_state_dict(state_dict)
        evaluator = evaluator.to(device).eval()

    score_rows = []
    grouped_records = []
    for uid, oid, record, outfit, offset in task_records:
        blank_idx = outfit.index(0)
        context_features = torch.from_numpy(np.asarray([item_features[item_id] for item_id in outfit])).float()
        candidate_slice = candidate_features[offset : offset + len(record["image_paths"])]
        compatibility_scores = None
        personal_scores = None
        pcar_scores = None
        if args.reward in {"compatibility", "bottleneck"}:
            outfits = context_features.unsqueeze(0).repeat(len(candidate_slice), 1, 1)
            outfits[:, blank_idx, :] = candidate_slice
            with torch.no_grad():
                compatibility_scores = torch.sigmoid(evaluator(outfits.to(device))).cpu().numpy()
        if args.reward in {"personal", "bottleneck"}:
            cate = int(np.asarray(record["cates"])[0])
            try:
                history_embedding = torch.as_tensor(history[int(uid)][cate], dtype=torch.float32, device=device)
            except KeyError as exc:
                raise KeyError(f"Missing test history embedding for uid={uid}, category={cate}") from exc
            candidate_norm = F.normalize(candidate_slice.to(device), dim=-1)
            history_norm = F.normalize(history_embedding.unsqueeze(0), dim=-1)
            with torch.no_grad():
                personal_scores = (100.0 * (candidate_norm @ history_norm.T).squeeze(1)).cpu().numpy()
        if args.reward == "pcar":
            cate = int(np.asarray(record["cates"])[0])
            try:
                history_embedding = torch.as_tensor(history[int(uid)][cate], dtype=torch.float32)
            except KeyError as exc:
                raise KeyError(f"Missing test history embedding for uid={uid}, category={cate}") from exc
            context_embedding = context_features[[index for index in range(len(outfit)) if index != blank_idx]].mean(dim=0)
            pcar_scores, pcar_unknown_category_fallback = score_pcar(
                pcar, pcar_categories, candidate_slice, history_embedding, context_embedding, cate, device
            )
        else:
            pcar_unknown_category_fallback = False

        def percentile(values):
            values = np.asarray(values, dtype=np.float64)
            if len(values) <= 1:
                return np.ones_like(values)
            order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
            return order / float(len(values) - 1)

        if args.reward == "compatibility":
            scores = compatibility_scores
        elif args.reward == "personal":
            scores = personal_scores
        elif args.reward == "pcar":
            scores = pcar_scores
        else:
            scores = np.full(len(candidate_slice), -np.inf, dtype=np.float64)

        for group_idx, group_start in enumerate(range(0, len(candidate_slice), args.group_size), start=1):
            group_end = group_start + args.group_size
            if args.reward == "bottleneck":
                compatibility_percentiles = percentile(compatibility_scores[group_start:group_end])
                personal_percentiles = percentile(personal_scores[group_start:group_end])
                group_scores = np.minimum(compatibility_percentiles, personal_percentiles)
                scores[group_start:group_end] = group_scores
            else:
                group_scores = scores[group_start:group_end]
            winner_rel_idx = int(np.argmax(group_scores))
            winner_idx = group_start + winner_rel_idx
            standard_idx = group_start
            standard = {
                "candidate_idx": standard_idx,
                "seed": int(record["candidate_seeds"][standard_idx]),
                "image_path": str(record["image_paths"][standard_idx]),
                "reward": float(scores[standard_idx]),
            }
            winner = {
                "candidate_idx": winner_idx,
                "seed": int(record["candidate_seeds"][winner_idx]),
                "image_path": str(record["image_paths"][winner_idx]),
                "reward": float(scores[winner_idx]),
            }
            grouped_records.append(
                {
                    "uid": uid,
                    "oid": oid,
                    "group_id": group_idx,
                    "standard": standard,
                    "bon": winner,
                    "source_record": record,
                }
            )
            for rel_idx, score in enumerate(group_scores):
                candidate_idx = group_start + rel_idx
                score_rows.append(
                    {
                        "uid": uid,
                        "oid": oid,
                        "group_id": group_idx,
                        "candidate_idx": candidate_idx,
                        "seed": int(record["candidate_seeds"][candidate_idx]),
                        f"{args.reward}_reward": float(score),
                        "pcar_unknown_category_fallback": pcar_unknown_category_fallback,
                        **({
                            "compatibility_score": float(compatibility_scores[candidate_idx]),
                            "personal_score": float(personal_scores[candidate_idx]),
                            "compatibility_percentile": float(compatibility_percentiles[rel_idx]),
                            "personal_percentile": float(personal_percentiles[rel_idx]),
                        } if args.reward == "bottleneck" else {}),
                        "is_standard": candidate_idx == standard_idx,
                        "is_bon_winner": candidate_idx == winner_idx,
                        "image_path": str(record["image_paths"][candidate_idx]),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_name = f"candidate_{args.reward}_scores.csv"
    with (args.output_dir / score_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    gallery_records = []
    for record in grouped_records:
        stem = f"{record['uid']}_{record['oid']}_group{record['group_id']}"
        standard_dest = args.output_dir / "selected_images" / "standard" / f"{stem}_seed{record['standard']['seed']}.png"
        bon_dest = args.output_dir / "selected_images" / "bon_n4" / f"{stem}_seed{record['bon']['seed']}.png"
        ground_source = grd_data[record["uid"]][record["oid"]]["image_paths"][0]
        ground_dest = args.output_dir / "selected_images" / "ground_truth_reference" / f"{stem}.png"
        gallery_records.append(
            {
                **{key: value for key, value in record.items() if key != "source_record"},
                "standard_image": copy_image(record["standard"]["image_path"], standard_dest),
                "bon_image": copy_image(record["bon"]["image_path"], bon_dest),
                "ground_truth_reference": copy_image(ground_source, ground_dest),
            }
        )

    standard_scores = [row["standard"]["reward"] for row in grouped_records]
    bon_scores = [row["bon"]["reward"] for row in grouped_records]
    summary = {
        "reward": {
            "personal": "100*cosine(normalized OpenCLIP candidate, normalized test_history_clipembs[uid][target_category])",
            "compatibility": "sigmoid(FashionEvaluator(OpenCLIP_ViT-H-14(candidate), fixed_iFashion_context))",
            "bottleneck": "min(percentile(Compatibility), percentile(Personal Sim)) within each candidate group",
            "pcar": "PCAR(history, observed FITB context, target category, candidate); trained on iFashion train only",
        }[args.reward],
        "selection_rule": f"Best-of-{args.group_size}: maximum {args.reward} reward within each consecutive group.",
        "standard_rule": "First candidate in the same N-candidate group; no reward-based selection.",
        "task_count": len(task_records),
        "group_count": len(grouped_records),
        "candidate_count": len(score_rows),
        "pcar_unknown_category_fallback_task_count": int(sum(
            bool(row["pcar_unknown_category_fallback"]) for row in score_rows[::args.group_size]
        )),
        "standard_mean_reward": float(np.mean(standard_scores)),
        "bon_mean_reward": float(np.mean(bon_scores)),
        "bon_minus_standard": float(np.mean(bon_scores) - np.mean(standard_scores)),
        "groups": gallery_records,
    }
    with (args.output_dir / "bon_n4_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_selected_payload(args.output_dir, args.checkpoint, grouped_records, grd_data, "standard")
    save_selected_payload(args.output_dir, args.checkpoint, grouped_records, grd_data, "bon")
    print(json.dumps({key: summary[key] for key in summary if key != "groups"}, indent=2))
    print(f"Saved candidate scores, selected images, and official-evaluation payloads to {args.output_dir}")


if __name__ == "__main__":
    main()
