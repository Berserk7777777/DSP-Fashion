"""Select PSP survivors from the offline x0 cache with compatibility reward.

The cache contains all eight independent trajectories at steps 7, 14, and 28.
This script applies the fixed PSP schedule 8 -> 4 -> 2 -> 1 without running
the diffusion model again, writes per-candidate reward records, and packs the
selected final image for DualFashion's official PFITB evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T

from eval.compatibility_evaluator.compatibility_net import FashionEvaluator


TASK_PATTERN = re.compile(r"^psp_offline_group(?P<group>\d+)_task(?P<task>\d+)_seed(?P<seed>\d+)$")
SCHEDULE = ((7, 4), (14, 2), (28, 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fitb-file", default="processed/fitb_test.npy")
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="0057501")
    parser.add_argument("--max-samples", type=int, default=2207)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reward",
        choices=("compatibility", "personal", "bottleneck", "pcar"),
        default="compatibility",
        help="PSP pruning reward. Bottleneck uses min of per-active-set Compatibility/Personal percentiles.",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help="test_history_clipembs.npy; required for --reward personal.",
    )
    parser.add_argument(
        "--evaluator-ckpt",
        type=Path,
        default=Path("eval/compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth"),
    )
    parser.add_argument("--pcar-checkpoint", type=Path, default=None, help="Frozen PCAR best.pt; required for --reward pcar.")
    parser.add_argument(
        "--clip-weights",
        type=Path,
        default=None,
        help="Local OpenCLIP ViT-H/14 .safetensors/.bin; omitted uses the cached laion model.",
    )
    parser.add_argument("--resume-from", default="auto", help="auto, none, or a non-negative task index")
    return parser.parse_args()


def parse_resume_start(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "auto"}:
        return 0
    if text == "none":
        return None
    try:
        index = int(text)
    except ValueError as exc:
        raise ValueError("--resume-from must be auto, none, or a non-negative task index") from exc
    if index < 0:
        raise ValueError("--resume-from task index must be non-negative")
    return index


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_local_clip(clip_weights: Path | None):
    import open_clip

    if clip_weights is None:
        return open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b-s32b-b79K")
    if not clip_weights.exists():
        raise FileNotFoundError(clip_weights)
    if clip_weights.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
        state = load_file(str(clip_weights), device="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"OpenCLIP checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
        return model, None, preprocess
    return open_clip.create_model_and_transforms("ViT-H-14", pretrained=str(clip_weights))


def load_pcar(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(path)
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from fashion_psp.rewards.train_pcar import PCAR

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    categories = [int(value) for value in payload["category_ids"]]
    model = PCAR(len(categories), config["hidden_dim"], config["embedding_dim"], config["category_dim"])
    model.load_state_dict(payload["model"])
    return model.to(device).eval(), {category: index for index, category in enumerate(categories)}


@torch.inference_mode()
def score_paths(
    image_paths: list[str],
    context_features: torch.Tensor,
    blank_idx: int,
    clip_model: torch.nn.Module,
    preprocess,
    evaluator: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    reward: str,
    history_embedding: torch.Tensor | None,
    pcar_model: torch.nn.Module | None,
    pcar_categories: dict[int, int] | None,
    pcar_context: torch.Tensor | None,
    target_category: int | None,
) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for image_path in batch_paths:
            with Image.open(image_path).convert("RGB") as image:
                images.append(preprocess(image))
        image_batch = torch.stack(images).to(device)
        candidate_features = clip_model.encode_image(image_batch).float()
        compatibility_scores = None
        personal_scores = None
        pcar_scores = None
        if reward in {"compatibility", "bottleneck"}:
            outfits = context_features.unsqueeze(0).repeat(len(candidate_features), 1, 1)
            outfits[:, blank_idx, :] = candidate_features.cpu()
            logits = evaluator(outfits.to(device))
            compatibility_scores = torch.sigmoid(logits).detach().cpu().tolist()
            del outfits, logits
        if reward in {"personal", "bottleneck"}:
            candidate_features = F.normalize(candidate_features, dim=-1)
            history_norm = F.normalize(history_embedding, dim=-1).unsqueeze(0)
            personal_scores = (100.0 * (candidate_features @ history_norm.T).squeeze(1)).detach().cpu().tolist()
        if reward == "pcar":
            if pcar_model is None or pcar_categories is None or pcar_context is None or target_category is None:
                raise RuntimeError("PCAR scorer was not initialized")
            count = len(candidate_features)
            history_batch = history_embedding.reshape(1, -1).expand(count, -1)
            context_batch = pcar_context.reshape(1, -1).expand(count, -1)
            if target_category in pcar_categories:
                pcar_scores = pcar_model(
                    history_batch,
                    context_batch,
                    torch.full((count,), pcar_categories[target_category], dtype=torch.long, device=device),
                    candidate_features.unsqueeze(1),
                ).squeeze(1)
                pcar_unknown_category_fallback = False
            else:
                # Use a vocabulary-level representation for an unseen test
                # category; never alias it to a known category or use GT.
                category_embedding = pcar_model.category_embedding.weight.mean(dim=0, keepdim=True).expand(count, -1)
                pcar_scores = pcar_model.forward_with_category_embedding(
                    history_batch, context_batch, category_embedding, candidate_features.unsqueeze(1)
                ).squeeze(1)
                pcar_unknown_category_fallback = True
            pcar_scores = pcar_scores.detach().float().cpu().tolist()
        else:
            pcar_unknown_category_fallback = False
        for index in range(len(batch_paths)):
            scores.append({
                "compatibility": float(compatibility_scores[index]) if compatibility_scores is not None else float("nan"),
                "personal": float(personal_scores[index]) if personal_scores is not None else float("nan"),
                "pcar": float(pcar_scores[index]) if pcar_scores is not None else float("nan"),
                "pcar_unknown_category_fallback": pcar_unknown_category_fallback,
            })
        del image_batch, candidate_features
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return scores


def cache_complete(task_dir: Path) -> bool:
    metadata_path = task_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        cache = payload["offline_candidate_cache"]
        rows = cache["images"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return bool(cache.get("enabled")) and list(cache.get("steps", [])) == [7, 14, 28] and len(rows) == 24 and all(
        Path(row.get("image_path", "")).exists() for row in rows
    )


def task_paths(cache_root: Path, group: int, task_index: int, seed: int) -> Path:
    return cache_root / f"psp_offline_group{group}_task{task_index}_seed{seed}"


def build_official_records(
    task_index: int,
    uid: int,
    oid: int,
    fitb: dict,
    grd: dict,
    image_paths: np.ndarray,
    image_root: Path,
    final_image: str,
    final_seed: int,
) -> tuple[dict, dict]:
    incomplete = np.asarray(fitb["outfits"][task_index], dtype=np.int64).copy()
    blank_idx = int(np.flatnonzero(incomplete == 0)[0])
    full_outfit = np.asarray(grd["outfits"][task_index], dtype=np.int64)
    target_item_id = int(full_outfit[blank_idx])
    target_category = int(fitb["category"][task_index][blank_idx])

    context_paths = []
    for index, item_id in enumerate(full_outfit.tolist()):
        if index == blank_idx:
            continue
        path = image_root / str(image_paths[int(item_id)])
        if not path.exists():
            raise FileNotFoundError(path)
        context_paths.append(str(path.resolve()))

    ground_path = image_root / str(image_paths[target_item_id])
    if not ground_path.exists():
        raise FileNotFoundError(ground_path)
    generated = {
        "image_paths": [str(Path(final_image).resolve())],
        "cates": np.asarray([target_category], dtype=np.int64),
        "candidate_seeds": [int(final_seed)],
        "incomplete_outfit": context_paths,
        "outfits": full_outfit,
    }
    ground = {
        "image_paths": [str(ground_path.resolve())],
        "cates": np.asarray([target_category], dtype=np.int64),
        "incomplete_outfit": context_paths,
        "outfits": full_outfit,
    }
    return generated, ground


def record_to_json(record: dict) -> dict:
    """Convert NumPy arrays in an official-eval record to JSON lists."""
    return {
        **record,
        "cates": np.asarray(record["cates"]).tolist(),
        "outfits": np.asarray(record["outfits"]).tolist(),
    }


def record_from_json(record: dict) -> dict:
    """Restore arrays after loading a resumable task selection."""
    return {
        **record,
        "cates": np.asarray(record["cates"], dtype=np.int64),
        "outfits": np.asarray(record["outfits"], dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    if args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("--max-samples and --batch-size must be positive")
    resume_start = parse_resume_start(args.resume_from)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fitb = np.load(args.data_root / args.fitb_file, allow_pickle=True).item()
    grd = np.load(args.data_root / "test_grd_dict.npy", allow_pickle=True).item()
    item_features = np.load(args.data_root / "cnn_features_clip.npy", mmap_mode="r")
    image_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    total_samples = min(len(fitb["uids"]), args.max_samples)
    if resume_start is not None and resume_start >= total_samples:
        raise ValueError("--resume-from must be smaller than --max-samples")

    expected_dirs = [task_paths(args.cache_root, args.group, index, args.base_seed) for index in range(total_samples)]
    missing = [index for index, path in enumerate(expected_dirs) if not cache_complete(path)]
    if missing:
        preview = ", ".join(str(index) for index in missing[:20])
        suffix = "..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"PSP offline cache is incomplete: {len(missing)}/{total_samples} tasks missing or invalid "
            f"(first task indices: {preview}{suffix}). Finish the cache runner first."
        )

    history = None
    if args.reward in {"personal", "bottleneck", "pcar"}:
        history_path = args.history_path or (args.data_root / "processed" / "test_history_clipembs.npy")
        history = np.load(history_path, allow_pickle=True).item()

    clip_model, _, preprocess = load_local_clip(args.clip_weights)
    clip_model = clip_model.to(device).eval()
    evaluator = None
    if args.reward in {"compatibility", "bottleneck"}:
        evaluator = FashionEvaluator(cnn_feat_dim=1024)
        state = torch.load(args.evaluator_ckpt, map_location="cpu")
        evaluator.load_state_dict(state)
        evaluator = evaluator.to(device).eval()
    if args.reward == "pcar":
        if args.pcar_checkpoint is None:
            raise ValueError("--pcar-checkpoint is required for --reward pcar")
        pcar_model, pcar_categories = load_pcar(args.pcar_checkpoint, device)
    else:
        pcar_model = pcar_categories = None

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    task_output = output_dir / "tasks"
    selected_images = output_dir / "selected_images"
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    packed_gen: dict[int, dict[int, dict]] = {}
    packed_grd: dict[int, dict[int, dict]] = {}
    completed_selection_indices: set[int] = set()
    for task_index in range(total_samples):
        selection_path = task_output / f"task_{task_index:04d}.json"
        if not selection_path.exists():
            continue
        try:
            existing = json.loads(selection_path.read_text(encoding="utf-8"))
            if not Path(existing["final"]["image_path"]).exists():
                continue
            completed_selection_indices.add(task_index)
            summary_rows.append(existing["summary"])
            packed_gen.setdefault(int(existing["uid"]), {})[
                int(existing["oid"])
            ] = record_from_json(existing["generated_record"])
            packed_grd.setdefault(int(existing["uid"]), {})[
                int(existing["oid"])
            ] = record_from_json(existing["ground_record"])
            all_rows.extend(existing.get("score_rows", []))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue

    reward_key = f"{args.reward}_reward"
    progress = tqdm(range(total_samples), desc=f"PSP {args.reward} selection", unit="task")

    for task_index in progress:
        if resume_start is not None and task_index < resume_start:
            progress.set_postfix_str("resume skip")
            continue
        task_dir = expected_dirs[task_index]
        selection_path = task_output / f"task_{task_index:04d}.json"
        if task_index in completed_selection_indices:
            progress.set_postfix_str("selected cached")
            continue

        metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
        uid = int(metadata["uid"])
        oid = int(metadata["oid"])
        cache_rows = metadata["offline_candidate_cache"]["images"]
        by_step: dict[int, dict[int, str]] = {7: {}, 14: {}, 28: {}}
        for row in cache_rows:
            by_step[int(row["step"])][int(row["seed"])] = str(row["image_path"])
        seeds = [int(seed) for seed in metadata["candidate_seeds"]]
        if any(set(by_step[step]) != set(seeds) for step in by_step):
            raise ValueError(f"{task_dir}: cache seeds are not identical across steps")

        incomplete = np.asarray(fitb["outfits"][task_index], dtype=np.int64)
        blank_idx = int(np.flatnonzero(incomplete == 0)[0])
        full_outfit = np.asarray(grd["outfits"][task_index], dtype=np.int64)
        context_features = torch.from_numpy(np.asarray([item_features[item_id] for item_id in full_outfit])).float()
        target_category = int(fitb["category"][task_index][blank_idx])
        if args.reward in {"personal", "bottleneck", "pcar"}:
            try:
                history_embedding = torch.as_tensor(history[uid][target_category], dtype=torch.float32, device=device)
            except KeyError as exc:
                raise KeyError(f"Missing test history embedding for uid={uid}, category={target_category}") from exc
        else:
            history_embedding = None
        pcar_context = context_features[[index for index in range(len(full_outfit)) if index != blank_idx]].mean(dim=0).to(device)
        active = list(seeds)
        score_rows: list[dict] = []
        trace: list[dict] = []

        for step, keep in SCHEDULE:
            # Score only currently active particles. This mirrors the online
            # PSP compute budget: 8 candidates at step 7, then 4, then 2.
            step_seeds = list(active)
            step_paths = [by_step[step][seed] for seed in step_seeds]
            raw_step_scores = score_paths(
                step_paths,
                context_features,
                blank_idx,
                clip_model,
                preprocess,
                evaluator,
                device,
                args.batch_size,
                args.reward,
                history_embedding,
                pcar_model,
                pcar_categories,
                pcar_context,
                target_category,
            )
            pcar_unknown_category_fallback = any(
                bool(row.get("pcar_unknown_category_fallback", False)) for row in raw_step_scores
            )
            compatibility_by_seed = {seed: row["compatibility"] for seed, row in zip(step_seeds, raw_step_scores)}
            personal_by_seed = {seed: row["personal"] for seed, row in zip(step_seeds, raw_step_scores)}
            pcar_by_seed = {seed: row["pcar"] for seed, row in zip(step_seeds, raw_step_scores)}
            if args.reward == "compatibility":
                score_by_seed = compatibility_by_seed
            elif args.reward == "personal":
                score_by_seed = personal_by_seed
            elif args.reward == "pcar":
                score_by_seed = pcar_by_seed
            else:
                def percentile_map(values):
                    ordered = sorted(values, key=lambda seed: (values[seed], seed))
                    if len(ordered) <= 1:
                        return {seed: 1.0 for seed in ordered}
                    return {seed: index / float(len(ordered) - 1) for index, seed in enumerate(ordered)}
                c_pct = percentile_map(compatibility_by_seed)
                p_pct = percentile_map(personal_by_seed)
                score_by_seed = {seed: min(c_pct[seed], p_pct[seed]) for seed in step_seeds}
            before = list(active)
            ranked = sorted(active, key=lambda seed: (-score_by_seed[seed], seed))
            active = ranked[:keep]
            trace.append(
                {
                    "step": step,
                    "keep": keep,
                    "active_before": before,
                    "active_after": active,
                    "scores": [
                        {
                            "seed": seed,
                            reward_key: score_by_seed[seed],
                            "compatibility_score": compatibility_by_seed[seed],
                            "personal_score": personal_by_seed[seed],
                            "pcar_score": pcar_by_seed[seed],
                            "pcar_unknown_category_fallback": pcar_unknown_category_fallback,
                        }
                        for seed in step_seeds
                    ],
                }
            )
            for seed in step_seeds:
                score = score_by_seed[seed]
                score_rows.append(
                    {
                        "task_index": task_index,
                        "uid": uid,
                        "oid": oid,
                        "step": step,
                        "seed": seed,
                        reward_key: score,
                        "compatibility_score": compatibility_by_seed[seed],
                        "personal_score": personal_by_seed[seed],
                        "pcar_score": pcar_by_seed[seed],
                        "pcar_unknown_category_fallback": pcar_unknown_category_fallback,
                        "active_before": seed in before,
                        "selected_after": seed in active,
                        "image_path": by_step[step][seed],
                    }
                )

        final_seed = active[0]
        final_path = by_step[28][final_seed]
        final_trace_seeds = [row["seed"] for row in trace[-1]["scores"]]
        oracle_seed = max(final_trace_seeds, key=lambda seed: (next(row[reward_key] for row in trace[-1]["scores"] if row["seed"] == seed), -seed))
        oracle_score = next(row[reward_key] for row in trace[-1]["scores"] if row["seed"] == oracle_seed)
        final_score = next(row[reward_key] for row in trace[-1]["scores"] if row["seed"] == final_seed)
        generated_record, ground_record = build_official_records(
            task_index,
            uid,
            oid,
            fitb,
            grd,
            image_paths,
            args.image_root,
            final_path,
            final_seed,
        )
        destination = selected_images / f"{uid}_{oid}_seed{final_seed}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_path, destination)
        generated_record["image_paths"] = [str(destination.resolve())]
        selection = {
            "task_index": task_index,
            "uid": uid,
            "oid": oid,
            "candidate_seeds": seeds,
            "pcar_unknown_category_fallback": any(
                bool(row.get("pcar_unknown_category_fallback", False)) for row in score_rows
            ),
            "reward": {
                "personal": "100*cosine(normalized OpenCLIP candidate, normalized test_history_clipembs[uid][target_category])",
                "compatibility": "sigmoid(FashionEvaluator(OpenCLIP ViT-H/14(candidate), fixed iFashion context))",
                "bottleneck": "min(percentile(Compatibility), percentile(Personal Sim)) within active PSP set",
                "pcar": "PCAR(history, observed FITB context, target category, candidate); trained on iFashion train only",
            }[args.reward],
            "schedule": [{"step": step, "keep": keep} for step, keep in SCHEDULE],
            "trace": trace,
            "final": {"seed": final_seed, "reward": final_score, "image_path": str(destination.resolve())},
            "step28_oracle": {"seed": oracle_seed, "reward": oracle_score, "image_path": by_step[28][oracle_seed]},
            "generated_record": record_to_json(generated_record),
            "ground_record": record_to_json(ground_record),
            "score_rows": score_rows,
            "summary": {
                "task_index": task_index,
                "uid": uid,
                "oid": oid,
                "psp_seed": final_seed,
                "psp_reward": final_score,
                "step28_oracle_seed": oracle_seed,
                "step28_oracle_reward": oracle_score,
                "psp_regret": oracle_score - final_score,
                "pcar_unknown_category_fallback": any(
                    bool(row.get("pcar_unknown_category_fallback", False)) for row in score_rows
                ),
            },
        }
        atomic_json(selection_path, selection)
        summary_rows.append(selection["summary"])
        packed_gen.setdefault(uid, {})[oid] = generated_record
        packed_grd.setdefault(uid, {})[oid] = ground_record
        completed_selection_indices.add(task_index)
        all_rows.extend(score_rows)
        progress.set_postfix_str(f"task={task_index} seed={final_seed} reward={final_score:.4f}")

    checkpoint_dir = output_dir / args.checkpoint
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.save(checkpoint_dir / "gen.npy", packed_gen)
    np.save(checkpoint_dir / "grd.npy", packed_grd)
    manifest = {
        "source_cache": str(args.cache_root.resolve()),
        "group": args.group,
        "base_seed": args.base_seed,
        "task_count": sum(len(oid_map) for oid_map in packed_gen.values()),
        "checkpoint": args.checkpoint,
        "reward": {
            "personal": "100*cosine(normalized OpenCLIP candidate, normalized test_history_clipembs[uid][target_category])",
            "compatibility": "sigmoid(FashionEvaluator(OpenCLIP ViT-H-14(candidate), fixed iFashion context))",
            "bottleneck": "min(percentile(Compatibility), percentile(Personal Sim)) within active PSP set",
            "pcar": "PCAR(history, observed FITB context, target category, candidate); trained on iFashion train only",
        }[args.reward],
        "selection": f"PSP {args.reward} pruning at step 7 keep 4, step 14 keep 2, step 28 keep 1",
        "step28_oracle_is_diagnostic_only": True,
        "pcar_unknown_category_fallback": "Unseen target categories use the mean trained PCAR category embedding; no target-image information is used.",
    }
    atomic_json(output_dir / "pack_manifest.json", manifest)
    with (output_dir / f"psp_{args.reward}_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        if all_rows:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    with (output_dir / "psp_selection_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if summary_rows:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(sorted(summary_rows, key=lambda row: int(row["task_index"])))
    print(json.dumps(manifest, indent=2))
    print(f"Saved official-evaluation payload: {checkpoint_dir}")


if __name__ == "__main__":
    main()
