"""Offline reward cross-validation for DualFashion iFashion GOR caches.

The cache stores complete sequential-outfit trajectories.  This evaluator
scores a complete four-item candidate outfit and writes frozen selections for
Compatibility, Personal, and Bottleneck rewards.  Optional GT-Sim and GAA are
strictly post-selection diagnostics: GT paths are not loaded until every
selection JSON has been written.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


REWARDS = ("compatibility", "personal", "bottleneck", "pcar")
ATTRIBUTE_BANKS = {
    "color": ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"],
    "pattern": ["solid", "striped", "plaid", "floral", "polka dot", "geometric", "animal print", "quilted", "textured"],
    "material": ["leather", "faux leather", "suede", "canvas", "denim", "knit", "cotton", "wool", "synthetic material"],
    "style": ["casual", "formal", "sporty", "elegant", "vintage", "minimalist", "streetwear"],
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, default=root / "DualFashion-main")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--compatibility-ckpt", type=Path, default=None)
    parser.add_argument("--pcar-checkpoint", type=Path, default=None, help="Frozen PCAR best.pt; required to evaluate PCAR.")
    parser.add_argument("--psp-keeps", default="4,2,1", help="Survivors after each cached PSP step.")
    parser.add_argument("--with-gt-metrics", action="store_true")
    parser.add_argument("--gaa-clip-weights", type=Path, default=Path(r"C:\Users\Administrator\.cache\clip\ViT-L-14.pt"))
    parser.add_argument("--resume-from", default="auto", help="auto, none, or non-negative task index")
    return parser.parse_args()


def parse_resume(value: str) -> int | None:
    text = value.strip().lower()
    if text == "none":
        return None
    if text in {"", "auto"}:
        return 0
    try:
        result = int(text)
    except ValueError as exc:
        raise ValueError("--resume-from must be auto, none, or a non-negative task index") from exc
    if result < 0:
        raise ValueError("--resume-from must be non-negative")
    return result


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batched(values: list, batch_size: int):
    for offset in range(0, len(values), batch_size):
        yield values[offset : offset + batch_size]


@torch.inference_mode()
def encode_images(model, preprocess, paths: list[Path], device: torch.device, batch_size: int, normalize: bool = True) -> np.ndarray:
    def load_one(path: Path):
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"))

    result = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for chunk in batched(paths, batch_size):
            pixels = torch.stack(list(pool.map(load_one, chunk))).to(device, dtype=next(model.parameters()).dtype)
            embedding = model.encode_image(pixels)
            embedding = embedding.float()
            if normalize:
                embedding = F.normalize(embedding, dim=-1)
            result.append(embedding.cpu().numpy())
    return np.concatenate(result, axis=0)


def percentile_scores(values: dict[int, float]) -> dict[int, float]:
    ordered = sorted(values, key=lambda seed: (values[seed], seed))
    if len(ordered) == 1:
        return {ordered[0]: 1.0}
    return {seed: index / float(len(ordered) - 1) for index, seed in enumerate(ordered)}


def expected_schedule(cache_steps: list[int], particle_count: int, keeps_text: str) -> list[tuple[int, int]]:
    if cache_steps == [28]:
        return [(28, 1)]
    keeps = [int(value.strip()) for value in keeps_text.split(",") if value.strip()]
    if len(keeps) != len(cache_steps) or keeps[-1] != 1:
        raise ValueError("--psp-keeps must provide one keep count per cached PSP step and end at 1")
    if any(keep < 1 or keep > particle_count for keep in keeps):
        raise ValueError("invalid --psp-keeps value")
    return list(zip(cache_steps, keeps))


def canonical_paths(metadata: dict, step: int, seed: int) -> list[Path]:
    rows = [
        row for row in metadata["offline_candidate_cache"]["images"]
        if int(row["step"]) == step and int(row["candidate_seed"]) == seed
    ]
    rows.sort(key=lambda row: int(row["original_outfit_position"]))
    positions = [int(row["original_outfit_position"]) for row in rows]
    expected = list(range(len(metadata["ground_truth_item_ids"])))
    if positions != expected:
        raise ValueError(f"invalid GOR candidate paths for seed={seed}, step={step}: positions={positions}")
    paths = [Path(row["image_path"]) for row in rows]
    if not all(path.is_file() for path in paths):
        missing = next(path for path in paths if not path.is_file())
        raise FileNotFoundError(missing)
    return paths


def load_models(args: argparse.Namespace, device: torch.device):
    import open_clip

    if str(args.dualfashion_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.dualfashion_root.resolve()))
    from eval.compatibility_evaluator.compatibility_net import FashionEvaluator

    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79K")
    clip = clip.to(device).eval()
    checkpoint = args.compatibility_ckpt or args.dualfashion_root / "eval" / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing compatibility evaluator: {checkpoint}")
    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    evaluator = evaluator.to(device).eval()
    if args.pcar_checkpoint is None or not args.pcar_checkpoint.is_file():
        raise FileNotFoundError("--pcar-checkpoint must point to the frozen PCAR best.pt")
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from fashion_psp.rewards.train_pcar import PCAR
    payload = torch.load(args.pcar_checkpoint, map_location="cpu", weights_only=False)
    config = payload["model_config"]
    category_ids = [int(value) for value in payload["category_ids"]]
    pcar = PCAR(len(category_ids), config["hidden_dim"], config["embedding_dim"], config["category_dim"])
    pcar.load_state_dict(payload["model"])
    pcar = pcar.to(device).eval()
    return clip, preprocess, evaluator, open_clip, pcar, {category: index for index, category in enumerate(category_ids)}


def score_task(
    metadata: dict,
    cache_steps: list[int],
    history: dict,
    clip,
    preprocess,
    compatibility_model,
    pcar_model,
    pcar_categories: dict[int, int],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[int, dict[int, dict]], dict[int, dict[int, list[str]]]]:
    seeds = [int(seed) for seed in metadata["candidate_seeds"]]
    categories = [int(value) for value in metadata["ground_truth_category_ids"]]
    uid = int(metadata["uid"])
    raw_history_vectors = torch.stack([
        torch.as_tensor(history[uid][category], dtype=torch.float32) for category in categories
    ]).to(device)
    history_vectors = F.normalize(raw_history_vectors, dim=-1)
    unseen_categories = sorted(set(category for category in categories if category not in pcar_categories))
    by_step: dict[int, dict[int, dict]] = {}
    path_lookup: dict[int, dict[int, list[str]]] = {}
    for step in cache_steps:
        flat_paths: list[Path] = []
        index: dict[int, slice] = {}
        for seed in seeds:
            paths = canonical_paths(metadata, step, seed)
            start = len(flat_paths)
            flat_paths.extend(paths)
            index[seed] = slice(start, start + len(paths))
        raw_features = torch.as_tensor(encode_images(clip, preprocess, flat_paths, device, batch_size, normalize=False), device=device)
        raw_candidates = torch.stack([raw_features[index[seed]] for seed in seeds])
        candidates = F.normalize(raw_candidates, dim=-1)
        if candidates.shape[-1] != 1024:
            raise RuntimeError(f"OpenCLIP feature dim {candidates.shape[-1]} does not match FashionEvaluator 1024")
        with torch.inference_mode():
            compatibility = torch.sigmoid(compatibility_model(candidates)).detach().cpu().numpy()
            personal_by_item = 100.0 * (F.normalize(candidates, dim=-1) * history_vectors.unsqueeze(0)).sum(dim=-1)
            personal = personal_by_item.mean(dim=1).detach().cpu().numpy()
            pcar_by_item = []
            for item_index in range(len(categories)):
                context = (raw_candidates.sum(dim=1) - raw_candidates[:, item_index]) / float(len(categories) - 1)
                history_batch = raw_history_vectors[item_index].reshape(1, -1).expand(len(seeds), -1)
                candidate_batch = raw_candidates[:, item_index].unsqueeze(1)
                category = categories[item_index]
                if category in pcar_categories:
                    category_indices = torch.full(
                        (len(seeds),), pcar_categories[category], dtype=torch.long, device=device
                    )
                    item_score = pcar_model(history_batch, context, category_indices, candidate_batch).squeeze(1)
                else:
                    # The test-only unseen category uses a vocabulary-level
                    # fallback; it never receives a known category or GT cue.
                    category_embedding = pcar_model.category_embedding.weight.mean(dim=0, keepdim=True).expand(len(seeds), -1)
                    item_score = pcar_model.forward_with_category_embedding(
                        history_batch, context, category_embedding, candidate_batch
                    ).squeeze(1)
                pcar_by_item.append(item_score)
            pcar_by_item = torch.stack(pcar_by_item, dim=1)
            pcar = pcar_by_item.mean(dim=1).detach().cpu().numpy()
        by_step[step] = {
            seed: {
                "compatibility": float(compatibility[index_seed]),
                "personal": float(personal[index_seed]),
                "personal_by_item": [float(value) for value in personal_by_item[index_seed].detach().cpu().tolist()],
                "pcar": float(pcar[index_seed]),
                "pcar_by_item": [float(value) for value in pcar_by_item[index_seed].detach().cpu().tolist()],
                "pcar_unknown_category_fallback": bool(unseen_categories),
                "pcar_unseen_categories": unseen_categories,
            }
            for index_seed, seed in enumerate(seeds)
        }
        path_lookup[step] = {seed: [str(path.resolve()) for path in flat_paths[index[seed]]] for seed in seeds}
        del raw_features, raw_candidates, candidates
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return by_step, path_lookup


def choose_reward(reward: str, values: dict[int, dict]) -> dict[int, float]:
    compatibility = {seed: row["compatibility"] for seed, row in values.items()}
    personal = {seed: row["personal"] for seed, row in values.items()}
    if reward == "compatibility":
        return compatibility
    if reward == "personal":
        return personal
    if reward == "pcar":
        return {seed: row["pcar"] for seed, row in values.items()}
    c_rank = percentile_scores(compatibility)
    p_rank = percentile_scores(personal)
    return {seed: min(c_rank[seed], p_rank[seed]) for seed in values}


def make_selection(
    reward: str,
    metadata: dict,
    cache_steps: list[int],
    schedule: list[tuple[int, int]],
    scores: dict[int, dict[int, dict]],
    paths: dict[int, dict[int, list[str]]],
) -> dict:
    seeds = [int(seed) for seed in metadata["candidate_seeds"]]
    mode = "bon" if cache_steps == [28] else "psp"
    trace = []
    if mode == "bon":
        final_scores = choose_reward(reward, scores[28])
        final_seed = min(seeds, key=lambda seed: (-final_scores[seed], seed))
        trace.append({"step": 28, "active_before": seeds, "active_after": [final_seed], "reward_scores": final_scores})
    else:
        active = list(seeds)
        for step, keep in schedule:
            subset = {seed: scores[step][seed] for seed in active}
            reward_scores = choose_reward(reward, subset)
            ranked = sorted(active, key=lambda seed: (-reward_scores[seed], seed))
            before = list(active)
            active = ranked[:keep]
            trace.append({"step": step, "active_before": before, "active_after": active, "reward_scores": reward_scores})
        final_seed = active[0]
    standard_seed = seeds[0]
    final_reward = trace[-1]["reward_scores"][final_seed]
    return {
        "task_index": int(metadata["task_index"]),
        "uid": int(metadata["uid"]),
        "oid": int(metadata["oid"]),
        "mode": mode,
        "reward": reward,
        "ground_truth_item_ids": metadata["ground_truth_item_ids"],
        "ground_truth_category_ids": metadata["ground_truth_category_ids"],
        "pcar_unknown_category_fallback": bool(
            scores[28][standard_seed].get("pcar_unknown_category_fallback", False)
        ),
        "standard": {"seed": standard_seed, "image_paths": paths[28][standard_seed], "metrics": scores[28][standard_seed]},
        "selected": {"seed": final_seed, "image_paths": paths[28][final_seed], "metrics": scores[28][final_seed], "selection_reward": final_reward},
        "trace": trace,
    }


def selected_rows(selections: list[dict]) -> list[dict]:
    rows = []
    for selection in selections:
        for method, value in (("standard", selection["standard"]), (selection["mode"], selection["selected"])):
            rows.append({
                "task_index": selection["task_index"], "uid": selection["uid"], "oid": selection["oid"],
                "mode": selection["mode"], "reward": selection["reward"], "method": method,
                "seed": value["seed"], "compatibility": value["metrics"]["compatibility"],
                "personal": value["metrics"]["personal"],
                "pcar": value["metrics"].get("pcar", ""),
                "pcar_unknown_category_fallback": value["metrics"].get("pcar_unknown_category_fallback", False),
                "selection_reward": value.get("selection_reward", ""),
                "image_paths_json": json.dumps(value["image_paths"]),
            })
    return rows


def gt_paths(selection: dict, item_paths: np.ndarray, image_root: Path) -> list[Path]:
    paths = []
    for item_id in selection["ground_truth_item_ids"]:
        path = image_root / str(item_paths[int(item_id)])
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
    return paths


def posthoc_gt_sim(selections: list[dict], item_paths: np.ndarray, image_root: Path, clip, preprocess, device, batch_size: int) -> dict[tuple, float]:
    unique_paths: list[Path] = []
    for selection in selections:
        unique_paths.extend(gt_paths(selection, item_paths, image_root))
        for value in (selection["standard"], selection["selected"]):
            unique_paths.extend(Path(path) for path in value["image_paths"])
    unique_paths = list(dict.fromkeys(unique_paths))
    features = encode_images(clip, preprocess, unique_paths, device, batch_size)
    lookup = {str(path.resolve()): feature for path, feature in zip(unique_paths, features)}
    result = {}
    for selection in selections:
        gt = np.stack([lookup[str(path.resolve())] for path in gt_paths(selection, item_paths, image_root)])
        for method, value in (("standard", selection["standard"]), (selection["mode"], selection["selected"])):
            generated = np.stack([lookup[str(Path(path).resolve())] for path in value["image_paths"]])
            result[(selection["reward"], method, selection["task_index"])] = float(np.mean(np.sum(generated * gt, axis=1)))
    return result


def load_gaa_model(weights: Path, device: torch.device, open_clip):
    if not weights.is_file():
        raise FileNotFoundError(f"missing GAA evaluator weights: {weights}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None, device=device, force_quick_gelu=True)
    try:
        open_clip.load_checkpoint(model, str(weights), strict=False, weights_only=False, device=device)
    except TypeError:
        # Older OpenCLIP versions do not expose the safe-loader arguments.
        # Their legacy loader calls torch.load() with PyTorch's new
        # ``weights_only=True`` default, which cannot read OpenAI's trusted
        # TorchScript archive. Load that local archive explicitly instead.
        try:
            archive = torch.jit.load(str(weights), map_location=device)
        except RuntimeError:
            open_clip.load_checkpoint(model, str(weights), strict=False)
        else:
            model.load_state_dict(archive.state_dict(), strict=False)
            del archive
    except RuntimeError as exc:
        if "TorchScript archive" not in str(exc):
            raise
        archive = torch.jit.load(str(weights), map_location=device)
        model.load_state_dict(archive.state_dict(), strict=False)
        del archive
    if device.type == "cuda":
        model = model.half()
    return model.eval(), preprocess


@torch.inference_mode()
def gaa_metrics(selections: list[dict], item_paths: np.ndarray, image_root: Path, args, device, open_clip) -> dict[tuple, dict]:
    model, preprocess = load_gaa_model(args.gaa_clip_weights, device, open_clip)
    all_paths: list[Path] = []
    for selection in selections:
        all_paths.extend(gt_paths(selection, item_paths, image_root))
        all_paths.extend(Path(path) for path in selection["standard"]["image_paths"])
        all_paths.extend(Path(path) for path in selection["selected"]["image_paths"])
    all_paths = list(dict.fromkeys(all_paths))
    embeddings = encode_images(model, preprocess, all_paths, device, args.batch_size)
    embedding = {str(path.resolve()): value for path, value in zip(all_paths, embeddings)}
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    text_banks = {}
    for group, values in ATTRIBUTE_BANKS.items():
        prompts = [f"a product photo of a fashion item that is {value}" for value in values]
        z = model.encode_text(tokenizer(prompts).to(device))
        text_banks[group] = F.normalize(z.float(), dim=-1).cpu().numpy()
    scale = float(model.logit_scale.exp().float().cpu())
    result: dict[tuple, dict] = {}
    for selection in selections:
        gt = np.stack([embedding[str(path.resolve())] for path in gt_paths(selection, item_paths, image_root)])
        for method, value in (("standard", selection["standard"]), (selection["mode"], selection["selected"])):
            generated = np.stack([embedding[str(Path(path).resolve())] for path in value["image_paths"]])
            groups = {}
            for group, text in text_banks.items():
                generated_prob = torch.softmax(torch.from_numpy(generated @ text.T * scale), dim=1).numpy()
                gt_prob = torch.softmax(torch.from_numpy(gt @ text.T * scale), dim=1).numpy()
                groups[f"gaa_{group}"] = float(np.minimum(generated_prob, gt_prob).sum(axis=1).mean())
            groups["gaa"] = float(np.mean(list(groups.values())))
            result[(selection["reward"], method, selection["task_index"])] = groups
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def summarize(rows: list[dict], gt_sim: dict | None, gaa: dict | None) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["reward"], row["method"])].append(row)
    summary = []
    for (reward, method), values in sorted(grouped.items()):
        record = {"reward": reward, "method": method, "n_tasks": len(values)}
        for name in ("compatibility", "personal"):
            metric = np.asarray([float(row[name]) for row in values])
            record[f"{name}_mean"] = float(metric.mean())
            record[f"{name}_std"] = float(metric.std(ddof=1)) if len(metric) > 1 else 0.0
        if gt_sim is not None:
            metric = np.asarray([gt_sim[(reward, method, int(row["task_index"]))] for row in values])
            record["gt_sim_mean"] = float(metric.mean())
            record["gt_sim_std"] = float(metric.std(ddof=1)) if len(metric) > 1 else 0.0
        if gaa is not None:
            for name in ("gaa", "gaa_color", "gaa_pattern", "gaa_material", "gaa_style"):
                metric = np.asarray([gaa[(reward, method, int(row["task_index"]))][name] for row in values])
                record[f"{name}_mean"] = float(metric.mean())
                record[f"{name}_std"] = float(metric.std(ddof=1)) if len(metric) > 1 else 0.0
        summary.append(record)
    return summary


def main() -> None:
    args = parse_args()
    if args.max_samples < 1 or args.batch_size < 1:
        raise ValueError("--max-samples and --batch-size must be positive")
    resume = parse_resume(args.resume_from)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metadata_paths = sorted(args.cache_root.glob("task*/metadata.json"))[: args.max_samples]
    if len(metadata_paths) != args.max_samples:
        raise RuntimeError(f"expected {args.max_samples} completed task caches under {args.cache_root}, found {len(metadata_paths)}")
    first = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    cache_steps = [int(step) for step in first["offline_candidate_cache"]["steps"]]
    particle_count = int(first["offline_candidate_cache"]["candidate_count"])
    schedule = expected_schedule(cache_steps, particle_count, args.psp_keeps)
    for path in metadata_paths:
        current = json.loads(path.read_text(encoding="utf-8"))
        cache = current["offline_candidate_cache"]
        if [int(step) for step in cache["steps"]] != cache_steps or int(cache["candidate_count"]) != particle_count:
            raise ValueError(f"inconsistent cache protocol: {path}")
    history_path = args.history_path or args.data_root / "processed" / "test_history_clipembs.npy"
    history = np.load(history_path, allow_pickle=True).item()
    clip, preprocess, compatibility_model, open_clip, pcar_model, pcar_categories = load_models(args, device)
    task_root = args.output_root / "tasks"
    task_root.mkdir(parents=True, exist_ok=True)
    selections: list[dict] = []
    progress = tqdm(enumerate(metadata_paths), total=len(metadata_paths), desc=f"GOR {cache_steps} reward selection", unit="task")
    for task_index, path in progress:
        if resume is not None and task_index < resume:
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        output_paths = {reward: task_root / reward / f"task_{task_index:04d}.json" for reward in REWARDS}
        existing = []
        for reward, output_path in output_paths.items():
            if output_path.is_file():
                try:
                    existing.append(json.loads(output_path.read_text(encoding="utf-8")))
                except json.JSONDecodeError:
                    existing = []
                    break
        if len(existing) == len(REWARDS):
            selections.extend(existing)
            progress.set_postfix_str("cached")
            continue
        scores, paths = score_task(metadata, cache_steps, history, clip, preprocess, compatibility_model, pcar_model, pcar_categories, device, args.batch_size)
        for reward, output_path in output_paths.items():
            selection = make_selection(reward, metadata, cache_steps, schedule, scores, paths)
            atomic_json(output_path, selection)
            selections.append(selection)
        progress.set_postfix_str(f"task={task_index}")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Include cached selections skipped by an explicit numeric --resume-from.
    for reward in REWARDS:
        known = {(entry["reward"], entry["task_index"]) for entry in selections}
        for path in sorted((task_root / reward).glob("task_*.json")):
            entry = json.loads(path.read_text(encoding="utf-8"))
            if (reward, entry["task_index"]) not in known:
                selections.append(entry)
    if len(selections) != args.max_samples * len(REWARDS):
        raise RuntimeError(f"incomplete selection set: found {len(selections)}, expected {args.max_samples * len(REWARDS)}")
    rows = selected_rows(selections)
    write_csv(args.output_root / "selected_outfits.csv", rows)
    gt_sim = gaa = None
    if args.with_gt_metrics:
        item_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
        gt_sim = posthoc_gt_sim(selections, item_paths, args.image_root, clip, preprocess, device, args.batch_size)
        gaa = gaa_metrics(selections, item_paths, args.image_root, args, device, open_clip)
        for row in rows:
            key = (row["reward"], row["method"], int(row["task_index"]))
            row["gt_sim"] = gt_sim[key]
            row.update(gaa[key])
        write_csv(args.output_root / "selected_outfits_with_gt_metrics.csv", rows)
    summary = summarize(rows, gt_sim, gaa)
    write_csv(args.output_root / "summary.csv", summary)
    atomic_json(args.output_root / "protocol.json", {
        "task_type": "GOR", "cache_root": str(args.cache_root.resolve()), "cache_steps": cache_steps,
        "candidate_count": particle_count, "schedule": schedule,
        "selection_unit": "complete sequential outfit trajectory",
        "compatibility": "sigmoid(FashionEvaluator([OpenCLIP ViT-H/14 generated item embeddings]))",
        "personal": "mean_i 100*cosine(OpenCLIP ViT-H/14(generated_i), user_history(uid, category_i))",
        "pcar": "mean_i PCAR(history(uid, category_i), other generated outfit items, category_i, generated_i)",
        "bottleneck": "min(within-active-set percentile(compatibility), percentile(personal))",
        "gt_usage": "post-selection diagnostics only" if args.with_gt_metrics else "not loaded",
        "gt_sim": "mean matched-position OpenCLIP ViT-H/14 cosine" if args.with_gt_metrics else None,
        "gaa": "OpenAI CLIP ViT-L/14 attribute-distribution overlap" if args.with_gt_metrics else None,
    })
    print(json.dumps({"output": str(args.output_root), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
