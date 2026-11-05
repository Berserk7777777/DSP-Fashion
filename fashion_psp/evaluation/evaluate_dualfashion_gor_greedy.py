"""Evaluate final selections from the online item-greedy DualFashion GOR runs.

The generator never reads ground truth.  GT-Sim and GAA in this script are
strictly post-hoc diagnostics over the saved final selected images.
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


ATTRIBUTE_BANKS = {
    "color": ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"],
    "pattern": ["solid", "striped", "plaid", "floral", "polka dot", "geometric", "animal print", "quilted", "textured"],
    "material": ["leather", "faux leather", "suede", "canvas", "denim", "knit", "cotton", "wool", "synthetic material"],
    "style": ["casual", "formal", "sporty", "elegant", "vintage", "minimalist", "streetwear"],
}


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dualfashion-root", type=Path, default=repo / "DualFashion-main")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--processed-info-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--personal-clip-weights", type=Path,
        default=Path(r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"),
    )
    parser.add_argument("--gaa-clip-weights", type=Path, default=Path(r"C:\Users\Administrator\.cache\clip\ViT-L-14.pt"))
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def batched(items: list, size: int):
    for offset in range(0, len(items), size):
        yield items[offset : offset + size]


@torch.inference_mode()
def encode_images(
    model, preprocess, paths: list[Path], device: torch.device, batch_size: int, normalize: bool = True
) -> np.ndarray:
    def load_one(path: Path):
        with Image.open(path) as image:
            return preprocess(image.convert("RGB"))

    vectors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for chunk in tqdm(list(batched(paths, batch_size)), desc="Encoding images", unit="batch"):
            pixels = torch.stack(list(pool.map(load_one, chunk))).to(device, dtype=next(model.parameters()).dtype)
            vectors.append(model.encode_image(pixels).float().cpu().numpy())
    result = np.concatenate(vectors, axis=0)
    if normalize:
        result /= np.linalg.norm(result, axis=-1, keepdims=True).clip(min=1e-12)
    return result


def load_personal_and_compatibility(args: argparse.Namespace, device: torch.device):
    import open_clip

    dualfashion = str(args.dualfashion_root.resolve())
    if dualfashion not in sys.path:
        sys.path.insert(0, dualfashion)
    from eval.compatibility_evaluator.compatibility_net import FashionEvaluator

    if not args.personal_clip_weights.is_file():
        raise FileNotFoundError(f"missing frozen Personal OpenCLIP weights: {args.personal_clip_weights}")
    # Do not use the pretrained registry here: it performs an unnecessary
    # HuggingFace metadata request before it consults the local cache.
    clip, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    from safetensors.torch import load_file
    state = load_file(str(args.personal_clip_weights), device="cpu")
    missing, unexpected = clip.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Personal CLIP checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    clip = clip.to(device).eval()
    checkpoint = args.dualfashion_root / "eval" / "compatibility_evaluator" / "ifashion-ckpt" / "ifashion_evaluator.pth"
    evaluator = FashionEvaluator(cnn_feat_dim=1024)
    evaluator.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return clip, preprocess, evaluator.to(device).eval(), open_clip


def gt_path(item_id: int, image_paths: np.ndarray, image_root: Path) -> Path:
    path = image_root / str(image_paths[item_id])
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_tasks(root: Path, maximum: int) -> list[dict]:
    paths = sorted(root.glob("task*/task_summary.json"))[:maximum]
    if len(paths) != maximum:
        raise RuntimeError(f"expected {maximum} complete tasks in {root}, found {len(paths)}")
    tasks = []
    for path in paths:
        task = json.loads(path.read_text(encoding="utf-8"))
        items = sorted(task["items"], key=lambda item: int(item["original_outfit_position"]))
        if [int(item["original_outfit_position"]) for item in items] != list(range(4)):
            raise ValueError(f"invalid original outfit order: {path}")
        if not all(Path(item["selected_image_path"]).is_file() for item in items):
            raise FileNotFoundError(f"missing selected image under {path.parent}")
        task["items"] = items
        tasks.append(task)
    return tasks


def score_personal_compatibility(tasks, history, clip, preprocess, compatibility, device, batch_size):
    generated_paths = [Path(item["selected_image_path"]) for task in tasks for item in task["items"]]
    raw_features = encode_images(clip, preprocess, generated_paths, device, batch_size, normalize=False)
    features = raw_features / np.linalg.norm(raw_features, axis=-1, keepdims=True).clip(min=1e-12)
    rows = []
    for index, task in enumerate(tasks):
        start = index * 4
        vectors = torch.as_tensor(features[start : start + 4], device=device)
        raw_vectors = torch.as_tensor(raw_features[start : start + 4], device=device)
        uid = int(task["uid"])
        histories = []
        for item in task["items"]:
            category = int(item["category_id"])
            vector = history[uid].get(category, history[uid].get(str(category)))
            if vector is None:
                raise KeyError(f"missing history uid={uid} category={category}")
            histories.append(torch.as_tensor(vector, dtype=torch.float32, device=device))
        history_vectors = F.normalize(torch.stack(histories), dim=-1)
        personal_by_item = 100.0 * (vectors * history_vectors).sum(dim=-1)
        with torch.inference_mode():
            # FashionEvaluator was trained on raw OpenCLIP features.  L2
            # normalization is valid for cosine metrics only, not its input.
            compatibility_score = torch.sigmoid(compatibility(raw_vectors.unsqueeze(0))).item()
        logged = [float(item["selected_personal_reward"]) for item in task["items"]]
        rows.append({
            "task_index": int(task["task_index"]), "uid": uid, "oid": int(task["oid"]),
            "method": task["method"], "online_personal_mean": float(np.mean(logged)),
            "personal": float(personal_by_item.mean().cpu()), "compatibility": float(compatibility_score),
            "item_personal_json": json.dumps([float(value) for value in personal_by_item.cpu().tolist()]),
            "item_paths_json": json.dumps([item["selected_image_path"] for item in task["items"]]),
            "transformer_forward_calls": int(sum(int(item["transformer_forward_calls"]) for item in task["items"])),
        })
    return rows, generated_paths, features


def gt_sim(tasks, generated_paths, generated_features, item_paths, image_root, clip, preprocess, device, batch_size):
    gt_paths = [gt_path(int(item["item_id"]), item_paths, image_root) for task in tasks for item in task["items"]]
    gt_features = encode_images(clip, preprocess, gt_paths, device, batch_size, normalize=True)
    scores = {}
    for index, task in enumerate(tasks):
        start = index * 4
        scores[int(task["task_index"])] = float((generated_features[start : start + 4] * gt_features[start : start + 4]).sum(axis=1).mean())
    return scores, gt_paths


def load_gaa_model(weights: Path, device: torch.device, open_clip):
    if not weights.is_file():
        raise FileNotFoundError(f"missing GAA OpenAI CLIP weights: {weights}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None, device=device, force_quick_gelu=True)
    try:
        open_clip.load_checkpoint(model, str(weights), strict=False, weights_only=False, device=device)
    except (TypeError, RuntimeError):
        archive = torch.jit.load(str(weights), map_location=device)
        model.load_state_dict(archive.state_dict(), strict=False)
        del archive
    if device.type == "cuda":
        model = model.half()
    return model.eval(), preprocess


@torch.inference_mode()
def gaa_scores(tasks, generated_paths, gt_paths, args, device, open_clip):
    model, preprocess = load_gaa_model(args.gaa_clip_weights, device, open_clip)
    all_paths = list(dict.fromkeys(generated_paths + gt_paths))
    vectors = encode_images(model, preprocess, all_paths, device, args.batch_size)
    lookup = {str(path.resolve()): vector for path, vector in zip(all_paths, vectors)}
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    text_banks = {}
    for group, labels in ATTRIBUTE_BANKS.items():
        prompts = [f"a product photo of a fashion item that is {label}" for label in labels]
        text_banks[group] = F.normalize(model.encode_text(tokenizer(prompts).to(device)).float(), dim=-1).cpu().numpy()
    scale = float(model.logit_scale.exp().float().cpu())
    result = {}
    for index, task in enumerate(tasks):
        selected = np.stack([lookup[str(path.resolve())] for path in generated_paths[index * 4 : index * 4 + 4]])
        gt = np.stack([lookup[str(path.resolve())] for path in gt_paths[index * 4 : index * 4 + 4]])
        record = {}
        for group, bank in text_banks.items():
            selected_prob = torch.softmax(torch.from_numpy(selected @ bank.T * scale), dim=1).numpy()
            gt_prob = torch.softmax(torch.from_numpy(gt @ bank.T * scale), dim=1).numpy()
            record[f"gaa_{group}"] = float(np.minimum(selected_prob, gt_prob).sum(axis=1).mean())
        record["gaa"] = float(np.mean(list(record.values())))
        result[int(task["task_index"])] = record
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = load_tasks(args.input_root, args.max_samples)
    history = np.load(args.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    item_paths = np.load(args.processed_info_root / "all_item_image_paths.npy", allow_pickle=True)
    clip, preprocess, compatibility, open_clip = load_personal_and_compatibility(args, device)
    rows, generated_paths, generated_features = score_personal_compatibility(
        tasks, history, clip, preprocess, compatibility, device, args.batch_size
    )
    gt_sim_values, gt_paths = gt_sim(tasks, generated_paths, generated_features, item_paths, args.image_root, clip, preprocess, device, args.batch_size)
    del clip, compatibility
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gaa = gaa_scores(tasks, generated_paths, gt_paths, args, device, open_clip)
    for row in rows:
        row["gt_sim"] = gt_sim_values[int(row["task_index"])]
        row.update(gaa[int(row["task_index"])])
    write_csv(args.output_root / "per_task_metrics.csv", rows)
    metric_names = ("online_personal_mean", "personal", "compatibility", "gt_sim", "gaa", "gaa_color", "gaa_pattern", "gaa_material", "gaa_style")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary = []
    for method, values in sorted(grouped.items()):
        record = {"method": method, "n_tasks": len(values), "transformer_forward_calls_per_outfit": int(np.mean([row["transformer_forward_calls"] for row in values]))}
        for metric in metric_names:
            data = np.asarray([float(row[metric]) for row in values])
            record[f"{metric}_mean"] = float(data.mean())
            record[f"{metric}_std"] = float(data.std(ddof=1))
        summary.append(record)
    write_csv(args.output_root / "summary.csv", summary)
    (args.output_root / "protocol.json").write_text(json.dumps({
        "task_type": "GOR", "selection": "online item-level greedy", "reward": "frozen Personal only",
        "compatibility": "sigmoid(FashionEvaluator([OpenCLIP ViT-H/14 generated item embeddings]))",
        "personal": "mean_i 100*cosine(OpenCLIP ViT-H/14(generated_i), user_history(uid, category_i))",
        "gt_sim": "post-hoc mean matched-position OpenCLIP ViT-H/14 cosine",
        "gaa": "post-hoc OpenAI CLIP ViT-L/14 attribute-distribution overlap",
        "gt_used_for_selection": False,
    }, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
