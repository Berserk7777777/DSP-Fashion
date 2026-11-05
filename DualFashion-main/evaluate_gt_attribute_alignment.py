"""GT Attribute Alignment (GAA) for frozen BoN/PSP selections.

GT is used only after selection as an evaluation reference.  This script must
never be imported by a reward scorer or a PSP pruning routine.

The selector's Personal Sim encoder is OpenCLIP ViT-H/14 (LAION).  GAA uses a
separate, frozen OpenAI CLIP ViT-L/14 checkpoint and compares zero-shot
attribute distributions, rather than whole-image embeddings.  It is an
automatic proxy for visual attributes, not human annotations.
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

# These banks are fixed before looking at the FITB test results.  They cover
# attributes visible across bags, shoes, and apparel without using GT labels.
ATTRIBUTE_BANKS = {
    "color": ["black", "white", "gray", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "beige"],
    "pattern": ["solid", "striped", "plaid", "floral", "polka dot", "geometric", "animal print", "quilted", "textured"],
    "material": ["leather", "faux leather", "suede", "canvas", "denim", "knit", "cotton", "wool", "synthetic material"],
    "style": ["casual", "formal", "sporty", "elegant", "vintage", "minimalist", "streetwear"],
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_combined_summary(selection_roots: dict[str, Path], gaa_summary: list[dict], output: Path) -> None:
    """Join post-selection GAA with the frozen selection evaluator summaries."""
    if not selection_roots:
        return
    gaa_by_key = {(row["reward"], row["method"]): row for row in gaa_summary}
    metric_names = ("compatibility_sigmoid", "personal_sim", "gt_sim", "IS", "is_acc_top3", "gt_lpips")
    combined = []
    for reward, root in selection_roots.items():
        summary_path = root / "official_eval" / "group_summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing frozen evaluation summary: {summary_path}")
        by_method: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in read_csv(summary_path):
            if row.get("metric") in metric_names:
                by_method[row["method"]][row["metric"]] = row
        for method in ("standard", "bon", "psp"):
            gaa = gaa_by_key[(reward, method)]
            record = {"reward": reward, "method": method, "n_tasks": gaa["n_tasks"]}
            for metric in metric_names:
                metric_row = by_method.get(method, {}).get(metric)
                if metric_row is None:
                    raise RuntimeError(f"Missing {metric} in {summary_path} for {method}")
                record[f"{metric}_mean"] = metric_row["mean"]
                record[f"{metric}_std"] = metric_row["std_sample"]
            for column in ("gaa", "gaa_color", "gaa_pattern", "gaa_material", "gaa_style"):
                record[f"{column}_mean"] = gaa[f"{column}_mean"]
                record[f"{column}_std"] = gaa[f"{column}_std"]
            combined.append(record)
    write_csv(output / "combined_evaluation_summary.csv", combined)


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.inference_mode()
def encode_images(model, preprocess, paths: list[Path], device, batch_size: int) -> np.ndarray:
    def load_one(path: Path):
        try:
            with Image.open(path) as image:
                return preprocess(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"cannot read image: {path}") from exc

    embeddings = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for batch in batched(paths, batch_size):
            pixels = torch.stack(list(pool.map(load_one, batch))).to(
                device, dtype=next(model.parameters()).dtype
            )
            z = model.encode_image(pixels)
            z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            embeddings.append(z.float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


@torch.inference_mode()
def encode_text(model, tokenizer, values: list[str], device) -> np.ndarray:
    prompts = [f"a product photo of a fashion item that is {value}" for value in values]
    z = model.encode_text(tokenizer(prompts).to(device))
    z = z / z.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return z.float().cpu().numpy()


def build_selected_rows(repo: Path, data_root: Path, task_limit: int | None) -> list[dict]:
    fitb = np.load(data_root / "processed/fitb_test.npy", allow_pickle=True).item()
    ground_truth = np.load(data_root / "test_grd.npy", allow_pickle=True).item()
    item_paths = np.load(data_root / "all_item_image_paths.npy")
    tasks = []
    for task, (uid, oid, outfit) in enumerate(zip(fitb["uids"], fitb["oids"], fitb["outfits"])):
        if task_limit is not None and task >= task_limit:
            break
        outfit = outfit.tolist() if hasattr(outfit, "tolist") else list(outfit)
        missing_position = outfit.index(0)
        gt_item_id = ground_truth[oid]["outfits"][missing_position]
        tasks.append((task, int(uid), int(oid), data_root / "semantic_category" / str(item_paths[gt_item_id])))

    selected = []
    for reward, files in REWARD_FILES.items():
        bon_rows = read_csv(repo / files["bon"])
        bon_by_task = defaultdict(list)
        for row in bon_rows:
            bon_by_task[(int(row["uid"]), int(row["oid"]))].append(row)
        psp_rows = read_csv(repo / files["psp_scores"])
        psp_by_key = {(int(row["task_index"]), int(row["seed"]), int(row["step"])): row for row in psp_rows}
        psp_selection = {int(row["task_index"]): row for row in read_csv(repo / files["psp_selection"])}

        for task, uid, oid, gt_path in tasks:
            candidates = bon_by_task[(uid, oid)]
            if not candidates:
                raise RuntimeError(f"missing BoN candidates for {reward}, task {task}: {uid}/{oid}")
            standard = next((row for row in candidates if row["is_standard"].lower() == "true"), None)
            winner = next((row for row in candidates if row["is_bon_winner"].lower() == "true"), None)
            if standard is None or winner is None:
                raise RuntimeError(f"missing standard or BoN winner for {reward}, task {task}")
            for method, row in (("standard", standard), ("bon", winner)):
                selected.append({"task_index": task, "uid": uid, "oid": oid, "reward": reward, "method": method,
                                 "seed": int(row["seed"]), "image_path": str(row["image_path"]), "gt_path": str(gt_path)})
            choice = psp_selection.get(task)
            if choice is None:
                raise RuntimeError(f"missing PSP selection for {reward}, task {task}")
            psp_row = psp_by_key.get((task, int(choice["psp_seed"]), 28))
            if psp_row is None:
                raise RuntimeError(f"missing final PSP image for {reward}, task {task}")
            selected.append({"task_index": task, "uid": uid, "oid": oid, "reward": reward, "method": "psp",
                             "seed": int(choice["psp_seed"]), "image_path": str(psp_row["image_path"]), "gt_path": str(gt_path)})
    return selected


def parse_selection_roots(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return {}
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--selection-root must be REWARD=PATH")
        reward, raw_path = value.split("=", 1)
        reward = reward.strip().lower()
        if not reward:
            raise ValueError("Reward name in --selection-root cannot be empty")
        if reward in roots:
            raise ValueError(f"Duplicate --selection-root for reward: {reward}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Selection root is unavailable: {path}")
        roots[reward] = path
    return roots


def build_selected_rows_from_indices(selection_roots: dict[str, Path], data_root: Path, image_root: Path, task_limit: int | None) -> list[dict]:
    """Read frozen DiFashion selected-image indices; GT is joined only here."""
    fitb_path = data_root / "processed" / "new_fitb_test.npy"
    ground_path = data_root / "test_grd.npy"
    item_paths_path = data_root / "new_all_item_image_paths.npy"
    for path in (fitb_path, ground_path, item_paths_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing DiFashion post-selection asset: {path}")
    fitb = np.load(fitb_path, allow_pickle=True).item()
    ground_truth = np.load(ground_path, allow_pickle=True).item()
    item_paths = np.load(item_paths_path, allow_pickle=True)
    tasks: dict[str, dict] = {}
    for task_index, (uid, oid, outfit, category) in enumerate(zip(
        fitb["uids"], fitb["oids"], fitb["outfits"], fitb["category"]
    )):
        if task_limit is not None and task_index >= task_limit:
            break
        uid, oid = int(uid), int(oid)
        outfit = np.asarray(outfit, dtype=np.int64).reshape(-1)
        category = np.asarray(category, dtype=np.int64).reshape(-1)
        blanks = np.flatnonzero(outfit == 0)
        if len(outfit) != 4 or len(category) != 4 or len(blanks) != 1:
            raise ValueError(f"Invalid FITB condition row {task_index}")
        blank = int(blanks[0])
        task_id = f"{task_index:05d}_{uid}_{oid}"
        gt = ground_truth.get(oid, ground_truth.get(str(oid)))
        if gt is None:
            raise KeyError(f"No GT record for DiFashion task {task_id}")
        gt_outfit = np.asarray(gt["outfits"], dtype=np.int64).reshape(-1)
        gt_category = np.asarray(gt["category"], dtype=np.int64).reshape(-1)
        if len(gt_outfit) != 4 or len(gt_category) != 4:
            raise ValueError(f"Malformed GT outfit for {task_id}")
        for index, item_id in enumerate(outfit):
            if index != blank and int(gt_outfit[index]) != int(item_id):
                raise ValueError(f"GT context mismatch for {task_id}, position {index}")
        if int(gt_category[blank]) != int(category[blank]):
            raise ValueError(f"GT category mismatch for {task_id}")
        gt_item_id = int(gt_outfit[blank])
        if not 0 <= gt_item_id < len(item_paths):
            raise IndexError(f"GT item id out of range for {task_id}: {gt_item_id}")
        tasks[task_id] = {"task_index": task_index, "uid": uid, "oid": oid,
                          "gt_path": str((image_root / str(item_paths[gt_item_id])).resolve())}

    selected: list[dict] = []
    for reward, root in selection_roots.items():
        index_path = root / "selected_images_index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing selected image index for {reward}: {index_path}")
        by_task: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in read_csv(index_path):
            task_id, method = row.get("task_id"), row.get("method")
            if task_id not in tasks:
                if task_limit is not None:
                    continue
                raise ValueError(f"Unexpected task in {index_path}: {task_id}")
            if row.get("reward") != reward:
                raise ValueError(f"Reward label mismatch in {index_path}: {row.get('reward')} != {reward}")
            if method not in ("standard", "bon", "psp"):
                raise ValueError(f"Unknown selection method in {index_path}: {method}")
            if method in by_task[task_id]:
                raise ValueError(f"Duplicate {method} selection for {task_id} in {index_path}")
            by_task[task_id][method] = row
        for task_id, task in tasks.items():
            methods = by_task.get(task_id, {})
            if set(methods) != {"standard", "bon", "psp"}:
                raise RuntimeError(f"Incomplete selections for {reward}, {task_id}: {sorted(methods)}")
            for method, row in methods.items():
                image_path = Path(row["final_image_path"]).resolve()
                selected.append({**task, "task_id": task_id, "reward": reward, "method": method,
                                 "seed": int(row["seed"]), "image_path": str(image_path)})
    return selected


def load_evaluator(weights: Path, device):
    # OpenAI's released CLIP checkpoints use QuickGELU.  The explicit flag is
    # required when loading the original .pt through OpenCLIP.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=None, device=device, force_quick_gelu=True
    )
    # The local .pt is the official OpenAI checkpoint. load_checkpoint handles
    # its TorchScript archive without a Hugging Face network request.
    try:
        open_clip.load_checkpoint(model, str(weights), strict=False, weights_only=False, device=device)
    except TypeError:
        # OpenCLIP releases before the safe-loader API do not expose the
        # weights_only/device keywords. The model was already created on the
        # requested device, so the legacy call is equivalent here.
        try:
            open_clip.load_checkpoint(model, str(weights), strict=False)
        except RuntimeError as exc:
            if "TorchScript archive" not in str(exc):
                raise
            archive = torch.jit.load(str(weights), map_location=device)
            model.load_state_dict(archive.state_dict(), strict=False)
            del archive
    if device.type == "cuda":
        model = model.half()
    model.eval()
    return model, preprocess


def main():
    parser = argparse.ArgumentParser(description="Evaluate GT Attribute Alignment for frozen FITB selections.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).parent)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\tmp\ifashion"))
    parser.add_argument("--image-root", type=Path, default=None,
                        help="Root for paths in new_all_item_image_paths.npy; defaults to --data-root.")
    parser.add_argument(
        "--selection-root",
        action="append",
        default=None,
        metavar="REWARD=PATH",
        help=("Read DiFashion selected_images_index.csv from frozen reward runs. "
              "Accepts one or more unique REWARD=PATH values."),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--clip-weights", type=Path, default=Path(r"C:\Users\Administrator\.cache\clip\ViT-L-14.pt"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test first N tasks only; never use for final results.")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not args.clip_weights.is_file():
        raise FileNotFoundError(f"missing independent GAA evaluator weights: {args.clip_weights}")
    output = args.output or args.repo / "output/ifashion_fitb_gt_attribute_alignment"
    output.mkdir(parents=True, exist_ok=True)
    selection_roots = parse_selection_roots(args.selection_root)
    image_root = (args.image_root or args.data_root).resolve()
    if selection_roots and not image_root.is_dir():
        raise FileNotFoundError(f"DiFashion GT image root is unavailable: {image_root}")
    rows = (
        build_selected_rows_from_indices(selection_roots, args.data_root, image_root, args.limit)
        if selection_roots else build_selected_rows(args.repo, args.data_root, args.limit)
    )
    paths = list(dict.fromkeys([row["image_path"] for row in rows] + [row["gt_path"] for row in rows]))
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image files missing; first: {missing[0]}")
    path_to_index = {path: i for i, path in enumerate(paths)}
    cache_path = output / "vitl14_embeddings.npz"
    cache_index_path = output / "vitl14_embedding_paths.json"
    if cache_path.exists() and cache_index_path.exists():
        cached_paths = json.loads(cache_index_path.read_text(encoding="utf-8"))
        if cached_paths != paths:
            raise RuntimeError("embedding cache does not match selected images; choose a new --output directory")
        embeddings = np.load(cache_path)["embeddings"]
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model, preprocess = load_evaluator(args.clip_weights, device)
        embeddings = encode_images(model, preprocess, [Path(path) for path in paths], device, args.batch_size)
        np.savez_compressed(cache_path, embeddings=embeddings)
        cache_index_path.write_text(json.dumps(paths), encoding="utf-8")
        text_embeddings = {
            group: encode_text(model, open_clip.get_tokenizer("ViT-L-14"), values, device)
            for group, values in ATTRIBUTE_BANKS.items()
        }
        np.savez_compressed(output / "attribute_text_embeddings.npz", **text_embeddings)
    if not (output / "attribute_text_embeddings.npz").is_file():
        raise RuntimeError("missing text embedding cache; use a fresh --output directory")
    # Keep provenance current even when image embeddings were reused from cache.
    (output / "evaluator_config.json").write_text(json.dumps({
        "model": "OpenAI CLIP ViT-L/14", "weights": str(args.clip_weights),
        "selection_source": "frozen DiFashion reward-specific selection runs" if selection_roots else "legacy DualFashion paths",
        "attribute_banks": ATTRIBUTE_BANKS,
        "agreement": "mean over groups of sum_k min(p_gen(k), p_gt(k)); p is native CLIP-temperature softmax over a group",
        "gt_usage": "post-hoc evaluation only; never selection or reward",
        "selection_sources": {reward: str(path) for reward, path in selection_roots.items()} or "legacy DualFashion paths",
        "image_root": str(image_root),
    }, indent=2), encoding="utf-8")
    text_embeddings = np.load(output / "attribute_text_embeddings.npz")
    # The native learned CLIP temperature is part of the frozen evaluator.
    # It avoids introducing a test-tuned softmax temperature.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = load_evaluator(args.clip_weights, device)
    logit_scale = float(model.logit_scale.exp().detach().float().cpu())
    del model
    for group in ATTRIBUTE_BANKS:
        text_z = text_embeddings[group]
        gen_z = embeddings[[path_to_index[row["image_path"]] for row in rows]]
        gt_z = embeddings[[path_to_index[row["gt_path"]] for row in rows]]
        gen_logits = torch.tensor(gen_z @ text_z.T * logit_scale)
        gt_logits = torch.tensor(gt_z @ text_z.T * logit_scale)
        gen_prob = torch.softmax(gen_logits, dim=1).numpy()
        gt_prob = torch.softmax(gt_logits, dim=1).numpy()
        agreement = np.minimum(gen_prob, gt_prob).sum(axis=1)
        for index, row in enumerate(rows):
            row[f"gaa_{group}"] = float(agreement[index])
    group_columns = [f"gaa_{group}" for group in ATTRIBUTE_BANKS]
    for row in rows:
        row["gaa"] = float(np.mean([row[column] for column in group_columns]))
    per_task_path = output / "gaa_per_task.csv"
    write_csv(per_task_path, rows)
    summary = []
    for reward in (selection_roots or REWARD_FILES):
        for method in ("standard", "bon", "psp"):
            subset = [row for row in rows if row["reward"] == reward and row["method"] == method]
            record = {"reward": reward, "method": method, "n_tasks": len(subset)}
            for column in ["gaa", *group_columns]:
                values = np.asarray([row[column] for row in subset], dtype=float)
                record[f"{column}_mean"] = float(values.mean())
                record[f"{column}_std"] = float(values.std(ddof=1))
            summary.append(record)
    write_csv(output / "gaa_summary.csv", summary)
    write_combined_summary(selection_roots, summary, output)
    with (output / "gaa_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# GT Attribute Alignment\n\n")
        handle.write("All values are post-hoc test-set diagnostics in [0, 1]. Higher is better. "
                     "GT was not used by BoN/PSP selection.\n\n")
        handle.write("| reward | method | n | GAA | color | pattern | material | style |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary:
            handle.write(f"| {row['reward']} | {row['method']} | {row['n_tasks']} | {row['gaa_mean']:.4f} | "
                         f"{row['gaa_color_mean']:.4f} | {row['gaa_pattern_mean']:.4f} | "
                         f"{row['gaa_material_mean']:.4f} | {row['gaa_style_mean']:.4f} |\n")
    print(json.dumps({"output": str(output), "n_rows": len(rows), "n_unique_images": len(paths), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
