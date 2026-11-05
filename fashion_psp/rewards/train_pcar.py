"""Train the frozen Preference-Context Alignment Reward (PCAR) on iFashion.

PCAR learns a query for ``(user history, observed outfit context, target
category)`` and ranks the observed training-set target above category-matched
catalog negatives.  It deliberately never reads FITB/GOR test ground truth.
The OpenCLIP item features distributed with iFashion stay frozen.

The feature cache contains only mean history/context vectors.  It makes later
training runs inexpensive and can be deleted after the final checkpoint has
been saved.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


FEATURE_DIM = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-file", default="processed/train.npy")
    parser.add_argument("--history-file", default="train_history.npy")
    parser.add_argument("--feature-file", default="cnn_features_clip.npy")
    parser.add_argument("--max-outfits", type=int, default=None)
    parser.add_argument("--examples-per-outfit", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--num-negatives", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--category-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=4, help="Stop after this many non-improving validation epochs; 0 disables.")
    parser.add_argument("--save-epoch-checkpoints", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 0.5:
        parser.error("--val-fraction must be in (0, 0.5)")
    return args


def load_dict(path: Path) -> dict:
    value = np.load(path, allow_pickle=True)
    if getattr(value, "shape", None) != ():
        raise ValueError(f"expected a serialized dict: {path}")
    return value.item()


def as_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return [int(item) for item in np.asarray(value).reshape(-1).tolist()]


def history_items(history: dict, uid: int, category: int) -> list[int]:
    user_history = history.get(uid, history.get(str(uid), {}))
    if not isinstance(user_history, dict):
        return []
    values = user_history.get(category, user_history.get(str(category), []))
    return [int(item) for item in values]


def cache_paths(cache_root: Path) -> dict[str, Path]:
    return {
        "manifest": cache_root / "manifest.json",
        "history": cache_root / "history.f16.mmap",
        "context": cache_root / "context.f16.mmap",
        "target": cache_root / "target_ids.i32.mmap",
        "category": cache_root / "category_ids.i32.mmap",
        "split": cache_root / "split.u8.mmap",
    }


def cache_is_valid(paths: dict[str, Path], expected: dict) -> bool:
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest == expected


def is_validation_outfit(index: int, seed: int, val_fraction: float) -> bool:
    """A stable split independent of Python hash randomization."""
    value = (index * 1103515245 + seed * 12345 + 1013904223) & 0xFFFFFFFF
    return value < int(val_fraction * (1 << 32))


def make_examples(
    train: dict, max_outfits: int | None, examples_per_outfit: int, seed: int, val_fraction: float
) -> list[tuple[int, int, int, list[int], int]]:
    total = len(train["uids"])
    if max_outfits is not None:
        total = min(total, max_outfits)
    examples = []
    for index in tqdm(range(total), desc="Preparing PCAR examples", unit="outfit"):
        uid = int(train["uids"][index])
        outfit = as_ids(train["outfits"][index])
        categories = as_ids(train["category"][index])
        if len(outfit) != 4 or len(categories) != 4 or any(item <= 0 for item in outfit):
            continue
        positions = range(4) if examples_per_outfit == 4 else [(index + offset) % 4 for offset in range(examples_per_outfit)]
        for position in positions:
            context = [item for item_index, item in enumerate(outfit) if item_index != position]
            split = 1 if is_validation_outfit(index, seed, val_fraction) else 0
            examples.append((uid, int(categories[position]), int(outfit[position]), context, split))
    if not examples:
        raise RuntimeError("no usable four-item training outfits were found")
    return examples


def make_category_pools(train: dict, active_categories: set[int], feature_count: int) -> dict[int, np.ndarray]:
    """Build catalog negatives from training outfits only.

    iFashion's ``id_cate_dict.npy`` maps category IDs to names, not item IDs
    to categories.  The paired ``outfits``/``category`` fields are therefore
    the authoritative item-category relation available in this release.
    """
    pools: dict[int, set[int]] = defaultdict(set)
    for outfit_value, category_value in tqdm(zip(train["outfits"], train["category"]), total=len(train["uids"]), desc="Indexing category negatives", unit="outfit"):
        for item_id, category in zip(as_ids(outfit_value), as_ids(category_value)):
            if category in active_categories and 0 < item_id < feature_count:
                pools[category].add(item_id)
    result = {category: np.asarray(sorted(items), dtype=np.int64) for category, items in pools.items() if len(items) >= 2}
    missing = sorted(active_categories - set(result))
    if missing:
        raise RuntimeError(f"cannot sample same-category negatives for categories: {missing}")
    return result


def build_cache(
    paths: dict[str, Path],
    expected: dict,
    examples: list[tuple[int, int, int, list[int], int]],
    history: dict,
    item_features: np.ndarray,
) -> None:
    cache_root = paths["manifest"].parent
    cache_root.mkdir(parents=True, exist_ok=True)
    count = len(examples)
    history_cache = np.memmap(paths["history"], dtype=np.float16, mode="w+", shape=(count, FEATURE_DIM))
    context_cache = np.memmap(paths["context"], dtype=np.float16, mode="w+", shape=(count, FEATURE_DIM))
    target_cache = np.memmap(paths["target"], dtype=np.int32, mode="w+", shape=(count,))
    category_cache = np.memmap(paths["category"], dtype=np.int32, mode="w+", shape=(count,))
    split_cache = np.memmap(paths["split"], dtype=np.uint8, mode="w+", shape=(count,))

    for index, (uid, category, target, context, split) in enumerate(tqdm(examples, desc="Caching PCAR features", unit="example")):
        context_cache[index] = np.asarray(item_features[np.asarray(context, dtype=np.int64)]).mean(axis=0, dtype=np.float32)
        user_items = [item for item in history_items(history, uid, category) if item != target]
        if user_items:
            history_cache[index] = np.asarray(item_features[np.asarray(user_items, dtype=np.int64)]).mean(axis=0, dtype=np.float32)
        else:
            history_cache[index] = 0.0
        target_cache[index] = target
        category_cache[index] = category
        split_cache[index] = split
    history_cache.flush()
    context_cache.flush()
    target_cache.flush()
    category_cache.flush()
    split_cache.flush()
    paths["manifest"].write_text(json.dumps(expected, indent=2), encoding="utf-8")


class CachedPCARDataset(Dataset):
    def __init__(self, paths: dict[str, Path], count: int):
        self.history = np.memmap(paths["history"], dtype=np.float16, mode="r", shape=(count, FEATURE_DIM))
        self.context = np.memmap(paths["context"], dtype=np.float16, mode="r", shape=(count, FEATURE_DIM))
        self.target = np.memmap(paths["target"], dtype=np.int32, mode="r", shape=(count,))
        self.category = np.memmap(paths["category"], dtype=np.int32, mode="r", shape=(count,))
        self.split = np.memmap(paths["split"], dtype=np.uint8, mode="r", shape=(count,))

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(np.array(self.history[index], dtype=np.float32, copy=True)),
            torch.from_numpy(np.array(self.context[index], dtype=np.float32, copy=True)),
            int(self.category[index]),
            int(self.target[index]),
        )


class PCAR(nn.Module):
    def __init__(self, category_count: int, hidden_dim: int, embedding_dim: int, category_dim: int):
        super().__init__()
        self.category_embedding = nn.Embedding(category_count, category_dim)
        self.query = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM * 2 + category_dim),
            nn.Linear(FEATURE_DIM * 2 + category_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.candidate = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07), dtype=torch.float32))

    def forward_with_category_embedding(
        self,
        history: torch.Tensor,
        context: torch.Tensor,
        category_embedding: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates from an explicit category embedding.

        Training calls :meth:`forward`; this interface permits a documented
        inference-only fallback when a test category is absent from the
        train-only PCAR category vocabulary.
        """
        query = self.query(torch.cat([history, context, category_embedding], dim=-1))
        query = F.normalize(query, dim=-1)
        candidate = F.normalize(self.candidate(candidates), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return scale * torch.einsum("bd,bkd->bk", query, candidate)

    def forward(self, history: torch.Tensor, context: torch.Tensor, categories: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return self.forward_with_category_embedding(
            history, context, self.category_embedding(categories), candidates
        )


def sample_negatives(categories: torch.Tensor, targets: torch.Tensor, pools: dict[int, np.ndarray], count: int, rng: np.random.Generator) -> np.ndarray:
    result = np.empty((len(categories), count), dtype=np.int64)
    categories_np = categories.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    for category in np.unique(categories_np):
        row_indices = np.flatnonzero(categories_np == category)
        pool = pools[int(category)]
        if len(pool) < 2:
            raise RuntimeError(f"category {category} has fewer than two catalog items")
        choices = rng.integers(0, len(pool), size=(len(row_indices), count))
        sampled = pool[choices]
        same = sampled == targets_np[row_indices, None]
        sampled[same] = pool[(choices[same] + 1) % len(pool)]
        result[row_indices] = sampled
    return result


@torch.inference_mode()
def evaluate(
    model: PCAR,
    loader: DataLoader,
    category_to_index: dict[int, int],
    pools: dict[int, np.ndarray],
    item_features: np.ndarray,
    num_negatives: int,
    device: torch.device,
    seed: int,
) -> tuple[float, float]:
    model.eval()
    rng = np.random.default_rng(seed)
    total_loss = total_top1 = 0.0
    seen = 0
    for history_batch, context_batch, category_batch, target_batch in loader:
        category_index = torch.as_tensor([category_to_index[int(value)] for value in category_batch.tolist()], dtype=torch.long)
        negative_ids = sample_negatives(category_batch, target_batch, pools, num_negatives, rng)
        candidate_ids = np.concatenate([target_batch.numpy()[:, None], negative_ids], axis=1)
        candidate_features = torch.from_numpy(np.asarray(item_features[candidate_ids], dtype=np.float32).copy())
        logits = model(
            history_batch.to(device, non_blocking=True),
            context_batch.to(device, non_blocking=True),
            category_index.to(device, non_blocking=True),
            candidate_features.to(device, non_blocking=True),
        )
        labels = torch.zeros(len(logits), dtype=torch.long, device=device)
        total_loss += float(F.cross_entropy(logits, labels)) * len(logits)
        total_top1 += float((logits.argmax(dim=1) == 0).sum())
        seen += len(logits)
    return total_loss / seen, total_top1 / seen


def main() -> None:
    args = parse_args()
    if args.num_negatives < 1 or args.epochs < 1 or args.batch_size < 1 or args.patience < 0:
        raise ValueError("--num-negatives, --epochs, and --batch-size must be positive")
    if args.num_workers != 0 and os.name == "nt":
        raise ValueError("use --num-workers 0 on Windows to avoid duplicating memory-mapped feature caches")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_root / args.train_file
    history_path = data_root / args.history_file
    feature_path = data_root / args.feature_file
    for path in (train_path, history_path, feature_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train = load_dict(train_path)
    history = load_dict(history_path)
    item_features = np.load(feature_path, mmap_mode="r")
    if item_features.ndim != 2 or item_features.shape[1] != FEATURE_DIM:
        raise ValueError(f"expected [N,{FEATURE_DIM}] features, found {item_features.shape}")

    examples = make_examples(train, args.max_outfits, args.examples_per_outfit, args.seed, args.val_fraction)
    max_id = max(max(target, *context) for _, _, target, context, _ in examples)
    if max_id >= len(item_features):
        raise IndexError(f"item id {max_id} exceeds feature table length {len(item_features)}")
    categories = sorted({category for _, category, _, _, _ in examples})
    category_to_index = {category: index for index, category in enumerate(categories)}
    pools_array = make_category_pools(train, set(categories), len(item_features))

    cache_root = output_dir / "feature_cache"
    paths = cache_paths(cache_root)
    expected = {
        "schema": 1,
        "feature_dim": FEATURE_DIM,
        "train_file": str(train_path),
        "history_file": str(history_path),
        "feature_file": str(feature_path),
        "max_outfits": args.max_outfits,
        "examples_per_outfit": args.examples_per_outfit,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "examples": len(examples),
    }
    if args.rebuild_cache or not cache_is_valid(paths, expected):
        build_cache(paths, expected, examples, history, item_features)

    dataset = CachedPCARDataset(paths, len(examples))
    split = np.asarray(dataset.split)
    train_indices = np.flatnonzero(split == 0).tolist()
    validation_indices = np.flatnonzero(split == 1).tolist()
    if not train_indices or not validation_indices:
        raise RuntimeError("the deterministic PCAR split yielded an empty train or validation subset")
    loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    model = PCAR(len(categories), args.hidden_dim, args.embedding_dim, args.category_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        if payload["category_ids"] != categories:
            raise ValueError("the resumed checkpoint was trained on different category IDs")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"])
        best_val_loss = float(payload.get("best_validation", {}).get("loss", float("inf")))
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))

    rng = np.random.default_rng(args.seed)
    run_config = vars(args).copy()
    run_config.update({
        "device_used": str(device), "category_ids": categories, "examples": len(examples),
        "train_examples": len(train_indices), "validation_examples": len(validation_indices),
        "objective": "same-category InfoNCE",
    })
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2, default=str), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        running_top1 = 0.0
        seen = 0
        progress = tqdm(loader, desc=f"PCAR epoch {epoch + 1}/{args.epochs}", unit="batch")
        for history_batch, context_batch, category_batch, target_batch in progress:
            category_values = category_batch.clone()
            category_index = torch.as_tensor([category_to_index[int(value)] for value in category_values.tolist()], dtype=torch.long)
            negative_ids = sample_negatives(category_values, target_batch, pools_array, args.num_negatives, rng)
            candidate_ids = np.concatenate([target_batch.numpy()[:, None], negative_ids], axis=1)
            candidate_features = torch.from_numpy(np.asarray(item_features[candidate_ids], dtype=np.float32).copy())

            history_batch = history_batch.to(device, non_blocking=True)
            context_batch = context_batch.to(device, non_blocking=True)
            category_index = category_index.to(device, non_blocking=True)
            candidate_features = candidate_features.to(device, non_blocking=True)
            logits = model(history_batch, context_batch, category_index, candidate_features)
            labels = torch.zeros(len(logits), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = len(logits)
            running_loss += float(loss.detach()) * batch_size
            running_top1 += float((logits.argmax(dim=1) == 0).sum().detach())
            seen += batch_size
            progress.set_postfix(loss=f"{running_loss / seen:.4f}", top1=f"{running_top1 / seen:.3f}")

        val_loss, val_top1 = evaluate(
            model, validation_loader, category_to_index, pools_array, item_features,
            args.num_negatives, device, args.seed + 991,
        )
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        training = {
            "loss": running_loss / seen,
            "top1": running_top1 / seen,
            "examples": seen,
            "validation_loss": val_loss,
            "validation_top1": val_top1,
            "validation_examples": len(validation_indices),
        }
        checkpoint = {
            "schema": 1,
            "reward": "PCAR",
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": {
                "feature_dim": FEATURE_DIM,
                "hidden_dim": args.hidden_dim,
                "embedding_dim": args.embedding_dim,
                "category_dim": args.category_dim,
            },
            "category_ids": categories,
            "training": training,
            "best_validation": {"loss": best_val_loss},
            "epochs_without_improvement": epochs_without_improvement,
            "gt_policy": "Training positives are only iFashion processed/train.npy targets; no FITB/GOR test GT was loaded.",
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        if args.save_epoch_checkpoints:
            torch.save(checkpoint, output_dir / f"epoch_{epoch + 1:02d}.pt")
        log_entry = {"epoch": epoch + 1, "improved": improved, **training}
        with (output_dir / "training_log.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry) + "\n")
        print(json.dumps(log_entry, indent=2))
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epochs_without_improvement} non-improving validation epochs.")
            break

    print(f"PCAR latest checkpoint: {output_dir / 'last.pt'}")
    print(f"PCAR best validation checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
