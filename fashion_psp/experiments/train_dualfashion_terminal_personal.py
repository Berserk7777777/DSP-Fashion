"""Train a causal terminal-Personal ensemble, calibrate uncertainty, replay PSP.

This fits the decision model only. No image generator or reward encoder is
trained or loaded. User-disjoint train/validation/calibration/test partitions
keep all seeds and timesteps of the same user together.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fashion_psp.search.terminal_personal import (
    FEATURE_NAMES, TerminalPersonalMLP, TerminalPersonalPredictor,
    prefix_features, uncertainty_survivors,
)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_trajectories(path, max_tasks):
    """Whitelist predictor inputs; GT/compatibility columns are never copied."""
    trajectories = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            task, seed, step = int(row["task_index"]), int(row["seed"]), int(row["step"])
            if task >= max_tasks or not 0 <= seed - (2026 + task * 10000) < 8:
                continue
            if not 1 <= step <= 28:
                raise ValueError(f"Unexpected step {step}")
            key = (task, seed)
            identity = (int(row["uid"]), int(row["oid"]))
            if key not in trajectories:
                trajectories[key] = {"task_index":task, "uid":identity[0], "oid":identity[1],
                                     "seed":seed, "personal":np.full(28, np.nan),
                                     "sigma":np.full(28, np.nan), "image_path":[None] * 28}
            item = trajectories[key]
            if (item["uid"], item["oid"]) != identity or np.isfinite(item["personal"][step - 1]):
                raise ValueError(f"Duplicate or inconsistent row {key}, step {step}")
            item["personal"][step - 1] = float(row["personal"])
            item["sigma"][step - 1] = float(row["sigma"])
            item["image_path"][step - 1] = row["image_path"]
    if {task for task, _ in trajectories} != set(range(max_tasks)):
        raise ValueError("Requested FITB task prefix is not fully covered")
    for task in range(max_tasks):
        values = [value for (index, _), value in trajectories.items() if index == task]
        if len(values) != 8 or len({(v['uid'], v['oid']) for v in values}) != 1:
            raise ValueError(f"Task {task} does not contain eight consistent particles")
        if any(not np.isfinite(v["personal"]).all() or not np.isfinite(v["sigma"]).all() for v in values):
            raise ValueError(f"Task {task} is missing steps 1..28")
    return trajectories


def split_users(trajectories, seed):
    users = np.asarray(sorted({row["uid"] for row in trajectories.values()}))
    if len(users) < 8:
        raise ValueError("Need at least eight users for four disjoint partitions")
    users = np.random.default_rng(seed).permutation(users)
    n_train, n_val, n_cal = max(1, int(len(users) * 0.60)), max(1, int(len(users) * 0.15)), max(1, int(len(users) * 0.15))
    boundaries = (n_train, n_train + n_val, n_train + n_val + n_cal)
    parts = np.split(users, boundaries)
    names = ("train", "validation", "calibration", "test")
    return {name:{"uids":[int(v) for v in part],
                  "task_indices":sorted({row["task_index"] for row in trajectories.values() if row["uid"] in set(part)})}
            for name, part in zip(names, parts)}


def build_arrays(trajectories):
    features, targets, current, users, tasks, seeds, steps = [], [], [], [], [], [], []
    for (task, seed), row in sorted(trajectories.items()):
        for step in range(1, 28):
            features.append(prefix_features(row["personal"][:step], row["sigma"][:step]))
            targets.append((row["personal"][27] - row["personal"][step - 1]) / 100.0)
            current.append(row["personal"][step - 1])
            users.append(row["uid"])
            tasks.append(task)
            seeds.append(seed)
            steps.append(step)
    return {"features":np.asarray(features, dtype=np.float32), "target":np.asarray(targets, dtype=np.float32),
            "current":np.asarray(current), "uid":np.asarray(users), "task":np.asarray(tasks),
            "seed":np.asarray(seeds), "step":np.asarray(steps)}


def cpu_state(model):
    return {key:value.detach().cpu().clone() for key, value in model.state_dict().items()}


def nll(model, x, y):
    mean, log_variance = model(x)
    return 0.5 * (log_variance + (y - mean).square() * (-log_variance).exp())


def train_member(args, member, arrays, splits, mean, scale, config_hash):
    folder = args.output_dir / f"member_{member:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    member_seed = args.seed + member * 1009
    torch.manual_seed(member_seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(member_seed)
    model = TerminalPersonalMLP(args.hidden_dim).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    train_mask = np.isin(arrays["uid"], splits["train"]["uids"])
    val_mask = np.isin(arrays["uid"], splits["validation"]["uids"])
    x = torch.as_tensor((arrays["features"][train_mask] - mean) / scale, device=args.device)
    y = torch.as_tensor(arrays["target"][train_mask], device=args.device)
    vx = torch.as_tensor((arrays["features"][val_mask] - mean) / scale, device=args.device)
    vy = torch.as_tensor(arrays["target"][val_mask], device=args.device)
    train_uids = np.asarray(splits["train"]["uids"])
    bootstrap = np.random.default_rng(member_seed).choice(train_uids, size=len(train_uids), replace=True)
    multiplicity = {int(uid):int(np.sum(bootstrap == uid)) for uid in train_uids}
    train_rows_uids = arrays["uid"][train_mask]
    counts = {int(uid):int(np.sum(train_rows_uids == uid)) for uid in train_uids}
    weights = np.asarray([multiplicity[int(uid)] / counts[int(uid)] for uid in train_rows_uids], dtype=np.float32)
    # Equal total weight per bootstrapped user, regardless of outfit count.
    weights /= weights.mean()
    weights = torch.as_tensor(weights, device=args.device)
    start_epoch, best_loss, stale, history = 1, float("inf"), 0, []
    latest_path, best_path = folder / "latest.pt", folder / "best.pt"
    if args.resume and latest_path.is_file():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state["config_hash"] != config_hash:
            raise ValueError("Member resume configuration mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best_loss, stale = state["epoch"] + 1, state["best_validation_nll"], state["stale"]
        history = state["history"]
        torch.set_rng_state(state["torch_rng"])
        if args.device.startswith("cuda") and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Member {member + 1}: resume epoch {start_epoch}, best val NLL={best_loss:.5f}", flush=True)
    if args.patience and stale >= args.patience:
        print(f"Member {member + 1}: previously early-stopped; retaining best checkpoint", flush=True)
    else:
        with tqdm(range(start_epoch, args.epochs + 1), desc=f"Terminal model {member + 1}/{args.ensemble_size}", unit="epoch") as bar:
            for epoch in bar:
                model.train()
                order = torch.randperm(len(x))
                train_loss = 0.0
                for begin in range(0, len(order), args.batch_size):
                    ids = order[begin:begin + args.batch_size].to(args.device)
                    optimizer.zero_grad(set_to_none=True)
                    losses = nll(model, x[ids], y[ids])
                    loss = (losses * weights[ids]).mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    train_loss += float(loss.detach()) * len(ids)
                model.eval()
                with torch.inference_mode():
                    val_loss = sum(float(nll(model, vx[b:b + args.batch_size], vy[b:b + args.batch_size]).sum())
                                   for b in range(0, len(vx), args.batch_size)) / len(vx)
                if not np.isfinite(val_loss):
                    raise RuntimeError("Nonfinite validation loss")
                improved = val_loss < best_loss - 1e-4
                if improved:
                    best_loss, stale = val_loss, 0
                    atomic_torch(best_path, {"model":cpu_state(model), "epoch":epoch,
                                            "validation_nll":val_loss, "config_hash":config_hash})
                else:
                    stale += 1
                history.append({"member":member, "epoch":epoch, "train_nll":train_loss / len(x),
                                "validation_nll":val_loss, "best_validation_nll":best_loss,
                                "improved":improved, "stale":stale})
                atomic_torch(latest_path, {"model":cpu_state(model), "optimizer":optimizer.state_dict(),
                                          "epoch":epoch, "best_validation_nll":best_loss, "stale":stale,
                                          "history":history, "config_hash":config_hash,
                                          "torch_rng":torch.get_rng_state(),
                                          "cuda_rng":torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else None})
                write_csv(folder / "training_log.csv", history)
                bar.set_postfix(train=f"{train_loss / len(x):.4f}", val=f"{val_loss:.4f}", best=f"{best_loss:.4f}")
                if args.patience and stale >= args.patience:
                    break
    if not best_path.is_file():
        raise RuntimeError(f"Missing best member model: {best_path}")
    return torch.load(best_path, map_location="cpu", weights_only=False)


def calibrate(bundle, arrays, splits, coverage, device):
    mask = np.isin(arrays["uid"], splits["calibration"]["uids"])
    predictor = TerminalPersonalPredictor(bundle, device)
    mean, std = predictor.predict_features(arrays["features"][mask], arrays["current"][mask])
    target = arrays["current"][mask] + arrays["target"][mask] * 100
    standardized_error = np.abs(target - mean) / std
    uids = arrays["uid"][mask]
    # One calibration score per independent user: maximum over ALL their
    # tasks, 8 particles, and 27 prefixes. Covers adaptive node access too,
    # under the standard exchangeability assumption for future users.
    scores = [float(standardized_error[uids == uid].max()) for uid in splits["calibration"]["uids"]]
    rank = math.ceil((len(scores) + 1) * coverage)
    if rank > len(scores):
        raise ValueError(f"{len(scores)} calibration users cannot support coverage={coverage}; lower --coverage or use more users")
    q = sorted(scores)[rank - 1]
    return {"q":q, "coverage_target":coverage, "group":"uid", "scope":"all tasks, 8 seeds, steps 1..27 per user",
            "finite_sample_order_statistic":rank, "n_calibration_users":len(scores),
            "user_max_standardized_errors":dict(zip(map(str, splits["calibration"]["uids"]), scores)),
            "assumption":"Exchangeable calibration and future users; model and normalization fixed before calibration.",
            "warning":"Forced budget pruning can remove overlapping contenders; interval coverage does not certify those removals."}


def forecast_diagnostics(bundle, arrays, splits, device, output):
    predictor = TerminalPersonalPredictor(bundle, device)
    results = []
    for split, partition in splits.items():
        mask = np.isin(arrays["uid"], partition["uids"])
        mean, std = predictor.predict_features(arrays["features"][mask], arrays["current"][mask])
        target = arrays["current"][mask] + arrays["target"][mask] * 100
        error = np.abs(target - mean)
        covered = error <= predictor.q * std
        users = arrays["uid"][mask]
        for step in (7, 14, -1):
            node = arrays["step"][mask] == step if step > 0 else np.ones(len(mean), dtype=bool)
            results.append({"split":split, "step":step, "n_prefixes":int(node.sum()),
                            "mae":float(error[node].mean()), "rmse":float(np.sqrt(np.mean(error[node] ** 2))),
                            "pointwise_coverage":float(covered[node].mean()),
                            "simultaneous_user_coverage":float(np.mean([covered[node & (users == uid)].all()
                                                                       for uid in partition["uids"]])),
                            "mean_interval_width":float((2 * predictor.q * std[node]).mean())})
    write_csv(output / "forecast_diagnostics.csv", results)


def simulate_task(trajectories, task, predictor, policy):
    rows = {seed:row for (index, seed), row in trajectories.items() if index == task}
    alive = sorted(rows)
    trace, calls, previous = [], 0, 0
    for step, keep in ((7, 4), (14, 2), (28, 1)):
        calls += len(alive) * (step - previous)
        previous = step
        if step == 28 or policy == "raw_personal":
            # Only the currently visible reward is read here.
            scores = {seed:rows[seed]["personal"][step - 1] for seed in alive}
            survivors = sorted(alive, key=lambda seed:(-scores[seed], seed))[:keep]
            detail = {"observed_personal":{str(seed):float(value) for seed, value in scores.items()}}
        else:
            mean, std = predictor.predict([rows[s]["personal"][:step] for s in alive],
                                          [rows[s]["sigma"][:step] for s in alive])
            if policy == "predicted_mean":
                indices = sorted(range(len(alive)), key=lambda i:(-mean[i], alive[i]))[:keep]
                survivors = [alive[i] for i in indices]
                detail = {"predicted_mean":{str(seed):float(mean[i]) for i, seed in enumerate(alive)}}
            else:
                survivors, detail = uncertainty_survivors(alive, mean, std, predictor.q, keep)
        trace.append({"step":step, "alive_before":alive[:], "survivors":survivors[:], **detail})
        alive = survivors
    if calls != 112:
        raise AssertionError(f"Unexpected compute budget: {calls}")
    return alive[0], calls, trace


def replay(bundle, trajectories, splits, output, device, posthoc_csv=None):
    predictor = TerminalPersonalPredictor(bundle, device)
    split_of = {uid:name for name, part in splits.items() for uid in part["uids"]}
    task_ids = sorted({task for task, _ in trajectories})
    rows, traces = [], {}
    for task in tqdm(task_ids, desc="112-step offline policy replay", unit="task"):
        candidates = {seed:row for (index, seed), row in trajectories.items() if index == task}
        first = candidates[min(candidates)]
        oracle = max(candidates, key=lambda seed:(candidates[seed]["personal"][27], -seed))
        for policy in ("raw_personal", "predicted_mean", "uncertainty_anchor_ucb", "bon4", "bon8"):
            if policy.startswith("bon"):
                pool = sorted(candidates)[:int(policy[3:])]
                winner = min(pool, key=lambda seed:(-candidates[seed]["personal"][27], seed))
                calls, trace = len(pool) * 28, []
            else:
                winner, calls, trace = simulate_task(trajectories, task, predictor, policy)
            rows.append({"split":split_of[first["uid"]], "task_index":task, "uid":first["uid"], "oid":first["oid"],
                         "policy":policy, "seed":winner, "particle_steps":calls,
                         "personal":float(candidates[winner]["personal"][27]),
                         "personal_regret8":float(candidates[oracle]["personal"][27] - candidates[winner]["personal"][27]),
                         "personal_oracle8_hit":int(winner == oracle),
                         "forced_prunes":sum(node.get("forced_prunes", 0) for node in trace),
                         "image_path":candidates[winner]["image_path"][27]})
            traces[f"{task}:{policy}"] = trace
    # Freeze complete selection mapping BEFORE optional GT diagnostics are read.
    write_csv(output / "selection_pre_gt.csv", rows)
    atomic_json(output / "selection_traces.json", traces)
    posthoc = {}
    if posthoc_csv is not None:
        with posthoc_csv.open(newline="", encoding="utf-8-sig") as handle:
            for record in csv.DictReader(handle):
                if int(record["step"]) == 28:
                    key = (int(record["task_index"]), int(record["seed"]))
                    posthoc[key] = record
        optional = [name for name in ("gt_sim", "gaa", "cis", "lpips")
                    if posthoc and name in next(iter(posthoc.values()))]
        for row in rows:
            record = posthoc[(row["task_index"], row["seed"])]
            if Path(record["image_path"]).resolve() != Path(row["image_path"]).resolve():
                raise ValueError("Post-hoc metrics refer to a different image/cache")
            for key in optional:
                row[f"posthoc_{key}"] = float(record[key])
        write_csv(output / "selection_posthoc.csv", rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["policy"])].append(row)
        groups[("all_exploratory", row["policy"])].append(row)
    summaries = []
    for (split, policy), values in sorted(groups.items()):
        summary = {"split":split, "policy":policy, "n_tasks":len(values),
                   "particle_steps":values[0]["particle_steps"]}
        for key in ("personal", "personal_regret8", "personal_oracle8_hit", "forced_prunes"):
            summary[key] = float(np.mean([row[key] for row in values]))
        for key in values[0]:
            if key.startswith("posthoc_"):
                summary[key] = float(np.mean([row[key] for row in values]))
        summaries.append(summary)
    write_csv(output / "policy_summary.csv", summaries)
    for row in summaries:
        if row["split"] == "test":
            print(f"[Held-out users] {row['policy']}: Personal={row['personal']:.3f}, "
                  f"regret8={row['personal_regret8']:.3f}, hit={row['personal_oracle8_hit']:.1%}, "
                  f"steps={row['particle_steps']}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=ROOT / "DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fashion_psp/artifacts/terminal_personal/fitb200_v1")
    parser.add_argument("--posthoc-metrics-csv", type=Path, default=None, help="Optional diagnostic CSV, opened only after selection is frozen.")
    parser.add_argument("--max-tasks", type=int, default=200)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15, help="0 disables early stopping.")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replay-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ensemble_size < 2 or args.epochs < 1 or args.batch_size < 1 or not 0 < args.coverage < 1:
        raise ValueError("Need >=2 ensemble members, positive epochs/batch size, and 0<coverage<1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate dualfashion or pass --device cpu")
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if args.device.startswith("cuda"):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
    trajectories = load_trajectories(args.metrics_csv, args.max_tasks)
    splits = split_users(trajectories, args.seed)
    n_calibration = len(splits["calibration"]["uids"])
    if math.ceil((n_calibration + 1) * args.coverage) > n_calibration:
        raise ValueError(f"Only {n_calibration} calibration users; lower --coverage or supply more tasks before training")
    arrays = build_arrays(trajectories)
    train_mask = np.isin(arrays["uid"], splits["train"]["uids"])
    mean = arrays["features"][train_mask].mean(0)
    scale = np.maximum(arrays["features"][train_mask].std(0), 1e-5)
    config = {"schema_version":1, "metrics_csv":str(args.metrics_csv.resolve()),
              "metrics_sha256":hashlib.sha256(args.metrics_csv.read_bytes()).hexdigest(),
              "max_tasks":args.max_tasks, "ensemble_size":args.ensemble_size, "hidden_dim":args.hidden_dim,
              "seed":args.seed, "batch_size":args.batch_size, "learning_rate":args.learning_rate,
              "patience":args.patience, "coverage":args.coverage, "feature_names":list(FEATURE_NAMES),
              "target":"(Personal(step28) - observed Personal(step t))/100; t=1..27",
              "reward_view":"cached decoded x0 at step28; not an independently rescored terminal scheduler decode",
              "seed_pool":"2026 + task_index*10000 + slot, slot=0..7",
              "generator_training":False, "uses_gt_for_training_or_selection":False,
              "split_policy":"uid-disjoint 60/15/15/remaining percent; validation chooses epochs; calibration only sets intervals",
              "splits":splits}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = args.output_dir / "training_manifest.json"
    if manifest.is_file():
        if json.loads(manifest.read_text(encoding="utf-8")) != config:
            raise ValueError("Configuration differs; use a new output directory")
        if not args.resume and not args.replay_only:
            raise FileExistsError("Run exists; pass --resume or use another output directory")
    else:
        if args.replay_only:
            raise FileNotFoundError("No completed training run to replay")
        atomic_json(manifest, config)
        atomic_json(args.output_dir / "split.json", splits)
    print(f"Python: {sys.executable}\nDevice: {args.device}; tasks={args.max_tasks}; trajectories={len(trajectories)}; prefix examples={len(arrays['target'])}", flush=True)
    for name, partition in splits.items():
        print(f"{name}: {len(partition['uids'])} users, {len(partition['task_indices'])} tasks", flush=True)
    if args.replay_only:
        bundle = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=False)
        if bundle["config_hash"] != config_hash:
            raise ValueError("Bundle configuration mismatch")
    else:
        states = [train_member(args, member, arrays, splits, mean, scale, config_hash)
                  for member in range(args.ensemble_size)]
        bundle = {"feature_names":list(FEATURE_NAMES), "feature_mean":mean.tolist(), "feature_scale":scale.tolist(),
                  "total_steps":28, "hidden_dim":args.hidden_dim, "config_hash":config_hash,
                  "member_states":[state["model"] for state in states],
                  "member_best_epochs":[state["epoch"] for state in states]}
        bundle["calibration"] = calibrate(bundle, arrays, splits, args.coverage, args.device)
        atomic_torch(args.output_dir / "best.pt", bundle)
        atomic_json(args.output_dir / "calibration.json", bundle["calibration"])
    print(f"Calibrated q={bundle['calibration']['q']:.3f}; this is NOT a guarantee for forced budget removals.", flush=True)
    forecast_diagnostics(bundle, arrays, splits, args.device, args.output_dir)
    replay(bundle, trajectories, splits, args.output_dir, args.device, args.posthoc_metrics_csv)
    print(f"Complete: {args.output_dir.resolve()}\nDeployable ensemble: {args.output_dir.resolve() / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
