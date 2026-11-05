"""Personal-only analysis of the complete 200-task/16-seed/28-step cache.

No parameterized model is fitted. All candidate choices use currently observed
score prefixes. Terminal rewards are used only to assess/tune global policies,
and never by the per-task pruning decision. Results are exploratory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_data(args):
    scores = np.full((200, 28, 16), np.nan)
    uids, oids = np.full(200, -1), np.full(200, -1)
    paths = {}
    with args.metrics_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            task, seed, step = int(row["task_index"]), int(row["seed"]), int(row["step"])
            slot = seed - (2026 + task * 10000)
            if not 0 <= task < 200 or not 0 <= slot < 16 or not 1 <= step <= 28:
                raise ValueError("Unexpected task/seed/step in the merged score file")
            if np.isfinite(scores[task, step - 1, slot]):
                raise ValueError("Duplicate trajectory observation")
            uid, oid = int(row["uid"]), int(row["oid"])
            if uids[task] != -1 and (uids[task], oids[task]) != (uid, oid):
                raise ValueError("Task identity mismatch")
            uids[task], oids[task] = uid, oid
            scores[task, step - 1, slot] = float(row["personal"])
            paths[(task, step, seed)] = Path(row["image_path"]).resolve()
    if not np.isfinite(scores).all():
        raise ValueError("Need all 89,600 finite score observations")
    meta_checks = []
    for task in tqdm(range(200), desc="Validate both cache shards", unit="task"):
        hashes = []
        reference = None
        for base, root in ((2026, args.cache_root), (2034, args.extra_cache_root)):
            path = root / f"psp_offline_group0_task{task}_seed{base}" / "metadata.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            if (meta["task_index"], meta["uid"], meta["oid"]) != (task, int(uids[task]), int(oids[task])):
                raise ValueError(f"Wrong metadata identity: {path}")
            if meta["candidate_seeds"] != [base + task * 10000 + i for i in range(8)]:
                raise ValueError("Unexpected seed pool")
            view = meta["offline_candidate_cache"]
            if view["steps"] != list(range(1, 29)) or len(view["images"]) != 224:
                raise ValueError("Incomplete shard metadata")
            for record in view["images"]:
                key = (task, int(record["step"]), int(record["seed"]))
                if paths[key] != Path(record["image_path"]).resolve():
                    raise ValueError("Merged scores point to a different image than cache metadata")
                if not paths[key].is_file():
                    raise FileNotFoundError(paths[key])
            fields = {k:meta[k] for k in ("checkpoint", "precision", "deterministic_context", "guidance_scales", "num_inference_steps")}
            if reference is not None and fields != reference:
                raise ValueError("The two shards have different generation configurations")
            reference = fields
            hashes.extend(meta["initial_latent_hashes"].values())
        if len(hashes) != 16 or len(set(hashes)) != 16:
            raise ValueError("Initial noise tensors overlap across particles")
        meta_checks.append({"task_index":task, "uid":int(uids[task]), "oid":int(oids[task]), "unique_initial_hashes":16})
    write_csv(args.output_dir / "cache_validation.csv", meta_checks)
    return scores, uids, oids, paths


def percentile_rank(values):
    # Average ties; larger values have larger ranks. Works on (..., particles).
    if values.shape[-1] == 1:
        return np.ones_like(values)
    less = (values[..., :, None] > values[..., None, :]).sum(-1)
    equal = (values[..., :, None] == values[..., None, :]).sum(-1)
    return (less + (equal - 1) / 2) / (values.shape[-1] - 1)


def survivor_indices(p, alive, step, keep, rule):
    current = np.take_along_axis(p[:, step - 1, :], alive, axis=1)
    # Prefix cohorts consist ONLY of still-live seeds. We never read scores
    # of a pruned particle at a later time for a decision.
    if rule == "raw":
        value = current
    else:
        window = int(rule.rsplit("_", 1)[1])
        start = max(0, step - window)
        history = np.take_along_axis(p[:, start:step, :], alive[:, None, :], axis=2)
        if rule.startswith("mean_"):
            value = history.mean(1)
        elif rule.startswith("rank_mean_"):
            value = percentile_rank(history).mean(1)
        elif rule.startswith("rank_trend_"):
            rank = percentile_rank(history)
            value = rank.mean(1) + 0.5 * (rank[:, -1, :] - rank[:, 0, :])
        elif rule.startswith("rank_recent_"):
            rank = percentile_rank(history)
            weights = np.arange(1, history.shape[1] + 1, dtype=float)
            value = (rank * weights[None, :, None]).sum(1) / weights.sum()
        elif rule.startswith("lcb_") or rule.startswith("ucb_"):
            delta = np.diff(history, axis=1)
            instability = delta.std(1) if len(delta[0]) else np.zeros_like(current)
            value = current + (1 if rule.startswith("ucb_") else -1) * instability
        elif rule.startswith("slope_") or rule.startswith("rescue_"):
            time = np.arange(history.shape[1], dtype=float)
            time -= time.mean()
            slope = (history * time[None, :, None]).sum(1) / max(float(time @ time), 1.0)
            if rule.startswith("rescue_"):
                order = np.argsort(-current, axis=1, kind="stable")
                if keep == 1:
                    return np.take_along_axis(alive, order[:, :1], axis=1)
                fixed = order[:, :keep - 1]
                fringe = order[:, keep - 1:min(2 * keep, alive.shape[1])]
                fringe_slopes = np.take_along_axis(slope, fringe, axis=1)
                rescue = np.take_along_axis(fringe, np.argmax(fringe_slopes, axis=1)[:, None], axis=1)
                # Spend the rescue slot only on positive trend; otherwise keep
                # the next raw-score candidate. This is a fixed heuristic.
                rising = fringe_slopes.max(1) > 0
                rescue = np.where(rising[:, None], rescue, order[:, keep - 1:keep])
                chosen = np.concatenate([fixed, rescue], axis=1)
                return np.take_along_axis(alive, chosen, axis=1)
            horizon = min(4, 28 - step)
            value = current + horizon * slope
        else:
            raise ValueError(f"Unknown rule {rule}")
    chosen = np.argsort(-value, axis=1, kind="stable")[:, :keep]
    return np.take_along_axis(alive, chosen, axis=1)


def schedule_cost(k, nodes):
    cost, previous, alive = 0, 0, k
    for step, keep in nodes:
        if not previous < step <= 28 or not 1 <= keep < alive:
            raise ValueError(f"Invalid pruning path {k}, {nodes}")
        cost += alive * (step - previous)
        previous, alive = step, keep
    return cost + alive * (28 - previous)


def reward_view_count(k, nodes, rule="raw"):
    """Minimum unique decoded/scored seed-step views, reusing past views."""
    if rule.startswith("adaptive"):
        return None  # task-dependent; handled separately in future online work
    window = 1 if rule == "raw" else int(rule.rsplit("_", 1)[1])
    observed, count, alive = set(), 0, k
    for step, keep in nodes:
        needed = set(range(max(1, step - window + 1), step + 1))
        count += len(needed - observed) * alive
        observed |= needed
        alive = keep
    count += (28 not in observed) * alive
    return count


def annotate_reward_costs(output):
    with (output / "best_by_initial_particles.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        count = reward_view_count(int(row["initial_particles"]), json.loads(row["nodes"]), row["rule"])
        row["reward_views"] = count if count is not None else "task-dependent"
    write_csv(output / "best_policies_with_reward_costs.csv", rows)


def pruning_diagnostics(p, output):
    """Terminal labels diagnose already-frozen pruning, never choose survivors."""
    with (output / "policy_summary.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [row for row in rows if int(row["particle_steps"]) == 112 and int(row["pool_offset"]) == 0]
    simple = []
    for k in (8, 12, 16):
        for max_nodes in (2, 3, 4, 5, 99):
            subset = [row for row in candidates if int(row["initial_particles"]) == k
                      and row["rule"] == "raw" and len(json.loads(row["nodes"])) <= max_nodes]
            if subset:
                best = max(subset, key=lambda row:float(row["personal"]))
                simple.append({"initial_particles":k, "max_intermediate_nodes":max_nodes,
                               "name":best["name"], "personal":best["personal"],
                               "delta_default8":best["delta_default8"], "nodes":best["nodes"],
                               "reward_views":reward_view_count(k, json.loads(best["nodes"]), "raw")})
    write_csv(output / "schedule_complexity_comparison.csv", simple)
    selected = [next(row for row in rows if row["name"] == "default_psp8")]
    for k in (8, 9, 10, 12, 16):
        selected.append(max((row for row in candidates if int(row["initial_particles"]) == k and row["rule"] == "raw"),
                            key=lambda row:float(row["personal"])))
    node_rows = []
    for row in selected:
        k, nodes = int(row["initial_particles"]), json.loads(row["nodes"])
        _, _, trace = evaluate(p, k, nodes, capture=True)
        oracle_id = np.argmax(p[:, 27, :k], axis=1)
        alive = np.tile(np.arange(k), (len(p), 1))
        total_loss = np.zeros(len(p))
        for step, survivors in trace:
            before = np.take_along_axis(p[:, 27, :], alive, axis=1).max(1)
            after = np.take_along_axis(p[:, 27, :], survivors, axis=1).max(1)
            total_loss += before - after
            node_rows.append({"name":row["name"], "step":step, "alive_before":alive.shape[1],
                              "alive_after":survivors.shape[1], "mean_irrecoverable_personal_loss":float((before - after).mean()),
                              "cumulative_personal_loss":float(total_loss.mean()),
                              "own_terminal_winner_survival":float((survivors == oracle_id[:, None]).any(1).mean()),
                              "best_remaining_terminal_personal":float(after.mean())})
            alive = survivors
        if not np.isclose(total_loss.mean(), float(row["regret_own_pool"]), atol=1e-6):
            raise AssertionError("Pruning losses do not sum to same-pool regret")
    write_csv(output / "irrecoverable_pruning_losses.csv", node_rows)


def evaluate(p, k, nodes, rule="raw", offset=0, capture=False):
    alive = np.tile(np.arange(offset, offset + k), (len(p), 1))
    trace = []
    for step, keep in nodes:
        alive = survivor_indices(p, alive, step, keep, rule)
        if capture:
            trace.append((step, alive.copy()))
    terminal = np.take_along_axis(p[:, 27, :], alive, axis=1)
    selected = np.argmax(terminal, axis=1)
    winner = np.take_along_axis(alive, selected[:, None], axis=1)[:, 0]
    result = p[np.arange(len(p)), 27, winner]
    return result, winner, trace


def practical_policy_checks(p, uids, oids, paths, output):
    """Freeze simple tuned schedules and measure seed-pool sensitivity.

    These are exploratory checks on the same tasks, not independent validation.
    The separate permutation RNG makes this check reproducible without replaying
    the large policy search or depending on its bootstrap RNG consumption.
    """
    policies = {
        "default8": [(7, 4), (14, 2)],
        "simple8_two_cuts": [(2, 5), (11, 3)],
        "simple8_three_cuts": [(4, 6), (8, 4), (16, 2)],
        "simple8_four_cuts": [(2, 6), (8, 5), (12, 3), (20, 2)],
        "best_searched8": [(3, 7), (4, 6), (8, 4), (14, 3), (19, 2)],
    }
    base, _, _ = evaluate(p, 8, policies["default8"])
    extra_base, _, _ = evaluate(p, 8, policies["default8"], offset=8)
    rows, selected, robustness = [], [], []
    rng = np.random.default_rng(2028)
    for repeat in range(32):
        permutation = np.argsort(rng.random((len(p), 16)), axis=1)
        permuted = np.take_along_axis(p, permutation[:, None, :], axis=2)
        random_base, _, _ = evaluate(permuted, 8, policies["default8"])
        for name, nodes in policies.items():
            value, _, _ = evaluate(permuted, 8, nodes)
            robustness.append({"name":name, "repeat":repeat,
                               "personal":float(value.mean()),
                               "delta_random_default8":float((value - random_base).mean())})
    for name, nodes in policies.items():
        value, winner, _ = evaluate(p, 8, nodes)
        extra, _, _ = evaluate(p, 8, nodes, offset=8)
        deltas = np.asarray([r["delta_random_default8"] for r in robustness if r["name"] == name])
        oracle = p[:, 27, :8].max(1)
        rows.append({"name":name, "nodes":json.dumps(nodes),
                     "particle_steps":schedule_cost(8, nodes),
                     "reward_views":reward_view_count(8, nodes),
                     "original8_personal":float(value.mean()),
                     "delta_original_default8":float((value - base).mean()),
                     "original8_regret":float((oracle - value).mean()),
                     "original8_oracle_hit":float(np.isclose(value, oracle, atol=1e-7, rtol=0).mean()),
                     "extra8_personal":float(extra.mean()),
                     "delta_extra_default8":float((extra - extra_base).mean()),
                     "mean_delta_random_default8":float(deltas.mean()),
                     "std_delta_random_default8":float(deltas.std()),
                     "positive_permutations":int((deltas > 0).sum()),
                     "permutations":len(deltas), "permutation_rng_seed":2028})
        for task, slot in enumerate(winner):
            seed = int(2026 + task * 10000 + slot)
            selected.append({"name":name, "task_index":task, "uid":int(uids[task]),
                             "oid":int(oids[task]), "seed":seed, "personal":float(value[task]),
                             "image_path":str(paths[(task, 28, seed)])})
    write_csv(output / "practical_policy_checks.csv", rows)
    write_csv(output / "practical_policy_permutations.csv", robustness)
    write_csv(output / "practical_policy_selected_images.csv", selected)


def schedule_key(k, nodes):
    return (k, tuple(tuple(node) for node in nodes))


def enumerate_candidates(args):
    schedules = {schedule_key(k, []):[] for k in (1, 2, 3, 4)}
    # Exhaust every 1- and 2-intermediate-pruning schedule for every initial K.
    for k in range(5, 17):
        for keep in range(1, k):
            for step in range(1, 28):
                nodes = [(step, keep)]
                if schedule_cost(k, nodes) == 112:
                    schedules[schedule_key(k, nodes)] = nodes
        for first in range(2, k):
            for second in range(1, first):
                for t1 in range(1, 27):
                    numerator = 112 - 28 * second - (k - first) * t1
                    if numerator % (first - second):
                        continue
                    t2 = numerator // (first - second)
                    if t1 < t2 < 28:
                        nodes = [(t1, first), (t2, second)]
                        schedules[schedule_key(k, nodes)] = nodes
    # Reuse the actual previously searched schedules rather than inventing a
    # claim of exhaustive search over arbitrarily many pruning stages.
    for source_index, path in enumerate(args.grid_csv):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for rank, row in enumerate(csv.DictReader(handle)):
                # The unrestricted historical search has millions of rows.
                # Its ranked prefix adds rich schedules without rebuilding a
                # multi-gigabyte exhaustive-policy registry in memory.
                if source_index > 0 and rank >= args.additional_grid_limit:
                    break
                k = int(row["initial_particles"])
                nodes = list(zip(json.loads(row["cutoff_times"]), json.loads(row["remaining_seeds"])))
                if 4 <= k <= 16 and schedule_cost(k, nodes) == 112:
                    schedules[schedule_key(k, nodes)] = nodes
    return schedules


def adaptive_evaluate(p, threshold, stable_window=3):
    # For 8 -> 4 -> 2, exact-budget feasibility requires t2=28-2*t1.
    values, winners, t1s = np.empty(len(p)), np.empty(len(p), dtype=int), np.empty(len(p), dtype=int)
    for task in range(len(p)):
        first = 9
        for step in range(5, 10):
            current = p[task, step - 1, :8]
            order = np.argsort(-current, kind="stable")
            gap = (current[order[3]] - current[order[4]]) / max(float(current.std()), 1e-8)
            window = p[task, max(0, step - stable_window):step, :8]
            sets = [set(np.argsort(-row, kind="stable")[:4]) for row in window]
            stable = len(set.intersection(*sets)) == 4
            if gap >= threshold and stable:
                first = step
                break
        second = 28 - 2 * first
        result, winner, _ = evaluate(p[task:task + 1], 8, [(first, 4), (second, 2)])
        values[task], winners[task], t1s[task] = result[0], winner[0], first
    return values, winners, t1s


def mechanism(p, output):
    rows = []
    for k in (8, 12, 16):
        final = p[:, 27, :k]
        best = np.argmax(final, axis=1)
        final_rank = percentile_rank(final)
        final_rank -= final_rank.mean(1, keepdims=True)
        for step in range(1, 29):
            current = p[:, step - 1, :k]
            rank = percentile_rank(current)
            centered = rank - rank.mean(1, keepdims=True)
            correlation = (centered * final_rank).sum(1) / np.maximum(np.sqrt((centered ** 2).sum(1) * (final_rank ** 2).sum(1)), 1e-10)
            order = np.argsort(-current, axis=1, kind="stable")
            row = {"initial_particles":k, "step":step, "mean_rank_correlation_with_terminal":float(correlation.mean())}
            for keep in sorted({1, 2, 4, k // 2}):
                row[f"terminal_winner_in_current_top{keep}"] = float(np.mean((order[:, :keep] == best[:, None]).any(1)))
            rows.append(row)
    # Different K has different top-K fields.
    columns = sorted(set().union(*(row.keys() for row in rows)))
    write_csv(output / "step_information.csv", [{key:row.get(key, "") for key in columns} for row in rows])
    pool_rows = [{"particles":k, "oracle_personal":float(p[:, 27, :k].max(1).mean()),
                  "full_generation_steps":k * 28,
                  "latest_possible_first_prune_to_complete_one":min(27, int(84 // (k - 1))) if k > 1 else 28}
                 for k in range(1, 17)]
    write_csv(output / "pool_oracles.csv", pool_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT / "DualFashion-main/output"
    parser.add_argument("--metrics-csv", type=Path, default=root / "ifashion_fitb_timestep200_16particle_analysis/candidate_metrics_personal_all_steps.csv")
    parser.add_argument("--cache-root", type=Path, default=root / "ifashion_fitb_timestep200_cache")
    parser.add_argument("--extra-cache-root", type=Path, default=root / "ifashion_fitb_timestep200_cache_extra8")
    parser.add_argument("--output-dir", type=Path, default=root / "ifashion_fitb_training_free_analysis")
    parser.add_argument("--grid-csv", type=Path, action="append", default=None)
    parser.add_argument("--shortlist-per-k", type=int, default=20)
    parser.add_argument("--additional-grid-limit", type=int, default=2000)
    parser.add_argument("--robustness-repeats", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.grid_csv is None:
        args.grid_csv = [root / "ifashion_fitb_timestep200_16particle_analysis/grid_search/personal_psp_style_kset_dense/grid_search_summary.csv",
                         root / "ifashion_fitb_timestep200_16particle_analysis/grid_search/personal_unified_exact112/grid_search_summary.csv"]
    p, uids, oids, paths = load_data(args)
    mechanism(p, args.output_dir)
    baseline_values, baseline_winners, _ = evaluate(p, 8, [(7, 4), (14, 2)])
    oracle16 = p[:, 27, :].max(1)
    results, candidates, all_values, all_winners = [], [], [], []

    def register(k, nodes, rule, values, winners, label=None, cost=None, offset=0):
        cost = schedule_cost(k, nodes) if cost is None else cost
        own_oracle = p[:, 27, offset:offset + k].max(1)
        name = label or f"K{k}:" + ",".join(f"{t}:{keep}" for t, keep in nodes) + f":{rule}"
        row = {"policy_id":len(results), "name":name, "initial_particles":k, "rule":rule,
               "nodes":json.dumps(nodes), "particle_steps":cost, "pool_offset":offset,
               "personal":float(values.mean()), "delta_default8":float((values - baseline_values).mean()),
               "regret_own_pool":float((own_oracle - values).mean()),
               "coverage_gap_to16":float((oracle16 - own_oracle).mean()),
               "regret_to16":float((oracle16 - values).mean()),
               "own_oracle_hit":float(np.isclose(values, own_oracle, atol=1e-7, rtol=0).mean()),
               "win_rate_default8":float((values > baseline_values + 1e-7).mean())}
        results.append(row)
        candidates.append((k, nodes, rule, offset))
        all_values.append(values.astype(np.float32))
        all_winners.append(winners.astype(np.int16))
        return row

    register(8, [(7, 4), (14, 2)], "raw", baseline_values, baseline_winners, "default_psp8")
    extra_values, extra_winners, _ = evaluate(p, 8, [(7, 4), (14, 2)], offset=8)
    register(8, [(7, 4), (14, 2)], "raw", extra_values, extra_winners, "default_psp8_extra_pool", offset=8)
    for k in (4, 8, 12, 16):
        value, winner, _ = evaluate(p, k, [])
        register(k, [], "raw", value, winner, f"bon{k}")
    schedules = enumerate_candidates(args)
    by_k = defaultdict(list)
    for (k, frozen_nodes), nodes in tqdm(schedules.items(), desc="Raw Personal schedule replay", unit="policy"):
        value, winner, _ = evaluate(p, k, nodes)
        row = register(k, nodes, "raw", value, winner)
        by_k[k].append(row)
    rules = ["mean_3", "mean_5", "rank_mean_3", "rank_mean_5", "rank_mean_7", "rank_recent_5",
             "rank_trend_5", "slope_3", "slope_5", "lcb_3", "ucb_3", "rescue_3", "rescue_5"]
    shortlist = {schedule_key(8, [(7, 4), (14, 2)]):[(7, 4), (14, 2)]}
    for k, values in by_k.items():
        for row in sorted(values, key=lambda row:-row["personal"])[:args.shortlist_per_k]:
            nodes = json.loads(row["nodes"])
            if nodes:
                shortlist[schedule_key(k, nodes)] = nodes
    for (k, _), nodes in tqdm(shortlist.items(), desc="Causal trajectory-rule replay", unit="schedule"):
        for rule in rules:
            value, winner, _ = evaluate(p, k, nodes, rule)
            register(k, nodes, rule, value, winner)
    for threshold in (0, 0.05, 0.1, 0.2, 0.4, 0.8):
        value, winner, times = adaptive_evaluate(p, threshold)
        register(8, [], f"adaptive_gap_{threshold}", value, winner,
                 label=f"adaptive8_gap{threshold}", cost=112)
        write_csv(args.output_dir / f"adaptive_gap{threshold}_times.csv",
                  [{"task_index":i, "first_prune":int(t), "second_prune":28 - 2 * int(t)} for i, t in enumerate(times)])
    # A diagnostic upper bound over the 200 tasks is not a deployable policy.
    # The same global policy MUST apply to each task in reported main rows.
    ranked = sorted(results, key=lambda row:-row["personal"])
    write_csv(args.output_dir / "policy_summary.csv", ranked)
    fixed_default_rows = [row for row in ranked if row["initial_particles"] == 8 and row["pool_offset"] == 0
                          and json.loads(row["nodes"]) == [[7, 4], [14, 2]]]
    write_csv(args.output_dir / "default_schedule_rule_comparison.csv", fixed_default_rows)
    fair = [row for row in ranked if row["particle_steps"] <= 112 and row["pool_offset"] == 0]
    best_by_k = [next(row for row in fair if row["initial_particles"] == k) for k in sorted({row["initial_particles"] for row in fair})]
    best_by_rule = [next(row for row in fair if row["rule"] == rule) for rule in sorted({row["rule"] for row in fair})]
    write_csv(args.output_dir / "best_by_initial_particles.csv", best_by_k)
    write_csv(args.output_dir / "best_by_rule.csv", best_by_rule)
    # User-group outer folds choose a GLOBAL strategy on the other users.
    # This is hyperparameter validation, not predictor fitting.
    values_array = np.stack(all_values)
    winner_array = np.stack(all_winners)
    fair_ids = np.asarray([row["policy_id"] for row in fair])
    users = np.random.default_rng(2026).permutation(sorted(set(uids.tolist())))
    folds = np.array_split(users, 5)
    cv_rows, cv_decisions = [], []
    for fold, heldout in enumerate(folds):
        test = np.isin(uids, heldout)
        training_means = values_array[fair_ids][:, ~test].mean(1)
        index = int(fair_ids[int(np.argmax(training_means))])
        for task in np.flatnonzero(test):
            cv_decisions.append({"task_index":int(task), "uid":int(uids[task]), "fold":fold,
                                 "policy_id":index, "policy_name":results[index]["name"],
                                 "seed":int(2026 + task * 10000 + winner_array[index, task]),
                                 "personal":float(values_array[index, task]), "default_personal":float(baseline_values[task])})
        cv_rows.append({"fold":fold, "n_heldout_users":len(heldout), "n_heldout_tasks":int(test.sum()),
                        "selected_policy":results[index]["name"], "policy_id":index,
                        "heldout_personal":float(values_array[index, test].mean()),
                        "default_personal":float(baseline_values[test].mean()),
                        "delta_default":float((values_array[index, test] - baseline_values[test]).mean())})
    write_csv(args.output_dir / "group_policy_cross_validation.csv", cv_rows)
    write_csv(args.output_dir / "group_policy_cross_validation_tasks.csv", cv_decisions)
    # Paired user-cluster bootstrap of exploratory gain; accounts for multiple
    # tasks per user but does not correct selection of the best strategy.
    selected_for_checks = {row["policy_id"]:row for row in best_by_k}
    for rule in ("raw", "rank_mean_3", "rank_mean_5", "rank_recent_5", "rescue_5"):
        row = next(row for row in fair if row["rule"] == rule)
        selected_for_checks[row["policy_id"]] = row
    chosen = list(selected_for_checks.values())
    unique_users = np.unique(uids)
    rng = np.random.default_rng(2027)
    bootstrap_tasks = [np.concatenate([np.flatnonzero(uids == uid) for uid in rng.choice(unique_users, len(unique_users), replace=True)]) for _ in range(1000)]
    interval_rows = []
    for row in chosen:
        delta = values_array[row["policy_id"]] - baseline_values
        means = [delta[indices].mean() for indices in bootstrap_tasks]
        interval_rows.append({"policy_id":row["policy_id"], "name":row["name"], "mean_delta":float(delta.mean()),
                              "user_cluster_bootstrap_p025":float(np.quantile(means, 0.025)),
                              "user_cluster_bootstrap_p975":float(np.quantile(means, 0.975)),
                              "selection_bias_corrected":False})
    write_csv(args.output_dir / "exploratory_gain_intervals.csv", interval_rows)
    # Test whether a benefit depends on which slots enter a K<16 pool. All
    # policies and baseline share the same random order within each repetition.
    robustness = []
    checks = [row for row in chosen if not row["rule"].startswith("adaptive")]
    for repeat in tqdm(range(args.robustness_repeats), desc="Seed-pool permutation checks", unit="repeat"):
        permutation = np.argsort(rng.random((200, 16)), axis=1)
        permuted = np.take_along_axis(p, permutation[:, None, :], axis=2)
        base, _, _ = evaluate(permuted, 8, [(7, 4), (14, 2)])
        for row in checks:
            k, nodes, rule, _ = candidates[row["policy_id"]]
            value, _, _ = evaluate(permuted, k, nodes, rule)
            robustness.append({"repeat":repeat, "policy_id":row["policy_id"], "name":row["name"],
                               "personal":float(value.mean()), "delta_random_default8":float((value - base).mean())})
    write_csv(args.output_dir / "seed_pool_permutations.csv", robustness)
    grouped = defaultdict(list)
    for row in robustness:
        grouped[row["policy_id"]].append(row)
    write_csv(args.output_dir / "seed_pool_robustness_summary.csv", [
        {"policy_id":index, "name":rows[0]["name"], "mean_personal":float(np.mean([r["personal"] for r in rows])),
         "mean_delta_random_default8":float(np.mean([r["delta_random_default8"] for r in rows])),
         "std_across_pool_permutations":float(np.std([r["delta_random_default8"] for r in rows])),
         "positive_repetitions":sum(r["delta_random_default8"] > 0 for r in rows), "repetitions":len(rows)}
        for index, rows in grouped.items()])
    # Freeze concrete winner images for a manageable set of useful policies.
    selected_rows = []
    for row in chosen:
        index = row["policy_id"]
        for task in range(200):
            slot = int(winner_array[index, task])
            seed = 2026 + task * 10000 + slot
            selected_rows.append({"policy_id":index, "name":row["name"], "task_index":task,
                                  "uid":int(uids[task]), "oid":int(oids[task]), "seed":seed,
                                  "personal":float(values_array[index, task]), "image_path":str(paths[(task, 28, seed)])})
    write_csv(args.output_dir / "selected_images.csv", selected_rows)
    manifest = {"metrics_csv":str(args.metrics_csv.resolve()), "metrics_sha256":hashlib.sha256(args.metrics_csv.read_bytes()).hexdigest(),
                "cache_roots":[str(args.cache_root.resolve()), str(args.extra_cache_root.resolve())],
                "tasks":200, "users":len(unique_users), "seeds_per_task":16, "steps":28,
                "seed_pool":"Nested first K slots; original8=slots0..7, extra8=slots8..15",
                "personal_only":True, "uses_gt":False, "fits_predictor":False,
                "budget":112, "raw_schedule_count":len(schedules), "evaluated_policies":len(results),
                "historical_grids":"All exact-112 schedules from the first grid; top additional_grid_limit rows from subsequent ranked grids",
                "additional_grid_limit":args.additional_grid_limit,
                "trajectory_rule_shortlist":"Default8 plus top raw schedules per K; not exhaustive over all rule/schedule combinations",
                "selection_warning":"All200 are exploratory tuning data. Group CV assesses global policy selection; permutation/bootstrap are not an independent final test.",
                "cost_warning":"Particle steps count denoising only. Prefix-based rules require extra decoding/reward evaluations online.",
                "cv_personal":float(np.mean([r["personal"] for r in cv_decisions])),
                "cv_default_personal":float(baseline_values.mean()),
                "default8_personal":float(baseline_values.mean()), "oracle16_personal":float(oracle16.mean()),
                "best_fair":fair[0]}
    write_json(args.output_dir / "analysis_manifest.json", manifest)
    annotate_reward_costs(args.output_dir)
    pruning_diagnostics(p, args.output_dir)
    practical_policy_checks(p, uids, oids, paths, args.output_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
