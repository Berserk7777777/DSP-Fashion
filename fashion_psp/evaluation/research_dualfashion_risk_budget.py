"""CPU replay of training-free, exact-budget multi-fidelity seed allocation.

Only Personal labels enter calibration/planning. Empirical transition tables are
planning surrogates, not exact Markov models of adaptive survivor populations.
The simultaneous error envelopes and their coverage are reported separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fashion_psp.evaluation.analyze_dualfashion_training_free_psp import (
    evaluate, load_data, write_csv, write_json,
)

HORIZON = 28
BUDGET = 112


def group_max(values, users):
    return np.stack([values[users == user].max(0) for user in np.unique(users)])


def conformal_quantile(values, alpha):
    index = int(np.ceil((len(values) + 1) * (1 - alpha)))
    if index > len(values):
        return float("inf")
    return float(np.sort(values)[index - 1])


def calibrate(plan, calibration, calibration_users, alpha=0.1):
    residual = plan[:, 27:28, :] - plan
    center = residual.mean((0, 2))
    pair_scale = np.maximum(np.ptp(residual, axis=2).mean(0), 1e-6)
    absolute_scale = np.maximum(np.abs(residual - center[None, :, None]).max(2).mean(0), 1e-6)
    cal_residual = calibration[:, 27:28, :] - calibration
    pair_scores = (np.ptp(cal_residual, axis=2) / pair_scale).max(1)
    absolute_scores = (np.abs(cal_residual - center[None, :, None]) / absolute_scale[None, :, None]).max((1, 2))
    # One user-block nonconformity score covers BOTH envelope families, all
    # steps and every seed/pair. Choosing between the policy families therefore
    # does not silently require an additional multiple-comparison correction.
    joint_factor = conformal_quantile(group_max(np.maximum(pair_scores, absolute_scores), calibration_users), alpha)
    pair_factor = absolute_factor = joint_factor
    if not np.isfinite([pair_factor, absolute_factor]).all():
        raise ValueError("Too few independent calibration user blocks for the requested alpha")
    pair_q = pair_scale * pair_factor
    absolute_q = absolute_scale * absolute_factor
    pair_q[27] = absolute_q[27] = center[27] = 0
    return {"center":center, "pair_q":pair_q, "absolute_q":absolute_q,
            "pair_factor":pair_factor, "absolute_factor":absolute_factor,
            "calibration_users":len(np.unique(calibration_users)), "alpha":alpha}


def coverage(p, users, calibration):
    residual = p[:, 27:28, :] - p
    pair = (np.ptp(residual, axis=2) <= calibration["pair_q"] + 1e-7).all(1)
    absolute = (np.abs(residual - calibration["center"][None, :, None])
                <= calibration["absolute_q"][None, :, None] + 1e-7).all((1, 2))
    return {"pair_task_coverage":float(pair.mean()), "absolute_task_coverage":float(absolute.mean()),
            "pair_user_coverage":float(np.mean([pair[users == u].all() for u in np.unique(users)])),
            "absolute_user_coverage":float(np.mean([absolute[users == u].all() for u in np.unique(users)]))}


def cohort_tables(p, pair_q, repeats=8):
    """Average over random cohorts; this does not assume they are actual survivors."""
    nmax = p.shape[2]
    empirical = np.zeros((29, nmax + 1, nmax + 1))
    interval = np.zeros_like(empirical)
    rng = np.random.default_rng(731)
    for _ in range(repeats):
        permutation = np.argsort(rng.random((len(p), nmax)), axis=1)
        shuffled = np.take_along_axis(p, permutation[:, None, :], axis=2)
        terminal = shuffled[:, 27, :]
        for n in range(2, nmax + 1):
            current = shuffled[:, :, :n]
            order = np.argsort(-current, axis=2, kind="stable")
            final_by_current_rank = np.take_along_axis(np.broadcast_to(terminal[:, None, :n], current.shape), order, axis=2)
            retained_best = np.maximum.accumulate(final_by_current_rank, axis=2)
            best_full = terminal[:, :n].max(1)
            sorted_current = np.take_along_axis(current, order, axis=2)
            for m in range(1, n):
                empirical[1:, n, m] += (best_full[:, None] - retained_best[:, :, m - 1]).mean(0) / repeats
                gap = sorted_current[:, :, 0] - sorted_current[:, :, m]
                interval[1:, n, m] += np.maximum(0, pair_q[None, :] - gap).mean(0) / repeats
    empirical[28] = interval[28] = 0
    return empirical, interval


def budget_dp(cut_cost, nmax, budget=BUDGET):
    """Exact minimization of the additive planning surrogate, not true regret.

    State: current step t, alive count n, remaining particle-step budget b.
    Action: keep m <= n, then advance all m by one step. Terminal b must be 0.
    Every monotone integer-count schedule is represented, without a cut-count cap.
    """
    value = np.full((29, nmax + 1, budget + 1), np.inf)
    value[28, 1:, 0] = 0
    for t in range(27, 0, -1):
        for n in range(1, nmax + 1):
            for m in range(1, n + 1):
                candidate = cut_cost[t, n, m] + value[t + 1, m, :budget + 1 - m]
                value[t, n, m:] = np.minimum(value[t, n, m:], candidate)
    return value


def select_keep(current, alive, step, remaining, value, pair_q, mode, empirical=None):
    n_tasks, nmax = current.shape
    n = alive.sum(1)
    order = np.argsort(-np.where(alive, current, -np.inf), axis=1, kind="stable")
    ranked = np.take_along_axis(current, order, axis=1)
    counts = np.arange(1, nmax + 1)
    legal_b = remaining[:, None] - counts[None, :]
    legal = (counts[None, :] <= n[:, None]) & (legal_b >= 0)
    future = value[step + 1, counts[None, :], np.maximum(0, legal_b)]
    legal &= np.isfinite(future)
    # Removed maximum is the candidate ranked m+1. No removal has zero risk.
    removed = np.take_along_axis(ranked, np.minimum(counts, nmax - 1)[None, :].repeat(n_tasks, 0), axis=1)
    epsilon = np.maximum(0, pair_q - (ranked[:, :1] - removed))
    epsilon[counts[None, :] == n[:, None]] = 0
    if mode == "risk_dp":
        objective = epsilon + future
    elif mode == "empirical_dp":
        objective = empirical[step, n[:, None], counts[None, :]] + future
    elif mode == "safe_budget":
        safe = legal & (epsilon <= 1e-10)
        has_safe = safe.any(1)
        # If safe cuts fit the exact budget, discard as many dominated seeds as
        # possible. Otherwise retain the largest feasible population.
        objective = np.where(has_safe[:, None], np.where(safe, counts, np.inf), -counts)
    else:
        raise ValueError(mode)
    objective = np.where(legal, objective, np.inf)
    if not np.isfinite(objective).any(1).all():
        raise AssertionError("No exact-budget continuation")
    keep = np.argmin(objective, axis=1) + 1
    next_alive = np.zeros_like(alive)
    mask = np.arange(nmax)[None, :] < keep[:, None]
    rows = np.broadcast_to(np.arange(n_tasks)[:, None], order.shape)
    next_alive[rows[mask], order[mask]] = True
    risk = epsilon[np.arange(n_tasks), keep - 1]
    return next_alive, risk


def replay_sync(p, k, calibration, value, mode, scale=1, empirical=None):
    tasks, _, nmax = p.shape
    alive = np.broadcast_to(np.arange(nmax) < k, (tasks, nmax)).copy()
    spent = np.full(tasks, k)
    views, cuts = np.zeros(tasks, int), np.zeros(tasks, int)
    bound = np.zeros(tasks)
    traces = [[] for _ in range(tasks)]
    for step in range(1, 28):
        before = alive.copy()
        count = before.sum(1)
        alive, risk = select_keep(p[:, step - 1, :], before, step, BUDGET - spent, value,
                                  calibration["pair_q"][step - 1] * scale, mode, empirical)
        kept = alive.sum(1)
        changed = kept < count
        views += np.where(changed, count, 0) if mode == "empirical_dp" else count
        cuts += changed
        bound += risk
        for task in np.flatnonzero(changed):
            traces[task].append({"step":step, "keep":int(kept[task]),
                                 "slots":np.flatnonzero(alive[task]).tolist(),
                                 "risk_bound":float(risk[task])})
        spent += kept
    views += alive.sum(1)
    if not np.all(spent == BUDGET):
        raise AssertionError("Unequal denoising budgets")
    terminal = np.where(alive, p[:, 27, :], -np.inf)
    winner = np.argmax(terminal, axis=1)
    return {"winner":winner, "personal":p[np.arange(tasks), 27, winner], "spent":spent,
            "reward_views":views, "cuts":cuts, "bound":bound, "trace":traces,
            "completed":alive.sum(1), "resumptions":np.zeros(tasks, int)}


def choose_resume(levels, lower, upper, budget_left, exponent):
    incomplete = levels < HORIZON
    completed = ~incomplete
    best_lower = lower.max(1)
    gain = np.maximum(0, upper - best_lower[:, None])
    distance = np.maximum(1, HORIZON - levels)
    priority = np.where(incomplete, gain / distance ** exponent, -np.inf)
    chosen = np.argmax(priority, axis=1)
    needs_completion = ~completed.any(1)
    shortest = np.where(incomplete, distance, HORIZON + 1).argmin(1)
    forced = needs_completion & (budget_left <= distance[np.arange(len(levels)), shortest])
    return np.where(forced, shortest, chosen)


def replay_resume(p, k, calibration, scale=1, exponent=1):
    tasks = len(p)
    levels = np.ones((tasks, k), int)
    current = p[:, 0, :k].copy()
    q = calibration["absolute_q"] * scale
    lower = current + calibration["center"][0] - q[0]
    upper = current + calibration["center"][0] + q[0]
    spent = k
    previous = np.full(tasks, -1)
    resumptions = np.zeros(tasks, int)
    traces = [[] for _ in range(tasks)]
    rows = np.arange(tasks)
    while spent < BUDGET:
        chosen = choose_resume(levels, lower, upper, BUDGET - spent, exponent)
        resumptions += (previous != chosen) & (levels[rows, chosen] > 1)
        levels[rows, chosen] += 1
        t = levels[rows, chosen] - 1
        observation = p[rows, t, chosen]
        current[rows, chosen] = observation
        # Use the current calibrated interval. A globally valid envelope need
        # not be nested; we do not disguise empty intersections as valid bounds.
        lower[rows, chosen] = observation + calibration["center"][t] - q[t]
        upper[rows, chosen] = observation + calibration["center"][t] + q[t]
        for task in rows:
            traces[task].append(int(chosen[task]))
        previous = chosen
        spent += 1
    completed = levels == HORIZON
    if not completed.any(1).all() or not np.all(levels.sum(1) == BUDGET):
        raise AssertionError("Resumable allocation failed to produce an exact-budget complete result")
    winner = np.argmax(np.where(completed, current, -np.inf), axis=1)
    # Every unfinished seed still has a calibrated terminal upper bound.
    bound = np.maximum(0, upper.max(1) - current[rows, winner])
    return {"winner":winner, "personal":current[rows, winner], "spent":np.full(tasks, spent),
            "reward_views":np.full(tasks, spent), "cuts":np.zeros(tasks, int),
            "bound":bound, "trace":traces, "completed":completed.sum(1),
            "resumptions":resumptions}


def configurations(max_particles=16):
    configs = []
    for k in range(4, max_particles + 1):
        configs.append({"mode":"empirical_dp", "k":k, "scale":1.0, "exponent":0})
        for scale in (0.125, 0.25, 0.5, 1.0, 2.0):
            for mode in ("risk_dp", "safe_budget"):
                configs.append({"mode":mode, "k":k, "scale":scale, "exponent":0})
        for scale in (0.25, 0.5, 1.0, 2.0):
            for exponent in (0, 1):
                configs.append({"mode":"resume_ucb", "k":k, "scale":scale, "exponent":exponent})
    return configs


def config_name(config):
    return f"{config['mode']}_K{config['k']}_q{config['scale']:g}_eta{config['exponent']}"


def run_suite(p, plan, calibration, configs, repeats):
    nmax = p.shape[2]
    empirical, _ = cohort_tables(plan, calibration["pair_q"], repeats)
    plans = {("empirical_dp", 1.0):budget_dp(empirical, nmax)}
    for scale in sorted({c["scale"] for c in configs if c["mode"] != "resume_ucb"}):
        _, intervals = cohort_tables(plan, calibration["pair_q"] * scale, repeats)
        plans[("risk_dp", scale)] = budget_dp(intervals, nmax)
    outputs = []
    for config in configs:
        mode, k, scale = config["mode"], config["k"], config["scale"]
        if mode == "resume_ucb":
            result = replay_resume(p, k, calibration, scale, config["exponent"])
        else:
            value = plans[("empirical_dp", 1.0)] if mode == "empirical_dp" else plans[("risk_dp", scale)]
            result = replay_sync(p, k, calibration, value, mode, scale, empirical)
        outputs.append(result)
    return outputs


def seed_pool_checks(p, calibration, summary, configs, output, repeats=32):
    """Frozen-policy sensitivity on the same tasks, not an independent test."""
    selected = {max((i for i, c in enumerate(configs) if c["mode"] == mode),
                    key=lambda i:summary[i]["personal"])
                for mode in sorted({c["mode"] for c in configs})}
    for k in (7, 8, 16):
        selected.add(next(i for i, c in enumerate(configs) if c["mode"] == "empirical_dp" and c["k"] == k))
    selected = sorted(selected)
    rows = []
    rng = np.random.default_rng(42028)
    # Empirical mean-risk tables are estimated once from the planning data.
    empirical, _ = cohort_tables(p, calibration["pair_q"], 8)
    value = budget_dp(empirical, p.shape[2])
    risk_plans = {}
    for i in selected:
        c = configs[i]
        if c["mode"] == "risk_dp":
            _, intervals = cohort_tables(p, calibration["pair_q"] * c["scale"], 8)
            risk_plans[c["scale"]] = budget_dp(intervals, p.shape[2])
    for repeat in range(repeats):
        order = np.argsort(rng.random((len(p), p.shape[2])), axis=1)
        permuted = np.take_along_axis(p, order[:, None, :], axis=2)
        base, _, _ = evaluate(permuted, 8, [(7, 4), (14, 2)])
        for i in selected:
            c = configs[i]
            if c["mode"] == "resume_ucb":
                result = replay_resume(permuted, c["k"], calibration, c["scale"], c["exponent"])
            else:
                v = risk_plans.get(c["scale"], value) if c["mode"] == "risk_dp" else value
                result = replay_sync(permuted, c["k"], calibration, v, c["mode"], c["scale"], empirical)
            rows.append({"repeat":repeat, "name":config_name(c),
                         "personal":float(result["personal"].mean()),
                         "delta_random_default8":float((result["personal"] - base).mean())})
    write_csv(output / "seed_pool_repetitions.csv", rows)
    summary_rows = []
    for i in selected:
        name = config_name(configs[i])
        delta = np.array([r["delta_random_default8"] for r in rows if r["name"] == name])
        summary_rows.append({"name":name, "original_personal":summary[i]["personal"],
                             "original_delta_default8":summary[i]["delta_default8"],
                             "mean_delta_random_default8":float(delta.mean()),
                             "std_delta_random_default8":float(delta.std()),
                             "positive_repetitions":int((delta > 0).sum()),
                             "repetitions":repeats, "rng_seed":42028})
    write_csv(output / "seed_pool_summary.csv", summary_rows)


def summarize(config, result, p, base, mask=None):
    if mask is None:
        mask = np.ones(len(p), bool)
    r = result["personal"][mask]
    oracle = p[mask, 27, :config["k"]].max(1)
    return {"name":config_name(config), **config, "tasks":int(mask.sum()),
            "personal":float(r.mean()), "delta_default8":float((r - base[mask]).mean()),
            "own_pool_regret":float((oracle - r).mean()),
            "regret_to16":float((p[mask, 27, :].max(1) - r).mean()),
            "oracle_hit":float(np.isclose(r, oracle, atol=1e-7, rtol=0).mean()),
            "particle_steps_min":int(result["spent"][mask].min()),
            "particle_steps_max":int(result["spent"][mask].max()),
            "mean_reward_views":float(result["reward_views"][mask].mean()),
            "mean_cuts":float(result["cuts"][mask].mean()),
            "mean_completed_seeds":float(result["completed"][mask].mean()),
            "mean_resumptions":float(result["resumptions"][mask].mean()),
            "mean_reported_bound":float(result["bound"][mask].mean()),
            "empirical_bound_coverage":float(((oracle - r) <= result["bound"][mask] + 1e-6).mean()),
            "unscaled_envelope":config["scale"] >= 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output = ROOT / "DualFashion-main/output"
    parser.add_argument("--metrics-csv", type=Path, default=output / "ifashion_fitb_timestep200_16particle_analysis/candidate_metrics_personal_all_steps.csv")
    parser.add_argument("--cache-root", type=Path, default=output / "ifashion_fitb_timestep200_cache")
    parser.add_argument("--extra-cache-root", type=Path, default=output / "ifashion_fitb_timestep200_cache_extra8")
    parser.add_argument("--output-dir", type=Path, default=output / "ifashion_fitb_risk_budget_research")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cohort-repeats", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    p, users, oids, paths = load_data(args)
    configs = configurations(p.shape[2])
    base, _, _ = evaluate(p, 8, [(7, 4), (14, 2)])
    all_cal = calibrate(p, p, users, args.alpha)
    print(f"All-task exploratory replay: {len(configs)} configurations", flush=True)
    results = run_suite(p, p, all_cal, configs, args.cohort_repeats)
    summary = [summarize(c, r, p, base) for c, r in zip(configs, results)]
    write_csv(args.output_dir / "exploratory_policy_summary.csv", sorted(summary, key=lambda r:-r["personal"]))
    bounds = [{"step":t + 1, "pairwise_error_bound":float(all_cal["pair_q"][t]),
               "absolute_error_bound":float(all_cal["absolute_q"][t]),
               "terminal_residual_center":float(all_cal["center"][t])} for t in range(28)]
    write_csv(args.output_dir / "exploratory_envelopes.csv", bounds)
    by_mode = [max((i for i, c in enumerate(configs) if c["mode"] == mode), key=lambda i:summary[i]["personal"])
               for mode in sorted({c["mode"] for c in configs})]
    best = int(np.argmax([r["personal"] for r in summary]))
    selected = sorted(set(by_mode + [best]))
    chosen_rows, trace_rows = [], []
    for index in selected:
        r, c = results[index], configs[index]
        for task in range(len(p)):
            seed = int(2026 + task * 10000 + r["winner"][task])
            chosen_rows.append({"name":config_name(c), "task_index":task, "uid":int(users[task]),
                                "oid":int(oids[task]), "seed":seed, "personal":float(r["personal"][task]),
                                "particle_steps":int(r["spent"][task]), "reward_views":int(r["reward_views"][task]),
                                "bound":float(r["bound"][task]), "image_path":str(paths[(task, 28, seed)])})
            trace_rows.append({"name":config_name(c), "task_index":task, "trace":json.dumps(r["trace"][task])})
    write_csv(args.output_dir / "selected_images.csv", chosen_rows)
    write_csv(args.output_dir / "selected_allocation_traces.csv", trace_rows)
    # Complete all-task results before group checks so an interrupted run still
    # leaves a concrete exploratory comparison.
    fold_users = np.array_split(np.random.default_rng(42027).permutation(np.unique(users)), args.folds)
    cv_tasks, cv_rows, coverage_rows, fixed_rows = [], [], [], []
    for fold in tqdm(range(args.folds), desc="User-block planning/calibration/evaluation", unit="fold"):
        test_users = fold_users[fold]
        cal_users = fold_users[(fold + 1) % args.folds]
        test_mask, cal_mask = np.isin(users, test_users), np.isin(users, cal_users)
        plan_mask = ~(test_mask | cal_mask)
        cal = calibrate(p[plan_mask], p[cal_mask], users[cal_mask], args.alpha)
        # Same statistical tables/envelopes for planning and evaluation tasks;
        # held-out terminal labels never enter a decision.
        fold_results = run_suite(p, p[plan_mask], cal, configs, args.cohort_repeats)
        train_means = np.array([r["personal"][plan_mask].mean() for r in fold_results])
        selected_index = int(np.argmax(train_means))
        certified_ids = [i for i, c in enumerate(configs) if c["mode"] != "empirical_dp" and c["scale"] >= 1]
        certified_index = max(certified_ids, key=lambda i:train_means[i])
        for group, idx in (("selected_all", selected_index), ("selected_unscaled_envelope", certified_index)):
            rr, cc = fold_results[idx], configs[idx]
            row = summarize(cc, rr, p, base, test_mask)
            cv_rows.append({"fold":fold, "selection_group":group,
                            "planning_tasks":int(plan_mask.sum()), "calibration_tasks":int(cal_mask.sum()), **row})
            for task in np.flatnonzero(test_mask):
                cv_tasks.append({"fold":fold, "selection_group":group, "task_index":int(task),
                                 "uid":int(users[task]), "name":config_name(cc),
                                 "personal":float(rr["personal"][task]), "default_personal":float(base[task]),
                                 "seed":int(2026 + task * 10000 + rr["winner"][task]),
                                 "particle_steps":int(rr["spent"][task])})
        for cc, rr in zip(configs, fold_results):
            fixed_rows.append({"fold":fold, **summarize(cc, rr, p, base, test_mask)})
        coverage_rows.append({"fold":fold, "calibration_users":cal["calibration_users"],
                              **coverage(p[test_mask], users[test_mask], cal)})
        write_csv(args.output_dir / "user_group_policy_selection.csv", cv_rows)
        write_csv(args.output_dir / "user_group_selected_tasks.csv", cv_tasks)
        write_csv(args.output_dir / "user_group_fixed_policies.csv", fixed_rows)
        write_csv(args.output_dir / "heldout_envelope_coverage.csv", coverage_rows)
    fixed_summary = []
    for c in configs:
        rows = [r for r in fixed_rows if r["name"] == config_name(c)]
        weight = np.array([r["tasks"] for r in rows], dtype=float)
        row = {"name":config_name(c), **c, "tasks":int(weight.sum())}
        for field in ("personal", "delta_default8", "mean_reward_views", "own_pool_regret", "mean_reported_bound", "empirical_bound_coverage"):
            row[field] = float(np.average([r[field] for r in rows], weights=weight))
        fixed_summary.append(row)
    write_csv(args.output_dir / "cross_fitted_fixed_policy_summary.csv", sorted(fixed_summary, key=lambda r:-r["personal"]))
    seed_pool_checks(p, all_cal, summary, configs, args.output_dir)
    manifest = {"metrics_csv":str(args.metrics_csv.resolve()),
                "metrics_sha256":hashlib.sha256(args.metrics_csv.read_bytes()).hexdigest(),
                "tasks":len(p), "users":len(np.unique(users)), "cached_particles":16,
                "particle_steps_per_task":BUDGET, "horizon":HORIZON, "configurations":len(configs),
                "initial_k_tested":list(range(4, 17)), "all_steps_eligible":list(range(1, 28)),
                "cut_count_limit":None, "theoretical_max_initial_k_one_step_then_complete_one":BUDGET - HORIZON + 1,
                "alpha":args.alpha, "folds":args.folds, "personal_only":True, "uses_gt":False,
                "trains_predictor":False, "planning_tables":"Empirical cohort averages; not exact adaptive-survivor transition models",
                "warning":"All200 were previously used in exploratory work. Cross-fitted calibration is diagnostic, not a pristine independent benchmark. Scales below1 lose nominal envelope coverage.",
                "budget_warning":"112 denoising particle-steps exactly. Decoder/reward costs and paused-latent memory are additional.",
                "default_personal":float(base.mean()), "bon4_personal":float(p[:, 27, :4].max(1).mean()),
                "best_exploratory":summary[best]}
    for group in ("selected_all", "selected_unscaled_envelope"):
        rows = [r for r in cv_tasks if r["selection_group"] == group]
        manifest[group + "_personal"] = float(np.mean([r["personal"] for r in rows]))
        manifest[group + "_delta_default"] = float(np.mean([r["personal"] - r["default_personal"] for r in rows]))
    write_json(args.output_dir / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
