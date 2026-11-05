"""State-conditioned empirical-risk DP on frozen Personal trajectories.

No generative/reward model weights are trained. Planning does use terminal
Personal labels from planning users. The small tabular state is an approximate
MDP, not a proof of Bayes-optimal pruning or a calibrated confidence interval.
Other metrics and GT are inaccessible to the policy module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from fashion_psp.evaluation.analyze_dualfashion_training_free_psp import load_data, evaluate, write_csv, write_json
from fashion_psp.evaluation.research_dualfashion_risk_budget import cohort_tables, budget_dp, replay_sync

H = 28
B = 112
Z = 4


def snapshot_features(scores, reference):
    """scores (..., n), in descending order. Only the present scores enter."""
    spread = scores.std(-1)
    top_gap = (scores[..., 0] - scores[..., min(1, scores.shape[-1] - 1)]) / np.maximum(spread, 1e-6)
    state = 2 * (top_gap > 0.4).astype(int) + (spread > reference).astype(int)
    return state, spread


def action_feature(scores, keep, spread):
    if keep == scores.shape[-1]:
        return np.zeros(scores.shape[:-1], int)
    boundary = (scores[..., keep - 1] - scores[..., keep]) / np.maximum(spread, 1e-6)
    leader = (scores[..., 0] - scores[..., keep]) / np.maximum(spread, 1e-6)
    return 2 * (boundary > 0.25).astype(int) + (leader > 1).astype(int)


def baseline_result(p, k, nodes):
    values, winners, raw_trace = evaluate(p, k, nodes, capture=True)
    trace = [[] for _ in p]
    for step, survivors in raw_trace:
        for task in range(len(p)):
            trace[task].append({"step":step, "slots":survivors[task].tolist(), "keep":survivors.shape[1]})
    views = k + sum(keep for _, keep in nodes)
    return {"personal":values, "winner":winners, "trace":trace,
            "views":np.full(len(p), views), "spent":np.full(len(p), sum([k * nodes[0][0]] +
             [nodes[i - 1][1] * (nodes[i][0] - nodes[i - 1][0]) for i in range(1, len(nodes))] +
             [nodes[-1][1] * (H - nodes[-1][0])]) if nodes else k * H)}


def actual_cohorts(p, result, k, weight):
    alive = np.broadcast_to(np.arange(p.shape[2]) < k, (len(p), p.shape[2])).copy()
    lookup = [{cut["step"]:cut for cut in trace} for trace in result["trace"]]
    for t in range(1, H):
        for n in np.unique(alive.sum(1)):
            tasks = np.flatnonzero(alive.sum(1) == n)
            slots = np.stack([np.flatnonzero(alive[i]) for i in tasks])
            yield t, tasks, slots, weight
        for task, cuts in enumerate(lookup):
            if t in cuts:
                alive[task] = False
                alive[task, cuts[t]["slots"]] = True


class StateRiskDP:
    def __init__(self, reference, core_loss, core_transition, loss, transition, support):
        self.reference, self.core_loss, self.core_transition = reference, core_loss, core_transition
        self.loss, self.transition, self.support = loss, transition, support
        self.nmax = reference.shape[1] - 1
        self.value = self._bellman()

    def _bellman(self):
        value = np.full((29, self.nmax + 1, B + 1, Z), np.inf)
        value[28, 1:, 0, :] = 0
        for t in range(27, 0, -1):
            for n in range(1, self.nmax + 1):
                for m in range(1, n + 1):
                    future = value[t + 1, m, :B + 1 - m]
                    feasible = np.isfinite(future).all(1)
                    expected = np.where(np.isfinite(future), future, 0) @ self.core_transition[t, n, m].T
                    candidate = self.core_loss[t, n, m][None, :] + expected
                    candidate[~feasible] = np.inf
                    value[t, n, m:] = np.minimum(value[t, n, m:], candidate)
        return value

    def decide(self, current, alive, t, remaining):
        """Pure current-state decision. No trajectory/terminal scores accepted."""
        tasks, nmax = current.shape
        counts = alive.sum(1)
        order = np.argsort(-np.where(alive, current, -np.inf), axis=1, kind="stable")
        ranked = np.take_along_axis(current, order, axis=1)
        action_cost = np.full((tasks, nmax), np.inf)
        estimated = np.zeros_like(action_cost)
        support = np.zeros_like(action_cost)
        for n in np.unique(counts):
            ids = np.flatnonzero(counts == n)
            scores = ranked[ids, :n]
            state, spread = snapshot_features(scores, self.reference[t, n])
            for m in range(1, n + 1):
                rem = remaining[ids] - m
                valid = rem >= 0
                if not valid.any():
                    continue
                who = ids[valid]
                z = state[valid]
                a = action_feature(scores[valid], m, spread[valid])
                future = self.value[t + 1, m, rem[valid]]
                feasible = np.isfinite(future).all(1)
                expected = (np.where(np.isfinite(future), future, 0) * self.transition[t, n, m, z, a]).sum(1)
                immediate = self.loss[t, n, m, z, a]
                action_cost[who, m - 1] = np.where(feasible, immediate + expected - m * 1e-10, np.inf)
                estimated[who, m - 1] = immediate
                support[who, m - 1] = self.support[t, n, m, z, a]
        if not np.isfinite(action_cost).any(1).all():
            raise AssertionError("No exact-budget continuation")
        keep = np.argmin(action_cost, axis=1) + 1
        new_alive = np.zeros_like(alive)
        mask = np.arange(nmax)[None, :] < keep[:, None]
        row = np.broadcast_to(np.arange(tasks)[:, None], order.shape)
        new_alive[row[mask], order[mask]] = True
        selected = estimated[np.arange(tasks), keep - 1]
        selected_support = support[np.arange(tasks), keep - 1]
        # Current scoring is necessary if a cut occurs or multiple actions are
        # feasible. A forced keep-all action needs no new reward observation.
        score_needed = (keep < counts) | (np.isfinite(action_cost).sum(1) > 1)
        return new_alive, selected, selected_support, score_needed

    def replay(self, p, k):
        alive = np.broadcast_to(np.arange(p.shape[2]) < k, (len(p), p.shape[2])).copy()
        spent, views = np.full(len(p), k), np.zeros(len(p), int)
        trace = [[] for _ in p]
        for t in range(1, H):
            before = alive.sum(1)
            alive, risk, support, observed = self.decide(p[:, t - 1], alive, t, B - spent)
            kept = alive.sum(1)
            views += before * observed
            for i in np.flatnonzero(kept < before):
                trace[i].append({"step":t, "keep":int(kept[i]), "slots":np.flatnonzero(alive[i]).tolist(),
                                 "estimated_loss":float(risk[i]), "weighted_support":float(support[i])})
            spent += kept
        views += alive.sum(1)
        if not (spent == B).all():
            raise AssertionError("Every task must consume exactly112 particle steps")
        winner = np.argmax(np.where(alive, p[:, 27], -np.inf), axis=1)
        return {"winner":winner, "personal":p[np.arange(len(p)), 27, winner],
                "trace":trace, "spent":spent, "views":views}

    def save(self, path):
        np.savez_compressed(path, reference=self.reference, core_loss=self.core_loss,
                            core_transition=self.core_transition, loss=self.loss,
                            transition=self.transition, support=self.support)

    @classmethod
    def load(cls, path):
        """Restore planning tables without accessing any candidate trajectories."""
        with np.load(path, allow_pickle=False) as tables:
            return cls(**{key: tables[key] for key in (
                "reference", "core_loss", "core_transition", "loss", "transition", "support"
            )})


def fit_once(p, feedback, repeats=4, shrinkage=12):
    nmax = p.shape[2]
    rng = np.random.default_rng(628)
    permutations = [np.argsort(rng.random((len(p), nmax)), axis=1) for _ in range(repeats)]
    reference = np.zeros((29, nmax + 1))
    for n in range(1, nmax + 1):
        spreads = [np.take_along_axis(p, order[:, None, :n], axis=2).std(2) for order in permutations]
        reference[1:, n] = np.median(np.concatenate(spreads), axis=0)
    shape = (29, nmax + 1, nmax + 1, Z, Z)
    count, total = np.zeros(shape), np.zeros(shape)
    transition = np.zeros(shape + (Z,))

    def consume(t, tasks, slots, weight):
        n = slots.shape[1]
        current = np.take_along_axis(p[tasks, t - 1], slots, axis=1)
        order = np.argsort(-current, axis=1, kind="stable")
        slots = np.take_along_axis(slots, order, axis=1)
        scores = np.take_along_axis(current, order, axis=1)
        z, spread = snapshot_features(scores, reference[t, n])
        final = np.take_along_axis(p[tasks, 27], slots, axis=1)
        full = final.max(1)
        prefix_best = np.maximum.accumulate(final, axis=1)
        next_scores = np.take_along_axis(p[tasks, t], slots, axis=1)
        for m in range(1, n + 1):
            a = action_feature(scores, m, spread)
            next_state, _ = snapshot_features(np.sort(next_scores[:, :m], axis=1)[:, ::-1], reference[t + 1, m])
            index = (t, n, m, z, a)
            np.add.at(count, index, weight)
            np.add.at(total, index, weight * (full - prefix_best[:, m - 1]))
            np.add.at(transition, index + (next_state,), weight)

    # The weights are task-equivalent counts, not independent pair counts.
    random_weight = 0.5 if not feedback else 0.25
    for order in permutations:
        for n in range(1, nmax + 1):
            for t in range(1, H):
                consume(t, np.arange(len(p)), order[:, :n], random_weight / repeats)
    base = baseline_result(p, 8, [(7, 4), (14, 2)])
    fixed_weight = 0.5 if not feedback else 0.25
    for batch in actual_cohorts(p, base, 8, fixed_weight):
        consume(*batch)
    if feedback:
        for k, result in feedback:
            for batch in actual_cohorts(p, result, k, 0.5 / len(feedback)):
                consume(*batch)
    parent_count = count.sum((3, 4))
    parent_loss = total.sum((3, 4)) / np.maximum(parent_count, 1e-12)
    parent_transition = transition.sum((3, 4)) / np.maximum(parent_count[..., None], 1e-12)
    core_count = count.sum(4)
    core_loss = (total.sum(4) + shrinkage * parent_loss[..., None]) / (core_count + shrinkage)
    core_transition = (transition.sum(4) + shrinkage * parent_transition[..., None, :]) / (core_count[..., None] + shrinkage)
    loss = (total + shrinkage * core_loss[..., None]) / (count + shrinkage)
    probabilities = (transition + shrinkage * core_transition[..., None, :]) / (count[..., None] + shrinkage)
    for n in range(1, nmax + 1):
        core_loss[:, n, n] = loss[:, n, n] = 0
    return StateRiskDP(reference, core_loss, core_transition, loss, probabilities, count)


def fit_model(p, repeats=4, shrinkage=12, rounds=1):
    model = fit_once(p, [], repeats, shrinkage)
    for _ in range(rounds):
        feedback = [(k, model.replay(p, k)) for k in range(4, p.shape[2] + 1)]
        model = fit_once(p, feedback, repeats, shrinkage)
    return model


def unconditional(p):
    costs, _ = cohort_tables(p, np.zeros(28), repeats=8)
    return costs, budget_dp(costs, p.shape[2])


def fixed_dp(p, costs, value, k=8):
    result = replay_sync(p, k, {"pair_q":np.zeros(28)}, value, "empirical_dp", empirical=costs)
    return {"winner":result["winner"], "personal":result["personal"], "trace":result["trace"],
            "spent":result["spent"], "views":result["reward_views"]}


def record_selection(rows, label, result, task_ids, users, oids, paths, fold=-1):
    for local, task in enumerate(task_ids):
        seed = int(2026 + task * 10000 + result["winner"][local])
        rows.append({"label":label, "task_index":int(task), "uid":int(users[task]), "oid":int(oids[task]),
                     "fold":fold, "seed":seed, "image_path":str(paths[(task, 28, seed)]),
                     "selection_personal":float(result["personal"][local]),
                     "particle_steps":int(result["spent"][local]), "reward_views":int(result["views"][local]),
                     "trace":json.dumps(result["trace"][local])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT / "DualFashion-main/output"
    parser.add_argument("--metrics-csv", type=Path, default=root / "ifashion_fitb_timestep200_16particle_analysis/candidate_metrics_personal_all_steps.csv")
    parser.add_argument("--cache-root", type=Path, default=root / "ifashion_fitb_timestep200_cache")
    parser.add_argument("--extra-cache-root", type=Path, default=root / "ifashion_fitb_timestep200_cache_extra8")
    parser.add_argument("--output-dir", type=Path, default=root / "ifashion_fitb_state_risk_dp")
    parser.add_argument("--cohort-repeats", type=int, default=4)
    parser.add_argument("--shrinkage", type=float, default=12)
    parser.add_argument("--feedback-rounds", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    p, users, oids, paths = load_data(args)
    model = fit_model(p, args.cohort_repeats, args.shrinkage, args.feedback_rounds)
    model.save(args.output_dir / "planning_statistics_all_tasks.npz")
    costs, value = unconditional(p)
    rows = []
    ids = np.arange(len(p))
    for k in range(4, 17):
        record_selection(rows, f"exploratory/state_dp{k}", model.replay(p, k), ids, users, oids, paths)
    record_selection(rows, "exploratory/empirical_dp8", fixed_dp(p, costs, value), ids, users, oids, paths)
    for label, k, nodes in (("default_psp8", 8, [(7, 4), (14, 2)]), ("bon4", 4, []), ("standard", 1, []), ("oracle_bon8", 8, [])):
        record_selection(rows, label, baseline_result(p, k, nodes), ids, users, oids, paths)
    folds = np.array_split(np.random.default_rng(42027).permutation(np.unique(users)), args.folds)
    choices = []
    for fold, test_users in enumerate(tqdm(folds, desc="State-risk DP user-group replay", unit="fold")):
        test_mask = np.isin(users, test_users)
        train_mask = ~test_mask
        train, test = p[train_mask], p[test_mask]
        trained = fit_model(train, args.cohort_repeats, args.shrinkage, args.feedback_rounds)
        trained.save(args.output_dir / f"planning_statistics_fold{fold}.npz")
        train_values = [trained.replay(train, k)["personal"].mean() for k in range(4, 17)]
        best_k = int(np.argmax(train_values)) + 4
        for k in range(4, 17):
            result = trained.replay(test, k)
            record_selection(rows, f"crossfit/state_dp{k}", result, ids[test_mask], users, oids, paths, fold)
            if k == best_k:
                record_selection(rows, "crossfit/state_dp_selected_k", result, ids[test_mask], users, oids, paths, fold)
        cc, vv = unconditional(train)
        record_selection(rows, "crossfit/empirical_dp8", fixed_dp(test, cc, vv), ids[test_mask], users, oids, paths, fold)
        choices.append({"fold":fold, "planning_users":int(len(np.unique(users[train_mask]))),
                        "evaluation_users":len(test_users), "planning_tasks":len(train),
                        "evaluation_tasks":len(test), "selected_k":best_k})
    write_csv(args.output_dir / "frozen_selections.csv", rows)
    write_csv(args.output_dir / "user_group_choices.csv", choices)
    summary = []
    for label in sorted({r["label"] for r in rows}):
        selected = [r for r in rows if r["label"] == label]
        if len(selected) != 200:
            raise AssertionError("Incomplete selection group")
        values = np.array([r["selection_personal"] for r in selected])
        schedules = {tuple((v["step"], v["keep"]) for v in json.loads(r["trace"])) for r in selected}
        summary.append({"label":label, "tasks":len(selected), "selection_personal":float(values.mean()),
                        "mean_particle_steps":float(np.mean([r["particle_steps"] for r in selected])),
                        "mean_reward_views":float(np.mean([r["reward_views"] for r in selected])),
                        "distinct_schedules":len(schedules)})
    write_csv(args.output_dir / "selection_summary.csv", summary)
    manifest = {"tasks":len(p), "users":len(np.unique(users)), "cached_particles":16,
                "particle_steps":112, "state_features":["normalized_top2_gap", "current_spread_vs_planning_median"],
                "action_features":["normalized_boundary_gap", "normalized_leader_to_first_removed_gap"],
                "core_states":4, "action_states":4, "shrinkage_task_equivalents":args.shrinkage,
                "cohort_repeats":args.cohort_repeats, "feedback_rounds":args.feedback_rounds,
                "replay_cohorts":"default8 plus actual planning-policy survivors; random subsets provide support for unvisited states",
                "training":"No model weight fitting; terminal Personal from planning users estimates statistical tables",
                "gt_in_selection":False, "other_metrics_in_selection":False,
                "evaluation_warning":"All200 are exploratory data used previously. Crossfit excludes current evaluation users from tables but is not a fresh independent benchmark.",
                "optimality_warning":"Bellman planning uses a coarse empirical MDP; no exact true-regret optimality or confidence guarantee",
                "metrics_sha256":hashlib.sha256(args.metrics_csv.read_bytes()).hexdigest(),
                "selections_sha256":hashlib.sha256((args.output_dir / "frozen_selections.csv").read_bytes()).hexdigest()}
    write_json(args.output_dir / "selection_protocol.json", manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
