"""Explicitly tuned DEVELOPMENT-set adaptive PSP search, original8, exact112.

No per-task fitted parameters, terminal lookup in policies, or old DP tables.
Terminal labels rank global policies only AFTER causal replay. No test claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import (
    CausalObserver, load_and_audit, posthoc, replay as replay_default,
    sha256, summarize, write_csv, write_json,
)


@dataclass(frozen=True)
class Policy:
    min_step: int = 7
    window: int = 3
    z: float = 2.0
    margin: float = 0.0
    protect_top: int = 2
    max_drop: int = 8
    trend_horizon: float = 0.0
    mean_weight: float = 0.0


def feasible_counts(step, spent, n):
    """Exact monotone integer continuation; independent of all reward data.

For h steps with next population m, achievable costs are every integer in
    [m+h-1, h*m]. Confirmed against recursive reachable_costs in tests.
"""
    h, remaining = 28 - step, 112 - spent
    lo = max(1, (remaining + h - 1) // h)
    hi = min(n, remaining - h + 1)
    if lo > hi:
        raise AssertionError("No exact112 continuation")
    return list(range(lo, hi + 1))


def decide_scalar(policy, history, step, spent):
    """Only histories of surviving seeds reach this policy function."""
    seeds = sorted(history)
    current = {s: history[s][-1] for s in seeds}
    leader = min(seeds, key=lambda s: (-current[s], s))
    width = min(policy.window, step)
    x = [i - (width - 1) / 2 for i in range(width)]
    denom = sum(v * v for v in x)
    proxy = {}
    for s in seeds:
        values = history[s][-width:]
        slope = sum(a * b for a, b in zip(x, values)) / denom if denom else 0.0
        proxy[s] = ((1 - policy.mean_weight) * current[s]
                    + policy.mean_weight * statistics.mean(values)
                    + policy.trend_horizon * slope)
    ranked = sorted(seeds, key=lambda s: (-proxy[s], s))
    protected = set(ranked[:policy.protect_top])
    eligible, evidence = [], {}
    if step >= max(policy.min_step, policy.window):
        for s in seeds:
            gaps = [a - b for a, b in zip(history[leader][-width:], history[s][-width:])]
            mean, std = statistics.mean(gaps), statistics.stdev(gaps)
            stable = all(g > 0 for g in gaps) and mean > policy.z * std + policy.margin
            marked = stable and s not in protected
            evidence[str(s)] = {"gaps": gaps, "mean": mean, "sample_std": std,
                                "stable": stable, "protected": s in protected, "eligible": marked}
            if marked:
                eligible.append(s)
    feasible = feasible_counts(step, spent, len(seeds))
    desired = len(seeds) - min(len(eligible), policy.max_drop)
    keep = max(min(feasible), min(desired, max(feasible)))
    worst = list(reversed(ranked))
    priority = [s for s in worst if s in eligible] + [s for s in worst if s not in eligible]
    removed = priority[:len(seeds) - keep]
    # The heuristic proposes at most max_drop. Any additional removals are forced.
    voluntary = set([s for s in priority if s in eligible][:min(len(removed), policy.max_drop)])
    reasons = {str(s): "stability_heuristic" if s in voluntary else "budget_forced" for s in removed}
    return sorted(set(seeds) - set(removed)), reasons, {
        "leader": leader, "ranking_scores": proxy, "protected_seeds": sorted(protected),
        "eligible_seeds": sorted(eligible), "feasible_keep_counts": feasible,
        "desired_keep_count": desired, "stability_evidence": evidence,
    }


def replay_scalar(policy, seeds, observe):
    active, history, spent, trace = sorted(seeds), {}, 0, []
    for step in range(1, 29):
        before = active.copy()
        spent += len(before)
        scores = observe(step, before, True)
        if set(scores) != set(before) or not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("Expected finite live scores")
        for s in before:
            history.setdefault(s, []).append(scores[s])
        if step == 28:
            assert spent == 112
            active = [min(before, key=lambda s: (-scores[s], s))]
            reasons = {str(s): "terminal_selection" for s in before if s not in active}
            detail = {}
        else:
            active, reasons, detail = decide_scalar(policy, history, step, spent)
        history = {s: history[s] for s in active}
        trace.append({"step": step, "alive_before": before, "alive_after": active.copy(),
                      "count_before": len(before), "count_after": len(active),
                      "removed_reasons": reasons, "personal_scores": scores, "observed": True,
                      "cumulative_particle_steps": spent, "remaining_particle_steps": 112 - spent,
                      "cumulative_decode_images": spent, "cumulative_personal_evaluations": spent,
                      "cumulative_scoring_rounds": step, **detail})
    return {"selected_seed": active[0], "particle_steps": spent, "decode_images": spent,
            "personal_evaluations": spent, "scoring_rounds": 28, "trace": trace}


def replay_batch(data, policy, return_masks=False):
    """Fast environment+policy for search. Read only reached live scores.

The final winner is additionally independently verified by the callback replay.
No post-prune values are copied to policy history, including historical windows.
"""
    n_tasks = len(data)
    rows = np.arange(n_tasks)
    active = np.ones((n_tasks, 8), bool)
    history = np.full_like(data, np.nan)
    spent = np.zeros(n_tasks, int)
    masks = []
    for t in range(28):
        step = t + 1
        n = active.sum(1)
        spent += n
        rr, ss = np.where(active)
        history[rr, t, ss] = data[rr, t, ss]
        current = np.where(active, history[:, t, :], -np.inf)
        if return_masks:
            masks.append(active.copy())
        if step == 28:
            assert (spent == 112).all()
            return np.argmax(current, axis=1), masks
        width = min(step, policy.window)
        x = np.arange(width, dtype=float) - (width - 1) / 2
        # Values of pruned candidates are excluded from ALL policy operations.
        values = np.where(active[:, None, :], history[:, step - width:step, :], 0.0)
        slope = (values * x[None, :, None]).sum(1) / max(float(x @ x), 1.0)
        proxy = np.where(active, (1 - policy.mean_weight) * values[:, -1, :]
                         + policy.mean_weight * values.mean(1) + policy.trend_horizon * slope, -np.inf)
        order = np.argsort(-proxy, axis=1, kind="stable")
        ranks = np.argsort(order, axis=1, kind="stable")
        eligible = np.zeros_like(active)
        if step >= max(policy.min_step, policy.window):
            leader = np.argmax(current, axis=1)
            gaps = values[rows, :, leader][:, :, None] - values
            eligible = (active & (ranks >= policy.protect_top) & (gaps > 0).all(1)
                        & (gaps.mean(1) > policy.z * gaps.std(1, ddof=1) + policy.margin))
        desired = n - np.minimum(eligible.sum(1), policy.max_drop)
        h, remaining = 28 - step, 112 - spent
        lo = np.maximum(1, (remaining + h - 1) // h)
        hi = np.minimum(n, remaining - h + 1)
        assert (lo <= hi).all()
        keep = np.maximum(lo, np.minimum(desired, hi))
        # Low integer priority = eligible first, worst rank first; dead last.
        priority = np.where(active, np.where(eligible, 0, 8) + 7 - ranks, 100)
        removal_order = np.argsort(priority, axis=1, kind="stable")
        removal_ranks = np.argsort(removal_order, axis=1, kind="stable")
        active &= removal_ranks >= (n - keep)[:, None]
    raise AssertionError("Missing terminal")


def policy_candidates():
    """Finite, logged global search; no task identity branches or fitted table."""
    seen = set()
    # First isolate conservative current-score policies, then widen the family.
    for first, z, margin, protect, cap in itertools.product(
            [5, 6, 7, 8, 9, 10], [1.0, 2.0, 3.0, 4.0], [0.0, 2.0, 4.0], [1, 2, 3], [1, 2, 8]):
        p = Policy(min_step=first, z=z, margin=margin, protect_top=protect, max_drop=cap)
        seen.add(p)
        yield "conservative_grid", p
    rng = random.Random(20261024)
    for _ in range(5000):
        p = Policy(min_step=rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 11]),
                   window=rng.choice([3, 4, 5, 6, 7]), z=rng.choice([0, 0.5, 1, 1.5, 2, 3, 4, 6]),
                   margin=rng.choice([0, 0.5, 1, 2, 3, 4, 6]), protect_top=rng.choice([1, 2, 3, 4]),
                   max_drop=rng.choice([1, 2, 3, 8]), trend_horizon=rng.choice([0, 0, 0.5, 1, 2, 3, 4]),
                   mean_weight=rng.choice([0, 0, 0.25, 0.5, 0.75, 1]))
        if p not in seen:
            seen.add(p)
            yield "seeded_prefix_search", p


def export_final(output, tasks, policy, fast_winners, summary_search, input_path):
    selections, traces, validations = [], [], []
    for task, entry in sorted(tasks.items()):
        for method in ("default", "adaptive"):
            observer = CausalObserver(entry["personal"], entry["seeds"])
            result = (replay_default(entry["seeds"], "default", observer) if method == "default"
                      else replay_scalar(policy, entry["seeds"], observer))
            if method == "adaptive":
                assert result["selected_seed"] == entry["seeds"][int(fast_winners[task])]
                seen = set(observer.accesses)
                # Change every unobserved trajectory value to a huge positive value.
                poisoned = {k: v if k in seen else 1e90 for k, v in entry["personal"].items()}
                rerun = replay_scalar(policy, entry["seeds"], CausalObserver(poisoned, entry["seeds"]))
                assert result == rerun
                validations.append({"task_index": task, "batch_scalar_same_seed": True,
                                    "unobserved_future_poisoning_invariance": True,
                                    "exact112": result["particle_steps"] == 112})
            assert len(observer.accesses) == result["personal_evaluations"]
            traces.extend({"task_index": task, "method": method, **r} for r in result.pop("trace"))
            selections.append({"task_index": task, "uid": entry["identity"][0],
                               "oid": entry["identity"][1], "method": method, **result})
    write_csv(output / "selection_pre_posthoc.csv", selections)
    with (output / "step_trace.jsonl").open("w", encoding="utf-8") as f:
        for r in traces:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    events = [{"event_type": "terminal_selection" if r["step"] == 28 else "pruning", **r}
              for r in traces if r["count_after"] < r["count_before"]]
    write_csv(output / "pruning_events.csv", events)
    evaluated = posthoc(input_path, tasks, selections)
    summary, pairs = summarize(evaluated, traces)
    summary["dataset_role"] = "adaptively_tuned_on_all200_development_tasks_NOT_independent_test"
    summary["search"] = summary_search
    schedules = Counter()
    for task in tasks:
        path = tuple((r["step"], r["count_after"]) for r in traces if r["task_index"] == task
                     and r["method"] == "adaptive" and r["step"] < 28 and r["count_after"] < r["count_before"])
        schedules[path] += 1
    summary["adaptive_unique_schedules"] = len(schedules)
    a, b = summary["methods"]["default"], summary["methods"]["adaptive"]
    summary["both_objectives_strictly_improved"] = (b["mean_personal"] > a["mean_personal"]
                                                   and b["mean_optimal_seed_hit"] > a["mean_optimal_seed_hit"])
    assert summary["both_objectives_strictly_improved"]
    assert len(schedules) > 1
    write_json(output / "summary.json", summary)
    write_csv(output / "selections.csv", evaluated)
    write_csv(output / "task_comparison.csv", pairs)
    write_csv(output / "validation_results.csv", validations)
    write_csv(output / "budget_checks.csv", [{k: r[k] for k in ("task_index", "method", "particle_steps",
              "decode_images", "personal_evaluations", "scoring_rounds")} for r in evaluated])
    write_csv(output / "adaptive_schedule_distribution.csv",
              [{"schedule": list(p), "tasks": c} for p, c in schedules.most_common()])
    write_report(output, summary, policy)
    return summary


def write_report(output, summary, policy):
    a, b = summary["methods"]["default"], summary["methods"]["adaptive"]
    comparison = summary["adaptive_vs_default"]
    lines = ["# 全200任务调参后的自适应Personal PSP", "",
             "本次按用户授权，在全部200个开发任务上搜索参数并选取方案，属于开发集拟合结果，不能宣称独立测试或泛化提升。旧版冻结实验保持不变。", "",
             "原始8seed×28步×200任务，来自ifashion_fitb_timestep200_cache及其既有Personal评分CSV。CSV与每任务metadata核对，未混入额外8seed。", "",
             "|方法|平均Personal|平均Compatibility|平均regret|最优seed命中率|",
             "|---|---:|---:|---:|---:|"]
    for name, v in [("默认PSP@8", a), ("调参后自适应PSP@8", b)]:
        lines.append(f"|{name}|{v['mean_personal']:.9f}|{v['mean_compatibility']:.9f}|{v['mean_personal_regret']:.9f}|{v['mean_optimal_seed_hit']:.1%}|")
    lines += ["", f"Personal差值：{b['mean_personal']-a['mean_personal']:+.9f}；最优seed命中率差值：{100*(b['mean_optimal_seed_hit']-a['mean_optimal_seed_hit']):+.1f}个百分点。",
              f"胜/平/负：{comparison['wins']}/{comparison['ties']}/{comparison['losses']}。两个主指标均严格高于默认：{summary['both_objectives_strictly_improved']}。", "",
              f"完整评估全局候选策略数：{summary['search']['policies_evaluated']}。固定随机搜索seed=20261024，按“两个指标均超过默认，然后命中数优先、Personal其次”选取。全部尝试保存在search_trials.csv，没有隐藏的任务专属规则。", "",
              "冻结的最终参数：", "```json", json.dumps(asdict(policy), ensure_ascii=False, indent=2), "```", "",
              "每步评分；当前Personal最高者为窗口分差参照。窗口内每步严格落后且分差均值>z×样本标准差+margin、并且不受前protect_top名保护，才标记可剪。排序分数=(1-mean_weight)×当前Personal+mean_weight×窗口均值+trend_horizon×窗口最小二乘斜率。达到min_step且窗口完整才允许主动剪枝，每步最多主动剪max_drop个。纯预算可达性约束数量，必要时强制剪枝；同分保留较小seed。终步按真实Personal选1个。稳定性与趋势项均是启发式，不是统计置信区间。", "",
              "候选都从step1推进，去噪步数相同；剪枝位置与数量由各任务已观察轨迹决定。", "",
              f"自适应产生{summary['adaptive_unique_schedules']}种非终步剪枝路径，分布见adaptive_schedule_distribution.csv。", "",
              "两方法每任务均严格112粒子步，各200任务合计22400粒子步。", "",
              "|每任务开销|默认|自适应|", "|---|---:|---:|",
              "|解码图片数|14|112|", "|Personal图片评估次数|14|112|", "|评分轮数|3|28|", "",
              "解码与reward计数是在线实施所需粒子图片数，不是GPU批次。自适应图片评估量为默认8倍，未声称等端到端成本。本次执行为缓存回放，实际神经模型调用为0。", "",
              "因果边界：策略仅接收存活候选已到达step的Personal历史；未访问GT、Compatibility、CIS、LPIPS或旧DP表。终值只用于每个策略完整回放后的开发集参数选择，以及事后oracle/regret评估。全局参数已由这些任务的结果调优，因此只有回放决策因果，不能把选参过程称为未见过终值。", "",
              "最终方案经过独立逐任务观察回调复核，200任务快速实现与标量实现所选seed一致。把所有未观察未来值替换为极大值后，200任务完整轨迹均不变。", "",
              "产物：selected_policy.json为最终参数；task_comparison.csv为逐任务seed与图片；selections.csv为完整指标；pruning_events.csv记录每次实际剪枝及原因/累计预算；step_trace.jsonl为全步证据；budget_checks.csv和validation_results.csv为预算/因果验证。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=ROOT / "DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "DualFashion-main/output/ifashion_fitb_timestep200_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fashion_psp/artifacts/adaptive_personal_development_search_v2")
    parser.add_argument("--policy", type=Path, help="Replay a saved selected_policy.json without searching")
    parser.add_argument("--limit", type=int, default=0, help="0 runs the complete predefined search")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    tasks, audit = load_and_audit(args.metrics_csv, args.cache_root, 200)
    write_csv(output / "candidate_pool_audit.csv", audit)
    data = np.array([[[e["personal"][(step, s)] for s in e["seeds"]] for step in range(1, 29)]
                     for _, e in sorted(tasks.items())])
    terminal = data[:, -1, :].copy()
    rows = np.arange(200)
    oracle = terminal.max(1)
    baseline_winners = np.array([tasks[t]["seeds"].index(replay_default(e["seeds"], "default", CausalObserver(e["personal"], e["seeds"]))["selected_seed"])
                                 for t, e in sorted(tasks.items())])
    baseline_values = terminal[rows, baseline_winners]
    base_mean, base_hits = float(baseline_values.mean()), int((baseline_values == oracle).sum())
    print(f"Original8 audited. Baseline Personal={base_mean:.9f}, hits={base_hits}/200", flush=True)
    if args.policy:
        saved = json.loads(args.policy.read_text(encoding="utf-8"))
        chosen = Policy(**saved["parameters"])
        search_info = saved["search"]
    else:
        trials, best = [], None
        search_file = output / "search_trials.jsonl"
        with search_file.open("w", encoding="utf-8") as log:
            for index, (phase, policy) in enumerate(policy_candidates(), 1):
                if args.limit and index > args.limit:
                    break
                winners, _ = replay_batch(data, policy)
                values = terminal[rows, winners]
                mean, hits = float(values.mean()), int((values == oracle).sum())
                qualifies = mean > base_mean and hits > base_hits
                trial = {"trial": index, "phase": phase, **asdict(policy), "mean_personal": mean,
                         "oracle_hits": hits, "hit_rate": hits / 200, "mean_regret": float((oracle-values).mean()),
                         "both_improved": qualifies}
                trials.append(trial)
                log.write(json.dumps(trial) + "\n")
                if qualifies and (best is None or (hits, mean) > best[:2]):
                    best = (hits, mean, policy, index)
                    print(f"New feasible best trial{index}: hits={hits}/200 Personal={mean:.9f} params={asdict(policy)}", flush=True)
                if index % 250 == 0:
                    log.flush()
                    print(f"Completed {index} policies in {time.monotonic()-start:.1f}s; best={best[:2] if best else None}", flush=True)
        write_csv(output / "search_trials.csv", trials)
        if best is None:
            write_json(output / "search_status.json", {"both_improved_found": False, "trials": len(trials)})
            raise RuntimeError("No policy met both targets in this search; inspect and extend the family")
        chosen = best[2]
        search_info = {"policies_evaluated": len(trials), "qualifying_policies": sum(r["both_improved"] for r in trials),
                       "selected_trial": best[3], "selection": "Both strictly better, then maximize oracle hit count, then Personal",
                       "random_seed": 20261024, "all200_used_for_tuning": True, "search_log_sha256": sha256(search_file)}
        write_csv(output / "top_policies.csv", sorted(trials, key=lambda r: (r["both_improved"], r["oracle_hits"], r["mean_personal"]), reverse=True)[:50])
    saved = {"version": "development_tuned_v2", "parameters": asdict(chosen), "search": search_info,
             "dataset_role": "all200_development_tuned_NOT_independent_test"}
    write_json(output / "selected_policy.json", saved)
    fast_winners, _ = replay_batch(data, chosen)
    summary = export_final(output, tasks, chosen, fast_winners, search_info, args.metrics_csv)
    write_json(output / "run_manifest.json", {"metrics_csv": str(args.metrics_csv.resolve()), "metrics_sha256": sha256(args.metrics_csv),
              "cache_root": str(args.cache_root.resolve()), "script_sha256": sha256(__file__),
              "selected_policy_sha256": sha256(output / "selected_policy.json"),
              "pool_audit_sha256": sha256(output / "candidate_pool_audit.csv"),
              "trace_sha256": sha256(output / "step_trace.jsonl"), "selection_sha256": sha256(output / "selection_pre_posthoc.csv"),
              "old_dp_table_used": False, "neural_inference_calls": 0,
              "elapsed_seconds": time.monotonic() - start, "python": sys.version, "numpy": np.__version__})
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
