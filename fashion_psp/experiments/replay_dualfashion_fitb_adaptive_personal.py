"""Frozen, task-local causal Personal PSP@8 replay on the development cache.

Standard-library only. No neural inference and no learned calibration tables.
Run as ``python -m fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal``.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
HORIZON, BUDGET = 28, 112
RULE = {
    "version": "personal_stability_v1",
    "status": "frozen_before_first_replay_no_parameter_search",
    "dataset_role": "development_set_online_replay_not_independent_test",
    "initial_particles": 8, "horizon": HORIZON, "particle_step_budget": BUDGET,
    "default_schedule": [[7, 4], [14, 2], [28, 1]],
    "observation": "Adaptive observes every live seed at every reached step, including step28.",
    "window": 3, "std_ddof": 1, "std_multiplier": 2,
    "reference": "Fix the current-step leader; compare its observed last3 scores to each live candidate.",
    "eligible": "All3 leader-minus-candidate gaps >0 and mean(gaps)>2*sample_std(gaps).",
    "interpretation": "Stability heuristic, NOT a statistical confidence interval.",
    "feasibility": "Exact integer reachability with only nonincreasing positive live counts; no reward input.",
    "count_choice": "Let desired=n-eligible_count. Choose smallest feasible count>=desired; if none, choose largest feasible count (minimum forced removals).",
    "identity_choice": "Prune eligible seeds first, worst current Personal first; then, only if necessary, prune unmarked seeds worst current Personal first.",
    "ties": "Higher Personal then smaller integer seed wins; worst order is reversed. No numeric tolerance.",
    "terminal": "At step28 assert spent=112, select highest current Personal, smaller seed breaks ties.",
    "causality": "Policy receives only initial seeds and an observer callback returning reached, live Personal scores. No future/compatibility/image/GT/calibration access.",
    "oracle": "Posthoc only, over the same metadata-confirmed original8 terminal seeds. Regret=max(finalPersonal)-selectedPersonal; hit means selected belongs to exact maximum tie set.",
    "cost": "Counts per particle-image and scoring round, not hardware forward batches. Adaptive112 views/28 rounds; default14 views/3 rounds. Cache replay itself invokes no decoders or reward models.",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


@lru_cache(None)
def reachable_costs(steps_left, next_count):
    """Pure combinatorics: next_count must run at the next denoising step."""
    if steps_left == 1:
        return frozenset([next_count])
    if steps_left < 1 or next_count < 1:
        raise ValueError("positive horizon/count required")
    return frozenset(next_count + cost
                     for later_count in range(1, next_count + 1)
                     for cost in reachable_costs(steps_left - 1, later_count))


def rank(scores):
    return sorted(scores, key=lambda seed: (-scores[seed], seed))


def adaptive_decision(history, step, spent):
    """History contains only reached scores of currently live candidates."""
    scores = {seed: values[-1] for seed, values in history.items()}
    ranked = rank(scores)
    leader = ranked[0]
    evidence, eligible = {}, []
    if step >= RULE["window"]:
        for seed in ranked:
            gaps = [a - b for a, b in zip(history[leader][-3:], history[seed][-3:])]
            mean, std = statistics.mean(gaps), statistics.stdev(gaps)
            marked = all(gap > 0 for gap in gaps) and mean > 2 * std
            evidence[str(seed)] = {"gaps": gaps, "gap_mean": mean, "gap_sample_std": std, "eligible": marked}
            if marked:
                eligible.append(seed)
    remaining = BUDGET - spent
    feasible = [m for m in range(1, len(ranked) + 1)
                if remaining in reachable_costs(HORIZON - step, m)]
    if not feasible:
        raise AssertionError("No exact112 continuation")
    desired = len(ranked) - len(eligible)
    sufficient = [m for m in feasible if m >= desired]
    keep = min(sufficient) if sufficient else max(feasible)
    worst = list(reversed(ranked))
    priority = [s for s in worst if s in eligible] + [s for s in worst if s not in eligible]
    removed = priority[:len(ranked) - keep]
    reasons = {str(s): "stability_heuristic" if s in eligible else "budget_forced" for s in removed}
    return ([s for s in ranked if s not in removed], reasons,
            {"leader": leader, "eligible_seeds": sorted(eligible), "feasible_keep_counts": feasible,
             "desired_keep_count": desired, "stability_evidence": evidence})


def replay(seeds, method, observe):
    """Policy boundary: no full trajectories, terminal oracle, or other rewards."""
    if method not in ("default", "adaptive"):
        raise ValueError(method)
    active, history, spent, views, rounds, trace = sorted(seeds), {}, 0, 0, 0, []
    for step in range(1, HORIZON + 1):
        before = sorted(active)
        spent += len(before)
        needs_scores = method == "adaptive" or step in (7, 14, 28)
        scores = observe(step, before, needs_scores)
        if needs_scores:
            if set(scores) != set(before) or not all(math.isfinite(v) for v in scores.values()):
                raise ValueError("Expected exactly one finite Personal per live seed")
            views += len(before)
            rounds += 1
            for seed in before:
                history.setdefault(seed, []).append(scores[seed])
        reasons, detail = {}, {}
        if step == HORIZON:
            assert spent == BUDGET
            active = rank(scores)[:1]
            reasons = {str(s): "terminal_selection" for s in before if s not in active}
        elif method == "adaptive":
            active, reasons, detail = adaptive_decision(history, step, spent)
        elif step in (7, 14):
            active = rank(scores)[:{7: 4, 14: 2}[step]]
            reasons = {str(s): "fixed_schedule" for s in before if s not in active}
        assert set(active) <= set(before) and active
        history = {seed: history[seed] for seed in active if seed in history}
        trace.append({"step": step, "alive_before": before, "alive_after": sorted(active),
                      "count_before": len(before), "count_after": len(active),
                      "removed_reasons": reasons, "personal_scores": scores,
                      "observed": needs_scores, "cumulative_particle_steps": spent,
                      "remaining_particle_steps": BUDGET - spent,
                      "cumulative_decode_images": views, "cumulative_personal_evaluations": views,
                      "cumulative_scoring_rounds": rounds, **detail})
    assert spent == BUDGET and len(active) == 1
    return {"selected_seed": active[0], "particle_steps": spent, "decode_images": views,
            "personal_evaluations": views, "scoring_rounds": rounds, "trace": trace}


class CausalObserver:
    """Replay environment; guards monotone time, survival and score accesses."""
    def __init__(self, personal, seeds):
        self.__personal = personal
        self.__alive = set(seeds)
        self.__step = 0
        self.accesses = []

    def __call__(self, step, seeds, needs_scores):
        if step != self.__step + 1 or not set(seeds) <= self.__alive or not seeds:
            raise AssertionError("Noncausal observation request")
        self.__step, self.__alive = step, set(seeds)
        if not needs_scores:
            return {}
        self.accesses.extend((step, seed) for seed in seeds)
        return {seed: self.__personal[(step, seed)] for seed in seeds}


def load_and_audit(metrics, cache_root, expected_tasks):
    """Audit identities/coverage; only Personal is parsed for replay."""
    tasks = {}
    with metrics.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"task_index", "uid", "oid", "seed", "step", "scheduler_timestep",
                    "sigma", "image_path", "personal", "compatibility"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Missing required CSV fields")
        for row in reader:
            task, step, seed = (int(row[k]) for k in ("task_index", "step", "seed"))
            entry = tasks.setdefault(task, {"identity": (int(row["uid"]), int(row["oid"])), "personal": {}, "paths": {}})
            if entry["identity"] != (int(row["uid"]), int(row["oid"])):
                raise ValueError("Task identity changed within CSV")
            key = (step, seed)
            if key in entry["personal"]:
                raise ValueError("Duplicate task/step/seed")
            value = float(row["personal"])
            if not math.isfinite(value):
                raise ValueError("Nonfinite Personal")
            entry["personal"][key], entry["paths"][key] = value, row["image_path"]
    if sorted(tasks) != list(range(expected_tasks)):
        raise ValueError("Unexpected task mapping/count")
    audit = []
    for task, entry in sorted(tasks.items()):
        # Explicit original cache identity, not first8 after arbitrary sorting.
        meta_path = cache_root / f"psp_offline_group0_task{task}_seed2026/metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        seeds = [int(seed) for seed in meta["candidate_seeds"]]
        expected = {(step, seed) for step in range(1, 29) for seed in seeds}
        if (len(seeds) != 8 or len(set(seeds)) != 8 or set(entry["personal"]) != expected
                or meta["task_index"] != task or entry["identity"] != (meta["uid"], meta["oid"])):
            raise ValueError(f"Original8 metadata mismatch: task{task}")
        if seeds != list(range(2026 + task * 10000, 2034 + task * 10000)):
            raise ValueError("Unexpected original seed mapping; investigate instead of remapping")
        cache = meta["offline_candidate_cache"]
        meta_images = {(int(r["step"]), int(r["seed"])): r["image_path"] for r in cache["images"]}
        if set(meta_images) != expected or cache["image_count"] != 224:
            raise ValueError("Incomplete metadata trajectory")
        for key, path in entry["paths"].items():
            if Path(path).resolve() != Path(meta_images[key]).resolve():
                raise ValueError("CSV/metadata image mismatch")
        entry["seeds"] = seeds
        audit.append({"task_index": task, "uid": entry["identity"][0], "oid": entry["identity"][1],
                      "candidate_seeds": seeds, "candidate_rows": len(expected),
                      "metadata_path": str(meta_path.resolve()), "metadata_sha256": sha256(meta_path),
                      "original8_metadata_match": True})
    return tasks, audit


def posthoc(metrics, tasks, selections):
    terminal = {}
    with metrics.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["step"]) == HORIZON:
                terminal[(int(row["task_index"]), int(row["seed"]))] = row
    evaluated = []
    for selection in selections:
        task, seed = selection["task_index"], selection["selected_seed"]
        pool = tasks[task]["seeds"]
        oracle = max(float(terminal[(task, s)]["personal"]) for s in pool)
        oracle_seeds = sorted(s for s in pool if float(terminal[(task, s)]["personal"]) == oracle)
        row = terminal[(task, seed)]
        personal, comp = float(row["personal"]), float(row["compatibility"])
        if not math.isfinite(comp) or not Path(row["image_path"]).is_file():
            raise ValueError("Missing selected image/nonfinite terminal Compatibility")
        evaluated.append({**selection, "image_path": row["image_path"], "personal": personal,
                          "compatibility": comp, "personal_regret": oracle - personal,
                          "optimal_seed_hit": int(seed in oracle_seeds), "oracle_personal": oracle,
                          "oracle_seeds": oracle_seeds})
    return evaluated


def summarize(evaluated, traces):
    summary = {"dataset_role": RULE["dataset_role"], "tasks": len(evaluated) // 2, "methods": {}}
    by_method = {m: [r for r in evaluated if r["method"] == m] for m in ("default", "adaptive")}
    for method, rows in by_method.items():
        mt = [r for r in traces if r["method"] == method]
        reason_counts = {}
        for r in mt:
            for reason in r["removed_reasons"].values():
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        summary["methods"][method] = {
            **{f"mean_{key}": statistics.mean(r[key] for r in rows)
               for key in ("personal", "compatibility", "personal_regret", "optimal_seed_hit")},
            "particle_steps_per_task_min": min(r["particle_steps"] for r in rows),
            "particle_steps_per_task_max": max(r["particle_steps"] for r in rows),
            "all_tasks_exact112": all(r["particle_steps"] == 112 for r in rows),
            **{f"total_{key}": sum(r[key] for r in rows)
               for key in ("particle_steps", "decode_images", "personal_evaluations", "scoring_rounds")},
            "mean_decode_images": statistics.mean(r["decode_images"] for r in rows),
            "mean_scoring_rounds": statistics.mean(r["scoring_rounds"] for r in rows),
            "nonterminal_pruning_events": sum(r["step"] < 28 and r["count_after"] < r["count_before"] for r in mt),
            "removed_particle_reasons": reason_counts,
            "tasks_with_budget_forced_pruning": len({r["task_index"] for r in mt if "budget_forced" in r["removed_reasons"].values()}),
        }
    pairs = []
    for a, b in zip(by_method["default"], by_method["adaptive"]):
        assert a["task_index"] == b["task_index"]
        delta = b["personal"] - a["personal"]
        pairs.append({"task_index": a["task_index"], "uid": a["uid"], "oid": a["oid"],
                      "default_seed": a["selected_seed"], "adaptive_seed": b["selected_seed"],
                      "default_image_path": a["image_path"], "adaptive_image_path": b["image_path"],
                      "default_personal": a["personal"], "adaptive_personal": b["personal"],
                      "personal_delta_adaptive_minus_default": delta,
                      "outcome": "win" if delta > 0 else "loss" if delta < 0 else "tie"})
    summary["adaptive_vs_default"] = {
        "wins": sum(r["outcome"] == "win" for r in pairs),
        "ties": sum(r["outcome"] == "tie" for r in pairs),
        "losses": sum(r["outcome"] == "loss" for r in pairs),
        "mean_personal_delta": statistics.mean(r["personal_delta_adaptive_minus_default"] for r in pairs),
        "same_selected_seed_tasks": sum(r["default_seed"] == r["adaptive_seed"] for r in pairs),
    }
    summary["actual_replay_neural_calls"] = {"denoiser": 0, "decoder": 0, "reward": 0}
    return summary, pairs


def report(output, summary):
    a, b = (summary["methods"][m] for m in ("default", "adaptive"))
    comparison = summary["adaptive_vs_default"]
    lines = ["# Personal PSP 开发集因果在线回放", "",
             "这200个任务已参与策略开发；本结果不是独立测试。规则在首次回放前冻结，本次未搜索或调整参数。", "",
             "原始CSV共44,800行，每任务恰好8个seed×28步；全部与原缓存metadata的task/uid/oid/candidate_seeds及图片路径一致，未混入额外8个seed。", "",
             "|方法|平均终值 Personal|平均终值 Compatibility|平均 Personal regret|最优seed命中率|",
             "|---|---:|---:|---:|---:|"]
    for name, values in (("默认 PSP@8", a), ("任务内自适应 PSP@8", b)):
        lines.append(f"|{name}|{values['mean_personal']:.9f}|{values['mean_compatibility']:.9f}|{values['mean_personal_regret']:.9f}|{values['mean_optimal_seed_hit']:.1%}|")
    lines += ["", f"自适应相对默认，按终值Personal：胜/平/负 = {comparison['wins']}/{comparison['ties']}/{comparison['losses']}；均值差 = {comparison['mean_personal_delta']:+.9f}。",
              "", "Regret/oracle限定在同一原始8-seed池；终值并列最大均算命中，胜平负使用CSV数值精确比较。Compatibility仅在选择冻结后读取。", "",
              "两方法每任务均恰好112粒子步（最小值=最大值=112），各200任务总计22,400粒子步。", "",
              "|在线实施所需开销|默认/任务|自适应/任务|默认总计|自适应总计|",
              "|---|---:|---:|---:|---:|",
              f"|解码图片数|14|112|{a['total_decode_images']}|{b['total_decode_images']}|",
              f"|Personal图片评估次数|14|112|{a['total_personal_evaluations']}|{b['total_personal_evaluations']}|",
              f"|评分轮数|3|28|{a['total_scoring_rounds']}|{b['total_scoring_rounds']}|", "",
              "图片评估/解码计数按粒子图片计，不等于GPU批次；终步解码已计入，选中图可复用。自适应图片评估量为默认8倍，评分轮数为28/3倍。相同去噪预算不代表相同端到端开销，未测量在线GPU时间。此次实际执行为缓存回放，未调用去噪器、解码器或reward模型；事后Compatibility/oracle为CSV查表，不计入策略在线开销。", "",
              f"自适应非终步实际剪枝事件：{b['nonterminal_pruning_events']}；剪除原因计数：{json.dumps(b['removed_particle_reasons'], ensure_ascii=False)}；发生预算强制剪枝的任务数：{b['tasks_with_budget_forced_pruning']}。", "",
              "冻结规则详见 frozen_rule.json：每步固定当前领先者作为最近3步参照；持续严格落后且均值>2×样本标准差（ddof=1）标记可剪。它是稳定性启发式，不是统计置信区间。精确预算可达性只看步数、粒子数和已消耗预算；候选按当前Personal及seed确定。可剪但会导致剩余预算无法用尽的剪枝会被延后；稳定性不足时为保持可达性执行预算强制剪枝。", "",
              "策略函数仅接收seed及受约束Personal观察回调；已剪粒子未来值、Compatibility、GT、CIS、LPIPS、旧DP风险表均不进入决策。先保存selection_pre_posthoc.csv和完整决策，再做终值oracle/regret和Compatibility评估。", "",
              "文件：task_comparison.csv为逐任务两方法seed及图片路径；selections.csv为400条完整选择和指标；pruning_events.csv为每次实际剪枝的步、前后粒子数、原因及累计预算（含终步选择，event_type区分）；step_trace.jsonl为每一步证据；budget_checks.csv为逐任务预算与评估次数；candidate_pool_audit.csv为metadata核对；run_manifest.json记录输入、规则、脚本及metadata摘要哈希。", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=ROOT / "DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "DualFashion-main/output/ifashion_fitb_timestep200_cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fashion_psp/artifacts/adaptive_personal_causal_replay_v1")
    parser.add_argument("--expected-tasks", type=int, default=200)
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen_rule.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != RULE:
            raise ValueError("Frozen rule differs; never silently overwrite")
    else:
        write_json(frozen, RULE)
    if args.freeze_only:
        print(f"Frozen rule: {frozen}; SHA256={sha256(frozen)}")
        return
    input_hash = sha256(args.metrics_csv)
    tasks, audit = load_and_audit(args.metrics_csv, args.cache_root, args.expected_tasks)
    write_csv(output / "candidate_pool_audit.csv", audit)
    selections, traces = [], []
    for task, entry in sorted(tasks.items()):
        for method in ("default", "adaptive"):
            observer = CausalObserver(entry["personal"], entry["seeds"])
            result = replay(entry["seeds"], method, observer)
            assert len(observer.accesses) == result["personal_evaluations"]
            traces.extend({"task_index": task, "method": method, **t} for t in result.pop("trace"))
            selections.append({"task_index": task, "uid": entry["identity"][0], "oid": entry["identity"][1], "method": method, **result})
    # Freeze causal choices and traces before any posthoc Compatibility access.
    write_csv(output / "selection_pre_posthoc.csv", selections)
    with (output / "step_trace.jsonl").open("w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    events = [{"task_index": r["task_index"], "method": r["method"],
               "event_type": "terminal_selection" if r["step"] == 28 else "pruning", **r}
              for r in traces if r["count_after"] < r["count_before"]]
    write_csv(output / "pruning_events.csv", events)
    evaluated = posthoc(args.metrics_csv, tasks, selections)
    write_csv(output / "selections.csv", evaluated)
    write_csv(output / "budget_checks.csv", [{k: r[k] for k in ("task_index", "method", "particle_steps", "decode_images", "personal_evaluations", "scoring_rounds")} | {"exact112": r["particle_steps"] == 112} for r in evaluated])
    summary, pairs = summarize(evaluated, traces)
    write_json(output / "summary.json", summary)
    write_csv(output / "task_comparison.csv", pairs)
    report(output, summary)
    assert input_hash == sha256(args.metrics_csv), "Input changed during replay"
    write_json(output / "run_manifest.json", {
        "dataset_role": RULE["dataset_role"], "metrics_csv": str(args.metrics_csv.resolve()),
        "metrics_sha256": input_hash, "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
        "frozen_rule_sha256": sha256(frozen), "pool_audit_sha256": sha256(output / "candidate_pool_audit.csv"),
        "frozen_selections_sha256": sha256(output / "selection_pre_posthoc.csv"),
        "step_trace_sha256": sha256(output / "step_trace.jsonl"),
        "tasks": len(tasks), "candidate_rows": sum(len(t["personal"]) for t in tasks.values()),
        "causal_access_checks_passed": True, "old_dp_calibration_used": False,
        "parameter_search_runs": 0, "actual_neural_inference_calls": 0,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
