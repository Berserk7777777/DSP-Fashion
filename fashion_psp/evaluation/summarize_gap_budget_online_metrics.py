"""Audit completed online gap-budget selections and summarize native metrics.

Does not generate/select images or read the old DP calibration. Native metrics
are produced by evaluate_dualfashion_fitb_native_metrics.py on frozen PNGs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT/"DualFashion-main/output/ifashion_fitb_gap_budget/personal_group0_seed2026"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit(root, out):
    config = read_json(root/"run_config.json")
    resume = read_json(root/"resume_state.json")
    total = resume["total_tasks"]
    check(resume["completed_count"] == total == 2207, "Expected all2207 tasks complete")
    check(sorted(resume["completed_indices"]) == list(range(total)), "Incomplete task indices")
    check(config["method"] == "frozen_gap_budget_psp8", "Unexpected policy")
    check(sha(config["policy_path"]) == config["policy_sha256"], "Policy file changed")
    generated = np.load(root/"0057501/gen.npy", allow_pickle=True).item()
    ground = np.load(root/"0057501/grd.npy", allow_pickle=True).item()
    fitb = np.load(config["fitb"]["path"], allow_pickle=True).item()
    packed = {(int(u), int(o)): r for u, outfits in generated.items() for o, r in outfits.items()}
    gt = {(int(u), int(o)): r for u, outfits in ground.items() for o, r in outfits.items()}
    files = sorted((root/"tasks").glob("*/task_summary.json"))
    check(len(files) == len(packed) == len(gt) == total, "Payload/task count mismatch")
    rows, indices, identities, branches = [], set(), set(), Counter()
    hash_cache = {}
    def file_hash(path):
        path = Path(path).resolve()
        if str(path) not in hash_cache:
            hash_cache[str(path)] = sha(path)
        return hash_cache[str(path)]
    for path in files:
        task = read_json(path)
        idx, uid, oid = task["task_index"], task["uid"], task["oid"]
        check(idx not in indices and (uid, oid) not in identities, "Duplicate identity")
        indices.add(idx)
        identities.add((uid, oid))
        check((uid, oid) == (int(fitb["uids"][idx]), int(fitb["oids"][idx])), "FITB task mapping mismatch")
        seeds = [config["seed"]+idx*config["seed_stride"]+s for s in range(8)]
        check(task["candidate_seeds"] == seeds, "Original8 seed mapping changed")
        check(task["policy_sha256"] == config["policy_sha256"], "Mixed policy checkpoints")
        trace = task["trace"]
        check([r["step"] for r in trace] == list(range(1, 29)), "Missing step trace")
        first = trace[6]
        score7 = {int(k): v for k, v in first["personal_scores"].items()}
        order7 = sorted(score7, key=lambda s: (-score7[s], s))
        check(set(score7) == set(range(8)), "Missing first scores")
        keep = 3 if score7[order7[2]]-score7[order7[3]] > score7[order7[3]]-score7[order7[4]] else 4
        second = 21 if keep == 3 else 14
        live, spent, views, rounds, batches = list(range(8)), 0, 0, 0, 0
        for row in trace:
            step = row["step"]
            check(row["alive_before"] == live, "Nonmonotone candidate trace")
            spent += len(live)
            observed = step in (7, second, 28)
            check(row["observed"] == observed, "Unexpected scoring time")
            scores = {int(k): v for k, v in row["personal_scores"].items()}
            check(set(scores) == (set(live) if observed else set()), "Unexpected score accesses")
            check(all(math.isfinite(v) for v in scores.values()), "Nonfinite observed Personal")
            if observed:
                views += len(live)
                rounds += 1
                batches += (len(live)+3)//4
                count = keep if step == 7 else 2 if step == second else 1
                live = sorted(sorted(live, key=lambda s: (-scores[s], s))[:count])
            check(row["alive_after"] == live and row["particle_steps_spent"] == spent, "Invalid pruning/budget")
        check(spent == task["particle_steps"] == 112, "Non112 task")
        check(views == task["reward_views"] == 8+keep+2, "Incorrect reward count")
        check(rounds == task["scoring_rounds"] == 3, "Incorrect scoring rounds")
        check(batches == task["clip_batches_padded4"] == 4, "Incorrect batch count")
        check(task["clip_image_slots_including_padding"] == 16, "Incorrect padded slots")
        check(task["selected_seed"] == seeds[live[0]], "Wrong winner")
        check(task["selected_personal"] == trace[-1]["personal_scores"][str(live[0])], "Wrong final Personal")
        selected = Path(task["selected_image_path"]).resolve()
        record, gt_record = packed[uid, oid], gt[uid, oid]
        check(len(record["image_paths"]) == len(gt_record["image_paths"]) == 1, "Not single selected image")
        check(Path(record["image_paths"][0]).resolve() == selected, "Packed image mismatch")
        check(list(record["candidate_seeds"]) == [task["selected_seed"]], "Packed seed mismatch")
        check(int(np.asarray(record["cates"]).reshape(-1)[0]) == task["category"], "Category mismatch")
        gt_path = Path(gt_record["image_paths"][0]).resolve()
        branches[str(keep)] += 1
        rows.append({"task_index": idx, "uid": uid, "oid": oid, "category": task["category"],
                     "selected_seed": task["selected_seed"], "image_path": str(selected),
                     "gt_image_path": str(gt_path), "image_sha256": file_hash(selected),
                     "gt_sha256": file_hash(gt_path), "task_summary_sha256": sha(path),
                     "particle_steps": spent, "reward_views": views, "first_keep": keep,
                     "second_pruning_step": second, "online_personal": task["selected_personal"],
                     "elapsed_s": task["elapsed_s"]})
    check(indices == set(range(total)) and identities == set(packed) == set(gt), "Task coverage differs")
    rows.sort(key=lambda r: r["task_index"])
    input_hashes = {str(p.resolve()): sha(p) for p in (root/"run_config.json", root/"0057501/gen.npy", root/"0057501/grd.npy")}
    audit_csv = out/"online_selection_audit.csv"
    write_csv(audit_csv, rows)
    summary = {
        "tasks": total, "users": len({r["uid"] for r in rows}), "all_tasks_exact112": True,
        "all_frozen_gate_decisions_verified": True, "all_payload_task_image_seed_mappings_verified": True,
        "branches_first_keep": dict(branches), "total_particle_steps": 112*total,
        "total_reward_views_online": sum(r["reward_views"] for r in rows),
        "mean_reward_views_online": statistics.mean(r["reward_views"] for r in rows),
        "total_scoring_rounds_online": 3*total,
        "total_clip_batches_online_padded4": 4*total, "total_clip_slots_online_padded4": 16*total,
        "online_logged_personal_mean": statistics.mean(r["online_personal"] for r in rows),
        "source_payload_sha256": input_hashes, "selection_audit_sha256": sha(audit_csv),
        "dataset_role": "Full2207 online test-split run includes previously inspected tasks; not independent method confirmation",
    }
    write_json(out/"input_audit.json", summary)
    return summary


def summarize(root, out):
    audit_summary = read_json(out/"input_audit.json")
    audit_rows = read_csv(out/"online_selection_audit.csv")
    for raw_path, digest in audit_summary["source_payload_sha256"].items():
        check(sha(raw_path) == digest, "Input payload changed during evaluation")
    check(sha(out/"online_selection_audit.csv") == audit_summary["selection_audit_sha256"], "Audit changed")
    metrics = read_csv(out/"per_task_metrics.csv")
    keyed = {(int(r["uid"]), int(r["oid"])): r for r in metrics}
    expected = {(int(r["uid"]), int(r["oid"])) for r in audit_rows}
    check(len(metrics) == len(keyed) == audit_summary["tasks"] and set(keyed) == expected, "Metric task coverage mismatch")
    merged, deltas = [], []
    for row in audit_rows:
        value = keyed[int(row["uid"]), int(row["oid"])]
        check(int(value["selected_seed"]) == int(row["selected_seed"]), "Metric seed mismatch")
        for field in ("image_path", "gt_image_path"):
            check(Path(value[field]).resolve() == Path(row[field]).resolve(), "Metric image mismatch")
        check(sha(row["image_path"]) == row["image_sha256"], "Selected PNG changed")
        check(sha(row["gt_image_path"]) == row["gt_sha256"], "GT PNG changed")
        scores = {k: float(value[k]) for k in ("cis", "lpips", "compatibility", "personal")}
        check(all(math.isfinite(v) for v in scores.values()), "Invalid metric")
        delta = scores["personal"]-float(row["online_personal"])
        deltas.append(delta)
        merged.append({**row, **scores, "personal_reevaluated_minus_online": delta})
    summary = {"tasks": len(merged), "users": audit_summary["users"], "method": "frozen_gap_budget_psp8",
               "metrics": {k: {"mean": statistics.mean(r[k] for r in merged),
                                "std_sample": statistics.stdev(r[k] for r in merged)}
                           for k in ("cis", "lpips", "compatibility", "personal")},
               "online_personal_mean": audit_summary["online_logged_personal_mean"],
               "personal_reevaluation_delta_mean": statistics.mean(deltas),
               "personal_reevaluation_mean_absolute_difference": statistics.mean(abs(d) for d in deltas),
               "personal_reevaluation_max_absolute_difference": max(abs(d) for d in deltas),
               "personal_difference_over_0_001_task_indices": [int(r["task_index"]) for r in merged
                                                               if abs(r["personal_reevaluated_minus_online"]) > 0.001],
               "all_tasks_exact112": True, "source_integrity_verified": True,
               "dataset_role": audit_summary["dataset_role"],
               "metric_file_sha256": sha(out/"per_task_metrics.csv")}
    write_csv(out/"per_task_metrics_with_online_audit.csv", merged)
    write_json(out/"verified_summary.json", summary)
    lines = ["# 冻结分差自适应PSP：在线生成结果四项指标", "",
             f"已实际评估全部{len(merged)}个FITB任务，每任务一张最终选中PNG。", "",
             "| 指标 | 全任务平均值 |", "|---|---:|"]
    for name, key in (("CIS ↑（100×CLIP cosine，生成图对GT）", "cis"),
                      ("LPIPS-VGG ↓（生成图对GT）", "lpips"),
                      ("Compatibility ↑（0–1）", "compatibility"),
                      ("Personal ↑（100×CLIP cosine，生成图对用户历史）", "personal")):
        lines.append(f"| {name} | {summary['metrics'][key]['mean']:.9f} |")
    lines.extend(["", "指标沿用evaluate_dualfashion_fitb_native_metrics.py的既有实现：OpenCLIP ViT-H/14 float32、FashionEvaluator和LPIPS-VGG，batch_size=4。CIS/LPIPS均使用最终选中图与对应GT图；LPIPS不是单图无法定义的候选间多样性。GT只用于事后评估。", "",
                  f"在线日志Personal均值为{summary['online_personal_mean']:.9f}；本次重算减日志的平均差为{summary['personal_reevaluation_delta_mean']:.9f}，逐任务平均绝对差{summary['personal_reevaluation_mean_absolute_difference']:.9f}，最大绝对差{summary['personal_reevaluation_max_absolute_difference']:.9f}。在线reward以同任务候选补齐batch4评分，本次按任务批量评分且采用既有评估器的特征点积实现，浮点计算路径不同，以上差值作为一致性记录。", "",
                  f"逐任务绝对差超过0.001的仅{summary['personal_difference_over_0_001_task_indices']}，对应评估末尾不足4张的批次；原评估器不对尾批补齐。该差异不改变冻结的在线选择，本报告保留既有评估器原始输出。", "",
                  f"完整性检查：2207个任务、gen/grd条目、原始seed映射、冻结分差门控、28步轨迹、最终图片/seed全部一致；每任务112粒子步，总{audit_summary['total_particle_steps']}。首次留3/4任务数为{audit_summary['branches_first_keep']}。在线候选评分总{audit_summary['total_reward_views_online']}张，均值{audit_summary['mean_reward_views_online']:.6f}；评分轮总{audit_summary['total_scoring_rounds_online']}。这些是搜索成本，本次事后四指标评估另算。", "",
                  "源gen/grd/config及所有选中PNG/GT的SHA256核验通过。逐任务指标见per_task_metrics_with_online_audit.csv，原评估器输出见per_task_metrics.csv、summary.csv和protocol.json。", "",
                  "这是完整2207任务的在线结果统计，包含此前研究过的测试任务，不能将整批作为未参与方法选择的独立确认集。"])
    (out/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=DEFAULT)
    p.add_argument("--audit-only", action="store_true")
    args = p.parse_args()
    out = args.run_root/"native_metrics"
    out.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        print(json.dumps(audit(args.run_root, out), ensure_ascii=False, indent=2))
    else:
        summarize(args.run_root, out)


if __name__ == "__main__":
    main()
