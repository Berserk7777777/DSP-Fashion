"""Development-tuned sparse observation PSP; original8, exact112, no future reads."""
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
    CausalObserver, load_and_audit, posthoc, replay, sha256, summarize, write_csv, write_json,
)
from fashion_psp.experiments.search_adaptive_personal_development import feasible_counts


@dataclass(frozen=True)
class SparsePolicy:
    first: int = 5
    margin: float = 3
    protect: int = 2
    max_drop: int = 4
    near_interval: int = 2
    far_interval: int = 7
    uncertainty_gap: float = 2


def observation_reason(policy, step, spent, count, next_observation):
    """Called BEFORE scores are available. spent includes this denoising step."""
    if step == 28:
        return "terminal"
    feasible = feasible_counts(step, spent, count)
    if min(feasible) == count:
        return ""  # No reduction is possible: do not decode for a non-action.
    if max(feasible) < count:
        return "budget_deadline"
    return "scheduled" if step >= next_observation else ""


def choose(policy, scores, step, spent):
    """Pure online decision: only current live scores, counts and budget."""
    ranked = sorted(scores, key=lambda s: (-scores[s], s))
    best = scores[ranked[0]]
    eligible = [s for s in ranked[policy.protect:] if best - scores[s] > policy.margin]
    desired = len(ranked) - min(len(eligible), policy.max_drop)
    feasible = feasible_counts(step, spent, len(ranked))
    keep = min(max(desired, min(feasible)), max(feasible))
    survivors = sorted(ranked[:keep])
    voluntary = min(len(eligible), policy.max_drop, len(ranked) - keep)
    marked_removed = set(ranked[len(ranked)-voluntary:]) if voluntary else set()
    reasons = {str(s): "score_gap" if s in marked_removed else "budget_forced" for s in ranked[keep:]}
    # Only surviving scores decide whether the next observation should be near.
    boundary_gap = scores[ranked[0]] - scores[ranked[keep-1]]
    interval = policy.near_interval if boundary_gap <= policy.uncertainty_gap else policy.far_interval
    return survivors, reasons, step + interval, {
        "feasible_keep_counts": feasible, "eligible_seeds": eligible,
        "survivor_score_spread": boundary_gap, "next_interval": interval,
    }


def sparse_replay(policy, seeds, observe):
    active, spent, views, rounds, next_at = sorted(seeds), 0, 0, 0, policy.first
    trace = []
    for step in range(1, 29):
        before = active.copy()
        spent += len(before)
        reason = observation_reason(policy, step, spent, len(before), next_at)
        scores = observe(step, before, bool(reason))
        removed, detail = {}, {}
        if reason:
            assert set(scores) == set(before) and all(np.isfinite(v) for v in scores.values())
            views += len(before)
            rounds += 1
            if step == 28:
                active = [min(before, key=lambda s: (-scores[s], s))]
                removed = {str(s): "terminal_selection" for s in before if s not in active}
            else:
                active, removed, next_at, detail = choose(policy, scores, step, spent)
        else:
            assert not scores
        trace.append({"step": step, "alive_before": before, "alive_after": active.copy(),
                      "count_before": len(before), "count_after": len(active),
                      "observed": bool(reason), "observation_reason": reason,
                      "personal_scores": scores, "removed_reasons": removed,
                      "next_observation": next_at, "cumulative_particle_steps": spent,
                      "cumulative_decode_images": views, "cumulative_personal_evaluations": views,
                      "cumulative_scoring_rounds": rounds, **detail})
    assert spent == 112
    return {"selected_seed": active[0], "particle_steps": spent, "decode_images": views,
            "personal_evaluations": views, "scoring_rounds": rounds, "trace": trace}


def batch_replay(data, policy):
    """Vectorized search environment; scores copied only at live observation events."""
    nt = len(data)
    alive = np.ones((nt, 8), bool)
    spent, views, rounds = (np.zeros(nt, int) for _ in range(3))
    next_at = np.full(nt, policy.first, int)
    for t in range(28):
        step = t + 1
        n = alive.sum(1)
        spent += n
        if step == 28:
            obs = np.ones(nt, bool)
        else:
            h, remaining = 28-step, 112-spent
            lo = np.maximum(1, (remaining+h-1)//h)
            hi = np.minimum(n, remaining-h+1)
            assert np.all(lo <= hi)
            obs = (lo < n) & ((hi < n) | (step >= next_at))
        ix = np.flatnonzero(obs)
        if not len(ix):
            continue
        scores = np.full((len(ix), 8), -np.inf)
        rr, ss = np.where(alive[ix])
        scores[rr, ss] = data[ix[rr], t, ss]
        views[ix] += n[ix]
        rounds[ix] += 1
        if step == 28:
            assert np.all(spent == 112)
            return np.argmax(scores, axis=1), views, rounds
        order = np.argsort(-scores, axis=1, kind="stable")
        ranks = np.argsort(order, axis=1, kind="stable")
        gaps = scores.max(1)[:, None] - scores
        eligible = alive[ix] & (ranks >= policy.protect) & (gaps > policy.margin)
        desired = n[ix] - np.minimum(eligible.sum(1), policy.max_drop)
        keep = np.maximum(lo[ix], np.minimum(desired, hi[ix]))
        spread = scores.max(1) - scores[np.arange(len(ix)), order[np.arange(len(ix)), keep-1]]
        next_at[ix] = step + np.where(spread <= policy.uncertainty_gap, policy.near_interval, policy.far_interval)
        alive[ix] &= ranks < keep[:, None]
    raise AssertionError("No terminal")


def candidates():
    seen = set()
    for first, margin, protect, cap, interval in itertools.product(
            [3,4,5,6,7,8,9], [0,1,2,3,4,6,8,12], [2,3], [2,4,8], [2,4,7,28]):
        p = SparsePolicy(first, margin, protect, cap, interval, interval, 2)
        seen.add(p)
        yield "periodic_budget_events", p
    rng = random.Random(20261025)
    for _ in range(4000):
        p = SparsePolicy(rng.choice([3,4,5,6,7,8,9,10]), rng.choice([0,0.5,1,1.5,2,3,4,5,6,8,10]),
                         rng.choice([1,2,3,4]), rng.choice([1,2,3,4,5,8]),
                         rng.choice([1,2,3,4]), rng.choice([4,5,7,10,14,28]),
                         rng.choice([0.5,1,2,3,4,6,8]))
        if p not in seen:
            seen.add(p)
            yield "score_adaptive_intervals", p


def fixed_schedule_ablation(tasks, metrics, output, adaptive_rows):
    """Distinguish changing the dominant schedule from adaptive branching.

    This diagnostic is run after global policy selection and does not select it.
    Like default PSP it ranks current live Personal at three fixed steps.
    """
    selected = []
    for task, e in sorted(tasks.items()):
        active, spent = sorted(e["seeds"]), 0
        observer = CausalObserver(e["personal"], active)
        for step in range(1, 29):
            spent += len(active)
            scores = observer(step, active, step in (8,16,28))
            if scores:
                keep = {8:3,16:2,28:1}[step]
                active = sorted(sorted(active,key=lambda s:(-scores[s],s))[:keep])
        assert spent == 112
        selected.append({"task_index":task,"uid":e["identity"][0],"oid":e["identity"][1],
                         "method":"fixed8_16_top3_top2","selected_seed":active[0],
                         "particle_steps":spent,"decode_images":13,"personal_evaluations":13,"scoring_rounds":3})
    evaluated = posthoc(metrics,tasks,selected)
    write_csv(output/"fixed_schedule_ablation_selections.csv",evaluated)
    deltas = [a["personal"]-f["personal"] for a,f in zip(adaptive_rows,evaluated)]
    write_json(output/"fixed_schedule_ablation.json",{
        "note":"Post-selection diagnostic, development set; dominant schedule fixed at8/top3,16/top2,28/top1",
        **{f"mean_{key}":statistics.mean(r[key] for r in evaluated)
           for key in ("personal","compatibility","personal_regret","optimal_seed_hit","decode_images","scoring_rounds")},
        "all_tasks_exact112":True,
        "adaptive_vs_fixed":{"wins":sum(d>0 for d in deltas),"ties":sum(d==0 for d in deltas),
                             "losses":sum(d<0 for d in deltas),"mean_personal_delta":statistics.mean(deltas)}})


def export(output, tasks, policy, metrics, search):
    selections, traces, checks = [], [], []
    for task, e in sorted(tasks.items()):
        for method in ("default", "adaptive"):
            observer = CausalObserver(e["personal"], e["seeds"])
            result = (replay(e["seeds"], "default", observer) if method == "default" else
                      sparse_replay(policy, e["seeds"], observer))
            if method == "adaptive":
                used = set(observer.accesses)
                poisoned = {k:v if k in used else 1e90 for k,v in e["personal"].items()}
                assert result == sparse_replay(policy, e["seeds"], CausalObserver(poisoned,e["seeds"]))
                checks.append({"task_index":task,"unobserved_score_poisoning_invariant":True,
                               "exact112":result["particle_steps"]==112})
            traces.extend({"task_index":task,"method":method,**r} for r in result.pop("trace"))
            selections.append({"task_index":task,"uid":e["identity"][0],"oid":e["identity"][1],"method":method,**result})
    write_csv(output / "selection_pre_posthoc.csv", selections)
    with (output / "step_trace.jsonl").open("w",encoding="utf-8") as f:
        for r in traces: f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
    evaluated = posthoc(metrics,tasks,selections)
    summary,pairs = summarize(evaluated,traces)
    summary["dataset_role"] = "all200_development_tuned_NOT_independent_test"
    summary["search"] = search
    schedules=Counter(tuple((r["step"],r["count_after"]) for r in traces
                    if r["method"]=="adaptive" and r["task_index"]==task and r["count_after"]<r["count_before"])
                      for task in tasks)
    observations=Counter(tuple(r["step"] for r in traces if r["method"]=="adaptive"
                               and r["task_index"]==task and r["observed"]) for task in tasks)
    summary["unique_pruning_schedules"] = len(schedules)
    summary["unique_observation_schedules"] = len(observations)
    selected=[r for r in evaluated if r["method"]=="adaptive"]
    summary["adaptive_view_range"]=[min(r["decode_images"] for r in selected),max(r["decode_images"] for r in selected)]
    summary["adaptive_round_range"]=[min(r["scoring_rounds"] for r in selected),max(r["scoring_rounds"] for r in selected)]
    for method in ("default", "adaptive"):
        # Online CLIP uses batches of4 padded by repeated images; these are not
        # extra VAE decodes, but they do consume reward-model computation.
        batches = sum((r["count_before"]+3)//4 for r in traces if r["method"]==method and r["observed"])
        summary["methods"][method]["total_clip_batches_padded4"] = batches
        summary["methods"][method]["mean_clip_image_slots_including_padding"] = 4*batches/len(tasks)
    write_json(output/"summary.json",summary)
    write_csv(output/"task_comparison.csv",pairs)
    write_csv(output/"selections.csv",evaluated)
    write_csv(output/"validation_results.csv",checks)
    write_csv(output/"pruning_events.csv",[r for r in traces if r["count_after"]<r["count_before"]])
    write_csv(output/"observation_events.csv",[r for r in traces if r["observed"]])
    write_csv(output/"schedule_distribution.csv",[{"schedule":p,"tasks":n} for p,n in schedules.most_common()])
    write_csv(output/"observation_distribution.csv",[{"steps":p,"tasks":n} for p,n in observations.most_common()])
    write_csv(output/"budget_checks.csv",[{k:r[k] for k in ("task_index","method","particle_steps","decode_images","personal_evaluations","scoring_rounds")} for r in evaluated])
    fixed_schedule_ablation(tasks,metrics,output,selected)
    return summary,selected


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",type=Path,default=ROOT/"fashion_psp/artifacts/sparse_personal_development_v3")
    ap.add_argument("--policy",type=Path)
    args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    metrics=ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv"
    cache=ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_cache"
    tasks,audit=load_and_audit(metrics,cache,200)
    write_csv(out/"candidate_pool_audit.csv",audit)
    task_order=sorted(tasks)
    positions={t:i for i,t in enumerate(task_order)}
    data=np.array([[[tasks[t]["personal"][(step,s)] for s in tasks[t]["seeds"]] for step in range(1,29)] for t in task_order])
    terminal=data[:,-1,:];oracle=terminal.max(1);rr=np.arange(len(task_order))
    baseline=[]
    for t in task_order:
        e=tasks[t]
        r=replay(e["seeds"],"default",CausalObserver(e["personal"],e["seeds"]))
        baseline.append(e["personal"][(28,r["selected_seed"])])
    base_mean=float(np.mean(baseline));base_hits=int((np.array(baseline)==oracle).sum())
    if args.policy:
        saved=json.loads(args.policy.read_text(encoding="utf-8"));policy=SparsePolicy(**saved["parameters"]);search=saved["search"]
    else:
        trials=[];start=time.monotonic()
        for i,(phase,p) in enumerate(candidates(),1):
            win,views,rounds=batch_replay(data,p)
            values=terminal[rr,win];hits=int((values==oracle).sum());mean=float(values.mean())
            trials.append({"trial":i,"phase":phase,**asdict(p),"personal":mean,"hits":hits,
                           "mean_views":float(views.mean()),"max_views":int(views.max()),
                           "mean_rounds":float(rounds.mean()),"qualifies":mean>base_mean and hits>base_hits})
            if i%500==0:print(f"{i} trials, {time.monotonic()-start:.1f}s",flush=True)
        write_csv(out/"search_trials.csv",trials)
        qualifying=[r for r in trials if r["qualifies"]]
        if not qualifying:raise RuntimeError("No qualifying sparse policy")
        # Cost ceilings are reported together, not silently ignored during selection.
        frontier=[]
        for ceiling in [14,16,20,24,28,32,40]:
            eligible=[r for r in qualifying if r["mean_views"]<=ceiling]
            if eligible:
                best=max(eligible,key=lambda r:(r["hits"],r["personal"],-r["mean_views"]))
                frontier.append({"mean_view_ceiling":ceiling,**best})
        write_csv(out/"cost_frontier.csv",frontier)
        # Prefer a <=20-image solution if possible; otherwise use the lowest feasible tier.
        eligible=[r for r in qualifying if r["mean_views"]<=20]
        if not eligible:eligible=[r for r in qualifying if r["mean_views"]<=min(x["mean_views"] for x in qualifying)+1e-9]
        best=max(eligible,key=lambda r:(r["hits"],r["personal"],-r["mean_views"]))
        policy=SparsePolicy(**{k:best[k] for k in asdict(SparsePolicy())})
        search={"policies_evaluated":len(trials),"qualifying_policies":len(qualifying),"selected_trial":best["trial"],
                "selection":"Both metrics >default; mean views<=20 if feasible; then hits, Personal, lower views",
                "all200_tuning":True,"rng_seed":20261025,"trials_sha256":sha256(out/"search_trials.csv")}
    write_json(out/"selected_policy.json",{"version":"sparse_personal_v3","parameters":asdict(policy),"search":search})
    summary,selected=export(out,tasks,policy,metrics,search)
    wins,views,rounds=batch_replay(data,policy)
    for row in selected:
        t=row["task_index"];i=positions[t]
        assert row["selected_seed"]==tasks[t]["seeds"][int(wins[i])]
        assert row["decode_images"]==views[i] and row["scoring_rounds"]==rounds[i]
    write_json(out/"run_manifest.json",{"script_sha256":sha256(__file__),"input_sha256":sha256(metrics),
               "policy_sha256":sha256(out/"selected_policy.json"),"trace_sha256":sha256(out/"step_trace.jsonl"),
               "scalar_batch_parity_tasks":200,"future_and_skipped_step_poisoning_tasks":200,"neural_calls":0})
    print(json.dumps({"policy":asdict(policy),**summary},indent=2),flush=True)


if __name__=="__main__": main()
