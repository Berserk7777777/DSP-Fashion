"""Posthoc difficulty diagnosis ONLY; no policy search, fit, calibration or tuning.

Reads the original8 test-derived cache. Full future/terminal access is deliberate
in this diagnostic and MUST NOT be imported into an online pruning decision.
The only policy evaluated here is the already specified default PSP7/14/28.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import statistics

from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import (
    CausalObserver, load_and_audit, replay, sha256, write_csv, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = {
    "dataset_role":"test_subset_posthoc_difficulty_diagnosis_not_for_tuning",
    "allowed":"all28 steps and terminal oracle for descriptive diagnosis only",
    "policy_evaluated":"preexisting default PSP7->4,14->2,28->1 only",
    "new_policy_candidates_evaluated":0,"parameter_search":False,
    "model_fitting":False,"threshold_selection":False,"calibration":False,
    "diagnostics":["all28 step oracle ranks and pair reversals",
                   "default pruning-stage oracle loss and telescoping regret",
                   "pure integer resource feasibility independent of scores"],
    "ranking":"descending Personal, ascending original seed; exact terminal ties are all oracle seeds",
    "caution":"Descriptive curves must not be used to pick pruning steps, thresholds or a winning policy.",
}


def ranked(scores):
    return sorted(scores,key=lambda s:(-scores[s],s))


def rank_correlation(a,b):
    """Spearman with average ranks for actual score ties, not seed tie breaks."""
    def average_ranks(scores):
        groups = {}
        for s,v in scores.items():
            groups.setdefault(v,[]).append(s)
        rank, result = 1, {}
        for value in sorted(groups,reverse=True):
            group = groups[value]
            for s in group:
                result[s] = rank+(len(group)-1)/2
            rank += len(group)
        return result
    x,y=average_ranks(a),average_ranks(b)
    mx,my=statistics.mean(x.values()),statistics.mean(y.values())
    cross=sum((x[s]-mx)*(y[s]-my) for s in x)
    denom=math.sqrt(sum((v-mx)**2 for v in x.values())*sum((v-my)**2 for v in y.values()))
    return cross/denom if denom else None


def task_diagnosis(task,e):
    seeds=e["seeds"]
    final={s:e["personal"][(28,s)] for s in seeds}
    order28=ranked(final); best=max(final.values())
    oracle={s for s in seeds if final[s]==best}
    observer=CausalObserver(e["personal"],seeds)
    result=replay(seeds,"default",observer)
    trace={r["step"]:r for r in result["trace"]}
    after7,after14=trace[7]["alive_after"],trace[14]["alive_after"]
    best7,best14=max(final[s] for s in after7),max(final[s] for s in after14)
    stage7,stage14=best-best7,best7-best14
    selected=result["selected_seed"];regret=best-final[selected]
    assert math.isclose(stage7+stage14,regret,rel_tol=0,abs_tol=1e-12)
    loss_step=7 if not oracle.intersection(after7) else 14 if not oracle.intersection(after14) else 0
    assert bool(loss_step)==(final[selected]<best)
    rows=[]
    for t in range(1,29):
        current={s:e["personal"][(t,s)] for s in seeds}
        order=ranked(current)
        oracle_rank=min(order.index(s)+1 for s in oracle)
        strict_pairs=ties=flips=0
        for a,b in itertools.combinations(seeds,2):
            dx,dy=current[a]-current[b],final[a]-final[b]
            if dx==0 or dy==0:
                ties+=1
            else:
                strict_pairs+=1
                flips+=int(dx*dy<0)
        rows.append({"task_index":task,"uid":e["identity"][0],"oid":e["identity"][1],"step":t,
                     "oracle_seeds":sorted(oracle),"best_oracle_rank_all8":oracle_rank,
                     "oracle_in_top1":oracle_rank<=1,"oracle_in_top2":oracle_rank<=2,"oracle_in_top4":oracle_rank<=4,
                     "current_leader_seed":order[0],"current_leader_final_regret":best-final[order[0]],
                     "oracle_gap_to_current_leader":current[order[0]]-max(current[s] for s in oracle),
                     "spearman_to_final_all8":rank_correlation(current,final),
                     "strict_pairs":strict_pairs,"tied_pairs":ties,"reversed_pairs_to_final":flips,
                     "ranking_seed_order":order})
    scores7={s:e["personal"][(7,s)] for s in seeds};order7=ranked(scores7)
    scores14={s:e["personal"][(14,s)] for s in after7};order14=ranked(scores14)
    info={"task_index":task,"uid":e["identity"][0],"oid":e["identity"][1],
          "oracle_seeds":sorted(oracle),"oracle_personal":best,"top1_minus_top2_terminal":best-final[order28[1]],
          "default_selected_seed":selected,"default_selected_image_path":e["paths"][(28,selected)],
          "default_personal":final[selected],"default_regret":regret,
          "oracle_loss_step":loss_step,"oracle_rank_step7_all8":rows[6]["best_oracle_rank_all8"],
          "step7_boundary_gap_rank4_minus_rank5":scores7[order7[3]]-scores7[order7[4]],
          "step14_boundary_gap_live_rank2_minus_rank3":scores14[order14[1]]-scores14[order14[2]],
          "stage7_terminal_best_loss":stage7,"stage14_terminal_best_loss":stage14,
          "particle_steps":result["particle_steps"],"decode_images":result["decode_images"],
          "oracle_terminal_paths":[e["paths"][(28,s)] for s in sorted(oracle)]}
    return info,rows


def resource_schedules():
    """All two-internal-node schedules under exact112 and <=14 views.

    No candidate rewards or labels are arguments. This is not a policy sweep.
    Both internal nodes must strictly reduce, with >=2 survivors until28.
    """
    schedules=[]
    for a in range(1,27):
        for b in range(a+1,28):
            for k1 in range(3,8):
                for k2 in range(2,k1):
                    cost=8*a+k1*(b-a)+k2*(28-b)
                    views=8+k1+k2
                    if cost==112 and views<=14:
                        schedules.append({"first_step":a,"first_keep":k1,"second_step":b,"second_keep":k2,
                                          "terminal_step":28,"particle_steps":cost,"candidate_views":views})
    return schedules


def plot_difficulty(out,aggregate,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4),gridspec_kw={"width_ratios":[1.7,1]})
    for k,color in ((1,"#be6a29"),(2,"#608a65"),(4,"#326c9a")):
        axes[0].plot([r["step"] for r in aggregate],
                     [100*r[f"oracle_in_top{k}"]/r["tasks"] for r in aggregate],
                     label=f"Final-best seed in current top{k}",color=color,lw=2)
    for step in (7,14):
        axes[0].axvline(step,color="#999999",ls=":",lw=1)
    axes[0].set(xlabel="Denoising step",ylabel="Tasks (%)",xlim=(1,28),ylim=(0,102),
                title="All original8 trajectories: posthoc ranks")
    axes[0].legend(loc="lower right",frameon=False,fontsize=8)
    d=summary["default"]
    bars=axes[1].bar(["Step7 pruning","Step14 pruning"],
                      [d["mean_stage7_loss"],d["mean_stage14_loss"]],color=["#326c9a","#6c94b4"],width=.6)
    for bar in bars:
        axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.02,f"{bar.get_height():.3f}",ha="center")
    axes[1].set(ylabel="Mean loss of best terminal Personal",ylim=(0,1.15),title="Default PSP regret attribution")
    fig.suptitle("200-task test subset: difficulty diagnosis only; no policy tuning",fontsize=12)
    fig.text(.02,.01,"All8 curves access future data for diagnosis. They are not online policy results.",fontsize=8,color="#555555")
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(out/"difficulty.png",dpi=180)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir",type=Path,default=ROOT/"fashion_psp/artifacts/personal_test_difficulty_diagnosis")
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    write_json(out/"diagnostic_protocol.json",PROTOCOL)
    # All following numerical outputs are descriptive. There is no optimizer.
    metrics=ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv"
    cache=ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_cache"
    before_hash=sha256(metrics)
    tasks,audit=load_and_audit(metrics,cache,200)
    detail,rows=[],[]
    for task,e in sorted(tasks.items()):
        info,curve=task_diagnosis(task,e);detail.append(info);rows.extend(curve)
    aggregate=[]
    for step in range(1,29):
        sample=[r for r in rows if r["step"]==step]
        corr=[r["spearman_to_final_all8"] for r in sample if r["spearman_to_final_all8"] is not None]
        strict=sum(r["strict_pairs"] for r in sample)
        aggregate.append({"step":step,"tasks":len(sample),
                          **{k:sum(int(r[k]) for r in sample) for k in ("oracle_in_top1","oracle_in_top2","oracle_in_top4")},
                          "mean_spearman_to_final_all8":statistics.mean(corr) if corr else None,
                          "defined_spearman_tasks":len(corr),
                          "reversed_strict_pairs_to_final":sum(r["reversed_pairs_to_final"] for r in sample),
                          "strict_pairs":strict,
                          "reversed_strict_pair_fraction":sum(r["reversed_pairs_to_final"] for r in sample)/strict if strict else None,
                          "mean_current_leader_final_regret":statistics.mean(r["current_leader_final_regret"] for r in sample)})
    by_stage={str(step):sum(r["oracle_loss_step"]==step for r in detail) for step in (0,7,14)}
    gaps=[r["top1_minus_top2_terminal"] for r in detail]
    summary={**PROTOCOL,"tasks":len(tasks),"users":len({e["identity"][0] for e in tasks.values()}),
             "default":{"mean_personal":statistics.mean(r["default_personal"] for r in detail),
                        "oracle_hit_tasks":by_stage["0"],"oracle_lost_at_step7":by_stage["7"],
                        "oracle_lost_at_step14":by_stage["14"],
                        "mean_regret":statistics.mean(r["default_regret"] for r in detail),
                        "mean_stage7_loss":statistics.mean(r["stage7_terminal_best_loss"] for r in detail),
                        "mean_stage14_loss":statistics.mean(r["stage14_terminal_best_loss"] for r in detail),
                        "stage7_missed_oracle_current_rank_histogram":dict(sorted(Counter(r["oracle_rank_step7_all8"] for r in detail if r["oracle_loss_step"]==7).items())),
                        "all_exact112":all(r["particle_steps"]==112 for r in detail),
                        "all_exact14_views":all(r["decode_images"]==14 for r in detail)},
             "terminal_best_second_gap":{"mean":statistics.mean(gaps),"median":statistics.median(gaps),"zero_tie_tasks":sum(g==0 for g in gaps)},
             "resource_only":{"latest_first_drop_if_two_must_reach28":(112-2*28)//(8-2),
                              "latest_first_drop_if_one_must_reach28":(112-28)//(8-1),
                              "two_node_14view_feasible_schedules":resource_schedules()},
             "no_new_adaptive_policy_result":True}
    assert by_stage["0"]+by_stage["7"]+by_stage["14"]==len(tasks)
    assert summary["default"]["oracle_hit_tasks"]==108  # Existing default reference, not a tuning target.
    assert math.isclose(summary["default"]["mean_stage7_loss"]+summary["default"]["mean_stage14_loss"],summary["default"]["mean_regret"],abs_tol=1e-12)
    write_json(out/"summary.json",summary)
    write_csv(out/"per_step_difficulty.csv",aggregate)
    write_csv(out/"per_task_diagnosis.csv",detail)
    write_csv(out/"per_task_step_ranks.csv",rows)
    write_csv(out/"candidate_pool_audit.csv",audit)
    write_csv(out/"resource_only_schedules.csv",resource_schedules())
    plot_difficulty(out,aggregate,summary)
    assert before_hash==sha256(metrics)
    write_json(out/"manifest.json",{"input":str(metrics),"input_sha256":before_hash,"script_sha256":sha256(__file__),
               "protocol_sha256":sha256(out/"diagnostic_protocol.json"),"summary_sha256":sha256(out/"summary.json"),
               "new_policy_evaluations":0,"neural_calls":0,"training_calls":0,"threshold_or_schedule_selection":False})
    print(json.dumps(summary,indent=2))
    print(json.dumps([r for r in aggregate if r["step"] in (3,7,14,21,28)],indent=2))


if __name__=="__main__":main()
