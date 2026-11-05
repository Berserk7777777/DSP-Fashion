"""ONE preregistered gap-budget rule; no search/fit/tune path or policy options."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

from fashion_psp.search.gap_budget_psp import RULE, replay as adaptive_replay
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import (
    CausalObserver,load_and_audit,posthoc,replay as default_replay,sha256,summarize,write_csv,write_json,
)

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/"fashion_psp/artifacts/gap_budget_psp_v1"


def frozen_document():
    return {"rule":RULE,
            "rule_source_sha256":sha256(ROOT/"fashion_psp/search/gap_budget_psp.py"),
            "replay_source_sha256":sha256(__file__),
            "dataset_role":"previously_inspected_test_subset_one_frozen_rule_no_new_tuning",
            "acceptance_reporting":"Report Personal, same-original8 hit and regret, win/tie/loss vs default and first4 BoN, all outcomes including deterioration.",
            "no_feedback_revision":True}


def bon4(seeds,observe):
    # Same original8 pool; the comparator's candidate pool is explicitly slots0..3.
    active=sorted(seeds)[:4];trace=[]
    for step in range(1,29):
        before=active.copy();scores=observe(step,before,step==28)
        if step==28:
            active=[min(before,key=lambda s:(-scores[s],s))]
        trace.append({"step":step,"alive_before":before,"alive_after":active.copy(),
                      "count_before":len(before),"count_after":len(active),"observed":step==28,
                      "personal_scores":scores,"removed_reasons":{str(s):"terminal_selection" for s in before if s not in active},
                      "cumulative_particle_steps":4*step,"cumulative_decode_images":4 if step==28 else 0})
    return {"selected_seed":active[0],"particle_steps":112,"decode_images":4,
            "personal_evaluations":4,"scoring_rounds":1,"trace":trace}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    p.add_argument("--freeze-only",action="store_true")
    args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    freeze=out/"frozen_rule.json"
    document=frozen_document()
    if args.freeze_only:
        if freeze.exists() and json.loads(freeze.read_text(encoding="utf-8"))!=document:
            raise ValueError("Existing freeze differs; do not silently replace it")
        write_json(freeze,document)
        print(f"Frozen without opening any trajectory data: {freeze}")
        return
    if not freeze.is_file() or json.loads(freeze.read_text(encoding="utf-8"))!=document:
        raise ValueError("Freeze matching source before running; do not revise from test outcomes")
    metrics=ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv"
    cache=ROOT/"DualFashion-main/output/ifashion_fitb_timestep200_cache"
    input_hash=sha256(metrics)
    tasks,audit=load_and_audit(metrics,cache,200)
    selections=[];traces=[];checks=[]
    for task,e in sorted(tasks.items()):
        for method,runner in (("default",lambda s,o:default_replay(s,"default",o)),
                              ("adaptive",adaptive_replay),("bon4_first4",bon4)):
            observer=CausalObserver(e["personal"],e["seeds"])
            result=runner(e["seeds"],observer)
            assert len(observer.accesses)==result["personal_evaluations"]
            if method=="adaptive":
                used=set(observer.accesses)
                poison={k:v if k in used else float("nan") for k,v in e["personal"].items()}
                assert result==adaptive_replay(e["seeds"],CausalObserver(poison,e["seeds"]))
                checks.append({"task_index":task,"exact112":result["particle_steps"]==112,
                               "views13_or14":result["decode_images"] in (13,14),"no_unobserved_score_influence":True})
            traces.extend({"method":method,"task_index":task,**r} for r in result.pop("trace"))
            selections.append({"method":method,"task_index":task,"uid":e["identity"][0],"oid":e["identity"][1],**result})
    write_csv(out/"selection_pre_posthoc.csv",selections)
    with (out/"step_trace.jsonl").open("w",encoding="utf-8") as f:
        for r in traces:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n")
    # All comparisons and oracle labels only after choices have been frozen.
    evaluated=posthoc(metrics,tasks,selections)
    two=[r for r in evaluated if r["method"] in ("default","adaptive")]
    summary,pairs=summarize(two,[r for r in traces if r["method"] in ("default","adaptive")])
    summary["dataset_role"]=document["dataset_role"]
    base=[r for r in evaluated if r["method"]=="bon4_first4"]
    summary["methods"]["bon4_first4"]={
        **{f"mean_{k}":statistics.mean(r[k] for r in base) for k in ("personal","compatibility","personal_regret","optimal_seed_hit")},
        "particle_steps_per_task_min":112,"particle_steps_per_task_max":112,"all_tasks_exact112":True,
        "mean_decode_images":4,"mean_scoring_rounds":1,"total_particle_steps":22400,
        "total_decode_images":800,"total_personal_evaluations":800,"total_scoring_rounds":200}
    adapt=[r for r in evaluated if r["method"]=="adaptive"]
    bp=[]
    for a,b in zip(adapt,base):
        assert a["task_index"]==b["task_index"]
        d=a["personal"]-b["personal"]
        bp.append({"task_index":a["task_index"],"adaptive_seed":a["selected_seed"],"bon4_seed":b["selected_seed"],
                   "personal_delta":d,"outcome":"win" if d>0 else "loss" if d<0 else "tie"})
    summary["adaptive_vs_bon4_first4"]={"wins":sum(r["outcome"]=="win" for r in bp),
        "ties":sum(r["outcome"]=="tie" for r in bp),"losses":sum(r["outcome"]=="loss" for r in bp),
        "mean_personal_delta":statistics.mean(r["personal_delta"] for r in bp)}
    branch_rows=[r for r in traces if r["method"]=="adaptive" and r["step"]==7]
    summary["branches"]={str(k):sum(r["count_after"]==k for r in branch_rows) for k in (3,4)}
    stage={7:0,14:0,21:0,0:0};stage_losses=[]
    for a in adapt:
        e=tasks[a["task_index"]];oracle=set(a["oracle_seeds"]);lost=0;previous=a["oracle_personal"]
        for r in traces:
            if r["method"]!="adaptive" or r["task_index"]!=a["task_index"] or r["count_after"]==r["count_before"]:
                continue
            best=max(e["personal"][(28,s)] for s in r["alive_after"])
            stage_losses.append({"task_index":a["task_index"],"step":r["step"],"loss_of_best_terminal":previous-best})
            previous=best
            if not lost and not oracle.intersection(r["alive_after"]):lost=r["step"]
        stage[lost]+=1
    summary["adaptive_oracle_first_loss_stage"]=stage
    summary["new_policies_evaluated"]=1
    summary["parameter_search_runs"]=0
    summary["compute_scope"]="112 denoising particle steps; NOT equal end-to-end compute to BoN4; decoder/reward separately counted"
    for method in ("default","adaptive","bon4_first4"):
        mt=[r for r in traces if r["method"]==method and r["observed"]]
        batches=sum((r["count_before"]+3)//4 for r in mt)
        summary["methods"][method]["logical_clip_batches_if_padded4"]=batches
        summary["methods"][method]["mean_clip_image_slots_if_padded4"]=4*batches/200
    write_json(out/"summary.json",summary)
    write_csv(out/"selections.csv",evaluated);write_csv(out/"adaptive_vs_default.csv",pairs)
    write_csv(out/"adaptive_vs_bon4.csv",bp);write_csv(out/"validation_results.csv",checks)
    write_csv(out/"branch_decisions.csv",branch_rows);write_csv(out/"stage_terminal_losses.csv",stage_losses)
    write_csv(out/"pruning_events.csv",[r for r in traces if r["count_after"]<r["count_before"]])
    write_csv(out/"candidate_pool_audit.csv",audit)
    write_csv(out/"budget_checks.csv",[{k:r[k] for k in ("task_index","method","particle_steps","decode_images","personal_evaluations","scoring_rounds")} for r in evaluated])
    assert input_hash==sha256(metrics)
    write_json(out/"manifest.json",{"frozen_rule_sha256":sha256(freeze),"input_sha256":input_hash,
        "trace_sha256":sha256(out/"step_trace.jsonl"),"selection_sha256":sha256(out/"selection_pre_posthoc.csv"),
        "new_policy_candidates":1,"parameter_search_runs":0,"future_poisoning_checks":200,"neural_calls":0})
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
