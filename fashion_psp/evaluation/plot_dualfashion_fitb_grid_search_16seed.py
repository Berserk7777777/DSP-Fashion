"""Reproducible retrospective Personal schedule search and plots on 200 tasks.

Prefix K pools, at-most112 by default (exact mode is also available). The search
domain is frozen before evaluation: all integer intermediate steps/counts,
at most four intermediate cuts. Selection reads current live Personal only.
This is exploratory reuse of inspected test tasks, not independent validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_dualfashion_fitb_timestep_strategies import enumerate_schedules, select_task
from plot_dualfashion_fitb_personal_pruning_schedules import profile

COUNTS = [1, 2, 4, 6, 8, 12, 16]
RED, BLUE = "#d62728", "#1261c9"


def interpolate_axis(values, source, target):
    """Piecewise-linear axis mapping with endpoint extrapolation."""
    values = np.asarray(values, dtype=float)
    mapped = np.interp(values, source, target)
    mapped = np.where(values < source[0], target[0] + (values-source[0]) *
                      (target[1]-target[0])/(source[1]-source[0]), mapped)
    return np.where(values > source[-1], target[-1] + (values-source[-1]) *
                    (target[-1]-target[-2])/(source[-1]-source[-2]), mapped)


def set_particle_axis(ax):
    # Equally spaced labeled levels, not a proportional particle-count axis.
    levels = np.asarray(COUNTS, dtype=float)
    positions = np.arange(len(levels), dtype=float)
    ax.set_yscale("function", functions=(
        lambda y: interpolate_axis(y, levels, positions),
        lambda y: interpolate_axis(y, positions, levels)))
    ax.set_ylim(.88, 17.2)
    ax.set_yticks(COUNTS, labels=list(map(str, COUNTS)))


def parse_args():
    root = Path(__file__).resolve().parents[2]
    analysis = root / "DualFashion-main/output/ifashion_fitb_timestep200_16particle_analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, default=analysis / "candidate_metrics_personal_all_steps.csv")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--budget-mode", choices=("at-most","exact"), default="at-most")
    parser.add_argument("--plot-only", action="store_true", help="Reuse the saved complete grid table for rendering changes.")
    parser.add_argument("--max-cutoffs", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_data(path):
    scores = np.full((200, 28, 16), np.nan, dtype=np.float64)
    seeds = np.empty((200, 16), dtype=np.int64)
    for task in range(200):
        seeds[task] = 2026 + task*10000 + np.arange(16)
    tasks = {}
    ids = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            task, step, seed = (int(row[key]) for key in ("task_index", "step", "seed"))
            slot = seed - (2026+task*10000)
            if not (0 <= task < 200 and 1 <= step <= 28 and 0 <= slot < 16):
                raise ValueError(f"Unexpected task/step/seed: {task}/{step}/{seed}")
            if np.isfinite(scores[task,step-1,slot]):
                raise ValueError(f"Duplicate task/step/seed: {task}/{step}/{seed}")
            value = float(row["personal"])
            if not np.isfinite(value):
                raise ValueError("Nonfinite Personal")
            scores[task,step-1,slot] = value
            uid_oid = (int(row["uid"]),int(row["oid"]))
            if task in ids and ids[task] != uid_oid:
                raise ValueError(f"Task mapping changes at {task}")
            ids[task] = uid_oid
            # Only Personal and source identity/path enter the replay data.
            tasks.setdefault(task,{}).setdefault(step,{})[seed] = dict(
                task_index=task, uid=uid_oid[0], oid=uid_oid[1], step=step,
                seed=seed, personal=value, image_path=row["image_path"])
    if not np.isfinite(scores).all():
        raise ValueError("Expected exactly 200*28*16 complete scores")
    return scores, seeds, tasks


def evaluate(schedule, scores):
    k = schedule["initial_particles"]
    active = np.tile(np.arange(k), (200,1))
    task_axis = np.arange(200)[:,None]
    spent, previous, views = 0, 0, 0
    for step, keep in zip(schedule["cutoff_times"],schedule["remaining_seeds"]):
        spent += active.shape[1]*(step-previous)
        views += active.shape[1]
        current = scores[task_axis,step-1,active]
        order = np.argsort(-current, axis=1, kind="stable")[:,:keep]
        active = np.sort(np.take_along_axis(active,order,axis=1),axis=1)
        previous = step
    spent += active.shape[1]*(28-previous)
    views += active.shape[1]
    assert spent == schedule["particle_steps"]
    final = scores[task_axis,27,active]
    winner = active[np.arange(200), np.argmax(final,axis=1)]
    selected = scores[np.arange(200),27,winner]
    pool_best = np.argmax(scores[:,27,:k],axis=1)
    global_best = np.argmax(scores[:,27,:],axis=1)
    return dict(selected_personal=float(selected.mean()),
                selected_personal_std=float(selected.std()),
                personal_regret_pool=float((scores[:,27,:k].max(axis=1)-selected).mean()),
                best_seed_hit_rate_pool=float(np.mean(winner==pool_best)),
                best_seed_hit_rate_16=float(np.mean(winner==global_best)),
                reward_views=views, vae_decode_views=views,
                scoring_rounds=len(schedule["cutoff_times"])+1), winner


def signature(row):
    return row["initial_particles"], tuple(row["cutoff_times"]), tuple(row["remaining_seeds"])


def rank_key(row):
    return (-row["selected_personal"], row["reward_views"], row["initial_particles"],
            tuple(row["cutoff_times"]), tuple(row["remaining_seeds"]))


def save_csv(path, rows):
    with path.open("w",encoding="utf-8-sig",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value) if isinstance(value,list) else value for key,value in row.items()})


def display_segment(row, rng, jitter=True):
    x,y=profile(row,28)
    if jitter:
        # Display-only offsets expose coincident policies as in the reference
        # figure. Exact steps/counts and budgets remain in the source tables.
        interior=(x>0)&(x<1)
        if row["cutoff_times"] and row["cutoff_times"][0]==7:
            interior &= ~np.isclose(x,7/28)
        x[interior]+=rng.uniform(-.010,.010)
        y*=np.exp(rng.uniform(-.010,.010))
    return np.column_stack((x,y))


def make_plots(output, rows, best, first7best, default, dpi, budget_mode):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "pdf.fonttype":42,
                         "svg.fonttype":"none", "savefig.facecolor":"white"})
    rng=np.random.default_rng(20261103)
    ordered = sorted(rows,key=lambda r:r["selected_personal"])
    segments = [display_segment(row,rng) for row in ordered]
    norm = Normalize(min(r["selected_personal"] for r in rows), max(r["selected_personal"] for r in rows))
    figure, ax = plt.subplots(figsize=(12.2,7.5))
    figure.subplots_adjust(left=.085,right=.87,bottom=.19,top=.89)
    collection = LineCollection(segments,cmap="viridis",norm=norm,linewidths=.42,alpha=.28,rasterized=True)
    collection.set_array(np.array([r["selected_personal"] for r in ordered]))
    ax.add_collection(collection)
    for row,color,style,lw,z in [(first7best,BLUE,"-",2.4,5),(best,RED,"-",2.8,6),(default,"black","--",2.1,7)]:
        x,y=profile(row,28)
        ax.plot(x,y,color=color,linestyle=style,lw=lw,zorder=z)
    ax.set(xlim=(-.01,1.01),ylim=(.88,18.2),xlabel="Inference progress (%)",ylabel="Particles alive")
    set_particle_axis(ax)
    ax.set_xticks([0,.25,.5,.75,1],labels=["0","25","50","75","100"])
    ax.grid(axis="y",color="#dddddd",lw=.65)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Personal pruning schedules: initial K = 1, 2, 4, 6, 8, 12, 16",pad=14)
    handles=[Line2D([],[],color=RED,lw=2.8,label=f"Grid best: K={best['initial_particles']}, Personal={best['selected_personal']:.3f}"),
             Line2D([],[],color=BLUE,lw=2.4,label=f"Best with first cut at step 7: K={first7best['initial_particles']}, Personal={first7best['selected_personal']:.3f}"),
             Line2D([],[],color="black",ls="--",lw=2.1,label=f"Default K=6: 7:4, 21:2, 28:1; Personal={default['selected_personal']:.3f}")]
    ax.legend(handles=handles,loc="upper right",framealpha=.96,fontsize=9)
    bar=figure.colorbar(ScalarMappable(norm=norm,cmap="viridis"),ax=ax,cax=figure.add_axes([.89,.24,.017,.60]))
    bar.set_label("Mean terminal Personal (200 tasks)")
    figure.text(.085,.105,f"{len(rows):,} policies | prefix K seeds per task | all integer cut steps/counts | at most 4 intermediate cuts",fontsize=9)
    budget_text=("All policies use <=112 particle-steps; no budget padding. " if budget_mode=="at-most" else
                 "K >=4: exactly112; K=1/2: 28/56-step references. ")
    figure.text(.085,.075,budget_text+"Y levels equally spaced. Background display offsets expose overlapping curves.",fontsize=8.5)
    figure.text(.085,.045,"Retrospective search on 200 previously inspected test tasks; not independent validation. Terminal selection is implicit at step 28.",fontsize=8.5,color="#555555")
    for suffix in ("png","pdf","svg"):
        figure.savefig(output/f"pruning_grid_16seed.{suffix}",dpi=dpi)
    plt.close(figure)
    del collection,segments,ordered

    # Per-K panels keep optimal policies readable even when the full grid overlaps.
    figure, axes = plt.subplots(2,4,figsize=(14,7.5),sharex=True,sharey=True)
    for ax,k in zip(axes.flat,COUNTS):
        subset = [r for r in rows if r["initial_particles"]==k]
        local = min(subset,key=rank_key)
        clouds = LineCollection([display_segment(r,rng) for r in subset],colors="#999999",lw=.45,alpha=.10,rasterized=True)
        ax.add_collection(clouds)
        first7 = [r for r in subset if r["cutoff_times"] and r["cutoff_times"][0]==7]
        if first7:
            b = min(first7,key=rank_key)
            x,y=profile(b,28);ax.plot(x,y,color=BLUE,lw=2.1)
        x,y=profile(local,28);ax.plot(x,y,color=RED,lw=2)
        x,y=profile(default,28);ax.plot(x,y,"--",color="black",lw=1.5,alpha=.8)
        noun="policy" if len(subset)==1 else "policies"
        ax.set_title(f"K={k} | best Personal {local['selected_personal']:.3f}\n{len(subset):,} {noun}",fontsize=10)
        ax.set(xlim=(0,1),ylim=(.85,18.2))
        set_particle_axis(ax)
        ax.set_xticks([0,.25,.5,.75,1],labels=["0","25","50","75","100"])
        ax.grid(axis="y",alpha=.25);ax.spines[["top","right"]].set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=[Line2D([],[],color=RED,lw=2,label="Best for this K"),
                              Line2D([],[],color=BLUE,lw=2,label="Best with first cut at step 7 (per K)"),
                              Line2D([],[],color="black",ls="--",lw=1.5,label="Default six-seed schedule")],loc="upper left",bbox_to_anchor=(.02,.9),fontsize=9)
    note=("Budget <=112 particle-steps.\nK=1/2 cannot consume all112.\nK=16 cannot wait until step7." if budget_mode=="at-most" else
          "K=1/2 use 28/56 steps.\nK=4 has no cut at exact112.\nK=16 cannot wait until step7.")
    axes.flat[-1].text(.05,.20,note,transform=axes.flat[-1].transAxes,fontsize=9)
    figure.supxlabel("Inference progress (%)",y=.08);figure.supylabel("Particles alive")
    figure.suptitle("Schedule grid by initial candidate count",y=.99,fontsize=14)
    figure.text(.5,.02,"Retrospective test-subset search; prefix candidate pools; at most four cuts. Y labels use equally spaced levels.",ha="center",fontsize=9,color="#555555")
    figure.tight_layout(rect=(.025,.10,1,.95))
    for suffix in ("png","pdf"):
        figure.savefig(output/f"pruning_grid_by_k.{suffix}",dpi=dpi)
    plt.close(figure)


def main():
    args=parse_args()
    if args.max_cutoffs!=4:
        raise ValueError("This figure freezes the existing max-four-cut search domain")
    analysis=args.metrics_csv.parent
    output=(args.output_root or analysis/f"grid_search/prefix_kset_{'atmost' if args.budget_mode=='at-most' else 'exact'}112_first7").resolve()
    output.mkdir(parents=True,exist_ok=True)
    scores,seeds,tasks=load_data(args.metrics_csv)
    rows=[]
    if args.plot_only:
        old=json.loads((output/"grid_search_manifest.json").read_text(encoding="utf-8"))
        assert old["source_sha256"]==hashlib.sha256(args.metrics_csv.read_bytes()).hexdigest()
        assert old["budget_mode"]==args.budget_mode
        with (output/"grid_search_summary.csv").open(encoding="utf-8-sig",newline="") as handle:
            for raw in csv.DictReader(handle):
                row={}
                for key,value in raw.items():
                    if key in ("cutoff_times","remaining_seeds"): row[key]=json.loads(value)
                    elif key in ("initial_particles","particle_steps","budget","reward_views","vae_decode_views","scoring_rounds"): row[key]=int(value)
                    elif key=="budget_status":row[key]=value
                    else:row[key]=float(value)
                rows.append(row)
        assert len(rows)==old["total_policies"]
    else:
        for k in COUNTS:
            schedules = ([dict(initial_particles=k,cutoff_times=[],remaining_seeds=[],particle_steps=28*k,budget=112)]
                         if args.budget_mode=="exact" and k<4 else
                         enumerate_schedules(k,28,112,args.budget_mode,4,list(range(1,28)),list(range(1,16))))
            for index,schedule in enumerate(schedules,1):
                metrics,_=evaluate(schedule,scores)
                status="exact112" if schedule["particle_steps"]==112 else "under_budget"
                rows.append(dict(**schedule,**metrics,budget_status=status))
                if index%20000==0:
                    print(f"K={k}: {index:,}/{len(schedules):,}",flush=True)
            print(f"K={k}: evaluated {len(schedules):,} schedules",flush=True)
    rows.sort(key=rank_key)
    best=min((r for r in rows if args.budget_mode=="at-most" or r["budget_status"]=="exact112"),key=rank_key)
    first7=min((r for r in rows if r["cutoff_times"] and r["cutoff_times"][0]==7),key=rank_key)
    default=next(r for r in rows if signature(r)==(6,(7,21),(4,2)))
    chosen=[]
    for k in COUNTS:
        subset=[r for r in rows if r["initial_particles"]==k]
        chosen.append(dict(role="best_by_k",**min(subset,key=rank_key)))
        first7_subset=[r for r in subset if r["cutoff_times"] and r["cutoff_times"][0]==7]
        if first7_subset:chosen.append(dict(role="best_first7_by_k",**min(first7_subset,key=rank_key)))
    chosen.extend([dict(role="global_best",**best),dict(role="global_best_first7",**first7),dict(role="default_six_seed",**default)])
    selections=[]
    # Cross-check the vector implementation with the existing scalar causal
    # replay on every task for every displayed highlighted policy.
    for policy in chosen:
        _,winners=evaluate(policy,scores)
        for task,by_step in tasks.items():
            window=seeds[task,:policy["initial_particles"]].tolist()
            selected,trace=select_task(by_step,policy,"personal",28,window)
            assert selected["seed"]==int(seeds[task,winners[task]])
            selections.append(dict(role=policy["role"],initial_particles=policy["initial_particles"],
                                   task_index=task,uid=selected["uid"],oid=selected["oid"],
                                   selected_seed=selected["seed"],personal=selected["personal"],
                                   image_path=selected["image_path"]))
    save_csv(output/"grid_search_summary.csv",rows)
    save_csv(output/"highlighted_strategies.csv",chosen)
    save_csv(output/"highlighted_task_selections.csv",selections)
    manifest=dict(source=str(args.metrics_csv.resolve()),source_sha256=hashlib.sha256(args.metrics_csv.read_bytes()).hexdigest(),
        tasks=200,steps=28,seeds_per_task=16,rows=200*28*16,initial_counts=COUNTS,
        seed_pool_policy="For each task K uses original slots0..K-1, not partitioned-window averages",
        seed_mapping="2026 + task_index*10000 + original_slot",cutoff_domain=list(range(1,28)),
        keep_domain="all positive integer counts strictly less than current population",max_intermediate_cuts=4,
        budget_mode=args.budget_mode,
        budget="All schedules <=112; no padding" if args.budget_mode=="at-most" else "K>=4 exactly112; K=1/2 unpruned28/56 references",
        plot_display="Fixed RNG20261103; background cut-time offset <=0.01 horizon and population factor exp(+-0.01), purely for visibility. Initial/terminal times, first-cut7 positions and all highlighted policies stay exact. No effect on replay metrics/budget.",
        plot_highlights="Red: global best. Blue (#1261c9): only the best policy whose first actual cut is step7 (per-K best in per-K panels). Black dashed: default six-seed schedule. Other first7 policies retain the background Personal color map.",
        plot_y_axis="Labeled particle levels 1,2,4,6,8,12,16 are equally spaced. Other counts are interpolated between labeled levels. This display axis is not proportional to count and must not be used to infer compute from plotted area.",
        objective="Highest mean terminal Personal; deterministic ties: fewer reward views, lower K, lexicographic schedule",
        total_policies=len(rows),first7_policies=sum(bool(r["cutoff_times"]) and r["cutoff_times"][0]==7 for r in rows),
        grid_best=best,best_first7=first7,default_six_seed=default,
        causal_replay="Only live candidates at reached steps determine task selections. All terminal scores are used solely for retrospective policy comparison and oracle metrics.",
        evaluation_role="Retrospective exploratory grid on previously inspected test subset; no independent validation or deployment-policy claim",
        verification="All 200 tasks for every highlighted policy agree exactly with the existing scalar select_task replay",
        command=[sys.executable,*sys.argv])
    (output/"grid_search_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    report = ["# Retrospective 16-seed Personal schedule grid", "",
        "This is an exploratory search on 200 previously inspected test tasks, not independent validation.", "",
        f"Each task uses its original first K seeds. Budget mode: {args.budget_mode}, upper limit112 particle-steps, no artificial padding.",
        "All integer cutoff steps1..27 and survivor counts1..15 are searched, with strictly decreasing populations and at most4 intermediate cuts.",
        "The red highlight maximizes mean terminal Personal within this declared grid. Blue highlights only the best policy whose first actual cut is step7 (the per-K best in each per-K panel). Other first7 policies retain the background Personal color map. Black dashes show the default6->4->2 schedule.",
        "Y-axis labels1,2,4,6,8,12,16 use equal spacing for display; intermediate counts are interpolated between these levels. Plotted area does not represent compute. Actual particle-step budgets remain unchanged in the CSV.",
        "Terminal selection at step28 is implicit; profiles show particle counts during denoising. Rewards/decodes are reported separately from particle-steps.", "",
        "Background curves have small fixed-seed display offsets to expose overlap, as in the supplied reference. Highlighted policies use exact coordinates. Actual times/counts/budgets are unchanged in the CSV.","",
        "| Role | K | Cutoffs | Keep counts | Personal | Pool best-seed hit | Reward views |", "|---|---:|---|---|---:|---:|---:|"]
    for role,row in [("Grid best",best),("Best first7",first7),("Default six-seed",default)]:
        report.append(f"| {role} | {row['initial_particles']} | {row['cutoff_times']} | {row['remaining_seeds']} | {row['selected_personal']:.6f} | {row['best_seed_hit_rate_pool']:.1%} | {row['reward_views']} |")
    report += ["", "Pool-best hit rates for different K use different oracle candidate pools; see best_seed_hit_rate_16 for a common16-seed oracle.",
        "", "Reproduce from the workspace:", "", "```powershell",
        "powershell -ExecutionPolicy Bypass -File .\\fashion_psp\\evaluation\\run_dualfashion_fitb_grid_search_16seed.ps1", "```", "",
        "Artifacts: grid_search_summary.csv (all policies), highlighted_strategies.csv, highlighted_task_selections.csv, grid_search_manifest.json, pruning_grid_16seed.png/pdf/svg, pruning_grid_by_k.png/pdf."]
    (output/"README.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("Saved results; rendering dense plots...",flush=True)
    make_plots(output,rows,best,first7,default,args.dpi,args.budget_mode)
    print(json.dumps({"output":str(output),"total_policies":len(rows),"best":best,"best_first7":first7,"default":default},indent=2))


if __name__=="__main__":
    main()
