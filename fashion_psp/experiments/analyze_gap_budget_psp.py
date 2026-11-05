"""Posthoc reporting only for the already frozen single-rule experiment."""
import csv
import json
from pathlib import Path
import random
import statistics

from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import write_json

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"fashion_psp/artifacts/gap_budget_psp_v1"


def main():
    summary=json.loads((OUT/"summary.json").read_text())
    with (OUT/"selections.csv").open(encoding="utf-8-sig",newline="") as f:
        records=list(csv.DictReader(f))
    by={}
    for r in records:by.setdefault(int(r["task_index"]),{})[r["method"]]=r
    groups={}
    for _,m in sorted(by.items()):groups.setdefault(int(m["adaptive"]["uid"]),[]).append(m)
    transitions={}
    intervals={}
    for reference in ("default","bon4_first4"):
        both=onlya=onlyb=neither=0
        for methods in by.values():
            a=int(methods["adaptive"]["optimal_seed_hit"]);b=int(methods[reference]["optimal_seed_hit"])
            both+=int(a and b);onlya+=int(a and not b);onlyb+=int(b and not a);neither+=int(not a and not b)
        transitions[reference]={"both_hit":both,"adaptive_only_hit":onlya,"reference_only_hit":onlyb,"neither_hit":neither}
        # Uncertainty description only; no rule/model/threshold selection.
        users=list(groups);rng=random.Random(20261025);pd=[];hd=[]
        stats=[]
        for uid in users:
            ms=groups[uid]
            stats.append((len(ms),sum(float(m["adaptive"]["personal"])-float(m[reference]["personal"]) for m in ms),
                          sum(int(m["adaptive"]["optimal_seed_hit"])-int(m[reference]["optimal_seed_hit"]) for m in ms)))
        for _ in range(10000):
            chosen=rng.choices(stats,k=len(users));n=sum(s[0] for s in chosen)
            pd.append(sum(s[1] for s in chosen)/n);hd.append(sum(s[2] for s in chosen)/n)
        pd.sort();hd.sort()
        intervals[reference]={"personal_delta_percentile95":[pd[249],pd[9749]],
                              "hit_fraction_delta_percentile95":[hd[249],hd[9749]]}
    evidence={"oracle_hit_transitions":transitions,"exploratory_user_cluster_bootstrap":intervals,
              "users":len(groups),"resamples":10000,"random_seed":20261025,
              "caution":"Intervals do not remove prior test inspection or establish independent confirmation."}
    write_json(OUT/"posthoc_analysis.json",evidence)
    methods=summary["methods"]
    lines=["# 单一冻结规则：分差自适应PSP机制实验","",
           "主计算口径按最新要求为每任务112去噪粒子步，对照BoN@4=4×28。解码和reward另列；不宣称端到端总FLOPs或耗时与BoN相同。", "",
           "本次只评估一个新规则，先保存规则和源码哈希，再读取轨迹回放；没有搜索分差阈值、节点或参数，结果之后没有修订规则。但200任务此前已被分析和用于旧策略开发，因此本次仍是已研究测试子集上的探索性机制结果，不是独立确认。","",
           "## 冻结规则","",
           "第7步观察原始8个存活候选的Personal，按分数降序/seed升序排序。设g3=r3−r4，g4=r4−r5。若g3>g4，保留前3个，并在第21步保留前2个；否则保留前4个，并在第14步保留前2个。第28步用实际终值选1个。相等时留4个。", "",
           "两个时间均由预算恒等式推出：8×7 + k×(t−7) + 2×(28−t)=112；k=3得到t=21，k=4得到t=14。首次节点保持原PSP的7，候选数/第二节点自适应；不声称首次节点也自适应。分差是启发式，不是置信度或未来反超概率。", "",
           "| 方法 | 平均Personal | 同原始8seed的最优命中 | 平均regret | Compatibility | 每任务解码/评分图 | 评分轮 |","|---|---:|---:|---:|---:|---:|---:|"]
    for method,label in (("bon4_first4","BoN@4，同池前4seed"),("default","默认PSP"),("adaptive","冻结分差自适应")):
        m=methods[method]
        lines.append(f"|{label}|{m['mean_personal']:.9f}|{m['mean_optimal_seed_hit']:.1%}|{m['mean_personal_regret']:.9f}|{m['mean_compatibility']:.9f}|{m['mean_decode_images']:.3f}|{m['mean_scoring_rounds']:.0f}|")
    for ref,key in (("默认PSP","adaptive_vs_default"),("BoN@4","adaptive_vs_bon4_first4")):
        d=summary[key]
        lines.extend(["",f"相对{ref}：Personal差{d['mean_personal_delta']:+.9f}，逐任务胜/平/负{d['wins']}/{d['ties']}/{d['losses']}。"])
    lines.extend(["","## 机制与限制","",
        "95个任务选择7→3、21→2、28→1，105个任务沿用7→4、14→2、28→1。所有600条方法/任务记录均为112粒子步；自适应每任务13或14张候选图，合计2705张/600轮。相同CLIP batch=4填充口径下，自适应与默认均为800个批次/3200个图像槽；BoN为200批次/800槽。这是调用计数，不是实测GPU时间。", "",
        "自适应第7步丢失oracle68次，默认62次：更激进的首次剪枝多损失6个赢家。自适应后续丢失19次（14步18次、21步1次），默认第二次丢失30次，因此最终净多命中5次。这个规则没有解决早期评分不可靠的问题，提供的是某些任务把第二次判断推迟的资源重分配机制。", "",
        "命中转移（相对默认）："+json.dumps(transitions["default"],ensure_ascii=False)+"。", "",
        "## 不确定性","",
        "按108个用户成组重采样10000次的探索性百分位95%区间：", ""])
    for ref in ("default","bon4_first4"):
        e=intervals[ref]
        lines.append(f"- 相对{ref}，Personal差区间{e['personal_delta_percentile95']}；命中率差区间{e['hit_fraction_delta_percentile95']}。")
    lines.extend(["","这些区间不能消除既有测试数据研究带来的偏差，不应把点估计上升写成已证明显著提升。未据结果继续挑规则；后续新假设应另行预先声明并采用独立确认数据。", "",
        "## 因果与复现","",
        "固定原始8seed/任务映射，不混入extra8。只访问存活候选在第7步、分支第二节点、28步的Personal；没有访问被剪未来或跳过评分的中间分数。全部200任务未访问值替换NaN后轨迹不变。选择在任何事后oracle/Compatibility分析前写出。未使用旧DP表、训练预测器、GT/CIS/LPIPS。", "",
        "BoN@4明确使用原池槽0..3，并不是历史seed42的独立BoN结果。它只能命中自己池内出现的原池oracle；不更换有利的4seed子集。BoN@8作为终值oracle需要224粒子步，不是等预算比较方法。", "",
        "结果文件：frozen_rule.json（预先冻结）；selections.csv（逐任务seed/路径/终值）；branch_decisions.csv（分差和动作）；pruning_events.csv与step_trace.jsonl（剪枝原因及累计预算）；budget_checks.csv、validation_results.csv与manifest.json（资源和因果审计）；posthoc_analysis.json（命中转移和探索性区间）。", "",
        "复现冻结回放：`python -m fashion_psp.experiments.replay_gap_budget_psp`。不要在新结果上修改冻结文件或规则以追求更高数值。", "",
        "在线实现已接入共享神经推理入口，全部200任务的模拟神经回调与冻结回放选择一致；每任务8个断点（6、7、13、14、20、21、27、28）恢复一致，共1600次恢复检查。新的控制测试及原密集/稀疏入口回归共9项通过。2207任务PrepareOnly已通过，没有启动完整GPU生成，也尚未验证新入口GPU图像与旧缓存的数值一致性。在线入口说明见`fashion_psp/experiments/GAP_BUDGET_PSP.md`。", ""])
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(evidence,indent=2))


if __name__=="__main__":main()
