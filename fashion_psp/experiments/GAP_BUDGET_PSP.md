# 单一冻结分差自适应PSP

这是一次机制实验：在BoN@4的112去噪粒子步预算内，从固定原始8seed保留终值Personal赢家。代码没有参数搜索入口。规则不依赖旧DP校准表或此前测试集训练的模型。

第7步比较当前第3/4名分差与第4/5名分差；前者严格更大则保留3个，到第21步保留2个，否则保留4个，到第14步保留2个。第28步按实际Personal选1。分差相同选保留4个；分数相同按seed升序。只让首次保留数与第二剪枝节点自适应，首次节点维持默认7。

`8×7 + 3×14 + 2×7 = 112`，`8×7 + 4×7 + 2×14 = 112`。解码/Personal评估分别为13、14张，不宣称端到端成本与仅评估4张图的BoN@4相同。

规则与源码在首次回放前冻结。本200任务曾被研究、也曾用于旧策略调参；本规则未搜索参数，仍只构成已研究测试子集上的探索结果。完整结果和局限见`fashion_psp/artifacts/gap_budget_psp_v1/report.md`。

## 冻结策略回放

```powershell
python -m fashion_psp.experiments.replay_gap_budget_psp
python -m fashion_psp.experiments.analyze_gap_budget_psp
```

首个命令要求已有匹配的frozen_rule.json，源码变化会拒绝继续，不静默覆盖冻结内容。没有修改规则、扫参数或自动择优逻辑。

## 在线入口

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP
python -m fashion_psp.experiments.run_dualfashion_fitb_gap_budget `
  --max-samples 2207 `
  --seed 2026 `
  --resume-from auto
```

输出隔离于`DualFashion-main/output/ifashion_fitb_gap_budget/personal_group0_seed2026`。每任务保存选中图、原始seed映射、评分次数、分差/分支、累计去噪预算、CLIP实际填充批次数和断点；Packed gen/grd仅供事后评价，GT不参与剪枝。启动完整2207任务不是本次工作的已执行结果。

控制逻辑已用全部200任务真实缓存分数验证，与冻结回放选择/剪枝/评分时点一致。每任务6、7、13、14、20、21、27、28步恢复均一致；神经回调用模拟数据结构，因此不是GPU生成精度/图像一致性的实测。PrepareOnly检查不启动模型生成。

```powershell
python -m unittest fashion_psp.experiments.test_gap_budget_psp fashion_psp.experiments.test_gap_budget_online -v
```

单次冻结测试结果只能用于报告此机制的行为；后续新假设须另外预先声明，不能根据本次点估计反复改规则直到优于PSP。
