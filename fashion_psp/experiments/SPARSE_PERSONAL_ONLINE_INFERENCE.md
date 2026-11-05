# 稀疏评估自适应 Personal PSP 在线推理

2026-10-25 更正：用户明确该200任务属于测试集的一部分，不允许调参。本页v3来自此前在这些任务上的参数搜索，只保留为历史探索；60%命中率不是未经测试调参的新方法证据。请遵循`PERSONAL_RESEARCH_PROTOCOL.md`，本页命令不作为新确认实验建议。

该入口默认使用已固定的 sparse_personal_v3 参数，维持原始 8 个 seed、28 步、112 粒子步；未评分步骤不会调用 VAE 解码或 Personal reward。它复用原在线入口的神经网络、精度和任务映射实现。输出目录与逐步评估 v2 隔离。

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP

powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_fitb_sparse_personal.ps1 `
  -MaxSamples 2207 `
  -Seed 2026 `
  -ResumeFrom auto
```

输出：`E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_sparse_personal\personal_group0_seed2026`。

启动器默认 Python：`D:\anaconda\envs\dualfashion\python.exe`。保留同一命令再次运行即可继续未完成的任务/步骤；不要把 v2 的输出路径传给此入口。配置/策略不一致时会拒绝混用。每个任务保存选中图片、seed、Personal、去噪预算、解码/reward 候选数、评分轮数、CLIP 填充批次数和逐步轨迹；`0057501/gen.npy`、`grd.npy` 供原有事后评估使用，GT 不参与剪枝。

200 开发任务真实缓存回放：Personal=71.844867535，最优 seed 命中=120/200，平均解码/Personal 候选评估=13.32 张，平均评分轮数=3.125；原默认 PSP 分别为 71.560855560、108/200、14 张、3 轮。两者每任务都恰好 112 粒子步。这些是开发集调参结果；2207 任务的实际得分、解码次数和耗时要以在线执行结果为准。

已经执行 2207 任务 PrepareOnly 检查和全200任务模拟神经回调的在线控制一致性/断点恢复测试；尚未执行新的 2207 任务 GPU 去噪。完整结果、Compatibility 代价和固定调度补充对照见 `fashion_psp/artifacts/sparse_personal_development_v3/report.md`。

## 复现冻结策略回放

```powershell
python -m fashion_psp.experiments.search_sparse_personal `
  --policy fashion_psp/artifacts/sparse_personal_development_v3/selected_policy.json `
  --output-dir fashion_psp/artifacts/sparse_personal_replay
```

复现开发集全局搜索（会重新评估固定的 5,314 个配置，不修改搜索空间）：

```powershell
python -m fashion_psp.experiments.search_sparse_personal `
  --output-dir fashion_psp/artifacts/sparse_personal_search_reproduced
```

检查：

```powershell
python -m unittest fashion_psp.experiments.test_sparse_personal -v
```

回放仅读取原有 CSV 和缓存 metadata/图片存在性，不重新解码或推理。搜索仅选择一个全局策略；在线入口仅加载冻结参数，不加载候选缓存分数、终值表或旧 DP 风险表。
