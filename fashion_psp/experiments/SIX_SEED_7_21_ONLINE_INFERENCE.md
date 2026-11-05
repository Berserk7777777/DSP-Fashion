# FITB全任务在线推理：6→4→2→1，Personal固定剪枝

初始化每个任务原始槽0..5共6个候选，完成第7步后按当前Personal保留4个，完成第21步后保留2个，第28步按实际终值选1。分数相同保留原始seed较小者。只在7/21/28评分，不读取缓存轨迹或旧DP经验表。

去噪预算：`6*7+4*14+2*7=112`粒子步。解码/Personal评估：`6+4+2=12`张，3轮；CLIP按4补齐：4批、16槽。端到端成本与仅4张终点评分的BoN@4不同。

## PowerShell启动全2207任务

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP

powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_fitb_six_seed_7_21.ps1 `
  -MaxSamples 2207 `
  -Seed 2026 `
  -ResumeFrom auto
```

默认解释器`D:\anaconda\envs\dualfashion\python.exe`，数据根目录`C:\tmp\ifashion`，生成模型`0057501`，生成/reward设备均为`cuda:0`，启用CPU offload。可通过`-PythonExe`、`-DataRoot`、`-Device`、`-RewardDevice`和`-OutputRoot`修改对应运行位置/设备。上述PowerShell脚本直接调用指定环境的解释器。

输出独立目录：

```text
E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_six_seed_7_21\personal_group0_seed2026
```

每任务保存`selected.png`和`task_summary.json`（seed、Personal、28步剪枝轨迹、预算、评分图数/轮数、CLIP补齐批次及耗时）；`resume_state.pt`用于未完成任务的逐步恢复，完成后删除。根目录`resume_state.json`记录完成索引；`0057501/gen.npy`和`grd.npy`可接既有四指标评估器，GT只用于事后打包/评估，未用于Personal剪枝。

中断后执行同一命令，自动跳过完成任务并恢复任务内断点。不同策略或配置不能复用同一输出目录，入口会拒绝配置不匹配。

## seed映射与任务身份

默认任务索引t的6个seed为`2026+t*10000+[0..5]`，保持原始8seed中的前6槽，实际仅初始化6个latent。不能把9248_17906的简称2029当作它在所有任务划分中固定的完整seed；以当前FITB任务的索引与保存的uid/oid/seed映射为准。完整2207任务的排列不能假设与旧200任务缓存索引一一相同。

## Python入口及只检查配置

```powershell
python -m fashion_psp.experiments.run_dualfashion_fitb_six_seed_7_21 `
  --max-samples 2207 `
  --seed 2026 `
  --seed-stride 10000 `
  --resume-from auto
```

PowerShell命令添加`-PrepareOnly`可核验策略、数据/上下文路径及输出配置，不加载生成模型。新策略及在线控制源码哈希保存于`fashion_psp/artifacts/fixed_six_seed_7_21/frozen_rule.json`，不匹配则拒绝启动。

## 已执行验证

2026-10-29：通过新PowerShell入口完成全部2207任务PrepareOnly，打印K=6；与200任务已保存固定6/7/21回放逐任务选择、剪枝过程及评分访问一致，模拟回调恢复检查1800次；另有10组合成轨迹的280个逐步恢复检查。新测试及既有入口回归共19项通过。已完成的8seed在线结果目录也通过原配置PrepareOnly兼容性检查。

这些控制验证没有启动新的GPU生成或2207任务长推理，不等于神经输出与旧缓存逐像素一致的验证。200任务测试用于核对既有固定规则的实现，不用于继续选择参数；之前的探索指标不能作为完整在线2207任务实测得分。
