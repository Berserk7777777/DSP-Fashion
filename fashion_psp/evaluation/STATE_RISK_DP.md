# DualFashion FITB 状态条件化经验风险 DP

规划及研究评估使用已有的 200 任务 × 16 seeds × 28 steps 图像缓存。另提供全量 FITB 在线入口 `fashion_psp/experiments/run_dualfashion_fitb_state_dp.ps1`，使用冻结规划表做真实采样，不读取离线候选图像或分数。

## 决策机制

在 step `t`，存活集合为 `A`，剩余去噪预算为 `b`。按当前 Personal 分数排序，动作 `m` 保留前 `m` 个粒子。规划数据中的单次损失为：

```text
loss(t,A,m) = max_{i in A} Personal_i(28)
              - max_{i in Top_m(Personal(t),A)} Personal_i(28)
```

该损失在连续剪枝上望远镜求和，等于初始池最佳终值与最终存活集合最佳终值之差。终值仅用于规划用户的统计表构建；回放中的当前任务决策不读取未来分数。

核心状态为当前 top1/top2 标准化分差、当前分数标准差相对规划中位数的大小，共 4 个 bin。每个动作再观察保留边界分差、第一名到第一个被移除者的分差，共 4 个 bin。分差阈值分别为 0.4、0.25、1。估计即时损失及下一步核心状态转移，按 12 个任务等效观测做分层收缩，缓解稀疏 bin。默认用随机候选集合、默认 PSP 的实际存活集合及一轮策略反馈得到的实际存活集合构建统计表。

Bellman 状态为 `(t, n, b, core_state)`：

```text
V(t,n,b,z) = min_m [ estimated_loss(t,n,m,z)
                    + E V(t+1,m,b-m,z_next) ]
```

当前动作还用观测到的动作 bin 修正即时风险及转移。它是粗粒度经验 MDP 的近似规划，不是真实条件风险的精确最优解，也不提供置信覆盖保证。当前版本仍按当前 Personal 排名保留 top-m；它自适应决定“何时剪、保留几个”，没有直接预测每个粒子的终值排序。

每任务预算严格为 112 个去噪粒子步；steps 1–27 均可决策，step 28 按 Personal 选最终结果。剪枝次数不设固定值。支持从现有缓存的 4–16 个初始粒子启动；主对照固定为相同的前 8 个 seeds。

## 复现与恢复

### 全部 2207 个 FITB 任务在线推理

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP
powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_fitb_state_dp.ps1 -MaxSamples 2207 -Seed 2026 -ResumeFrom auto
```

默认加载 `planning_statistics_all_tasks.npz`（在原 200 任务上拟合的固定表），并非交叉用户评估时的折内表。初始 K=8，28 steps，严格112粒子步，seed 为 `2026 + task_index*10000 + slot`。denoising 和 VAE 解码均为单粒子串行；CPU offload 在生成器与 reward 之间切换显存；不生成文本。

Personal 使用原始8粒子缓存评分时的 float32、batch4、cuDNN TF32 设置；存活粒子不足4时复制补齐后仅取真实候选得分。此设备上同一 PNG 用 batch1 和 batch4 得分可能相差约0.06，因此在线不随存活粒子数改变 reward 批形状。该额外计算不计入去噪112粒子步。reward 图像的 uint8 量化也沿用原 timestep 缓存保存 PNG 的舍入规则。

输出为 `DualFashion-main/output/ifashion_fitb_state_dp/personal_group0_seed2026/`：`tasks/taskXXXX_uid_oid/selected.png`、`task_summary.json`，以及供评估使用的 `0057501/gen.npy`、`grd.npy`。不保存全量中间图像。终端显示总任务数、ETA 和当前 step/存活数/已用预算。

默认 `-ResumeFrom auto` 跳过已完成任务，并从未完成任务上一次完整 step 的 `resume_state.pt` 继续；每任务结束自动删除其 step 状态以节省空间。配置或规划表变化时拒绝混用旧目录。可用 `-ResumeFrom 1000` 指定起始任务下标（从0开始），或 `-PrepareOnly` 只检查路径和配置。每个完成任务原子更新评估 payload，若提交任务后中断，会从任务记录补齐 payload。

### 离线研究回放及四项指标

Windows PowerShell，在项目根目录运行：

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP
powershell -ExecutionPolicy Bypass -File .\fashion_psp\evaluation\run_dualfashion_state_risk_dp.ps1 -BatchSize 4
```

已有冻结选图、只继续四项指标评估：

```powershell
powershell -ExecutionPolicy Bypass -File .\fashion_psp\evaluation\run_dualfashion_state_risk_dp.ps1 -SkipSelection -BatchSize 4
```

可用 `-PythonExe D:\anaconda\envs\dualfashion\python.exe` 显式指定解释器，避免 PowerShell 子进程选错环境。选图阶段为确定性 CPU 回放；评估缓存按图像与 GT 的字节哈希及评估器签名校验，自动恢复。

主要代码为 `research_dualfashion_state_risk_dp.py`；`StateRiskDP.load(path)` 从 NPZ 恢复规划表，`decide(current, alive, t, remaining)` 仅接收当前分数、存活掩码、step、剩余预算。返回新存活掩码、估计即时损失、统计支持量及本步是否需要 reward 观测。

## 结果与解释边界

输出目录：`DualFashion-main/output/ifashion_fitb_state_risk_dp`。

- `frozen_selections.csv`：各方法每任务选图、seed、完整剪枝轨迹、预算及 reward 图像数。
- `planning_statistics_all_tasks.npz` / `planning_statistics_fold*.npz`：全样本或用户分组规划表。
- `native_metrics/summary.csv`：所有 33 个方法标签的 Personal、Compatibility、CIS、LPIPS。
- `native_metrics/per_task_metrics.csv`：逐任务四项指标。
- `native_metrics/paired_deltas.csv`：相对默认 PSP 的配对差异及用户聚类 bootstrap 区间。
- `research_summary.md`：本次结果分析。

`exploratory` 在相同 200 个任务上建立统计并回放；`crossfit` 按 108 位用户分为 5 组，仅用其余用户构建规划表，再回放当前用户。所有 200 个任务此前都已参与方法研究，所以交叉评估仍属于探索性证据，不是全新的独立测试。

不更新生成器或 reward 网络权重，但需要终值 Personal 做经验校准。Compatibility、CIS、LPIPS、GT 均不参与剪枝或粒子数选择。去噪粒子步相等不等于总运行时间相等：本策略需要更多中间图像解码及 reward 评估。
