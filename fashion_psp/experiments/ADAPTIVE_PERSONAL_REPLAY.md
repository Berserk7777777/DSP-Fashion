# 冻结规则与实现审查

本脚本为开发集的因果在线**缓存回放**，不运行生成模型，不提供独立测试结论。
规则在首次200任务回放前写入输出目录的 `frozen_rule.json`；脚本再次运行时逐字段核对，不允许静默更改。
没有阈值搜索、跨任务拟合、旧DP经验风险表或按结果调参。

## 策略语义

1. 8个候选一起从step1开始；每推进一名存活候选一步，消耗1粒子步。
2. 自适应每步观察全部存活候选的Personal，包括仅剩1粒子时和step28。
3. 从step3起，以**当前步**Personal最高者作为固定参照，比较该参照和每个存活候选最近3步的分数。参照同分取较小seed。
4. 三个分差都严格大于0，且分差均值严格大于2倍样本标准差（ddof=1），标记可剪。领先者变动时重新以本步领先者计算窗口；不用每个历史步不同领先者拼接差值。
5. 这是稳定性启发式。3点相关时间序列不能据此形成统计置信区间；分差恒定且为正时标准差为0，也会标记。
6. 设本步完成后有n个候选，e个标记，目标存活数d=n-e。枚举所有能在剩余步数内恰好消耗剩余预算的存活数F。优先取F中不小于d的最小数；如果不存在，则取F的最大数，以最少强制剪枝维持预算可行性。
7. 在目标数量内，先剪标记候选，再剪未标记候选；各组按当前Personal低到高、seed大到小剪除。后者记录为budget_forced。标记但受预算限制不能剪除者继续存活。
8. step28完成后要求累计预算恰好112；按当前终值Personal及seed选择1个。终步选择单独标记，不计为稳定性/预算强制剪枝。

## 纯预算可行性

定义C(h,m)为剩余h步、下一步必须推进m个粒子时所有可达成本：

```
C(1,m) = {m}
C(h,m) = union over k=1..m of {m+c : c in C(h-1,k)}
```

仅允许粒子数保持或减少，每一步至少1个；不复活、不空跑、不提前终止、不使用reward。当前动作保留m个后，必须满足 `112-spent in C(28-step,m)`。
这张整数可达集合不是跨任务校准表，也不含经验风险。
终步才选1个不额外消耗粒子步。默认预算为8×7+4×7+2×14=112。

## 因果边界与代码审查

- 原PSP `PSP-main/Fk-Diffusion-Steering/text_to_image/run_psp.py` 的step-end回调只在预定节点解码、评分、缩减粒子；它本身不是自适应方法。
- `fashion_psp/search/psp.py` 已规定Personal降序、seed升序的确定性排序，本脚本使用相同语义。
- 原步序列评估器 `evaluate_dualfashion_fitb_timestep.py` 的默认节点为7→4、14→2、28→1；其metadata校验要求28步、8seed。
- 旧 `run_dualfashion_fitb_state_dp.py` 导入StateRiskDP并读取planning_statistics；本次独立脚本没有导入该模块或读取其任何校准文件。
- 输入审计逐任务核对原group0缓存metadata的task/uid/oid/candidate_seeds和全部图片路径。发现额外seed、缺步、重复行或映射变化会报错，不截取排序后的前8个，不重排任务。
- 策略入口 `replay(seeds, method, observe)` 只获得初始seed及观察回调。`adaptive_decision`只获得存活候选已观察的历史；删除的历史立即移除。
- 环境 `CausalObserver` 每步推进时间，只返回本步存活候选的Personal，并拒绝时间跳跃或复活请求。它保存访问记录用于核对评估次数。
- 选择与完整轨迹先落盘，再单独读取终值Compatibility与所有原始8seed终值计算oracle/regret。无GT、Compatibility、CIS、LPIPS决策输入。
- 回放前的6项单元验证覆盖：小规模预算路径穷举对照、样本标准差与严格不等式、当前领先者窗口语义、全同分强制剪枝、未访问未来值污染不影响决策、禁止时间跳跃和复活。

## 运行与产物

在工作区根目录使用Python3.10+（仅标准库）：

```
python -m unittest fashion_psp.experiments.test_replay_dualfashion_fitb_adaptive_personal -v
python -m fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal --freeze-only
python -m fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal
```

默认输出为 `fashion_psp/artifacts/adaptive_personal_causal_replay_v1`。
`run_manifest.json`保存输入、脚本、规则、候选映射及冻结选择哈希；`report.md`报告真实运行结果。
解码/Personal评估次数为该策略在线实施所需的粒子图片计数；实际缓存回放神经模型调用为0。
评分轮数与硬件batch数不同，既有缓存构建成本和事后oracle查表成本不属于策略在线预算。
