# 全200任务自适应剪枝开发搜索

用户已明确授权在当前全部200任务上反复调整，以平均Personal与最优seed命中率都超过默认PSP为目标。本实验因此与此前“先冻结、不调参”的v1分开保存，不覆盖v1。所有结果均为开发集调参结果，不是独立测试。

## 实际运行结果

评估5628个全局参数配置，136个同时超过默认。选取规则预先写在搜索程序中：先要求两指标都严格超过默认，然后优先最大化命中数，同命中数再比较平均Personal。选择trial2770。

|方法|Personal均值|最优seed命中数|命中率|平均regret|Compatibility均值|
|---|---:|---:|---:|---:|---:|
|默认7→4、14→2、28→1|71.560855560|108/200|54%|1.664545231|0.729010790|
|调参后自适应|71.952895527|122/200|61%|1.272505264|0.732375811|

Personal差值+0.392039967；命中率+7个百分点。胜/平/负=51/114/35。挽回默认漏掉的39个最优seed，同时丢失默认保住的25个，净增加14个命中任务。

## 最终规则（全局一致，无任务专属分支）

1. 采用metadata确认的每任务原始8seed，28步，预算112粒子步。
2. 每步对存活候选评估Personal；step1–3不主动剪枝。
3. 从step4起，按本步Personal降序、seed升序排序，保护本步前2名。
4. 固定本步领先者为参照。其余候选若最近3步均严格落后于这个参照，则标记可剪。
5. 每步最多主动剪除2个标记候选，当前Personal较差者先剪；同分较大seed先剪。
6. 纯整数预算可行性进一步约束数量，保证剩余粒子能在剩余步数中恰好用完112预算。不足以继续保留时强制剪枝，单独记录。
7. step28按真实终值Personal选1个。

搜索选中z=0、margin=0、trend_horizon=0、mean_weight=0，因此最终方案没有使用标准差门槛、趋势外推或均值平滑。它是启发式，不是置信区间；不得把原版2×标准差的描述沿用到本方案。

在200任务中出现44种不同剪枝路径；其中最常见路径（74任务）是step4留6、step5留4、step10留3，最后3个候选到step28选1。预算为8×4+6×1+4×5+3×18=112。此路径是决策结果，不是预设调度。5个任务发生预算强制剪枝，共5个粒子。

## 搜索与因果边界

搜索先枚举648个保守策略配置，再使用固定随机种子20261024产生5000次参数提案；去重后总共评估5628个配置。涉及最早主动剪枝步、窗口长度、分差标准差系数、绝对分差门槛、保护名次数量、单步主动剪枝上限、趋势外推及平滑权重。

终值仅用于每个策略回放结束后的全局参数评价与选择。最终策略在线函数没有task/uid/oid输入，没有访问完整未来序列，没有使用旧DP表、GT、Compatibility、CIS、LPIPS或任务查找表。选择器未训练模型，但参数使用全部200任务调优，不能据此声称对这些任务完全无监督或完全未使用终值。

搜索的向量化实现逐步仅把当前存活候选本步Personal写入历史；最终使用独立标量实现与CausalObserver回调逐任务复核。记录每步全部合法Personal访问、保护集合、标记候选、可行粒子数、预算和真实剪枝原因。

## 验证

- 全部44800张缓存图片存在，CSV与原始metadata的身份、seed、图片路径匹配。
- 默认所选seed、图片、Personal与历史默认记录200/200完全一致。
- 400条任务/方法轨迹、11200条逐步记录的存活关系、剪枝集合和预算独立核对通过。
- 快速实现与标量实现所选seed在200/200任务一致。
- 将每个任务所有未访问分数替换为极大值，最终策略200/200完整决策轨迹不变。
- 保存参数后在临时目录独立复跑，summary、选择、逐步轨迹、剪枝事件等9个产物逐字节相同。
- 三项单元测试通过：预算可达集合与递归枚举全量比对；合成轨迹上的快慢实现与未来污染测试；预热/保护/单步剪枝上限。

## 开销

两方法每任务均112粒子步，各200任务共22400粒子步。默认每任务14次解码与Personal图片评估，自适应112次（8倍）；评分轮数3对28。以上为在线实施所需图片数，非GPU批次或耗时。本次真实运行是缓存回放，神经模型调用为0，未进行额外图片解码或reward重算。

## 复现

在E:\difashion-PSP运行，需要Python3.10+与NumPy：

```powershell
python -m unittest fashion_psp.experiments.test_search_adaptive_personal_development -v

# 完整复现搜索，使用新目录保存
python -m fashion_psp.experiments.search_adaptive_personal_development `
  --output-dir fashion_psp/artifacts/adaptive_personal_search_reproduction

# 仅复跑最终策略，不重新搜索
python -m fashion_psp.experiments.search_adaptive_personal_development `
  --policy fashion_psp/artifacts/adaptive_personal_development_search_v2/selected_policy.json `
  --output-dir fashion_psp/artifacts/adaptive_personal_selected_replay
```

原始实际输出目录为 `fashion_psp/artifacts/adaptive_personal_development_search_v2`，包含全部搜索记录、最终参数、逐任务结果、剪枝证据、预算和复现验证。`independent_verification.json`中的independent指验证程序独立核对，不表示独立测试集。
