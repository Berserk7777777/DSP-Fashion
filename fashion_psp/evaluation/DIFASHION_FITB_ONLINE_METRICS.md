# Native metrics for frozen DiFashion online FITB results

```powershell
conda activate difashion
Set-Location 'E:\difashion-PSP'
python -m fashion_psp.evaluation.evaluate_difashion_fitb_online `
  --run-root 'E:\difashion-PSP\DiFashion-main\online_fitb_psp6_personal_25_75_100_v1' `
  --output-root 'E:\difashion-PSP\fashion_psp\artifacts\difashion_fitb_psp6_online_native_metrics_v1' `
  --device cuda:0 `
  --batch-size 8 `
  --resume
```

Reuse the same command after interruption. Add `--audit-only` to validate
existing per-task metric records and regenerate aggregate tables without
loading GPU models. An incomplete audit exits with status 2 and writes the
missing task IDs in `integrity_report.json`.

The evaluator uses helper functions from
`DiFashion-main/tools/evaluate_fitb_offline_native.py` and its original
DiFashion compatibility checkpoint, without selecting any new candidates.
All selected PNGs and their task/seed identities are checked and hashed before
GT is opened. Metric implementations match the previously used native FITB
evaluation: CS/CIS/Personal use FP32 OpenCLIP ViT-H-14/laion2b-s32b-b79K;
LPIPS uses VGG and RGB [-1,1] images at original resolution; Compatibility
uses raw OpenCLIP features in the incomplete-outfit context and sigmoid of
the DiFashion FashionEvaluator logit. Only CIS and LPIPS use GT.

Outputs:

- `input_integrity.json`: task/image/selection checks before evaluation.
- `evaluation_config.json`: input/source hashes, method, group, and evaluator settings.
- `per_task/<task_id>.json`: atomic resumable metrics bound to task/seed/image hashes.
- `task_metrics.csv`: all task metrics, selected seeds/paths, and post-selection GT paths.
- `summary.csv`: task means, sample standard deviations, valid/missing counts.
- `integrity_report.json`: completion, missing IDs, and online CPU versus evaluation GPU Personal rounding difference.
- `metric_contract.json`: metric formulas and evaluator provenance.

Current runs use one group (group 0). Comparing policies on the inspected test
tasks is exploratory. This evaluation does not measure inference runtime,
change pruning, or consume saved intermediate predictions for final metrics.
