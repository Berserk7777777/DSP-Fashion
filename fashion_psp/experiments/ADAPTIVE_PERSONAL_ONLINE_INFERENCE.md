# 2207-task adaptive Personal PSP online inference

2026-10-25 correction: the200-task cache is part of the test set, per the user.
This v2 policy was previously tuned on those tasks and is historical exploration,
not an untuned final experiment recommendation. See `PERSONAL_RESEARCH_PROTOCOL.md`.

This page documents the older dense v2 policy (112 candidate reward views/task).
For the sparse v3 policy requested to reduce decoding cost, use
`SPARSE_PERSONAL_ONLINE_INFERENCE.md` and `run_dualfashion_fitb_sparse_personal.ps1`.

Run from `E:\difashion-PSP` in the existing `dualfashion` environment:

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP

powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_fitb_adaptive_personal.ps1 `
  -MaxSamples 2207 `
  -Seed 2026 `
  -ResumeFrom auto
```

The launcher uses the selected global development-tuned policy at
`fashion_psp/artifacts/adaptive_personal_development_search_v2/selected_policy.json`.
It does not load or import the old state-risk DP calibration table. Output is
isolated at `DualFashion-main/output/ifashion_fitb_adaptive_personal/personal_group0_seed2026`.

`-ResumeFrom auto` resumes incomplete tasks and step checkpoints after an
interruption. Keep all arguments unchanged when resuming. To start a fresh
independent output directory, pass e.g. `-OutputRoot
E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_adaptive_personal\personal_group0_seed2027`
and set `-Seed 2027` only if that seed change is intended; do not change seeds
when comparing against the existing PSP mapping.

The launcher was smoke-tested using `-MaxSamples 1 -PrepareOnly`. The complete
2207-task diffusion run was not launched as part of this command preparation.

Cost at the selected algorithm: 247,184 particle steps and up to 247,184
decode/Personal candidate views over 2207 tasks, assuming all tasks are new.
The default 14 Personal views/task would total 30,898; this adaptive online
policy evaluates all surviving candidates at every step and costs 8x as many
candidate reward views. GPU time depends on hardware and batching.
