# Terminal Personal forecasting and uncertainty-aware PSP

2026-10-25 correction: the user clarified all200 source tasks belong to the test
set and must not be used for fitting, calibration or tuning. The internal splits
and trained artifacts below are historical exploration, not permissible independent
training data for the new research. Do not rerun the default training command as
part of the new protocol. See `PERSONAL_RESEARCH_PROTOCOL.md`.

This pilot trains a small decision model from existing DualFashion FITB
trajectory scores. It does not train or load DualFashion or OpenCLIP.
The objective is **cached step-28 Personal**, not CIS/LPIPS or a GT label.
The native dataset uses decoded x0 reward views; an online terminal scheduler
decode must be checked separately before claiming exact agreement with it.

## Training in Windows PowerShell

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP

powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_terminal_personal.ps1 `
  -Epochs 100 `
  -EnsembleSize 5 `
  -BatchSize 1024 `
  -Device cuda:0 `
  -Resume
```

The wrapper resolves `python.exe` from the active `CONDA_PREFIX`, avoiding
the Python interpreter mismatch seen in previous experiments. `-Python
D:\anaconda\envs\dualfashion\python.exe` explicitly selects it if necessary.
Only PyTorch, NumPy, and tqdm are needed. There are no model downloads.
`-Device cpu` is also supported; this model is small enough for CPU training.

Default input:
`DualFashion-main/output/ifashion_fitb_timestep200_personal_analysis/candidate_metrics_pre_gt.csv`.
The loader only reads identity, Personal, sigma, and image paths, discarding
other metric columns. The selected pool is always
`2026 + task_index * 10000 + slot`, for slots 0..7. Supplementary slots 8..15
are not silently added. All 28 steps must exist for all eight trajectories.

The 200-task input has 108 users. Seed 2026 produces these user-disjoint groups:

| Partition | Users | Tasks | Purpose |
| --- | ---: | ---: | --- |
| Train | 64 | 119 | Fit model parameters and feature normalization |
| Validation | 16 | 32 | Choose each ensemble member's best epoch |
| Calibration | 16 | 28 | Calibrate prediction intervals only |
| Test | 12 | 21 | Report held-out diagnostics and policy results |

All seeds and steps belonging to one user remain in one partition. These
are exploratory splits of an existing test-derived cache, previously used
for other tuning, not a newly collected independent benchmark. `split.json`
records the precise assignments. There are 43,200 prefix examples overall:
200 tasks * 8 trajectories * 27 nonterminal prefixes.

## Model and calibration

Five independently initialized, user-bootstrap ensemble members have two
64-unit SiLU hidden layers. Inputs are 20 causal statistics of Personal
scores and scheduler sigma, including current reward, observed changes,
slopes, and prefix variation. The model receives no user/task/seed IDs,
compatibility, GT image, GT similarity, GAA, CIS, or LPIPS as features.

Each member estimates a Gaussian mean and variance for
`(Personal_28 - Personal_t) / 100`. Gaussian negative log likelihood is used;
negative NLL values are valid for continuous densities. No monotonic reward
curve is assumed. Bootstrap weights give equal total weight to sampled users.
At inference, terminal uncertainty combines member disagreement and learned
residual variance.

Calibration uses the maximum standardized prediction error per calibration
user, over all their tasks, eight seeds, and steps 1..27. The finite-sample
order statistic sets interval multiplier q. Default target coverage is 90%
at this user-group scope, under exchangeability assumptions. This may give
wide intervals. Coverage is measured separately on held-out users; it is not
a guarantee that forced pruning preserves the best seed.

## What is actually changed in PSP

This first experiment fixes the default 7/14/28 nodes and retains 4/2/1 seeds.
The denoiser budget remains exactly `8*7 + 4*7 + 2*14 = 112` particle steps.
It isolates the effect of forecasting/uncertainty from schedule optimization.

- `raw_personal`: original PSP, ranking currently visible Personal.
- `predicted_mean`: rank forecast terminal Personal at steps 7 and 14.
- `uncertainty_anchor_ucb`: retain the best predicted-mean anchor, then
  optimistic upper-bound contenders. Clearly dominated contenders rank last.
- `bon4`: terminal Personal best of the first four seeds of the same pool;
  112 particle steps. This is a nested same-pool diagnostic, distinct from
  the historical independently sampled seed-42 BoN experiment.
- `bon8`: terminal Personal best of all eight; 224 particle steps.

At step 28, all PSP variants select by the actually observed Personal.
Budget-required removal of overlapping contenders is labeled `forced_prunes`.
The algorithm cannot promise confidence-certified elimination at every fixed
budget decision. Mean/UCB comparisons may underperform; no performance gain
is assumed before the full run is evaluated.

This is **offline replay**, not an integrated online generator. Forecast
features use every observed Personal score up to a pruning point. An online
implementation therefore needs extra intermediate decoding/scoring relative
to the original three-view PSP. The 112-step figure counts denoiser work;
it does not imply equal wall time or evaluator cost. Check those costs before
claiming online speedups. The reusable predictor API lives in
`fashion_psp/search/terminal_personal.py`.

## Outputs and interruption recovery

Default directory: `fashion_psp/artifacts/terminal_personal/fitb200_v1/`.

- `best.pt`: deployable ensemble of each member's best validation checkpoint,
  training-only normalization, feature schema, and calibration.
- `member_00` through `member_04`: individual `best.pt`, `latest.pt`, and
  `training_log.csv`. Latest includes optimizer, RNG state, epoch, and logs.
- `training_manifest.json`, `split.json`, `calibration.json`: provenance,
  input hash, group assignments, and interval calibration.
- `forecast_diagnostics.csv`: prediction MAE/RMSE, interval coverage/width,
  including separate step 7 and 14 results. Step -1 means all prefixes 1..27.
- `selection_pre_gt.csv`, `selection_traces.json`: frozen seed choices,
  intermediate predictions/intervals, and forced-pruning diagnostics.
- `policy_summary.csv`: Personal, regret against the same eight-seed pool,
  best-seed hit rate, and denoiser budget, by partition and policy. Judge
  generalization on `split=test`; `all_exploratory` includes training data.

After interruption, rerun the exact training command with `-Resume`.
Progress is saved after each epoch; a partially completed epoch is repeated.
Completed members are reused. Patience 15 stops members without validation
improvement; increasing the requested epoch limit does not bypass an already
triggered early stop. Use a new output directory for changed hyperparameters
or different input data. Keep the same environment/device for reproducibility.

To rerun the frozen selection and add **post-hoc GT-Sim/GAA** without training:

```powershell
powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_terminal_personal.ps1 `
  -ReplayOnly `
  -PosthocGT `
  -Device cuda:0
```

The GT diagnostic CSV is opened only after `selection_pre_gt.csv` is frozen.
Input paths must match the actual selected cache images. These diagnostics
are GT-Sim/GAA, not an already completed native CIS/LPIPS evaluation. Native
CIS/LPIPS require scoring the newly frozen selected images separately; do
not reuse values for a different seed or another generation cache.

Verification includes prefix/future access isolation, GT-column exclusion,
user-disjoint splitting, exact 112-step budgets, forced-pruning labels, a
CPU/CUDA smoke run, and exact CPU resumed-versus-uninterrupted checkpoint
and selection agreement. Short smoke runs verify execution, not efficacy.
