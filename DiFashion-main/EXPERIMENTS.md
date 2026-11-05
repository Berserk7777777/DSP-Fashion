# DiFashion seed and Best-of-N experiment

## Scope

Run the released DiFashion checkpoint first. Do not train a new model before
the published inference and evaluation path is reproduced. The initial study is
PFITB on one dataset (start with iFashion), with every model, data split,
guidance scale, denoising step count, and preprocessing operation fixed. The
only changing variable is the initial diffusion-noise seed.

## Required assets

The files committed in `datasets/` are metadata only. Before inference, obtain
the released DiFashion checkpoint and the data package linked from the project
README. For the selected dataset, the following must be present:

- item images, including `empty_image.png` at item id 0;
- `new_id_cate_dict.npy`, `new_all_item_image_paths.npy`, and `cnn_features_clip.npy`;
- `processed/new_train.npy`, `processed/new_fitb_{valid,test}.npy`, and the
  corresponding history-latent files; and
- an output directory containing `checkpoint-15000` from the released model.

The official evaluator additionally needs the released fine-tuned Inception
model under `Evaluation/finetuned_inception/`.

After downloading, verify the local directory shape before launching inference:

```powershell
python tools\check_difashion_assets.py `
  --data-path E:\data\ifashion `
  --img-folder-path E:\data\ifashion_images `
  --output-dir E:\checkpoints\difashion_ifashion
```

## Phase 0: environment check

Use a CUDA build of PyTorch that supports the local GPU, then install the
versions listed in the DiFashion README: Python 3.8.13, torch 2.0.1,
torchvision 0.15.2, transformers 4.32.1, accelerate 0.20.3, diffusers 0.18.2,
open-clip-torch 2.20.0, xformers 0.0.22, pytorch-fid 0.3.0, and lpips 0.1.4.
The original versions are the reproduction target; a newer CUDA-compatible
PyTorch may require a compatibility patch to old diffusers/xformers.

`DiFashion/config.yaml` is configured for local GPU 0. Confirm it matches the
actual CUDA device before launching.

## Phase 1: show seed sensitivity

Run a small deterministic PFITB sweep first. Replace all placeholder paths.
The checkpoint directory is both the input checkpoint root and the parent for
the generated outputs because this is how the upstream script is structured.

```powershell
Set-Location E:\difashion-PSP\DiFashion-main\DiFashion
accelerate launch --config_file config.yaml inf4eval.py `
  --task FITB --mode test --use_ema --use_ema_fashion `
  --data_path E:\data --dataset_name ifashion `
  --img_folder_path E:\data\ifashion_images `
  --output_dir E:\checkpoints\difashion_ifashion `
  --run_name seed_sweep_smoke --seeds 101,202,303,404 `
  --max_batches 8 --deterministic_inference --mixed_precision fp16
```

The `--deterministic_inference` flag fixes CUDA behavior and changes the
default random crop to a center crop. Consequently, changing `--seeds` changes
only the initial latent noise. Inspect the grids and the nonzero pixel-distance
counts:

```powershell
python ..\tools\analyze_seed_sweep.py `
  --run-dirs `
    E:\checkpoints\difashion_ifashion\seed_sweep_smoke\eval-test\FITB-checkpoint-15000-cate12.0-mutual5.0-hist4.0-seed101 `
    E:\checkpoints\difashion_ifashion\seed_sweep_smoke\eval-test\FITB-checkpoint-15000-cate12.0-mutual5.0-hist4.0-seed202 `
    E:\checkpoints\difashion_ifashion\seed_sweep_smoke\eval-test\FITB-checkpoint-15000-cate12.0-mutual5.0-hist4.0-seed303 `
    E:\checkpoints\difashion_ifashion\seed_sweep_smoke\eval-test\FITB-checkpoint-15000-cate12.0-mutual5.0-hist4.0-seed404 `
  --output-dir E:\checkpoints\difashion_ifashion\seed_sweep_smoke\seed_report
```

`summary.json`, `per_item_pairwise.csv`, and `grids/` form the direct evidence
that different initial seeds produce different samples for the same user,
outfit, and category.

## Phase 4: Offline Candidate Replay Bundle (iFashion FITB)

`tools/collect_ifashion_fitb_candidates.py` reuses DiFashion's own
`DiFashion.fashion_generation` path and its PNDMScheduler. It reads only
`processed/new_fitb_test.npy` and `processed/test_hist_latents.npy`; it does not
load `test_grd.npy` or FITB retrieval candidates. The missing outfit item is
the zero item already present in each processed FITB row.

The collector fixes DiFashion's original `num_inference_steps=50` argument.
With the installed Diffusers 0.33.0 PNDMScheduler and released scheduler
config this produces 51 actual denoising-loop calls. PSP checkpoints are thus
one-based loop steps 13, 26, and 51, corresponding to `ceil(25% * 51)`,
`ceil(50% * 51)`, and 100%. The checkpoint's SD2 PNDM scheduler is retained.
PSP uses seeds `2026 +
task_index*10000 + candidate_index` (8 candidates); BoN uses the separate pool
`42 + task_index*10000 + candidate_index` (4 candidates). Startup checks the
full task/seed map for collisions. The manifest key includes row index, uid,
and oid; output directories use the validated-unique `uid_oid` form.

At each saved PSP checkpoint, `x_prediction` is the clamped clean estimate
from the actual DiFashion helper
`pred_ori_sample_given_epsilon(t, x_t_before_step, guided_noise_pred)`. The
collector decodes it to a PNG for later reward evaluation and intentionally
does not write latent tensors. The decoded prediction is not the continuation
latent. Metadata records zero-based loop index, one-based step number, and
actual PNDM timestep. `step_51` is retained as a diagnostic x-prediction, but
the selection protocol is **step 13 -> step 26 -> final**: at the final node,
the two surviving seeds are scored on the terminal scheduler decode in
`final/`, which is also the image used by the official post-selection metrics.
It must not be replaced by `step_51` because the latter estimates x0 before
the final PNDM update. The output layout is:

```text
fitb_ifashion_offline_data/
  bon/<uid>_<oid>/seed_<seed>.png
  psp/<uid>_<oid>/step_13/seed_<seed>.png
  psp/<uid>_<oid>/step_26/seed_<seed>.png
  psp/<uid>_<oid>/step_51/seed_<seed>.png
  psp/<uid>_<oid>/final/seed_<seed>.png
  metadata/<uid>_<oid>.json
```

The PSP cache physically runs all eight candidates for all 51 loops: 408 UNet
calls per task. The later online PSP schedule would cost 206 calls
(`8*13 + 4*(26-13) + 2*(51-26)`); this replay does not demonstrate online
compute savings. BoN@4 costs 204 calls (`4*51`). Both costs are stored in
`run_config.json`, each task metadata file, and `missing_report.json`.

Example using the locally verified paths:

```powershell
conda activate difashion
Set-Location E:\difashion-PSP\DiFashion-main
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$python = 'D:\anaconda\envs\difashion\python.exe'
& $python tools\collect_ifashion_fitb_candidates.py `
  --data-root D:\DiFashion `
  --image-root D:\DiFashion\ifashion\images `
  --checkpoint-root D:\DiFashion `
  --base-model E:\difashion-PSP\DiFashion-main\pretrained_models\stable-diffusion-2-base `
  --output-root E:\difashion-PSP\DiFashion-main\fitb_ifashion_offline_data `
  --resume-from 0
```

`--resume-from 1000` starts at FITB row 1000. Completed task directories are
verified and skipped automatically; interrupted staging directories are left
untouched for inspection. `--max-tasks 1` is available for a smoke run, but the
full-dataset audit will correctly list the remaining tasks as missing.

Refresh the current completeness report without loading the model or starting
inference:

```powershell
& $python tools\collect_ifashion_fitb_candidates.py `
  --data-root D:\DiFashion `
  --image-root D:\DiFashion\ifashion\images `
  --checkpoint-root D:\DiFashion `
  --base-model E:\difashion-PSP\DiFashion-main\pretrained_models\stable-diffusion-2-base `
  --output-root E:\difashion-PSP\DiFashion-main\fitb_ifashion_offline_data `
  --audit-only
```

Completion requires `missing_report.json` to contain
`"complete": true` and `verified_complete_tasks` equal to the task count
loaded from `new_fitb_test.npy`. Every committed task contains four BoN images,
eight PSP terminal images, and 24 PSP x-prediction reward images; metadata is
stored outside the image directories. `manifest.jsonl` records candidate and
task states.

## Phase 2: reproduce DiFashion

Remove `--max_batches`, use the same four seed values, and evaluate every seed
separately with the official FITB script. Report each seed and mean plus sample
standard deviation for FID, IS, CLIP, CLIP retrieval accuracy, LPIPS,
personalization, and compatibility. This verifies baseline behavior and shows
whether seed changes quality, not merely pixels.

Example for one seed:

```powershell
Set-Location E:\difashion-PSP\DiFashion-main\Evaluation
python -u evaluate_fitb.py `
  --dataset ifashion --mode test --task FITB --ckpts "[15000]" `
  --data_path E:\data\ifashion `
  --output_dir E:\checkpoints\difashion_ifashion `
  --eval_version seed_sweep_full `
  --result_suffix -seed101 `
  --gpu 0 --log_name seed101
```

## Phase 3: Best-of-N

Best-of-N is not the mean of N independent runs. For every test sample, it must
generate N candidates and choose one candidate before final evaluation. Use the
frozen DiFashion compatibility evaluator as the initial selector: its
`CompatibilityEvaluator.evaluate_compatibility` method returns one score per
outfit. For PFITB, select the candidate whose generated item gives the highest
compatibility score with the observed items. Tune any combined reward on the
validation split only; test labels and ground-truth image similarity must not
choose the winner.

Compare standard sampling (N=1) and BoN with N=2, 4, and 8 under matched
denoising settings. Persist all N candidates, reward scores, selected seed, and
the winner. Report the official test metrics only for the selected winner.
This candidate manifest is also the data source needed by PSP: PSP will score
intermediate predicted clean samples with the same frozen reward and prune
seeds according to a fixed budget schedule.

## Phase 5: Offline BoN@4 and PSP@8 selection/evaluation

`tools/offline_fitb_select_and_evaluate.py` is an independent consumer of the
candidate cache. It reads the FITB condition rows from
`processed/new_fitb_test.npy`, candidate/task metadata, the frozen DualFashion
compatibility evaluator, `cnn_features_clip.npy`, and
`processed/test_history_clipembs.npy`. It does **not** read `test_grd.npy`, a
retrieval-candidate file, or the GT FITB item while scoring candidates or
choosing a seed.

The cache was audited against the currently installed data on 2026-10-01:
2,207 condition rows, 2,207 complete task directories, and zero missing or
invalid cached candidates. The task key is `task_index_uid_oid`; `oid` alone
is not unique.

Use a fresh result directory for each frozen reward and group. This command
scores all 28 reward images per task (four terminal BoN candidates and the
three saved x-prediction images for each of eight PSP candidates), selects
BoN@4 and PSP@8, and evaluates selected terminal images:

```powershell
conda activate difashion
Set-Location E:\difashion-PSP\DiFashion-main
$python = 'D:\anaconda\envs\difashion\python.exe'
& $python tools\offline_fitb_select_and_evaluate.py `
  --cache-root E:\difashion-PSP\DiFashion-main\fitb_candidate_replay_ifashion_28step `
  --data-root D:\DiFashion\ifashion `
  --dualfashion-root E:\difashion-PSP\DualFashion-main `
  --output-root E:\difashion-PSP\DiFashion-main\offline_selection_eval_compat_group0_v3 `
  --reward compatibility `
  --group-id 0 `
  --batch-size 8 `
  --device cuda:0
```

For an interrupted result directory, retain all files and resume with exactly
the same cache/data/reward/group arguments plus `--resume`. Task-atomic score
files are reused; complete selection JSON is reused only after its winner is
recomputed consistently. `--resume-from 1000` only schedules row 1000 onward;
already complete rows remain available in output tables.

```powershell
& $python tools\offline_fitb_select_and_evaluate.py `
  --cache-root E:\difashion-PSP\DiFashion-main\fitb_candidate_replay_ifashion_28step `
  --data-root D:\DiFashion\ifashion `
  --dualfashion-root E:\difashion-PSP\DualFashion-main `
  --output-root E:\difashion-PSP\DiFashion-main\offline_selection_eval_compat_group0_v3 `
  --reward compatibility --group-id 0 --batch-size 8 --device cuda:0 `
  --resume --resume-from 1000 --check-repeat
```

Available built-in rewards are `compatibility`, `personal`, `bottleneck`, and
`pareto`. Compatibility is the frozen sigmoid FashionEvaluator score; Personal
Sim is `100 * cosine` of normalized OpenCLIP candidate/history embeddings.
`bottleneck` uses the minimum of average-tie percentile ranks of these two
quantities **within the active candidate set**, and `pareto` uses non-dominated
fronts with a recorded deterministic tie order. Neither combines raw scores of
different units. A plugin reward uses `--reward plugin --plugin module:function`
and must return one finite scalar. All reward choices and parameters require
validation-set freezing; with the currently available test-only cache, outputs
are marked exploratory and must not be treated as an unbiased tuned result.

The output directory contains:

- `run_config.json`: immutable input/reward contract and explicit leakage/cost caveats.
- `score_tasks/<task>.json`: task-atomic reusable Compatibility/Personal Sim
  components for all 28 reward images.
- `candidate_scores.csv` and `.jsonl`: flat candidate-level score table.
- `selections/bon/<task>.json` and `selections/psp/<task>.json`: complete
  reward trace, active seed lists, formula, tie break, loop index, PNDM
  timestep, and selected terminal image path.
- `selected_images_index.csv`: final terminal image path for `standard`, `bon`,
  and `psp`; no images are copied. `standard` is always the BoN independent
  pool's `candidate_index=0` / first seed and performs no reward selection.
- `official_eval/task_metrics.csv`, `group_summary.csv`, and
  `aggregate_summary.csv`: selected-image frozen Compatibility, Personal Sim,
  and, when local weights load, frozen Inception metrics. `LPIP score` is
  explicitly N/A because one selected image per task has no within-task pair.
- The same tables now also include post-selection-only GT metrics: `gt_sim` is
  `100 * cosine` between predicted and GT item
  OpenCLIP ViT-H/14 embeddings (higher is better); `gt_lpips` is LPIPS-VGG on
  selected and GT RGB tensors scaled to `[-1, 1]` (lower is better). The tool
  validates the GT outfit's three observed items and target category against
  the FITB condition before evaluating. It reads `test_grd.npy` only at this
  final evaluation stage, never for candidate scoring or selection.
- `official_eval/metric_contract.json`: implementation, input, formula,
  direction, unit, and missing-value contract for every reported metric.
- `integrity_report.json` and `missing_tasks.csv`: coverage, file, seed, and
  selection checks.

Before an expensive run, audit the immutable cache only:

```powershell
& $python tools\offline_fitb_select_and_evaluate.py `
  --cache-root E:\difashion-PSP\DiFashion-main\fitb_candidate_replay_ifashion_28step `
  --data-root D:\DiFashion\ifashion `
  --dualfashion-root E:\difashion-PSP\DualFashion-main `
  --output-root E:\difashion-PSP\DiFashion-main\offline_selection_eval_audit_v3 `
  --audit-only
Get-Content E:\difashion-PSP\DiFashion-main\offline_selection_eval_audit_v3\integrity_report.json
```

A completed full run has three selected-image rows per FITB task (6,621 for this
split), two complete method summaries, `cache_problem_count: 0`, and no rows
in `missing_tasks.csv`. The cache costs are deliberately separate: physical
candidate collection was 112 UNet calls/task for BoN@4 and 224 for the full
PSP cache. The PSP replay's nominal online schedule is 112 calls/task
(`8*7 + 4*7 + 2*14`) but this offline result is not a measurement of online
dynamic-pruning runtime.

To add or refresh only these final metrics for an existing selection run, use
the same frozen reward/group configuration and `--resume --evaluate-only`.
This does not rewrite candidate scores or selected seeds:

```powershell
& $python tools\offline_fitb_select_and_evaluate.py `
  --cache-root E:\difashion-PSP\DiFashion-main\fitb_candidate_replay_ifashion_28step `
  --data-root D:\DiFashion\ifashion `
  --image-root D:\DiFashion\ifashion\images `
  --dualfashion-root E:\difashion-PSP\DualFashion-main `
  --output-root E:\difashion-PSP\DiFashion-main\offline_selection_eval_compat_group0_v3 `
  --reward compatibility --group-id 0 --batch-size 8 --device cuda:0 `
  --resume --evaluate-only
```

After upgrading an older two-method result directory to schema 4, first add
the deterministic Standard index without loading reward models, then run the
post-selection evaluation command above. The index must report 2,207 rows for
each of `standard`, `bon`, and `psp` before final evaluation:

```powershell
& $python tools\offline_fitb_select_and_evaluate.py `
  --cache-root E:\difashion-PSP\DiFashion-main\fitb_candidate_replay_ifashion_28step `
  --data-root D:\DiFashion\ifashion `
  --dualfashion-root E:\difashion-PSP\DualFashion-main `
  --output-root E:\difashion-PSP\DiFashion-main\offline_selection_eval_compat_group0_v3 `
  --reward compatibility --group-id 0 `
  --select-only --resume --check-repeat
```

To preserve per-group results before making an across-group view, aggregate
the individual `group_summary.csv` files rather than pooling task rows:

```powershell
& $python tools\aggregate_offline_fitb_groups.py `
  --group-summary E:\difashion-PSP\DiFashion-main\offline_selection_eval_compat_group0_v3\official_eval\group_summary.csv `
  --group-summary E:\difashion-PSP\DiFashion-main\offline_selection_eval_compat_group1_v3\official_eval\group_summary.csv `
  --output-dir E:\difashion-PSP\DiFashion-main\offline_selection_eval_compat_groups_v3
```
