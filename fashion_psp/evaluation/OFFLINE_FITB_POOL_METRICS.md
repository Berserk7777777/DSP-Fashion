# FITB offline pool metrics

This evaluator reads saved PNGs, never diffusion checkpoints. It does not tune
or replay a pruning policy. Original dataset row numbers, group IDs, seed bases,
and slot order come from the committed cache metadata and are verified.

## Windows: DiFashion strict50

```powershell
conda activate difashion
Set-Location 'E:\difashion-PSP'
powershell -ExecutionPolicy Bypass -File .\fashion_psp\evaluation\run_difashion_fitb_offline_pool_metrics.ps1 `
  -Groups '0,1,2' -MaxSamples 2207 -BatchSize 4 -Device 'cuda:0'
```

Default inputs: `DiFashion-main/offline_fitb_strict50/`, data
`D:/DiFashion/ifashion/`. Default outputs: `<cache-root>/evaluation/`.

## Windows: DualFashion 28 steps

```powershell
conda activate dualfashion
Set-Location 'E:\difashion-PSP'
powershell -ExecutionPolicy Bypass -File .\fashion_psp\evaluation\run_dualfashion_fitb_offline_pool_metrics.ps1 `
  -Groups '0,1,2' -MaxSamples 2207 -BatchSize 4 -Device 'cuda:0'
```

Default inputs: `DualFashion-main/output/ifashion_fitb_16seed_offline/`, data
`C:/tmp/ifashion/`. Default outputs: `<cache-root>/evaluation/`.

Both commands automatically reuse evaluated images and score newly committed
candidates. Missing groups/tasks/candidates are recorded, not assigned zero.
An incomplete DiFashion candidate trajectory is deferred until its metadata is
committed. DualFashion commits metadata per task, so an unfinished task is
deferred. No concurrent GPU evaluation workers are started. `BatchSize` batches
image embeddings, not diffusion generation; the LPIPS terminal diagnostics are
processed one image at a time.

Use the same command again after collection progresses. `-AuditOnly` inventories
without model loading; `-ExportOnly` skips scoring and regenerates exports from
saved metrics for currently verified inputs. `-NoExcel` saves SQLite/CSV only.
`-RequireComplete` exits nonzero until the requested scope is fully evaluated.
Use `-Groups '0'` or `-Methods 'psp'` for a smaller scope. `-MaxSamples` is the
number of original task rows, not the count of completed tasks.

Keep the same batch size/protocol/assets when resuming; incompatible evaluator
changes require a new `-OutputRoot`. Outputs are separate from generation files.
The evaluation run inventories a snapshot; rerun to include images committed
during an ongoing evaluation. Do not modify committed source images.

## Metrics and dependencies

BoN: every seed's final saved image gets CS, CIS, LPIPS, Compatibility, Personal.
PSP: every saved step gets Compatibility/Personal; only the final saved image
also gets CS/CIS/LPIPS. The last step is 50 for the new DiFashion strict50 cache
(not the old 51-call PNDM protocol), and 28 for DualFashion. Intermediate CS/CIS/
LPIPS columns are blank rather than zero.

- OpenCLIP ViT-H-14/laion2b-s32b-b79K, FP32, no autocast/TF32.
- CS, CIS, Personal: cosine multiplied by 100.
- Compatibility: sigmoid of the backbone's native FashionEvaluator checkpoint,
  with raw CLIP features and fixed observed outfit item features.
- LPIPS: VGG, original saved image resolution, pixels normalized to [-1,1].
- GT is read only for terminal CIS/LPIPS and never affects reward computation.

Required evaluation assets, additional to generation inputs:

1. `cnn_features_clip.npy`, `processed/test_history_clipembs.npy` in data root.
2. DiFashion: `processed/new_fitb_test.npy`, `new_id_cate_dict.npy`,
   `new_all_item_image_paths.npy`, `test_grd.npy`, `images/`.
3. DualFashion: `processed/fitb_test.npy`, `id_cate_dict.npy`,
   `all_item_image_paths.npy`, `test_grd_dict.npy`, `semantic_category/`.
4. Each backbone's `Evaluation/compatibility_evaluator/ifashion-ckpt/` or
   `eval/compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth` and its
   `compatibility_net.py` implementation.
5. OpenCLIP safetensors (override using `-PersonalClipWeights`), torchvision's
   cached VGG16 backbone weights, and LPIPS's packaged VGG calibration weights.
   The evaluator sets HF offline mode. Provision all weights on the server.

Evaluation environment needs `open_clip_torch`, `lpips`, `safetensors`, torch,
torchvision, numpy, Pillow, tqdm. Excel uses the Codex artifact-tool runtime
when available locally. On a server without it, install the streaming fallback:

```bash
python -m pip install openpyxl
```

Linux uses the Python module directly; `--data-root` points to `ifashion/`:

```bash
python -m fashion_psp.evaluation.evaluate_fitb_offline_pools \
  --model difashion --cache-root /outputs/offline_fitb_strict50 \
  --data-root /data/DiFashion/ifashion \
  --personal-clip-weights /models/open_clip_model.safetensors \
  --groups 0,1,2 --max-tasks 2207 --batch-size 4 --device cuda:0

python -m fashion_psp.evaluation.evaluate_fitb_offline_pools \
  --model dualfashion --cache-root /outputs/ifashion_fitb_16seed_offline \
  --data-root /data/DualFashion/ifashion \
  --personal-clip-weights /models/open_clip_model.safetensors \
  --groups 0,1,2 --max-tasks 2207 --batch-size 4 --device cuda:0
```

DualFashion's Windows absolute image paths are resolved by their verified task
directory suffix after relocation. Metadata seed/task identities are retained.

## Tables

`excel/summary.xlsx`:

| Sheet | Meaning |
|---|---|
| Summary | Five terminal metrics per method/group, populations separately labeled |
| Coverage | Expected/available/evaluated candidate and image counts per method/group |
| PSP_steps | Mean Personal/Compatibility per step/group, actual image/task counts |
| Seed_slots | Terminal five-metric means by original seed slot, actual task counts |
| Definitions | Units, metric definitions, populations, data role |

`excel/terminal_group{g}_*.xlsx`: one row per terminal candidate, all five metrics,
task/uid/oid/seed/slot/image path, complete-pool flag, final Personal rank/regret.
Ranks and regret are blank until the full declared seed pool is evaluated.

`excel/psp_trajectories_group{g}_*.xlsx`: one row per seed, original task/uid/oid/
group/slot/seed, number of evaluated steps, `personal_s01` ... `personal_s50`
(or s28), followed by `compatibility_s01` ... `compatibility_s50` (or s28).
Missing values remain blank. Wide format avoids the Excel row limit. Each part
has at most 5000 trajectories (configurable with Python `--excel-rows`); terminal
tables split at 50000 rows. `excel/manifest.json` lists generated workbooks.

Canonical machine-readable tables:

- `metrics.sqlite`: image-level reusable metrics, committed per scoring batch.
- `terminal_candidates.csv`: all evaluated terminal candidates and five metrics.
- `psp_steps_group0/1/2.csv`: long trajectories, one row per task/seed/step,
  including image path, scheduler timestep, sigma where available, two rewards;
  the terminal rows also contain CS/CIS/LPIPS.
- `psp_trajectories_group0/1/2.csv`: the wide trajectory tables used by Excel.
- `task_oracle.csv`: full-pool final Personal oracle per task/method/group.
- `coverage.csv`, `coverage_summary.csv`, `evaluation_status.json`: missing and
  pending entries, actual denominators, overall completeness.
- `summary.csv`, `step_summary.csv`, `slot_summary.csv`: Excel summary sources.
- `protocol.json`: metric formulas, precision, evaluator/assets/version signatures.

`all_terminal_candidates` is a candidate average, not a selected BoN/PSP result.
`full_pool_personal_oracle_posthoc` is a retrospective upper bound, not an actual
PSP/DSP policy. Ties use ascending seed. Group pools are never combined. The
offline tables can later drive causal policy replay; no pruning policy or
threshold is selected by this evaluation. The previously examined 200 tasks
remain part of the test set and are not used for fitting.
