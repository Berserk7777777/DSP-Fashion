# Controlled preference rewrite pilot

For terminal-reward forecasting and uncertainty-aware PSP training, see
[terminal_personal_README.md](terminal_personal_README.md).

The first experiment is DualFashion FITB task 395 (`9674_18277`, hats). Eight
particles share the exact initial Gaussian tensor with resolved noise seed
3952026 (`2026 + 395 * 10000`). Candidate IDs are rewrite IDs, never noise
seeds. The first rewrite is the original attribute JSON. Four cyclic field
orders and four reversed cyclic orders give eight distinct token sequences,
while preserving every attribute value. This tests serialization sensitivity;
it does not invent new preference modes or establish that an LLM rewrite helps.

The native generator uses checkpoint 0057501, BF16, deterministic context
encoding, 28 flow-matching steps, and text/context/preference CFG 8/7/8.
Personal is the frozen native `100*cosine(ViT-H/14(image), category history)`.
The original history vector stays fixed for all candidates. Default reward
inference runs on CPU to keep its weights out of the 16-GiB GPU.

Online pruning is 8 -> 4 at step 7, 4 -> 2 at step 14, and final selection
at 28. This costs 112 particle steps. All reward views are native pre-update
flow `x0` predictions; `selected.png` is the terminal scheduler decode.
The terminal decode is scored separately and does not change the last
selection. The objective is high Personal and agreement of the actual input
preference with its output image. Reports do not compare against GT.
After selection is frozen, the canonical rewrite and winning rewrite are
independently replayed under the current implementation (normally 56 extra
particle steps, or 28 if rewrite 0 wins). These diagnostic calls are separate
from the 112 online calls. The winner's replay checks exact PNG reproducibility.
The historical cache is a visual reference; it is not assumed to be an exact
canonical baseline for a newer implementation/environment.

Run in Windows PowerShell:

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP
powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_fitb_rewrite.ps1 -Resume
```

`-PrepareOnly` validates all eight preference dictionaries and untruncated
token sequences without loading the generator or reward. `-Resume` restores
the last fully checkpointed denoising step, including survivors and their
conditioning IDs. A completed run returns immediately. Rerunning without
`-Resume` starts again if the run was interrupted. Completed outputs are
protected against overwriting, and differing manifests require a new directory.

To inspect the full trajectories of all eight rewrites, use
`-AuditAllRewrites -OutputDir <new-directory> -Resume`. This runs online PSP
first and then 224 additional particle steps instead of the default canonical
and winner replay for diagnosis. Audit calls are
reported separately and are not part of the 112-step online budget. Audit
restarts from the beginning after interruption; the online run resumes.

Default outputs are under
`DualFashion-main/output/ifashion_fitb_rewrite/task395_9674_18277/`:

- `rewrite_manifest.json`: original profile, eight key orders, token hashes,
  noise mapping, configuration, and reference initial-noise hash.
- `predictions/step_07`, `step_14`, `step_28`: 8, 4, 2 decoded reward images,
  respectively, plus raw predicted-clean latents and diagnostic RGB projections.
- `resume_state.pt`: task-local checkpoint, updated atomically after each step.
- `scores.csv`, `metadata.json`: Personal scores, survivors, selected rewrite,
  actual model calls, shared-noise hashes, elapsed time, and peak allocated VRAM.
- `online_selection.json`: decision frozen before alignment diagnostics.
- `personal_comparison.json`: canonical versus selected PNG Personal, plus
  the winning rewrite's independent replay comparison.
- `selected.png`, `rewrite_psp_gallery.png`: terminal result and a comparison
  panel with the canonical rerun, winner, and historical cache.
- `canonical_rewrite_psp.png`: compact canonical/winner comparison.
- `selected_preference.json`: original user profile and exact conditioning
  preference fixed before generation. A high Personal score does not verify
  text/image alignment. Old GT reports are retained as historical artifacts.

Reward PNGs serialize with the native scorer's truncation to uint8 rather
than torchvision's rounding. This makes rescoring the saved PNG reproduce
the actual online Personal score to floating-point tolerance.

The old seed cache cannot supply rewritten trajectories: a different text
condition changes the velocity field from the first denoising step.

## Coherent history-mode pilot (no GT objective)

The aggregate profile lists alternative colors, materials, and designs from
different past items. Reordering its keys cannot make it an item-specific
description. `-RewriteMode history` instead takes the attributes of eight
distinct historical items in their existing order. This preserves observed
attribute co-occurrence and avoids arbitrary cross-products. It specializes
the profile into hypotheses, rather than paraphrasing an identical meaning.
No LLM rewrite or image captioning is used. For this task all eight source
items are hats. Other tasks fail if fewer than eight complete unique modes
are available; this is a single-task pilot, not a general fallback policy.

The same 3952026 noise, 7/14/28 pruning and fixed original Personal history
vector are used. The canonical aggregate profile is a separate audit-only
condition (ID 8). Default diagnostic replay cost remains 56 particle steps;
`-AuditAllRewrites` replays eight modes plus canonical (252 audit steps).

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP
$python = Join-Path $env:CONDA_PREFIX "python.exe"
powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_fitb_rewrite.ps1 -RewriteMode history -Resume

& $python .\fashion_psp\experiments\evaluate_preference_image_alignment.py `
  --output-dir .\DualFashion-main\output\ifashion_fitb_rewrite\task395_9674_18277_history_modes `
  --device cuda:0
if ($LASTEXITCODE -ne 0) { throw "Preference/image alignment diagnostics failed." }
```

The independent frozen OpenAI CLIP ViT-L/14 diagnostic produces
`preference_alignment.json` and `preference_image_pair.png`. It loads only
the frozen preferences and generated images; it does not load GT, change
Personal, rerank, or caption the image. It reports paired text/image cosine,
the pair's rank among the eight candidate texts, and per-field cosine.
Long descriptions are detected rather than silently truncated. Neither
cosine nor rank is a calibrated correctness probability. Visually check
color, material appearance, shape, and details separately, especially when
a Personal winner disagrees with its own conditioning. Such a result is a
failed alignment case, even if Personal is high. Do not relabel its output
preference after generation and call that successful conditional generation.

## Online FITB state-conditioned risk DP@8

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP
powershell -ExecutionPolicy Bypass -File .\fashion_psp\experiments\run_dualfashion_fitb_state_dp.ps1 -MaxSamples 2207 -Seed 2026 -ResumeFrom auto
```

This runs real denoising with the frozen `planning_statistics_all_tasks.npz`
table, Personal only, initial8 particles, 28 steps and exactly112 particle
steps per task. Generation and decoding use one particle per call. Generator
CPU offload and reward offload prevent them from occupying GPU memory together.
Reward uses float32 CLIP, batches of4 (padded when needed), the original cuDNN
TF32 setting and timestep-cache PNG quantization. Fixed reward batch shape is
necessary here: batch1 and batch4 can differ numerically on identical images.
No candidate cache is read and no text branch is generated.

Outputs live in `DualFashion-main/output/ifashion_fitb_state_dp/personal_group0_seed2026`:
final PNGs and traces under `tasks`, plus native `0057501/gen.npy` and `grd.npy`.
The total-task progress bar includes ETA and current step/alive/budget status.
Rerunning the same command skips finished tasks and resumes unfinished tasks
at their saved completed step. Intermediate PNGs are not retained. `-PrepareOnly`
validates inputs; `-MaxSamples 1 -OutputRoot <new-path>` runs a bounded smoke test.
Use `-PythonExe` to override the explicit dualfashion interpreter.

The full-table run is distinct from the prior user-crossfit research results:
the table uses all200 planning tasks, which remain part of the full task set.
See `fashion_psp/evaluation/STATE_RISK_DP.md` for method and metric definitions.
