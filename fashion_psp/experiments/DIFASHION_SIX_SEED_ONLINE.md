# DiFashion FITB online PSP6: 25%, 75%, final

The inference argument remains DiFashion's native default 50. The audited
PNDMScheduler produces 51 actual loop calls. Percentages are rounded up against
that actual count: ceil(51*.25)=13, ceil(51*.75)=39, terminal=51.

Initialize six original seeds, keep four after call 13, keep two after call 39,
and choose one after terminal call 51. The reward is frozen Personal Sim:
100*cosine(OpenCLIP ViT-H-14/laion2b-s32b-b79K generated image, user/category
history embedding). Ties select the smaller original seed. Intermediate reward
images decode DiFashion's clamped x_prediction; final reward decodes the real
terminal scheduler latent. GT and retrieval files are never loaded.

This is serial ONLINE generation: only surviving trajectories continue. It is
not an offline full-trajectory replay or concurrent six-seed batch. Actual UNet
forward calls are counted and must equal 6*13+4*26+2*12=206 per completed task;
BoN@4 would use 4*51=204. Personal scoring adds 6+4+2=12 image evaluations in
three rounds, so these methods do not have identical end-to-end costs.

## PowerShell

```powershell
conda activate difashion
Set-Location 'E:\difashion-PSP'

# Check data, checkpoint, seed mapping and history vectors; no GPU inference.
python -m fashion_psp.experiments.run_difashion_fitb_six_seed `
  --output-root 'E:\difashion-PSP\DiFashion-main\online_fitb_psp6_personal_25_75_100_v1' `
  --preflight-only

# All FITB tasks; step count is inherited from DiFashion's default.
python -m fashion_psp.experiments.run_difashion_fitb_six_seed `
  --output-root 'E:\difashion-PSP\DiFashion-main\online_fitb_psp6_personal_25_75_100_v1' `
  --seed-base 2026 `
  --device cuda:0 `
  --reward-device cpu `
  --resume-from 0

# Verify every task, selected image, recorded reward selection and call budget.
python -m fashion_psp.experiments.run_difashion_fitb_six_seed `
  --output-root 'E:\difashion-PSP\DiFashion-main\online_fitb_psp6_personal_25_75_100_v1' `
  --audit-only
```

Reward runs on CPU by default to conserve GPU memory. `--reward-device cuda:0`
keeps the Personal model on GPU alongside DiFashion; use a different output
directory for this configuration. CPU and GPU score arithmetic can differ.

For a one-task smoke run use a separate output directory and `--max-tasks 1`.
Omitting that option processes all task rows from new_fitb_test.npy (currently
2207, never hardcoded by this runner). Task identities are original row index,
uid, oid; seed = 2026 + original_row_index*10000 + original_slot[0..5].

## Outputs and recovery

Each `<row_index>_<uid>_<oid>/` contains `step_13/seed_<seed>.png` (six images),
`step_39/seed_<seed>.png` (four), `step_51/seed_<seed>.png` (two), `selected.png`,
and `task_summary.json`. The summary records current scores, seeds before/after
every pruning node, actual timestep, call budget, and selected image identity.
It reports only the time spent in completed sampling segments and scoring;
this is not a standardized end-to-end runtime/VRAM benchmark.

`resume_state.pt` is atomically replaced after each seed reaches a node and
after pruning. It stores continuation latents and PNDM multistep state on CPU;
it is removed after the task summary and selected image are verified. A killed
incomplete seed segment is rerun from its previous saved node. Such discarded
attempts and model loading are excluded from the completed-trajectory budget.
Only load resume files created by this runner in a trusted output directory.

Rerun the identical command after interruption: completed tasks are verified
and skipped, remaining tasks restore their latest saved segment. Numeric
`--resume-from 874` skips rows before 874; only use it when earlier tasks have
already completed, or the final coverage audit will report them missing.
Configuration/source changes require a fresh output directory.

The audited scheduler-metadata repair is compatible with the original runner
revision: Diffusers can reorder the private `_use_default_values` list across
Python processes. Continuation now compares all public scheduler parameters
and the full timestep list, ignoring private metadata. Only the exact known
legacy source hashes are accepted for this migration, with every other run
setting still checked. The original run config and task hashes remain intact;
`resume_compatibility_fix.json` records the old and current source hashes.

`RUN_COMPLETE` plus `missing_report.json` with `complete: true` confirms coverage
of the requested task range. `manifest.json` indexes task summaries. Errors
stop execution, preserve resumable state and write per-task `error.json`.
Re-auditing an interrupted run lists all missing/invalid task IDs without GPU
inference. No existing experiment outputs are overwritten.

## Verification

```powershell
python -m unittest fashion_psp.experiments.test_difashion_segmented_psp -v
```

The tests run the actual DiFashion generation loop and actual PNDMScheduler
with small deterministic neural stubs. They compare serialized/interleaved
segmented trajectories to uninterrupted sampling, check exact forward counts,
terminal latent versus x_prediction, ties and invalid scores, and reject
changed scheduler configurations. They do not replace a pretrained-model GPU
smoke run. All test-set outputs remain exploratory.

The real pretrained-model smoke run also completed task 00000_9239_17895 in
`DiFashion-main/online_fitb_psp6_personal_25_75_100_smoke_v1/`, with actual
forward count 206 and selected seed 2031. Node image counts were 6/4/2.
Both surviving terminal PNGs (seeds 2030 and 2031) have identical file hashes
to their original full-trajectory offline-cache terminal PNGs.
Rerunning the identical command verified the completed task and skipped model
loading. Full-dataset preflight found all 2207 FITB conditions and valid
Personal history vectors. No full-dataset generation was launched.
