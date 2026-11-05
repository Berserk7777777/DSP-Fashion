# `fashion_psp` Module

`fashion_psp` is the shared inference orchestration layer for DiFashion,
DualFashion, FashionDPO, and later baselines.  It owns protocols that must be
identical across backbones; it does **not** own a diffusion model, scheduler,
or dataset-specific preprocessing implementation.

## Environment Boundary

The three backbones must remain in separate environments.  DiFashion,
DualFashion, and FashionDPO have incompatible dependency pins, particularly
for PyTorch, Diffusers, and Transformers.  Install each upstream project in
its own conda environment and activate that environment before invoking its
adapter.  The shared package deliberately has no runtime import of a backbone
or model library, allowing the same seed, schedule, artifact, and evaluation
contracts to be reused from each environment.

## Design Rules

- A backbone adapter retains its native scheduler, default inference argument,
  precision, conditioning path, and image decoder.
- The adapter resolves actual loop calls at runtime.  A requested step count
  is never assumed to equal the scheduler loop count.
- The search engine only receives decoded reward views and stable particle IDs.
  It never reads a GT item, test retrieval candidates, or a post-hoc metric.
- Pruning is atomic: the same survivor indices are applied to latents,
  conditioning tensors, scheduler state, generators, seeds, and trace rows.
- An experiment records all seed mappings, schedule nodes, reward versions,
  model checkpoints, actual forward-call counts, and selected image paths.
- Offline replay and online PSP are different claims.  Replay is for rapid
  reward research; only online PSP can establish runtime/compute savings.

## Layout

```text
fashion_psp/
├── backbones/       # Native-generator adapters and declared defaults
├── search/          # Seed pools, schedules, deterministic BoN/PSP policy
├── rewards/         # Reward interface and registry only
├── artifacts/       # Run/task/candidate manifests and atomic writers
├── evaluation/      # Metric contract, not a replacement evaluator
├── baselines/       # Extension point for FK-Steering, BFS, DSearch, ...
├── config.py        # Backbone specs and run configuration validation
└── contracts.py     # Adapter, particle, reward-view protocols
```

## Adapter Contract

An adapter needs to implement the following lifecycle.  The state is opaque to
the PSP core so that PNDM, DDIM, and flow-matching schedulers remain native.

```python
prepare_task(task) -> PreparedTask
initialize(prepared, particles) -> SamplingState
step(state) -> StepOutput
make_reward_view(state, step_output) -> list[RewardView]
prune(state, survivor_indices) -> SamplingState
finalize(state) -> list[FinalArtifact]
```

`make_reward_view` returns a decoded predicted-clean image at intermediate
nodes.  `finalize` returns the terminal scheduler decode.  These are not
interchangeable.  The DiFashion offline protocol demonstrates why: the final
PNDM `x_prediction` is a pre-update estimate, whereas the terminal decode is
the actual generated result.

## Default Step Semantics

| Adapter | Native argument | Runtime loop count | Historical PSP nodes |
|---|---:|---:|---|
| DiFashion | 50 | 51 | 13, 26, final 51 |
| DualFashion | 28 | 28 | 7, 14, final 28 |
| FashionDPO | 50 | validate at runtime | proposed 13, 25, final 50 |

`FashionDPO` must validate that DDIM actually has 50 loop calls before its
default schedule is used.  A schedule is represented as node/keep pairs, not
as hard-coded percentages.

The module computes and stores both the actual PSP forward count and the BoN
reference count.  For a schedule with nodes `(s1, s2, T)` and counts
`(N, K1, K2)`, PSP calls are:

```text
N * s1 + K1 * (s2 - s1) + K2 * (T - s2)
```

For historical DiFashion nodes `13, 26, 51`, this is 206 calls for
`8 -> 4 -> 2 -> 1`, while BoN@4 is 204 calls.  The runner must report this
fact; it must not label the comparison exact-budget by default.

## Existing Integration Anchors

| Backend | Native entry point | Existing reusable evidence |
|---|---|---|
| DiFashion | `DiFashion-main/DiFashion/models/difashion.py` | `tools/collect_ifashion_fitb_candidates.py`, `tools/offline_fitb_select_and_evaluate.py` |
| DualFashion | `DualFashion-main/run_psp_ifashion_smoke.py` | offline cache, selectors, official evaluator |
| FashionDPO | `FashionDPO-main/fashiondpo/sample.py` / `inf4eval.py` | 50-step DDIM candidate generation and its expert evaluators |

Adapters should call or minimally extract these paths.  They should not fork
the models into a fourth incompatible implementation.

## Implementation Sequence

1. Preserve existing offline collectors/selectors and export their data into
   the common artifact schema.
2. Integrate DiFashion online first, with a one-task trace test proving that
   `prune()` reindexes every batch-aligned field and preserves a survivor's
   non-pruned trajectory exactly.
3. Integrate the DualFashion flow-matching adapter and verify its `x0`
   semantics independently.
4. Integrate FashionDPO's DDIM sampler after confirming the released checkpoint
   and evaluator assets.
5. Add baselines through `baselines/`, with their own environments and a common
   final-artifact manifest.  FK-Steering is not a PSP adapter; it is a
   competing search policy.

## Frozen adaptive Personal replay

`experiments/replay_dualfashion_fitb_adaptive_personal.py` compares the default
7/14/28 PSP@8 with a frozen, task-local three-step Personal stability heuristic
under an exact112 particle-step budget. It audits the original8 metadata seed
mapping, guards causal observations, and separates posthoc oracle/Compatibility
evaluation from selection. This is a development-set cache replay.

```powershell
python -m fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal
```

See `experiments/ADAPTIVE_PERSONAL_REPLAY.md` for rule semantics, budget
reachability, validation and reproducibility. Outputs, including per-task
images/seeds, pruning traces and separate decode/reward cost counts, are in
`artifacts/adaptive_personal_causal_replay_v1/`.

## Adaptive Personal development search

`experiments/search_adaptive_personal_development.py` tunes global adaptive
pruning heuristics on all200 development tasks, with the original8 seeds and
exact112 particle steps. This is explicitly a tuned development experiment,
separate from the frozen v1 replay. The selected policy improves both mean
Personal and oracle-seed hit rate on these200 tasks; it is not an independent
test result. All trials, the selected policy, causal traces and verification
artifacts are in `artifacts/adaptive_personal_development_search_v2/`.
See `experiments/ADAPTIVE_PERSONAL_DEVELOPMENT_SEARCH.md` for the actual results,
final rule and commands to reproduce the search or replay only the saved policy.

## PCAR Reward Training

`rewards/train_pcar.py` trains a small frozen-at-inference reward for iFashion.
It uses the actual target item in `processed/train.npy` as its positive and
samples negatives only from the same target category.  The visual backbone is
the already supplied frozen `cnn_features_clip.npy`; DualFashion itself is
not trained or invoked.

The reward input is the mean target-category user history, the three observed
outfit-item features, and the target category.  It must be trained only on
the training split.  FITB/GOR GT remains unavailable to the later selector.

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP

python .\fashion_psp\rewards\train_pcar.py `
  --data-root C:\tmp\ifashion `
  --output-dir .\fashion_psp\artifacts\pcar\ifashion_train_v2 `
  --max-outfits 102642 `
  --examples-per-outfit 1 `
  --num-negatives 15 `
  --epochs 30 `
  --batch-size 512 `
  --device cuda:0 `
  --num-workers 0 `
  --seed 2026
```

The script deterministically withholds 5% of training outfits.  It writes
`last.pt` after every epoch, `best.pt` only when held-out InfoNCE loss improves,
and `training_log.jsonl` with both train and held-out metrics.  It stops after
four non-improving validation epochs by default.  The first run creates an
approximately 400 MiB `feature_cache` for 102,642 examples, allowing resumed
runs to avoid rebuilding user-history and context means.  After an
interruption, keep all data and architecture arguments the same, omit
`--rebuild-cache`, and set a larger terminal epoch count:

```powershell
python .\fashion_psp\rewards\train_pcar.py `
  --data-root C:\tmp\ifashion `
  --output-dir .\fashion_psp\artifacts\pcar\ifashion_train_v2 `
  --max-outfits 102642 `
  --examples-per-outfit 1 `
  --num-negatives 15 `
  --epochs 40 `
  --batch-size 512 `
  --device cuda:0 `
  --num-workers 0 `
  --seed 2026 `
  --resume .\fashion_psp\artifacts\pcar\ifashion_train_v2\last.pt
```
