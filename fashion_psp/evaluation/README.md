# Evaluation Integration

FashionPSP standardizes the input manifest and evaluation policy.  It does not
replace verified backbone-specific evaluators until the same frozen assets and
preprocessing are proven equivalent.

## Canonical Input

Every evaluator consumes selected final images through a manifest with at
least:

```text
task_id, task_index, uid, oid, target_category,
method, reward, group_id, seed, image_path, selected
```

Candidate-level records additionally contain the reward node, reward score,
and whether the image is an intermediate predicted-clean decode or a terminal
scheduler decode.  Official metrics use terminal selected images only.

## Existing Evaluator Reuse

| Backend | Existing source | Reuse policy |
|---|---|---|
| DiFashion | `DiFashion-main/tools/offline_fitb_select_and_evaluate.py` | Canonical source for its FITB condition parsing, compatibility/personal scoring, selected-image metrics, integrity reports, and per-group aggregation. |
| DiFashion | `DiFashion-main/append_standard_gt_sim.py` and its GT-attribute experiment outputs | Post-hoc only. GT information cannot enter rewards or selection. |
| DualFashion | `DualFashion-main/evaluate_PFITB.py` | Canonical source for its official Inception, compatibility, LPIPS, and personalization metrics. |
| DualFashion | `DualFashion-main/evaluate_gt_attribute_alignment.py` | Independent post-hoc GAA diagnostic; it must remain separate from selection. |
| FashionDPO | `FashionDPO-main/evaluation/multiple_evaluate.py` | Preserve its Quality/VBPR/Personal definitions as FashionDPO-specific supplementary metrics until assets and calibration are aligned. |

## Cross-Backbone Main Metrics

For a common iFashion FITB table, only compare scores computed with the same
frozen assets and preprocessing.  The existing protocol in
`FashionPSP/unified_evaluation_protocol.md` identifies compatibility,
personalization, category accuracy, retrieval, and post-hoc GT diagnostics.

The evaluator must always emit:

```text
run_manifest.json
candidate_metrics.csv
selected_metrics.csv
task_metrics.csv
group_summary.csv
integrity_report.json
```

The integrity report checks task coverage, duplicate/missing seeds, candidate
counts, selected image readability, finite metrics, forbidden GT access during
selection, and the relation between selected rows and final image paths.

## DualFashion iFashion GOR Offline Cross-Validation

`evaluate_dualfashion_gor_offline.py` evaluates the completed GOR offline
caches as complete sequential-outfit trajectories. It supports the 200-task
caches in `DualFashion-main/output/ifashion_gor/`:

- BoN: four complete trajectories, scored at step 28;
- offline PSP: eight complete trajectories cached at steps `7,14,28`, with
  reward-specific replay of the `8 -> 4 -> 2 -> 1` schedule.

The selector always computes all three frozen candidate metrics:

- `compatibility`: sigmoid of the released iFashion `FashionEvaluator` over
  four generated OpenCLIP ViT-H/14 item embeddings;
- `personal`: mean of four per-category user-history cosine scores, scaled by
  100;
- `bottleneck`: minimum of Compatibility and Personal within-active-set
  percentiles. It is a selection objective, not a calibrated measurement.

GT is never loaded for candidate scoring or seed selection. Passing
`--with-gt-metrics` freezes all reward-specific selections first, then
computes post-hoc `gt_sim` (matched-position ViT-H/14 similarity) and `gaa`
(independent OpenAI CLIP ViT-L/14 zero-shot attribute-distribution overlap).

Run under the DualFashion environment:

```powershell
conda activate dualfashion
Set-Location E:\difashion-PSP
powershell -ExecutionPolicy Bypass -File .\fashion_psp\evaluation\run_dualfashion_gor_cross_validation.ps1
```

Results go to `DualFashion-main/output/ifashion_gor/evaluation/{bon,psp}`.
Each root contains frozen task selections for the three reward functions,
`selected_outfits_with_gt_metrics.csv`, and `summary.csv`. For a faster
selection-only pass, add `-SkipGtMetrics` to the PowerShell runner.

This is an offline reward study, not an online GOR PSP compute-savings claim:
the cache has already completed each sequential outfit trajectory. The pruning
unit is one whole trajectory, never a single GOR item.
