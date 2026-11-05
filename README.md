# DSP-Fashion

**Inference-Time Scaling for Preference Alignment in Generative Fashion Recommendation**

DSP (**Delayed Seed Pruning**) searches random seeds while delaying candidate
elimination until later denoising stages. The current DualFashion DSP-6 uses
`6 -> 4 -> 2 -> 1` candidates at steps `7, 21, 28`, with 112 particle-denoising
steps. This configuration has fixed pruning nodes; it is not adaptive pruning.

## Projects

```text
DSP-Fashion/
├── DiFashion-main/      # Backbone and local inference modifications
├── DualFashion-main/    # Backbone and local inference modifications
├── FashionDPO-main/     # FashionDPO reference implementation
├── fashion_psp/         # Shared experiments, reward evaluation, and replay
├── environments/        # Core versions from the working environments
├── scripts/             # Packaging and GitHub upload helpers
└── SERVER_DEPLOYMENT.md
```

The Python package remains `fashion_psp` to preserve existing imports. The
original `PSP-main` reference checkout is not included; current FITB offline
collection and metric evaluation do not require it. Keep the backbones in
separate Python environments.

## Deployment and offline evaluation

See [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md) for setup and external assets.
Dataset arrays/images, model weights, generated caches, and environments are
transferred separately rather than included in Git.

- DiFashion strict50: BoN saves only step 50; PSP saves steps 1--50. This
  protocol uses PNDM argument 49 to execute exactly 50 UNet calls. The old
  argument-50 protocol executes 51 calls and is a different experiment.
- DualFashion: BoN saves only step 28; PSP saves steps 1--28.
- Both collections support 16 candidates, groups 0/1/2, original task ordering,
  and concurrent task workers. The seed mapping is
  `base_seed + group_id*1000000 + original_task_index*10000 + slot`, with
  base 42 for BoN and 2026 for PSP.
- Offline metrics: BoN terminal images get CS/CIS/LPIPS/Compatibility/Personal;
  PSP gets Compatibility/Personal at every step and CS/CIS/LPIPS at the final
  step. SQLite, CSV, and Excel exports support partial pools and resume.

Commands and tables:
[OFFLINE_FITB_POOL_METRICS.md](fashion_psp/evaluation/OFFLINE_FITB_POOL_METRICS.md).
Historical contracts: [fashion_psp/README.md](fashion_psp/README.md).

## Evaluation boundaries

Pruning uses only the current task's surviving candidates and reached reward
observations. GT and CS/CIS/LPIPS never enter pruning. Final-score oracles and
regret are posthoc diagnostics, not policy results. The previously examined
200 tasks belong to the test set; exploratory analysis on them is not
independent validation and must not be used to fit pruning thresholds.

## Upstream attribution

The backbone directories retain their upstream readmes and paper citations.
Refer to them for terms and attribution. This repository does not assign a
new blanket license to third-party implementations.
