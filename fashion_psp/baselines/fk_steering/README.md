# FK-Steering Baseline

This directory adapts the *policy* of PSP-main's Feynman-Kac Diffusion
Steering (FKD) to fashion FITB.  It deliberately does not import or vendor
`PSP-main/Fk-Diffusion-Steering`: its custom pipelines target text-to-image
Diffusers versions and cannot safely replace DiFashion's PNDM or
DualFashion's flow-matching sampler.

## Difference from PSP

| Property | PSP | FK-Steering |
|---|---|---|
| Particle count | Decreases by hard pruning | Stays constant at `K` |
| Parent choice | Deterministic top-k without replacement | Stochastic resampling, normally with replacement |
| Search behavior | Explores more initial seeds, then allocates later compute to survivors | Reallocates a fixed particle population toward high-reward ancestors |
| Native model calls | Depends on pruning schedule | `K * T` regardless of resampling events |

For a fair first comparison, use `K=4` for FK-Steering and BoN@4.  With
DualFashion's 28 steps, both use 112 denoiser calls.  Compare PSP@8 with its
equal-budget `7 -> 14 -> 28`, `8 -> 4 -> 2 -> 1` schedule.  For DiFashion,
use an explicitly reported schedule/budget: its historical `13,26,51` PSP
schedule has 206 calls, whereas FK@4 and BoN@4 have 204.

## Fashion Reward Interface

At a resampling node, the adapter decodes each particle's predicted-clean
view and passes it with the fixed FITB condition to a frozen reward function:

```text
x_t^i -> predicted clean image x0_hat^i -> reward(x0_hat^i, outfit context, user history)
```

The initial experiments should separately use `compatibility`, `personal`,
and frozen `bottleneck` reward configurations.  GT item/image, retrieval
candidates, GT CLIP similarity, LPIPS-to-GT, and GT Attribute Alignment are
strictly post-hoc and prohibited here.

Raw reward scales change across models and intermediate denoising nodes.  Do
not tune `lambda` on the test set.  Choose potential type, `lambda`, resampling
nodes, and ESS threshold on validation data, then freeze them for test runs.

## Required Adapter Operation

FK resampling draws a parent index for every output particle.  An adapter must
implement a `resample_with_parents(state, parent_indices)` operation that
copies/reindexes **all** per-particle state:

```text
current latent
previous latent / x0 estimate, if retained
all CFG, category, mutual, and history conditions aligned to particles
per-particle generator state
native scheduler history buffers and step counter
particle seed, ancestor id, and trace state
```

This is more demanding than PSP's subset prune because parent indices may
contain duplicates.  PNDM is multi-step: its internal model-output history
must be copied by parent index, not left as an unmodified batch tensor.

## Deterministic-Sampler Gate

The released DiFashion PNDM path and the current DualFashion flow-matching
path are effectively deterministic after initial-noise sampling.  If a parent
is copied and no subsequent stochastic transition is introduced, its children
will remain pixel-identical.  In that situation ordinary with-replacement FK
resampling can reduce effective sample size without creating new trajectories.

Before reporting FK-Steering, run this gate on at least 20 validation tasks:

1. resample one high-scoring particle into two children at an intermediate
   node;
2. finish both children with the native sampler;
3. calculate a pixel hash and LPIPS between children; and
4. log the duplicate fraction and number of distinct terminal images.

If children are identical, do **not** present that result as a meaningful FK
baseline.  The defensible options are to use a validated stochastic transition
kernel (with its noise process and compute reported), add a formally specified
rejuvenation kernel as a separate method, or report a no-replacement variant
under a distinct name rather than calling it standard FK-Steering.

## DualFashion Controlled Stochastic Flow Solver

DualFashion's released `FlowMatchEulerDiscreteScheduler` accepts stochastic
arguments but its `step` implementation is deterministic.  The online FK
runner therefore follows the controlled stochastic Euler wrapper used for
SD3.5 in the PSP paper.  At **every** Euler transition, after the native
deterministic update, every active particle gets an independent perturbation:

```text
x_det_next = x + (sigma_next - sigma) * velocity
x_next = x_det_next + sigma * sqrt((1 + gamma)^2 - 1) * Normal(0, I)
```

This is enabled only by `--fk-controlled-stochastic`; `gamma=0` recovers the
native deterministic solver.  The candidate reward remains well-defined:
the x0 estimate is calculated directly as `x - sigma * velocity` *before*
the stochastic transition, never by differencing its perturbed output.

`gamma` must be calibrated on held-out iFashion validation tasks.  The PSP
paper found `gamma=0.005` for SD3.5, but that value is only an initial
candidate for DualFashion, not a transferred hyperparameter.  Every run
records sigma, noise scale, and a distinct reproducible seed for every
particle-step in `metadata.json`; resampling traces retain the parent-child
lineage separately.

## DiFashion Online Diagnostic

`run_difashion_fitb_fk.py` is the implemented DiFashion adapter for one FITB
task. It runs four particles together in the native 50-argument / 51-loop
PNDM sampler and reindexes the latent batch plus `ets`, `cur_sample`, and
`cur_model_output` whenever a parent is selected. It saves decoded clean
predictions, scores, normalized weights, ESS, parent indices, lineage, and
terminal image hashes.

The default intermediate nodes are actual one-based loop calls `10, 20, 31,
41`, the rounded 20/40/60/80 percent points of DiFashion's 51 PNDM calls.
Loop 51 is the FKD-style terminal forced-resampling decision, not an extra
denoising call. This is distinct from DiFashion PSP's historical `13,26,51`
hard-pruning schedule.

The runner currently exposes frozen Personal Sim only. Its `lambda` is
required at the command line because PSP-main's ImageReward temperature is
not transferable to Personal Sim's 0--100 scale. A run is explicitly
exploratory until the potential, temperature, node schedule, and resampling
strategy have been frozen on validation tasks.

## Initial Protocol

1. Implement and unit-test the pure weighting/resampling policy in this
   directory.
2. Use the DiFashion online diagnostic to compare unresampled and resampled
   traces, including its PNDM history state.
3. Run the deterministic-sampler gate.
4. Tune only on FITB validation data, using fixed seeds independent of PSP and
   BoN pools.
5. Run one frozen test group, producing candidate ancestry and reward traces.
6. Use the existing official/post-hoc evaluators only after final selection.

Every task writes a trace with `node`, raw reward, potential, normalized
weight, ESS, resampling RNG seed, parent index, ancestor seed, and final
selected seed.  This makes ancestry collapse measurable rather than hidden.
