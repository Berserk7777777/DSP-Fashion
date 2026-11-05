# Backbone Adapters

An adapter is a narrow bridge from a native generator to the interfaces in
`fashion_psp.contracts`.  It must not duplicate the model implementation.

## Required Runtime Evidence

Before an adapter is declared `online_ready`, it must produce a one-task trace
showing all of the following:

1. resolved scheduler timesteps and actual loop count;
2. initial seeds and initial latent checksums;
3. each intermediate `RewardView` image and its exact x0 semantics;
4. survivor seeds, scores, deterministic tie-breaking, and batch indices;
5. reindexing of every batch-aligned latent/condition/generator state; and
6. equality between a surviving particle's PSP trajectory and its unpruned
   native trajectory under the same deterministic configuration.

## Adapter-Specific Notes

### DiFashion

Native argument `num_inference_steps=50` resolves to 51 PNDM iterations in the
audited environment.  Intermediate reward views decode
`pred_ori_sample_given_epsilon(t, x_t_before_step, guided_noise_pred)`.  The
terminal image must decode the latent after the final scheduler update.

### DualFashion

Native default is 28 flow-matching steps.  The reward view is an x0 estimate
recovered from the actual scheduler transition; scheduler step-index handling
must be preserved after pruning.

### FashionDPO

Native `sample.py` and `inf4eval.py` specify 50 DDIM steps.  Confirm its loop
count, LoRA/checkpoint loading path, and predicted-clean decoding before
enabling online pruning.  Its existing multi-expert evaluator is not assumed
to be numerically interchangeable with the DiFashion/DualFashion evaluator.
