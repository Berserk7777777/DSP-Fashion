"""Guardrails for common evaluation rather than another evaluator implementation."""

SELECTION_FORBIDDEN_INPUTS = frozenset({
    "ground_truth_item", "ground_truth_image", "test_grd.npy", "retrieval_candidates",
    "gt_attribute_alignment", "gt_clip_similarity", "lpips_to_ground_truth",
})

POSTHOC_ONLY_METRICS = frozenset({
    "is_acc", "retrieval_accuracy", "ground_truth_clip_similarity",
    "ground_truth_lpips", "gt_attribute_alignment", "fid", "inception_score",
})
