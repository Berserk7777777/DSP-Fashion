"""Declarative backbone defaults; actual scheduler state remains authoritative."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    sampler_family: Literal["diffusion", "flow_matching"]
    default_step_argument: int
    expected_loop_calls: int
    default_schedule: Tuple[Tuple[int, int], ...]
    upstream_entrypoints: Tuple[str, ...]
    clean_prediction_semantics: str

    def validate_loop_count(self, actual_loop_calls: int) -> None:
        if actual_loop_calls != self.expected_loop_calls:
            raise ValueError(
                f"{self.name} expected {self.expected_loop_calls} loop calls for native "
                f"step argument {self.default_step_argument}, received {actual_loop_calls}. "
                "Resolve a new schedule explicitly; do not silently reuse this default."
            )


BACKBONE_SPECS: Dict[str, BackboneSpec] = {
    "difashion": BackboneSpec(
        name="DiFashion",
        sampler_family="diffusion",
        default_step_argument=50,
        expected_loop_calls=51,
        default_schedule=((13, 4), (26, 2), (51, 1)),
        upstream_entrypoints=(
            "DiFashion-main/DiFashion/models/difashion.py::DiFashion.fashion_generation",
            "DiFashion-main/tools/collect_ifashion_fitb_candidates.py",
        ),
        clean_prediction_semantics="pred_ori_sample_given_epsilon before PNDM scheduler.step",
    ),
    "dualfashion": BackboneSpec(
        name="DualFashion",
        sampler_family="flow_matching",
        default_step_argument=28,
        expected_loop_calls=28,
        default_schedule=((7, 4), (14, 2), (28, 1)),
        upstream_entrypoints=(
            "DualFashion-main/run_psp_ifashion_smoke.py",
            "DualFashion-main/Inference_ifashion_PFITB.py",
        ),
        clean_prediction_semantics="x0 estimated from a flow-matching scheduler transition",
    ),
    "fashiondpo": BackboneSpec(
        name="FashionDPO",
        sampler_family="diffusion",
        default_step_argument=50,
        expected_loop_calls=50,
        default_schedule=((13, 4), (25, 2), (50, 1)),
        upstream_entrypoints=(
            "FashionDPO-main/fashiondpo/sample.py",
            "FashionDPO-main/fashiondpo/inf4eval.py",
        ),
        clean_prediction_semantics="DDIM predicted clean sample; adapter implementation pending",
    ),
}


@dataclass(frozen=True)
class RunConfig:
    """Configuration shared by a search policy and a specific backbone run."""

    backbone: str
    method: Literal["standard", "bon", "psp"]
    num_particles: int
    schedule: Tuple[Tuple[int, int], ...]
    seed_base: int
    seed_stride: int = 10_000

    def validate(self, actual_loop_calls: int) -> None:
        if self.backbone not in BACKBONE_SPECS:
            raise ValueError(f"unknown backbone: {self.backbone}")
        if self.num_particles < 1 or self.seed_stride < self.num_particles:
            raise ValueError("invalid particle count or seed stride")
        previous_step = 0
        previous_keep = self.num_particles
        for step, keep in self.schedule:
            if not 1 <= step <= actual_loop_calls:
                raise ValueError(f"schedule step {step} outside [1, {actual_loop_calls}]")
            if step <= previous_step or not 1 <= keep <= previous_keep:
                raise ValueError("schedule must have increasing nodes and non-increasing keep counts")
            previous_step, previous_keep = step, keep
        if self.method == "psp" and (not self.schedule or self.schedule[-1] != (actual_loop_calls, 1)):
            raise ValueError("PSP schedule must end with a single final survivor")
