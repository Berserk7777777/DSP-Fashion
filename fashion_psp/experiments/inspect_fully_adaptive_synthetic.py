"""Save mechanism examples, NOT a benchmark or a policy-selection experiment."""
import json
from pathlib import Path

from fashion_psp.search.deferred_boundary_psp import PartialCausalObserver
from fashion_psp.search.fully_adaptive_psp import replay

ROOT = Path(__file__).resolve().parents[2]


def main():
    examples = {}
    for name, top, delayed in (("steady_top3", 3, False), ("steady_top4", 4, False),
                                ("steady_top5", 5, False), ("later_stabilization", 3, True),
                                ("all_ties", 8, False)):
        p = {(t, s): float(10 if s < top else 9 if delayed and t == 5 else 0)
             for t in range(1, 29) for s in range(8)}
        observer = PartialCausalObserver(p, range(8))
        result = replay(list(range(8)), observer)
        examples[name] = {
            "particle_steps": result["particle_steps"],
            "decode_images": result["decode_images"],
            "personal_evaluations": result["personal_evaluations"],
            "scoring_rounds": result["scoring_rounds"],
            "clip_batches_padded4": result["clip_batches_padded4"],
            "clip_image_slots_with_padding": 4*result["clip_batches_padded4"],
            "observation_steps": [r["step"] for r in result["trace"] if r["observed"]],
            "pruning_events": [{k: r[k] for k in ("step", "count_before", "count_after",
                                                   "removed_reasons", "cumulative_particle_steps")}
                               for r in result["trace"] if r["count_before"] > r["count_after"]],
            "trace": result["trace"],
        }
        assert len(observer.accesses) == result["decode_images"]
    payload = {
        "status": "SYNTHETIC_MECHANISM_EXAMPLES_ONLY",
        "real_trajectory_reads": 0, "real_test_policy_evaluations": 0,
        "parameter_search_runs": 0,
        "interpretation": "Shows dynamic times/counts and costs. No evidence of real Personal/hit improvement. Synthetic scores are not diffusion simulations.",
        "examples": examples,
    }
    target = ROOT/"fashion_psp/artifacts/fully_adaptive_psp_v1/synthetic_examples.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, r in examples.items():
        internal = [(e["step"], e["count_after"]) for e in r["pruning_events"] if e["step"] < 28]
        print(f"SYNTHETIC {name}: prune={internal}; denoise={r['particle_steps']}; views={r['decode_images']}; rounds={r['scoring_rounds']}")
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
