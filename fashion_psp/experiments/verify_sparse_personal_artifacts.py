"""Check repeated replay/search outputs and account for padded CLIP work."""
import json
from pathlib import Path
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "fashion_psp/artifacts/sparse_personal_development_v3"


def main():
    files = ["selected_policy.json", "summary.json", "step_trace.jsonl", "selections.csv",
             "budget_checks.csv", "fixed_schedule_ablation.json", "validation_results.csv"]
    matches = {}
    for folder in ("reproduction_check", "search_reproduction_check"):
        matches[folder] = {f:sha256(OUT/f)==sha256(OUT/folder/f) for f in files}
        assert all(matches[folder].values()), matches
    assert sha256(OUT/"search_trials.csv") == sha256(OUT/"search_reproduction_check/search_trials.csv")
    summary = json.loads((OUT/"summary.json").read_text())
    cost = {}
    sources = {"default":OUT/"step_trace.jsonl", "sparse":OUT/"step_trace.jsonl",
               "dense_v2":ROOT/"fashion_psp/artifacts/adaptive_personal_development_search_v2/step_trace.jsonl"}
    for name, path in sources.items():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        wanted = "default" if name=="default" else "adaptive"
        rows = [r for r in rows if r["method"]==wanted and r["observed"]]
        n = len({r["task_index"] for r in rows})
        views = sum(len(r["alive_before"]) for r in rows)
        batches = sum((len(r["alive_before"])+3)//4 for r in rows)
        cost[name] = {"tasks":n,"total_candidate_views":views,"mean_candidate_views":views/n,
                      "total_scoring_rounds":len(rows),"mean_scoring_rounds":len(rows)/n,
                      "total_clip_batches_padded4":batches,"mean_clip_image_slots_including_padding":4*batches/n}
    assert summary["methods"]["adaptive"]["all_tasks_exact112"]
    assert summary["methods"]["default"]["all_tasks_exact112"]
    scripts = ["search_sparse_personal.py", "run_dualfashion_fitb_sparse_personal.py",
               "run_dualfashion_fitb_adaptive_personal.py", "run_dualfashion_fitb_sparse_personal.ps1",
               "test_sparse_personal.py", "verify_sparse_personal_artifacts.py"]
    result = {"repeated_file_hash_matches":matches,"full_search_trials_hash_identical":True,
              "cost_accounting":cost,
              "source_hashes":{s:sha256(ROOT/"fashion_psp/experiments"/s) for s in scripts},
              "neural_inference_performed_by_this_check":False}
    write_json(OUT/"independent_verification.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
