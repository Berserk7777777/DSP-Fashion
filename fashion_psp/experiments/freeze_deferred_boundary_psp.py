"""Freeze a prototype WITHOUT loading any real candidate data or evaluating it."""
import json
from pathlib import Path

from fashion_psp.search.deferred_boundary_psp import RULE
from fashion_psp.experiments.replay_dualfashion_fitb_adaptive_personal import sha256,write_json

ROOT=Path(__file__).resolve().parents[2]


def main():
    out=ROOT/"fashion_psp/artifacts/deferred_boundary_psp_v1"
    out.mkdir(parents=True,exist_ok=True)
    record={"rule":RULE,
            "rule_source_sha256":sha256(ROOT/"fashion_psp/search/deferred_boundary_psp.py"),
            "gap_gate_source_sha256":sha256(ROOT/"fashion_psp/search/gap_budget_psp.py"),
            "online_source_sha256":sha256(ROOT/"fashion_psp/experiments/run_dualfashion_fitb_deferred_boundary.py"),
            "status":"frozen_prototype_synthetic_only_not_real_test_evaluated",
            "data_access_by_this_command":"None; source files only",
            "evaluation_boundary":"No repeated rule selection on the already-inspected200 test tasks. Independent validation/confirmation data are not yet identified.",
            "new_real_data_policy_evaluations":0,"parameter_search_runs":0}
    target=out/"frozen_rule.json"
    if target.is_file() and json.loads(target.read_text(encoding="utf-8"))!=record:
        raise ValueError("Do not silently replace an existing different frozen prototype")
    write_json(target,record)
    print(f"Prototype frozen; no trajectory data loaded: {target}")


if __name__=="__main__":main()
