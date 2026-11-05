"""Freeze source and rule only; does NOT load real trajectories or tune policy."""
import hashlib
import json
from pathlib import Path

from fashion_psp.search.fully_adaptive_psp import RULE

ROOT = Path(__file__).resolve().parents[2]


def main():
    paths = {"rule_source_sha256": ROOT/"fashion_psp/search/fully_adaptive_psp.py",
             "online_source_sha256": ROOT/"fashion_psp/experiments/run_dualfashion_fitb_fully_adaptive.py"}
    record = {"rule": RULE, **{k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()},
              "status": "frozen_prototype_synthetic_only_not_real_test_evaluated",
              "data_access_by_this_command": "None; source files only",
              "parameter_search_runs": 0, "new_real_data_policy_evaluations": 0}
    target = ROOT/"fashion_psp/artifacts/fully_adaptive_psp_v1/frozen_rule.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and json.loads(target.read_text(encoding="utf-8")) != record:
        raise ValueError("Do not silently replace an existing different frozen rule")
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Frozen without reading any real trajectories: {target}")


if __name__ == "__main__":
    main()
