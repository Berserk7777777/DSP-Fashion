"""Record fixed6/7/21 rule and code hashes without loading trajectory data."""
import hashlib
import json
from pathlib import Path

from fashion_psp.search.fixed_six_seed_psp import RULE

ROOT = Path(__file__).resolve().parents[2]


def main():
    sources = {"rule_source_sha256": ROOT/"fashion_psp/search/fixed_six_seed_psp.py",
               "online_source_sha256": ROOT/"fashion_psp/experiments/run_dualfashion_fitb_six_seed_7_21.py"}
    record = {"rule": RULE, **{k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in sources.items()},
              "status": "user_specified_fixed_schedule_online_implementation",
              "data_access_by_this_command": "None; source files only"}
    target = ROOT/"fashion_psp/artifacts/fixed_six_seed_7_21/frozen_rule.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and json.loads(target.read_text(encoding="utf-8")) != record:
        raise ValueError("Existing frozen rule/source differs; do not overwrite silently")
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"Frozen fixed6/7/21 online policy: {target}")


if __name__ == "__main__":
    main()
