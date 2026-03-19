from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_eval import run_evaluation


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "full": {},
    "no_retrieval": {"use_retrieval": False},
    "no_specialist": {"use_specialist": False},
    "no_controller": {"use_controller": False},
    "no_rule_memory": {"use_rule_memory": False},
}


def run_ablations(
    *,
    dataset_root: str,
    limit: int | None,
    split_json: str | None,
    split_name: str | None,
    controller_state_in: str | None,
    bank_state_in: str | None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for name, overrides in ABLATIONS.items():
        result = run_evaluation(
            dataset_root=dataset_root,
            limit=limit,
            split_json=split_json,
            split_name=split_name,
            use_retrieval=overrides.get("use_retrieval", True),
            use_specialist=overrides.get("use_specialist", True),
            use_reflection=False,
            use_controller=overrides.get("use_controller", True),
            controller_state_in=controller_state_in,
            bank_state_in=bank_state_in,
            update_online=False,
            use_rule_memory=overrides.get("use_rule_memory", True),
            enable_rule_compression=False,
        )
        results[name] = {
            "metrics": result.get("metrics", {}),
            "counts": result.get("counts", {}),
            "runtime_flags": result.get("runtime_flags", {}),
            "split": result.get("split", {}),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation suite on DermAgent checkpoints.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--controller-state-in", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = run_ablations(
        dataset_root=args.dataset_root,
        limit=args.limit,
        split_json=args.split_json,
        split_name=args.split_name,
        controller_state_in=args.controller_state_in,
        bank_state_in=args.bank_state_in,
    )
    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
