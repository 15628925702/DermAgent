from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_compare import run_compare
from scripts.run_ablations import run_ablations
from scripts.run_eval import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact experiment suite on DermAgent checkpoints.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--controller-state-in", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--compare-limit", type=int, default=10)
    parser.add_argument("--enable-controller", action="store_true")
    parser.add_argument("--enable-final-scorer", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    full_eval = run_evaluation(
        dataset_root=args.dataset_root,
        limit=args.limit,
        split_json=args.split_json,
        split_name=args.split_name,
        use_retrieval=True,
        use_specialist=True,
        use_reflection=False,
        use_controller=args.enable_controller,
        use_final_scorer=args.enable_final_scorer and args.enable_controller,
        controller_state_in=args.controller_state_in,
        bank_state_in=args.bank_state_in,
        update_online=False,
        use_rule_memory=True,
        enable_rule_compression=False,
    )
    ablations = run_ablations(
        dataset_root=args.dataset_root,
        limit=args.limit,
        split_json=args.split_json,
        split_name=args.split_name,
        controller_state_in=args.controller_state_in,
        bank_state_in=args.bank_state_in,
    )

    result: Dict[str, Any] = {
        "full_eval": {
            "metrics": full_eval.get("metrics", {}),
            "counts": full_eval.get("counts", {}),
            "split": full_eval.get("split", {}),
            "runtime_flags": full_eval.get("runtime_flags", {}),
        },
        "ablations": ablations,
    }

    if not args.skip_compare:
        compare_result = run_compare(
            dataset_root=args.dataset_root,
            limit=args.compare_limit,
            include_per_case=False,
            controller_state_in=args.controller_state_in,
            bank_state_in=args.bank_state_in,
            split_json=args.split_json,
            split_name=args.split_name,
            use_controller=args.enable_controller,
            use_final_scorer=args.enable_final_scorer and args.enable_controller,
        )
        result["compare"] = {
            "baseline_direct_gpt": compare_result.get("baseline_direct_gpt", {}),
            "agent_architecture": compare_result.get("agent_architecture", {}),
            "delta": compare_result.get("delta", {}),
            "compare_valid": compare_result.get("compare_valid", False),
        }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
