from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.splits import load_or_create_split_manifest, load_split_manifest, select_split_cases, summarize_split_cases
from evaluation.run_compare import (
    _build_delta,
    _build_error_summary,
    _merge_per_case,
    _run_agent_variant,
    _run_direct_gpt_baseline,
)
from evaluation.run_eval import load_pad_ufes20_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fast split-first compare on a small subset.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--pre-limit", type=int, default=None, help="Optional cap before split selection.")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-limit", type=int, default=40, help="Number of cases to compare after selecting the split.")
    parser.add_argument("--include-per-case", action="store_true")
    parser.add_argument("--controller-state-in", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    all_cases = load_pad_ufes20_cases(dataset_root=args.dataset_root, limit=args.pre_limit)
    resolved_split_path = args.split_json
    if args.split_json:
        split_payload = load_split_manifest(args.split_json)
    else:
        default_split_path = Path("outputs/splits") / f"{Path(args.dataset_root).name}_seed42.json"
        split_payload = load_or_create_split_manifest(all_cases, default_split_path, seed=42)
        resolved_split_path = str(default_split_path)

    cases = select_split_cases(all_cases, split_payload, args.split_name)
    if args.split_limit is not None and args.split_limit > 0:
        cases = cases[: args.split_limit]

    baseline = _run_direct_gpt_baseline(cases)
    agent = _run_agent_variant(
        cases,
        controller_state_in=args.controller_state_in,
        bank_state_in=args.bank_state_in,
    )
    result = {
        "dataset_root": str(args.dataset_root),
        "num_cases": len(cases),
        "split": {
            "name": args.split_name,
            "path": resolved_split_path,
            "summary": summarize_split_cases(cases),
        },
        "baseline_direct_gpt": baseline["summary"],
        "agent_architecture": agent["summary"],
        "agent_checkpoint": agent["checkpoint"],
        "compare_valid": baseline["summary"].get("counts", {}).get("errors", 0) == 0,
        "baseline_error_summary": _build_error_summary(baseline["per_case"]),
        "delta": _build_delta(
            baseline["summary"].get("metrics", {}),
            agent["summary"].get("metrics", {}),
        ),
    }
    if not result["compare_valid"]:
        result["warning"] = "Direct GPT baseline has execution errors. Current metric delta is not a fair comparison until baseline errors are resolved."
    if args.include_per_case:
        result["per_case"] = _merge_per_case(baseline["per_case"], agent["per_case"])

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
