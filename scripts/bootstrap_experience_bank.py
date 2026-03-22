#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.run_agent import run_agent
from datasets.splits import load_or_create_split_manifest, select_split_cases
from evaluation.run_eval import load_pad_ufes20_cases
from integrations.openai_client import OpenAICompatClient
from memory.compressor import ExperienceCompressor
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _run_pass(
    *,
    cases: List[Dict[str, Any]],
    bank: ExperienceBank,
    reranker: UtilityAwareExperienceReranker,
    skill_index: Any,
    perception_model: str,
    report_model: str,
    use_retrieval: bool,
    use_specialist: bool,
    pass_name: str,
) -> Dict[str, Any]:
    correct_top1 = 0
    errors = 0

    for idx, case in enumerate(cases):
        if idx % 25 == 0:
            print(f"  {pass_name}: {idx}/{len(cases)}")

        result = run_agent(
            case=case,
            bank=bank,
            skill_index=skill_index,
            reranker=reranker,
            learning_components={},
            use_retrieval=use_retrieval,
            use_specialist=use_specialist,
            use_reflection=True,
            use_controller=False,
            use_compare=True,
            use_malignancy=True,
            use_metadata_consistency=True,
            use_final_scorer=False,
            update_online=True,
            use_rule_memory=True,
            enable_rule_compression=True,
            perception_model=perception_model,
            report_model=report_model,
        )

        final_decision = result.get("final_decision", {}) or {}
        pred_label = _norm_label(final_decision.get("final_label") or final_decision.get("diagnosis"))
        true_label = _norm_label(case.get("label"))

        if result.get("error"):
            errors += 1
        elif pred_label == true_label:
            correct_top1 += 1

    compressor = ExperienceCompressor()
    compression = compressor.compress(bank, include_rules=True)
    return {
        "pass_name": pass_name,
        "num_cases": len(cases),
        "top1_accuracy": round(correct_top1 / len(cases), 4) if cases else 0.0,
        "errors": errors,
        "bank_stats": bank.stats(),
        "compression": compression,
        "used_retrieval": use_retrieval,
        "used_specialist": use_specialist,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap DermAgent experience bank offline from a split.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-name", default="train", choices=["train", "val", "test"])
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--bank-state-out", default="outputs/checkpoints/bootstrap_bank.json")
    parser.add_argument("--perception-model", default=None)
    parser.add_argument("--report-model", default=None)
    parser.add_argument("--disable-specialist", action="store_true")
    parser.add_argument("--output", default="outputs/train_runs/bootstrap_bank_summary.json")
    args = parser.parse_args()

    all_cases = load_pad_ufes20_cases(dataset_root=args.dataset_root, limit=args.limit)
    split_path = args.split_json or str(Path("outputs/splits") / f"{Path(args.dataset_root).name}_seed{args.seed}.json")
    split_payload = load_or_create_split_manifest(all_cases, split_path, seed=args.seed)
    cases = select_split_cases(all_cases, split_payload, args.split_name)
    if not cases:
        raise RuntimeError(f"Split '{args.split_name}' is empty.")

    base_client = OpenAICompatClient()
    perception_model = str(args.perception_model or base_client.model)
    report_model = str(args.report_model or perception_model)

    bank = ExperienceBank.from_json(args.bank_state_in) if args.bank_state_in else ExperienceBank()
    skill_index = build_default_skill_index()
    reranker = UtilityAwareExperienceReranker()

    history: List[Dict[str, Any]] = []
    num_passes = max(1, int(args.passes))
    for pass_idx in range(num_passes):
        use_retrieval = pass_idx > 0
        pass_name = f"pass_{pass_idx + 1}"
        print(f"Running {pass_name} on {len(cases)} {args.split_name} cases (retrieval={use_retrieval})")
        history.append(
            _run_pass(
                cases=cases,
                bank=bank,
                reranker=reranker,
                skill_index=skill_index,
                perception_model=perception_model,
                report_model=report_model,
                use_retrieval=use_retrieval,
                use_specialist=not args.disable_specialist,
                pass_name=pass_name,
            )
        )

    bank_path = bank.save_json(args.bank_state_out)
    summary = {
        "dataset_root": args.dataset_root,
        "split_name": args.split_name,
        "split_path": split_path,
        "num_cases": len(cases),
        "passes": num_passes,
        "perception_model": perception_model,
        "report_model": report_model,
        "history": history,
        "final_bank_stats": bank.stats(),
        "bank_state_out": str(bank_path),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved bank bootstrap summary to {output_path}")
    print(f"Saved bank checkpoint to {bank_path}")


if __name__ == "__main__":
    main()
