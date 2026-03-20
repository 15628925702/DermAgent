from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import LearnableSkillController
from agent.final_scorer import LearnableFinalScorer
from agent.rule_scorer import LearnableRuleScorer
from agent.run_agent import run_agent
from datasets.splits import load_or_create_split_manifest, load_split_manifest, select_split_cases, summarize_split_cases
from evaluation.run_compare import _build_delta, _build_error_summary, _merge_per_case, _run_direct_gpt_baseline, _summarize_predictions
from evaluation.run_eval import load_pad_ufes20_cases
from memory.controller_store import load_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index


def run_compare(
    dataset_root: str | Path = "data/pad_ufes_20",
    limit: int | None = None,
    include_per_case: bool = False,
    controller_state_in: str | Path | None = None,
    bank_state_in: str | Path | None = None,
    split_json: str | None = None,
    split_name: str | None = None,
    use_controller: bool = False,
    use_compare: bool = True,
    use_malignancy: bool = True,
    use_metadata_consistency: bool = True,
    use_final_scorer: bool = False,
) -> Dict[str, Any]:
    all_cases = load_pad_ufes20_cases(dataset_root=dataset_root, limit=limit)
    resolved_split_path = split_json
    split_payload = None
    if split_json:
        split_payload = load_split_manifest(split_json)
    elif split_name:
        default_split_path = Path("outputs/splits") / f"{Path(dataset_root).name}_seed42.json"
        split_payload = load_or_create_split_manifest(all_cases, default_split_path, seed=42)
        resolved_split_path = str(default_split_path)

    cases = select_split_cases(all_cases, split_payload, split_name)
    if not split_name:
        cases = list(all_cases)

    baseline = _run_direct_gpt_baseline(cases)
    agent = _run_agent_variant(
        cases,
        controller_state_in=controller_state_in,
        bank_state_in=bank_state_in,
        use_controller=use_controller,
        use_compare=use_compare,
        use_malignancy=use_malignancy,
        use_metadata_consistency=use_metadata_consistency,
        use_final_scorer=use_final_scorer,
    )
    result: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "num_cases": len(cases),
        "split": {
            "name": split_name,
            "path": resolved_split_path,
            "summary": summarize_split_cases(cases),
        },
        "baseline_direct_gpt": baseline["summary"],
        "agent_architecture": agent["summary"],
        "agent_checkpoint": agent["checkpoint"],
        "compare_valid": baseline["summary"].get("counts", {}).get("errors", 0) == 0,
        "baseline_error_summary": _build_error_summary(baseline["per_case"]),
        "delta": _build_delta(baseline["summary"].get("metrics", {}), agent["summary"].get("metrics", {})),
    }
    if not result["compare_valid"]:
        result["warning"] = "Direct GPT baseline has execution errors. Current metric delta is not a fair comparison until baseline errors are resolved."
    if include_per_case:
        result["per_case"] = _merge_per_case(baseline["per_case"], agent["per_case"])
    return result


def _run_agent_variant(
    cases: List[Dict[str, Any]],
    *,
    controller_state_in: str | Path | None = None,
    bank_state_in: str | Path | None = None,
    use_controller: bool = False,
    use_compare: bool = True,
    use_malignancy: bool = True,
    use_metadata_consistency: bool = True,
    use_final_scorer: bool = False,
) -> Dict[str, Any]:
    checkpoint_loaded = False
    bank_loaded = False
    final_scorer_loaded = False
    rule_scorer_loaded = False

    bank = ExperienceBank.from_json(bank_state_in) if bank_state_in else ExperienceBank()
    bank_loaded = bool(bank_state_in)

    if controller_state_in:
        skill_index, controller_payload, final_scorer_payload, rule_scorer_payload = load_controller_checkpoint(controller_state_in)
        checkpoint_loaded = True
    else:
        skill_index = build_default_skill_index()
        controller_payload = {}
        final_scorer_payload = {}
        rule_scorer_payload = {}

    reranker = UtilityAwareExperienceReranker()
    controller = LearnableSkillController(skill_index) if use_controller else None
    if controller is not None and controller_payload:
        controller.load_state(controller_payload)
    final_scorer = LearnableFinalScorer() if use_controller and use_final_scorer else None
    if final_scorer is not None and final_scorer_payload:
        final_scorer.load_state(final_scorer_payload)
        final_scorer_loaded = True
    rule_scorer = LearnableRuleScorer() if use_controller else None
    if rule_scorer is not None and rule_scorer_payload:
        rule_scorer.load_state(rule_scorer_payload)
        rule_scorer_loaded = True

    per_case: List[Dict[str, Any]] = []
    for case in cases:
        result = run_agent(
            case=case,
            bank=bank,
            skill_index=skill_index,
            reranker=reranker,
            controller=controller,
            final_scorer=final_scorer,
            rule_scorer=rule_scorer,
            use_retrieval=True,
            use_specialist=True,
            use_reflection=False,
            use_controller=use_controller,
            use_compare=use_compare,
            use_malignancy=use_malignancy,
            use_metadata_consistency=use_metadata_consistency,
            use_final_scorer=use_final_scorer,
            update_online=False,
            use_rule_memory=True,
            enable_rule_compression=False,
            update_rule_scorer=False,
        )
        final_decision = result.get("final_decision", {}) or {}
        per_case.append({
            "case_id": case.get("file", ""),
            "true_label": str(case.get("label", "")).strip().upper(),
            "predicted_label": str(final_decision.get("final_label") or final_decision.get("diagnosis") or "").strip().upper(),
            "top3": _normalize_top_k(final_decision.get("top_k", [])),
            "error": result.get("error"),
        })

    return {
        "summary": _summarize_predictions(per_case),
        "per_case": per_case,
        "checkpoint": {
            "controller_loaded": checkpoint_loaded,
            "controller_input_path": str(controller_state_in) if controller_state_in else None,
            "bank_loaded": bank_loaded,
            "bank_input_path": str(bank_state_in) if bank_state_in else None,
            "final_scorer_loaded": final_scorer_loaded,
            "rule_scorer_loaded": rule_scorer_loaded,
            "use_controller": use_controller,
            "use_compare": use_compare,
            "use_malignancy": use_malignancy,
            "use_metadata_consistency": use_metadata_consistency,
            "use_final_scorer": use_final_scorer,
        },
    }


def _normalize_top_k(value: Any) -> List[str]:
    items = value if isinstance(value, list) else []
    out: List[str] = []
    for item in items:
        label = str(item.get("name", "")).strip().upper() if isinstance(item, dict) else str(item).strip().upper()
        if label and label not in out:
            out.append(label)
    return out[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct GPT baseline and agent architecture.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default=None, choices=["train", "val", "test"])
    parser.add_argument("--include-per-case", action="store_true")
    parser.add_argument("--enable-controller", action="store_true")
    parser.add_argument("--enable-final-scorer", action="store_true")
    parser.add_argument("--disable-compare", action="store_true")
    parser.add_argument("--disable-malignancy", action="store_true")
    parser.add_argument("--disable-metadata-consistency", action="store_true")
    parser.add_argument("--controller-state-in", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_compare(
        dataset_root=args.dataset_root,
        limit=args.limit,
        include_per_case=args.include_per_case,
        controller_state_in=args.controller_state_in,
        bank_state_in=args.bank_state_in,
        split_json=args.split_json,
        split_name=args.split_name,
        use_controller=args.enable_controller,
        use_compare=not args.disable_compare,
        use_malignancy=not args.disable_malignancy,
        use_metadata_consistency=not args.disable_metadata_consistency,
        use_final_scorer=args.enable_final_scorer and args.enable_controller,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
