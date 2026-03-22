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
from evaluation.run_eval import load_pad_ufes20_cases
from integrations.openai_client import OpenAICompatClient
from memory.controller_store import load_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index

MALIGNANT_LABELS = {"MEL", "BCC", "SCC"}
ACK_SCC_LABELS = {"ACK", "SCC"}


def run_compare(
    dataset_root: str | Path = "data/pad_ufes_20",
    limit: int | None = None,
    include_per_case: bool = False,
    controller_state_in: str | Path | None = None,
    bank_state_in: str | Path | None = None,
    split_json: str | None = None,
    split_name: str | None = None,
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
    agent = _run_agent_variant(cases, controller_state_in=controller_state_in, bank_state_in=bank_state_in)
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


def _run_direct_gpt_baseline(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    client = OpenAICompatClient()
    per_case: List[Dict[str, Any]] = []
    for case in cases:
        error = None
        diagnosis = ""
        top3: List[str] = []
        try:
            raw = client.infer_derm_direct_diagnosis(image_path=str(case.get("image_path", "")), metadata=case.get("metadata", {}) or {})
            parsed = json.loads((raw or "{}").replace("```json", "").replace("```", "").strip())
            diagnosis = _norm_label(parsed.get("diagnosis"))
            top3 = _normalize_top_k(parsed.get("top_k", []))
            if diagnosis and diagnosis not in top3:
                top3.insert(0, diagnosis)
                top3 = top3[:3]
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        per_case.append({"case_id": case.get("file", ""), "true_label": _norm_label(case.get("label")), "predicted_label": diagnosis, "top3": top3, "error": error})
    return {"summary": _summarize_predictions(per_case), "per_case": per_case}


def _run_agent_variant(cases: List[Dict[str, Any]], *, controller_state_in: str | Path | None = None, bank_state_in: str | Path | None = None) -> Dict[str, Any]:
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
    controller = LearnableSkillController(skill_index)
    if controller_payload:
        controller.load_state(controller_payload)
    final_scorer = LearnableFinalScorer()
    if final_scorer_payload:
        final_scorer.load_state(final_scorer_payload)
        final_scorer_loaded = True
    rule_scorer = LearnableRuleScorer()
    if rule_scorer_payload:
        rule_scorer.load_state(rule_scorer_payload)
        rule_scorer_loaded = True

    per_case: List[Dict[str, Any]] = []
    for case in cases:
        result = run_agent(
            case=case,
            bank=bank,
            skill_index=skill_index,
            reranker=reranker,
            learning_components={
                "controller": controller,
                "final_scorer": final_scorer,
                "rule_scorer": rule_scorer,
            },
            use_retrieval=True,
            use_specialist=True,
            use_reflection=False,
            use_controller=True,
            update_online=False,
            use_rule_memory=True,
            enable_rule_compression=False,
            update_rule_scorer=False,
        )
        final_decision = result.get("final_decision", {}) or {}
        per_case.append({
            "case_id": case.get("file", ""),
            "true_label": _norm_label(case.get("label")),
            "predicted_label": _norm_label(final_decision.get("final_label") or final_decision.get("diagnosis")),
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
        },
    }


def _summarize_predictions(per_case: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(per_case)
    correct_top1 = correct_top3 = malignant_total = malignant_hit = ack_scc_total = ack_scc_correct = errors = 0
    for item in per_case:
        true_label = _norm_label(item.get("true_label"))
        pred_label = _norm_label(item.get("predicted_label"))
        top3 = _normalize_top_k(item.get("top3", []))
        if item.get("error"):
            errors += 1
        if pred_label == true_label:
            correct_top1 += 1
        if true_label in top3:
            correct_top3 += 1
        if true_label in MALIGNANT_LABELS:
            malignant_total += 1
            if pred_label in MALIGNANT_LABELS:
                malignant_hit += 1
        if true_label in ACK_SCC_LABELS:
            ack_scc_total += 1
            if pred_label == true_label:
                ack_scc_correct += 1
    return {
        "metrics": {
            "accuracy_top1": _safe_div(correct_top1, total),
            "accuracy_top3": _safe_div(correct_top3, total),
            "malignant_recall": _safe_div(malignant_hit, malignant_total),
            "confusion_accuracy": _safe_div(ack_scc_correct, ack_scc_total),
        },
        "counts": {
            "correct_top1": correct_top1,
            "correct_top3": correct_top3,
            "malignant_total": malignant_total,
            "malignant_hit": malignant_hit,
            "ack_scc_total": ack_scc_total,
            "ack_scc_correct": ack_scc_correct,
            "errors": errors,
        },
    }


def _build_delta(baseline_metrics: Dict[str, Any], agent_metrics: Dict[str, Any]) -> Dict[str, float]:
    keys = ["accuracy_top1", "accuracy_top3", "malignant_recall", "confusion_accuracy"]
    return {key: round(float(agent_metrics.get(key, 0.0)) - float(baseline_metrics.get(key, 0.0)), 4) for key in keys}


def _merge_per_case(baseline_cases: List[Dict[str, Any]], agent_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "case_id": b.get("case_id"),
        "true_label": b.get("true_label"),
        "baseline_pred": b.get("predicted_label"),
        "agent_pred": a.get("predicted_label"),
        "baseline_top3": b.get("top3", []),
        "agent_top3": a.get("top3", []),
        "baseline_error": b.get("error"),
        "agent_error": a.get("error"),
    } for b, a in zip(baseline_cases, agent_cases)]


def _build_error_summary(per_case: List[Dict[str, Any]]) -> Dict[str, Any]:
    error_examples: List[Dict[str, Any]] = []
    error_types: Dict[str, int] = {}
    for item in per_case:
        raw_error = item.get("error")
        if raw_error is None:
            continue
        error = str(raw_error).strip()
        if not error:
            continue
        error_type = error.split(":", 1)[0].strip() if ":" in error else error
        error_types[error_type] = error_types.get(error_type, 0) + 1
        if len(error_examples) < 3:
            error_examples.append({"case_id": item.get("case_id", ""), "error": error})
    return {"num_error_cases": sum(error_types.values()), "error_types": error_types, "examples": error_examples}


def _normalize_top_k(value: Any) -> List[str]:
    items = value if isinstance(value, list) else []
    out: List[str] = []
    for item in items:
        label = _norm_label(item.get("name")) if isinstance(item, dict) else _norm_label(item)
        if label and label not in out:
            out.append(label)
    return out[:3]


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _safe_div(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare direct GPT baseline and agent architecture.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default=None, choices=["train", "val", "test"])
    parser.add_argument("--include-per-case", action="store_true")
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
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
