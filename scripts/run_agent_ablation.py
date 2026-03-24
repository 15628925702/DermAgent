#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import LearnableSkillController
from agent.evidence_calibrator import LearnableEvidenceCalibrator
from agent.final_scorer import LearnableFinalScorer
from agent.rule_scorer import LearnableRuleScorer
from agent.run_agent import run_agent
from datasets.splits import load_or_create_split_manifest, select_split_cases
from evaluation.case_selection import resolve_eval_cases
from evaluation.run_eval import load_dataset_cases, normalize_dataset_type
from integrations.openai_client import OpenAICompatClient
from memory.controller_store import load_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index
from scripts.compare_agent_vs_qwen import MALIGNANT_LABELS, QwenDirectInference


ABLATION_CONFIGS = [
    {
        "name": "agent_perception_only",
        "title": "Agent perception only",
        "use_retrieval": False,
        "use_compare": False,
        "use_malignancy": False,
        "use_metadata_consistency": False,
        "use_specialist": False,
        "use_controller": False,
        "no_harm_mode": "off",
    },
    {
        "name": "agent_plus_compare",
        "title": "+ compare",
        "use_retrieval": False,
        "use_compare": True,
        "use_malignancy": False,
        "use_metadata_consistency": False,
        "use_specialist": False,
        "use_controller": False,
        "no_harm_mode": "off",
    },
    {
        "name": "agent_plus_metadata",
        "title": "+ metadata",
        "use_retrieval": False,
        "use_compare": True,
        "use_malignancy": False,
        "use_metadata_consistency": True,
        "use_specialist": False,
        "use_controller": False,
        "no_harm_mode": "off",
    },
    {
        "name": "agent_plus_malignancy",
        "title": "+ malignancy",
        "use_retrieval": False,
        "use_compare": True,
        "use_malignancy": True,
        "use_metadata_consistency": True,
        "use_specialist": False,
        "use_controller": False,
        "no_harm_mode": "off",
    },
    {
        "name": "agent_plus_specialists",
        "title": "+ specialists",
        "use_retrieval": False,
        "use_compare": True,
        "use_malignancy": True,
        "use_metadata_consistency": True,
        "use_specialist": True,
        "use_controller": False,
        "no_harm_mode": "off",
    },
    {
        "name": "agent_plus_controller",
        "title": "+ controller/scorers",
        "use_retrieval": False,
        "use_compare": True,
        "use_malignancy": True,
        "use_metadata_consistency": True,
        "use_specialist": True,
        "use_controller": True,
        "no_harm_mode": "off",
    },
    {
        "name": "agent_full",
        "title": "Full agent (+ retrieval)",
        "use_retrieval": True,
        "use_compare": True,
        "use_malignancy": True,
        "use_metadata_consistency": True,
        "use_specialist": True,
        "use_controller": True,
        "no_harm_mode": "off",
    },
    {
        "name": "agent_full_no_harm",
        "title": "Full agent (+ retrieval, no-harm)",
        "use_retrieval": True,
        "use_compare": True,
        "use_malignancy": True,
        "use_metadata_consistency": True,
        "use_specialist": True,
        "use_controller": True,
        "no_harm_mode": "conservative",
    },
]


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _confidence_to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    mapping = {"low": 0.33, "medium": 0.66, "high": 0.90}
    return mapping.get(str(value or "").strip().lower(), 0.5)


def _extract_top_k_labels(final_decision: Dict[str, Any], top_n: int = 3) -> List[str]:
    labels: List[str] = []
    for item in (final_decision or {}).get("top_k", [])[:top_n]:
        label = _norm_label(item.get("name")) if isinstance(item, dict) else _norm_label(item)
        if label:
            labels.append(label)
    return labels


def _resolve_agent_components(checkpoint_path: str | None, enable_controller: bool) -> Dict[str, Any]:
    if checkpoint_path:
        (
            skill_index,
            controller_payload,
            final_scorer_payload,
            rule_scorer_payload,
            evidence_calibrator_payload,
        ) = load_controller_checkpoint(checkpoint_path)
    else:
        skill_index = build_default_skill_index()
        controller_payload = {}
        final_scorer_payload = {}
        rule_scorer_payload = {}
        evidence_calibrator_payload = {}

    controller = LearnableSkillController(skill_index) if enable_controller else None
    final_scorer = LearnableFinalScorer() if enable_controller else None
    rule_scorer = LearnableRuleScorer() if enable_controller else None
    evidence_calibrator = LearnableEvidenceCalibrator() if enable_controller else None

    if controller is not None and controller_payload:
        controller.load_state(controller_payload)
    if final_scorer is not None and final_scorer_payload:
        final_scorer.load_state(final_scorer_payload)
    if rule_scorer is not None and rule_scorer_payload:
        rule_scorer.load_state(rule_scorer_payload)
    if evidence_calibrator is not None and evidence_calibrator_payload:
        evidence_calibrator.load_state(evidence_calibrator_payload)

    return {
        "skill_index": skill_index,
        "controller": controller,
        "final_scorer": final_scorer,
        "rule_scorer": rule_scorer,
        "evidence_calibrator": evidence_calibrator,
    }


def _direct_top_k(pred: Dict[str, Any]) -> List[str]:
    raw_top_k = pred.get("raw_response", {}).get("top_k")
    if isinstance(raw_top_k, list):
        return [_norm_label(item) for item in raw_top_k[:3] if _norm_label(item)]
    pred_label = _norm_label(pred.get("diagnosis"))
    return [pred_label] if pred_label else []


def _build_direct_row(*, case: Dict[str, Any], pred: Dict[str, Any], idx: int) -> Dict[str, Any]:
    true_label = _norm_label(case.get("label"))
    pred_label = _norm_label(pred.get("diagnosis"))
    top3 = _direct_top_k(pred)
    error = pred.get("error")
    return {
        "case_id": case.get("file", f"case_{idx}"),
        "true_label": true_label,
        "predicted_label": pred_label,
        "top3": top3,
        "confidence": _confidence_to_float(pred.get("confidence")),
        "is_top1_correct": (not error) and pred_label == true_label,
        "is_top3_correct": (not error) and true_label in top3,
        "is_malignant_case": true_label in MALIGNANT_LABELS,
        "malignant_recalled": (not error) and true_label in MALIGNANT_LABELS and pred_label in MALIGNANT_LABELS,
        "error": error,
    }


def _build_agent_row(*, case: Dict[str, Any], result: Dict[str, Any], idx: int) -> Dict[str, Any]:
    true_label = _norm_label(case.get("label"))
    final_decision = result.get("final_decision", {}) or {}
    pred_label = _norm_label(final_decision.get("final_label") or final_decision.get("diagnosis"))
    top3 = _extract_top_k_labels(final_decision, top_n=3)
    planner = result.get("planner", {}) or {}
    planner_flags = planner.get("flags", {}) or {}
    aggregator_debug = (final_decision.get("aggregator_debug", {}) or {})
    no_harm_debug = aggregator_debug.get("no_harm_guard", {}) or {}
    evidence_summary = final_decision.get("evidence_summary", {}) or {}
    perception = result.get("perception", {}) or {}
    perception_top1 = _norm_label((perception.get("most_likely", {}) or {}).get("name"))
    if not perception_top1:
        top_names = [
            _norm_label(item.get("name"))
            for item in (perception.get("ddx_candidates", []) or [])[:1]
            if isinstance(item, dict)
        ]
        perception_top1 = top_names[0] if top_names else ""
    error = result.get("error")
    return {
        "case_id": case.get("file", f"case_{idx}"),
        "true_label": true_label,
        "predicted_label": pred_label,
        "top3": top3,
        "confidence": _confidence_to_float(final_decision.get("confidence")),
        "confidence_text": final_decision.get("confidence", "low"),
        "is_top1_correct": (not error) and pred_label == true_label,
        "is_top3_correct": (not error) and true_label in top3,
        "is_malignant_case": true_label in MALIGNANT_LABELS,
        "malignant_recalled": (not error) and true_label in MALIGNANT_LABELS and pred_label in MALIGNANT_LABELS,
        "error": error,
        "selected_skills": list(result.get("selected_skills", []) or []),
        "planner_mode": planner.get("planning_mode"),
        "stop_probability": planner.get("stop_probability"),
        "retrieval_confidence": planner_flags.get("retrieval_confidence"),
        "used_sources": list(evidence_summary.get("used_sources", []) or []),
        "perception_top1": perception_top1,
        "perception_correct": bool(perception_top1) and perception_top1 == true_label,
        "changed_from_perception_top1": bool(perception_top1) and pred_label != perception_top1,
        "no_harm_applied": bool(no_harm_debug.get("applied", False)),
        "no_harm_reason": no_harm_debug.get("reason", ""),
        "safety_override_applied": bool((aggregator_debug.get("safety_override", {}) or {}).get("applied", False)),
        "aggregator_top_scores": dict((aggregator_debug.get("candidate_scores", {}) or {})),
    }


def _compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(rows)
    valid = [row for row in rows if not row.get("error")]
    malignant_rows = [row for row in valid if row.get("is_malignant_case")]
    malignant_hits = sum(1 for row in malignant_rows if row.get("malignant_recalled"))
    return {
        "accuracy_top1": sum(1 for row in valid if row.get("is_top1_correct")) / len(valid) if valid else 0.0,
        "accuracy_top3": sum(1 for row in valid if row.get("is_top3_correct")) / len(valid) if valid else 0.0,
        "malignant_recall": malignant_hits / len(malignant_rows) if malignant_rows else 0.0,
        "error_rate": (total - len(valid)) / max(total, 1),
    }


def _run_agent_variant(
    *,
    cases: List[Dict[str, Any]],
    config: Dict[str, Any],
    checkpoint_path: str | None,
    bank_state_in: str | None,
    perception_model: str,
    report_model: str,
) -> Dict[str, Any]:
    components = _resolve_agent_components(checkpoint_path, enable_controller=bool(config["use_controller"]))
    bank = ExperienceBank.from_json(bank_state_in) if bank_state_in else ExperienceBank()
    reranker = UtilityAwareExperienceReranker()
    rows: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases):
        if idx % 10 == 0:
            print(f"  {config['title']} progress: {idx}/{len(cases)}")
        result = run_agent(
            case=case,
            bank=bank,
            skill_index=components["skill_index"],
            reranker=reranker,
            learning_components={
                "controller": components["controller"],
                "final_scorer": components["final_scorer"],
                "rule_scorer": components["rule_scorer"],
                "evidence_calibrator": components["evidence_calibrator"],
            } if config["use_controller"] else {},
            use_retrieval=config["use_retrieval"],
            use_specialist=config["use_specialist"],
            use_reflection=False,
            use_controller=config["use_controller"],
            use_compare=config["use_compare"],
            use_malignancy=config["use_malignancy"],
            use_metadata_consistency=config["use_metadata_consistency"],
            use_final_scorer=config["use_controller"],
            update_online=False,
            use_rule_memory=True,
            enable_rule_compression=True,
            perception_model=perception_model,
            report_model=report_model,
            no_harm_mode=config.get("no_harm_mode", "off"),
        )
        rows.append(_build_agent_row(case=case, result=result, idx=idx))

    return {
        "name": config["name"],
        "title": config["title"],
        "config": config,
        "metrics": _compute_metrics(rows),
        "per_case": rows,
        "total_cases": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
    }


def _run_direct_qwen(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline = QwenDirectInference()
    predictions = baseline.batch_infer(cases)
    rows = [
        _build_direct_row(case=case, pred=pred, idx=idx)
        for idx, (case, pred) in enumerate(zip(cases, predictions))
    ]
    return {
        "name": "direct_qwen",
        "title": "Direct Qwen",
        "metrics": _compute_metrics(rows),
        "per_case": rows,
        "total_cases": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "model": getattr(baseline.client, "model", OpenAICompatClient().model) if baseline.available else OpenAICompatClient().model,
    }


def _per_case_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("case_id")): row for row in rows}


def _route_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    skill_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    for row in rows:
        skill_counts.update(str(x) for x in (row.get("selected_skills", []) or []) if str(x).strip())
        source_counts.update(str(x) for x in (row.get("used_sources", []) or []) if str(x).strip())
        pred = _norm_label(row.get("predicted_label"))
        if pred:
            pred_counts[pred] += 1
    return {
        "selected_skill_counts": dict(skill_counts.most_common()),
        "used_source_counts": dict(source_counts.most_common()),
        "predicted_label_counts": dict(pred_counts.most_common()),
    }


def _harm_analysis(variant_rows: List[Dict[str, Any]], direct_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    direct_index = _per_case_index(direct_rows)
    harmed_cases: List[Dict[str, Any]] = []
    helped_cases: List[Dict[str, Any]] = []
    harmful_skills: Counter[str] = Counter()
    harmful_sources: Counter[str] = Counter()
    helped_skills: Counter[str] = Counter()

    changed_from_direct = 0
    direct_correct_variant_wrong = 0
    direct_wrong_variant_correct = 0
    perception_correct_variant_wrong = 0
    changed_from_perception_top1 = 0
    no_harm_applied = 0

    for row in variant_rows:
        case_id = str(row.get("case_id"))
        direct_row = direct_index.get(case_id, {})
        direct_correct = bool(direct_row.get("is_top1_correct"))
        variant_correct = bool(row.get("is_top1_correct"))
        if _norm_label(row.get("predicted_label")) != _norm_label(direct_row.get("predicted_label")):
            changed_from_direct += 1
        if row.get("changed_from_perception_top1"):
            changed_from_perception_top1 += 1
        if row.get("no_harm_applied"):
            no_harm_applied += 1
        if row.get("perception_correct") and not variant_correct:
            perception_correct_variant_wrong += 1
        if direct_correct and not variant_correct:
            direct_correct_variant_wrong += 1
            harmful_skills.update(str(x) for x in (row.get("selected_skills", []) or []) if str(x).strip())
            harmful_sources.update(str(x) for x in (row.get("used_sources", []) or []) if str(x).strip())
            harmed_cases.append(
                {
                    "case_id": case_id,
                    "true_label": row.get("true_label"),
                    "direct_pred": direct_row.get("predicted_label"),
                    "variant_pred": row.get("predicted_label"),
                    "perception_top1": row.get("perception_top1"),
                    "selected_skills": row.get("selected_skills", []),
                    "used_sources": row.get("used_sources", []),
                    "retrieval_confidence": row.get("retrieval_confidence"),
                    "stop_probability": row.get("stop_probability"),
                    "no_harm_reason": row.get("no_harm_reason"),
                }
            )
        elif (not direct_correct) and variant_correct:
            direct_wrong_variant_correct += 1
            helped_skills.update(str(x) for x in (row.get("selected_skills", []) or []) if str(x).strip())
            helped_cases.append(
                {
                    "case_id": case_id,
                    "true_label": row.get("true_label"),
                    "direct_pred": direct_row.get("predicted_label"),
                    "variant_pred": row.get("predicted_label"),
                    "perception_top1": row.get("perception_top1"),
                    "selected_skills": row.get("selected_skills", []),
                    "used_sources": row.get("used_sources", []),
                }
            )

    return {
        "changed_from_direct_count": changed_from_direct,
        "direct_correct_variant_wrong_count": direct_correct_variant_wrong,
        "direct_wrong_variant_correct_count": direct_wrong_variant_correct,
        "perception_correct_variant_wrong_count": perception_correct_variant_wrong,
        "changed_from_perception_top1_count": changed_from_perception_top1,
        "no_harm_applied_count": no_harm_applied,
        "harmful_skill_counts": dict(harmful_skills.most_common()),
        "harmful_source_counts": dict(harmful_sources.most_common()),
        "helped_skill_counts": dict(helped_skills.most_common()),
        "sample_harmed_cases": harmed_cases[:20],
        "sample_helped_cases": helped_cases[:20],
    }


def _load_cases(
    *,
    dataset_type: str | None,
    dataset_root: str,
    split_json: str | None,
    split_name: str | None,
    seed: int,
    test_limit: int | None,
    case_manifest_in: str | None,
) -> Dict[str, Any]:
    return resolve_eval_cases(
        dataset_type=dataset_type,
        dataset_root=dataset_root,
        split_json=split_json,
        split_name=split_name,
        seed=seed,
        limit=test_limit,
        case_manifest_in=case_manifest_in,
        case_manifest_out=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged ablation and baseline-harm analysis for DermAgent.")
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--dataset-type", default=None, choices=["pad_ufes20", "pad_ufes_20", "ham10000"])
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-checkpoint", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--perception-model", default=None)
    parser.add_argument("--report-model", default=None)
    parser.add_argument("--case-manifest-in", default=None, help="Replay the exact evaluation cases used by compare_agent_vs_qwen.py.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_client = OpenAICompatClient()
    direct_baseline = QwenDirectInference()
    direct_model = getattr(direct_baseline.client, "model", "") if direct_baseline.available else ""
    perception_model = str(args.perception_model or direct_model or base_client.model)
    report_model = str(args.report_model or perception_model)

    print("Loading evaluation cases...")
    case_bundle = _load_cases(
        dataset_type=args.dataset_type,
        dataset_root=args.dataset_root,
        split_json=args.split_json,
        split_name=args.split_name,
        seed=args.seed,
        test_limit=args.test_limit,
        case_manifest_in=args.case_manifest_in,
    )
    cases = case_bundle["cases"]
    resolved_dataset_type = case_bundle["dataset_type"]
    resolved_split_path = case_bundle["resolved_split_path"]
    case_manifest_path = case_bundle["case_manifest_path"]
    print(f"  dataset_type: {resolved_dataset_type}")
    print(f"  cases: {len(cases)}")

    print("Running Direct Qwen baseline...")
    direct_result = _run_direct_qwen(cases)
    results: List[Dict[str, Any]] = [direct_result]

    for config in ABLATION_CONFIGS:
        print(f"Running {config['title']}...")
        results.append(
            _run_agent_variant(
                cases=cases,
                config=config,
                checkpoint_path=args.controller_checkpoint,
                bank_state_in=args.bank_state_in,
                perception_model=perception_model,
                report_model=report_model,
            )
        )

    direct_rows = direct_result["per_case"]
    for item in results[1:]:
        item["route_stats"] = _route_stats(item["per_case"])
        item["harm_analysis_vs_direct"] = _harm_analysis(item["per_case"], direct_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_type": resolved_dataset_type,
        "dataset_root": args.dataset_root,
        "split_json": resolved_split_path,
        "split_name": args.split_name,
        "seed": args.seed,
        "test_count": len(cases),
        "case_manifest_path": case_manifest_path,
        "perception_model": perception_model,
        "report_model": report_model,
        "controller_checkpoint": args.controller_checkpoint,
        "bank_state_in": args.bank_state_in,
        "results": results,
    }

    json_path = output_dir / f"ablation_report_{timestamp}.json"
    csv_path = output_dir / f"ablation_metrics_{timestamp}.csv"
    txt_path = output_dir / f"ablation_summary_{timestamp}.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stage",
                "top1_accuracy",
                "top3_accuracy",
                "malignant_recall",
                "error_rate",
                "direct_correct_variant_wrong",
                "direct_wrong_variant_correct",
                "perception_correct_variant_wrong",
                "changed_from_perception_top1",
                "no_harm_applied",
            ]
        )
        for item in results:
            metrics = item["metrics"]
            harm = item.get("harm_analysis_vs_direct", {})
            writer.writerow(
                [
                    item["title"],
                    f"{metrics['accuracy_top1']:.4f}",
                    f"{metrics['accuracy_top3']:.4f}",
                    f"{metrics['malignant_recall']:.4f}",
                    f"{metrics['error_rate']:.4f}",
                    harm.get("direct_correct_variant_wrong_count", ""),
                    harm.get("direct_wrong_variant_correct_count", ""),
                    harm.get("perception_correct_variant_wrong_count", ""),
                    harm.get("changed_from_perception_top1_count", ""),
                    harm.get("no_harm_applied_count", ""),
                ]
            )

    lines: List[str] = []
    lines.append("DermAgent Staged Ablation")
    lines.append(f"timestamp: {report['timestamp']}")
    lines.append(f"dataset_type: {resolved_dataset_type}")
    lines.append(f"dataset_root: {args.dataset_root}")
    lines.append(f"split_json: {resolved_split_path}")
    lines.append(f"split_name: {args.split_name}")
    lines.append(f"seed: {args.seed}")
    lines.append(f"test_count: {len(cases)}")
    lines.append("")
    for item in results:
        metrics = item["metrics"]
        lines.append(item["title"])
        lines.append(f"  top1_accuracy: {metrics['accuracy_top1']:.4f}")
        lines.append(f"  top3_accuracy: {metrics['accuracy_top3']:.4f}")
        lines.append(f"  malignant_recall: {metrics['malignant_recall']:.4f}")
        lines.append(f"  error_rate: {metrics['error_rate']:.4f}")
        harm = item.get("harm_analysis_vs_direct")
        if harm:
            lines.append(f"  direct_correct_variant_wrong: {harm['direct_correct_variant_wrong_count']}")
            lines.append(f"  direct_wrong_variant_correct: {harm['direct_wrong_variant_correct_count']}")
            lines.append(f"  perception_correct_variant_wrong: {harm['perception_correct_variant_wrong_count']}")
            lines.append(f"  changed_from_perception_top1: {harm['changed_from_perception_top1_count']}")
            lines.append(f"  no_harm_applied: {harm['no_harm_applied_count']}")
            harmful_skills = ", ".join(f"{k}={v}" for k, v in list((harm.get("harmful_skill_counts") or {}).items())[:6]) or "none"
            harmful_sources = ", ".join(f"{k}={v}" for k, v in list((harm.get("harmful_source_counts") or {}).items())[:6]) or "none"
            lines.append(f"  harmful_skills: {harmful_skills}")
            lines.append(f"  harmful_sources: {harmful_sources}")
        lines.append("")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nAblation summary")
    for item in results:
        metrics = item["metrics"]
        harm = item.get("harm_analysis_vs_direct", {})
        summary = (
            f"  {item['title']}: top1={metrics['accuracy_top1']:.4f} "
            f"top3={metrics['accuracy_top3']:.4f} "
            f"mal_recall={metrics['malignant_recall']:.4f} "
            f"error={metrics['error_rate']:.4f}"
        )
        if harm:
            summary += (
                f" harm={harm.get('direct_correct_variant_wrong_count', 0)}"
                f" help={harm.get('direct_wrong_variant_correct_count', 0)}"
            )
        print(summary)
    print(f"\nSaved ablation report to {json_path}")
    print(f"Saved ablation metrics to {csv_path}")
    print(f"Saved ablation summary to {txt_path}")


if __name__ == "__main__":
    main()
