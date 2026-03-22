#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
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
from evaluation.run_eval import load_pad_ufes20_cases
from integrations.openai_client import OpenAICompatClient
from memory.controller_store import load_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index
from scripts.compare_agent_vs_qwen import MALIGNANT_LABELS, QwenDirectInference


ABLATION_CONFIGS = [
    {
        "name": "agent_no_retrieval",
        "title": "Agent without retrieval",
        "use_retrieval": False,
        "use_compare": True,
        "use_malignancy": True,
        "use_metadata_consistency": False,
        "use_specialist": False,
        "use_controller": False,
    },
    {
        "name": "agent_plus_metadata",
        "title": "+ metadata",
        "use_retrieval": False,
        "use_compare": True,
        "use_malignancy": True,
        "use_metadata_consistency": True,
        "use_specialist": False,
        "use_controller": False,
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
    },
]


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


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

    correct_top1 = 0
    malignant_total = 0
    malignant_hit = 0
    errors = 0
    per_case: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases):
        if idx % 10 == 0:
            print(f"  {config['title']} 进度: {idx}/{len(cases)}")

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
        )

        true_label = _norm_label(case.get("label"))
        final_decision = result.get("final_decision", {}) or {}
        pred_label = _norm_label(final_decision.get("final_label") or final_decision.get("diagnosis"))
        top3 = _extract_top_k_labels(final_decision, top_n=3)
        has_error = result.get("error") is not None

        if has_error:
            errors += 1
        else:
            if pred_label == true_label:
                correct_top1 += 1
            if true_label in MALIGNANT_LABELS:
                malignant_total += 1
                if pred_label in MALIGNANT_LABELS:
                    malignant_hit += 1

        per_case.append(
            {
                "case_id": case.get("file", f"case_{idx}"),
                "true_label": true_label,
                "predicted_label": pred_label,
                "top3": top3,
                "error": result.get("error"),
            }
        )

    valid_cases = len(cases) - errors
    return {
        "config": config,
        "metrics": {
            "accuracy_top1": correct_top1 / valid_cases if valid_cases > 0 else 0.0,
            "malignant_recall": malignant_hit / malignant_total if malignant_total > 0 else 0.0,
            "error_rate": errors / max(len(cases), 1),
        },
        "per_case": per_case,
        "total_cases": len(cases),
        "errors": errors,
    }


def _run_direct_qwen(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline = QwenDirectInference()
    predictions = baseline.batch_infer(cases)

    correct_top1 = 0
    malignant_total = 0
    malignant_hit = 0
    errors = 0
    per_case: List[Dict[str, Any]] = []

    for idx, (case, pred) in enumerate(zip(cases, predictions)):
        true_label = _norm_label(case.get("label"))
        pred_label = _norm_label(pred.get("diagnosis"))
        has_error = pred.get("error") is not None
        raw_top_k = pred.get("raw_response", {}).get("top_k")
        if isinstance(raw_top_k, list):
            top3 = [_norm_label(item) for item in raw_top_k[:3] if _norm_label(item)]
        else:
            top3 = [pred_label] if pred_label else []

        if has_error:
            errors += 1
        else:
            if pred_label == true_label:
                correct_top1 += 1
            if true_label in MALIGNANT_LABELS:
                malignant_total += 1
                if pred_label in MALIGNANT_LABELS:
                    malignant_hit += 1

        per_case.append(
            {
                "case_id": case.get("file", f"case_{idx}"),
                "true_label": true_label,
                "predicted_label": pred_label,
                "top3": top3,
                "error": pred.get("error"),
            }
        )

    valid_cases = len(cases) - errors
    return {
        "name": "direct_qwen",
        "title": "Direct Qwen",
        "metrics": {
            "accuracy_top1": correct_top1 / valid_cases if valid_cases > 0 else 0.0,
            "malignant_recall": malignant_hit / malignant_total if malignant_total > 0 else 0.0,
            "error_rate": errors / max(len(cases), 1),
        },
        "per_case": per_case,
        "total_cases": len(cases),
        "errors": errors,
        "model": getattr(baseline.client, "model", OpenAICompatClient().model) if baseline.available else OpenAICompatClient().model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run staged ablation for DermAgent.")
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--controller-checkpoint", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--perception-model", default=None)
    parser.add_argument("--report-model", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_client = OpenAICompatClient()
    direct_baseline = QwenDirectInference()
    direct_model = getattr(direct_baseline.client, "model", "") if direct_baseline.available else ""
    perception_model = str(args.perception_model or direct_model or base_client.model)
    report_model = str(args.report_model or perception_model)

    print("加载对比样本...")
    cases = load_pad_ufes20_cases(limit=args.test_limit)
    print(f"  样本数: {len(cases)}")

    results: List[Dict[str, Any]] = []
    print("运行 Direct Qwen 基线...")
    results.append(_run_direct_qwen(cases))

    for config in ABLATION_CONFIGS:
        print(f"运行 {config['title']} ...")
        stage_result = _run_agent_variant(
            cases=cases,
            config=config,
            checkpoint_path=args.controller_checkpoint,
            bank_state_in=args.bank_state_in,
            perception_model=perception_model,
            report_model=report_model,
        )
        stage_result["name"] = config["name"]
        stage_result["title"] = config["title"]
        results.append(stage_result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_count": len(cases),
        "perception_model": perception_model,
        "report_model": report_model,
        "controller_checkpoint": args.controller_checkpoint,
        "bank_state_in": args.bank_state_in,
        "results": results,
    }

    json_path = output_dir / f"ablation_report_{timestamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / f"ablation_metrics_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "top1_accuracy", "malignant_recall", "error_rate"])
        for item in results:
            metrics = item["metrics"]
            writer.writerow(
                [
                    item["title"],
                    f"{metrics['accuracy_top1']:.4f}",
                    f"{metrics['malignant_recall']:.4f}",
                    f"{metrics['error_rate']:.4f}",
                ]
            )

    print("\nAblation summary")
    for item in results:
        metrics = item["metrics"]
        print(
            f"  {item['title']}: top1={metrics['accuracy_top1']:.4f} "
            f"mal_recall={metrics['malignant_recall']:.4f} error={metrics['error_rate']:.4f}"
        )
    print(f"\nSaved ablation report to {json_path}")
    print(f"Saved ablation metrics to {csv_path}")


if __name__ == "__main__":
    main()
