#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
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
from datasets.ddi import (
    BENIGN_BINARY_LABEL,
    MALIGNANT_AGENT_LABELS,
    MALIGNANT_BINARY_LABEL,
    load_ddi_cases,
)
from integrations.openai_client import OpenAICompatClient
from memory.controller_store import load_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index
from scripts.compare_agent_vs_qwen import QwenDirectInference


VALID_LABELS = {"MEL", "BCC", "SCC", "NEV", "ACK", "SEK"}


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


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
    return mapping.get(_norm_text(value), 0.5)


def _binary_from_agent_label(value: Any) -> str:
    label = _norm_label(value)
    if label in VALID_LABELS:
        return MALIGNANT_BINARY_LABEL if label in MALIGNANT_AGENT_LABELS else BENIGN_BINARY_LABEL
    return "UNKNOWN"


def _compute_ece(items: List[Dict[str, Any]], field: str, num_bins: int = 10) -> Dict[str, Any]:
    if not items:
        return {"ece": 0.0, "bins": []}
    bins: List[List[Dict[str, Any]]] = [[] for _ in range(num_bins)]
    for item in items:
        conf = max(0.0, min(0.999999, _safe_float(item.get("confidence"), 0.5)))
        idx = min(num_bins - 1, int(conf * num_bins))
        bins[idx].append(item)

    ece = 0.0
    rendered_bins: List[Dict[str, Any]] = []
    total = len(items)
    for idx, bucket in enumerate(bins):
        if not bucket:
            continue
        avg_conf = sum(_safe_float(x.get("confidence"), 0.5) for x in bucket) / len(bucket)
        avg_acc = sum(1.0 if x.get(field) else 0.0 for x in bucket) / len(bucket)
        frac = len(bucket) / total
        ece += abs(avg_conf - avg_acc) * frac
        rendered_bins.append(
            {
                "bin": idx,
                "count": len(bucket),
                "avg_confidence": round(avg_conf, 4),
                "avg_accuracy": round(avg_acc, 4),
            }
        )
    return {"ece": round(ece, 4), "bins": rendered_bins}


def _resolve_components(checkpoint_path: str | None) -> Dict[str, Any]:
    if checkpoint_path:
        skill_index, controller_payload, final_scorer_payload, rule_scorer_payload, evidence_calibrator_payload = load_controller_checkpoint(checkpoint_path)
    else:
        skill_index = build_default_skill_index()
        controller_payload = {}
        final_scorer_payload = {}
        rule_scorer_payload = {}
        evidence_calibrator_payload = {}

    controller = LearnableSkillController(skill_index)
    final_scorer = LearnableFinalScorer()
    rule_scorer = LearnableRuleScorer()
    evidence_calibrator = LearnableEvidenceCalibrator()
    if controller_payload:
        controller.load_state(controller_payload)
    if final_scorer_payload:
        final_scorer.load_state(final_scorer_payload)
    if rule_scorer_payload:
        rule_scorer.load_state(rule_scorer_payload)
    if evidence_calibrator_payload:
        evidence_calibrator.load_state(evidence_calibrator_payload)
    return {
        "skill_index": skill_index,
        "controller": controller,
        "final_scorer": final_scorer,
        "rule_scorer": rule_scorer,
        "evidence_calibrator": evidence_calibrator,
    }


def _subgroup_metrics(rows: List[Dict[str, Any]], correctness_field: str) -> Dict[str, Dict[str, Any]]:
    axes = ("fitzpatrick", "skin_tone")
    output: Dict[str, Dict[str, Any]] = {}
    for axis in axes:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(axis, "unknown") or "unknown")].append(row)
        output[axis] = {
            name: _binary_metrics(items, correctness_field=correctness_field)
            for name, items in sorted(grouped.items())
        }
    return output


def _binary_metrics(rows: List[Dict[str, Any]], *, correctness_field: str) -> Dict[str, Any]:
    total = len(rows)
    if total <= 0:
        return {
            "num_cases": 0,
            "binary_accuracy": 0.0,
            "malignant_recall": 0.0,
            "specificity": 0.0,
            "balanced_accuracy": 0.0,
            "error_rate": 0.0,
        }

    malignant_rows = [row for row in rows if row.get("true_binary_label") == MALIGNANT_BINARY_LABEL]
    benign_rows = [row for row in rows if row.get("true_binary_label") == BENIGN_BINARY_LABEL]
    malignant_hits = sum(1 for row in malignant_rows if row.get("pred_binary_label") == MALIGNANT_BINARY_LABEL)
    benign_hits = sum(1 for row in benign_rows if row.get("pred_binary_label") == BENIGN_BINARY_LABEL)
    malignant_recall = malignant_hits / len(malignant_rows) if malignant_rows else 0.0
    specificity = benign_hits / len(benign_rows) if benign_rows else 0.0

    return {
        "num_cases": total,
        "binary_accuracy": round(sum(1 for row in rows if row.get(correctness_field)) / total, 4),
        "malignant_recall": round(malignant_recall, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round((malignant_recall + specificity) / 2, 4),
        "error_rate": round(sum(1 for row in rows if row.get("error")) / total, 4),
    }


def _six_class_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled_rows = [row for row in rows if row.get("true_label")]
    total = len(labeled_rows)
    if total <= 0:
        return {"num_cases": 0, "accuracy_top1": 0.0}
    return {
        "num_cases": total,
        "accuracy_top1": round(sum(1 for row in labeled_rows if row.get("is_top1_correct")) / total, 4),
    }


def _paired_mcnemar(rows_agent: List[Dict[str, Any]], rows_direct: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    agent_only = 0
    direct_only = 0
    for agent_row, direct_row in zip(rows_agent, rows_direct):
        agent_correct = bool(agent_row.get(field))
        direct_correct = bool(direct_row.get(field))
        if agent_correct and not direct_correct:
            agent_only += 1
        elif direct_correct and not agent_correct:
            direct_only += 1
    n = agent_only + direct_only
    if n <= 0:
        p_value = 1.0
    else:
        k = min(agent_only, direct_only)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)
    return {
        "agent_only_correct": agent_only,
        "direct_only_correct": direct_only,
        "p_value": round(p_value, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DermAgent and direct Qwen on external DDI data.")
    parser.add_argument("--dataset-root", default="data/ddi")
    parser.add_argument("--metadata-csv", default=None)
    parser.add_argument("--images-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--controller-checkpoint", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--perception-model", default=None)
    parser.add_argument("--report-model", default=None)
    parser.add_argument("--output-dir", default="outputs/external_ddi")
    parser.add_argument("--disable-direct-qwen", action="store_true")
    parser.add_argument("--disable-controller", action="store_true")
    args = parser.parse_args()

    cases = load_ddi_cases(
        dataset_root=args.dataset_root,
        metadata_csv=args.metadata_csv,
        images_dir=args.images_dir,
        limit=args.limit,
        split_name=args.split_name,
    )
    if not cases:
        raise RuntimeError("No DDI cases were loaded. Check your CSV path, image layout, or split filter.")

    components = _resolve_components(args.controller_checkpoint)
    bank = ExperienceBank.from_json(args.bank_state_in) if args.bank_state_in else ExperienceBank()
    reranker = UtilityAwareExperienceReranker()
    direct = None if args.disable_direct_qwen else QwenDirectInference()
    compat_client = OpenAICompatClient()
    perception_model = str(args.perception_model or compat_client.model)
    report_model = str(args.report_model or perception_model)

    agent_rows: List[Dict[str, Any]] = []
    direct_rows: List[Dict[str, Any]] = []
    agent_confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    direct_confusion: Dict[str, Counter[str]] = defaultdict(Counter)

    print(f"Running external DDI eval on {len(cases)} cases...")
    for idx, case in enumerate(cases):
        if idx % 10 == 0:
            print(f"  DDI progress: {idx}/{len(cases)}")

        true_label = _norm_label(case.get("label"))
        true_binary = _norm_label(case.get("binary_label"))
        metadata = case.get("metadata", {}) or {}
        fitzpatrick = str(metadata.get("fitzpatrick", "") or "unknown")
        skin_tone = str(metadata.get("skin_tone", "") or "unknown")

        agent_result = run_agent(
            case=case,
            bank=bank,
            skill_index=components["skill_index"],
            reranker=reranker,
            learning_components=None if args.disable_controller else {
                "controller": components["controller"],
                "final_scorer": components["final_scorer"],
                "rule_scorer": components["rule_scorer"],
                "evidence_calibrator": components["evidence_calibrator"],
            },
            use_retrieval=True,
            use_specialist=True,
            use_reflection=False,
            use_controller=not args.disable_controller,
            use_final_scorer=not args.disable_controller,
            update_online=False,
            use_rule_memory=True,
            enable_rule_compression=True,
            perception_model=perception_model,
            report_model=report_model,
        )
        agent_decision = agent_result.get("final_decision", {}) or {}
        agent_label = _norm_label(agent_decision.get("final_label") or agent_decision.get("diagnosis"))
        agent_binary = _binary_from_agent_label(agent_label)
        agent_conf = _confidence_to_float(agent_decision.get("confidence"))
        agent_row = {
            "case_id": case.get("file", f"ddi_{idx}"),
            "true_label": true_label,
            "true_binary_label": true_binary,
            "predicted_label": agent_label,
            "pred_binary_label": agent_binary,
            "confidence": agent_conf,
            "is_top1_correct": bool(true_label) and agent_label == true_label,
            "is_binary_correct": agent_binary == true_binary,
            "malignant_recalled": true_binary == MALIGNANT_BINARY_LABEL and agent_binary == MALIGNANT_BINARY_LABEL,
            "fitzpatrick": fitzpatrick,
            "skin_tone": skin_tone,
            "error": agent_result.get("error"),
            "perception_fallback": bool((agent_result.get("perception", {}) or {}).get("fallback_reason")),
            "report_fallback": _norm_text((agent_result.get("report", {}) or {}).get("generation_mode")) == "fallback",
        }
        agent_rows.append(agent_row)
        agent_confusion[true_binary][agent_binary] += 1

        if direct is not None:
            direct_result = direct.infer_case(case)
            direct_label = _norm_label(direct_result.get("diagnosis"))
            direct_binary = _binary_from_agent_label(direct_label)
            direct_conf = _safe_float(direct_result.get("confidence"), 0.5)
            direct_row = {
                "case_id": case.get("file", f"ddi_{idx}"),
                "true_label": true_label,
                "true_binary_label": true_binary,
                "predicted_label": direct_label,
                "pred_binary_label": direct_binary,
                "confidence": direct_conf,
                "is_top1_correct": bool(true_label) and direct_label == true_label,
                "is_binary_correct": direct_binary == true_binary,
                "malignant_recalled": true_binary == MALIGNANT_BINARY_LABEL and direct_binary == MALIGNANT_BINARY_LABEL,
                "fitzpatrick": fitzpatrick,
                "skin_tone": skin_tone,
                "error": direct_result.get("error"),
            }
            direct_rows.append(direct_row)
            direct_confusion[true_binary][direct_binary] += 1

    agent_binary = _binary_metrics(agent_rows, correctness_field="is_binary_correct")
    agent_six = _six_class_metrics(agent_rows)
    agent_ece = _compute_ece(agent_rows, field="is_binary_correct")

    direct_summary: Dict[str, Any] | None = None
    paired = None
    if direct_rows:
        direct_binary = _binary_metrics(direct_rows, correctness_field="is_binary_correct")
        direct_six = _six_class_metrics(direct_rows)
        direct_ece = _compute_ece(direct_rows, field="is_binary_correct")
        direct_summary = {
            "binary_metrics": direct_binary,
            "six_class_metrics": direct_six,
            "subgroup_metrics": _subgroup_metrics(direct_rows, correctness_field="is_binary_correct"),
            "calibration": direct_ece,
            "confusion_matrix": {label: dict(sorted(counter.items())) for label, counter in sorted(direct_confusion.items())},
            "per_case": direct_rows,
            "model_info": {
                "direct_model": str(getattr(direct.client, "model", "")) if direct.client is not None else "",
                "base_url": str(getattr(direct.client, "base_url", "")) if direct.client is not None else "",
            },
        }
        paired = {
            "binary_accuracy_diff": round(agent_binary["binary_accuracy"] - direct_binary["binary_accuracy"], 4),
            "malignant_recall_diff": round(agent_binary["malignant_recall"] - direct_binary["malignant_recall"], 4),
            "specificity_diff": round(agent_binary["specificity"] - direct_binary["specificity"], 4),
            "binary_mcnemar": _paired_mcnemar(agent_rows, direct_rows, field="is_binary_correct"),
        }

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": args.dataset_root,
        "metadata_csv": args.metadata_csv,
        "images_dir": args.images_dir,
        "split_name": args.split_name,
        "num_cases": len(cases),
        "label_coverage": {
            "with_six_class_label": sum(1 for case in cases if case.get("label")),
            "binary_benign": sum(1 for case in cases if case.get("binary_label") == BENIGN_BINARY_LABEL),
            "binary_malignant": sum(1 for case in cases if case.get("binary_label") == MALIGNANT_BINARY_LABEL),
        },
        "agent_results": {
            "binary_metrics": agent_binary,
            "six_class_metrics": agent_six,
            "subgroup_metrics": _subgroup_metrics(agent_rows, correctness_field="is_binary_correct"),
            "calibration": agent_ece,
            "confusion_matrix": {label: dict(sorted(counter.items())) for label, counter in sorted(agent_confusion.items())},
            "fallbacks": {
                "perception": sum(1 for row in agent_rows if row.get("perception_fallback")),
                "report": sum(1 for row in agent_rows if row.get("report_fallback")),
            },
            "per_case": agent_rows,
            "model_info": {
                "perception_model": perception_model,
                "report_model": report_model,
                "controller_enabled": not args.disable_controller,
                "checkpoint_path": args.controller_checkpoint,
            },
        },
        "qwen_direct_results": direct_summary,
        "comparison": paired,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"external_ddi_report_{ts}.json"
    csv_path = output_dir / f"external_ddi_metrics_{ts}.csv"
    txt_path = output_dir / f"external_ddi_summary_{ts}.txt"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "agent", "direct_qwen"])
        writer.writerow(["binary_accuracy", agent_binary["binary_accuracy"], "" if direct_summary is None else direct_summary["binary_metrics"]["binary_accuracy"]])
        writer.writerow(["malignant_recall", agent_binary["malignant_recall"], "" if direct_summary is None else direct_summary["binary_metrics"]["malignant_recall"]])
        writer.writerow(["specificity", agent_binary["specificity"], "" if direct_summary is None else direct_summary["binary_metrics"]["specificity"]])
        writer.writerow(["balanced_accuracy", agent_binary["balanced_accuracy"], "" if direct_summary is None else direct_summary["binary_metrics"]["balanced_accuracy"]])
        writer.writerow(["six_class_accuracy_subset", agent_six["accuracy_top1"], "" if direct_summary is None else direct_summary["six_class_metrics"]["accuracy_top1"]])
        writer.writerow(["ece_binary", agent_ece["ece"], "" if direct_summary is None else direct_summary["calibration"]["ece"]])

    lines = [
        "DermAgent External DDI Evaluation",
        f"num_cases: {summary['num_cases']}",
        f"with_six_class_label: {summary['label_coverage']['with_six_class_label']}",
        "",
        "Agent",
        f"  binary_accuracy: {agent_binary['binary_accuracy']:.4f}",
        f"  malignant_recall: {agent_binary['malignant_recall']:.4f}",
        f"  specificity: {agent_binary['specificity']:.4f}",
        f"  balanced_accuracy: {agent_binary['balanced_accuracy']:.4f}",
        f"  six_class_accuracy_subset: {agent_six['accuracy_top1']:.4f}",
        f"  ece_binary: {agent_ece['ece']:.4f}",
        "",
    ]
    if direct_summary is not None:
        direct_binary_metrics = direct_summary["binary_metrics"]
        direct_six_metrics = direct_summary["six_class_metrics"]
        lines.extend(
            [
                "Direct Qwen",
                f"  binary_accuracy: {direct_binary_metrics['binary_accuracy']:.4f}",
                f"  malignant_recall: {direct_binary_metrics['malignant_recall']:.4f}",
                f"  specificity: {direct_binary_metrics['specificity']:.4f}",
                f"  balanced_accuracy: {direct_binary_metrics['balanced_accuracy']:.4f}",
                f"  six_class_accuracy_subset: {direct_six_metrics['accuracy_top1']:.4f}",
                f"  ece_binary: {direct_summary['calibration']['ece']:.4f}",
                "",
                "Paired Comparison",
                f"  binary_accuracy_diff: {summary['comparison']['binary_accuracy_diff']:+.4f}",
                f"  malignant_recall_diff: {summary['comparison']['malignant_recall_diff']:+.4f}",
                f"  specificity_diff: {summary['comparison']['specificity_diff']:+.4f}",
                f"  binary_mcnemar_p: {summary['comparison']['binary_mcnemar']['p_value']:.6f}",
            ]
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved external DDI JSON to {json_path}")
    print(f"Saved external DDI CSV to {csv_path}")
    print(f"Saved external DDI summary to {txt_path}")


if __name__ == "__main__":
    main()
