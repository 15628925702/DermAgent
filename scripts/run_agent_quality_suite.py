#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from datasets.splits import load_or_create_split_manifest, select_split_cases
from evaluation.run_eval import load_pad_ufes20_cases
from integrations.openai_client import OpenAICompatClient
from memory.controller_store import load_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index


MALIGNANT_LABELS = {"MEL", "BCC", "SCC"}
VALID_LABELS = {"MEL", "BCC", "SCC", "NEV", "ACK", "SEK"}


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _confidence_to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    mapping = {"low": 0.33, "medium": 0.66, "high": 0.90}
    return mapping.get(_norm_text(value), 0.5)


def _age_group(metadata: Dict[str, Any]) -> str:
    age = _safe_int(metadata.get("age"))
    if age is None:
        return "unknown"
    if age <= 18:
        return "pediatric"
    if age < 30:
        return "young"
    if age < 60:
        return "middle"
    return "older"


def _site_group(metadata: Dict[str, Any]) -> str:
    site = _norm_text(metadata.get("location") or metadata.get("site") or metadata.get("anatomical_site"))
    if not site:
        return "unknown"
    if any(token in site for token in ["face", "scalp", "ear", "neck", "nose", "temple", "cheek", "hand", "forearm", "lip"]):
        return "sun_exposed"
    if any(token in site for token in ["trunk", "back", "chest", "abdomen"]):
        return "trunk"
    if any(token in site for token in ["leg", "foot", "toe"]):
        return "lower_limb"
    return "other"


def _top_k_labels(final_decision: Dict[str, Any], top_n: int = 3) -> List[str]:
    labels: List[str] = []
    for item in (final_decision or {}).get("top_k", [])[:top_n]:
        if isinstance(item, dict):
            label = _norm_label(item.get("name"))
        else:
            label = _norm_label(item)
        if label:
            labels.append(label)
    return labels


def _compute_ece(items: List[Dict[str, Any]], num_bins: int = 10) -> Dict[str, Any]:
    if not items:
        return {"ece": 0.0, "bins": []}
    bins: List[List[Dict[str, Any]]] = [[] for _ in range(num_bins)]
    for item in items:
        conf = max(0.0, min(0.999999, _safe_float(item.get("confidence"), default=0.5)))
        idx = min(num_bins - 1, int(conf * num_bins))
        bins[idx].append(item)

    ece = 0.0
    rendered_bins: List[Dict[str, Any]] = []
    total = len(items)
    for idx, bucket in enumerate(bins):
        if not bucket:
            continue
        avg_conf = sum(_safe_float(x.get("confidence"), 0.5) for x in bucket) / len(bucket)
        avg_acc = sum(1.0 if x.get("correct") else 0.0 for x in bucket) / len(bucket)
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


def _group_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    if total <= 0:
        return {"num_cases": 0, "accuracy_top1": 0.0, "malignant_recall": 0.0, "error_rate": 0.0}
    malignant_total = sum(1 for x in items if x.get("true_label") in MALIGNANT_LABELS)
    malignant_hit = sum(1 for x in items if x.get("true_label") in MALIGNANT_LABELS and x.get("pred_label") in MALIGNANT_LABELS)
    return {
        "num_cases": total,
        "accuracy_top1": round(sum(1 for x in items if x.get("correct")) / total, 4),
        "malignant_recall": round(malignant_hit / malignant_total, 4) if malignant_total else 0.0,
        "error_rate": round(sum(1 for x in items if x.get("error")) / total, 4),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibration, robustness, and safety analysis for DermAgent.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-checkpoint", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--perception-model", default=None)
    parser.add_argument("--report-model", default=None)
    parser.add_argument("--output-dir", default="outputs/quality")
    args = parser.parse_args()

    all_cases = load_pad_ufes20_cases(dataset_root=args.dataset_root, limit=None)
    split_path = args.split_json or str(Path("outputs/splits") / f"{Path(args.dataset_root).name}_seed{args.seed}.json")
    split_payload = load_or_create_split_manifest(all_cases, split_path, seed=args.seed)
    cases = select_split_cases(all_cases, split_payload, args.split_name)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise RuntimeError(f"Split '{args.split_name}' is empty.")

    components = _resolve_components(args.controller_checkpoint)
    bank = ExperienceBank.from_json(args.bank_state_in) if args.bank_state_in else ExperienceBank()
    reranker = UtilityAwareExperienceReranker()
    client = OpenAICompatClient()
    perception_model = str(args.perception_model or client.model)
    report_model = str(args.report_model or perception_model)

    rows: List[Dict[str, Any]] = []
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)

    print(f"Running quality suite on {len(cases)} {args.split_name} cases...")
    for idx, case in enumerate(cases):
        if idx % 10 == 0:
            print(f"  quality progress: {idx}/{len(cases)}")
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
            },
            use_retrieval=True,
            use_specialist=True,
            use_reflection=False,
            use_controller=True,
            use_final_scorer=True,
            update_online=False,
            use_rule_memory=True,
            enable_rule_compression=True,
            perception_model=perception_model,
            report_model=report_model,
        )
        final_decision = result.get("final_decision", {}) or {}
        pred_label = _norm_label(final_decision.get("final_label") or final_decision.get("diagnosis"))
        true_label = _norm_label(case.get("label"))
        top3 = _top_k_labels(final_decision, 3)
        retrieval_summary = (result.get("retrieval", {}) or {}).get("retrieval_summary", {}) or {}
        metadata = case.get("metadata", {}) or {}
        error = result.get("error")
        confidence = _confidence_to_float(final_decision.get("confidence"))
        row = {
            "case_id": case.get("file", f"case_{idx}"),
            "true_label": true_label,
            "pred_label": pred_label,
            "top3": top3,
            "correct": pred_label == true_label,
            "top3_hit": true_label in top3,
            "confidence": confidence,
            "error": error,
            "age_group": _age_group(metadata),
            "site_group": _site_group(metadata),
            "has_metadata": bool(metadata),
            "retrieval_confidence": _norm_text(retrieval_summary.get("retrieval_confidence")) or "unknown",
            "has_confusion_support": bool(retrieval_summary.get("has_confusion_support", False)),
            "perception_fallback": bool((result.get("perception", {}) or {}).get("fallback_reason")),
            "report_fallback": _norm_text((result.get("report", {}) or {}).get("generation_mode")) == "fallback",
        }
        rows.append(row)
        confusion[true_label][pred_label or "UNKNOWN"] += 1

    total = len(rows)
    malignant_total = sum(1 for row in rows if row["true_label"] in MALIGNANT_LABELS)
    malignant_hits = sum(1 for row in rows if row["true_label"] in MALIGNANT_LABELS and row["pred_label"] in MALIGNANT_LABELS)
    brier = sum((row["confidence"] - (1.0 if row["correct"] else 0.0)) ** 2 for row in rows) / max(total, 1)
    ece = _compute_ece(rows)

    per_label: Dict[str, Dict[str, Any]] = {}
    for label in sorted(VALID_LABELS):
        tp = sum(1 for row in rows if row["true_label"] == label and row["pred_label"] == label)
        fp = sum(1 for row in rows if row["true_label"] != label and row["pred_label"] == label)
        fn = sum(1 for row in rows if row["true_label"] == label and row["pred_label"] != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": sum(1 for row in rows if row["true_label"] == label),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    subgroup_axes = ["age_group", "site_group", "retrieval_confidence"]
    subgroup_metrics: Dict[str, Dict[str, Any]] = {}
    for axis in subgroup_axes:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row.get(axis, "unknown"))].append(row)
        subgroup_metrics[axis] = {
            name: _group_metrics(items)
            for name, items in sorted(grouped.items())
        }

    ood_slices = {
        "missing_metadata": [row for row in rows if not row["has_metadata"]],
        "low_retrieval": [row for row in rows if row["retrieval_confidence"] == "low"],
        "perception_fallback": [row for row in rows if row["perception_fallback"]],
        "rare_site_proxy": [row for row in rows if row["site_group"] in {"other", "unknown"}],
    }
    ood_metrics = {name: _group_metrics(items) for name, items in ood_slices.items()}

    malignant_misses = [
        {
            "case_id": row["case_id"],
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "confidence": row["confidence"],
            "retrieval_confidence": row["retrieval_confidence"],
        }
        for row in rows
        if row["true_label"] in MALIGNANT_LABELS and row["pred_label"] not in MALIGNANT_LABELS
    ]
    benign_false_alarms = [
        {
            "case_id": row["case_id"],
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "confidence": row["confidence"],
            "retrieval_confidence": row["retrieval_confidence"],
        }
        for row in rows
        if row["true_label"] not in MALIGNANT_LABELS and row["pred_label"] in MALIGNANT_LABELS
    ]

    summary = {
        "dataset_root": args.dataset_root,
        "split_path": split_path,
        "split_name": args.split_name,
        "num_cases": total,
        "metrics": {
            "accuracy_top1": round(sum(1 for row in rows if row["correct"]) / max(total, 1), 4),
            "accuracy_top3": round(sum(1 for row in rows if row["top3_hit"]) / max(total, 1), 4),
            "malignant_recall": round(malignant_hits / malignant_total, 4) if malignant_total else 0.0,
            "error_rate": round(sum(1 for row in rows if row["error"]) / max(total, 1), 4),
            "brier_top1": round(brier, 4),
            "expected_calibration_error": ece["ece"],
        },
        "per_label": per_label,
        "confusion_matrix": {label: dict(sorted(counter.items())) for label, counter in sorted(confusion.items())},
        "subgroup_metrics": subgroup_metrics,
        "ood_proxy_metrics": ood_metrics,
        "safety": {
            "malignant_miss_count": len(malignant_misses),
            "benign_false_alarm_count": len(benign_false_alarms),
            "malignant_misses": malignant_misses[:25],
            "benign_false_alarms": benign_false_alarms[:25],
        },
        "fallbacks": {
            "perception": sum(1 for row in rows if row["perception_fallback"]),
            "report": sum(1 for row in rows if row["report_fallback"]),
        },
        "calibration_bins": ece["bins"],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"quality_suite_{ts}.json"
    txt_path = output_dir / f"quality_suite_{ts}.txt"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "DermAgent Quality Suite",
        f"split: {args.split_name}",
        f"num_cases: {total}",
        "",
        "Core Metrics",
        f"  top1_accuracy: {summary['metrics']['accuracy_top1']:.4f}",
        f"  top3_accuracy: {summary['metrics']['accuracy_top3']:.4f}",
        f"  malignant_recall: {summary['metrics']['malignant_recall']:.4f}",
        f"  error_rate: {summary['metrics']['error_rate']:.4f}",
        f"  brier_top1: {summary['metrics']['brier_top1']:.4f}",
        f"  expected_calibration_error: {summary['metrics']['expected_calibration_error']:.4f}",
        "",
        "Safety",
        f"  malignant_miss_count: {summary['safety']['malignant_miss_count']}",
        f"  benign_false_alarm_count: {summary['safety']['benign_false_alarm_count']}",
        "",
        "Fallbacks",
        f"  perception_fallbacks: {summary['fallbacks']['perception']}",
        f"  report_fallbacks: {summary['fallbacks']['report']}",
        "",
        "OOD Proxy Slices",
    ]
    for name, metrics in summary["ood_proxy_metrics"].items():
        lines.append(f"  {name}: n={metrics['num_cases']} top1={metrics['accuracy_top1']:.4f} malignant_recall={metrics['malignant_recall']:.4f}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved quality suite JSON to {json_path}")
    print(f"Saved quality suite summary to {txt_path}")


if __name__ == "__main__":
    main()
