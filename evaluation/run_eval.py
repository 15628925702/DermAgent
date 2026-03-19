from __future__ import annotations

import argparse
import csv
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
from datasets.splits import (
    load_or_create_split_manifest,
    load_split_manifest,
    select_split_cases,
    summarize_split_cases,
)
from memory.controller_store import load_controller_checkpoint, save_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index


MALIGNANT_LABELS = {"MEL", "BCC", "SCC"}
ACK_SCC_LABELS = {"ACK", "SCC"}
SUPPORTED_LABELS = {"MEL", "NEV", "SCC", "BCC", "ACK", "SEK"}


def load_pad_ufes20_cases(dataset_root: str | Path = "data/pad_ufes_20", limit: int | None = None) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    metadata_path = root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    image_index = {path.name: str(path) for path in root.rglob("*.png")}
    cases: List[Dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = _norm_label(row.get("diagnostic"))
            img_id = str(row.get("img_id", "")).strip()
            if not img_id or label not in SUPPORTED_LABELS:
                continue
            image_path = image_index.get(img_id, str(root / img_id))
            metadata = {
                "age": _safe_int(row.get("age")),
                "sex": row.get("gender") or row.get("sex") or "",
                "location": row.get("region") or row.get("location") or "",
                "site": row.get("region") or row.get("site") or "",
                "clinical_history": _build_history_text(row),
                "patient_id": row.get("patient_id", ""),
                "lesion_id": row.get("lesion_id", ""),
            }
            cases.append({"file": img_id, "image_path": image_path, "metadata": metadata, "text": "", "label": label})
            if limit is not None and len(cases) >= limit:
                break
    return cases


def run_evaluation(
    dataset_root: str | Path = "data/pad_ufes_20",
    limit: int | None = None,
    split_json: str | None = None,
    split_name: str | None = None,
    use_retrieval: bool = True,
    use_specialist: bool = True,
    use_reflection: bool = True,
    use_controller: bool = True,
    controller_state_in: str | None = None,
    controller_state_out: str | None = None,
    bank_state_in: str | None = None,
    bank_state_out: str | None = None,
    update_online: bool = True,
    use_rule_memory: bool = True,
    enable_rule_compression: bool = True,
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

    bank = ExperienceBank.from_json(bank_state_in) if bank_state_in else ExperienceBank()
    bank_loaded = bool(bank_state_in)

    checkpoint_loaded = False
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
    final_scorer = LearnableFinalScorer() if use_controller else None
    rule_scorer = LearnableRuleScorer() if use_controller else None
    if controller is not None and controller_payload:
        controller.load_state(controller_payload)
    if final_scorer is not None and final_scorer_payload:
        final_scorer.load_state(final_scorer_payload)
    if rule_scorer is not None and rule_scorer_payload:
        rule_scorer.load_state(rule_scorer_payload)

    total = correct_top1 = correct_top3 = malignant_total = malignant_hit = ack_scc_total = ack_scc_correct = errors = 0
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
            use_retrieval=use_retrieval,
            use_specialist=use_specialist,
            use_reflection=use_reflection,
            use_controller=use_controller,
            update_online=update_online,
            use_rule_memory=use_rule_memory,
            enable_rule_compression=enable_rule_compression,
        )
        true_label = _norm_label(case.get("label"))
        final_decision = result.get("final_decision", {}) or {}
        pred_label = _norm_label(final_decision.get("final_label") or final_decision.get("diagnosis"))
        top3 = _extract_top_k_labels(final_decision, top_n=3)

        total += 1
        if result.get("error"):
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

        per_case.append({
            "case_id": case.get("file", ""),
            "true_label": true_label,
            "predicted_label": pred_label,
            "top3": top3,
            "is_top1_correct": pred_label == true_label,
            "is_top3_correct": true_label in top3,
            "is_malignant_case": true_label in MALIGNANT_LABELS,
            "malignant_recalled": true_label in MALIGNANT_LABELS and pred_label in MALIGNANT_LABELS,
            "is_ack_scc_case": true_label in ACK_SCC_LABELS,
            "ack_scc_correct": true_label in ACK_SCC_LABELS and pred_label == true_label,
            "error": result.get("error"),
            "selected_skills": result.get("selected_skills", []),
            "planner_mode": (result.get("planner", {}) or {}).get("planning_mode"),
            "stop_probability": (result.get("planner", {}) or {}).get("stop_probability"),
        })

    result = {
        "dataset_root": str(dataset_root),
        "num_cases": total,
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
        "runtime_flags": {
            "use_retrieval": use_retrieval,
            "use_specialist": use_specialist,
            "use_reflection": use_reflection,
            "use_controller": use_controller,
            "update_online": update_online,
            "use_rule_memory": use_rule_memory,
            "enable_rule_compression": enable_rule_compression,
        },
        "split": {
            "name": split_name,
            "path": resolved_split_path,
            "summary": summarize_split_cases(cases),
        },
        "checkpoint": {
            "loaded": checkpoint_loaded,
            "input_path": controller_state_in,
            "output_path": controller_state_out,
            "contains_final_scorer": use_controller,
            "contains_rule_scorer": use_controller,
        },
        "bank_checkpoint": {
            "loaded": bank_loaded,
            "input_path": bank_state_in,
            "output_path": bank_state_out,
        },
        "bank_stats": bank.stats(),
        "skill_index": skill_index.as_dict(),
        "final_scorer": final_scorer.to_dict() if final_scorer is not None else None,
        "rule_scorer": rule_scorer.to_dict() if rule_scorer is not None else None,
        "per_case": per_case,
    }
    if controller_state_out:
        saved_path = save_controller_checkpoint(
            controller_state_out,
            skill_index=skill_index,
            controller=controller,
            final_scorer=final_scorer,
            rule_scorer=rule_scorer,
            metadata={
                "dataset_root": str(dataset_root),
                "num_cases": total,
                "metrics": result["metrics"],
                "update_online": update_online,
                "split_name": split_name,
                "split_path": resolved_split_path,
            },
        )
        result["checkpoint"]["saved_to"] = str(saved_path)
    if bank_state_out:
        saved_bank_path = bank.save_json(bank_state_out)
        result["bank_checkpoint"]["saved_to"] = str(saved_bank_path)
        result["bank_stats"] = bank.stats()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DermAgent evaluation.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default=None, choices=["train", "val", "test"])
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--disable-specialist", action="store_true")
    parser.add_argument("--disable-reflection", action="store_true")
    parser.add_argument("--disable-controller", action="store_true")
    parser.add_argument("--disable-rule-memory", action="store_true")
    parser.add_argument("--disable-rule-compression", action="store_true")
    parser.add_argument("--controller-state-in", default=None)
    parser.add_argument("--controller-state-out", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--bank-state-out", default=None)
    parser.add_argument("--freeze-learning", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_evaluation(
        dataset_root=args.dataset_root,
        limit=args.limit,
        split_json=args.split_json,
        split_name=args.split_name,
        use_retrieval=not args.disable_retrieval,
        use_specialist=not args.disable_specialist,
        use_reflection=not args.disable_reflection,
        use_controller=not args.disable_controller,
        controller_state_in=args.controller_state_in,
        controller_state_out=args.controller_state_out,
        bank_state_in=args.bank_state_in,
        bank_state_out=args.bank_state_out,
        update_online=not args.freeze_learning,
        use_rule_memory=not args.disable_rule_memory,
        enable_rule_compression=not args.disable_rule_compression,
    )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


def _extract_top_k_labels(final_decision: Dict[str, Any], top_n: int) -> List[str]:
    labels: List[str] = []
    for item in (final_decision or {}).get("top_k", [])[:top_n]:
        label = _norm_label(item.get("name")) if isinstance(item, dict) else _norm_label(item)
        if label:
            labels.append(label)
    return labels


def _build_history_text(row: Dict[str, Any]) -> str:
    history_bits = []
    for key in ["itch", "grew", "hurt", "changed", "bleed", "elevation"]:
        if str(row.get(key, "")).strip().lower() == "true":
            history_bits.append(key)
    return "; ".join(history_bits)


def _safe_div(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 4)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


if __name__ == "__main__":
    main()
