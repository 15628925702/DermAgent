#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
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
from memory.controller_store import load_controller_checkpoint, save_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index


MALIGNANT_LABELS = {"MEL", "BCC", "SCC"}


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


def _evaluate_cases(
    *,
    cases: List[Dict[str, Any]],
    bank: ExperienceBank,
    skill_index: Any,
    reranker: UtilityAwareExperienceReranker,
    controller: LearnableSkillController | None,
    final_scorer: LearnableFinalScorer | None,
    rule_scorer: LearnableRuleScorer | None,
    evidence_calibrator: LearnableEvidenceCalibrator | None,
    use_retrieval: bool,
    use_specialist: bool,
    use_controller: bool,
    perception_model: str,
    report_model: str,
) -> Dict[str, Any]:
    total = 0
    correct_top1 = 0
    correct_top3 = 0
    malignant_total = 0
    malignant_hit = 0
    errors = 0

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
                "evidence_calibrator": evidence_calibrator,
            } if use_controller else None,
            use_retrieval=use_retrieval,
            use_specialist=use_specialist,
            use_reflection=False,
            use_controller=use_controller,
            use_final_scorer=use_controller,
            update_online=False,
            use_rule_memory=True,
            enable_rule_compression=True,
            perception_model=perception_model,
            report_model=report_model,
        )

        total += 1
        true_label = _norm_label(case.get("label"))
        final_decision = result.get("final_decision", {}) or {}
        pred_label = _norm_label(final_decision.get("final_label") or final_decision.get("diagnosis"))
        top3 = _extract_top_k_labels(final_decision, top_n=3)

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

    return {
        "num_cases": total,
        "accuracy_top1": 0.0 if total <= 0 else round(correct_top1 / total, 4),
        "accuracy_top3": 0.0 if total <= 0 else round(correct_top3 / total, 4),
        "malignant_recall": 0.0 if malignant_total <= 0 else round(malignant_hit / malignant_total, 4),
        "errors": errors,
        "bank_stats": bank.stats(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DermAgent learned components with checkpointed evaluation.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-checkpoint-in", default=None)
    parser.add_argument("--controller-checkpoint-out", default="outputs/checkpoints/learned_controller_best.json")
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--bank-state-out", default="outputs/checkpoints/learned_bank_best.json")
    parser.add_argument("--perception-model", default=None)
    parser.add_argument("--report-model", default=None)
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--disable-specialist", action="store_true")
    parser.add_argument("--output", default="outputs/train_runs/learned_components_training_summary.json")
    args = parser.parse_args()

    all_cases = load_pad_ufes20_cases(dataset_root=args.dataset_root, limit=args.limit)
    split_path = args.split_json or str(Path("outputs/splits") / f"{Path(args.dataset_root).name}_seed{args.seed}.json")
    split_payload = load_or_create_split_manifest(all_cases, split_path, seed=args.seed)
    train_cases = select_split_cases(all_cases, split_payload, "train")
    val_cases = select_split_cases(all_cases, split_payload, "val")
    test_cases = select_split_cases(all_cases, split_payload, "test")

    if not train_cases:
        raise RuntimeError("Training split is empty.")

    base_client = OpenAICompatClient()
    perception_model = str(args.perception_model or base_client.model)
    report_model = str(args.report_model or perception_model)
    use_retrieval = not args.disable_retrieval
    use_specialist = not args.disable_specialist

    bank = ExperienceBank.from_json(args.bank_state_in) if args.bank_state_in else ExperienceBank()
    if args.controller_checkpoint_in:
        skill_index, controller_payload, final_scorer_payload, rule_scorer_payload, evidence_calibrator_payload = load_controller_checkpoint(args.controller_checkpoint_in)
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

    reranker = UtilityAwareExperienceReranker()
    best_val = -1.0
    history: List[Dict[str, Any]] = []

    for epoch in range(args.epochs):
        epoch_cases = list(train_cases)
        random.Random(args.seed + epoch).shuffle(epoch_cases)
        train_correct = 0

        print(f"Epoch {epoch + 1}/{args.epochs}: training on {len(epoch_cases)} cases")
        for idx, case in enumerate(epoch_cases):
            if idx % 20 == 0:
                print(f"  train progress: {idx}/{len(epoch_cases)}")
            result = run_agent(
                case=case,
                bank=bank,
                skill_index=skill_index,
                reranker=reranker,
                learning_components={
                    "controller": controller,
                    "final_scorer": final_scorer,
                    "rule_scorer": rule_scorer,
                    "evidence_calibrator": evidence_calibrator,
                },
                use_retrieval=use_retrieval,
                use_specialist=use_specialist,
                use_reflection=True,
                use_controller=True,
                use_final_scorer=True,
                update_online=True,
                use_rule_memory=True,
                enable_rule_compression=True,
                perception_model=perception_model,
                report_model=report_model,
            )
            pred = _norm_label((result.get("final_decision", {}) or {}).get("final_label"))
            true = _norm_label(case.get("label"))
            if pred == true:
                train_correct += 1

        train_top1 = round(train_correct / len(epoch_cases), 4)
        val_metrics = _evaluate_cases(
            cases=val_cases,
            bank=bank,
            skill_index=skill_index,
            reranker=reranker,
            controller=controller,
            final_scorer=final_scorer,
            rule_scorer=rule_scorer,
            evidence_calibrator=evidence_calibrator,
            use_retrieval=use_retrieval,
            use_specialist=use_specialist,
            use_controller=True,
            perception_model=perception_model,
            report_model=report_model,
        )

        epoch_summary = {
            "epoch": epoch + 1,
            "train_top1": train_top1,
            "val_metrics": val_metrics,
            "bank_stats": bank.stats(),
        }
        history.append(epoch_summary)
        print(f"  train_top1={train_top1:.4f} val_top1={val_metrics['accuracy_top1']:.4f} val_malignant_recall={val_metrics['malignant_recall']:.4f}")

        if val_metrics["accuracy_top1"] >= best_val:
            best_val = val_metrics["accuracy_top1"]
            save_controller_checkpoint(
                args.controller_checkpoint_out,
                skill_index=skill_index,
                controller=controller,
                final_scorer=final_scorer,
                rule_scorer=rule_scorer,
                evidence_calibrator=evidence_calibrator,
                metadata={
                    "best_val_accuracy_top1": best_val,
                    "epoch": epoch + 1,
                    "perception_model": perception_model,
                    "report_model": report_model,
                    "use_retrieval": use_retrieval,
                    "use_specialist": use_specialist,
                },
            )
            ExperienceBank(initial_items=bank.list_all()).save_json(args.bank_state_out)

    test_metrics = _evaluate_cases(
        cases=test_cases,
        bank=bank,
        skill_index=skill_index,
        reranker=reranker,
        controller=controller,
        final_scorer=final_scorer,
        rule_scorer=rule_scorer,
        evidence_calibrator=evidence_calibrator,
        use_retrieval=use_retrieval,
        use_specialist=use_specialist,
        use_controller=True,
        perception_model=perception_model,
        report_model=report_model,
    )

    summary = {
        "dataset_root": args.dataset_root,
        "limit": args.limit,
        "split_path": split_path,
        "epochs": args.epochs,
        "perception_model": perception_model,
        "report_model": report_model,
        "use_retrieval": use_retrieval,
        "use_specialist": use_specialist,
        "best_val_accuracy_top1": best_val,
        "history": history,
        "final_bank_stats": bank.stats(),
        "test_metrics": test_metrics,
        "controller_checkpoint_out": args.controller_checkpoint_out,
        "bank_state_out": args.bank_state_out,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved training summary to {output_path}")
    print(f"Best controller checkpoint: {args.controller_checkpoint_out}")
    print(f"Best bank checkpoint: {args.bank_state_out}")


if __name__ == "__main__":
    main()
