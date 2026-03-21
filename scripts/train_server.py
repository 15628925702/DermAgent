from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
from evaluation.run_eval import load_pad_ufes20_cases
from memory.controller_store import load_controller_checkpoint, save_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_designer import SkillEvolutionDesigner
from memory.skill_index import build_default_skill_index

MALIGNANT_LABELS = {"MEL", "BCC", "SCC"}
ACK_SCC_LABELS = {"ACK", "SCC"}
RUN_ARTIFACT_NAMES = {
    "best_bank.json",
    "best_controller.json",
    "best_skill_designer.json",
    "latest_bank.json",
    "latest_controller.json",
    "latest_skill_designer.json",
    "run_manifest.json",
    "skill_evolution.jsonl",
    "train_log.jsonl",
    "train_summary.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Server-style training loop for DermAgent v2.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-dir", default="outputs/train_runs/latest")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--init-mode", default="clean")
    parser.add_argument("--base-run-dir", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--controller-state-in", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty-save-dir", action="store_true")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--rule-compression-start-epoch", type=int, default=4)
    parser.add_argument("--rule-memory-start-epoch", type=int, default=8)
    parser.add_argument("--rule-learning-start-epoch", type=int, default=10)
    parser.add_argument("--skill-evolution-start-epoch", type=int, default=12)
    parser.add_argument("--skill-evolution-every", type=int, default=3)
    parser.add_argument("--enable-controller", action="store_true")
    parser.add_argument("--enable-final-scorer", action="store_true")
    parser.add_argument("--disable-compare", action="store_true")
    parser.add_argument("--enable-malignancy", action="store_true")
    parser.add_argument("--disable-malignancy", action="store_true")
    parser.add_argument("--disable-metadata-consistency", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    malignancy_enabled = bool(args.enable_malignancy)
    if args.disable_malignancy:
        malignancy_enabled = False

    all_cases = load_pad_ufes20_cases(dataset_root=args.dataset_root, limit=args.limit)
    split_path = Path(args.split_json) if args.split_json else save_dir / "resolved_split.json"
    if split_path.exists():
        split_payload = load_split_manifest(split_path)
    else:
        split_payload = load_or_create_split_manifest(
            all_cases,
            split_path,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )

    train_cases = select_split_cases(all_cases, split_payload, "train")
    val_cases = select_split_cases(all_cases, split_payload, "val")
    test_cases = select_split_cases(all_cases, split_payload, "test")

    if args.resume:
        controller_in = args.controller_state_in or str(save_dir / "latest_controller.json")
        bank_in = args.bank_state_in or str(save_dir / "latest_bank.json")
    else:
        controller_in = args.controller_state_in
        bank_in = args.bank_state_in

    validate_save_dir_state(
        save_dir=save_dir,
        resume=args.resume,
        allow_dirty_save_dir=args.allow_dirty_save_dir,
    )

    manifest_path = save_dir / "run_manifest.json"
    init_summary = build_init_summary(
        init_mode=args.init_mode,
        base_run_dir=args.base_run_dir,
        controller_in=controller_in,
        bank_in=bank_in,
        resume=args.resume,
    )
    run_config = build_run_config(args=args, save_dir=save_dir)
    write_run_manifest(
        path=manifest_path,
        payload={
            "format_version": 1,
            "run_name": args.run_name or save_dir.name,
            "save_dir": str(save_dir),
            "status": "running",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "initialization": init_summary,
            "config": run_config,
            "artifacts": build_artifact_summary(save_dir),
        },
        resume=args.resume,
    )

    bank = ExperienceBank.from_json(bank_in) if bank_in and Path(bank_in).exists() else ExperienceBank()

    if controller_in and Path(controller_in).exists():
        skill_index, controller_payload, final_scorer_payload, rule_scorer_payload = load_controller_checkpoint(controller_in)
    else:
        skill_index = build_default_skill_index()
        controller_payload = {}
        final_scorer_payload = {}
        rule_scorer_payload = {}

    controller = LearnableSkillController(skill_index)
    if controller_payload:
        controller.load_state(controller_payload)
    final_scorer = LearnableFinalScorer()
    if final_scorer_payload:
        final_scorer.load_state(final_scorer_payload)
    rule_scorer = LearnableRuleScorer()
    if rule_scorer_payload:
        rule_scorer.load_state(rule_scorer_payload)
    reranker = UtilityAwareExperienceReranker()
    designer = SkillEvolutionDesigner()

    latest_designer_path = save_dir / "latest_skill_designer.json"
    if latest_designer_path.exists():
        designer.load_state(json.loads(latest_designer_path.read_text(encoding="utf-8")))

    train_log_path = save_dir / "train_log.jsonl"
    evolution_log_path = save_dir / "skill_evolution.jsonl"
    best_metric = float("-inf")
    best_epoch = 0

    split_summary = {
        "train": summarize_split_cases(train_cases),
        "val": summarize_split_cases(val_cases),
        "test": summarize_split_cases(test_cases),
        "path": str(split_path),
    }
    print(
        f"train_cases={len(train_cases)} val_cases={len(val_cases)} test_cases={len(test_cases)} epochs={args.epochs}"
    )

    for epoch in range(1, args.epochs + 1):
        epoch_rng = random.Random(args.seed + epoch)
        shuffled = list(train_cases)
        epoch_rng.shuffle(shuffled)
        schedule = build_epoch_schedule(
            epoch,
            rule_compression_start_epoch=args.rule_compression_start_epoch,
            rule_memory_start_epoch=args.rule_memory_start_epoch,
            rule_learning_start_epoch=args.rule_learning_start_epoch,
        )

        train_summary = run_pass(
            cases=shuffled,
            bank=bank,
            skill_index=skill_index,
            reranker=reranker,
            controller=controller,
            final_scorer=final_scorer,
            rule_scorer=rule_scorer,
            update_online=True,
            use_reflection=True,
            use_controller=args.enable_controller,
            use_compare=not args.disable_compare,
            use_malignancy=malignancy_enabled,
            use_metadata_consistency=not args.disable_metadata_consistency,
            use_final_scorer=args.enable_final_scorer and args.enable_controller,
            use_rule_memory=schedule["use_rule_memory"],
            enable_rule_compression=schedule["enable_rule_compression"],
            update_rule_scorer=schedule["update_rule_scorer"],
        )
        train_summary["schedule"] = schedule

        evolution_summary: Dict[str, Any] | None = None
        if should_run_skill_evolution(
            epoch,
            start_epoch=args.skill_evolution_start_epoch,
            every=args.skill_evolution_every,
        ):
            evolution_summary = designer.evolve(bank=bank, skill_index=skill_index)
            train_summary["skill_evolution"] = evolution_summary

        latest_controller_path = save_dir / "latest_controller.json"
        latest_bank_path = save_dir / "latest_bank.json"
        save_controller_checkpoint(
            latest_controller_path,
            skill_index=skill_index,
            controller=controller,
            final_scorer=final_scorer,
            rule_scorer=rule_scorer,
            metadata={
                "epoch": epoch,
                "stage": "latest",
                "train_summary": train_summary,
                "split_summary": split_summary,
                "designer_state_path": str(latest_designer_path),
                "run_name": args.run_name or save_dir.name,
                "init_mode": args.init_mode,
                "resume": args.resume,
            },
        )
        bank.save_json(latest_bank_path)
        latest_designer_path.write_text(json.dumps(designer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        record: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "schedule": schedule,
            "train": train_summary,
            "latest_controller": str(latest_controller_path),
            "latest_bank": str(latest_bank_path),
            "latest_skill_designer": str(latest_designer_path),
        }
        if evolution_summary is not None:
            record["skill_evolution"] = evolution_summary
            append_jsonl(
                evolution_log_path,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "epoch": epoch,
                    "skill_evolution": evolution_summary,
                },
            )

        val_summary: Dict[str, Any] | None = None
        if val_cases and epoch % max(1, args.eval_every) == 0:
            val_summary = run_pass(
                cases=val_cases,
                bank=bank,
                skill_index=skill_index,
                reranker=reranker,
                controller=controller,
                final_scorer=final_scorer,
                rule_scorer=rule_scorer,
                update_online=False,
                use_reflection=False,
                use_controller=args.enable_controller,
                use_compare=not args.disable_compare,
                use_malignancy=malignancy_enabled,
                use_metadata_consistency=not args.disable_metadata_consistency,
                use_final_scorer=args.enable_final_scorer and args.enable_controller,
                use_rule_memory=True,
                enable_rule_compression=False,
                update_rule_scorer=False,
            )
            record["val"] = val_summary
            metric_value = float(val_summary["metrics"]["accuracy_top1"])
            if metric_value > best_metric:
                best_metric = metric_value
                best_epoch = epoch
                best_controller_path = save_dir / "best_controller.json"
                best_bank_path = save_dir / "best_bank.json"
                save_controller_checkpoint(
                    best_controller_path,
                    skill_index=skill_index,
                    controller=controller,
                    final_scorer=final_scorer,
                    rule_scorer=rule_scorer,
                    metadata={
                        "epoch": epoch,
                        "stage": "best",
                        "val_summary": val_summary,
                        "split_summary": split_summary,
                        "designer_state_path": str(latest_designer_path),
                        "run_name": args.run_name or save_dir.name,
                        "init_mode": args.init_mode,
                        "resume": args.resume,
                    },
                )
                bank.save_json(best_bank_path)
                best_designer_path = save_dir / "best_skill_designer.json"
                best_designer_path.write_text(json.dumps(designer.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                record["best"] = {
                    "epoch": best_epoch,
                    "metric": round(best_metric, 4),
                    "controller": str(best_controller_path),
                    "bank": str(best_bank_path),
                    "skill_designer": str(best_designer_path),
                }

        append_jsonl(train_log_path, record)
        print(render_epoch_line(epoch=epoch, train=train_summary, val_summary=val_summary, best_metric=best_metric))

    latest_test_summary = run_pass(
        cases=test_cases,
        bank=bank,
        skill_index=skill_index,
        reranker=reranker,
        controller=controller,
        final_scorer=final_scorer,
        rule_scorer=rule_scorer,
        update_online=False,
        use_reflection=False,
        use_controller=args.enable_controller,
        use_compare=not args.disable_compare,
        use_malignancy=malignancy_enabled,
        use_metadata_consistency=not args.disable_metadata_consistency,
        use_final_scorer=args.enable_final_scorer and args.enable_controller,
        use_rule_memory=True,
        enable_rule_compression=False,
        update_rule_scorer=False,
    ) if test_cases else None

    best_test_summary = None
    best_controller_path = save_dir / "best_controller.json"
    best_bank_path = save_dir / "best_bank.json"
    if test_cases and best_controller_path.exists():
        best_bank = ExperienceBank.from_json(best_bank_path) if best_bank_path.exists() else bank
        best_skill_index, best_controller_payload, best_final_payload, best_rule_payload = load_controller_checkpoint(best_controller_path)
        best_controller = LearnableSkillController(best_skill_index)
        if best_controller_payload:
            best_controller.load_state(best_controller_payload)
        best_final_scorer = LearnableFinalScorer()
        if best_final_payload:
            best_final_scorer.load_state(best_final_payload)
        best_rule_scorer = LearnableRuleScorer()
        if best_rule_payload:
            best_rule_scorer.load_state(best_rule_payload)
        best_test_summary = run_pass(
            cases=test_cases,
            bank=best_bank,
            skill_index=best_skill_index,
            reranker=UtilityAwareExperienceReranker(),
            controller=best_controller,
            final_scorer=best_final_scorer,
            rule_scorer=best_rule_scorer,
            update_online=False,
            use_reflection=False,
            use_controller=args.enable_controller,
            use_compare=not args.disable_compare,
            use_malignancy=malignancy_enabled,
            use_metadata_consistency=not args.disable_metadata_consistency,
            use_final_scorer=args.enable_final_scorer and args.enable_controller,
            use_rule_memory=True,
            enable_rule_compression=False,
            update_rule_scorer=False,
        )

    summary = {
        "run_name": args.run_name or save_dir.name,
        "dataset_root": args.dataset_root,
        "num_total_cases": len(all_cases),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_metric": round(best_metric, 4) if best_metric > float("-inf") else None,
        "save_dir": str(save_dir),
        "initialization": init_summary,
        "config": run_config,
        "split": split_summary,
        "curriculum": {
            "rule_compression_start_epoch": args.rule_compression_start_epoch,
            "rule_memory_start_epoch": args.rule_memory_start_epoch,
            "rule_learning_start_epoch": args.rule_learning_start_epoch,
            "skill_evolution_start_epoch": args.skill_evolution_start_epoch,
            "skill_evolution_every": args.skill_evolution_every,
        },
        "latest_controller": str(save_dir / "latest_controller.json"),
        "latest_bank": str(save_dir / "latest_bank.json"),
        "latest_skill_designer": str(latest_designer_path),
        "best_controller": str(best_controller_path),
        "best_bank": str(best_bank_path),
        "best_skill_designer": str(save_dir / "best_skill_designer.json"),
        "train_log": str(train_log_path),
        "skill_evolution_log": str(evolution_log_path),
        "latest_test": latest_test_summary,
        "best_test": best_test_summary,
    }
    (save_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_run_manifest(
        path=manifest_path,
        payload={
            "format_version": 1,
            "run_name": args.run_name or save_dir.name,
            "save_dir": str(save_dir),
            "status": "completed",
            "created_at": load_manifest_created_at(manifest_path),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "initialization": init_summary,
            "config": run_config,
            "artifacts": build_artifact_summary(save_dir),
            "summary_path": str(save_dir / "train_summary.json"),
            "best_metric": summary["best_metric"],
            "best_epoch": summary["best_epoch"],
        },
        resume=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_epoch_schedule(
    epoch: int,
    *,
    rule_compression_start_epoch: int,
    rule_memory_start_epoch: int,
    rule_learning_start_epoch: int,
) -> Dict[str, Any]:
    return {
        "phase": _phase_name(
            epoch,
            rule_compression_start_epoch=rule_compression_start_epoch,
            rule_memory_start_epoch=rule_memory_start_epoch,
            rule_learning_start_epoch=rule_learning_start_epoch,
        ),
        "use_rule_memory": epoch >= rule_memory_start_epoch,
        "enable_rule_compression": epoch >= rule_compression_start_epoch,
        "update_rule_scorer": epoch >= rule_learning_start_epoch,
    }


def _phase_name(
    epoch: int,
    *,
    rule_compression_start_epoch: int,
    rule_memory_start_epoch: int,
    rule_learning_start_epoch: int,
) -> str:
    if epoch < rule_compression_start_epoch:
        return "memory_warmup"
    if epoch < rule_memory_start_epoch:
        return "rule_build_only"
    if epoch < rule_learning_start_epoch:
        return "rule_inference_warmup"
    return "full_training"


def should_run_skill_evolution(epoch: int, *, start_epoch: int, every: int) -> bool:
    if epoch < start_epoch:
        return False
    return (epoch - start_epoch) % max(1, every) == 0


def run_pass(
    *,
    cases: Sequence[Dict[str, Any]],
    bank: ExperienceBank,
    skill_index,
    reranker: UtilityAwareExperienceReranker,
    controller: LearnableSkillController,
    final_scorer: LearnableFinalScorer,
    rule_scorer: LearnableRuleScorer,
    update_online: bool,
    use_reflection: bool,
    use_controller: bool,
    use_compare: bool,
    use_malignancy: bool,
    use_metadata_consistency: bool,
    use_final_scorer: bool,
    use_rule_memory: bool,
    enable_rule_compression: bool,
    update_rule_scorer: bool,
) -> Dict[str, Any]:
    total = correct_top1 = correct_top3 = malignant_total = malignant_hit = ack_scc_total = ack_scc_correct = errors = 0
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
            use_reflection=use_reflection,
            use_controller=use_controller,
            use_compare=use_compare,
            use_malignancy=use_malignancy,
            use_metadata_consistency=use_metadata_consistency,
            use_final_scorer=use_final_scorer,
            update_online=update_online,
            use_rule_memory=use_rule_memory,
            enable_rule_compression=enable_rule_compression,
            update_rule_scorer=update_rule_scorer,
        )
        total += 1
        if result.get("error"):
            errors += 1
        true_label = str(case.get("label", "")).strip().upper()
        final_decision = result.get("final_decision", {}) or {}
        pred_label = str(final_decision.get("final_label") or final_decision.get("diagnosis") or "").strip().upper()
        top3 = []
        for item in (final_decision.get("top_k", []) or [])[:3]:
            name = str(item.get("name", "")).strip().upper() if isinstance(item, dict) else str(item).strip().upper()
            if name:
                top3.append(name)
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
        "num_cases": total,
        "metrics": {
            "accuracy_top1": safe_div(correct_top1, total),
            "accuracy_top3": safe_div(correct_top3, total),
            "malignant_recall": safe_div(malignant_hit, malignant_total),
            "confusion_accuracy": safe_div(ack_scc_correct, ack_scc_total),
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
        "bank_stats": bank.stats(),
        "update_online": update_online,
        "use_reflection": use_reflection,
        "use_controller": use_controller,
        "use_compare": use_compare,
        "use_malignancy": use_malignancy,
        "use_metadata_consistency": use_metadata_consistency,
        "use_final_scorer": use_final_scorer,
        "use_rule_memory": use_rule_memory,
        "enable_rule_compression": enable_rule_compression,
        "update_rule_scorer": update_rule_scorer,
    }


def render_epoch_line(*, epoch: int, train: Dict[str, Any], val_summary: Dict[str, Any] | None, best_metric: float) -> str:
    train_metrics = train.get("metrics", {}) or {}
    schedule = train.get("schedule", {}) or {}
    text = (
        f"epoch={epoch} phase={schedule.get('phase')} "
        f"train_top1={train_metrics.get('accuracy_top1', 0.0)} "
        f"train_top3={train_metrics.get('accuracy_top3', 0.0)}"
    )
    if val_summary is not None:
        val_metrics = val_summary.get("metrics", {}) or {}
        best_text = "None" if best_metric == float("-inf") else round(best_metric, 4)
        text += (
            f" val_top1={val_metrics.get('accuracy_top1', 0.0)} "
            f"val_top3={val_metrics.get('accuracy_top3', 0.0)} "
            f"best_top1={best_text}"
        )
    return text


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def safe_div(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 4)


def validate_save_dir_state(*, save_dir: Path, resume: bool, allow_dirty_save_dir: bool) -> None:
    existing_artifacts = find_existing_run_artifacts(save_dir)
    if not existing_artifacts:
        return
    if resume or allow_dirty_save_dir:
        return
    artifact_list = ", ".join(existing_artifacts)
    raise SystemExit(
        f"Refusing to start a fresh run in non-empty save dir: {save_dir}. "
        f"Found existing run artifacts: {artifact_list}. "
        "Use a new SAVE_DIR, or pass --resume for continuation."
    )


def find_existing_run_artifacts(save_dir: Path) -> List[str]:
    if not save_dir.exists():
        return []
    found = sorted(name for name in RUN_ARTIFACT_NAMES if (save_dir / name).exists())
    if found:
        return found
    generated_files = []
    for child in sorted(save_dir.iterdir()):
        if child.is_file():
            generated_files.append(child.name)
    return generated_files


def build_init_summary(
    *,
    init_mode: str,
    base_run_dir: str | None,
    controller_in: str | None,
    bank_in: str | None,
    resume: bool,
) -> Dict[str, Any]:
    controller_path = str(controller_in) if controller_in else None
    bank_path = str(bank_in) if bank_in else None
    return {
        "mode": str(init_mode).strip() or "clean",
        "resume": bool(resume),
        "base_run_dir": str(base_run_dir) if base_run_dir else None,
        "controller_state_in": controller_path,
        "bank_state_in": bank_path,
        "controller_state_exists": bool(controller_path and Path(controller_path).exists()),
        "bank_state_exists": bool(bank_path and Path(bank_path).exists()),
    }


def build_run_config(*, args: argparse.Namespace, save_dir: Path) -> Dict[str, Any]:
    return {
        "run_name": args.run_name or save_dir.name,
        "dataset_root": args.dataset_root,
        "limit": args.limit,
        "epochs": args.epochs,
        "seed": args.seed,
        "eval_every": args.eval_every,
        "save_dir": str(save_dir),
        "split_json": args.split_json,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "rule_compression_start_epoch": args.rule_compression_start_epoch,
        "rule_memory_start_epoch": args.rule_memory_start_epoch,
        "rule_learning_start_epoch": args.rule_learning_start_epoch,
        "skill_evolution_start_epoch": args.skill_evolution_start_epoch,
        "skill_evolution_every": args.skill_evolution_every,
        "enable_controller": bool(args.enable_controller),
        "enable_final_scorer": bool(args.enable_final_scorer),
        "disable_compare": bool(args.disable_compare),
        "enable_malignancy": bool(args.enable_malignancy) and not bool(args.disable_malignancy),
        "disable_metadata_consistency": bool(args.disable_metadata_consistency),
        "allow_dirty_save_dir": bool(args.allow_dirty_save_dir),
        "notes": str(args.notes or ""),
        "argv": list(sys.argv),
    }


def build_artifact_summary(save_dir: Path) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    for name in sorted(RUN_ARTIFACT_NAMES):
        path = save_dir / name
        if path.exists():
            artifacts[name] = str(path)
    return artifacts


def write_run_manifest(*, path: Path, payload: Dict[str, Any], resume: bool) -> None:
    if path.exists() and not resume:
        raise SystemExit(
            f"Run manifest already exists: {path}. "
            "Use a new SAVE_DIR, or pass --resume if you mean to continue this run."
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest_created_at(path: Path) -> str:
    if not path.exists():
        return datetime.now().isoformat(timespec="seconds")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload.get("created_at") or datetime.now().isoformat(timespec="seconds"))


if __name__ == "__main__":
    main()
