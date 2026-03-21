from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval import run_evaluation


PROTECTED_METRICS = (
    "accuracy_top1",
    "accuracy_top3",
    "malignant_recall",
    "confusion_accuracy",
)


def evaluate_run(
    *,
    run_dir: str | Path,
    dataset_root: str,
    split_json: str | None,
    split_name: str | None,
    limit: int | None,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    controller_path = run_path / "best_controller.json"
    bank_path = run_path / "best_bank.json"
    manifest_path = run_path / "run_manifest.json"
    if not controller_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {controller_path}")
    if not bank_path.exists():
        raise FileNotFoundError(f"Missing bank checkpoint: {bank_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    result = run_evaluation(
        dataset_root=dataset_root,
        limit=limit,
        split_json=split_json,
        split_name=split_name,
        use_retrieval=True,
        use_specialist=True,
        use_reflection=True,
        use_controller=False,
        use_compare=True,
        use_malignancy=False,
        use_metadata_consistency=True,
        use_final_scorer=False,
        controller_state_in=str(controller_path),
        bank_state_in=str(bank_path),
        update_online=False,
        use_rule_memory=True,
        enable_rule_compression=False,
    )
    return {
        "run_dir": str(run_path),
        "manifest": manifest,
        "controller_path": str(controller_path),
        "bank_path": str(bank_path),
        "metrics": result.get("metrics", {}) or {},
        "counts": result.get("counts", {}) or {},
        "bank_stats": result.get("bank_stats", {}) or {},
        "runtime_flags": result.get("runtime_flags", {}) or {},
        "split": result.get("split", {}) or {},
    }


def compare_runs(
    *,
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    min_top1_gain: float,
    min_top3_gain: float,
    min_malignant_recall_gain: float,
    min_confusion_gain: float,
) -> Dict[str, Any]:
    thresholds = {
        "accuracy_top1": min_top1_gain,
        "accuracy_top3": min_top3_gain,
        "malignant_recall": min_malignant_recall_gain,
        "confusion_accuracy": min_confusion_gain,
    }
    deltas: Dict[str, float] = {}
    failures = []
    for metric in PROTECTED_METRICS:
        baseline_value = float((baseline.get("metrics", {}) or {}).get(metric, 0.0))
        candidate_value = float((candidate.get("metrics", {}) or {}).get(metric, 0.0))
        delta = round(candidate_value - baseline_value, 4)
        deltas[metric] = delta
        if delta < thresholds[metric]:
            failures.append(
                {
                    "metric": metric,
                    "baseline": round(baseline_value, 4),
                    "candidate": round(candidate_value, 4),
                    "delta": delta,
                    "required_min_delta": thresholds[metric],
                }
            )

    decision = "approve" if not failures else "reject"
    return {
        "decision": decision,
        "deltas": deltas,
        "thresholds": thresholds,
        "failures": failures,
    }


def promote_run(candidate_run_dir: str | Path, promote_dir: str | Path) -> Dict[str, str]:
    candidate_path = Path(candidate_run_dir)
    promote_path = Path(promote_dir)
    promote_path.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, str] = {}
    for name in ["best_controller.json", "best_bank.json", "best_skill_designer.json", "train_summary.json", "run_manifest.json"]:
        src = candidate_path / name
        if not src.exists():
            continue
        dst = promote_path / name
        shutil.copy2(src, dst)
        copied[name] = str(dst)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a training run against a baseline and optionally delete or promote it.")
    parser.add_argument("--baseline-run-dir", required=True)
    parser.add_argument("--candidate-run-dir", required=True)
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--split-json", default="outputs/splits/pad_ufes20_full.json")
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-top1-gain", type=float, default=0.0)
    parser.add_argument("--min-top3-gain", type=float, default=0.0)
    parser.add_argument("--min-malignant-recall-gain", type=float, default=0.0)
    parser.add_argument("--min-confusion-gain", type=float, default=0.0)
    parser.add_argument("--delete-rejected", action="store_true")
    parser.add_argument("--promote-dir", default=None)
    args = parser.parse_args()

    baseline = evaluate_run(
        run_dir=args.baseline_run_dir,
        dataset_root=args.dataset_root,
        split_json=args.split_json,
        split_name=args.split_name,
        limit=args.limit,
    )
    candidate = evaluate_run(
        run_dir=args.candidate_run_dir,
        dataset_root=args.dataset_root,
        split_json=args.split_json,
        split_name=args.split_name,
        limit=args.limit,
    )
    comparison = compare_runs(
        baseline=baseline,
        candidate=candidate,
        min_top1_gain=args.min_top1_gain,
        min_top3_gain=args.min_top3_gain,
        min_malignant_recall_gain=args.min_malignant_recall_gain,
        min_confusion_gain=args.min_confusion_gain,
    )

    deleted = False
    promoted_files: Dict[str, str] = {}
    candidate_path = Path(args.candidate_run_dir)

    if comparison["decision"] == "reject" and args.delete_rejected and candidate_path.exists():
        shutil.rmtree(candidate_path)
        deleted = True

    if comparison["decision"] == "approve" and args.promote_dir:
        promoted_files = promote_run(candidate_path, args.promote_dir)

    result = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
        "deleted_rejected_run": deleted,
        "promoted_files": promoted_files,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
