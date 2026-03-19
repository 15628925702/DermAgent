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

from agent.run_agent import run_agent
from memory.experience_bank import ExperienceBank


MALIGNANT_LABELS = {"MEL", "BCC", "SCC"}
ACK_SCC_LABELS = {"ACK", "SCC"}
SUPPORTED_LABELS = {"MEL", "NEV", "SCC", "BCC", "ACK", "SEK"}


def load_pad_ufes20_cases(
    dataset_root: str | Path = "data/pad_ufes_20",
    limit: int | None = None,
) -> List[Dict[str, Any]]:
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

            cases.append(
                {
                    "file": img_id,
                    "image_path": image_path,
                    "metadata": metadata,
                    "text": "",
                    "label": label,
                }
            )

            if limit is not None and len(cases) >= limit:
                break

    return cases


def run_evaluation(
    dataset_root: str | Path = "data/pad_ufes_20",
    limit: int | None = None,
    use_retrieval: bool = True,
    use_specialist: bool = True,
    use_reflection: bool = True,
) -> Dict[str, Any]:
    cases = load_pad_ufes20_cases(dataset_root=dataset_root, limit=limit)
    bank = ExperienceBank()

    total = 0
    correct_top1 = 0
    correct_top3 = 0
    malignant_total = 0
    malignant_hit = 0
    ack_scc_total = 0
    ack_scc_correct = 0
    errors = 0

    per_case: List[Dict[str, Any]] = []

    for case in cases:
        result = run_agent(
            case=case,
            bank=bank,
            use_retrieval=use_retrieval,
            use_specialist=use_specialist,
            use_reflection=use_reflection,
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

        per_case.append(
            {
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
            }
        )

    return {
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
        },
        "bank_stats": bank.stats(),
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DermAgent evaluation.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--disable-specialist", action="store_true")
    parser.add_argument("--disable-reflection", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_evaluation(
        dataset_root=args.dataset_root,
        limit=args.limit,
        use_retrieval=not args.disable_retrieval,
        use_specialist=not args.disable_specialist,
        use_reflection=not args.disable_reflection,
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
        if isinstance(item, dict):
            label = _norm_label(item.get("name"))
        else:
            label = _norm_label(item)
        if label:
            labels.append(label)
    return labels


def _build_history_text(row: Dict[str, Any]) -> str:
    history_bits = []
    for key in ["itch", "grew", "hurt", "changed", "bleed", "elevation"]:
        value = str(row.get(key, "")).strip().lower()
        if value == "true":
            history_bits.append(key)
    return "; ".join(history_bits)


def _safe_div(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


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
