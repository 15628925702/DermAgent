from __future__ import annotations

import argparse
import json
from pathlib import Path


def _safe_float(value: object) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize quick compare results and print a simple verdict.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--min-top1-gain", type=float, default=0.10)
    parser.add_argument("--min-confusion-gain", type=float, default=0.00)
    parser.add_argument("--max-mal-drop", type=float, default=0.05)
    parser.add_argument("--max-top3-drop", type=float, default=0.10)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    delta = payload.get("delta", {}) or {}
    compare_valid = bool(payload.get("compare_valid", False))
    num_cases = int(payload.get("num_cases", 0) or 0)

    top1_delta = _safe_float(delta.get("accuracy_top1"))
    top3_delta = _safe_float(delta.get("accuracy_top3"))
    mal_delta = _safe_float(delta.get("malignant_recall"))
    confusion_delta = _safe_float(delta.get("confusion_accuracy"))

    if not compare_valid:
        verdict = "INVALID"
        reason = "baseline_failed"
    elif (
        top1_delta >= args.min_top1_gain
        and confusion_delta >= args.min_confusion_gain
        and mal_delta >= -args.max_mal_drop
        and top3_delta >= -args.max_top3_drop
    ):
        verdict = "PROMISING"
        reason = "keep_going"
    elif top1_delta > 0.0 or confusion_delta > 0.0:
        verdict = "MIXED"
        reason = "some_gains_but_not_stable"
    else:
        verdict = "REJECT"
        reason = "no_useful_gain"

    print(f"quick_cases={num_cases}")
    print(
        "quick_delta="
        f"top1:{top1_delta:.4f} "
        f"top3:{top3_delta:.4f} "
        f"mal_recall:{mal_delta:.4f} "
        f"ack_scc:{confusion_delta:.4f}"
    )
    print(f"quick_verdict={verdict} reason={reason}")

    if verdict == "PROMISING":
        print("quick_next=run_longer")
    elif verdict == "MIXED":
        print("quick_next=inspect_then_tune")
    elif verdict == "INVALID":
        print("quick_next=fix_baseline_or_service")
    else:
        print("quick_next=discard_this_direction")


if __name__ == "__main__":
    main()
