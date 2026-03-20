from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_eval import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compact DermAgent evaluation summary.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default=None, choices=["train", "val", "test"])
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--disable-specialist", action="store_true")
    parser.add_argument("--disable-reflection", action="store_true")
    parser.add_argument("--enable-controller", action="store_true")
    parser.add_argument("--disable-controller", action="store_true")
    parser.add_argument("--disable-compare", action="store_true")
    parser.add_argument("--disable-malignancy", action="store_true")
    parser.add_argument("--disable-metadata-consistency", action="store_true")
    parser.add_argument("--enable-final-scorer", action="store_true")
    parser.add_argument("--disable-final-scorer", action="store_true")
    parser.add_argument("--disable-rule-memory", action="store_true")
    parser.add_argument("--disable-rule-compression", action="store_true")
    parser.add_argument("--controller-state-in", default=None)
    parser.add_argument("--controller-state-out", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--bank-state-out", default=None)
    parser.add_argument("--freeze-learning", action="store_true")
    args = parser.parse_args()

    controller_enabled = bool(args.enable_controller)
    if args.disable_controller:
        controller_enabled = False
    final_scorer_enabled = bool(args.enable_final_scorer)
    if args.disable_final_scorer:
        final_scorer_enabled = False
    if not controller_enabled:
        final_scorer_enabled = False

    result = run_evaluation(
        dataset_root=args.dataset_root,
        limit=args.limit,
        split_json=args.split_json,
        split_name=args.split_name,
        use_retrieval=not args.disable_retrieval,
        use_specialist=not args.disable_specialist,
        use_reflection=not args.disable_reflection,
        use_controller=controller_enabled,
        use_compare=not args.disable_compare,
        use_malignancy=not args.disable_malignancy,
        use_metadata_consistency=not args.disable_metadata_consistency,
        use_final_scorer=final_scorer_enabled,
        controller_state_in=args.controller_state_in,
        controller_state_out=args.controller_state_out,
        bank_state_in=args.bank_state_in,
        bank_state_out=args.bank_state_out,
        update_online=not args.freeze_learning,
        use_rule_memory=not args.disable_rule_memory,
        enable_rule_compression=not args.disable_rule_compression,
    )

    metrics = result.get("metrics", {}) or {}
    counts = result.get("counts", {}) or {}
    runtime = result.get("runtime_flags", {}) or {}
    bank_stats = result.get("bank_stats", {}) or {}
    split = result.get("split", {}) or {}

    print(f"cases={result.get('num_cases', 0)} split={split.get('name') or 'all'}")
    print(
        "metrics="
        f"top1:{metrics.get('accuracy_top1', 0.0)} "
        f"top3:{metrics.get('accuracy_top3', 0.0)} "
        f"mal_recall:{metrics.get('malignant_recall', 0.0)} "
        f"ack_scc:{metrics.get('confusion_accuracy', 0.0)}"
    )
    print(
        "counts="
        f"correct_top1:{counts.get('correct_top1', 0)} "
        f"correct_top3:{counts.get('correct_top3', 0)} "
        f"errors:{counts.get('errors', 0)}"
    )
    print(
        "runtime="
        f"retrieval:{runtime.get('use_retrieval')} "
        f"specialist:{runtime.get('use_specialist')} "
        f"reflection:{runtime.get('use_reflection')} "
        f"controller:{runtime.get('use_controller')} "
        f"compare:{runtime.get('use_compare')} "
        f"malignancy:{runtime.get('use_malignancy')} "
        f"metadata:{runtime.get('use_metadata_consistency')} "
        f"final_scorer:{runtime.get('use_final_scorer')} "
        f"rule_memory:{runtime.get('use_rule_memory')} "
        f"update_online:{runtime.get('update_online')}"
    )
    print(
        "bank="
        f"total:{bank_stats.get('total', 0)} "
        f"raw:{bank_stats.get('raw_case', 0)} "
        f"prototype:{bank_stats.get('prototype', 0)} "
        f"confusion:{bank_stats.get('confusion', 0)} "
        f"rule:{bank_stats.get('rule', 0)} "
        f"hard:{bank_stats.get('hard_case', 0)}"
    )


if __name__ == "__main__":
    main()
