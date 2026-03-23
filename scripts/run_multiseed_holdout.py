#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_agent_vs_qwen import ComparisonFramework


def _mean_std(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": round(values[0], 4), "std": 0.0}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated holdout comparison across multiple random seeds.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--controller-checkpoint", default=None)
    parser.add_argument("--bank-state-in", default=None)
    parser.add_argument("--disable-retrieval", action="store_true")
    parser.add_argument("--disable-specialist", action="store_true")
    parser.add_argument("--enable-controller", action="store_true")
    parser.add_argument("--disable-controller", action="store_true")
    parser.add_argument("--output-dir", default="outputs/multiseed")
    args = parser.parse_args()

    use_controller = None
    if args.enable_controller:
        use_controller = True
    if args.disable_controller:
        use_controller = False

    runs: List[Dict[str, Any]] = []
    for seed in args.seeds:
        split_json = str(Path("outputs/splits") / f"{Path(args.dataset_root).name}_seed{seed}.json")
        framework = ComparisonFramework(
            test_limit=args.test_limit,
            dataset_root=args.dataset_root,
            split_json=split_json,
            split_name=args.split_name,
            seed=seed,
            controller_checkpoint=args.controller_checkpoint,
            bank_state_in=args.bank_state_in,
            online_learning=False,
            use_retrieval=not args.disable_retrieval,
            use_specialist=not args.disable_specialist,
            use_controller=use_controller,
        )
        print(f"\nRunning multiseed holdout for seed={seed} split={args.split_name}")
        report = framework.run_full_comparison()
        comparison = report.get("comparison", {}) or {}
        agent_metrics = (report.get("agent_results") or {}).get("metrics", {}) or {}
        qwen_metrics = (report.get("qwen_direct_results") or {}).get("metrics", {}) or {}
        runs.append(
            {
                "seed": seed,
                "split_json": split_json,
                "test_count": report.get("test_count", 0),
                "agent_top1": float(agent_metrics.get("accuracy_top1", 0.0)),
                "agent_malignant_recall": float(agent_metrics.get("malignant_recall", 0.0)),
                "qwen_top1": float(qwen_metrics.get("accuracy_top1", 0.0)),
                "qwen_malignant_recall": float(qwen_metrics.get("malignant_recall", 0.0)),
                "fairness": comparison.get("fairness", {}) or {},
            }
        )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset_root": args.dataset_root,
        "split_name": args.split_name,
        "seeds": args.seeds,
        "runs": runs,
        "aggregate": {
            "agent_top1": _mean_std([x["agent_top1"] for x in runs]),
            "agent_malignant_recall": _mean_std([x["agent_malignant_recall"] for x in runs]),
            "qwen_top1": _mean_std([x["qwen_top1"] for x in runs]),
            "qwen_malignant_recall": _mean_std([x["qwen_malignant_recall"] for x in runs]),
            "top1_improvement": _mean_std([x["agent_top1"] - x["qwen_top1"] for x in runs]),
            "malignant_recall_improvement": _mean_std([x["agent_malignant_recall"] - x["qwen_malignant_recall"] for x in runs]),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"multiseed_holdout_{ts}.json"
    txt_path = output_dir / f"multiseed_holdout_{ts}.txt"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "DermAgent Multiseed Holdout Summary",
        f"split_name: {args.split_name}",
        f"seeds: {args.seeds}",
        "",
        f"agent_top1 mean/std: {summary['aggregate']['agent_top1']['mean']} / {summary['aggregate']['agent_top1']['std']}",
        f"qwen_top1 mean/std: {summary['aggregate']['qwen_top1']['mean']} / {summary['aggregate']['qwen_top1']['std']}",
        f"top1 improvement mean/std: {summary['aggregate']['top1_improvement']['mean']} / {summary['aggregate']['top1_improvement']['std']}",
        f"agent malignant recall mean/std: {summary['aggregate']['agent_malignant_recall']['mean']} / {summary['aggregate']['agent_malignant_recall']['std']}",
        f"qwen malignant recall mean/std: {summary['aggregate']['qwen_malignant_recall']['mean']} / {summary['aggregate']['qwen_malignant_recall']['std']}",
        f"malignant recall improvement mean/std: {summary['aggregate']['malignant_recall_improvement']['mean']} / {summary['aggregate']['malignant_recall_improvement']['std']}",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved multiseed JSON to {json_path}")
    print(f"Saved multiseed summary to {txt_path}")


if __name__ == "__main__":
    main()
