from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.splits import load_or_create_split_manifest, select_split_cases, summarize_split_cases
from evaluation.run_eval import load_pad_ufes20_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or reuse a deterministic PAD-UFES-20 split manifest.")
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cases = load_pad_ufes20_cases(dataset_root=args.dataset_root, limit=args.limit)
    default_output = Path("outputs/splits") / f"{Path(args.dataset_root).name}_seed{args.seed}.json"
    output_path = Path(args.output) if args.output else default_output
    payload = load_or_create_split_manifest(
        cases,
        output_path,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    summary = {
        "split_path": str(output_path),
        "dataset_root": str(args.dataset_root),
        "num_cases": len(cases),
        "splits": {
            split_name: summarize_split_cases(select_split_cases(cases, payload, split_name))
            for split_name in ["train", "val", "test"]
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
