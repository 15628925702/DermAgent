from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.ham10000 import load_ham10000_cases, summarize_ham10000_metadata
from datasets.splits import save_split_manifest, select_split_cases, summarize_split_cases


DEFAULT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}


def build_grouped_ham10000_split(
    cases: Sequence[Dict[str, Any]],
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    ratios = _normalize_ratios(
        {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        }
    )
    grouped_by_label: Dict[str, List[List[Dict[str, Any]]]] = defaultdict(list)
    for _, group_cases in _group_cases_by_lesion(cases).items():
        label = str(group_cases[0].get("label", "")).strip().upper() or "UNKNOWN"
        grouped_by_label[label].append(group_cases)

    split_case_ids = {"train": [], "val": [], "test": []}
    split_label_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for label, label_groups in sorted(grouped_by_label.items()):
        shuffled_groups = list(label_groups)
        label_seed = seed + sum(ord(ch) for ch in label)
        random.Random(label_seed).shuffle(shuffled_groups)
        allocation = _allocate_counts(len(shuffled_groups), ratios)

        cursor = 0
        for split_name in ("train", "val", "test"):
            num_groups = allocation[split_name]
            selected_groups = shuffled_groups[cursor:cursor + num_groups]
            cursor += num_groups

            split_cases = [case for group in selected_groups for case in group]
            split_case_ids[split_name].extend(_case_id(case) for case in split_cases)
            split_label_counts[split_name][label] = len(split_cases)

    return {
        "format_version": 1,
        "seed": seed,
        "ratios": ratios,
        "grouping": "lesion_id",
        "num_cases": len(cases),
        "splits": split_case_ids,
        "label_counts": split_label_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic grouped HAM10000 split manifest.")
    parser.add_argument("--dataset-root", default="data/ham10000")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--output", default="outputs/splits/ham10000_seed42.json")
    args = parser.parse_args()

    cases = load_ham10000_cases(dataset_root=args.dataset_root, limit=args.limit)
    manifest = build_grouped_ham10000_split(
        cases,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    output_path = save_split_manifest(manifest, args.output)
    metadata_summary = summarize_ham10000_metadata(dataset_root=args.dataset_root)

    print("HAM10000 mapped label counts:")
    for label, count in sorted(metadata_summary["mapped_agent_label_counts"].items()):
        print(f"  {label}: {count}")

    print("HAM10000 dropped raw labels:")
    for raw_label, count in sorted(metadata_summary["dropped_raw_label_counts"].items()):
        print(f"  {raw_label}: {count}")

    print("Split case counts:")
    split_summary = {}
    for split_name in ("train", "val", "test"):
        summary = summarize_split_cases(select_split_cases(cases, manifest, split_name))
        split_summary[split_name] = summary
        print(f"  [{split_name}] total={summary['num_cases']}")
        for label, count in sorted((summary.get("label_counts") or {}).items()):
            print(f"    {label}: {count}")

    payload = {
        "dataset_root": str(args.dataset_root),
        "output_path": str(output_path),
        "num_cases": len(cases),
        "split_summary": split_summary,
        "mapping_summary": metadata_summary,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _group_cases_by_lesion(cases: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        metadata = case.get("metadata", {}) or {}
        lesion_id = str(metadata.get("lesion_id", "")).strip() or _case_id(case)
        groups[lesion_id].append(case)
    return groups


def _case_id(case: Dict[str, Any]) -> str:
    file_name = str(case.get("file", "")).strip()
    if file_name:
        return file_name
    image_path = str(case.get("image_path", "")).strip()
    if image_path:
        return Path(image_path).name
    return "unknown_case"


def _normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in ratios.values())
    if total <= 0:
        return dict(DEFAULT_RATIOS)
    return {
        key: round(max(0.0, float(value)) / total, 6)
        for key, value in ratios.items()
    }


def _allocate_counts(num_items: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if num_items <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if num_items == 1:
        return {"train": 1, "val": 0, "test": 0}
    if num_items == 2:
        return {"train": 1, "val": 0, "test": 1}
    if num_items == 3:
        return {"train": 1, "val": 1, "test": 1}
    if num_items == 4:
        return {"train": 2, "val": 1, "test": 1}

    raw = {
        split_name: float(num_items) * float(ratios.get(split_name, 0.0))
        for split_name in ("train", "val", "test")
    }
    counts = {split_name: int(raw[split_name]) for split_name in raw}
    counts["train"] = max(1, counts["train"])
    counts["val"] = max(1, counts["val"])
    counts["test"] = max(1, counts["test"])

    while sum(counts.values()) > num_items:
        largest = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        if counts[largest] > 1:
            counts[largest] -= 1
        else:
            break

    remainders = sorted(
        (raw[split_name] - int(raw[split_name]), split_name)
        for split_name in ("train", "val", "test")
    )
    while sum(counts.values()) < num_items:
        _, split_name = remainders.pop()
        counts[split_name] += 1
        if not remainders:
            remainders = sorted(
                (raw[name] - int(raw[name]), name)
                for name in ("train", "val", "test")
            )

    return counts


if __name__ == "__main__":
    main()
