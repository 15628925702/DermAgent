from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}


def get_case_id(case: Dict[str, Any]) -> str:
    file_name = str(case.get("file", "")).strip()
    if file_name:
        return file_name
    image_path = str(case.get("image_path", "")).strip()
    if image_path:
        return Path(image_path).name
    return "unknown_case"


def build_stratified_split(
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
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        label = str(case.get("label", "")).strip().upper() or "UNKNOWN"
        by_label[label].append(case)

    split_case_ids = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_label_counts: Dict[str, Dict[str, int]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for label, label_cases in sorted(by_label.items()):
        shuffled = list(label_cases)
        label_seed = seed + sum(ord(ch) for ch in label)
        random.Random(label_seed).shuffle(shuffled)
        allocation = _allocate_counts(len(shuffled), ratios)

        cursor = 0
        for split_name in ["train", "val", "test"]:
            count = allocation[split_name]
            selected = shuffled[cursor:cursor + count]
            cursor += count
            ids = [get_case_id(case) for case in selected]
            split_case_ids[split_name].extend(ids)
            split_label_counts[split_name][label] = len(ids)

    return {
        "format_version": 1,
        "seed": seed,
        "ratios": ratios,
        "num_cases": len(cases),
        "splits": split_case_ids,
        "label_counts": split_label_counts,
    }


def save_split_manifest(payload: Dict[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def load_split_manifest(path: str | Path) -> Dict[str, Any]:
    input_path = Path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def load_or_create_split_manifest(
    cases: Sequence[Dict[str, Any]],
    path: str | Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Any]:
    split_path = Path(path)
    if split_path.exists():
        return load_split_manifest(split_path)
    payload = build_stratified_split(
        cases,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    save_split_manifest(payload, split_path)
    return payload


def select_split_cases(
    cases: Sequence[Dict[str, Any]],
    split_payload: Dict[str, Any] | None,
    split_name: str | None,
) -> List[Dict[str, Any]]:
    if not split_payload or not split_name:
        return list(cases)
    target = str(split_name).strip().lower()
    case_ids = {
        str(case_id).strip()
        for case_id in ((split_payload.get("splits", {}) or {}).get(target, []) or [])
        if str(case_id).strip()
    }
    if not case_ids:
        return []
    return [case for case in cases if get_case_id(case) in case_ids]


def summarize_split_cases(cases: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    label_counts = Counter(str(case.get("label", "")).strip().upper() or "UNKNOWN" for case in cases)
    return {
        "num_cases": len(cases),
        "label_counts": dict(sorted(label_counts.items())),
    }


def _normalize_ratios(ratios: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in ratios.values())
    if total <= 0:
        return dict(DEFAULT_SPLIT_RATIOS)
    return {
        key: round(max(0.0, float(value)) / total, 6)
        for key, value in ratios.items()
    }


def _allocate_counts(num_cases: int, ratios: Dict[str, float]) -> Dict[str, int]:
    if num_cases <= 0:
        return {"train": 0, "val": 0, "test": 0}
    if num_cases == 1:
        return {"train": 1, "val": 0, "test": 0}
    if num_cases == 2:
        return {"train": 1, "val": 0, "test": 1}
    if num_cases == 3:
        return {"train": 1, "val": 1, "test": 1}
    if num_cases == 4:
        return {"train": 2, "val": 1, "test": 1}

    raw = {
        split_name: float(num_cases) * float(ratios.get(split_name, 0.0))
        for split_name in ["train", "val", "test"]
    }
    counts = {split_name: int(raw[split_name]) for split_name in raw}
    counts["train"] = max(1, counts["train"])
    counts["val"] = max(1, counts["val"])
    counts["test"] = max(1, counts["test"])

    while sum(counts.values()) > num_cases:
        largest = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
        if counts[largest] > 1:
            counts[largest] -= 1
        else:
            break

    remainders = sorted(
        (
            raw[split_name] - int(raw[split_name]),
            split_name,
        )
        for split_name in ["train", "val", "test"]
    )
    while sum(counts.values()) < num_cases:
        _, split_name = remainders.pop()
        counts[split_name] += 1
        if not remainders:
            remainders = sorted(
                (
                    raw[name] - int(raw[name]),
                    name,
                )
                for name in ["train", "val", "test"]
            )

    return counts
