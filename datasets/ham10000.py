from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


VALID_AGENT_LABELS = {"MEL", "BCC", "SCC", "NEV", "ACK", "SEK"}
MALIGNANT_AGENT_LABELS = {"MEL", "BCC", "SCC"}
SUPPORTED_HAM_LABELS = {"mel", "nv", "bcc", "akiec"}
DROPPED_HAM_LABELS = {"bkl", "df", "vasc"}
HAM10000_LABEL_TO_AGENT = {
    "mel": "MEL",
    "nv": "NEV",
    "bcc": "BCC",
    # HAM10000 "akiec" merges actinic keratosis and intraepithelial carcinoma.
    # We map it conservatively into ACK instead of SCC to avoid overstating invasive SCC coverage.
    "akiec": "ACK",
}
HAM10000_LABEL_DESCRIPTIONS = {
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "bcc": "basal cell carcinoma",
    "akiec": "actinic keratoses / intraepithelial carcinoma",
    "bkl": "benign keratosis-like lesions",
    "df": "dermatofibroma",
    "vasc": "vascular lesions",
}

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_ham10000_cases(
    dataset_root: str | Path = "data/ham10000",
    *,
    metadata_csv: str | Path | None = None,
    limit: int | None = None,
    strict_images: bool = True,
) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    metadata_path = _resolve_metadata_csv(root, metadata_csv)
    image_index = _build_image_index(root)

    cases: List[Dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_label = _norm_text(row.get("dx"))
            agent_label = map_ham10000_label(raw_label)
            if not agent_label:
                continue

            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                continue

            image_path = _resolve_image_path(root, image_id, image_index=image_index)
            if not image_path and strict_images:
                continue

            metadata = {
                "age": _safe_int(row.get("age")),
                "sex": str(row.get("sex", "") or ""),
                "location": str(row.get("localization", "") or ""),
                "site": str(row.get("localization", "") or ""),
                "clinical_history": "",
                "lesion_id": str(row.get("lesion_id", "") or ""),
                "image_id": image_id,
                "dx_type": str(row.get("dx_type", "") or ""),
                "source_dataset": "HAM10000",
                "raw_diagnosis": raw_label,
            }
            cases.append(
                {
                    "file": image_id,
                    "image_path": image_path,
                    "metadata": metadata,
                    "text": "",
                    "label": agent_label,
                    "true_label": agent_label,
                    "binary_label": "MALIGNANT" if agent_label in MALIGNANT_AGENT_LABELS else "BENIGN",
                    "source_dataset": "HAM10000",
                    "raw_label": raw_label,
                }
            )

            if limit is not None and len(cases) >= limit:
                break

    return cases


def map_ham10000_label(value: Any) -> str:
    raw_label = _norm_text(value)
    return HAM10000_LABEL_TO_AGENT.get(raw_label, "")


def summarize_ham10000_metadata(
    dataset_root: str | Path = "data/ham10000",
    *,
    metadata_csv: str | Path | None = None,
) -> Dict[str, Any]:
    root = Path(dataset_root)
    metadata_path = _resolve_metadata_csv(root, metadata_csv)

    raw_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_label = _norm_text(row.get("dx"))
            raw_counts[raw_label or "unknown"] += 1
            agent_label = map_ham10000_label(raw_label)
            if agent_label:
                mapped_counts[agent_label] += 1
            else:
                dropped_counts[raw_label or "unknown"] += 1

    return {
        "dataset_root": str(root),
        "metadata_csv": str(metadata_path),
        "raw_label_counts": dict(sorted(raw_counts.items())),
        "mapped_agent_label_counts": dict(sorted(mapped_counts.items())),
        "dropped_raw_label_counts": dict(sorted(dropped_counts.items())),
    }


def iter_ham10000_mapping_rows() -> Iterable[Dict[str, str]]:
    for raw_label, description in sorted(HAM10000_LABEL_DESCRIPTIONS.items()):
        yield {
            "raw_label": raw_label,
            "description": description,
            "mapped_label": HAM10000_LABEL_TO_AGENT.get(raw_label, ""),
            "action": "keep" if raw_label in SUPPORTED_HAM_LABELS else "drop",
        }


def _resolve_metadata_csv(root: Path, metadata_csv: str | Path | None) -> Path:
    if metadata_csv is not None:
        candidate = Path(metadata_csv)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"HAM10000 metadata file not found: {candidate}")
        return candidate

    candidates = [
        root / "HAM10000_metadata.csv",
        root / "ham10000_metadata.csv",
        root / "metadata.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"No HAM10000 metadata CSV found under {root}")


def _build_image_index(root: Path) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for search_root in _candidate_image_roots(root):
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            absolute = str(path)
            index.setdefault(path.name.lower(), absolute)
            index.setdefault(path.stem.lower(), absolute)
            index.setdefault(str(path.relative_to(root)).replace("\\", "/").lower(), absolute)
    return index


def _resolve_image_path(root: Path, image_id: str, *, image_index: Dict[str, str]) -> str:
    normalized = str(image_id).strip()
    if not normalized:
        return ""

    direct_candidates = [
        root / normalized,
        root / f"{normalized}.jpg",
        root / f"{normalized}.jpeg",
        root / f"{normalized}.png",
    ]
    for candidate_root in _candidate_image_roots(root):
        direct_candidates.extend(
            [
                candidate_root / normalized,
                candidate_root / f"{normalized}.jpg",
                candidate_root / f"{normalized}.jpeg",
                candidate_root / f"{normalized}.png",
            ]
        )

    for candidate in direct_candidates:
        if candidate.exists():
            return str(candidate)

    lookup_keys = [
        normalized.lower(),
        Path(normalized).name.lower(),
        Path(normalized).stem.lower(),
    ]
    for key in lookup_keys:
        if key in image_index:
            return image_index[key]
    return ""


def _candidate_image_roots(root: Path) -> List[Path]:
    candidates = [
        root / "HAM10000_images_part_1",
        root / "HAM10000_images_part_2",
        root / "images",
        root,
    ]
    seen: set[str] = set()
    deduped: List[Path] = []
    for candidate in candidates:
        text = str(candidate)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(candidate)
    return deduped


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
