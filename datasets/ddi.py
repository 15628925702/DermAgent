from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List


VALID_AGENT_LABELS = {"MEL", "BCC", "SCC", "NEV", "ACK", "SEK"}
MALIGNANT_AGENT_LABELS = {"MEL", "BCC", "SCC"}
BENIGN_BINARY_LABEL = "BENIGN"
MALIGNANT_BINARY_LABEL = "MALIGNANT"

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_IMAGE_COLUMN_CANDIDATES = (
    "image_path",
    "image",
    "image_file",
    "image_filename",
    "file_path",
    "file",
    "filename",
    "ddi_file",
    "img_id",
)
_ID_COLUMN_CANDIDATES = ("case_id", "image_id", "lesion_id", "id")
_SPLIT_COLUMN_CANDIDATES = ("split", "partition", "subset", "set")
_DIAGNOSIS_COLUMN_CANDIDATES = (
    "diagnosis",
    "label",
    "disease",
    "disease_label",
    "condition",
    "clinical_diagnosis",
)
_BINARY_COLUMN_CANDIDATES = (
    "binary_label",
    "benign_malignant",
    "malignant_benign",
    "is_malignant",
    "malignant",
    "cancer",
)
_AGE_COLUMN_CANDIDATES = ("age", "patient_age")
_SEX_COLUMN_CANDIDATES = ("sex", "gender")
_SITE_COLUMN_CANDIDATES = ("location", "site", "anatomical_site", "body_site", "body_location")
_FITZ_COLUMN_CANDIDATES = ("fitzpatrick", "fitzpatrick_scale", "skin_type")
_SKIN_TONE_COLUMN_CANDIDATES = ("skin_tone", "skintone", "tone", "tone_group")
_HISTORY_COLUMN_CANDIDATES = ("history", "clinical_history", "notes", "symptoms")


def load_ddi_cases(
    dataset_root: str | Path = "data/ddi",
    *,
    metadata_csv: str | Path | None = None,
    images_dir: str | Path | None = None,
    limit: int | None = None,
    split_name: str | None = None,
    strict_images: bool = True,
) -> List[Dict[str, Any]]:
    root = Path(dataset_root)
    csv_path = _resolve_metadata_csv(root, metadata_csv)
    image_index = _build_image_index(root, images_dir)

    cases: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            norm_row = {str(k).strip().lower(): v for k, v in (row or {}).items() if k is not None}

            if split_name:
                split_value = _pick_first(norm_row, _SPLIT_COLUMN_CANDIDATES)
                if split_value and _norm_text(split_value) != _norm_text(split_name):
                    continue

            raw_diagnosis = _pick_first(norm_row, _DIAGNOSIS_COLUMN_CANDIDATES)
            agent_label = map_ddi_diagnosis_to_agent_label(raw_diagnosis)
            binary_label = map_ddi_binary_label(
                explicit_value=_pick_first(norm_row, _BINARY_COLUMN_CANDIDATES),
                diagnosis_value=raw_diagnosis,
                agent_label=agent_label,
            )
            if not binary_label:
                continue

            image_ref = _pick_first(norm_row, _IMAGE_COLUMN_CANDIDATES)
            image_path = _resolve_image_path(root, image_ref, image_index=image_index, images_dir=images_dir)
            if not image_path and strict_images:
                continue

            case_id = (
                _pick_first(norm_row, _ID_COLUMN_CANDIDATES)
                or image_ref
                or Path(image_path).name
                or f"ddi_case_{len(cases)}"
            )
            fitzpatrick = _pick_first(norm_row, _FITZ_COLUMN_CANDIDATES)
            skin_tone = _pick_first(norm_row, _SKIN_TONE_COLUMN_CANDIDATES)
            site = _pick_first(norm_row, _SITE_COLUMN_CANDIDATES)

            metadata = {
                "age": _safe_int(_pick_first(norm_row, _AGE_COLUMN_CANDIDATES)),
                "sex": _pick_first(norm_row, _SEX_COLUMN_CANDIDATES) or "",
                "location": site or "",
                "site": site or "",
                "clinical_history": _pick_first(norm_row, _HISTORY_COLUMN_CANDIDATES) or "",
                "fitzpatrick": fitzpatrick or "",
                "skin_tone": skin_tone or "",
                "source_dataset": "DDI",
                "raw_diagnosis": raw_diagnosis or "",
            }

            case: Dict[str, Any] = {
                "file": str(case_id).strip(),
                "image_path": image_path,
                "metadata": metadata,
                "text": "",
                "binary_label": binary_label,
                "source_dataset": "DDI",
                "raw_row": norm_row,
            }
            if agent_label in VALID_AGENT_LABELS:
                case["label"] = agent_label
                case["true_label"] = agent_label

            cases.append(case)
            if limit is not None and len(cases) >= limit:
                break

    return cases


def map_ddi_diagnosis_to_agent_label(value: Any) -> str:
    text = _norm_text(value)
    if not text:
        return ""

    replacements = {
        "seborrhoeic": "seborrheic",
        "_": " ",
        "-": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    mapping = [
        ("melanoma", "MEL"),
        ("melanocytic nevus", "NEV"),
        ("nevus", "NEV"),
        ("naevus", "NEV"),
        ("mole", "NEV"),
        ("basal cell carcinoma", "BCC"),
        ("bcc", "BCC"),
        ("squamous cell carcinoma", "SCC"),
        ("scc", "SCC"),
        ("actinic keratosis", "ACK"),
        ("solar keratosis", "ACK"),
        ("ak", "ACK"),
        ("seborrheic keratosis", "SEK"),
        ("seborrheic keratoses", "SEK"),
        ("sk", "SEK"),
    ]
    for key, label in mapping:
        if key in text:
            return label
    return ""


def map_ddi_binary_label(
    *,
    explicit_value: Any = None,
    diagnosis_value: Any = None,
    agent_label: str = "",
) -> str:
    explicit = _norm_text(explicit_value)
    if explicit:
        malignant_tokens = {
            "1",
            "true",
            "yes",
            "y",
            "malignant",
            "cancer",
            "positive",
        }
        benign_tokens = {
            "0",
            "false",
            "no",
            "n",
            "benign",
            "negative",
            "non-malignant",
            "non malignant",
        }
        if explicit in malignant_tokens:
            return MALIGNANT_BINARY_LABEL
        if explicit in benign_tokens:
            return BENIGN_BINARY_LABEL

    if agent_label in VALID_AGENT_LABELS:
        return MALIGNANT_BINARY_LABEL if agent_label in MALIGNANT_AGENT_LABELS else BENIGN_BINARY_LABEL

    diagnosis = _norm_text(diagnosis_value)
    if not diagnosis:
        return ""

    malignant_keywords = (
        "melanoma",
        "basal cell carcinoma",
        "squamous cell carcinoma",
        "malignant",
    )
    benign_keywords = (
        "nevus",
        "naevus",
        "seborrheic keratosis",
        "actinic keratosis",
        "benign",
    )
    if any(token in diagnosis for token in malignant_keywords):
        return MALIGNANT_BINARY_LABEL
    if any(token in diagnosis for token in benign_keywords):
        return BENIGN_BINARY_LABEL
    return ""


def _resolve_metadata_csv(root: Path, metadata_csv: str | Path | None) -> Path:
    if metadata_csv is not None:
        candidate = Path(metadata_csv)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"DDI metadata file not found: {candidate}")
        return candidate

    candidates = [
        root / "ddi_metadata.csv",
        root / "metadata.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    csv_files = sorted(root.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    raise FileNotFoundError(f"No DDI metadata CSV found under {root}")


def _build_image_index(root: Path, images_dir: str | Path | None) -> Dict[str, str]:
    search_roots = list(_candidate_image_roots(root, images_dir))
    search_roots.append(root)

    index: Dict[str, str] = {}
    seen_roots = []
    for search_root in search_roots:
        resolved = str(search_root.resolve()) if search_root.exists() else str(search_root)
        if resolved in seen_roots or not search_root.exists():
            continue
        seen_roots.append(resolved)
        for path in search_root.rglob("*"):
            if path.suffix.lower() not in _IMAGE_EXTENSIONS or not path.is_file():
                continue
            absolute = str(path)
            relative_to_root = str(path.relative_to(root)).replace("\\", "/").lower()
            index.setdefault(path.name.lower(), absolute)
            index.setdefault(relative_to_root, absolute)
    return index


def _resolve_image_path(
    root: Path,
    image_ref: str | None,
    *,
    image_index: Dict[str, str],
    images_dir: str | Path | None,
) -> str:
    ref = str(image_ref or "").strip()
    if not ref:
        return ""

    candidate = Path(ref)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)

    search_bases: List[Path] = []
    search_bases.extend(_candidate_image_roots(root, images_dir))
    search_bases.append(root)

    for base in search_bases:
        path = base / ref
        if path.exists():
            return str(path)

    normalized_ref = ref.replace("\\", "/").lower()
    if normalized_ref in image_index:
        return image_index[normalized_ref]

    basename = Path(ref).name.lower()
    if basename in image_index:
        return image_index[basename]

    stem = Path(ref).stem.lower()
    for extension in _IMAGE_EXTENSIONS:
        key = f"{stem}{extension}"
        if key in image_index:
            return image_index[key]

    return ""


def _pick_first(row: Dict[str, Any], candidates: Iterable[str]) -> Any:
    for key in candidates:
        value = row.get(key)
        if value is None:
            continue
        if str(value).strip() != "":
            return value
    return None


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


def _candidate_image_roots(root: Path, images_dir: str | Path | None) -> List[Path]:
    if images_dir is not None:
        image_root = Path(images_dir)
        if not image_root.is_absolute():
            image_root = root / image_root
        return [image_root]

    candidates: List[Path] = []
    for dirname in ("images", "Images"):
        candidate = root / dirname
        if candidate.exists():
            candidates.append(candidate)
    return candidates
