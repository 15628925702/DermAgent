from __future__ import annotations

import json
from pathlib import Path

from evaluation.case_selection import resolve_eval_cases


def test_case_manifest_replays_same_sample():
    artifact_dir = Path("outputs/test_artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "case_selection_manifest_test.json"
    if manifest_path.exists():
        manifest_path.unlink()

    first = resolve_eval_cases(
        dataset_type="pad_ufes_20",
        dataset_root="data/pad_ufes_20",
        split_json=None,
        split_name="test",
        seed=42,
        limit=12,
        case_manifest_out=str(manifest_path),
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["num_cases"] == len(first["cases"])
    assert payload["case_ids"] == first["case_ids"]

    second = resolve_eval_cases(
        dataset_type="pad_ufes_20",
        dataset_root="data/pad_ufes_20",
        split_json=None,
        split_name="test",
        seed=999,
        limit=3,
        case_manifest_in=str(manifest_path),
    )

    assert second["case_ids"] == first["case_ids"]
