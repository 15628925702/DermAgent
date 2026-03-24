from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets.splits import load_or_create_split_manifest, select_split_cases
from evaluation.run_eval import load_dataset_cases, normalize_dataset_type, stratified_subsample_cases


def resolve_eval_cases(
    *,
    dataset_type: str | None,
    dataset_root: str,
    split_json: str | None,
    split_name: str | None,
    seed: int,
    limit: int | None,
    case_manifest_in: str | None = None,
    case_manifest_out: str | None = None,
) -> Dict[str, Any]:
    manifest_payload = None
    if case_manifest_in:
        manifest_payload = json.loads(Path(case_manifest_in).read_text(encoding="utf-8"))

    resolved_type = normalize_dataset_type(dataset_type, dataset_root)
    cases = load_dataset_cases(dataset_type=resolved_type, dataset_root=dataset_root, limit=None)
    resolved_split_path = split_json or (manifest_payload or {}).get("split_json")
    if split_name:
        resolved_split_path = resolved_split_path or str(Path("outputs/splits") / f"{Path(dataset_root).name}_seed{seed}.json")
        split_payload = load_or_create_split_manifest(cases, resolved_split_path, seed=seed)
        cases = select_split_cases(cases, split_payload, split_name)

    if case_manifest_in:
        requested_case_ids = [str(item).strip() for item in (manifest_payload or {}).get("case_ids", []) if str(item).strip()]
        case_index = {str(case.get("file", "")).strip(): case for case in cases}
        selected_cases: List[Dict[str, Any]] = []
        missing_case_ids: List[str] = []
        for case_id in requested_case_ids:
            case = case_index.get(case_id)
            if case is None:
                missing_case_ids.append(case_id)
                continue
            selected_cases.append(case)
        if missing_case_ids:
            raise RuntimeError(
                "Case manifest contains ids that are not available in the requested dataset/split: "
                + ", ".join(missing_case_ids[:10])
            )
        cases = selected_cases
        manifest_path = case_manifest_in
    else:
        cases = stratified_subsample_cases(cases, limit, seed=seed)
        manifest_path = None
        if case_manifest_out:
            manifest_payload = {
                "dataset_type": resolved_type,
                "dataset_root": dataset_root,
                "split_json": resolved_split_path,
                "split_name": split_name,
                "seed": seed,
                "requested_limit": limit,
                "num_cases": len(cases),
                "case_ids": [str(case.get("file", "")).strip() for case in cases],
            }
            output_path = Path(case_manifest_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest_path = str(output_path)

    return {
        "dataset_type": resolved_type,
        "dataset_root": dataset_root,
        "cases": cases,
        "resolved_split_path": resolved_split_path,
        "case_manifest_path": manifest_path,
        "case_ids": [str(case.get("file", "")).strip() for case in cases],
    }
