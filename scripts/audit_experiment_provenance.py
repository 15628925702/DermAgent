#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(args: List[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_last_touch(path: Path) -> Dict[str, Any]:
    rel = path.relative_to(PROJECT_ROOT).as_posix()
    line = _run_git(["log", "-1", "--format=%H|%cI|%s", "--", rel])
    if not line:
        return {}
    commit, commit_time, subject = (line.split("|", 2) + ["", "", ""])[:3]
    return {
        "commit": commit,
        "commit_time": commit_time,
        "subject": subject,
    }


def _safe_rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return str(path)


def _file_info(path_str: str | None) -> Dict[str, Any]:
    if not path_str:
        return {}
    path = (PROJECT_ROOT / path_str).resolve() if not Path(path_str).is_absolute() else Path(path_str).resolve()
    info: Dict[str, Any] = {
        "requested_path": path_str,
        "resolved_path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return info
    stat = path.stat()
    info.update(
        {
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "sha256": _sha256(path),
            "git_last_touch": _git_last_touch(path),
            "project_relative_path": _safe_rel(path),
        }
    )
    return info


def _extract_case_ids(rows: Iterable[Dict[str, Any]], limit: int = 10) -> List[str]:
    case_ids: List[str] = []
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            case_ids.append(case_id)
        if len(case_ids) >= limit:
            break
    return case_ids


def _summarize_report(report_path: Path) -> Dict[str, Any]:
    report = _read_json(report_path)
    comparison = report.get("comparison", {}) or {}
    agent = report.get("agent_results", {}) or {}
    direct = report.get("qwen_direct_results", {}) or {}
    agent_model = (comparison.get("model_info", {}) or {}).get("agent", {}) or {}
    sample_alignment = comparison.get("sample_alignment", {}) or {}
    return {
        "file": _file_info(str(report_path)),
        "report_timestamp": report.get("timestamp"),
        "dataset_type": report.get("dataset_type"),
        "dataset_root": report.get("dataset_root"),
        "split_json": report.get("split_json"),
        "split_name": report.get("split_name"),
        "seed": report.get("seed"),
        "test_count": report.get("test_count"),
        "requested_limit": report.get("requested_limit"),
        "agent_metrics": agent.get("metrics"),
        "direct_metrics": direct.get("metrics"),
        "comparison_conclusion": comparison.get("conclusion"),
        "agent_model_info": agent_model,
        "agent_first_cases": _extract_case_ids(agent.get("per_case", []) or []),
        "direct_first_cases": _extract_case_ids(direct.get("per_case", []) or []),
        "sample_alignment": sample_alignment,
    }


def _compare_reports(left_path: Path, right_path: Path) -> Dict[str, Any]:
    left = _read_json(left_path)
    right = _read_json(right_path)
    left_agent = (left.get("agent_results", {}) or {}).get("metrics", {}) or {}
    right_agent = (right.get("agent_results", {}) or {}).get("metrics", {}) or {}
    left_direct = (left.get("qwen_direct_results", {}) or {}).get("metrics", {}) or {}
    right_direct = (right.get("qwen_direct_results", {}) or {}).get("metrics", {}) or {}
    left_cases = _extract_case_ids((left.get("agent_results", {}) or {}).get("per_case", []) or [], limit=1000)
    right_cases = _extract_case_ids((right.get("agent_results", {}) or {}).get("per_case", []) or [], limit=1000)
    return {
        "left_report": _safe_rel(left_path),
        "right_report": _safe_rel(right_path),
        "same_seed": left.get("seed") == right.get("seed"),
        "same_split_name": left.get("split_name") == right.get("split_name"),
        "same_dataset_root": left.get("dataset_root") == right.get("dataset_root"),
        "same_requested_limit": left.get("requested_limit") == right.get("requested_limit"),
        "same_case_order": left_cases == right_cases,
        "left_first_cases": left_cases[:10],
        "right_first_cases": right_cases[:10],
        "agent_metric_delta": {
            key: round(float(right_agent.get(key, 0.0)) - float(left_agent.get(key, 0.0)), 6)
            for key in sorted(set(left_agent) | set(right_agent))
        },
        "direct_metric_delta": {
            key: round(float(right_direct.get(key, 0.0)) - float(left_direct.get(key, 0.0)), 6)
            for key in sorted(set(left_direct) | set(right_direct))
        },
    }


def _default_split_path(dataset_root: str | None, seed: Any) -> Path | None:
    if not dataset_root or seed is None:
        return None
    return PROJECT_ROOT / "outputs" / "splits" / f"{Path(str(dataset_root)).name}_seed{seed}.json"


def _split_info(split_json: str | None, dataset_root: str | None, seed: Any) -> Dict[str, Any]:
    candidate = Path(split_json) if split_json else _default_split_path(dataset_root, seed)
    if candidate is None:
        return {}
    path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    info = _file_info(str(path))
    if not path.exists():
        return info
    payload = _read_json(path)
    split_rows = payload.get("splits", payload)
    summary: Dict[str, Any] = {}
    for key in ["train", "val", "test"]:
        rows = split_rows.get(key, []) if isinstance(split_rows, dict) else []
        summary[key] = {
            "count": len(rows) if isinstance(rows, list) else None,
            "first_case_ids": [str(item) for item in rows[:10]] if isinstance(rows, list) else [],
        }
    info["split_summary"] = summary
    return info


def _maybe_checkpoint_info(path_str: str | None) -> Dict[str, Any]:
    info = _file_info(path_str)
    resolved = info.get("resolved_path")
    if not resolved or not info.get("exists"):
        return info
    path = Path(resolved)
    try:
        payload = _read_json(path)
    except Exception as exc:
        info["json_error"] = str(exc)
        return info
    if isinstance(payload, dict):
        info["top_level_keys"] = list(payload.keys())[:20]
        if "metadata" in payload:
            info["metadata"] = payload.get("metadata")
        if "stats" in payload:
            info["stats"] = payload.get("stats")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit experiment provenance for reports, checkpoints, and split manifests.")
    parser.add_argument("--report", required=True, help="Primary comparison report JSON.")
    parser.add_argument("--compare-report", default=None, help="Optional second report to compare against.")
    parser.add_argument("--controller", default=None, help="Optional controller checkpoint override.")
    parser.add_argument("--bank", default=None, help="Optional bank checkpoint override.")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = (PROJECT_ROOT / report_path).resolve()
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")

    primary = _summarize_report(report_path)
    agent_model_info = primary.get("agent_model_info", {}) or {}
    controller_path = args.controller or agent_model_info.get("checkpoint_path")
    bank_path = args.bank or "outputs/checkpoints/learned_bank_best.json"

    result: Dict[str, Any] = {
        "primary_report": primary,
        "controller_checkpoint": _maybe_checkpoint_info(controller_path),
        "bank_checkpoint": _maybe_checkpoint_info(bank_path),
        "split_manifest": _split_info(
            primary.get("split_json"),
            primary.get("dataset_root"),
            primary.get("seed"),
        ),
        "git_status_short": _run_git(["status", "--short"]),
    }

    if args.compare_report:
        compare_path = Path(args.compare_report)
        if not compare_path.is_absolute():
            compare_path = (PROJECT_ROOT / compare_path).resolve()
        if not compare_path.exists():
            raise FileNotFoundError(f"Missing compare report: {compare_path}")
        result["report_comparison"] = _compare_reports(report_path, compare_path)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
