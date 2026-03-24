#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_report(path: str | None) -> Dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    comparison_dir = Path("outputs/comparison")
    candidates = sorted(comparison_dir.glob("comparison_report_*.json"))
    if not candidates:
        raise RuntimeError("No comparison report found under outputs/comparison. Pass --report explicitly.")
    return json.loads(candidates[-1].read_text(encoding="utf-8"))


def _top_counts(items: Dict[str, Any], limit: int) -> str:
    pairs = list((items or {}).items())[:limit]
    if not pairs:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize module participation from a comparison report.")
    parser.add_argument("--report", default=None, help="Path to comparison_report_*.json. Defaults to latest report.")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    report = _load_report(args.report)
    module = (((report.get("comparison") or {}).get("module_participation") or {}).get("agent") or {})
    if not module:
        raise RuntimeError("The report does not contain agent module participation. Re-run compare_agent_vs_qwen.py.")

    print("Module Participation Summary")
    print(f"report_timestamp: {report.get('timestamp', '')}")
    print(f"test_count: {report.get('test_count', 0)}")
    print(f"cases_with_memory_recommendations: {module.get('cases_with_memory_recommendations', 0)}")
    print(f"cases_with_rule_recommendations: {module.get('cases_with_rule_recommendations', 0)}")
    print(f"cases_with_applied_rules: {module.get('cases_with_applied_rules', 0)}")
    print(f"cases_with_hybrid_retention: {module.get('cases_with_hybrid_retention', 0)}")
    print(f"cases_with_confusion_support: {module.get('cases_with_confusion_support', 0)}")
    print(f"cases_with_supports_top1: {module.get('cases_with_supports_top1', 0)}")
    print(f"selected_skill_counts: {_top_counts(module.get('selected_skill_counts', {}), args.top_k)}")
    print(f"memory_recommended_skill_counts: {_top_counts(module.get('memory_recommended_skill_counts', {}), args.top_k)}")
    print(f"rule_recommended_skill_counts: {_top_counts(module.get('rule_recommended_skill_counts', {}), args.top_k)}")
    print(f"hybrid_retained_skill_counts: {_top_counts(module.get('hybrid_retained_skill_counts', {}), args.top_k)}")


if __name__ == "__main__":
    main()
