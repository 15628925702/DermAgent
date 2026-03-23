#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


MALIGNANT_LABELS = {"MEL", "BCC", "SCC"}


def _latest(pattern: str) -> str:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return paths[-1]


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _exact_mcnemar_pvalue(b: int, c: int) -> float:
    n = b + c
    if n <= 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _paired_bootstrap_diff(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    *,
    metric_name: str,
    malignant_recall: bool = False,
    num_samples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    if len(rows_a) != len(rows_b):
        raise ValueError("Paired bootstrap requires equal-length row lists.")
    rng = random.Random(seed)
    diffs: List[float] = []
    n = len(rows_a)
    indices = list(range(n))
    for _ in range(num_samples):
        sample = [rng.choice(indices) for _ in indices]
        if malignant_recall:
            def _rec(rows: List[Dict[str, Any]]) -> float:
                malignant = [rows[i] for i in sample if rows[i].get("true_label") in MALIGNANT_LABELS]
                if not malignant:
                    return 0.0
                hits = sum(1 for item in malignant if item.get(metric_name))
                return hits / len(malignant)
            diffs.append(_rec(rows_a) - _rec(rows_b))
        else:
            score_a = sum(1.0 if rows_a[i].get(metric_name) else 0.0 for i in sample) / n
            score_b = sum(1.0 if rows_b[i].get(metric_name) else 0.0 for i in sample) / n
            diffs.append(score_a - score_b)
    diffs.sort()
    low = diffs[int(0.025 * len(diffs))]
    high = diffs[int(0.975 * len(diffs))]
    mean = sum(diffs) / max(len(diffs), 1)
    return {
        "mean_diff": round(mean, 4),
        "ci95_low": round(low, 4),
        "ci95_high": round(high, 4),
        "num_samples": num_samples,
    }


def _load_comparison(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    agent = ((payload.get("agent_results") or {}).get("per_case")) or []
    qwen = ((payload.get("qwen_direct_results") or {}).get("per_case")) or []
    by_case_qwen = {str(item.get("case_id", "")).strip(): item for item in qwen}
    paired: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for item in agent:
        cid = str(item.get("case_id", "")).strip()
        if cid in by_case_qwen:
            paired.append((item, by_case_qwen[cid]))
    return {
        "payload": payload,
        "agent_rows": [x[0] for x in paired],
        "qwen_rows": [x[1] for x in paired],
    }


def _load_ablation(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload.get("results", []) or []
    mapped = {}
    for item in results:
        title = str(item.get("title", "")).strip()
        per_case = item.get("per_case", []) or []
        mapped[title] = {
            "metrics": item.get("metrics", {}) or {},
            "per_case": per_case,
            "by_case": {str(x.get("case_id", "")).strip(): x for x in per_case},
        }
    return {"payload": payload, "results": mapped}


def _comparison_significance(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    *,
    name_a: str,
    name_b: str,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    b = 0
    c = 0
    for a, b_row in zip(rows_a, rows_b):
        a_correct = bool(a.get("is_top1_correct") or a.get("correct"))
        b_correct = bool(b_row.get("is_top1_correct") or b_row.get("correct"))
        if a_correct and not b_correct:
            b += 1
        elif b_correct and not a_correct:
            c += 1

    return {
        "pair": [name_a, name_b],
        "paired_cases": len(rows_a),
        "mcnemar": {
            "agent_only_correct": b,
            "baseline_only_correct": c,
            "p_value": round(_exact_mcnemar_pvalue(b, c), 6),
        },
        "top1_bootstrap": _paired_bootstrap_diff(rows_a, rows_b, metric_name="is_top1_correct", num_samples=bootstrap_samples, seed=seed),
        "malignant_recall_bootstrap": _paired_bootstrap_diff(rows_a, rows_b, metric_name="malignant_recalled", malignant_recall=True, num_samples=bootstrap_samples, seed=seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper-ready significance tests for DermAgent outputs.")
    parser.add_argument("--comparison-report", default=None)
    parser.add_argument("--ablation-report", default=None)
    parser.add_argument("--external-ddi-report", default=None)
    parser.add_argument("--output-dir", default="outputs/paper_stats")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    comparison_path = args.comparison_report or _latest("outputs/comparison/comparison_report_*.json")
    ablation_path = args.ablation_report or _latest("outputs/ablation/ablation_report_*.json")
    comparison = _load_comparison(comparison_path)
    ablation = _load_ablation(ablation_path)

    summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "comparison_report": comparison_path,
        "ablation_report": ablation_path,
        "external_ddi_report": args.external_ddi_report,
        "bootstrap_samples": args.bootstrap_samples,
        "direct_vs_agent": _comparison_significance(
            comparison["agent_rows"],
            comparison["qwen_rows"],
            name_a="Agent",
            name_b="Direct Qwen",
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        ),
        "ablation_vs_full_agent": {},
    }

    if args.external_ddi_report:
        external_payload = _load_json(args.external_ddi_report)
        agent_rows = ((external_payload.get("agent_results") or {}).get("per_case")) or []
        direct_rows = ((external_payload.get("qwen_direct_results") or {}).get("per_case")) or []
        if agent_rows and direct_rows:
            summary["external_ddi"] = {
                "paired_binary": _comparison_significance(
                    agent_rows,
                    direct_rows,
                    name_a="External DDI Agent",
                    name_b="External DDI Direct Qwen",
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            }
        else:
            summary["external_ddi"] = {
                "paired_binary": None,
                "note": "External DDI report did not contain paired direct-Qwen predictions."
            }

    full = ablation["results"].get("Full agent (+ retrieval)")
    if full is not None:
        full_by_case = full["by_case"]
        for title, item in ablation["results"].items():
            if title == "Full agent (+ retrieval)":
                continue
            paired_full: List[Dict[str, Any]] = []
            paired_other: List[Dict[str, Any]] = []
            for cid, full_row in full_by_case.items():
                other_row = item["by_case"].get(cid)
                if other_row is None:
                    continue
                paired_full.append(full_row)
                paired_other.append(other_row)
            if not paired_full:
                continue
            summary["ablation_vs_full_agent"][title] = _comparison_significance(
                paired_full,
                paired_other,
                name_a="Full agent (+ retrieval)",
                name_b=title,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"significance_{ts}.json"
    txt_path = output_dir / f"significance_{ts}.txt"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "DermAgent Significance Summary",
        f"comparison_report: {comparison_path}",
        f"ablation_report: {ablation_path}",
        "",
        "Direct vs Agent",
    ]
    main_result = summary["direct_vs_agent"]
    lines.extend(
        [
            f"  McNemar p-value: {main_result['mcnemar']['p_value']}",
            f"  top1 diff mean/95CI: {main_result['top1_bootstrap']['mean_diff']} [{main_result['top1_bootstrap']['ci95_low']}, {main_result['top1_bootstrap']['ci95_high']}]",
            f"  malignant recall diff mean/95CI: {main_result['malignant_recall_bootstrap']['mean_diff']} [{main_result['malignant_recall_bootstrap']['ci95_low']}, {main_result['malignant_recall_bootstrap']['ci95_high']}]",
            "",
        ]
    )
    if summary.get("external_ddi", {}).get("paired_binary"):
        external = summary["external_ddi"]["paired_binary"]
        lines.extend(
            [
                "External DDI",
                f"  McNemar p-value: {external['mcnemar']['p_value']}",
                f"  top1 diff mean/95CI: {external['top1_bootstrap']['mean_diff']} [{external['top1_bootstrap']['ci95_low']}, {external['top1_bootstrap']['ci95_high']}]",
                f"  malignant recall diff mean/95CI: {external['malignant_recall_bootstrap']['mean_diff']} [{external['malignant_recall_bootstrap']['ci95_low']}, {external['malignant_recall_bootstrap']['ci95_high']}]",
                "",
            ]
        )
    lines.extend(
        [
            "Ablation vs Full Agent",
        ]
    )
    for title, item in summary["ablation_vs_full_agent"].items():
        lines.append(
            f"  {title}: p={item['mcnemar']['p_value']} top1_diff={item['top1_bootstrap']['mean_diff']} "
            f"[{item['top1_bootstrap']['ci95_low']}, {item['top1_bootstrap']['ci95_high']}]"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved significance JSON to {json_path}")
    print(f"Saved significance summary to {txt_path}")


if __name__ == "__main__":
    main()
