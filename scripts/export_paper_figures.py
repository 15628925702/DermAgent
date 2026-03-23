#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def _svg_write(path: Path, width: int, height: int, body: str) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Arial, sans-serif">{body}</svg>'
    )
    path.write_text(svg, encoding="utf-8")


def _bar_chart_svg(
    *,
    title: str,
    items: List[Tuple[str, float]],
    width: int = 900,
    height: int = 520,
    color: str = "#1f77b4",
    x_label_rotation: int = 0,
) -> str:
    left = 80
    right = 40
    top = 70
    bottom = 120 if x_label_rotation else 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max([value for _, value in items] + [1e-6])
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    parts.append(f'<text x="{width/2}" y="34" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="2"/>')
    for step in range(6):
        y = top + plot_h - plot_h * (step / 5)
        v = max_val * (step / 5)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#e3e3e3" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+5}" font-size="12" text-anchor="end">{v:.2f}</text>')
    bar_w = plot_w / max(1, len(items)) * 0.6
    gap = plot_w / max(1, len(items))
    for idx, (label, value) in enumerate(items):
        x = left + idx * gap + (gap - bar_w) / 2
        h = 0 if max_val <= 0 else plot_h * (value / max_val)
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}" rx="4"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-8:.1f}" font-size="12" text-anchor="middle">{value:.3f}</text>')
        if x_label_rotation:
            tx = x + bar_w / 2
            ty = top + plot_h + 18
            parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="12" text-anchor="end" transform="rotate({x_label_rotation} {tx:.1f},{ty:.1f})">{label}</text>')
        else:
            parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top+plot_h+22}" font-size="12" text-anchor="middle">{label}</text>')
    return "".join(parts)


def _line_chart_svg(
    *,
    title: str,
    points: List[Tuple[float, float]],
    width: int = 900,
    height: int = 520,
    color: str = "#d62728",
) -> str:
    left, right, top, bottom = 80, 40, 70, 80
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_x = max([x for x, _ in points] + [1.0])
    max_y = max([y for _, y in points] + [1.0])
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    parts.append(f'<text x="{width/2}" y="34" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top}" stroke="#cfcfcf" stroke-dasharray="6,6"/>')
    for step in range(6):
        x = left + plot_w * (step / 5)
        y = top + plot_h - plot_h * (step / 5)
        vx = max_x * (step / 5)
        vy = max_y * (step / 5)
        parts.append(f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}" stroke="#f0f0f0"/>')
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#f0f0f0"/>')
        parts.append(f'<text x="{x}" y="{top+plot_h+24}" font-size="12" text-anchor="middle">{vx:.2f}</text>')
        parts.append(f'<text x="{left-12}" y="{y+5}" font-size="12" text-anchor="end">{vy:.2f}</text>')
    path_cmd = []
    for idx, (xv, yv) in enumerate(points):
        x = left + (0 if max_x <= 0 else plot_w * (xv / max_x))
        y = top + plot_h - (0 if max_y <= 0 else plot_h * (yv / max_y))
        path_cmd.append(f'{"M" if idx == 0 else "L"} {x:.1f},{y:.1f}')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    parts.append(f'<path d="{" ".join(path_cmd)}" fill="none" stroke="{color}" stroke-width="3"/>')
    return "".join(parts)


def _heatmap_svg(
    *,
    title: str,
    matrix: Dict[str, Dict[str, int]],
    labels: List[str],
    width: int = 920,
    height: int = 760,
) -> str:
    left, right, top, bottom = 160, 40, 100, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    cell_w = plot_w / max(1, len(labels))
    cell_h = plot_h / max(1, len(labels))
    max_val = max([value for row in matrix.values() for value in row.values()] + [1])
    parts = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>']
    parts.append(f'<text x="{width/2}" y="40" font-size="24" text-anchor="middle" font-weight="bold">{title}</text>')
    for i, row_label in enumerate(labels):
        parts.append(f'<text x="{left-12}" y="{top + i*cell_h + cell_h/2 + 5:.1f}" font-size="12" text-anchor="end">{row_label}</text>')
        parts.append(f'<text x="{left + i*cell_w + cell_w/2:.1f}" y="{top-12}" font-size="12" text-anchor="middle">{row_label}</text>')
        for j, col_label in enumerate(labels):
            val = int(((matrix.get(row_label) or {}).get(col_label)) or 0)
            shade = int(255 - 180 * (val / max_val))
            color = f'rgb({shade},{shade},{255})'
            x = left + j * cell_w
            y = top + i * cell_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color}" stroke="#ddd"/>')
            parts.append(f'<text x="{x + cell_w/2:.1f}" y="{y + cell_h/2 + 5:.1f}" font-size="12" text-anchor="middle">{val}</text>')
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready SVG figures from DermAgent reports.")
    parser.add_argument("--comparison-report", default=None)
    parser.add_argument("--ablation-report", default=None)
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--external-ddi-report", default=None)
    parser.add_argument("--output-dir", default="outputs/paper_figures")
    args = parser.parse_args()

    comparison = _load_json(args.comparison_report or _latest("outputs/comparison/comparison_report_*.json"))
    ablation = _load_json(args.ablation_report or _latest("outputs/ablation/ablation_report_*.json"))
    quality = _load_json(args.quality_report or _latest("outputs/quality/quality_suite_*.json"))
    external_ddi = _load_json(args.external_ddi_report) if args.external_ddi_report else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    comp_metrics = (comparison.get("comparison") or {})
    agent_metrics = (comparison.get("agent_results") or {}).get("metrics", {}) or {}
    qwen_metrics = (comparison.get("qwen_direct_results") or {}).get("metrics", {}) or {}
    if not agent_metrics and comp_metrics:
        agent_metrics = {
            "accuracy_top1": _safe_float(((comp_metrics.get("accuracy_improvement") or {}).get("agent")), 0.0),
            "malignant_recall": _safe_float(((comp_metrics.get("malignant_recall_improvement") or {}).get("agent")), 0.0),
        }
    if not qwen_metrics and comp_metrics:
        qwen_metrics = {
            "accuracy_top1": _safe_float(((comp_metrics.get("accuracy_improvement") or {}).get("qwen_direct")), 0.0),
            "malignant_recall": _safe_float(((comp_metrics.get("malignant_recall_improvement") or {}).get("qwen_direct")), 0.0),
        }
    compare_svg = _bar_chart_svg(
        title="Agent vs Direct Qwen",
        items=[
            ("Agent Top1", float(agent_metrics.get("accuracy_top1", 0.0))),
            ("Direct Top1", float(qwen_metrics.get("accuracy_top1", 0.0))),
            ("Agent Mal Recall", float(agent_metrics.get("malignant_recall", 0.0))),
            ("Direct Mal Recall", float(qwen_metrics.get("malignant_recall", 0.0))),
        ],
        color="#2a6f97",
    )
    compare_path = out_dir / f"figure_compare_{ts}.svg"
    _svg_write(compare_path, 900, 520, compare_svg)

    ablation_items = []
    for item in ablation.get("results", []) or []:
        title = str(item.get("title", "")).strip()
        top1 = float((item.get("metrics") or {}).get("accuracy_top1", 0.0))
        if title:
            ablation_items.append((title, top1))
    ablation_path = out_dir / f"figure_ablation_{ts}.svg"
    _svg_write(
        ablation_path,
        1000,
        560,
        _bar_chart_svg(title="Ablation Top1 Accuracy", items=ablation_items, color="#e07a5f", x_label_rotation=-28),
    )

    per_label = quality.get("per_label", {}) or {}
    label_items = [(label, float((per_label.get(label) or {}).get("f1", 0.0))) for label in sorted(per_label.keys())]
    label_path = out_dir / f"figure_per_label_f1_{ts}.svg"
    _svg_write(label_path, 900, 520, _bar_chart_svg(title="Per-label F1", items=label_items, color="#3d405b"))

    bins = quality.get("calibration_bins", []) or []
    points = [(float(item.get("avg_confidence", 0.0)), float(item.get("avg_accuracy", 0.0))) for item in bins]
    calibration_path = out_dir / f"figure_calibration_{ts}.svg"
    _svg_write(calibration_path, 900, 520, _line_chart_svg(title="Calibration Curve", points=points))

    labels = sorted((quality.get("per_label") or {}).keys())
    confusion_path = out_dir / f"figure_confusion_{ts}.svg"
    _svg_write(
        confusion_path,
        920,
        760,
        _heatmap_svg(title="Confusion Matrix", matrix=quality.get("confusion_matrix", {}) or {}, labels=labels),
    )

    external_binary_path = None
    external_skin_path = None
    if external_ddi is not None:
        agent_binary = ((external_ddi.get("agent_results") or {}).get("binary_metrics")) or {}
        direct_binary = ((external_ddi.get("qwen_direct_results") or {}).get("binary_metrics")) or {}
        ddi_items = [
            ("Agent Binary Acc", float(agent_binary.get("binary_accuracy", 0.0))),
            ("Agent Mal Recall", float(agent_binary.get("malignant_recall", 0.0))),
            ("Agent Specificity", float(agent_binary.get("specificity", 0.0))),
            ("Agent Bal Acc", float(agent_binary.get("balanced_accuracy", 0.0))),
        ]
        if direct_binary:
            ddi_items.extend(
                [
                    ("Direct Binary Acc", float(direct_binary.get("binary_accuracy", 0.0))),
                    ("Direct Mal Recall", float(direct_binary.get("malignant_recall", 0.0))),
                    ("Direct Specificity", float(direct_binary.get("specificity", 0.0))),
                    ("Direct Bal Acc", float(direct_binary.get("balanced_accuracy", 0.0))),
                ]
            )
        external_binary_path = out_dir / f"figure_external_ddi_binary_{ts}.svg"
        _svg_write(
            external_binary_path,
            1100,
            560,
            _bar_chart_svg(title="External DDI Binary Metrics", items=ddi_items, color="#457b9d", x_label_rotation=-25),
        )

        tone_metrics = (((external_ddi.get("agent_results") or {}).get("subgroup_metrics") or {}).get("skin_tone")) or {}
        if tone_metrics:
            tone_items = [
                (label, float((metrics or {}).get("balanced_accuracy", 0.0)))
                for label, metrics in tone_metrics.items()
            ]
            external_skin_path = out_dir / f"figure_external_ddi_skin_tone_{ts}.svg"
            _svg_write(
                external_skin_path,
                980,
                540,
                _bar_chart_svg(title="External DDI Skin-tone Balanced Accuracy", items=tone_items, color="#6d597a", x_label_rotation=-18),
            )

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "comparison_report": args.comparison_report,
        "ablation_report": args.ablation_report,
        "quality_report": args.quality_report,
        "external_ddi_report": args.external_ddi_report,
        "figures": {
            "compare": str(compare_path),
            "ablation": str(ablation_path),
            "per_label_f1": str(label_path),
            "calibration": str(calibration_path),
            "confusion": str(confusion_path),
        },
    }
    if external_binary_path is not None:
        manifest["figures"]["external_ddi_binary"] = str(external_binary_path)
    if external_skin_path is not None:
        manifest["figures"]["external_ddi_skin_tone"] = str(external_skin_path)
    manifest_path = out_dir / f"figure_manifest_{ts}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved figure manifest to {manifest_path}")


if __name__ == "__main__":
    main()
