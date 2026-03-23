#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready statistics and figures from existing DermAgent reports.")
    parser.add_argument("--comparison-report", default=None)
    parser.add_argument("--ablation-report", default=None)
    parser.add_argument("--quality-report", default=None)
    parser.add_argument("--external-ddi-report", default=None)
    args = parser.parse_args()

    python = sys.executable
    sig_cmd = [python, "scripts/run_significance_tests.py"]
    fig_cmd = [python, "scripts/export_paper_figures.py"]
    for cmd in [sig_cmd, fig_cmd]:
        if args.comparison_report:
            cmd.extend(["--comparison-report", args.comparison_report])
        if args.ablation_report:
            cmd.extend(["--ablation-report", args.ablation_report])
        if args.quality_report:
            cmd.extend(["--quality-report", args.quality_report])
        if args.external_ddi_report:
            cmd.extend(["--external-ddi-report", args.external_ddi_report])
    _run(sig_cmd)
    _run(fig_cmd)


if __name__ == "__main__":
    main()
