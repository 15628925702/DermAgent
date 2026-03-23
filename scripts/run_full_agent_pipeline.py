#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full DermAgent overnight pipeline.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--bootstrap-passes", type=int, default=2)
    parser.add_argument("--ablation-limit", type=int, default=100)
    parser.add_argument("--compare-limit", type=int, default=100)
    parser.add_argument("--quality-limit", type=int, default=None)
    parser.add_argument("--split-json", default="outputs/splits/pad_ufes_20_seed42.json")
    parser.add_argument("--bootstrap-bank", default="outputs/checkpoints/bootstrap_bank.json")
    parser.add_argument("--learned-bank", default="outputs/checkpoints/learned_bank_best.json")
    parser.add_argument("--controller-checkpoint", default="outputs/checkpoints/learned_controller_best.json")
    parser.add_argument("--external-ddi-root", default=None)
    parser.add_argument("--external-ddi-limit", type=int, default=None)
    args = parser.parse_args()

    python = sys.executable

    _run(
        [
            python,
            "scripts/bootstrap_experience_bank.py",
            "--split-name",
            "train",
            "--passes",
            str(args.bootstrap_passes),
            "--split-json",
            args.split_json,
            "--bank-state-out",
            args.bootstrap_bank,
        ]
    )
    _run(
        [
            python,
            "scripts/train_learned_components.py",
            "--epochs",
            str(args.epochs),
            "--split-json",
            args.split_json,
            "--bank-state-in",
            args.bootstrap_bank,
            "--bank-state-out",
            args.learned_bank,
            "--controller-checkpoint-out",
            args.controller_checkpoint,
        ]
    )
    _run(
        [
            python,
            "scripts/run_agent_ablation.py",
            "--test-limit",
            str(args.ablation_limit),
            "--controller-checkpoint",
            args.controller_checkpoint,
            "--bank-state-in",
            args.learned_bank,
        ]
    )
    _run(
        [
            python,
            "scripts/compare_agent_vs_qwen.py",
            "--test-limit",
            str(args.compare_limit),
            "--controller-checkpoint",
            args.controller_checkpoint,
            "--bank-state-in",
            args.learned_bank,
        ]
    )
    quality_cmd = [
        python,
        "scripts/run_agent_quality_suite.py",
        "--split-name",
        "test",
        "--split-json",
        args.split_json,
        "--controller-checkpoint",
        args.controller_checkpoint,
        "--bank-state-in",
        args.learned_bank,
    ]
    if args.quality_limit is not None:
        quality_cmd.extend(["--limit", str(args.quality_limit)])
    _run(quality_cmd)

    if args.external_ddi_root:
        ddi_cmd = [
            python,
            "scripts/run_external_ddi_eval.py",
            "--dataset-root",
            args.external_ddi_root,
            "--controller-checkpoint",
            args.controller_checkpoint,
            "--bank-state-in",
            args.learned_bank,
        ]
        if args.external_ddi_limit is not None:
            ddi_cmd.extend(["--limit", str(args.external_ddi_limit)])
        _run(ddi_cmd)


if __name__ == "__main__":
    main()
