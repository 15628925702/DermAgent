# Run Modes

This project now treats run initialization as an explicit choice instead of an implicit side effect.

## Modes

- `INIT_MODE=clean`
  - start from empty controller state and empty memory bank
  - use this for new experiments by default
- `INIT_MODE=seed`
  - warm-start from tracked seed checkpoints under `outputs/checkpoints/seed/`
  - use this only when you intentionally want the tracked seed baseline
- `INIT_MODE=run_best`
  - warm-start from `best_controller.json` and `best_bank.json` inside `BASE_RUN_DIR`
  - use this when continuing from a promoted baseline
- `INIT_MODE=resume`
  - continue from `latest_controller.json` and `latest_bank.json` already inside `SAVE_DIR`
  - use this only for interrupted or unfinished runs

## Safety Rules

- `clean`, `seed`, and `run_best` should use a fresh `SAVE_DIR`
- `resume` should reuse the same `SAVE_DIR`
- if a populated run directory must be reused intentionally, pass `ALLOW_DIR_REUSE=1`

## Recorded Metadata

Every training run writes `run_manifest.json` in its run directory.

That manifest records:

- run name
- save directory
- init mode
- input checkpoint paths
- base run directory
- config values
- command argv
- artifact paths

So after training, you can inspect:

- `outputs/train_runs/<run_name>/run_manifest.json`
- `outputs/train_runs/<run_name>/train_summary.json`

before comparing or promoting a run.

## Naming

Recommended prefixes:

- clean runs: `clean_*`, `mainline_*`
- seeded runs: `seed_*`
- baseline warm-start runs: `frombest_*`, `ablation_*`
- resumed runs: keep the original run name

Avoid reusing one run name for different init modes.

## Server Commands

Minimal sync and test:

```bash
git pull
python -m pytest tests/test_smoke.py -q
```

Clean smoke run:

```bash
INIT_MODE=clean EPOCHS=1 SAVE_DIR=outputs/train_runs/clean_smoke_try1 bash scripts/linux_train_8h.sh
```

Seeded smoke run:

```bash
INIT_MODE=seed EPOCHS=1 SAVE_DIR=outputs/train_runs/seed_smoke_try1 bash scripts/linux_train_8h.sh
```

Warm-start from promoted baseline:

```bash
INIT_MODE=run_best BASE_RUN_DIR=outputs/train_runs/mainline_current EPOCHS=1 SAVE_DIR=outputs/train_runs/frombest_smoke_try1 bash scripts/linux_train_8h.sh
```

Resume an interrupted run:

```bash
INIT_MODE=resume EPOCHS=1 SAVE_DIR=outputs/train_runs/mainline_overnight_try1 bash scripts/linux_train_8h.sh
```
