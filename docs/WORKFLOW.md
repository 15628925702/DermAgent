# Mainline Workflow

This project should use a small-step workflow:

1. Change one small module.
2. Run a smoke train.
3. Review the new run against the baseline.
4. Only if the run is approved, start a longer training run.

## Rules

- Do not go straight from code change to overnight training.
- Do not compare a new run by memory. Always run the review script.
- A rejected run should not become the new baseline.
- Keep the current promoted baseline in `outputs/train_runs/mainline_current`.

## Smoke

Use this after a small code change:

```bash
STAGE=smoke RUN_NAME=mainline_smoke_try1 bash scripts/linux_mainline_workflow.sh
```

This workflow now defaults to a clean initialization. It will not silently warm-start from
seed checkpoints or older runs unless you explicitly pass an init mode.

See [RUN_MODES.md](/g:/0-newResearch/derm_agent/docs/RUN_MODES.md) for the recommended run semantics and server commands.

Default smoke settings:

- limit: 80
- epochs: 1
- review split: full test split

## Overnight

Use this only after smoke is acceptable:

```bash
STAGE=overnight RUN_NAME=mainline_overnight_try1 FULL_EPOCHS=7 bash scripts/linux_mainline_workflow.sh
```

If you want a warm start, pass it explicitly:

```bash
INIT_MODE=seed STAGE=overnight RUN_NAME=mainline_seed_try1 FULL_EPOCHS=7 bash scripts/linux_mainline_workflow.sh
```

```bash
INIT_MODE=run_best BASE_RUN_DIR=outputs/train_runs/mainline_current STAGE=overnight RUN_NAME=mainline_from_best_try1 FULL_EPOCHS=7 bash scripts/linux_mainline_workflow.sh
```

```bash
INIT_MODE=resume RUN_NAME=mainline_resume_try1 STAGE=overnight FULL_EPOCHS=7 bash scripts/linux_mainline_workflow.sh
```

## Review Only

Use this to re-check an existing run:

```bash
STAGE=review CANDIDATE_RUN_DIR=outputs/train_runs/mainline_overnight_try1 bash scripts/linux_mainline_workflow.sh
```

## Delete Rejected Runs

If you want rejected runs to be removed automatically:

```bash
DELETE_REJECTED=1 STAGE=review CANDIDATE_RUN_DIR=outputs/train_runs/bad_run bash scripts/linux_mainline_workflow.sh
```

## Promote Good Runs

Approved runs are copied into:

`outputs/train_runs/mainline_current`

This is handled by the review workflow automatically.

## Useful Thresholds

Default review requires no metric drop:

- top1 delta >= 0
- top3 delta >= 0
- malignant recall delta >= 0
- confusion accuracy delta >= 0

You can make review stricter:

```bash
MIN_TOP1_GAIN=0.01 MIN_CONFUSION_GAIN=0.01 STAGE=review CANDIDATE_RUN_DIR=outputs/train_runs/some_run bash scripts/linux_mainline_workflow.sh
```
