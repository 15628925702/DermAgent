# Output Layout

This project keeps generated artifacts under `outputs/` with a small number of top-level buckets.

## Active Layout

- `outputs/checkpoints/seed/`
  - tracked seed artifacts used as default warm-start inputs
  - currently includes `controller_v2.json` and `experience_bank.json`
- `outputs/train_runs/`
  - active or recent experiment directories you still compare against
  - each run should include `run_manifest.json` and `train_summary.json`
- `outputs/splits/`
  - dataset split manifests
- `outputs/logs/`
  - local runtime logs such as vLLM or manual training logs
- `outputs/test_artifacts/`
  - pytest temp files and cache

## Archive Layout

- `outputs/archive/train_runs/`
  - older or superseded experiment directories
- `outputs/archive/checkpoints/`
  - backup checkpoint directories or retired standalone backups

## Archiving Rule

Keep under `outputs/train_runs/` only:

- current baseline runs
- the latest experiments you are actively comparing
- runs still referenced in notes or current scripts

Move to `outputs/archive/` when a run is:

- clearly older than the current baseline
- superseded by a renamed or newer version
- a one-off smoke/debug run you no longer compare against

## Current Convention

Recommended naming:

- long runs: `final_*`, `mainline_*`
- clean runs: `clean_*`
- seeded runs: `seed_*`
- baseline warm-start runs: `frombest_*`
- focused experiments: `writegate_*`, `ablation_*`
- quick local checks: `quick_*`, then archive quickly if no longer needed

Avoid generic names like just `smoke/` for long-term retention.
