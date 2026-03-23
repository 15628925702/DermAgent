# Finalize Results Checklist

Use this checklist after the final long training run.

## Internal PAD-UFES-20
- Run final bootstrap and learned-component training.
- Run frozen same-model compare with the final checkpoint.
- Run full ablation with the final checkpoint.
- Run quality suite on the test split.
- Export significance tests and paper figures.

## External DDI
- Run [scripts/run_external_ddi_eval.py](/g:/0-newResearch/derm_agent/scripts/run_external_ddi_eval.py) with the final checkpoint and learned bank.
- Export paired significance if direct-Qwen predictions are included.
- Export DDI figures from the paper evidence suite.

## Replace in Manuscript
- Abstract: update the final reported top-1 and malignant-recall gains.
- Section 7.1: replace the controlled compare table if final numbers differ.
- Section 7.2: replace the ablation table with the final trained-checkpoint ablation.
- Section 7.3: insert final DDI binary metrics and subgroup findings.
- Add significance values where appropriate.

## Suggested Final Command Order
```bash
python scripts/run_full_agent_pipeline.py --epochs 2 --bootstrap-passes 2 --external-ddi-root data/ddi

python scripts/run_paper_evidence_suite.py ^
  --comparison-report outputs/comparison/<final_comparison>.json ^
  --ablation-report outputs/ablation/<final_ablation>.json ^
  --quality-report outputs/quality/<final_quality>.json ^
  --external-ddi-report outputs/external_ddi/<final_ddi>.json
```
