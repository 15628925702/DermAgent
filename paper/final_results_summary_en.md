# DermAgent Final Results Summary

## Scope

This file summarizes the locally synced final experiment artifacts from March 24, 2026 and identifies which results are safe to cite in the paper draft.

Source artifacts:

- `outputs/comparison/comparison_report_20260324_004023.json`
- `outputs/ablation/ablation_report_20260324_002952.json`
- `outputs/quality/quality_suite_20260324_010432.json`
- `outputs/external_ddi/external_ddi_report_20260324_020035.json`
- `outputs/paper_stats/significance_20260324_021221.json`
- `outputs/paper_figures/figure_manifest_20260324_021227.json`

## 1. PAD-UFES-20 Main Comparison

Frozen same-model comparison on 100 test cases:

- Agent+Qwen top-1 accuracy: `0.4300`
- Direct Qwen top-1 accuracy: `0.1818`
- Absolute top-1 gain: `+0.2482`
- Agent malignant recall: `0.9333`
- Direct Qwen malignant recall: `0.7727`
- Absolute malignant recall gain: `+0.1606`
- Agent error rate: `0.0000`
- Direct Qwen error rate: `0.0100`
- Sample alignment: `True`
- Same-model fairness check: `True`

Interpretation:

- The agent framework is substantially stronger than direct Qwen on the in-domain PAD-UFES-20 evaluation.
- The comparison is suitable as the main internal-result table.

## 2. PAD-UFES-20 Ablation

100-case staged ablation:

- Direct Qwen: top-1 `0.1818`, malignant recall `0.7500`
- Agent without retrieval: top-1 `0.3200`, malignant recall `1.0000`
- `+ metadata`: top-1 `0.3200`, malignant recall `1.0000`
- `+ specialists`: top-1 `0.3200`, malignant recall `1.0000`
- `+ controller/scorers`: top-1 `0.3600`, malignant recall `1.0000`
- Full agent `(+ retrieval)`: top-1 `0.3300`, malignant recall `1.0000`

Interpretation:

- The main in-domain gain comes from the agent framework itself relative to direct Qwen.
- Learned controller/scorers provide the clearest additional gain over the no-retrieval agent baseline.
- Retrieval is not yet a stable positive contributor in the current configuration.

## 3. PAD-UFES-20 Quality Suite

Full PAD-UFES-20 test split (`n=344`):

- Top-1 accuracy: `0.4012`
- Top-3 accuracy: `0.6541`
- Malignant recall: `0.9146`
- Error rate: `0.0000`
- Brier top-1: `0.3072`
- Expected calibration error: `0.2588`

Safety summary:

- Malignant miss count: `14`
- Benign false alarm count: `89`

OOD proxy slices:

- Rare-site proxy: `n=39`, top-1 `0.2564`, malignant recall `0.8000`
- Perception fallback: `n=4`, top-1 `0.7500`, malignant recall `0.5000`

Interpretation:

- In-domain safety recall is relatively strong.
- Calibration is still weak and should be described as an open limitation.
- Performance on rare-site slices is notably worse than overall performance.

## 4. External DDI Evaluation

DDI external test set:

- Total cases: `656`
- Cases with 6-class mapped label: `230`
- Binary benign / malignant: `485 / 171`

Agent results:

- Binary accuracy: `0.7485`
- Malignant recall: `0.0409`
- Specificity: `0.9979`
- Balanced accuracy: `0.5194`
- 6-class accuracy on mapped subset: `0.1522`
- Binary ECE: `0.0885`

Direct Qwen results:

- Binary accuracy: `0.9314`
- Malignant recall: `0.7953`
- Specificity: `0.9794`
- Balanced accuracy: `0.8874`
- 6-class accuracy on mapped subset: `0.9565`
- Binary ECE: `0.0797`

Paired external comparison:

- Binary accuracy difference (Agent - Direct): `-0.1829`
- Malignant recall difference: `-0.7544`
- Specificity difference: `+0.0185`
- McNemar p-value: `0.000000`

Observed failure mode:

- The agent predicted `BENIGN` for `648/656` DDI cases.
- The dominant predicted label was `NEV` (`632` cases).
- This is a clear benign-collapse under external domain shift.

Interpretation:

- External generalization is currently unsuccessful.
- DDI should be framed as a failure analysis / limitation section, not as supporting evidence of robust external performance.

## 5. Significance Results

Reliable significance result:

- Direct vs Agent McNemar p-value: `4.1e-05`
- Top-1 bootstrap mean difference: `0.2504`, 95% CI `[0.14, 0.36]`
- Malignant recall bootstrap mean difference: `0.1777`, 95% CI `[0.0732, 0.2941]`

Reliable external significance result:

- External DDI McNemar p-value: `0.0`
- Top-1 bootstrap mean difference: `-0.2823`, 95% CI `[-0.3171, -0.2485]`
- Malignant recall bootstrap mean difference: `-0.9110`, 95% CI `[-0.9583, -0.8571]`

Important caution:

- The current `ablation_vs_full_agent` significance block should **not** be cited.
- The reason is implementation-related: the ablation per-case records do not store `is_top1_correct` / `malignant_recalled`, so the significance script collapses those comparisons to zero difference.

## 6. Paper Positioning

Recommended paper positioning:

- Strong in-domain research prototype
- Strong PAD-UFES-20 baseline versus direct Qwen
- Useful mechanistic evidence from ablation and quality analysis
- Not yet an externally robust or clinical-grade system

Recommended claim style:

- Claim strong internal improvement.
- Claim meaningful evidence that controller/scorers help.
- Explicitly acknowledge poor external DDI generalization and benign-collapse.
- Position the system as a research prototype rather than a deployment-ready medical system.

## 7. Available Figures

Generated figure manifest:

- `outputs/paper_figures/figure_compare_20260324_021227.svg`
- `outputs/paper_figures/figure_ablation_20260324_021227.svg`
- `outputs/paper_figures/figure_per_label_f1_20260324_021227.svg`
- `outputs/paper_figures/figure_calibration_20260324_021227.svg`
- `outputs/paper_figures/figure_confusion_20260324_021227.svg`
- `outputs/paper_figures/figure_external_ddi_binary_20260324_021227.svg`
- `outputs/paper_figures/figure_external_ddi_skin_tone_20260324_021227.svg`
