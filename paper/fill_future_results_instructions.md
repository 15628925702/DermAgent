# Future Result Maintenance Instructions

This file now serves as a maintenance guide for the already filled paper draft. It records what has been updated, what can still be updated in later full-training rounds, and how to revise the paper conservatively if future results differ from the current staged-validation numbers.

## 1. What Has Already Been Updated

The following result groups have already been inserted into the manuscript:

- PAD main comparison
- PAD staged ablation
- PAD full-test quality suite
- PAD significance statistics
- retained checkpoint metadata for the staged controller and learned memory bank

Updated files:

- `paper/paper_draft_zh.md`
- `paper/paper_draft_en.md`
- `paper/abstract_tbd_zh.md`
- `paper/abstract_tbd_en.md`
- `paper/results_placeholders.md`
- `paper/contributions_tbd.md` if you choose to refresh contribution wording later

Current source-of-truth files used for this fill pass:

- `outputs/comparison/comparison_report_20260324_004023.json`
- `outputs/ablation/ablation_report_20260324_002952.json`
- `outputs/quality/quality_suite_20260324_010432.json`
- `outputs/paper_stats/significance_20260324_021221.json`
- `outputs/checkpoints/learned_controller_best.json`
- `outputs/checkpoints/learned_bank_best.json`

## 2. If You Later Re-run Main PAD Comparison

Update source:
- `scripts/compare_agent_vs_qwen.py`

Primary fields to refresh:
- `agent_results.metrics.accuracy_top1`
- `agent_results.metrics.malignant_recall`
- `comparison.agent_efficiency.error_rate`
- `qwen_direct_results.metrics.accuracy_top1`
- `qwen_direct_results.metrics.malignant_recall`
- `comparison.qwen_efficiency.error_rate`
- `comparison.accuracy_improvement.absolute_improvement`
- `comparison.malignant_recall_improvement.absolute_improvement`

Paper locations to update:
- Chinese/English Table 1
- Chinese/English Abstract
- Chinese/English Section 5.1 narrative paragraph
- Chinese/English Conclusion

Only strengthen wording if:
- the same-model paired comparison remains aligned and frozen
- the new gains still move in the same direction

## 3. If You Later Re-run Significance Statistics

Update source:
- `scripts/run_significance_tests.py`

Primary fields to refresh:
- McNemar `p`
- bootstrap mean top-1 difference and 95% CI
- bootstrap mean malignant-recall difference and 95% CI

Paper locations to update:
- Chinese/English Abstract
- Chinese/English Section 5.1

Safe wording rules:
- If `p < 0.05`, you may say “statistically significant paired gain”.
- If `p >= 0.05`, replace that with “positive trend” or “directional improvement”.
- Do not call the result “robust” or “conclusive” unless multi-seed evidence is also available.

## 4. If You Later Re-run Ablation

Update source:
- `scripts/run_agent_ablation.py`

Primary fields to refresh:
- top-1 and malignant recall for all ablation stages

Paper locations to update:
- Chinese/English Table 2
- Chinese/English Section 5.2 interpretation paragraph

Interpretation rules:
- If `+ controller/scorers` remains the best stage before full retrieval, keep the current claim that the learned controller/scorer stack is the clearest incremental contributor.
- If full retrieval overtakes `+ controller/scorers`, revise the text to say retrieval becomes beneficial after stronger training.
- If stages become non-monotonic, state that explicitly instead of forcing a monotonic narrative.

## 5. If You Later Re-run Quality Suite

Update source:
- `scripts/run_agent_quality_suite.py`

Primary fields to refresh:
- `metrics.accuracy_top1`
- `metrics.accuracy_top3`
- `metrics.malignant_recall`
- `metrics.error_rate`
- `metrics.brier_top1`
- `metrics.expected_calibration_error`
- `safety.malignant_miss_count`
- `safety.benign_false_alarm_count`
- per-label `f1`
- OOD-proxy slice metrics if they materially change

Paper locations to update:
- Chinese/English Table 3
- Chinese/English Table 4
- Chinese/English Section 5.3
- Chinese/English Abstract if calibration materially improves
- Chinese/English Limitations if the class imbalance profile materially changes

Safe wording rules:
- If ECE remains high, keep “calibration remains weak”.
- If ECE becomes clearly lower while malignant recall stays high, you may say “benefits extend beyond accuracy to confidence quality”.
- If MEL/SCC/SEK remain weak, keep the limitation paragraph explicit.

## 6. If You Later Recover the Original Training Summary JSON

Currently missing from the retained bundle:
- staged held-out test top-1
- staged held-out test top-3
- staged held-out test malignant recall

If you recover a file such as:
- `outputs/train_runs/learned_components_*.json`

Then update:
- Chinese/English Table 5
- Chinese/English Section 5.4

Recommended fields:
- best validation top-1
- held-out test top-1
- held-out test top-3
- held-out test malignant recall
- final bank statistics

If those metrics remain unavailable:
- keep the current conservative Table 5 based only on retained checkpoint metadata
- do not invent summary test metrics

## 7. What Not to Add Automatically in This Revision

- Do not expand DDI-related problem discussion in the current paper text unless the writing direction changes explicitly.
- Do not convert the manuscript into a clinical-readiness claim.
- Do not claim strong external generalization from the current in-domain PAD evidence.

## 8. Conservative Wording Templates

Use these if future results weaken:

Chinese:
- “当前 staged validation 显示 learned components 已开始影响决策，但收益幅度仍有限。”
- “DermAgent 在当前设置下呈现正向趋势，统计显著性仍待更大规模评测确认。”
- “当前收益更多体现在 malignant recall 与风险敏感行为上，而不是整体 top-1 的大幅跃升。”

English:
- “The staged validation indicates that the learned components have begun to affect decision making, although the gains remain limited.”
- “DermAgent shows a favorable trend under the current setup, while stronger statistical confirmation is deferred to larger-scale evaluation.”
- “The current gains are concentrated more on malignant-case sensitivity and risk-aware behavior than on a large jump in overall top-1 accuracy.”

## 9. Final Consistency Checklist

Before submission, re-check:

1. Table 1 numbers match the abstract and conclusion.
2. Significance wording matches the latest p-value.
3. Table 2 interpretation matches the actual best ablation stage.
4. Table 3 and Table 4 match the latest quality JSON.
5. Limitation wording does not contradict the current calibration or per-class results.
6. No DDI-specific failure-analysis prose has been reintroduced unless intentionally desired.
