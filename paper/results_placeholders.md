# Results Fill Status

This file records which placeholder groups from the earlier draft have already been resolved, where the values were inserted, and which originally planned fields are still unavailable in the retained artifact bundle.

## 1. Main Comparison Placeholders

Source file:
- `outputs/comparison/comparison_report_20260324_004023.json`

Resolved values:

| Placeholder | Value | Inserted into |
| --- | --- | --- |
| `[TBD_COMPARE_QWEN_TOP1]` | `0.1818` | Chinese/English Table 1 |
| `[TBD_COMPARE_QWEN_MAL_RECALL]` | `0.7727` | Chinese/English Table 1 |
| `[TBD_COMPARE_QWEN_ERROR_RATE]` | `0.0100` | Chinese/English Table 1 |
| `[TBD_COMPARE_AGENT_TOP1]` | `0.4300` | Chinese/English Table 1, abstract, conclusion |
| `[TBD_COMPARE_AGENT_MAL_RECALL]` | `0.9333` | Chinese/English Table 1, abstract, conclusion |
| `[TBD_COMPARE_AGENT_ERROR_RATE]` | `0.0000` | Chinese/English Table 1 |
| `[TBD_COMPARE_TOP1_ABS_DELTA]` | `+0.2482` | Chinese/English Table 1, abstract |
| `[TBD_COMPARE_MAL_RECALL_ABS_DELTA]` | `+0.1606` | Chinese/English Table 1, abstract |
| `[TBD_COMPARE_ERROR_RATE_ABS_DELTA]` | `-0.0100` | Chinese/English Table 1 |
| `[TBD_COMPARE_TOP1_REL_DELTA]` | `+136.5%` | documented here only |
| `[TBD_COMPARE_MAL_RECALL_REL_DELTA]` | `+20.8%` | documented here only |

## 2. Significance Placeholders

Source file:
- `outputs/paper_stats/significance_20260324_021221.json`

Resolved values:

| Placeholder | Value | Inserted into |
| --- | --- | --- |
| `[TBD_COMPARE_MCNEMAR_P]` | `4.1e-05` | Chinese/English Section 5.1, abstract |
| `[TBD_COMPARE_TOP1_BOOTSTRAP_CI]` | `mean 0.2504, 95% CI [0.14, 0.36]` | Chinese/English Section 5.1, abstract |
| `[TBD_COMPARE_MAL_RECALL_BOOTSTRAP_CI]` | `mean 0.1777, 95% CI [0.0732, 0.2941]` | Chinese/English Section 5.1, abstract |

Note:
- The `ablation_vs_full_agent` significance block was intentionally not cited because the retained statistics summary itself marks that block as unreliable for direct use.

## 3. Ablation Placeholders

Source file:
- `outputs/ablation/ablation_report_20260324_002952.json`

Resolved values:

| Placeholder | Value | Inserted into |
| --- | --- | --- |
| `[TBD_ABL_DIRECT_QWEN_TOP1]` | `0.1818` | Chinese/English Table 2 |
| `[TBD_ABL_DIRECT_QWEN_MAL_RECALL]` | `0.7500` | Chinese/English Table 2 |
| `[TBD_ABL_NO_RETRIEVAL_TOP1]` | `0.3200` | Chinese/English Table 2 |
| `[TBD_ABL_NO_RETRIEVAL_MAL_RECALL]` | `1.0000` | Chinese/English Table 2 |
| `[TBD_ABL_PLUS_METADATA_TOP1]` | `0.3200` | Chinese/English Table 2 |
| `[TBD_ABL_PLUS_METADATA_MAL_RECALL]` | `1.0000` | Chinese/English Table 2 |
| `[TBD_ABL_PLUS_SPECIALISTS_TOP1]` | `0.3200` | Chinese/English Table 2 |
| `[TBD_ABL_PLUS_SPECIALISTS_MAL_RECALL]` | `1.0000` | Chinese/English Table 2 |
| `[TBD_ABL_PLUS_CONTROLLER_TOP1]` | `0.3600` | Chinese/English Table 2 |
| `[TBD_ABL_PLUS_CONTROLLER_MAL_RECALL]` | `1.0000` | Chinese/English Table 2 |
| `[TBD_ABL_FULL_TOP1]` | `0.3300` | Chinese/English Table 2 |
| `[TBD_ABL_FULL_MAL_RECALL]` | `1.0000` | Chinese/English Table 2 |

## 4. Quality-suite Placeholders

Source file:
- `outputs/quality/quality_suite_20260324_010432.json`

Resolved values:

| Placeholder | Value | Inserted into |
| --- | --- | --- |
| `[TBD_QUALITY_TOP1]` | `0.4012` | Chinese/English Table 3 |
| `[TBD_QUALITY_TOP3]` | `0.6541` | Chinese/English Table 3 |
| `[TBD_QUALITY_MAL_RECALL]` | `0.9146` | Chinese/English Table 3, abstract |
| `[TBD_QUALITY_ERROR_RATE]` | `0.0000` | Chinese/English Table 3 |
| `[TBD_QUALITY_BRIER]` | `0.3072` | Chinese/English Table 3, text |
| `[TBD_QUALITY_ECE]` | `0.2588` | Chinese/English Table 3, abstract, conclusion |
| `[TBD_QUALITY_MALIGNANT_MISS_COUNT]` | `14` | Chinese/English Table 3 |
| `[TBD_QUALITY_BENIGN_FALSE_ALARM_COUNT]` | `89` | Chinese/English Table 3 |
| `[TBD_QUALITY_F1_ACK]` | `0.1600` | Chinese/English Table 4 |
| `[TBD_QUALITY_F1_BCC]` | `0.5903` | Chinese/English Table 4 |
| `[TBD_QUALITY_F1_MEL]` | `0.0000` | Chinese/English Table 4 |
| `[TBD_QUALITY_F1_NEV]` | `0.3607` | Chinese/English Table 4 |
| `[TBD_QUALITY_F1_SCC]` | `0.0870` | Chinese/English Table 4 |
| `[TBD_QUALITY_F1_SEK]` | `0.0526` | Chinese/English Table 4 |

Additional quality observations incorporated into the draft:
- rare-site proxy: `n=39`, top-1 `0.2564`, malignant recall `0.8000`
- perception fallback: `n=4`, top-1 `0.7500`, malignant recall `0.5000`

## 5. Training-stage Placeholders from the Earlier Draft

Originally planned placeholders:
- `[TBD_TRAIN_BEST_VAL_TOP1]`
- `[TBD_TRAIN_TEST_TOP1]`
- `[TBD_TRAIN_TEST_TOP3]`
- `[TBD_TRAIN_TEST_MAL_RECALL]`
- `[TBD_BANK_TOTAL]`
- `[TBD_BANK_PROTO_CONF_RULE]`

Current status:
- The retained synced artifact bundle does not include the original staged training summary JSON with held-out test metrics.
- Instead of fabricating missing values, the paper draft was revised to report only the checkpoint metadata that are actually available.

Available retained values used in Table 5:

| Field | Value | Source |
| --- | --- | --- |
| Best validation Top-1 | `0.4220` | `outputs/checkpoints/learned_controller_best.json` -> `metadata.best_val_accuracy_top1` |
| Epoch | `1` | `outputs/checkpoints/learned_controller_best.json` -> `metadata.epoch` |
| Final memory-bank size | `1523` | `outputs/checkpoints/learned_bank_best.json` -> `stats.total` |
| Raw-case memories | `824` | `outputs/checkpoints/learned_bank_best.json` -> `stats.raw_case` |
| Prototype / Confusion / Rule memories | `31 / 3 / 11` | `outputs/checkpoints/learned_bank_best.json` -> `stats.prototype/confusion/rule` |
| Hard-case memories | `654` | `outputs/checkpoints/learned_bank_best.json` -> `stats.hard_case` |

Unresolved and intentionally omitted from the paper:
- staged held-out test top-1
- staged held-out test top-3
- staged held-out test malignant recall

Reason:
- those metrics are not present in the retained result bundle provided for this fill pass.

## 6. Scope Note

- This fill pass intentionally updates PAD-UFES-20 in-domain results only.
- DDI-related problem analysis was intentionally not expanded in the paper text for this revision.
