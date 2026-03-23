# DermAgent 最终结果汇总

## 范围

本文档汇总了 2026 年 3 月 24 日本地已同步的最终实验结果，并标明哪些结果可以直接写入论文，哪些结果需要谨慎解释。

对应结果文件：

- `outputs/comparison/comparison_report_20260324_004023.json`
- `outputs/ablation/ablation_report_20260324_002952.json`
- `outputs/quality/quality_suite_20260324_010432.json`
- `outputs/external_ddi/external_ddi_report_20260324_020035.json`
- `outputs/paper_stats/significance_20260324_021221.json`
- `outputs/paper_figures/figure_manifest_20260324_021227.json`

## 1. PAD-UFES-20 主对比结果

在 100 个测试样本上的冻结同模对比：

- Agent+Qwen top-1 准确率：`0.4300`
- Direct Qwen top-1 准确率：`0.1818`
- top-1 绝对提升：`+0.2482`
- Agent 恶性召回：`0.9333`
- Direct Qwen 恶性召回：`0.7727`
- 恶性召回绝对提升：`+0.1606`
- Agent 错误率：`0.0000`
- Direct Qwen 错误率：`0.0100`
- 样本对齐：`True`
- 同模公平性检查：`True`

结论：

- 在 PAD-UFES-20 域内测试中，Agent 框架显著优于直接调用 Qwen。
- 这一部分可以作为论文主结果表。

## 2. PAD-UFES-20 消融实验

100 样本分阶段消融结果：

- Direct Qwen：top-1 `0.1818`，恶性召回 `0.7500`
- Agent without retrieval：top-1 `0.3200`，恶性召回 `1.0000`
- `+ metadata`：top-1 `0.3200`，恶性召回 `1.0000`
- `+ specialists`：top-1 `0.3200`，恶性召回 `1.0000`
- `+ controller/scorers`：top-1 `0.3600`，恶性召回 `1.0000`
- Full agent `(+ retrieval)`：top-1 `0.3300`，恶性召回 `1.0000`

结论：

- 域内主要收益首先来自 Agent 框架本身，相对于直接 Qwen 有明显提升。
- learned controller/scorers 带来了最清晰的额外收益。
- 当前 retrieval 还不是稳定正收益模块。

## 3. PAD-UFES-20 质量评测

完整测试集 `n=344`：

- Top-1 准确率：`0.4012`
- Top-3 准确率：`0.6541`
- 恶性召回：`0.9146`
- 错误率：`0.0000`
- Brier top-1：`0.3072`
- 期望校准误差 ECE：`0.2588`

安全性摘要：

- 漏诊恶性数：`14`
- 良性误报数：`89`

OOD 代理切片：

- rare-site proxy：`n=39`，top-1 `0.2564`，恶性召回 `0.8000`
- perception fallback：`n=4`，top-1 `0.7500`，恶性召回 `0.5000`

结论：

- 域内恶性召回相对较强。
- 但校准仍明显不足，应该在论文中作为限制说明。
- 少见部位样本上的性能明显弱于总体表现。

## 4. 外部 DDI 评测

DDI 外部测试集：

- 总样本数：`656`
- 可映射到 6 类标签的样本数：`230`
- 二分类 benign / malignant：`485 / 171`

Agent 结果：

- 二分类准确率：`0.7485`
- 恶性召回：`0.0409`
- 特异性：`0.9979`
- 平衡准确率：`0.5194`
- 映射子集上的 6 类 top-1：`0.1522`
- 二分类 ECE：`0.0885`

Direct Qwen 结果：

- 二分类准确率：`0.9314`
- 恶性召回：`0.7953`
- 特异性：`0.9794`
- 平衡准确率：`0.8874`
- 映射子集上的 6 类 top-1：`0.9565`
- 二分类 ECE：`0.0797`

成对外部比较：

- 二分类准确率差值（Agent - Direct）：`-0.1829`
- 恶性召回差值：`-0.7544`
- 特异性差值：`+0.0185`
- McNemar p 值：`0.000000`

观察到的失败模式：

- Agent 在 `656` 个 DDI 样本中有 `648` 个被预测为 `BENIGN`
- 主导预测标签是 `NEV`，共 `632` 个样本
- 这是非常明确的外部域 benign-collapse

结论：

- 当前系统的外部泛化是失败的。
- DDI 不适合写成“外部验证成功”，而应写成 failure analysis / limitation。

## 5. 统计显著性

可以直接使用的统计结果：

- Direct vs Agent 的 McNemar p 值：`4.1e-05`
- top-1 bootstrap 均值差：`0.2504`，95% CI `[0.14, 0.36]`
- 恶性召回 bootstrap 均值差：`0.1777`，95% CI `[0.0732, 0.2941]`

可以直接使用的外部统计结果：

- External DDI McNemar p 值：`0.0`
- top-1 bootstrap 均值差：`-0.2823`，95% CI `[-0.3171, -0.2485]`
- 恶性召回 bootstrap 均值差：`-0.9110`，95% CI `[-0.9583, -0.8571]`

重要提醒：

- 当前 `ablation_vs_full_agent` 这部分统计结果 **不能直接引用**。
- 原因是实现层面的字段不匹配：ablation 的 per-case 记录没有保存 `is_top1_correct` 和 `malignant_recalled`，导致统计脚本把差异错误算成了 0。

## 6. 论文定位建议

建议的论文定位：

- 一个较强的域内研究原型
- 一个在 PAD-UFES-20 上明显优于 Direct Qwen 的 baseline
- 有比较完整的机制分析、质量分析和失败分析
- 还不是外部稳健或临床级系统

建议的论述方式：

- 强调内部数据上的显著提升
- 强调 controller/scorers 的有效性
- 明确承认 DDI 外部泛化失败和 benign-collapse
- 将系统定位为 research prototype，而不是可部署的临床系统

## 7. 已生成图表

已生成的图表包括：

- `outputs/paper_figures/figure_compare_20260324_021227.svg`
- `outputs/paper_figures/figure_ablation_20260324_021227.svg`
- `outputs/paper_figures/figure_per_label_f1_20260324_021227.svg`
- `outputs/paper_figures/figure_calibration_20260324_021227.svg`
- `outputs/paper_figures/figure_confusion_20260324_021227.svg`
- `outputs/paper_figures/figure_external_ddi_binary_20260324_021227.svg`
- `outputs/paper_figures/figure_external_ddi_skin_tone_20260324_021227.svg`
