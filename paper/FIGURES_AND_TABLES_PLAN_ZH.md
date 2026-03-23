# 论文图表清单

本文件用于明确最终论文需要哪些图和表，以及它们分别由哪些脚本生成。

## 主表
### 表 1：PAD-UFES-20 主对比结果
包含：
- Direct Qwen
- DermAgent
- Top-1 accuracy
- Malignant recall
- Error rate

来源：
- `scripts/compare_agent_vs_qwen.py`

### 表 2：系统级 ablation
包含：
- Direct Qwen
- Agent without retrieval
- + metadata
- + specialists
- + controller/scorers
- Full agent (+ retrieval)

来源：
- `scripts/run_agent_ablation.py`

### 表 3：外部 DDI 结果
包含：
- binary accuracy
- malignant recall
- specificity
- balanced accuracy
- 如果可用，再加 direct Qwen 对照

来源：
- `scripts/run_external_ddi_eval.py`

## 主图
### 图 1：Agent vs Direct Qwen 主对比图
来源：
- `scripts/export_paper_figures.py`

### 图 2：Ablation 图
来源：
- `scripts/export_paper_figures.py`

### 图 3：内部 per-label F1
来源：
- `scripts/export_paper_figures.py`

### 图 4：内部 calibration curve
来源：
- `scripts/export_paper_figures.py`

### 图 5：内部 confusion matrix
来源：
- `scripts/export_paper_figures.py`

### 图 6：外部 DDI binary 指标图
来源：
- `scripts/export_paper_figures.py --external-ddi-report ...`

### 图 7：外部 DDI subgroup 图
来源：
- `scripts/export_paper_figures.py --external-ddi-report ...`

## 统计结果
### 显著性检验
包括：
- McNemar p-value
- paired bootstrap 95% CI

来源：
- `scripts/run_significance_tests.py`

## 最终生成顺序
建议按以下顺序整理论文图表：

1. 跑最终 compare
2. 跑最终 ablation
3. 跑 quality suite
4. 跑 external DDI eval
5. 跑 paper evidence suite
6. 从输出目录中挑选最终图表与表格数据
