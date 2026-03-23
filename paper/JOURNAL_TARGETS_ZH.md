# 投稿目标建议

本文件给出当前项目在最终结果补齐后的投稿建议，按“最稳妥”到“最冲刺”的顺序排列。

## 第一梯队：优先考虑
### 1. Artificial Intelligence in Medicine
- 最匹配当前工作形态。
- 适合“有明确方法设计 + 医疗任务验证 + 系统性实验”的论文。
- 你这篇的 agent 架构、经验库、learned scorer、公平对比协议都比较适合该刊口味。

### 2. Computer Methods and Programs in Biomedicine
- 对系统型方法论文比较友好。
- 如果最终结果扎实、实验完整，这个方向很稳。
- 适合强调“完整工作流、实验脚本、可复现实验和外部验证”。

## 第二梯队：有机会冲
### 3. IEEE Journal of Biomedical and Health Informatics
- 更偏 biomedical informatics / digital health / signal and imaging intelligence。
- 如果长训练后的结果、显著性和外部验证都比较漂亮，可以尝试。
- 需要更注意论文整体表达的规范性和实验完整度。

### 4. Journal of Biomedical Informatics
- 如果你更强调“系统、决策支持、知识和经验建模”，这也是合适目标。

## 第三梯队：冲刺选择
### 5. npj Digital Medicine
- 难度明显更高。
- 如果没有更强的外部验证、多中心或更接近临床的证据，不建议作为首投。
- 更适合作为“如果最终结果特别漂亮，再考虑”的冲刺目标。

## 当前最现实的排序建议
1. Artificial Intelligence in Medicine
2. Computer Methods and Programs in Biomedicine
3. IEEE Journal of Biomedical and Health Informatics
4. Journal of Biomedical Informatics
5. npj Digital Medicine

## 影响投稿档次的关键因素
不是代码还改不改，而是最终实验能不能把下面几项补齐：

- PAD-UFES-20 全量或更稳健的最终主结果
- 最终 ablation
- DDI 外部结果
- calibration / subgroup / fairness
- 显著性检验
- 最好再有 multi-seed 或更稳健的 holdout 结果

## 结论
以当前项目进度看：

- **B 档期刊比较稳**
- **A 档期刊可以冲**
- 不建议现在就按临床顶刊标准预期

等最终长训练结果出来后，可以再按真实实验强度重排目标。
