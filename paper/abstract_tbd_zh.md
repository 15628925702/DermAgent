# 中文摘要

本文提出 DermAgent，一个面向多模态皮肤病分类的分阶段参数化诊断代理。DermAgent 围绕四个核心设计展开：显式 self-awareness signal、可学习 skill policy、learned experience bank，以及 uncertainty-aware aggregation。与仅依赖静态提示或固定规则路由的流程不同，系统保留规则作为安全先验，同时引入 learned controller、final scorer、rule scorer 与 evidence calibrator，从而实现从规则编排到参数化决策的平滑过渡。

我们在 PAD-UFES-20 上将 DermAgent 作为 staged-validation 系统进行评估。在冻结的同模型 100 例配对主对比中，DermAgent 将 Top-1 accuracy 从 0.1818 提升到 0.4300，将 malignant recall 从 0.7727 提升到 0.9333，对应绝对增益分别为 +0.2482 和 +0.1606，同时将 error rate 从 0.0100 降到 0.0000。该配对收益具有统计显著性，McNemar `p=4.1e-05`；bootstrap 得到的 Top-1 平均差为 0.2504，95% CI 为 `[0.14, 0.36]`，malignant recall 平均差为 0.1777，95% CI 为 `[0.0732, 0.2941]`。分阶段消融表明，最清晰的额外域内收益来自 learned controller/scorer 组件，而 retrieval 在当前预算下尚未成为稳定的正收益模块。在完整 PAD-UFES-20 测试集的 quality suite（`n=344`）中，DermAgent 达到 Top-1 0.4012、Top-3 0.6541、malignant recall 0.9146、Brier score 0.3072、ECE 0.2588，说明系统已具备较强恶性敏感性，但 calibration 与类别均衡性仍明显不足。

这些结果表明，在最终 full training 完成之前，learned components 已经开始以可测方式影响最终决策。更重要的是，本文给出了一条可操作的医学 AI agent 升级路径：先建立可靠的规则骨架，再逐步验证 controller、memory、calibration 与 scoring 模块是否真正开始带来收益。
