# 标题候选
1. DermAgent：从规则驱动到参数化皮肤病诊断代理的分阶段验证
2. DermAgent：融合自我认知信号、可学习技能策略与经验记忆的多模态皮肤病诊断系统
3. 从规则系统到参数化系统：面向皮肤病诊断的分阶段可学习代理

说明：本稿已根据 2026 年 3 月 24 日同步完成的 PAD-UFES-20 结果文件完成回填。本版本按你的要求不展开 DDI 相关问题分析。

## 摘要

本文提出 DermAgent，一个面向多模态皮肤病分类的分阶段参数化诊断代理。系统围绕四个核心设计展开：显式 self-awareness signal、可学习的 skill policy、可持续演化的 learned experience bank，以及 uncertainty-aware aggregation。与仅依赖静态提示词或固定路由规则的流程不同，DermAgent 保留规则作为安全先验，同时引入 learned controller、final scorer、rule scorer 与 evidence calibrator，使系统能够从“规则编排”逐步过渡到“参数化决策”。

本文将 DermAgent 作为一个 preliminary staged-validation 系统进行评估，而不是将其表述为最终 full training 模型。当前保留的结果文件覆盖三组互补协议：100 例冻结同模型主对比、100 例分阶段消融，以及覆盖 PAD-UFES-20 全部测试集的 quality suite（`n=344`）。在主对比中，DermAgent 将 Top-1 accuracy 从 0.1818 提升到 0.4300，将 malignant recall 从 0.7727 提升到 0.9333，对应绝对增益分别为 +0.2482 和 +0.1606，同时 error rate 从 0.0100 降至 0.0000。该配对结果具有统计显著性，McNemar `p=4.1e-05`；bootstrap 得到的 Top-1 平均差为 0.2504，95% CI 为 `[0.14, 0.36]`，malignant recall 平均差为 0.1777，95% CI 为 `[0.0732, 0.2941]`。在消融实验中，最清晰的额外域内收益来自 learned controller/scorer 组件；而 retrieval 在当前预算下尚未成为稳定的正收益模块。quality suite 显示，系统在全测试集上达到 Top-1 0.4012、Top-3 0.6541、malignant recall 0.9146、Brier score 0.3072、ECE 0.2588，说明其恶性敏感性已较强，但 calibration 与类别均衡性仍明显不足。

这些结果表明，在尚未完成最终大规模训练之前，learned components 已经开始以可测方式影响最终决策。本文的贡献不仅在于当前性能增益，更在于给出了一条可操作的系统升级路径：先建立可靠的规则骨架，再逐步验证 controller、memory、calibration 与 scoring 模块是否真正开始起作用。这一路径不仅适用于皮肤病诊断代理，也对更广泛的医学 AI 多模块系统具有参考价值。

## 1 引言

皮肤病诊断是一个高混淆、多类别、强风险约束的视觉决策问题。单次前向推理往往会在若干临床相近类别之间摇摆，例如 ACK 与 SCC、BCC 与 SEK、MEL 与 NEV。现实诊断中，模型不仅要“猜一个标签”，还需要结合病灶部位、患者年龄、病史线索，以及一个更关键的元层问题：当前证据是否足够稳定，系统到底有多不确定。近年来，多模态基础模型显著提升了这一任务的上限，但若将其直接当作 one-shot 分类器使用，仍存在若干现实缺口。

第一，单步模型通常缺乏稳定的自我认知接口。即使模型内部存在不确定性，也未必会以结构化状态暴露出来，供后续模块据此决定是否检索、是否调用专科技能、是否更保守地处理高风险样本。第二，纯规则式 agent 流程虽然可解释、可控，却难以随着经验积累自然改进。它很难真正“学会”何时该调用 specialist skill、何时该相信 retrieval、何时应该停止加证据并保留 perception anchor。第三，完全端到端的大规模训练固然方向明确，但在系统研发过程中，更紧迫的问题往往不是“最终极限能到多高”，而是“新加入的 learned components 是否已经开始有用”。

本文关注的正是这个过渡性问题：当我们把一个皮肤病诊断代理从规则系统升级为参数化系统时，是否能在中等训练预算下，提前看到 learned components 已经开始发挥作用的证据？围绕这一问题，我们构建了 DermAgent。它不是简单地在大模型外包一层流程控制，而是把诊断流程拆成一组显式、可观察、可学习、可写回的模块：perception 负责产生 differential diagnosis 候选与 uncertainty 线索；retrieval 从 experience bank 中取回相似病例、prototype、confusion 与 rule 证据；planner 基于 case state 触发候选技能；learned controller 在规则先验上进一步学习 skill routing；aggregator 将 perception、retrieval、metadata、specialist 与 malignancy 线索做 uncertainty-aware 融合；reflection 与 writeback 则把高价值样本沉淀为未来可复用的记忆。

DermAgent 的关键设计不在于抛弃规则，而在于重新定义规则的角色。规则不再是最终决策本身，而是变成安全护栏和初始化先验；memory 不再只是静态 few-shot prompt，而成为一个会增长、会压缩、会反馈到路由和聚合中的结构化经验库；最终决策也不再只由人工固定加权决定，而是允许 learned final scorer 和 evidence calibrator 在约束下调整候选排序。这样的设计非常适合 staged validation：我们可以在完整 full training 之前，先验证 learned controller、learned scorer 与 learned memory 是否已经开始影响结果。

当前结果对这个问题给出了肯定但克制的回答。在冻结的同模型 100 例主对比中，DermAgent 相比 direct Qwen 在 Top-1 accuracy 和 malignant recall 上均取得显著提升；在消融实验中，learned controller/scorer 是最清晰的额外增益来源；但在全测试集 quality suite 中，系统仍暴露出 calibration 偏弱、类别表现不均衡、稀有切片性能下降等问题。因此，本文要表达的不是“系统已经完成”，而是“从规则到参数化的过渡已经开始产生可验证收益”。

本文贡献可概括为三点。第一，我们提出了一个显式围绕“从规则系统到参数化系统”过渡而设计的皮肤病诊断代理。第二，我们以 self-awareness signal、learnable skill policy、learned experience bank 和 uncertainty-aware aggregation 具体实现这一过渡。第三，我们通过同模型冻结主对比、分阶段消融、统计显著性检验和 quality suite 分析，证明 learned components 已开始在域内决策中产生可测收益，同时明确暴露出当前版本尚未解决的问题。

## 2 相关工作

### 2.1 多模态皮肤病诊断系统

现有皮肤病 AI 大致可以分为两类。一类是传统图像分类器或图像加元数据预测器，直接输出疾病标签；另一类则依赖更强的多模态基础模型，执行解释生成、问答或报告辅助。前者监督目标清晰，但在经验复用和显式推理方面能力有限；后者表达灵活，却往往缺乏稳定的中间状态与可控的决策接口。DermAgent 位于二者之间：它利用多模态模型完成感知与报告，但把检索、路由、校准与记忆写回都结构化、显式化。

### 2.2 医学 AI 中的 agent 化推理

越来越多医学 AI 系统采用 agent 化拆解，将诊断流程分解为证据收集、候选生成、鉴别诊断、风险评估和最终综合。这种设计能够提升可解释性，也更方便插入安全约束。然而，当全部工具调用都依赖手工脚本编排时，系统容易变脆，且难以随着数据积累真正变强。DermAgent 并不否定规则式 agent，而是把它升级成 hybrid 形式：规则负责安全与先验，controller 与 scorer 负责学习更细粒度的决策行为。

### 2.3 检索增强、病例推理与经验记忆

诊断任务天然适合经验检索，因为相似病例、疾病原型和高混淆对信息都能补足单个样本缺失的上下文。传统 case-based reasoning 往往依赖固定相似度；更近一步的 memory-augmented 方法则尝试把成功经验压缩成可复用知识。DermAgent 将 memory 明确组织为 raw case、prototype、confusion、rule 与 hard case，并允许通过 reflection 和 compression 持续演化。在这个意义上，本文所说的 learned memory 并不是完全可微的外部存储，而是一个会随训练演化、会影响后续推理的结构化经验系统。

### 2.4 不确定性、校准与安全性

医学 AI 的评估不能只看 Top-1 accuracy。不确定性质量、恶性召回、置信度校准和 subgroup 行为都直接影响风险。已有研究强调 calibration、selective prediction 与 subgroup evaluation 的重要性。本文虽不是临床部署研究，但已经把 uncertainty-aware planning、evidence calibration、恶性风险敏感聚合和专门的 quality suite 纳入同一框架，因此不仅能回答“有没有更准”，也能回答“是否更可分析、是否更风险敏感”。

## 3 方法

### 3.1 从规则系统到参数化系统

DermAgent 的总体流程如下：perception 产生初始 ddx 候选与不确定性线索；retrieval 从 experience bank 中找回相关案例、prototype、confusion 和 rule；planner 基于当前 case state 生成候选技能；learned controller 在规则先验上细化技能选择；各技能产生局部证据；uncertainty-aware aggregator 结合 learned final scorer 与 evidence calibrator 输出最终诊断；随后 reflection 和 writeback 决定哪些样本写回记忆库。

关键点在于，系统并非从一开始就是完全参数化的。相反，我们保留了规则骨架，以保证早期阶段仍具备可解释性与安全性，再在其上叠加四类 learned components：

1. 负责技能选择与 stop probability 的 learnable controller。
2. 负责候选重排序的 learned final scorer。
3. 负责 rule memory 建议加权的 learned rule scorer。
4. 负责证据权重和阈值调节的 learned evidence calibrator。

因此，DermAgent 并不把“规则”和“学习”视为二选一，而是让规则作为安全先验存在，让参数化模块逐步替代脆弱的人工加权逻辑。

### 3.2 Self-awareness signal

本文将 self-awareness signal 定义为：在最终诊断之前，系统用来刻画“当前这个样本有多难、证据稳不稳、支撑是否不足”的结构化内部状态。这些信号来自系统中间过程，而不是事后人工标注。当前实现至少暴露以下信息：

- perception uncertainty（`low`、`medium`、`high`）；
- 候选前两名之间的分数间隔 `top_gap`；
- retrieval confidence，以及 retrieval 是否支持当前 top-1；
- confusion memory 是否给出支持；
- metadata 是否存在，以及年龄/部位是否与候选疾病一致；
- 是否存在恶性候选、是否命中高混淆疾病对；
- controller 估计的 stop probability。

这些信号有两个作用。其一，它们驱动 planner 层的规则触发逻辑，例如 uncertainty 高、top-gap 小、memory 有建议时，更倾向触发 compare skill 或 specialist skill。其二，它们被编码成 controller 的数值特征，使 controller 能学习在什么状态下更应相信哪类证据、保留哪些技能、何时可以提前停止。这也是系统从“只看输入”过渡到“同时看内部状态”的关键。

### 3.3 可学习技能策略

DermAgent 包含始终开启的 uncertainty assessment skill，以及 compare skill、malignancy risk skill、metadata consistency skill，和若干针对皮肤病高混淆对的 specialist skills。在纯规则系统中，这些技能通常由固定条件触发；而在 DermAgent 中，controller 会针对每个技能学习一个条件概率，决定当前 case state 下是否值得执行。

controller 的输入同时包含 perception 特征、retrieval 特征、planner 先验和 memory 推荐。训练阶段中，它依据病例结果和局部证据是否真正有帮助的启发式目标进行更新。因此，controller 学到的不是抽象的语言建模行为，而是“在当前诊断状态下，哪些技能值得执行”的决策策略。

### 3.4 Learned memory bank / experience bank

DermAgent 的 memory bank 不是简单的文本缓存，而是结构化经验库。当前实现至少支持以下五类对象：

- `raw_case`：成功解决的原始案例；
- `prototype`：疾病级原型摘要；
- `confusion`：高混淆疾病对知识；
- `rule`：压缩后的规则型记忆，包含触发条件和推荐技能；
- `hard_case`：困难样本或近错样本。

writeback 决定哪些内容写入 bank，compression 则进一步从累计案例中提炼出 prototype、confusion 与 rule。在推理阶段，retriever 根据疾病匹配、metadata 相似性、不确定性对齐等因素检索经验，并输出 consensus label、prototype vote、confusion vote、retrieval confidence 与 recommended skills。因而，这里的 learned memory 有两层含义：其一，记忆内容会随训练演化；其二，这些演化后的记忆会直接参与后续路由和聚合。

### 3.5 Uncertainty-aware aggregation

最终诊断由 uncertainty-aware aggregation 产生，而不是简单投票。perception 始终作为 anchor 存在，retrieval、prototype、confusion、metadata、specialist 与 malignancy 线索都可能改变候选排序。为避免未训练充分的晚期模块破坏候选覆盖，aggregator 会保留 perception shortlist，并限制低支持候选被过度提升。

在此基础上，learned final scorer 对候选进行重排序。其特征包括 perception 支持、retrieval 一致性、skill 一致性、多源证据加成、memory consensus，以及恶性候选指标。evidence calibrator 则进一步调节 specialist 支持、metadata 支持和 planner threshold 等关键权重，使最终决策从固定人工加权，过渡到受约束的参数化融合。

### 3.6 面向风险的聚合行为

DermAgent 还引入了恶性敏感的聚合逻辑。当 malignancy risk skill 认为样本存在中高风险，而当前 top-1 为良性类别时，aggregator 会检查 specialist、retrieval 与候选结构中是否存在足够的恶性支持，从而在必要时采取更保守的输出。这还远不是临床级安全层，但它至少让 staged validation 可以评估系统是否正在朝着“更风险敏感”而非单纯“更自信”演化。

## 4 实验设计

### 4.1 数据集与任务

本文主任务为 PAD-UFES-20 六分类皮肤病诊断，类别包括 ACK、BCC、MEL、NEV、SCC 与 SEK。当前保留的 split 对应仓库标准划分，共 2298 例，其中 train/val/test 分别为 1608/346/344。本文定位为 staged validation，而非最终 full training 论文。

### 4.2 本稿使用的保留结果文件

当前论文直接基于以下已落盘结果文件：

- 主对比：`outputs/comparison/comparison_report_20260324_004023.json`
- 消融：`outputs/ablation/ablation_report_20260324_002952.json`
- 质量评测：`outputs/quality/quality_suite_20260324_010432.json`
- 显著性统计：`outputs/paper_stats/significance_20260324_021221.json`
- controller 检查点元信息：`outputs/checkpoints/learned_controller_best.json`
- memory bank 统计：`outputs/checkpoints/learned_bank_best.json`

其中，主对比和消融均为冻结的 100 例测试切片评估；quality suite 则覆盖 PAD-UFES-20 全测试集（`n=344`）。主对比报告还记录了 direct baseline 与 agent 使用相同的底层模型后端 `Qwen/Qwen2.5-VL-7B-Instruct`，样本完全对齐，且 benchmark 为 frozen evaluation。

### 4.3 评测协议

本文采用三组互补评测协议。

1. 主对比：在相同 100 例冻结测试切片上，将 DermAgent 与 direct Qwen 进行同模型配对比较。
2. 分阶段消融：依次评测 Direct Qwen、agent without retrieval、`+ metadata`、`+ specialists`、`+ controller/scorers` 和 full agent `(+ retrieval)`。
3. Quality suite：在完整 344 例测试集上评估 Top-k accuracy、malignant recall、calibration、安全计数与类别级表现。

这种设计将“参数化系统是否在配对切片上明显优于 direct baseline”与“完整冻结测试集上的整体质量轮廓如何”区分开来。

### 4.4 指标

主对比报告 Top-1 accuracy、malignant recall 与 error rate，并通过显著性文件提供 McNemar p-value 和 paired bootstrap 置信区间。消融主要关注 Top-1 accuracy 和 malignant recall。Quality suite 则报告 Top-1、Top-3、Brier score、ECE、安全计数、OOD proxy 切片以及 per-label precision/recall/F1。

本文特别将 malignant recall 视为一等指标，因为皮肤病诊断代理不应以牺牲恶性病例敏感性为代价换取更高的平均准确率。

### 4.5 论文定位

本文有意被定位为一篇 preliminary staged-validation 论文，而非最终 benchmark 论文。目标不是宣称系统已完全训练完成，而是回答一个更贴近系统研发的问题：这些 learned components 是否已经开始起作用，如果起作用，是通过哪些模块、以什么代价、在哪些维度上体现出来。

## 5 结果与计划中的后续完整版训练

### 5.1 主对比结果

表 1 给出主对比结果。DermAgent 将 Top-1 accuracy 从 0.1818 提升到 0.4300，将 malignant recall 从 0.7727 提升到 0.9333，同时将 error rate 从 0.0100 降至 0.0000。

表 1：冻结 100 例配对测试切片上的主对比结果

| 模型 | Top-1 Accuracy | Malignant Recall | Error Rate | 备注 |
| --- | --- | --- | --- | --- |
| Direct Qwen | 0.1818 | 0.7727 | 0.0100 | 同后端 direct multimodal baseline |
| DermAgent（冻结 staged checkpoint） | 0.4300 | 0.9333 | 0.0000 | learned controller + scorer + memory bank |
| 绝对差值（Agent - Direct） | +0.2482 | +0.1606 | -0.0100 | 对 accuracy/recall 正值更好；对 error rate 负值更好 |

这一结果不只是“趋势向好”，而是具备统计支撑。配对 McNemar 检验得到 `p=4.1e-05`。bootstrap 给出的 Top-1 平均差为 0.2504，95% CI 为 `[0.14, 0.36]`；malignant recall 平均差为 0.1777，95% CI 为 `[0.0732, 0.2941]`。由于 direct baseline 与 agent 使用相同的 Qwen 后端、相同样本顺序且均为冻结评测，这一提升更能支持“收益来自 agent 结构与 learned decision stack”的解释，而非来自模型后端差异。

### 5.2 分阶段消融

表 2 展示了收益如何随着系统逐步参数化而积累。

表 2：冻结 100 例测试切片上的分阶段消融

| 阶段 | Top-1 Accuracy | Malignant Recall | 解释 |
| --- | --- | --- | --- |
| Direct Qwen | 0.1818 | 0.7500 | 无 agent 结构 |
| Agent without retrieval | 0.3200 | 1.0000 | 基础 agent，不使用记忆检索 |
| + metadata | 0.3200 | 1.0000 | 加入 metadata consistency |
| + specialists | 0.3200 | 1.0000 | 加入高混淆专科技能 |
| + controller/scorers | 0.3600 | 1.0000 | 加入 learned controller、final scorer、rule scorer 与 evidence calibrator |
| Full agent (+ retrieval) | 0.3300 | 1.0000 | 完整 staged agent |

这一消融结果有三点尤其重要。第一，相对于 direct Qwen，agent 结构本身就带来了主要的首轮增益。第二，最清晰的额外收益来自 learned controller/scorer 组件，它将 Top-1 从 0.3200 提升到 0.3600，同时在该切片上保持 1.0000 的 malignant recall。第三，retrieval 在当前预算下尚未成为稳定的正收益模块：full agent 仍显著优于 direct Qwen，但其 0.3300 的 Top-1 低于 `+ controller/scorers` 阶段的 0.3600。这正是 staged validation 应该提前暴露出来的问题。

### 5.3 Quality suite、校准与安全性

Quality suite 在完整 PAD-UFES-20 测试集（`n=344`）上评估冻结 agent，因此比 100 例主对比切片更能反映系统的整体轮廓。

表 3：完整测试集上的核心 quality 指标

| 指标 | 数值 |
| --- | --- |
| Top-1 Accuracy | 0.4012 |
| Top-3 Accuracy | 0.6541 |
| Malignant Recall | 0.9146 |
| Error Rate | 0.0000 |
| Brier Score | 0.3072 |
| Expected Calibration Error | 0.2588 |
| Malignant Miss Count | 14 |
| Benign False Alarm Count | 89 |

表 4：完整测试集上的 per-label F1

| 类别 | F1 |
| --- | --- |
| ACK | 0.1600 |
| BCC | 0.5903 |
| MEL | 0.0000 |
| NEV | 0.3607 |
| SCC | 0.0870 |
| SEK | 0.0526 |

Quality suite 呈现出一种“有价值但不平衡”的轮廓。一方面，系统在完整测试集上维持了较高的 malignant recall（0.9146），且冻结评测中的 runtime error rate 为 0.0000。另一方面，calibration 仍明显偏弱，ECE 达到 0.2588，Brier score 为 0.3072；类别级表现也很不均衡，F1 主要集中在 BCC（0.5903）和部分 NEV（0.3607），而 MEL、SCC、SEK 的 F1 仍很低。这说明系统已经在“恶性敏感性”上有所建立，但还远未达到“类别均衡且置信可依赖”的状态。

OOD proxy 切片也支持这一判断。在 rare-site proxy（`n=39`）上，Top-1 accuracy 下降到 0.2564，malignant recall 下降到 0.8000，表明系统对分布边缘样本仍不稳。perception fallback 切片样本量很小（`n=4`），不适合做强结论，但同样提示边缘场景需要进一步优化。

### 5.4 可用的训练阶段元信息与后续 full training 扩展

当前同步结果包并未保留最初 staged training summary JSON 中的 held-out test metrics，因此本文不编造这些数值。相反，表 5 只报告目前实际可从 retained controller checkpoint 与 learned bank artifact 中提取的训练阶段元信息。

表 5：当前可用的 staged checkpoint 元信息

| 项目 | 数值 |
| --- | --- |
| Staged checkpoint epoch | 1 |
| Best validation Top-1 | 0.4220 |
| Final memory-bank size | 1523 |
| Raw-case memories | 824 |
| Prototype / Confusion / Rule memories | 31 / 3 / 11 |
| Hard-case memories | 654 |

这些数字应被理解为“学习和记忆增长已经在发生”的内部证据，而不是最终 benchmark 主表。后续 full training 扩展仍应补上 retained staged test metrics、多 seed 验证、更长程的 retrieval 分析和更稳定的 calibration 研究。但就本文当前的论点而言，主对比、消融、quality suite 和显著性分析已经足以支撑“learned components 开始起作用”的核心结论。

## 6 讨论

### 6.1 为什么阶段性结果仍然有意义

Staged validation 的核心价值在于，它能在最终训练完成前回答研发中最关键的问题：learned components 是否已经开始有用？当前答案是肯定的。同模型配对主对比显示，DermAgent 相比 direct Qwen 在 Top-1 和 malignant recall 上均有显著提升；消融结果进一步表明，learned controller/scorer 是最清晰的额外收益来源。这说明“从规则骨架到参数化诊断代理”的转变不是停留在架构设想，而已经开始在结果上可测。

### 6.2 为什么仍然需要后续 full training

但阶段性收益并不能替代最终结论。当前 retrieval 尚不是稳定正收益模块，这意味着 memory 质量、检索打分或聚合策略仍可能欠训练。Quality suite 也显示，高 malignant recall 并不自动等价于高 calibration 或高类别均衡性。因此，full training 的意义在于把问题从“学习是否开始有用”升级为“学习是否稳定、有显著性、可复现，并且伴随更合理的风险分布”。

### 6.3 当前版本已经验证了什么

当前版本已经验证了若干关键假设。第一，显式 self-awareness signal 可以作为有效的决策状态。第二，learned controller 能在规则先验之上细化 skill routing。第三，experience bank 不只是被动存储，而会通过 retrieval 输出参与路由和聚合。第四，uncertainty-aware aggregation 与 learned scoring 的组合，能够在域内同模型公平对比下击败 direct one-shot multimodal baseline。

### 6.4 当前版本尚未验证什么

当前版本尚不能证明最终性能上限、多 seed 稳定性、强外部泛化能力或临床级 calibration。它同样没有证明每个新增模块都已成熟有效：retrieval 是当前最明显仍待优化的模块之一。保留这些边界非常重要，因为本文要呈现的是“一个成功启动中的过渡过程”，而不是“一个已经完成的终局系统”。

## 7 局限性

本文存在以下局限性。第一，训练范式仍是 staged、medium-scale，而非 fully optimized regime。第二，主对比和消融都基于 100 例冻结切片，虽然足以发现方向性乃至显著效应，但仍不足以覆盖全部 subgroup 风险。第三，quality suite 已显示出显著的类别不均衡表现，包括 MEL 的 F1 为 0.0000，SCC 与 SEK 的 F1 也较低，因此当前系统显然不适合任何部署性表述。第四，尽管 malignant recall 较高，但 calibration 仍弱，ECE 0.2588 说明当前置信度不应被理解为临床可依赖概率。第五，本文中的 learned memory 主要指结构化写回、压缩与经验复用，而不是完全端到端可微记忆。最后，按照当前写作约束，本文不展开 DDI 外部失败分析，因此应将其理解为一篇域内 staged-validation 论文，而非完整外部验证论文。

## 8 结论

本文提出 DermAgent，一个面向皮肤病分类的分阶段参数化诊断代理。相对于同模型 direct Qwen baseline，DermAgent 在冻结配对测试切片上将 Top-1 accuracy 从 0.1818 提升到 0.4300，将 malignant recall 从 0.7727 提升到 0.9333，并获得统计显著的配对收益。消融分析表明，learned controller/scorer 组件已经开始产生清晰增益；而完整 quality suite 也表明，系统在恶性敏感性方面表现较强，但 calibration 与类别均衡性仍是主要限制。

从更广泛的角度看，本文的核心价值不只是当前这一组结果数字，而是一条可执行的系统升级路径：先搭建可靠的规则骨架，再逐步引入 controller、memory、calibration 与 scoring 等参数化模块，并在每一步用 staged validation 检查它们是否真的开始起作用。DermAgent 说明这条路径是可行的，但也同样清楚地揭示出，在更强结论成立之前，还有哪些模块和指标需要进一步打磨。
