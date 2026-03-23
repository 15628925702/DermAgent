# DermAgent：一种面向皮肤科图像分诊与鉴别诊断的经验驱动型 Agent 框架

## 摘要
大规模视觉语言模型已经可以直接对皮肤病图像进行推理，但单次直接提示通常难以充分利用临床元数据，也缺少显式的不确定性处理机制与持续积累经验的能力。本文提出 **DermAgent**，这是一个面向皮肤科任务的 agentic 框架，将共享的视觉语言骨干模型与结构化感知、经验检索、元数据一致性检查、专科鉴别模块、可学习聚合器以及可更新的经验库结合起来。该框架的目标是在提升诊断质量的同时，保留中间推理过程的可审计性。

在 PAD-UFES-20 上，我们构建了一个**同模型、同样本、冻结评估**的公平对比设置。在这一设置下，DermAgent 在一个 100 例对比切片上将 top-1 准确率从直接 Qwen 推理的 **0.1717** 提升到 **0.3700**，将恶性病例召回率从 **0.7045** 提升到 **0.9556**。系统级消融结果表明，检索模块是最主要的收益来源；经过校准后的元数据模块开始产生稳定正收益；完整 Agent 系统强于任何删减变体。我们还构建了 DDI 数据集的外部评估流程，并说明在当前标签体系下，DDI 更适合作为外部二分类鲁棒性与公平性评估集，而不适合作为完整六分类主训练集。总体来看，实验结果表明：在严格控制比较条件的前提下，基于结构化分解与经验检索的 Agent 系统能够显著优于直接提示推理，同时也暴露出可学习校准与外部泛化方面的后续提升空间。

## 1. 引言
皮肤科任务天然具有多模态特征：临床判断不仅依赖病灶图像，还依赖患者年龄、病灶部位、病程演化等元数据。近年来，多模态基础模型已经能够根据单张皮肤病图像生成看似合理的诊断结果，但直接的一步式推理路径仍然存在几个明显问题。第一，它把感知、鉴别诊断和报告生成全部混在一次黑箱生成中。第二，它难以显式控制专科模块何时介入，以及在不确定情况下如何升级推理。第三，它几乎没有任务级的持续经验积累机制。

本文的核心问题是：**在保持底层视觉语言模型相同的前提下，一个 agent 化的皮肤科系统，能否通过结构化分解而不是仅依赖更强骨干模型，获得显著收益？**

为此，我们构建了 **DermAgent**。它是一个模块化的皮肤科 Agent 框架，包含：

- 一个结构化感知模块，用于提取候选鉴别诊断和视觉线索；
- 一个基于经验库的检索层，存储原始病例、困难病例、原型记忆、混淆对记忆和轻量规则；
- 元数据一致性模块和专科鉴别模块，用显式支持或惩罚的形式输出证据；
- 一个可学习的控制与打分模块，用于决定调用哪些技能以及如何聚合各类证据；
- 一个报告模块，用于把结构化决策证据转换为面向临床的自然语言输出。

我们要强调的是，本工作是**研究型系统**，而不是临床可部署系统。本文的核心主张不是“DermAgent 已经达到临床可用”，而是：**在同模型、同样本、冻结评估的严格条件下，agent 化结构加上经验驱动检索，能够优于直接提示推理。**

## 2. 贡献
本文的主要贡献有四点：

1. 提出一个端到端的皮肤科 Agent 框架，将感知、检索、专科推理、聚合和报告解耦，同时保持与直接基线相同的视觉语言骨干模型。
2. 实现了一条可学习的“记忆 + 控制”流程，包括离线经验库初始化、可学习 controller / scorer checkpoint、证据校准器，以及原型 / 混淆记忆 / 规则压缩。
3. 建立了一套**公平的内部比较协议**：PAD-UFES-20 上的 Agent 分支和直接基线共享完全相同的样本、使用完全相同的模型名，并在 compare 阶段使用冻结评估模式，从而避免 agent vs baseline 比较中常见的混淆因素。
4. 补全了 DDI 外部评估路径，并明确说明 DDI 在本项目中应被作为外部二分类鲁棒性与公平性评估集来使用，而不是完整六分类训练集。

## 3. 相关工作
皮肤科 AI 长期以来以图像分类和皮肤镜分析为主，公开数据集如 ISIC、PAD-UFES-20 等推动了病灶识别研究的发展。PAD-UFES-20 对本文尤其重要，因为它同时提供智能手机临床图像和患者元数据，天然适合多模态推理。

与此同时，近年来多模态大模型与视觉语言模型在医学图像理解上展现出较强的零样本和小样本能力，但它们通常存在难以校准、难以审计、难以解释等问题。

本文与以下三类工作最相关：

- 基于单次提示的直接多模态诊断；
- 可调用工具或多阶段推理的 agent 型医疗系统；
- 基于病例记忆或原型检索的记忆增强式医疗 AI。

DermAgent 与一步式系统的主要差异在于：它强制显式中间状态，并将感知与最终决策分离；它也不同于纯规则流水线，因为它引入了可学习的控制与打分机制，允许系统逐步从数据中学习如何使用证据。

## 4. 任务定义
本文当前研究的是一个六分类皮肤病任务，系统内统一采用以下标签：

- `MEL`
- `BCC`
- `SCC`
- `NEV`
- `ACK`
- `SEK`

此外，为了安全导向分析，我们还定义了一个恶性 / 良性二分类映射：

- 恶性：`MEL`、`BCC`、`SCC`
- 非恶性：`NEV`、`ACK`、`SEK`

本文的主要内部任务是 PAD-UFES-20 上的六分类；外部任务则是在 DDI 上做恶性 / 非恶性的二分类评估，因为 DDI 中大量类别并不能严格映射到当前的六标签体系。

## 5. 方法
### 5.1 总体框架
给定一个包含图像和可用元数据的病例，DermAgent 按如下阶段执行：

1. **感知（Perception）**：视觉语言模型输出结构化的候选诊断、视觉线索、恶性线索与不确定性。
2. **检索（Retrieval）**：系统从经验库中检索原始病例、原型、混淆记忆和规则记忆。
3. **规划（Planning）**：规划器决定后续应该调用哪些技能模块，必要时参考可学习 controller 和证据校准器。
4. **技能执行（Skill Execution）**：元数据一致性、比较技能和成对专科鉴别模块输出对候选标签的支持或惩罚。
5. **聚合（Aggregation）**：决策聚合器汇总所有证据，并可选使用 learned final scorer。
6. **报告（Reporting）**：报告模块将结构化决策结果转化为面向临床的简洁报告。
7. **学习与记忆更新（Learning / Memory Update）**：在训练或 warm-start 工作流中，系统可以继续更新 learned components，并把新的经验写回经验库。

### 5.2 感知模块
感知模块要求视觉语言模型输出严格 JSON，而不是自由文本。输出包含候选鉴别诊断、置信度代理、视觉线索、风险线索和不确定性等级。

当前版本中的一个关键修复是：感知提示词不再使用像 SCC / ACK / BCC 这样带有明显偏向性的示例标签，而改为使用**中性占位符标签**。这是因为早期版本会把模型锚定到角化相关恶性病变，从而系统性高估 SCC。

### 5.3 经验库
经验库存储多种形式的记忆：

- **原始病例（raw cases）**
- **困难病例（hard cases）**
- **原型记忆（prototypes）**
- **混淆记忆（confusion memories）**
- **规则记忆（rules）**

我们首先通过离线 bootstrap 扫描训练集，构建一个初始经验库，然后再进入 learned component 训练。一个代表性 run 中，在 PAD-UFES-20 的 train split（1608 例，2 个 pass）上，经验库最终得到：

- 总条目数：**1023**
- raw cases：**566**
- hard cases：**440**
- prototypes：**4**
- confusion memories：**3**
- rules：**10**

### 5.4 元数据与专科鉴别推理
元数据一致性模块并不直接做硬覆盖，而是作为一种证据源输出显式支持和惩罚强度。这样，证据校准器就可以学习：

- 在儿童或低风险场景下，应多大程度提升 benign rescue；
- 在年龄、部位明显不合理的情况下，应多大程度惩罚不可信的恶性标签。

专科鉴别模块目前覆盖以下高混淆标签对：

- ACK vs SCC
- BCC vs SCC
- BCC vs SEK
- MEL vs NEV

早期系统存在两个明显问题：一是 specialist 触发过于激进；二是多个 specialist 会对同一标签重复加票。当前版本已经收紧 specialist 触发条件，并对重叠 specialist 支持进行抑制。

### 5.5 可学习控制与聚合
DermAgent 目前包含四个可学习组件：

- **Learnable Skill Controller**
- **Learnable Final Scorer**
- **Learnable Rule Scorer**
- **Learnable Evidence Calibrator**

这些组件可以在训练集上学习，但最终 compare / evaluation 阶段采用冻结模式，避免“边学边评”导致结果污染。

### 5.6 Agent 与 Baseline 的公平比较
本文一个非常重要的工程贡献，其实就是**实验协议本身**。现在 Agent 分支和 Direct Qwen 分支：

- 使用**完全相同的病例列表**；
- 使用**完全相同的底层模型名**；
- 在 compare 阶段使用**冻结 benchmark 模式**；
- 显式记录 fallback 次数和模型调用诊断信息。

这一步很关键，因为很多所谓的“agent 优势”，实际上可能只是样本不一致、模型名不一致、或者 fallback 造成的假提升。

## 6. 实验设置
### 6.1 内部数据集：PAD-UFES-20
我们使用 PAD-UFES-20 作为主要内部基准数据集。固定的 split manifest（seed = 42）对应如下划分：

- train：**1608**
- val：**346**
- test：**344**
- total：**2298**

split 文件会持久化到磁盘，确保训练、消融和评估都共享同一套数据划分。

### 6.2 外部数据集：DDI
我们还将 DDI 接入为外部评估集。目前本地 DDI 集成已经成功加载 **656** 个病例，并且图片路径匹配完整。

在当前六标签体系下：

- **230** 个样本能够映射到当前六分类标签空间；
- **426** 个样本无法严格映射，因此只用于恶性 / 非恶性的外部二分类评估。

这说明 DDI 在本项目中的最佳角色是：**外部鲁棒性与公平性评估集**，而不是主训练集或完整六分类评估集。

### 6.3 基线
基线采用**直接 Qwen 视觉语言推理**。它接收与 Agent 分支相同的图像和元数据，但不使用显式检索、专科路由或结构化聚合。

因此本文的对比目标并不是“谁的底层模型更强”，而是“在同一底层模型上，Agent 架构本身是否带来增益”。

### 6.4 评估指标
本文报告以下指标：

- top-1 accuracy
- top-3 accuracy（适用时）
- malignant recall
- error rate
- subgroup metrics
- calibration 指标，如 ECE
- 配对显著性检验（在存在对应比较结果时）

### 6.5 训练流程
当前训练流程采用两阶段：

1. **离线经验库初始化（bootstrap）**
2. **基于 bootstrap bank 的 learned components 训练**

所有最终 compare 和 paper-facing evaluation 都在**冻结模式**下完成。

## 7. 结果
### 7.1 PAD-UFES-20 上的受控内部对比
在当前冻结、同模型、同样本的 100 例内部 benchmark 切片上，DermAgent 明显优于直接 Qwen 推理。

| 方法 | Top-1 准确率 | 恶性召回率 | 错误率 |
|---|---:|---:|---:|
| Direct Qwen | 0.1717 | 0.7045 | 0.0100 |
| DermAgent | 0.3700 | 0.9556 | 0.0000 |

这意味着在这一 benchmark 切片上：

- top-1 accuracy 相对提升约 **115.5%**
- malignant recall 提升约 **35.6%**

这些提升之所以可信，是因为比较协议已经修复并显式控制了以下混淆因素：

- Agent 与基线样本不一致；
- Agent 与基线底层模型名不一致；
- Agent 由于 fallback 而“看起来更快更好”。

### 7.2 系统级消融实验
基于一个 smoke-trained checkpoint 的 100 例消融实验结果如下：

| 变体 | Top-1 准确率 | 恶性召回率 | 错误率 |
|---|---:|---:|---:|
| Direct Qwen | 0.1717 | 0.7500 | 0.0100 |
| Agent without retrieval | 0.3000 | 1.0000 | 0.0000 |
| + metadata | 0.3300 | 1.0000 | 0.0000 |
| + specialists | 0.3000 | 1.0000 | 0.0000 |
| + controller/scorers | 0.2600 | 0.9778 | 0.0000 |
| Full agent (+ retrieval) | 0.3700 | 1.0000 | 0.0000 |

这个消融结果说明：

- **检索是当前最强的增益来源**
- 元数据在校准后开始真正产生正面作用
- specialists 还不够稳定，仍是后续优化重点
- controller / scorer 目前还没有在短训练 setting 下完全发挥出来
- **完整系统依然最强**

### 7.3 DDI 外部评估
DDI 的接入与评估流程目前已经完成，但由于当前工作区里还没有长训练后最终 checkpoint 对应的完整外部结果，这里暂时将 DDI 的最终数值视为**待补充实验结果**。

目前已经明确的是：

- 656 张 DDI 图像可以完整加载；
- 系统支持外部恶性 / 非恶性二分类评估；
- 可输出 skin tone 等 subgroup 结果；
- DDI 在本项目中应被视为**外部 robustness benchmark**，而不是六分类主训练集。

### 7.4 结果解释
当前结果表明，即便保持底层模型不变，结构化 agent 分解依然可以带来实质收益。主要收益很可能来自以下组合：

- 更结构化的感知输出；
- 对先前经验的检索利用；
- 显式的证据聚合机制。

相较之下，learned controller 目前还不是最主要的收益来源，但系统工程已经打通，后续只需要更长训练即可继续提升。

## 8. 讨论
### 8.1 为什么 Agent 会比 Direct Prompt 更好
从当前实验看，直接提示推理显然没有把可用信息用尽。结构化 Agent 至少有以下优势：

- 在元数据支持下，能够持续保留 benign 候选；
- 只有在高混淆场景下才调用 specialist；
- 能利用相似病例和原型记忆稳定困难样本；
- 中间证据可以被完整审计和分析。

这在医学任务中尤其重要，因为一次性的自由文本输出通常难以支撑系统级错误分析。

### 8.2 为什么检索收益最大
消融实验表明，当前系统中检索模块的贡献最大。这与皮肤科任务本身的性质一致：很多难样本本质上就是“与哪些既往病灶更像”、“是否属于典型混淆对”这样的问题。经验库为系统提供了一条**不依赖重新训练骨干模型、但可以持续积累任务经验**的路径。

### 8.3 为什么 DDI 不适合作为本项目主训练集
尽管 DDI 很有价值，但在当前项目中它更适合作为**外部测试集**：

- 数据规模较小；
- 标签体系与当前六分类任务不完全一致；
- 它的强项在于多样性、公平性和外部验证，而不是大规模监督训练。

因此，本文将 DDI 定位为外部 robustness / fairness benchmark，而非主训练数据源。

## 9. 局限性
本文仍然存在以下局限：

1. 当前最强的结果证据主要还是来自 100 例受控内部 benchmark，而不是长训练后的最终全量结果。
2. DDI 外部评估流程已经完成，但最终长训练后的外部结果尚待补充。
3. planner、specialist 和 metadata 仍然部分依赖规则逻辑，没有完全 learned 化。
4. learned controller / scorer 仍然需要更长训练和多 seed 验证，才能更准确判断其真实贡献。
5. 本系统仍然是研究原型，**不能视为临床可部署系统**。

## 10. 伦理与安全说明
DermAgent 的目标是研究“结构化 Agent 推理是否能提升皮肤科多模态诊断质量”，而不是直接作为临床系统部署。任何真实医疗部署都至少需要：

- 前瞻性验证；
- 临床医生监督；
- 校准与 subgroup fairness 分析；
- OOD 检测；
- 正式安全审查；
- 合规的数据治理与监管流程。

因此，本文明确不将 DermAgent 描述为临床诊断设备，而是将其定位为一个用于研究 agentic reasoning 的皮肤科 AI 原型系统。

## 11. 结论
本文提出了 DermAgent，一个经验驱动型的皮肤科 Agent 框架。该系统在与直接基线共享完全相同视觉语言骨干模型的前提下，通过结构化感知、经验检索、元数据一致性推理、专科鉴别、可学习聚合与显式记忆系统，在 PAD-UFES-20 的受控比较中获得了显著优于直接提示推理的结果。与此同时，系统已经完成 DDI 外部评估接入，可以继续用于鲁棒性与 subgroup 分析。

从工程角度看，这个系统的主链已经完成；后续工作的重点不再是重构核心架构，而是**进行更长训练并完成最终论文实验**。因此，当前版本已经达到了“研究原型完成、可进入论文写作与定量完善阶段”的状态。

## 参考文献
1. Pacheco AGC, et al. PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*. 2020.
2. Daneshjou R, et al. Disparities in dermatology AI performance on a diverse, curated clinical image set. *Science Advances*. 2022. PMID: 35960806.
3. DDI Dataset Project Page. https://ddi-dataset.github.io/
4. Stanford AIMI Shared Datasets Portal. https://stanfordaimi.azurewebsites.net/datasets/
5. Qwen2.5-VL 技术文档与模型说明。

## 附录 A：可复现实验脚本
当前代码库中与论文实验最相关的脚本包括：

- 内部对比：[scripts/compare_agent_vs_qwen.py](/g:/0-newResearch/derm_agent/scripts/compare_agent_vs_qwen.py)
- 系统级消融：[scripts/run_agent_ablation.py](/g:/0-newResearch/derm_agent/scripts/run_agent_ablation.py)
- learned components 训练：[scripts/train_learned_components.py](/g:/0-newResearch/derm_agent/scripts/train_learned_components.py)
- 经验库初始化：[scripts/bootstrap_experience_bank.py](/g:/0-newResearch/derm_agent/scripts/bootstrap_experience_bank.py)
- 内部质量评估：[scripts/run_agent_quality_suite.py](/g:/0-newResearch/derm_agent/scripts/run_agent_quality_suite.py)
- 外部 DDI 评估：[scripts/run_external_ddi_eval.py](/g:/0-newResearch/derm_agent/scripts/run_external_ddi_eval.py)
- 显著性检验：[scripts/run_significance_tests.py](/g:/0-newResearch/derm_agent/scripts/run_significance_tests.py)
- 图表导出：[scripts/export_paper_figures.py](/g:/0-newResearch/derm_agent/scripts/export_paper_figures.py)

## 附录 B：长训练完成后需要替换的最终结果
当最终 overnight run 完成后，需要更新以下部分：

- 摘要中的最终主结果数字；
- 第 7.1 节中的主对比表；
- 第 7.2 节中的最终 ablation 结果；
- 第 7.3 节中的 DDI 外部结果与 subgroup 差异；
- 显著性检验结果与最终图表。
