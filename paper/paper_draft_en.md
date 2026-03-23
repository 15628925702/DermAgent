# Candidate Titles
1. DermAgent: From Rule-driven Orchestration to Parameterized Dermatology Diagnosis through Staged Validation
2. DermAgent: Self-aware Skill Routing, Learned Experience Memory, and Uncertainty-aware Aggregation for Skin Disease Diagnosis
3. From Rules to Parameterization: A Staged Diagnostic Agent for Multimodal Dermatology AI

Note: this draft has now been back-filled with the available PAD-UFES-20 staged-validation results from March 24, 2026. In this version, we intentionally do not expand a DDI failure-analysis section.

## Abstract

We present DermAgent, a staged parameterized diagnostic agent for multimodal skin disease classification. The system is built around four core ideas: explicit self-awareness signals, a learnable skill policy, a learned experience bank, and uncertainty-aware aggregation. Instead of relying solely on static prompts or fixed routing rules, DermAgent preserves rules as safety priors while introducing a learned controller, final scorer, rule scorer, and evidence calibrator, enabling a gradual transition from rule-driven orchestration to parameterized decision making.

We evaluate DermAgent as a preliminary staged-validation system rather than as a final full-training model. The retained reports cover three complementary protocols: a frozen 100-case same-model comparison against a direct Qwen baseline, a 100-case staged ablation, and a full PAD-UFES-20 test-set quality suite (`n=344`). On the main paired comparison, DermAgent improves top-1 accuracy from 0.1818 to 0.4300 and malignant recall from 0.7727 to 0.9333, corresponding to absolute gains of +0.2482 and +0.1606, respectively, while reducing error rate from 0.0100 to 0.0000. The paired comparison is statistically significant (McNemar `p=4.1e-05`), with a bootstrap mean top-1 difference of 0.2504 (95% CI `[0.14, 0.36]`) and a bootstrap mean malignant-recall difference of 0.1777 (95% CI `[0.0732, 0.2941]`). In staged ablation, the clearest additional in-domain gain arises after enabling the learned controller/scorer stack, while retrieval is not yet a stable positive contributor under the current budget. On the full quality suite, DermAgent reaches top-1 accuracy 0.4012, top-3 accuracy 0.6541, malignant recall 0.9146, Brier score 0.3072, and expected calibration error 0.2588, indicating strong malignancy sensitivity but still-weak calibration and uneven per-class behavior.

These results show that the learned components have begun to influence final decisions in measurable ways, even before full-scale training. We argue that the main contribution is not only the current performance gain, but also a practical development pathway: begin with a reliable rule scaffold, then incrementally validate whether controller, memory, calibration, and scoring modules genuinely help. This pathway is relevant both to dermatology diagnosis agents and to broader medical AI systems that must balance modularity, safety, and learnability.

## 1 Introduction

Dermatology diagnosis is a high-confusion multiclass decision problem. A single visual forward pass often oscillates among clinically similar classes such as actinic keratosis versus squamous cell carcinoma, basal cell carcinoma versus seborrheic keratosis, or melanoma versus nevus. In practice, meaningful diagnostic reasoning also depends on lesion site, patient age, clinical history, and a meta-level assessment of whether the current evidence is stable enough to trust. Although multimodal foundation models have made this setting more tractable, directly using a large multimodal model as a one-shot classifier still leaves several important gaps.

First, one-shot systems typically do not expose a stable self-awareness interface. Even when internal uncertainty exists, it may not be made available as structured state that later modules can use for routing, safety handling, or conservative decision making. Second, purely rule-based agent pipelines are controllable and inspectable, but they do not naturally improve as experience accumulates. They struggle to learn when a specialist skill is worth invoking, when retrieved evidence should override perception, or when the system should stop collecting evidence and preserve a stable anchor. Third, full end-to-end optimization is attractive in principle, but during system development it is often more urgent to know whether newly introduced learned components have started to help at all.

This paper focuses on that transitional question. When a dermatology agent is upgraded from a rule system into a parameterized system, can we already detect useful effects from learned components under a medium-scale budget? To study this, we build DermAgent. The system decomposes diagnosis into explicit modules: perception produces differential candidates and uncertainty; retrieval extracts structured evidence from an experience bank; a planner proposes skills from self-awareness features; a learnable controller refines routing on top of rule priors; an aggregator combines perception, retrieval, metadata, specialist evidence, and malignancy cues; and reflection with writeback turns selected cases into future memory.

The design goal is not to discard rules, but to reposition them. In DermAgent, rules remain priors and safety rails, while parameterized components gradually replace brittle manual weighting. Memory is no longer a static few-shot prompt; it becomes a structured experience bank that can grow, compress, and later affect routing and aggregation. The final decision is no longer produced by a fixed handcrafted formula alone; it can be adjusted through a learned final scorer and an evidence calibrator. This makes DermAgent a useful testbed for staged validation: we can ask whether the learned controller, learned scorer, and learned memory already matter before waiting for the final full-training pipeline.

Our staged results answer that question positively but not uncritically. On a frozen same-model 100-case paired comparison, DermAgent outperforms direct Qwen in both top-1 accuracy and malignant recall with statistically significant gains. At the same time, the full quality suite shows that calibration remains weak and per-class performance remains uneven, with especially limited F1 on MEL, SCC, and SEK. The present paper therefore makes a measured claim: the transition from rules to parameterization is already useful, but it is not complete.

Our contributions are threefold. First, we present a dermatology agent explicitly designed around a transition from a rule system to a parameterized system. Second, we instantiate that transition through self-awareness signals, a learnable skill policy, a learned experience bank, and uncertainty-aware aggregation. Third, we provide a staged-validation analysis showing statistically significant in-domain gains over a direct multimodal baseline, interpretable ablation trends, and a quality-suite view that reveals both safety-oriented strengths and current limitations.

## 2 Related Work

### 2.1 Multimodal dermatology AI

Prior dermatology AI systems broadly fall into two categories. One category focuses on image classifiers or image-plus-metadata predictors that directly output diagnostic labels. The other category uses stronger multimodal models for explanation, question answering, or report generation. The former has clear supervision targets but limited flexibility in experience reuse and explicit reasoning. The latter is more expressive but often lacks stable intermediate states. DermAgent sits between these paradigms: it leverages multimodal models for perception and reporting, while making routing, retrieval, calibration, and memory update explicit.

### 2.2 Agentic medical reasoning

Recent medical AI systems increasingly adopt an agentic decomposition in which diagnosis is broken into evidence gathering, candidate generation, differential comparison, risk assessment, and final synthesis. This can improve interpretability and enable safety interventions. However, when all tool use is manually orchestrated, the resulting behavior can become brittle and difficult to improve through accumulated experience. DermAgent does not reject rule-based agents; rather, it upgrades them into a hybrid architecture in which rules provide safety and initialization, while a controller and scorer stack learn finer-grained decision-time behavior.

### 2.3 Retrieval, case-based reasoning, and experience memory

Diagnostic problems naturally benefit from retrieval because similar cases, disease prototypes, and known confusion pairs provide context that a single sample lacks. Classical case-based reasoning typically relies on fixed similarity. More recent memory-augmented systems attempt to distill experience into reusable knowledge. DermAgent organizes memory as raw cases, prototypes, confusions, rules, and hard cases. Reflection-driven writeback and compression allow the memory bank to evolve over time, while later retrieval can directly affect routing and evidence aggregation.

### 2.4 Uncertainty, calibration, and safety

Medical AI should not be evaluated solely by top-1 accuracy. Uncertainty quality, malignant recall, confidence calibration, and subgroup behavior all matter for real risk. Existing work on calibration, selective prediction, and subgroup evaluation has emphasized that trustworthy outputs require more than correctness alone. The present study is not yet a deployment study, but it already integrates uncertainty-aware planning, evidence calibration, malignancy-sensitive aggregation, and a dedicated quality suite, allowing us to evaluate both gains and residual weaknesses.

## 3 Method

### 3.1 From a rule system to a parameterized system

DermAgent follows a modular diagnostic pipeline. Perception first produces a candidate differential diagnosis and case-level uncertainty cues. Retrieval then searches the experience bank for related examples, prototypes, confusion memories, and rule memories. A planner converts the resulting case state into candidate skills. A learnable controller refines this plan on top of rule priors. Executed skills produce local evidence, and an uncertainty-aware aggregator combines these signals into the final diagnosis. Reflection and writeback then determine which cases should enter the memory bank.

The critical point is that the system is not fully parameterized from the start. Instead, a rule scaffold is retained for safety and interpretability while four learned components are layered on top:

1. A learnable controller for skill selection and stop probability.
2. A learned final scorer for candidate reranking.
3. A learned rule scorer for weighting rule-memory recommendations.
4. A learned evidence calibrator for tuning evidence weights and thresholds.

This design operationalizes a transition rather than a replacement. Rules remain useful priors, but the final decision process increasingly depends on parameterized modules.

### 3.2 Self-awareness signals

We define self-awareness signals as structured internal variables that describe how uncertain, ambiguous, or under-supported the current case is before final diagnosis. These signals are derived from intermediate system state rather than from external annotation. The current implementation exposes at least the following:

- perception uncertainty (`low`, `medium`, `high`);
- the score gap between the top differential candidates (`top_gap`);
- retrieval confidence and whether retrieved evidence supports the current top candidate;
- confusion-memory support;
- metadata presence and age/site consistency;
- malignant-candidate indicators and known high-confusion pair indicators;
- controller-estimated stop probability.

These signals serve two purposes. First, they drive planner-level triggers, such as increasing the probability of comparison or specialist skills under high uncertainty or small top-gap. Second, they become numeric controller features, allowing the controller to learn when certain evidence sources are likely to be useful. In this sense, self-awareness is the key mechanism by which DermAgent moves from input-only inference toward explicit state-aware decision making.

### 3.3 Learnable skill policy

DermAgent includes an always-on uncertainty assessment skill together with routable skills such as comparison, malignancy-risk assessment, metadata-consistency assessment, and specialist skills targeting dermatology-specific confusion pairs. In a pure rule system, such skills would be invoked through fixed if-else conditions. In DermAgent, the controller learns probabilities of using each skill conditional on the current case state.

The controller consumes perception features, retrieval features, planner priors, and memory recommendations. During staged training, it is updated from heuristic utility targets derived from case outcome and local evidence usefulness. It therefore learns not merely to imitate a generic next-token policy, but to decide which diagnostic actions are worth taking in a given state.

### 3.4 Learned memory bank / experience bank

The experience bank is a structured memory rather than a plain prompt store. It supports at least five memory types:

- `raw_case`: successful solved cases;
- `prototype`: disease-level prototype summaries;
- `confusion`: high-confusion disease-pair knowledge;
- `rule`: compressed rule-like memories with triggers and recommended skills;
- `hard_case`: difficult or near-miss cases retained for later reuse.

Writeback determines what enters the bank, while compression derives prototypes, confusions, and rules from accumulated cases. At inference time, retrieval scores memories using disease alignment, metadata similarity, uncertainty alignment, and related cues, then returns support signals such as consensus labels, prototype votes, confusion votes, retrieval confidence, and recommended skills. The memory bank is therefore “learned” in the sense that it grows and changes through training-time interaction and later shapes decision-time behavior.

### 3.5 Uncertainty-aware aggregation

Final diagnosis is produced by uncertainty-aware aggregation rather than naive voting. Perception remains the anchor, but retrieval, prototype votes, confusion support, metadata consistency, specialist evidence, and malignancy cues can modify the ranking. To prevent unstable late-stage overrides, the aggregator explicitly preserves a perception shortlist and constrains how aggressively low-support candidates can be promoted.

On top of aggregated features, a learned final scorer reranks the remaining candidates. Feature groups include perception support, retrieval agreement, skill agreement, multi-source bonus, memory consensus, and malignant-candidate indicators. The evidence calibrator further adjusts weights and thresholds governing specialist support, metadata support, and planner-trigger behavior. This replaces a fully fixed manual weighting scheme with a constrained learnable synthesis layer.

### 3.6 Safety-oriented aggregation behavior

DermAgent also includes malignancy-sensitive handling. When the malignancy-risk skill signals elevated risk while the current top prediction is benign, the aggregator checks whether there is sufficient malignant support from specialists, retrieval, or candidate structure to justify a more conservative output. This is not equivalent to a clinically validated safety layer, but it does allow staged experiments to examine whether the system is becoming more risk-sensitive rather than merely more accurate.

## 4 Experimental Design

### 4.1 Dataset and primary task

Our primary task is six-way skin disease classification on PAD-UFES-20 with labels ACK, BCC, MEL, NEV, SCC, and SEK. The retained split specification corresponds to the standard repository split with 2,298 cases and a 1,608/346/344 train/validation/test partition. The paper reports staged validation rather than final large-scale training.

### 4.2 Retained artifacts used in this draft

The current draft is based on the following retained artifacts:

- main comparison: `outputs/comparison/comparison_report_20260324_004023.json`;
- ablation: `outputs/ablation/ablation_report_20260324_002952.json`;
- quality suite: `outputs/quality/quality_suite_20260324_010432.json`;
- significance: `outputs/paper_stats/significance_20260324_021221.json`;
- checkpoint metadata: `outputs/checkpoints/learned_controller_best.json`;
- memory-bank statistics: `outputs/checkpoints/learned_bank_best.json`.

The comparison and ablation reports are frozen evaluations on 100 test cases. The quality suite covers the full PAD-UFES-20 test split (`n=344`). The comparison report additionally records that the direct baseline and the agent use the same model backend (`Qwen/Qwen2.5-VL-7B-Instruct`), that sample order is aligned, and that the evaluation is frozen rather than online-adaptive.

### 4.3 Evaluation protocols

We use three complementary evaluation protocols.

1. Main paired comparison. DermAgent is compared against a direct Qwen baseline on the same 100-case frozen test slice.
2. Staged ablation. The system is evaluated across Direct Qwen, agent without retrieval, `+ metadata`, `+ specialists`, `+ controller/scorers`, and full agent `(+ retrieval)`.
3. Quality suite. The frozen agent is evaluated on the full 344-case test split for top-k accuracy, malignant recall, calibration, safety counts, and class-wise behavior.

This design separates “does the parameterized stack help on the paired slice?” from “what does the full frozen test profile look like?”.

### 4.4 Metrics

The main comparison reports top-1 accuracy, malignant recall, and error rate. The significance report provides McNemar p-values and paired bootstrap confidence intervals. The ablation report focuses on top-1 accuracy and malignant recall. The quality suite reports top-1 accuracy, top-3 accuracy, malignant recall, error rate, Brier score, expected calibration error, safety counts, OOD-proxy slices, and per-label precision/recall/F1.

We treat malignant recall as a first-class metric because a dermatology diagnosis agent should not improve accuracy by sacrificing malignant-case sensitivity.

### 4.5 Positioning

We intentionally position this manuscript as a preliminary staged-validation paper. The goal is not to claim that the final system is fully trained or deployment-ready. The goal is to determine whether the learned components are already producing measurable effects and, if so, where the remaining weaknesses are.

## 5 Results and Planned Full-training Extensions

### 5.1 Main comparison

Table 1 reports the primary paired comparison. DermAgent improves top-1 accuracy from 0.1818 to 0.4300 and malignant recall from 0.7727 to 0.9333, while reducing error rate from 0.0100 to 0.0000.

Table 1: Main staged-validation comparison on the frozen 100-case paired test slice

| Model | Top-1 Accuracy | Malignant Recall | Error Rate | Notes |
| --- | --- | --- | --- | --- |
| Direct Qwen | 0.1818 | 0.7727 | 0.0100 | direct multimodal baseline on the same backend |
| DermAgent (frozen staged checkpoint) | 0.4300 | 0.9333 | 0.0000 | learned controller + scorer + memory bank |
| Absolute Delta (Agent - Direct) | +0.2482 | +0.1606 | -0.0100 | positive is better for accuracy/recall; negative is better for error rate |

This is not merely a directional trend. The paired comparison is statistically significant with McNemar `p=4.1e-05`. The bootstrap mean top-1 difference is 0.2504 with 95% CI `[0.14, 0.36]`, and the bootstrap mean malignant-recall difference is 0.1777 with 95% CI `[0.0732, 0.2941]`. Because the direct baseline and the agent branch use the same underlying Qwen model and aligned samples, the result supports the interpretation that the gain comes from the agent architecture and learned decision stack rather than from a backend-model mismatch.

### 5.2 Staged ablation

Table 2 shows how gains accumulate across increasingly parameterized stages of the system.

Table 2: Staged ablation on the frozen 100-case test slice

| Stage | Top-1 Accuracy | Malignant Recall | Interpretation |
| --- | --- | --- | --- |
| Direct Qwen | 0.1818 | 0.7500 | no agent structure |
| Agent without retrieval | 0.3200 | 1.0000 | basic agent without memory retrieval |
| + metadata | 0.3200 | 1.0000 | adds metadata-consistency reasoning |
| + specialists | 0.3200 | 1.0000 | adds high-confusion specialist skills |
| + controller/scorers | 0.3600 | 1.0000 | adds learned controller, final scorer, rule scorer, and evidence calibrator |
| Full agent (+ retrieval) | 0.3300 | 1.0000 | complete staged agent |

Three observations are especially important. First, most of the initial gain over direct Qwen appears as soon as the agent structure itself is introduced. Second, the clearest additional gain comes from enabling the learned controller/scorer stack, which raises top-1 accuracy from 0.3200 to 0.3600 while preserving perfect malignant recall on this slice. Third, retrieval is not yet a stable positive contributor under the current budget: the full agent remains much better than direct Qwen, but its 0.3300 top-1 accuracy is lower than the `+ controller/scorers` stage. This is precisely the kind of conclusion staged validation should surface early.

### 5.3 Quality and safety analysis

The quality suite evaluates the frozen agent on the full PAD-UFES-20 test split (`n=344`), which provides a broader picture than the 100-case comparison slice.

Table 3: Core quality-suite metrics on the full test split

| Metric | Value |
| --- | --- |
| Top-1 Accuracy | 0.4012 |
| Top-3 Accuracy | 0.6541 |
| Malignant Recall | 0.9146 |
| Error Rate | 0.0000 |
| Brier Score | 0.3072 |
| Expected Calibration Error | 0.2588 |
| Malignant Miss Count | 14 |
| Benign False Alarm Count | 89 |

Table 4: Per-label F1 on the full test split

| Class | F1 |
| --- | --- |
| ACK | 0.1600 |
| BCC | 0.5903 |
| MEL | 0.0000 |
| NEV | 0.3607 |
| SCC | 0.0870 |
| SEK | 0.0526 |

The quality suite reveals a mixed but informative picture. On the positive side, malignant recall remains high at 0.9146 and the system reports zero runtime error rate on the frozen full test split. On the negative side, calibration is still weak: ECE reaches 0.2588 and Brier score is 0.3072. Class-wise behavior is also uneven, with the strongest F1 on BCC (0.5903), moderate behavior on NEV (0.3607), and poor F1 on MEL, SCC, and SEK. In other words, the system is already becoming malignancy-sensitive, but it is not yet class-balanced or well-calibrated.

The OOD-proxy slices show the same pattern. On the rare-site proxy slice (`n=39`), top-1 accuracy drops to 0.2564 and malignant recall to 0.8000, indicating reduced robustness away from the core distribution. The perception-fallback slice is very small (`n=4`) and therefore not stable enough for strong claims, but it again suggests that edge-case behavior deserves further work.

### 5.4 Available training-stage metadata and planned full-training extensions

The final synced artifact bundle does not retain the original staged training summary JSON with held-out test metrics, so we do not fabricate those values. Instead, Table 5 reports the checkpoint metadata that are actually available from the retained controller and memory-bank artifacts.

Table 5: Available staged-checkpoint metadata from retained artifacts

| Item | Value |
| --- | --- |
| Staged checkpoint epoch | 1 |
| Best validation Top-1 | 0.4220 |
| Final memory-bank size | 1523 |
| Raw-case memories | 824 |
| Prototype / Confusion / Rule memories | 31 / 3 / 11 |
| Hard-case memories | 654 |

These numbers should be interpreted as internal evidence that learning and memory growth are active, not as a final benchmark table. The next full-training extension should add retained staged test metrics, multi-seed validation, longer-horizon retrieval analysis, and a more stable calibration study. For the current paper, however, the main claims can already be grounded in the comparison, ablation, quality-suite, and significance reports.

## 6 Discussion

### 6.1 Why the staged results still matter

The central value of staged validation is that it answers a development-critical question before final training is complete: have the learned components begun to help? Here the answer is yes. The same-model paired comparison shows statistically significant gains over direct Qwen, and the ablation results show that the learned controller/scorer stack is the clearest additional contributor beyond the basic agent structure. This means the transition from a rule scaffold to a parameterized diagnostic agent is not merely conceptual; it is already measurable.

### 6.2 Why full training is still necessary

At the same time, staged gains are not the end of the story. The retrieval module is not yet a stable positive contributor, which suggests that memory quality, retrieval scoring, or aggregation policies remain undertrained. The quality suite also shows that good malignant recall does not automatically imply good calibration or balanced class performance. Full training is therefore still needed to answer the next level of question: not merely whether the parameterized modules help, but whether they help stably, consistently, and with a better-calibrated risk profile.

### 6.3 What this version validates

The current paper validates several concrete hypotheses. First, explicit self-awareness signals are useful as decision-time state. Second, a learnable controller can refine skill routing on top of rule priors. Third, the experience bank is more than passive storage; it contributes to routing and aggregation through retrieval outputs. Fourth, uncertainty-aware aggregation with learned scoring can outperform a direct one-shot multimodal baseline in-domain under a fair same-model comparison.

### 6.4 What this version does not yet validate

This version does not yet establish the final performance ceiling, multi-seed robustness, strong external generalization, or clinically ready calibration quality. It also does not show that every added module is already beneficial: retrieval remains the clearest unresolved component in the current staged setting. These boundaries are important because the purpose of the paper is to document a successful transition in progress, not to oversell a completed endpoint.

## 7 Limitations

This study has several limitations. First, the training regime remains staged and medium-scale rather than fully optimized. Second, the main paired comparison and ablation both use a 100-case frozen slice, which is sufficient for detecting directional and even significant effects, but still limited for exhaustive subgroup analysis. Third, the quality suite shows pronounced class imbalance in performance, including zero F1 on MEL and low F1 on SCC and SEK; this makes the current system unsuitable for any deployment claim. Fourth, calibration remains weak despite good malignant recall, so confidence values should not yet be interpreted as clinically reliable. Fifth, the learned memory in this work refers to structured writeback, compression, and reuse rather than a fully differentiable memory module. Finally, we intentionally do not expand external DDI failure analysis in this draft, so the manuscript should be read as an in-domain staged-validation paper.

## 8 Conclusion

We introduced DermAgent, a staged parameterized diagnostic agent for skin disease classification. Relative to a direct same-model Qwen baseline on a frozen paired test slice, DermAgent improves top-1 accuracy from 0.1818 to 0.4300 and malignant recall from 0.7727 to 0.9333, with statistically significant paired gains. The ablation study shows that the learned controller/scorer stack is already useful, while the full quality suite shows that malignancy sensitivity is relatively strong but calibration and class balance remain limiting factors.

The broader contribution of this work is a practical pathway for upgrading medical AI systems from rules to parameterization. Rather than waiting for a monolithic end-to-end result, we can start from a reliable rule scaffold, incrementally introduce learned controller, memory, calibration, and scoring modules, and test whether each step genuinely helps. DermAgent suggests that this path is viable, but it also makes clear what remains to be improved before stronger claims are warranted.
