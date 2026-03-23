# DermAgent: An Experience-Grounded Agentic Framework for Dermatology Image Triage and Differential Diagnosis

## Abstract
Large vision-language models can perform direct dermatology image inference, but naive single-shot prompting often underuses clinical metadata, lacks explicit uncertainty handling, and provides little mechanism for iterative experience accumulation. We present **DermAgent**, an agentic dermatology framework that combines a shared vision-language backbone with modular perception, retrieval, metadata-consistency checking, specialist reasoning, learned aggregation, and an experience bank that is updated offline and online. The framework is designed to improve differential diagnosis quality while preserving auditability through structured intermediate outputs.

On a frozen, same-model comparison setting using PAD-UFES-20, DermAgent improves top-1 accuracy from **0.1717** for direct Qwen inference to **0.3700**, and malignant-case recall from **0.7045** to **0.9556** on a 100-case benchmark slice, with matched samples and identical base model selection in both branches. A system-level ablation shows that retrieval is the strongest contributor, metadata is beneficial after calibration, and the full agent remains stronger than any reduced variant. We also build an external evaluation pipeline for the DDI dataset and show that DDI is suitable primarily as an external binary robustness and fairness benchmark rather than a full six-class training source under our ontology. Taken together, these results suggest that agentic decomposition plus experience-grounded retrieval can substantially outperform direct prompting under controlled settings, while exposing clear avenues for further improvement in learned calibration and external robustness.

## 1. Introduction
Dermatology is a natural setting for multimodal medical AI: diagnosis depends not only on lesion appearance, but also on metadata such as age, lesion site, and clinical evolution. Recent multimodal foundation models can generate plausible dermatology judgments from single images, yet a direct one-shot inference path has several practical limitations. First, it entangles perception, differential diagnosis, and reporting inside one opaque generation. Second, it offers limited control over specialist reasoning and uncertainty-driven escalation. Third, it lacks a persistent mechanism for accumulating task-specific experience over time.

This work asks whether an **agentic dermatology system**, built on top of the same underlying vision-language model used for direct inference, can improve performance through explicit decomposition rather than stronger base modeling alone. To answer this question, we build **DermAgent**, a modular framework with:

- a perception module that extracts structured differential candidates and visual cues;
- a retrieval layer over an experience bank containing raw cases, hard cases, prototypes, confusion memories, and lightweight rules;
- metadata-consistency and specialist skills that produce explicit supporting and penalizing evidence;
- a learned controller and scorer stack that calibrates which skills to use and how to weight their outputs;
- a reporting module that converts structured evidence into clinician-facing summaries.

The system is designed for research rather than clinical deployment. Our central claim is not that DermAgent is clinically ready, but that **agentic structure plus learned and memory-based calibration can outperform direct prompting under a controlled same-model comparison**.

## 2. Contributions
This paper makes four contributions.

1. We introduce an end-to-end dermatology agent framework that separates perception, retrieval, specialist reasoning, aggregation, and reporting while sharing the same underlying vision-language model as a direct baseline.
2. We implement a learnable memory-and-control pipeline including offline bank bootstrapping, learnable controller and scorer checkpoints, evidence calibration, and rule/prototype/confusion compression.
3. We establish a **fair internal comparison protocol** on PAD-UFES-20 with matched samples, identical backbone model names, and frozen benchmarking semantics, preventing the confounds that often affect agent-vs-baseline comparisons.
4. We extend the system with an external DDI evaluation path and show how to use DDI appropriately as a binary external benchmark for robustness and subgroup analysis.

## 3. Related Work
Dermatology AI has a long history of image classification and dermoscopic analysis, with public datasets such as ISIC and PAD-UFES-20 enabling lesion recognition research. PAD-UFES-20 is especially relevant here because it contains smartphone clinical images plus patient metadata, making it well suited to multimodal reasoning. In parallel, recent multimodal large language models and vision-language models have shown strong zero-shot or few-shot capabilities on medical image understanding tasks, but often remain difficult to calibrate and audit in deployment-like workflows.

Our work is most closely related to three threads:

- **Direct multimodal diagnosis** using a single prompt over image plus metadata.
- **Tool-using and agentic medical systems**, where models can invoke specialized modules or staged reasoning pipelines.
- **Memory-augmented and retrieval-augmented clinical AI**, where previous cases or prototypes are reused to stabilize decisions.

DermAgent differs from prior one-shot systems in that it enforces explicit intermediate states and separates perception from final decision-making. It also differs from static rule-based pipelines by allowing learned control and scoring to adapt which evidence is emphasized.

## 4. Task Definition
We study lesion classification and triage under a six-label ontology used throughout the current system:

- `MEL`
- `BCC`
- `SCC`
- `NEV`
- `ACK`
- `SEK`

For safety-oriented reporting, we additionally collapse predictions into a binary malignant/benign split:

- malignant: `MEL`, `BCC`, `SCC`
- benign/non-malignant: `NEV`, `ACK`, `SEK`

The primary internal task is six-class prediction on PAD-UFES-20. The primary external task on DDI is binary malignant-vs-benign assessment, because many DDI disease categories do not map cleanly into the six-label ontology.

## 5. Method
### 5.1 Overview
Given an input case containing an image and available metadata, DermAgent executes the following stages:

1. **Perception**: a vision-language prompt produces structured differential candidates, visual cues, malignant cues, and uncertainty.
2. **Retrieval**: the system queries an experience bank for raw prior cases, prototypes, confusion memories, and rules.
3. **Planning**: a planner determines which downstream skills to invoke, optionally informed by a learnable controller and evidence calibrator.
4. **Skill execution**: metadata-consistency checks, comparison skills, and pairwise specialists provide structured support or penalties for candidate labels.
5. **Aggregation**: all evidence is combined by a decision aggregator, optionally using learned final scoring.
6. **Reporting**: a reporting model turns the structured decision into a concise clinical-style output.
7. **Learning and memory update**: during training or warm-start workflows, the system can update learned components and compress new experiences back into memory.

### 5.2 Perception Module
The perception stage uses a vision-language model to emit structured JSON rather than free-form text. This output includes differential candidates, confidence proxies, visual cues, risk cues, and uncertainty. A key design decision in the current version is that the perception prompt uses **neutral placeholder labels** rather than anchoring examples such as SCC/ACK/BCC, which previously biased the model toward keratinocytic malignancies.

### 5.3 Experience Bank
The experience bank stores multiple forms of memory:

- **raw cases**, preserving concrete prior examples;
- **hard cases**, preserving failures and edge cases;
- **prototypes**, representing compressed class-level or subgroup-level exemplars;
- **confusion memories**, representing systematic near-miss pairs;
- **rules**, lightweight structured recommendations distilled from repeated patterns.

An offline bootstrap pass scans the training split and builds an initial bank before final evaluation. In a representative run over the PAD-UFES-20 train split (1608 cases, 2 passes), the bank produced **1023** total entries: **566 raw cases**, **440 hard cases**, **4 prototypes**, **3 confusion memories**, and **10 rules**.

### 5.4 Metadata and Specialist Reasoning
Metadata-consistency is treated as evidence, not as a hard override. The current system emits explicit support and penalty strengths rather than simple yes/no gates. This allows the evidence calibrator to learn how strongly to encourage benign rescue in pediatric or low-risk contexts, or how strongly to penalize implausible malignancies given age and site.

Specialist modules cover high-confusion label pairs:

- ACK vs SCC
- BCC vs SCC
- BCC vs SEK
- MEL vs NEV

Earlier versions over-triggered specialists and over-counted overlapping votes. The current version tightens specialist activation and dampens duplicate support for the same label.

### 5.5 Learned Control and Aggregation
DermAgent includes several learned components:

- a **learnable skill controller** for stop/use decisions;
- a **learnable final scorer** for evidence-weighted label aggregation;
- a **learnable rule scorer** for adjusting rule utility;
- a **learnable evidence calibrator** for reweighting metadata and other evidence sources.

These components can be trained from train-split experience while keeping the final compare/evaluation path frozen.

### 5.6 Fairness of the Agent-vs-Baseline Comparison
A major engineering contribution of this work is the experimental protocol itself. The agent branch and direct baseline now:

- operate on the **same exact case list**;
- use the **same underlying model name**;
- run in a **frozen evaluation mode** without online learning during compare;
- expose fallback counts and model-usage diagnostics.

This matters because apparent gains from agentic methods can otherwise be artifacts of sample mismatch, fallback behavior, or hidden model-name differences.

## 6. Experimental Setup
### 6.1 Internal Dataset: PAD-UFES-20
We use PAD-UFES-20 as the main internal benchmark. A fixed split manifest with seed 42 yields:

- train: **1608**
- val: **346**
- test: **344**
- total: **2298**

The split infrastructure is persisted to disk so that all training, ablation, and evaluation runs share the same split definition.

### 6.2 External Dataset: DDI
We also prepare the Diverse Dermatology Images (DDI) dataset as an external benchmark. The local DDI integration currently loads **656** cases with complete image-path resolution. Under the current six-class ontology:

- **230** samples map directly into the six-label space;
- **426** samples do not map cleanly and are therefore used only for binary malignant-vs-benign evaluation.

This is an important methodological point: DDI is valuable for **external robustness and fairness analysis**, but in this project it is not a suitable replacement for the internal six-class training objective.

### 6.3 Baseline
The baseline is a **direct Qwen vision-language inference** path that receives the same image and metadata but does not use retrieval, explicit specialist routing, or structured aggregation. The purpose of the comparison is to isolate the value of the agent architecture, not to compare different backbone models.

### 6.4 Metrics
We report:

- top-1 accuracy;
- top-3 accuracy where applicable;
- malignant-case recall;
- error rate;
- subgroup metrics;
- calibration metrics such as expected calibration error (ECE);
- paired significance where comparison outputs are available.

### 6.5 Training Protocol
The current workflow uses a two-stage training regime:

1. **offline bank bootstrap** over the train split;
2. **learned controller/scorer training** initialized from the bootstrapped bank.

Final comparison and paper-facing evaluation are then run in **frozen mode**.

## 7. Results
### 7.1 Controlled Internal Comparison on PAD-UFES-20
In the current frozen 100-case internal benchmark slice, DermAgent outperforms direct Qwen inference under same-model and sample-aligned conditions.

| Method | Top-1 Accuracy | Malignant Recall | Error Rate |
|---|---:|---:|---:|
| Direct Qwen | 0.1717 | 0.7045 | 0.0100 |
| DermAgent | 0.3700 | 0.9556 | 0.0000 |

This corresponds to a relative top-1 improvement of approximately **115.5%** and a malignant-recall improvement of **35.6%** on this benchmark slice.

These gains are meaningful because the comparison was explicitly corrected to avoid three common confounds:

- mismatched evaluation samples;
- different model names between agent and direct branches;
- hidden fallback usage masquerading as faster inference.

### 7.2 System-Level Ablation
A 100-case ablation using a smoke-trained checkpoint yields the following pattern:

| Variant | Top-1 Accuracy | Malignant Recall | Error Rate |
|---|---:|---:|---:|
| Direct Qwen | 0.1717 | 0.7500 | 0.0100 |
| Agent without retrieval | 0.3000 | 1.0000 | 0.0000 |
| + metadata | 0.3300 | 1.0000 | 0.0000 |
| + specialists | 0.3000 | 1.0000 | 0.0000 |
| + controller/scorers | 0.2600 | 0.9778 | 0.0000 |
| Full agent (+ retrieval) | 0.3700 | 1.0000 | 0.0000 |

This ablation suggests:

- **retrieval is the strongest single contributor**;
- metadata becomes helpful once calibrated properly;
- specialists help less consistently and remain a target for further refinement;
- controller/scorer learning is promising but undertrained in short smoke settings;
- the **full system remains strongest** once all components are combined.

### 7.3 External Evaluation on DDI
The DDI integration is complete and can now be evaluated through the same paper pipeline. However, because the currently available local DDI run is only a short smoke check and the long-run final checkpoint has not yet been applied in this workspace, we treat external DDI numbers as **pending final long-training evaluation**.

What is already established is the evaluation protocol:

- all 656 DDI images load successfully;
- external binary malignant/benign evaluation is supported end-to-end;
- subgroup summaries can be produced over available skin-tone metadata;
- DDI is treated as an **external robustness benchmark**, not as a six-class primary training source.

### 7.4 Interpretation
The results indicate that structured agentic decomposition adds value even when the base model is held fixed. The dominant gain appears to come from combining:

- better-structured perception;
- retrieval over prior experience;
- explicit evidence aggregation.

The learned controller stack is not yet the main source of gain, but the system is now engineered so that longer training can improve it without changing the core evaluation protocol.

## 8. Discussion
### 8.1 Why the Agent Helps
The strongest evidence from the current runs is that direct prompting leaves significant value on the table. A structured agent can:

- keep benign alternatives active when metadata supports them;
- invoke specialists only when confusion is plausible;
- use retrieved cases and prototypes to stabilize difficult decisions;
- produce auditable intermediate evidence.

This is especially important in medical settings, where a single free-form model output is often insufficient for error analysis.

### 8.2 Why Retrieval Matters Most
The ablation indicates that retrieval is the most important module in the current system. This is consistent with the design of dermatology diagnosis, where many difficult cases are resolved by comparison to similar prior examples or by remembering systematic confusion patterns. The experience bank provides a mechanism for task-specific adaptation without retraining the base vision-language model itself.

### 8.3 Why DDI Should Not Be the Main Training Set Here
Although DDI is extremely valuable, it is best used here as an **external test set**:

- it is relatively small;
- its label space does not fully match the six-class PAD-UFES-20 ontology;
- its strength is diversity and fairness-oriented evaluation rather than large-scale supervision.

This project therefore treats DDI as an external check on robustness rather than a replacement for the internal training source.

## 9. Limitations
This study has several limitations.

1. The strongest current quantitative evidence is still on a 100-case controlled internal slice rather than the final full long-run benchmark.
2. External DDI evaluation infrastructure is complete, but final external results remain to be filled in after long-run training.
3. Some specialist and planner behaviors remain partially rule-informed rather than fully learned.
4. The learned controller and scorer components likely require longer training and multi-seed evaluation before their contribution can be estimated precisely.
5. The system is a research prototype and is **not** validated for clinical deployment.

## 10. Ethical and Safety Considerations
DermAgent is intended for research on assistive diagnostic reasoning, not for autonomous clinical use. Any medical deployment would require:

- prospective validation;
- clinician oversight;
- calibration and subgroup fairness analysis;
- out-of-distribution detection;
- formal safety review;
- compliance with regulatory and data governance requirements.

We explicitly avoid presenting this system as a clinical diagnostic device. The goal is to study whether structured agentic reasoning can improve the reliability and interpretability of multimodal dermatology AI.

## 11. Conclusion
We presented DermAgent, an experience-grounded agentic dermatology framework built around the same vision-language backbone as a direct baseline. By combining structured perception, retrieval, metadata-consistency reasoning, specialist skills, learned aggregation, and an explicit memory system, DermAgent substantially improves performance in a frozen same-model comparison on PAD-UFES-20. The system now also supports external DDI evaluation for robustness and subgroup analysis.

The main engineering work is complete: the remaining step is to run longer training and finalize the paper-facing experiments. This makes the current system a strong **research prototype and paper-ready baseline**, with a clear path toward stronger empirical validation in the next stage.

## References
1. Pacheco AGC, et al. PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*. 2020.
2. Daneshjou R, et al. Disparities in dermatology AI performance on a diverse, curated clinical image set. *Science Advances*. 2022. PMID: 35960806.
3. DDI Dataset Project Page. https://ddi-dataset.github.io/
4. Stanford AIMI Shared Datasets Portal. https://stanfordaimi.azurewebsites.net/datasets/
5. Qwen2.5-VL technical and model documentation.

## Appendix A. Reproducibility Notes
The codebase includes the following relevant scripts:

- internal compare: [scripts/compare_agent_vs_qwen.py](/g:/0-newResearch/derm_agent/scripts/compare_agent_vs_qwen.py)
- ablation: [scripts/run_agent_ablation.py](/g:/0-newResearch/derm_agent/scripts/run_agent_ablation.py)
- learned-component training: [scripts/train_learned_components.py](/g:/0-newResearch/derm_agent/scripts/train_learned_components.py)
- bank bootstrap: [scripts/bootstrap_experience_bank.py](/g:/0-newResearch/derm_agent/scripts/bootstrap_experience_bank.py)
- internal quality suite: [scripts/run_agent_quality_suite.py](/g:/0-newResearch/derm_agent/scripts/run_agent_quality_suite.py)
- external DDI eval: [scripts/run_external_ddi_eval.py](/g:/0-newResearch/derm_agent/scripts/run_external_ddi_eval.py)
- significance tests: [scripts/run_significance_tests.py](/g:/0-newResearch/derm_agent/scripts/run_significance_tests.py)
- figure export: [scripts/export_paper_figures.py](/g:/0-newResearch/derm_agent/scripts/export_paper_figures.py)

## Appendix B. Final Numbers to Replace After Long Training
When the final overnight run is complete, update the following sections with the final checkpoint outputs:

- Table 1: replace the 100-case controlled compare numbers if a full test-split compare is used in the final paper.
- Table 2: replace the smoke-checkpoint ablation with the final trained-checkpoint ablation.
- Section 7.3: insert the final DDI external binary results and subgroup gaps.
- Add statistical significance values from the final paper evidence suite.
