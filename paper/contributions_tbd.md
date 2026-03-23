# Contribution Options

This file provides three contribution variants aligned with the currently filled PAD-UFES-20 results. None of the variants below expands DDI-specific problem analysis.

## A Version: Conservative

1. We present DermAgent, a dermatology diagnostic agent that explicitly transitions from a rule-driven pipeline to a parameterized system through self-awareness signals, a learnable skill policy, a learned experience bank, and uncertainty-aware aggregation.
2. We introduce a staged-validation protocol that allows learned controller, scorer, and memory components to be evaluated before full-scale training is complete.
3. On PAD-UFES-20, DermAgent shows statistically supported in-domain gains over a direct same-model Qwen baseline, while the quality suite also reveals the remaining calibration and per-class limitations of the current stage.

## B Version: Standard

1. We propose DermAgent, a hybrid medical AI agent that upgrades a controllable rule scaffold into a parameterized diagnostic system by adding explicit self-awareness state, learnable skill routing, structured experience memory, and learned evidence aggregation.
2. We show that this transition is already measurable under staged validation: on a frozen same-model 100-case paired comparison, DermAgent improves top-1 accuracy from 0.1818 to 0.4300 and malignant recall from 0.7727 to 0.9333, with statistically significant paired gains.
3. We provide a structured analysis beyond headline accuracy, including staged ablation, calibration, safety counts, and per-class quality profiling, showing that the learned controller/scorer stack is already helpful while calibration and class balance remain open challenges.

## C Version: Stronger Wording If You Want a More Assertive Abstract

Use this only if you are comfortable with a stronger research framing and keep the limitation section explicit.

1. We demonstrate that a dermatology diagnosis system can be upgraded from static rule orchestration to parameterized agent behavior without waiting for monolithic end-to-end training, using self-aware routing, learned memory, and uncertainty-aware aggregation.
2. We establish statistically significant in-domain gains over a direct same-model multimodal baseline, indicating that the learned components are already contributing materially to decision quality in the current staged regime.
3. We show that the benefit is mechanistically interpretable rather than purely empirical: ablation localizes the clearest incremental gain to the learned controller/scorer stack, while quality-suite analysis exposes the exact calibration and class-wise weaknesses that remain for the next training stage.

## Recommended Choice

- Use Version A if you want the safest framing.
- Use Version B for most conference-style drafts.
- Use Version C only if the rest of the paper keeps the caveats visible and does not overstate deployment readiness.
