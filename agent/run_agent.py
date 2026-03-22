from __future__ import annotations

from typing import Any, Dict

from agent.aggregator import DecisionAggregator
from agent.controller import LearnableSkillController
from agent.evidence_calibrator import LearnableEvidenceCalibrator
from agent.final_scorer import LearnableFinalScorer
from agent.planner import ExperienceSkillPlanner
from agent.reflection import ReflectionEngine
from agent.registry import build_skill_registry
from agent.router import SkillRouter
from agent.rule_scorer import LearnableRuleScorer
from agent.state import create_case_state
from memory.compressor import ExperienceCompressor
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import SkillIndex, build_default_skill_index
from memory.writer import ExperienceWriter
from memory.weights_manager import weights_manager


USE_RETRIEVAL = True
USE_SPECIALIST = True
USE_REFLECTION = True
USE_CONTROLLER = False
USE_COMPARE = True
USE_MALIGNANCY = False
USE_METADATA_CONSISTENCY = True
USE_FINAL_SCORER = False


def run_agent(
    case: Dict[str, Any],
    bank: ExperienceBank | None = None,
    skill_index: SkillIndex | None = None,
    reranker: UtilityAwareExperienceReranker | None = None,
    learning_components: Dict[str, Any] | None = None,
    *,
    use_retrieval: bool = USE_RETRIEVAL,
    use_specialist: bool = USE_SPECIALIST,
    use_reflection: bool = USE_REFLECTION,
    use_controller: bool = USE_CONTROLLER,
    use_compare: bool = USE_COMPARE,
    use_malignancy: bool = USE_MALIGNANCY,
    use_metadata_consistency: bool = USE_METADATA_CONSISTENCY,
    use_final_scorer: bool = USE_FINAL_SCORER,
    update_online: bool = True,
    use_rule_memory: bool = True,
    enable_rule_compression: bool = True,
    update_rule_scorer: bool | None = None,
    perception_model: str | None = None,
    report_model: str | None = None,
) -> Dict[str, Any]:
    if bank is None:
        bank = ExperienceBank()
    if skill_index is None:
        skill_index = build_default_skill_index()

    # 初始化学习组件
    if learning_components is None:
        learning_components = weights_manager.initialize_components(skill_index)

    # 提取各个组件
    controller = learning_components.get("controller")
    final_scorer = learning_components.get("final_scorer")
    rule_scorer = learning_components.get("rule_scorer")
    retrieval_scorer = learning_components.get("retrieval_scorer")
    evidence_calibrator = learning_components.get("evidence_calibrator")

    # 如果需要，使用学习组件
    if use_controller and controller is None:
        controller = learning_components.get("controller")
    if use_controller and use_final_scorer and final_scorer is None:
        final_scorer = learning_components.get("final_scorer")
    if use_controller and rule_scorer is None:
        rule_scorer = learning_components.get("rule_scorer")

    state = create_case_state(case)
    runtime_flags = {
        "use_retrieval": use_retrieval,
        "use_specialist": use_specialist,
        "use_reflection": use_reflection,
        "use_controller": use_controller,
        "use_compare": use_compare,
        "use_malignancy": use_malignancy,
        "use_metadata_consistency": use_metadata_consistency,
        "use_final_scorer": use_final_scorer,
        "update_online": update_online,
        "use_rule_memory": use_rule_memory,
        "enable_rule_compression": enable_rule_compression,
        "perception_model": perception_model or "",
        "report_model": report_model or "",
    }
    state.trace("config", "success", "Runtime flags initialized", payload=runtime_flags)

    try:
        disable_skills = []
        if not use_compare:
            disable_skills.append("compare_skill")
        if not use_malignancy:
            disable_skills.append("malignancy_risk_skill")
        if not use_metadata_consistency:
            disable_skills.append("metadata_consistency_skill")
        if not use_specialist:
            disable_skills.extend(
                [
                    "ack_scc_specialist_skill",
                    "bcc_scc_specialist_skill",
                    "bcc_sek_specialist_skill",
                    "mel_nev_specialist_skill",
                ]
            )
        registry = build_skill_registry(
            bank,
            skill_index=skill_index,
            reranker=reranker,
            retrieval_scorer=retrieval_scorer,
            config={
                "disable_skills": disable_skills,
                "perception_model": perception_model,
                "report_model": report_model,
            },
        )
        state.trace("registry", "success", "Skill registry built")
    except Exception as e:
        state.trace("registry", "failed", f"Failed to build registry: {e}")
        return _export_state(state, error=f"Failed to build registry: {e}", runtime_flags=runtime_flags)

    try:
        registry["perception_skill"].run(state)
    except Exception as e:
        state.trace("perception", "failed", f"Perception skill failed: {e}")
        return _export_state(state, error=f"Perception skill failed: {e}", runtime_flags=runtime_flags)

    if use_retrieval:
        try:
            registry["retrieval_skill"].run(state)
            if not use_rule_memory:
                _strip_rule_memory_from_state(state)
                state.trace("rule_memory", "skipped", "Rule memory disabled by runtime flag")
        except Exception as e:
            state.trace("retrieval", "failed", f"Retrieval skill failed: {e}")
            return _export_state(state, error=f"Retrieval skill failed: {e}", runtime_flags=runtime_flags)
    else:
        state.retrieval = _empty_retrieval_result(reason="retrieval_disabled")
        state.trace("retrieval", "skipped", "Retrieval disabled by runtime flag")

    try:
        planner = ExperienceSkillPlanner(
            use_specialist=use_specialist,
            controller=controller if use_controller else None,
            evidence_calibrator=evidence_calibrator,
            rule_scorer=rule_scorer if use_controller else None,
            planning_mode="learnable_hybrid" if use_controller else "rules_only",
            enabled_skills=set(registry.keys()),
        )
        planner.plan(state)
    except Exception as e:
        state.trace("planner", "failed", f"Planner failed: {e}")
        return _export_state(state, error=f"Planner failed: {e}", runtime_flags=runtime_flags)

    try:
        router = SkillRouter(registry)
        router.execute(state)
    except Exception as e:
        state.trace("router", "failed", f"Router execution failed: {e}")
        return _export_state(state, error=f"Router execution failed: {e}", runtime_flags=runtime_flags)

    try:
        aggregator = DecisionAggregator(
            final_scorer=final_scorer if use_controller and use_final_scorer else None,
            evidence_calibrator=evidence_calibrator,
        )
        aggregator.aggregate(state)
    except Exception as e:
        state.trace("aggregator", "failed", f"Aggregator failed: {e}")
        return _export_state(state, error=f"Aggregator failed: {e}", runtime_flags=runtime_flags)

    try:
        registry["report_skill"].run(state)
    except Exception as e:
        state.trace("report", "failed", f"Report skill failed: {e}")

    if use_reflection:
        try:
            reflection = ReflectionEngine()
            reflection.summarize(state)
        except Exception as e:
            state.trace("reflection", "failed", f"Reflection failed: {e}")
            state.reflection = {}
    else:
        state.reflection = {"status": "skipped", "reason": "reflection_disabled"}
        state.trace("reflection", "skipped", "Reflection disabled by runtime flag")

    if use_reflection and update_online:
        try:
            writer = ExperienceWriter()
            writeback_summary = writer.write_case(state=state, bank=bank, auto=True)
            state.reflection.setdefault("memory_update", {})["writeback"] = writeback_summary
        except Exception as e:
            state.trace("writeback", "failed", f"Experience writeback failed: {e}")

        try:
            compressor = ExperienceCompressor()
            compression_summary = compressor.compress(bank, include_rules=enable_rule_compression)
            state.reflection.setdefault("memory_update", {})["compression"] = compression_summary
            state.trace(
                "compression",
                "success",
                "Experience compression completed",
                payload={
                    "prototype_count": compression_summary.get("prototype_count", 0),
                    "confusion_count": compression_summary.get("confusion_count", 0),
                    "rule_count": compression_summary.get("rule_count", 0),
                },
            )
        except Exception as e:
            state.trace("compression", "failed", f"Experience compression failed: {e}")

        # 更新检索器的学习参数
        try:
            if retrieval_scorer:
                # 从检索结果中提取有用性反馈
                retrieved_cases = state.retrieval.get("raw_case_hits", []) or []
                was_helpful = _evaluate_retrieval_helpfulness(state)
                retrieval_scorer.update_from_feedback(retrieved_cases, was_helpful)
                state.trace("retriever_update", "success", "Retriever parameters updated from feedback")
        except Exception as e:
            state.trace("retriever_update", "failed", f"Retriever update failed: {e}")
    else:
        state.trace("writeback", "skipped", "Writeback skipped because online updates are disabled or reflection is off")
        state.trace("compression", "skipped", "Compression skipped because online updates are disabled or reflection is off")

    if use_controller and controller is not None and update_online:
        try:
            learning_feedback = controller.update_from_case(state)
            state.controller["learning_feedback"] = learning_feedback
            state.trace(
                "controller",
                "success",
                "Controller parameters updated",
                payload={
                    "stop_target": learning_feedback.get("stop_target"),
                    "stop_prediction": learning_feedback.get("stop_prediction"),
                },
            )
        except Exception as e:
            state.trace("controller", "failed", f"Controller update failed: {e}")
    elif use_controller:
        state.trace("controller", "skipped", "Controller update skipped because online updates are disabled")

    if use_controller and use_final_scorer and final_scorer is not None and update_online:
        try:
            scorer_feedback = final_scorer.update_from_case(state)
            state.controller["final_scorer_feedback"] = scorer_feedback
            state.trace(
                "final_scorer",
                "success",
                "Final scorer updated",
                payload={
                    "updated": scorer_feedback.get("updated"),
                    "margin": scorer_feedback.get("margin"),
                    "true_label": scorer_feedback.get("true_label"),
                    "predicted_label": scorer_feedback.get("predicted_label"),
                },
            )
        except Exception as e:
            state.trace("final_scorer", "failed", f"Final scorer update failed: {e}")
    elif use_controller and use_final_scorer:
        state.trace("final_scorer", "skipped", "Final scorer update skipped because online updates are disabled")

    if evidence_calibrator is not None and update_online:
        try:
            calibrator_feedback = evidence_calibrator.update_from_case(state)
            state.controller["evidence_calibrator_feedback"] = calibrator_feedback
            state.trace(
                "evidence_calibrator",
                "success",
                "Evidence calibration updated",
                payload={
                    "updated": calibrator_feedback.get("updated"),
                    "true_label": calibrator_feedback.get("true_label"),
                    "predicted_label": calibrator_feedback.get("predicted_label"),
                },
            )
        except Exception as e:
            state.trace("evidence_calibrator", "failed", f"Evidence calibration update failed: {e}")
    elif evidence_calibrator is not None:
        state.trace("evidence_calibrator", "skipped", "Evidence calibration update skipped because online updates are disabled")

    effective_rule_update = update_online if update_rule_scorer is None else bool(update_rule_scorer)

    if use_controller and rule_scorer is not None and effective_rule_update:
        try:
            rule_feedback = rule_scorer.update_from_case(state)
            state.controller["rule_scorer_feedback"] = rule_feedback
            state.trace(
                "rule_scorer",
                "success",
                "Rule scorer updated",
                payload={
                    "updated": rule_feedback.get("updated"),
                    "num_updated_rules": len(rule_feedback.get("updated_rules", [])),
                },
            )
        except Exception as e:
            state.trace("rule_scorer", "failed", f"Rule scorer update failed: {e}")
    elif use_controller:
        state.trace("rule_scorer", "skipped", "Rule scorer update skipped because rule updates are disabled")

    return _export_state(state, runtime_flags=runtime_flags)


def _export_state(state: Any, error: str | None = None, runtime_flags: Dict[str, Any] | None = None) -> Dict[str, Any]:
    result = {
        "case_info": state.case_info,
        "perception": state.perception,
        "retrieval": state.retrieval,
        "controller": state.controller,
        "planner": state.planner,
        "selected_skills": state.selected_skills,
        "skill_outputs": state.skill_outputs,
        "final_decision": state.final_decision,
        "report": state.report,
        "reflection": state.reflection,
        "trace": state.execution_trace,
    }
    if runtime_flags is not None:
        result["runtime_flags"] = runtime_flags
    if error:
        result["error"] = error
    return result


def _empty_retrieval_result(reason: str) -> Dict[str, Any]:
    return {
        "top_k": 0,
        "raw_case_hits": [],
        "prototype_hits": [],
        "confusion_hits": [],
        "rule_hits": [],
        "retrieval_summary": {
            "support_labels": [],
            "prototype_votes": {},
            "confusion_votes": {},
            "memory_consensus_label": "",
            "retrieval_confidence": "low",
            "has_confusion_support": False,
            "confusion_pairs": [],
            "memory_recommended_skills": [],
            "rule_recommended_skills": [],
            "recommended_skills": [],
            "support_strength": {"raw_case": 0, "prototype": 0, "confusion": 0, "rule": 0},
            "top_support_case_ids": [],
            "supports_top1": False,
            "status": "skipped",
            "reason": reason,
        },
    }


def _strip_rule_memory_from_state(state: Any) -> None:
    retrieval = dict(state.retrieval or {})
    summary = dict(retrieval.get("retrieval_summary", {}) or {})
    memory_recommended = [
        str(x).strip()
        for x in summary.get("memory_recommended_skills", []) or []
        if str(x).strip()
    ]
    summary["rule_recommended_skills"] = []
    summary["recommended_skills"] = memory_recommended
    support_strength = dict(summary.get("support_strength", {}) or {})
    support_strength["rule"] = 0
    summary["support_strength"] = support_strength
    retrieval["rule_hits"] = []
    retrieval["retrieval_summary"] = summary
    state.retrieval = retrieval


def _evaluate_retrieval_helpfulness(state: Any) -> bool:
    """评估检索结果是否有帮助"""
    retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}

    # 如果检索提供了技能推荐，且这些技能被使用了，认为是helpful
    recommended_skills = set(retrieval_summary.get("recommended_skills", []))
    selected_skills = set(state.selected_skills or [])
    skill_overlap = len(recommended_skills & selected_skills)

    # 如果检索置信度高且最终决策正确，认为是helpful
    retrieval_confidence = retrieval_summary.get("retrieval_confidence", "low")
    final_decision = state.final_decision or {}
    true_label = str((state.case_info or {}).get("true_label", "")).strip().upper()
    final_label = str(final_decision.get("final_label") or final_decision.get("diagnosis") or "").strip().upper()
    is_correct = bool(true_label) and final_label == true_label

    # 综合判断
    confidence_bonus = {"high": 2, "medium": 1, "low": 0}.get(retrieval_confidence, 0)
    helpful_score = skill_overlap + confidence_bonus + (2 if is_correct else 0)

    return helpful_score >= 3  # 阈值判断
