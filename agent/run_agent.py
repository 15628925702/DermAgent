from __future__ import annotations

from typing import Any, Dict

from agent.aggregator import DecisionAggregator
from agent.planner import ExperienceSkillPlanner
from agent.reflection import ReflectionEngine
from agent.registry import build_skill_registry
from agent.router import SkillRouter
from agent.state import create_case_state
from memory.experience_bank import ExperienceBank
from memory.writer import ExperienceWriter


USE_RETRIEVAL = True
USE_SPECIALIST = True
USE_REFLECTION = True


def run_agent(
    case: Dict[str, Any],
    bank: ExperienceBank | None = None,
    *,
    use_retrieval: bool = USE_RETRIEVAL,
    use_specialist: bool = USE_SPECIALIST,
    use_reflection: bool = USE_REFLECTION,
) -> Dict[str, Any]:
    if bank is None:
        bank = ExperienceBank()

    state = create_case_state(case)
    runtime_flags = {
        "use_retrieval": use_retrieval,
        "use_specialist": use_specialist,
        "use_reflection": use_reflection,
    }
    state.trace("config", "success", "Runtime flags initialized", payload=runtime_flags)

    try:
        disable_skills = []
        if not use_specialist:
            disable_skills.extend(
                [
                    "ack_scc_specialist_skill",
                    "mel_nev_specialist_skill",
                ]
            )
        registry = build_skill_registry(
            bank,
            config={"disable_skills": disable_skills},
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
        except Exception as e:
            state.trace("retrieval", "failed", f"Retrieval skill failed: {e}")
            return _export_state(state, error=f"Retrieval skill failed: {e}", runtime_flags=runtime_flags)
    else:
        state.retrieval = _empty_retrieval_result(reason="retrieval_disabled")
        state.trace("retrieval", "skipped", "Retrieval disabled by runtime flag")

    try:
        planner = ExperienceSkillPlanner(use_specialist=use_specialist)
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
        aggregator = DecisionAggregator()
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
        state.reflection = {
            "status": "skipped",
            "reason": "reflection_disabled",
        }
        state.trace("reflection", "skipped", "Reflection disabled by runtime flag")

    if use_reflection:
        try:
            writer = ExperienceWriter()
            writer.write_case(
                state=state,
                bank=bank,
                auto=True,
            )
        except Exception as e:
            state.trace("writeback", "failed", f"Experience writeback failed: {e}")
    else:
        state.trace("writeback", "skipped", "Writeback skipped because reflection is disabled")

    return _export_state(state, runtime_flags=runtime_flags)


def _export_state(
    state: Any,
    error: str | None = None,
    runtime_flags: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    result = {
        "case_info": state.case_info,
        "perception": state.perception,
        "retrieval": state.retrieval,
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
            "retrieval_confidence": "low",
            "has_confusion_support": False,
            "confusion_pairs": [],
            "recommended_skills": [],
            "support_strength": {
                "raw_case": 0,
                "prototype": 0,
                "confusion": 0,
                "rule": 0,
            },
            "top_support_case_ids": [],
            "supports_top1": False,
            "status": "skipped",
            "reason": reason,
        },
    }
