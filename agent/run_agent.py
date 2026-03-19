"""
主执行入口。

基础版固定主干：
state init -> perception -> retrieval -> planner -> router -> aggregator -> report -> reflection

相关文件：
- agent/registry.py
- agent/planner.py
- agent/router.py
- agent/aggregator.py
- agent/reflection.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.aggregator import DecisionAggregator
from agent.planner import ExperienceSkillPlanner
from agent.reflection import ReflectionEngine
from agent.registry import build_skill_registry
from agent.router import SkillRouter
from agent.state import create_case_state
from memory.experience_bank import ExperienceBank


def run_agent(case: Dict[str, Any], bank: ExperienceBank | None = None) -> Dict[str, Any]:
    state = create_case_state(case)
    registry = build_skill_registry(bank)

    registry["perception_skill"].run(state)
    registry["retrieval_skill"].run(state)

    planner = ExperienceSkillPlanner()
    planner.plan(state)

    router = SkillRouter(registry)
    router.execute(state)

    aggregator = DecisionAggregator()
    aggregator.aggregate(state)

    registry["report_skill"].run(state)

    reflection = ReflectionEngine()
    reflection.summarize(state)

    return {
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
