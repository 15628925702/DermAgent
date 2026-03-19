"""
不确定性分析 skill。

职责：
- 统一读取 perception/retrieval 信号
- 决定是否需要 compare 或 specialist

相关文件：
- agent/planner.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from skills.base import BaseSkill


class UncertaintyAssessmentSkill(BaseSkill):
    name = "uncertainty_assessment_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        result = state.perception.get("uncertainty", {"level": "high", "reason": "missing_perception"})
        state.skill_outputs[self.name] = result
        state.trace(self.name, "success", "Uncertainty assessment executed")
        return result
