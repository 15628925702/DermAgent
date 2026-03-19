"""
恶性风险 skill。

职责：
- 不是直接定病种
- 而是输出 benign / suspicious / malignant-like 倾向
- 供 aggregator 作为 safety evidence 使用

相关文件：
- skills/perception.py
- agent/aggregator.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from skills.base import BaseSkill


class MalignancyRiskSkill(BaseSkill):
    name = "malignancy_risk_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        cues = (state.perception.get("risk_cues", {}) or {}).get("malignant_cues", [])
        level = "high" if cues else "low"
        result = {"risk_level": level, "evidence": cues}
        state.skill_outputs[self.name] = result
        state.trace(self.name, "success", "Malignancy stub executed")
        return result
