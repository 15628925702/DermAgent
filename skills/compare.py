"""
候选对比 skill。

职责：
- 比较 top-2 / top-3 候选谁更合理
- 用于不确定病例、混淆对病例

相关文件：
- agent/planner.py
- memory/retriever.py
- agent/aggregator.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from skills.base import BaseSkill


class CompareSkill(BaseSkill):
    name = "compare_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", [])
        result = {
            "pair": [x.get("name") for x in ddx[:2]],
            "winner": ddx[0].get("name") if ddx else "unknown",
            "reason": "stub_compare",
        }
        state.skill_outputs[self.name] = result
        state.trace(self.name, "success", "Compare stub executed")
        return result
