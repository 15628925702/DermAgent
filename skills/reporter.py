"""
报告生成 skill。

职责：
- 基于 final_decision 生成结构化报告
- 报告不是最终判断器，只负责表达输出

相关文件：
- agent/aggregator.py
- agent/run_agent.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from skills.base import BaseSkill


class ReportSkill(BaseSkill):
    name = "report_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        final_name = state.final_decision.get("final_label", "unknown")
        result = {
            "summary": f"Predicted diagnosis: {final_name}",
            "final_impression": final_name,
            "next_step": "clinical correlation recommended",
        }
        state.report = result
        state.trace(self.name, "success", "Report stub executed")
        return result
