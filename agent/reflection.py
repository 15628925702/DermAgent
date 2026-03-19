"""
Reflection 模块。

职责：
- 诊断结束后总结本病例经验
- 形成可写回 experience bank 的中间对象

相关文件：
- memory/schema.py
- memory/writer.py
"""

from __future__ import annotations

from typing import Dict

from agent.state import CaseState


class ReflectionEngine:
    def summarize(self, state: CaseState) -> Dict[str, object]:
        result = {
            "summary": f"Case {state.case_info.get('file', '')} -> {state.final_decision.get('final_label', 'unknown')}",
            "selected_skills": list(state.selected_skills),
            "final_label": state.final_decision.get("final_label", "unknown"),
            "confusion_tag": None,
        }
        state.reflection = result
        state.trace("reflection", "success", "Reflection stub executed")
        return result
