"""
经验写回器。

职责：
- 把 reflection 结果转换成 schema item
- 写回 bank

相关文件：
- memory/schema.py
- memory/experience_bank.py
- agent/reflection.py
"""

from __future__ import annotations

from agent.state import CaseState
from memory.experience_bank import ExperienceBank
from memory.schema import build_raw_case_experience


class ExperienceWriter:
    def write_case(self, state: CaseState, bank: ExperienceBank) -> None:
        item = build_raw_case_experience(
            case_id=str(state.case_info.get("file", "unknown_case")),
            payload={
                "final_label": state.final_decision.get("final_label", "unknown"),
                "reflection": state.reflection,
                "selected_skills": list(state.selected_skills),
            },
        )
        bank.add(item)
