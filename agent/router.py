"""
Router / executor。

职责：
- 按 planner 选择的顺序执行 skills
- 将 skill 输出写入 state.skill_outputs

相关文件：
- agent/registry.py
- agent/planner.py
"""

from __future__ import annotations

from agent.state import CaseState


class SkillRouter:
    def __init__(self, registry: dict) -> None:
        self.registry = registry

    def execute(self, state: CaseState) -> None:
        for skill_name in state.selected_skills:
            skill = self.registry.get(skill_name)
            if skill is None:
                state.trace("router", "failed", f"Missing skill: {skill_name}")
                continue
            skill.run(state)
