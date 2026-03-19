"""
Router / executor。

职责：
- 按 planner 选择的顺序执行 skills
- 将 skill 输出写入 state.skill_outputs
- 记录执行轨迹
- 对单个 skill 的失败做隔离，避免整个流程中断

相关文件：
- agent/registry.py
- agent/planner.py
- agent/run_agent.py
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState


class SkillRouter:
    """
    基础版 skill 执行器。

    约定：
    - skill 对象应提供 run(state) 方法
    - skill 可以：
        1. 自己在内部写 state.skill_outputs
        2. 返回 dict，由 router 兜底写入
    """

    # 主干技能通常已经在 run_agent 主流程里执行，不应在这里重复执行
    RESERVED_SKILLS = {
        "perception_skill",
        "retrieval_skill",
        "report_skill",
    }

    def __init__(self, registry: Dict[str, Any]) -> None:
        self.registry = registry

    def execute(self, state: CaseState) -> None:
        executed: List[str] = []
        skipped: List[str] = []
        failed: List[Dict[str, str]] = []

        for skill_name in state.selected_skills:
            if skill_name in self.RESERVED_SKILLS:
                skipped.append(skill_name)
                state.trace(
                    "router",
                    "skipped",
                    f"Reserved skill is executed outside router: {skill_name}",
                )
                continue

            skill = self.registry.get(skill_name)
            if skill is None:
                failed.append({"skill": skill_name, "reason": "missing_skill"})
                state.trace("router", "failed", f"Missing skill: {skill_name}")
                continue

            try:
                output = skill.run(state)

                # 兼容两种写法：
                # 1. skill.run() 自己写入 state.skill_outputs
                # 2. skill.run() 返回 dict，由 router 兜底写入
                if isinstance(output, dict):
                    if skill_name not in state.skill_outputs:
                        state.skill_outputs[skill_name] = output

                # 若 skill 既没返回 dict，也没写入 state.skill_outputs，给一个占位输出
                if skill_name not in state.skill_outputs:
                    state.skill_outputs[skill_name] = {
                        "status": "executed",
                        "note": "Skill executed without explicit structured output.",
                    }

                executed.append(skill_name)
                state.trace(
                    "router",
                    "success",
                    f"Executed skill: {skill_name}",
                    payload={
                        "skill": skill_name,
                        "has_output": skill_name in state.skill_outputs,
                    },
                )
            except Exception as e:
                failed.append({"skill": skill_name, "reason": str(e)})
                state.skill_outputs[skill_name] = {
                    "status": "failed",
                    "error": str(e),
                }
                state.trace(
                    "router",
                    "failed",
                    f"Skill execution failed: {skill_name} | {e}",
                )

        state.planner["execution_summary"] = {
            "executed_skills": executed,
            "skipped_skills": skipped,
            "failed_skills": failed,
        }