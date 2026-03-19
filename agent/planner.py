"""
基础版 planner。

这是规则式 planner，不是可训练 controller。
以后你做进阶版时，可以保持同样接口，替换内部实现。

相关文件：
- agent/router.py
- skills/uncertainty.py
- integrations/langchain/planner_chain.py
"""

from __future__ import annotations

from typing import Dict, List

from agent.state import CaseState


class ExperienceSkillPlanner:
    def plan(self, state: CaseState) -> Dict[str, object]:
        selected: List[str] = ["uncertainty_assessment_skill"]

        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [str(x.get("name", "")).upper() for x in ddx[:3]]
        uncertainty = (state.perception.get("uncertainty", {}) or {}).get("level", "high")

        if uncertainty in {"medium", "high"}:
            selected.append("compare_skill")

        if {"MEL", "BCC", "SCC"}.intersection(top_names):
            selected.append("malignancy_risk_skill")

        if "ACK" in top_names and "SCC" in top_names:
            selected.append("ack_scc_specialist_skill")

        if "MEL" in top_names and "NEV" in top_names:
            selected.append("mel_nev_specialist_skill")

        selected.append("metadata_consistency_skill")

        state.selected_skills = list(dict.fromkeys(selected))
        state.planner = {
            "selected_skills": state.selected_skills,
            "reason": {
                "top_names": top_names,
                "uncertainty": uncertainty,
            },
        }
        state.trace("planner", "success", f"Selected {state.selected_skills}")
        return state.planner
