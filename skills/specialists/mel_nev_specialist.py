from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from skills.base import BaseSkill


class MelNevSpecialistSkill(BaseSkill):
    """
    MEL / NEV 混淆组 specialist。

    什么时候用：
    - perception top-k 同时出现 MEL 和 NEV
    - 存在色素不均、边界不规则等线索

    相关文件：
    - agent/planner.py
    - skills/compare.py
    - memory/retriever.py
    - agent/aggregator.py
    """

    name = "mel_nev_specialist_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        result = {
            "target_group": ["MEL", "NEV"],
            "recommendation": "MEL",
            "supporting_evidence": ["stub"],
            "confidence": 0.5,
        }
        state.skill_outputs[self.name] = result
        state.trace(self.name, "success", "MEL/NEV specialist stub executed")
        return result
