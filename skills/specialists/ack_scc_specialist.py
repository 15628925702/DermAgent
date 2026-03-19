from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from skills.base import BaseSkill


class AckSccSpecialistSkill(BaseSkill):
    """
    ACK / SCC 混淆组 specialist。

    什么时候用：
    - perception top-k 同时出现 ACK 和 SCC
    - retrieval 中 ACK / SCC 混淆明显

    相关文件：
    - agent/planner.py
    - skills/compare.py
    - memory/retriever.py
    - agent/aggregator.py
    """

    name = "ack_scc_specialist_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        result = {
            "target_group": ["ACK", "SCC"],
            "recommendation": "ACK",
            "supporting_evidence": ["stub"],
            "confidence": 0.5,
        }
        state.skill_outputs[self.name] = result
        state.trace(self.name, "success", "ACK/SCC specialist stub executed")
        return result
