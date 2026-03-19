"""
证据聚合器。

职责：
- 汇总 perception / retrieval / compare / risk / specialist 输出
- 给出 final_decision

相关文件：
- skills/reporter.py
- agent/reflection.py
"""

from __future__ import annotations

from typing import Dict

from agent.state import CaseState


class DecisionAggregator:
    def aggregate(self, state: CaseState) -> Dict[str, object]:
        most_likely = (state.perception.get("most_likely", {}) or {}).get("name", "unknown")
        compare_winner = (state.skill_outputs.get("compare_skill", {}) or {}).get("winner")
        specialist_pref = None
        for key in ["mel_nev_specialist_skill", "ack_scc_specialist_skill"]:
            if key in state.skill_outputs:
                specialist_pref = state.skill_outputs[key].get("recommendation")
                break

        final_label = specialist_pref or compare_winner or most_likely
        result = {
            "final_label": final_label,
            "evidence": {
                "most_likely": most_likely,
                "compare_winner": compare_winner,
                "specialist_pref": specialist_pref,
            },
        }
        state.final_decision = result
        state.trace("aggregator", "success", f"Final label: {final_label}")
        return result
