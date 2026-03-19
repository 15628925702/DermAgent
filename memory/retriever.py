"""
经验检索器。

基础版先做简单检索接口。
你当前 `skills/retrieval.py` 的逻辑可以逐步迁到这里。

相关文件：
- memory/experience_bank.py
- skills/retrieval.py
- agent/planner.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from memory.experience_bank import ExperienceBank


class ExperienceRetriever:
    def __init__(self, bank: ExperienceBank) -> None:
        self.bank = bank

    def retrieve(self, state: CaseState, top_k: int = 5) -> Dict[str, Any]:
        # TODO: 后续迁移你旧 retrieval 里的 metadata / label / coarse category 打分逻辑。
        return {
            "top_k": top_k,
            "raw_case_hits": self.bank.list_all()[:top_k],
            "prototype_hits": [],
            "confusion_hits": [],
            "rule_hits": [],
            "retrieval_summary": {"support_labels": [], "retrieval_confidence": "low"},
        }
