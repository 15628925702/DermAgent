"""
检索 skill 薄封装。

为什么保留这个文件：
- 你论文里可能仍然希望把 retrieval 看成一个 skill
- 但真正的检索逻辑放在 `memory/retriever.py`

相关文件：
- memory/retriever.py
- memory/experience_bank.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from memory.retriever import ExperienceRetriever
from skills.base import BaseSkill


class RetrievalSkill(BaseSkill):
    name = "retrieval_skill"

    def __init__(self, retriever: ExperienceRetriever) -> None:
        self.retriever = retriever

    def run(self, state: CaseState) -> Dict[str, Any]:
        result = self.retriever.retrieve(state)
        state.retrieval = result
        state.trace(self.name, "success", "Experience retrieval executed")
        return result
