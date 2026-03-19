"""
检索 skill 薄封装。

为什么保留这个文件：
- 论文里仍然可以把 retrieval 视为一个独立 skill
- 真正的检索逻辑放在 `memory/retriever.py`
- 这里负责把 retriever 接入 agent 主流程

职责：
- 调用 ExperienceRetriever
- 将结果写入 state.retrieval
- 记录 trace
- 提供轻量配置入口

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

    def __init__(self, retriever: ExperienceRetriever, top_k: int = 5) -> None:
        self.retriever = retriever
        self.top_k = top_k

    def run(self, state: CaseState) -> Dict[str, Any]:
        try:
            result = self.retriever.retrieve(state, top_k=self.top_k)
            state.retrieval = result

            retrieval_summary = result.get("retrieval_summary", {}) or {}
            state.trace(
                self.name,
                "success",
                "Experience retrieval executed",
                payload={
                    "top_k": self.top_k,
                    "retrieval_confidence": retrieval_summary.get("retrieval_confidence", "low"),
                    "support_labels": retrieval_summary.get("support_labels", []),
                    "has_confusion_support": retrieval_summary.get("has_confusion_support", False),
                },
            )
            return result
        except Exception as e:
            error_result = {
                "top_k": self.top_k,
                "raw_case_hits": [],
                "prototype_hits": [],
                "confusion_hits": [],
                "rule_hits": [],
                "retrieval_summary": {
                    "support_labels": [],
                    "retrieval_confidence": "low",
                    "has_confusion_support": False,
                    "confusion_pairs": [],
                    "recommended_skills": [],
                    "support_strength": {
                        "raw_case": 0,
                        "prototype": 0,
                        "confusion": 0,
                        "rule": 0,
                    },
                    "top_support_case_ids": [],
                    "supports_top1": False,
                },
                "error": str(e),
            }
            state.retrieval = error_result
            state.trace(
                self.name,
                "failed",
                f"Experience retrieval failed: {e}",
            )
            return error_result