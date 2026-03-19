"""
metadata 一致性校验 skill。

职责：
- 当视觉证据和 metadata 冲突时提供修正信息
- 可用于年龄、部位、病史先验检查

相关文件：
- agent/planner.py
- agent/aggregator.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from skills.base import BaseSkill


class MetadataConsistencySkill(BaseSkill):
    name = "metadata_consistency_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        meta = state.case_info.get("metadata", {}) or {}
        result = {"consistency": "unknown", "meta_keys": list(meta.keys())}
        state.skill_outputs[self.name] = result
        state.trace(self.name, "success", "Metadata consistency stub executed")
        return result
