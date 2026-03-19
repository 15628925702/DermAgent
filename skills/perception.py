"""
通用感知 skill。

职责：
- 输入图像与 metadata
- 提取粗粒度 visual cues
- 生成 top-k ddx
- 估计 uncertainty

相关文件：
- models/vlm_backend.py
- llm/gpt_api.py
- agent/planner.py
- agent/state.py
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState
from skills.base import BaseSkill


class PerceptionSkill(BaseSkill):
    name = "perception_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        # TODO: 迁移你当前 skills/perception.py 的核心逻辑到这里。
        result = {
            "coarse_category": "uncertain",
            "visual_findings": {
                "lesion_type": "unknown",
                "border": "unknown",
                "symmetry": "unknown",
                "surface_scale": "unknown",
                "pigment_pattern": "unknown",
                "color": [],
            },
            "risk_cues": {
                "malignant_cues": [],
                "benign_cues": [],
            },
            "ddx_candidates": [],
            "most_likely": {"name": "unknown", "score": 0.0},
            "uncertainty": {"level": "high", "reason": "stub"},
        }
        state.perception = result
        state.trace(self.name, "success", "Perception stub executed")
        return result
