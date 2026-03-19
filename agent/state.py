"""
共享病例状态定义。

这是整个项目最核心的共享对象。
以后所有 skill 都通过它读写信息。

相关文件：
- agent/run_agent.py
- agent/planner.py
- agent/router.py
- agent/aggregator.py
- agent/reflection.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CaseState:
    # 原始输入
    case_info: Dict[str, Any]

    # 中间结果
    perception: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    planner: Dict[str, Any] = field(default_factory=dict)
    selected_skills: List[str] = field(default_factory=list)
    skill_outputs: Dict[str, Any] = field(default_factory=dict)
    final_decision: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    reflection: Dict[str, Any] = field(default_factory=dict)

    # 调试与追踪
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)

    def trace(self, step: str, status: str, note: str = "") -> None:
        self.execution_trace.append({"step": step, "status": status, "note": note})


def create_case_state(case: Dict[str, Any]) -> CaseState:
    return CaseState(case_info={
        "file": case.get("file", ""),
        "image_path": case.get("image_path", case.get("file", "")),
        "metadata": case.get("metadata", {}),
        "text": case.get("text", ""),
        "true_label": case.get("label", case.get("true_label", "unknown")),
    })
