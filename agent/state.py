"""
共享病例状态定义。

这是整个项目最核心的共享对象。
以后所有 skill 都通过它读写信息。

基础版目标：
- 统一所有模块的输入输出
- 给 planner / aggregator / reflection 留好字段
- 保持后续可升级到 learnable controller 的接口兼容性

相关文件：
- agent/run_agent.py
- agent/planner.py
- agent/router.py
- agent/aggregator.py
- agent/reflection.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CaseState:
    """
    单个病例在 agent 全流程中的共享状态。

    设计原则：
    1. 所有模块只读写 state，不互相直接耦合
    2. 字段命名尽量稳定，便于后续升级 controller / memory
    3. 既支持基础版规则路由，也预留进阶版可学习信号
    """

    # =========================
    # 原始输入
    # =========================
    case_info: Dict[str, Any]

    # =========================
    # 中间结果：主干模块
    # =========================
    perception: Dict[str, Any] = field(default_factory=dict)
    retrieval: Dict[str, Any] = field(default_factory=dict)
    planner: Dict[str, Any] = field(default_factory=dict)

    # planner 最终选中的 skills
    selected_skills: List[str] = field(default_factory=list)

    # 每个 skill 的输出，key = skill_name
    skill_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # 最终聚合决策
    final_decision: Dict[str, Any] = field(default_factory=dict)

    # 最终报告
    report: Dict[str, Any] = field(default_factory=dict)

    # 反思与经验写回
    reflection: Dict[str, Any] = field(default_factory=dict)

    # =========================
    # 调试与追踪
    # =========================
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)

    def trace(
        self,
        step: str,
        status: str,
        note: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        统一 trace 记录接口。

        参数：
        - step: 当前步骤名，如 perception / planner / aggregator
        - status: success / failed / skipped / warning
        - note: 简短说明
        - payload: 可选调试附加信息
        """
        item: Dict[str, Any] = {
            "step": step,
            "status": status,
            "note": note,
        }
        if payload is not None:
            item["payload"] = payload
        self.execution_trace.append(item)

    def get_top_ddx_names(self, top_k: int = 3) -> List[str]:
        """
        便捷函数：获取 perception 里的 top-k 候选病名。
        """
        ddx = self.perception.get("ddx_candidates", []) or []
        names: List[str] = []
        for x in ddx[:top_k]:
            name = str(x.get("name", "")).strip()
            if name:
                names.append(name.upper())
        return names

    def get_uncertainty_level(self) -> str:
        """
        便捷函数：读取 perception 中的不确定性等级。
        默认 high，避免 planner 误判为低风险。
        """
        uncertainty = self.perception.get("uncertainty", {}) or {}
        return str(uncertainty.get("level", "high")).lower()

    def get_metadata(self) -> Dict[str, Any]:
        """
        统一读取 metadata。
        """
        return self.case_info.get("metadata", {}) or {}

    def get_case_id(self) -> str:
        """
        统一病例 id。
        优先 file，其次 image_path，最后 unknown_case。
        """
        file_name = str(self.case_info.get("file", "")).strip()
        if file_name:
            return file_name

        image_path = str(self.case_info.get("image_path", "")).strip()
        if image_path:
            return image_path

        return "unknown_case"

    def to_dict(self) -> Dict[str, Any]:
        """
        导出完整状态，用于调试或保存。
        """
        return {
            "case_info": self.case_info,
            "perception": self.perception,
            "retrieval": self.retrieval,
            "planner": self.planner,
            "selected_skills": self.selected_skills,
            "skill_outputs": self.skill_outputs,
            "final_decision": self.final_decision,
            "report": self.report,
            "reflection": self.reflection,
            "trace": self.execution_trace,
        }


def create_case_state(case: Dict[str, Any]) -> CaseState:
    """
    从输入样本构造统一 CaseState。

    兼容你现在已有的数据字段：
    - file
    - image_path
    - metadata
    - text
    - label / true_label
    """
    state = CaseState(
        case_info={
            "file": case.get("file", ""),
            "image_path": case.get("image_path", case.get("file", "")),
            "metadata": case.get("metadata", {}) or {},
            "text": case.get("text", ""),
            "true_label": case.get("label", case.get("true_label", "unknown")),
        }
    )
    state.trace("state_init", "success", "CaseState created")
    return state