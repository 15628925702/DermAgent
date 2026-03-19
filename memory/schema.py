"""
Experience schema。

基础版统一四类经验：
- raw_case：每个病例的完整记录
- prototype：典型疾病原型
- confusion：混淆对经验
- rule：可复用规则（轻量版）

设计原则：
- 全部是结构化 dict，方便后续 embedding / 检索
- 字段尽量稳定，为进阶版预留扩展空间
- payload 不做过度嵌套，保证可读性

相关文件：
- memory/writer.py
- memory/experience_bank.py
- memory/retriever.py
- agent/reflection.py
"""

from __future__ import annotations

from typing import Any, Dict, List


# =========================
# 通用工具
# =========================

def _base_experience(exp_type: str) -> Dict[str, Any]:
    return {
        "experience_type": exp_type,
    }


# =========================
# 1. Raw Case Experience
# =========================

def build_raw_case_experience(
    case_id: str,
    perception: Dict[str, Any],
    final_decision: Dict[str, Any],
    selected_skills: List[str],
    retrieval: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    每个病例一条。

    用于：
    - retrieval
    - 失败分析
    - prototype / confusion 抽取（后续）
    """
    return {
        **_base_experience("raw_case"),
        "case_id": case_id,

        # 输入
        "metadata": metadata,

        # perception 结果
        "perception": {
            "ddx": perception.get("ddx_candidates", []),
            "uncertainty": perception.get("uncertainty", {}),
            "visual_cues": perception.get("visual_cues", []),
        },

        # 决策
        "final_decision": final_decision,

        # 使用了哪些技能
        "selected_skills": selected_skills,

        # 检索支持
        "retrieval_summary": {
            "num_raw_case_hits": len(retrieval.get("raw_case_hits", [])),
            "num_prototype_hits": len(retrieval.get("prototype_hits", [])),
            "num_confusion_hits": len(retrieval.get("confusion_hits", [])),
        },

        # 简单标签（方便后续统计）
        "tags": {
            "uncertainty_level": perception.get("uncertainty", {}).get("level", "high"),
            "fallback_reason": perception.get("fallback_reason"),
        },
    }


# =========================
# 2. Prototype Experience
# =========================

def build_prototype_experience(
    disease: str,
    typical_cues: List[str],
    typical_metadata: Dict[str, Any],
    common_confusions: List[str],
    recommended_skills: List[str],
) -> Dict[str, Any]:
    """
    每类病的典型特征总结。

    用于：
    - retrieval 提供先验
    - planner 辅助决策（进阶版更重要）
    """
    return {
        **_base_experience("prototype"),
        "disease": disease.upper(),

        "typical_cues": typical_cues,
        "typical_metadata": typical_metadata,
        "common_confusions": [x.upper() for x in common_confusions],

        "recommended_skills": recommended_skills,
    }


# =========================
# 3. Confusion Experience
# =========================

def build_confusion_experience(
    disease_a: str,
    disease_b: str,
    distinguishing_clues: List[str],
    useful_skills: List[str],
    failure_modes: List[str],
) -> Dict[str, Any]:
    """
    混淆对经验（非常关键）。

    用于：
    - compare skill
    - planner（识别是否进入 specialist）
    """
    return {
        **_base_experience("confusion"),

        "pair": sorted([disease_a.upper(), disease_b.upper()]),

        "distinguishing_clues": distinguishing_clues,
        "useful_skills": useful_skills,

        "failure_modes": failure_modes,
    }


# =========================
# 4. Rule Experience（轻量版）
# =========================

def build_rule_experience(
    rule_name: str,
    trigger_conditions: Dict[str, Any],
    action: Dict[str, Any],
    priority: int = 1,
) -> Dict[str, Any]:
    """
    可复用规则（基础版只是存，不自动执行）。

    未来可以：
    - 转成 planner policy
    - 作为 learnable controller 的监督数据
    """
    return {
        **_base_experience("rule"),

        "rule_name": rule_name,

        "trigger_conditions": trigger_conditions,
        "action": action,

        "priority": priority,
    }


def build_hard_case_experience(
    case_id: str,
    final_label: str,
    top_ddx: List[str],
    uncertainty: str,
    learning_signals: Dict[str, Any],
    selected_skills: List[str],
) -> Dict[str, Any]:
    return {
        **_base_experience("hard_case"),
        "case_id": case_id,
        "final_label": str(final_label).strip().upper(),
        "top_ddx": [str(x).strip().upper() for x in top_ddx if str(x).strip()],
        "uncertainty": uncertainty,
        "learning_signals": learning_signals,
        "selected_skills": selected_skills,
    }


# =========================
# 类型判断工具
# =========================

def is_raw_case(exp: Dict[str, Any]) -> bool:
    return exp.get("experience_type") == "raw_case"


def is_prototype(exp: Dict[str, Any]) -> bool:
    return exp.get("experience_type") == "prototype"


def is_confusion(exp: Dict[str, Any]) -> bool:
    return exp.get("experience_type") == "confusion"


def is_rule(exp: Dict[str, Any]) -> bool:
    return exp.get("experience_type") == "rule"


def is_hard_case(exp: Dict[str, Any]) -> bool:
    return exp.get("experience_type") == "hard_case"
