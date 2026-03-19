"""
Skill 注册表。

职责：
- 集中创建所有 skills
- 注入依赖（如 experience retriever）
- 支持后续扩展（ablation / debug / config）

设计原则：
- registry 是唯一 skill 构造入口
- router 只依赖 registry，不关心 skill 如何创建
- 为进阶版 skill embedding / controller 预留接口

相关文件：
- agent/router.py
- skills/*
- memory/retriever.py
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.retriever import ExperienceRetriever
from memory.skill_index import SkillIndex

# skills
from skills.compare import CompareSkill
from skills.malignancy import MalignancyRiskSkill
from skills.metadata_consistency import MetadataConsistencySkill
from skills.perception import PerceptionSkill
from skills.reporter import ReportSkill
from skills.retrieval import RetrievalSkill
from skills.uncertainty import UncertaintyAssessmentSkill
from skills.specialists.ack_scc_specialist import AckSccSpecialistSkill
from skills.specialists.bcc_scc_specialist import BccSccSpecialistSkill
from skills.specialists.bcc_sek_specialist import BccSekSpecialistSkill
from skills.specialists.mel_nev_specialist import MelNevSpecialistSkill


def build_skill_registry(
    bank: Optional[ExperienceBank] = None,
    skill_index: Optional[SkillIndex] = None,
    reranker: Optional[UtilityAwareExperienceReranker] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建 skill registry。

    参数：
    - bank: 经验库
    - config: 可选配置（用于 ablation / debug / future controller）

    config 示例：
    {
        "disable_skills": ["compare_skill"],
        "debug": True
    }
    """
    bank = bank or ExperienceBank()
    config = config or {}

    disable_skills = set(config.get("disable_skills", []))
    debug = bool(config.get("debug", False))

    # =========================
    # 依赖构建
    # =========================
    retriever = ExperienceRetriever(bank, reranker=reranker)

    # =========================
    # Skill 实例化
    # =========================
    registry: Dict[str, Any] = {
        "perception_skill": PerceptionSkill(),
        "retrieval_skill": RetrievalSkill(retriever),
        "uncertainty_assessment_skill": UncertaintyAssessmentSkill(),
        "compare_skill": CompareSkill(),
        "malignancy_risk_skill": MalignancyRiskSkill(),
        "metadata_consistency_skill": MetadataConsistencySkill(),
        "ack_scc_specialist_skill": AckSccSpecialistSkill(),
        "bcc_scc_specialist_skill": BccSccSpecialistSkill(),
        "bcc_sek_specialist_skill": BccSekSpecialistSkill(),
        "mel_nev_specialist_skill": MelNevSpecialistSkill(),
        "report_skill": ReportSkill(),
    }

    # =========================
    # Ablation 支持（关闭某些 skill）
    # =========================
    if disable_skills:
        for name in list(registry.keys()):
            if name in disable_skills:
                registry.pop(name)

    # =========================
    # Debug 信息（可选）
    # =========================
    if debug:
        registry["_debug_info"] = {
            "num_skills": len(registry),
            "skills": list(registry.keys()),
            "disabled_skills": list(disable_skills),
            "skill_index_enabled": skill_index is not None,
            "reranker_enabled": reranker is not None,
        }

    return registry
