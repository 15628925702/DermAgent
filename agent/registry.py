"""
Skill 注册表。

职责：
- 集中创建所有 skills
- router 只通过注册表拿 skill

相关文件：
- agent/router.py
- skills/*
- memory/retriever.py
"""

from __future__ import annotations

from typing import Dict

from memory.experience_bank import ExperienceBank
from memory.retriever import ExperienceRetriever
from skills.compare import CompareSkill
from skills.malignancy import MalignancyRiskSkill
from skills.metadata_consistency import MetadataConsistencySkill
from skills.perception import PerceptionSkill
from skills.reporter import ReportSkill
from skills.retrieval import RetrievalSkill
from skills.uncertainty import UncertaintyAssessmentSkill
from skills.specialists.ack_scc_specialist import AckSccSpecialistSkill
from skills.specialists.mel_nev_specialist import MelNevSpecialistSkill


def build_skill_registry(bank: ExperienceBank | None = None) -> Dict[str, object]:
    bank = bank or ExperienceBank()
    retriever = ExperienceRetriever(bank)
    return {
        "perception_skill": PerceptionSkill(),
        "retrieval_skill": RetrievalSkill(retriever),
        "uncertainty_assessment_skill": UncertaintyAssessmentSkill(),
        "compare_skill": CompareSkill(),
        "malignancy_risk_skill": MalignancyRiskSkill(),
        "metadata_consistency_skill": MetadataConsistencySkill(),
        "ack_scc_specialist_skill": AckSccSpecialistSkill(),
        "mel_nev_specialist_skill": MelNevSpecialistSkill(),
        "report_skill": ReportSkill(),
    }
