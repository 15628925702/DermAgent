from agent.controller import LearnableSkillController
from agent.evidence_calibrator import LearnableEvidenceCalibrator
from agent.planner import ExperienceSkillPlanner
from agent.state import create_case_state
from memory.skill_index import build_default_skill_index


def test_planner_records_case_features_and_decision_trace():
    state = create_case_state(
        {
            "file": "planner1.png",
            "metadata": {"age": 68, "site": "face", "clinical_history": "bleeding"},
            "text": "",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "BCC", "score": 0.61},
            {"name": "SCC", "score": 0.56},
            {"name": "SEK", "score": 0.18},
        ],
        "uncertainty": {"level": "high"},
        "risk_cues": {"malignant_cues": ["ulcerated"]},
    }
    state.retrieval = {
        "confusion_hits": [{"pair": ["BCC", "SCC"]}],
        "rule_hits": [],
        "retrieval_summary": {
            "retrieval_confidence": "medium",
            "supports_top1": False,
            "has_confusion_support": True,
            "memory_recommended_skills": ["bcc_scc_specialist_skill"],
            "recommended_skills": ["compare_skill", "bcc_scc_specialist_skill"],
        },
    }

    planner = ExperienceSkillPlanner(
        use_specialist=True,
        planning_mode="rules_only",
        enabled_skills={
            "uncertainty_assessment_skill",
            "compare_skill",
            "malignancy_risk_skill",
            "metadata_consistency_skill",
            "bcc_scc_specialist_skill",
        },
    )
    result = planner.plan(state)

    assert "case_features" in result
    assert result["case_features"]["has_bcc_scc_pair"] is True
    assert result["case_features"]["strong_invasive_history"] is True
    assert "decision_trace" in result
    compare_items = [item for item in result["decision_trace"] if item["skill"] == "compare_skill"]
    assert compare_items and compare_items[0]["selected"] is True
    specialist_items = [item for item in result["decision_trace"] if item["skill"] == "bcc_scc_specialist_skill"]
    assert specialist_items and specialist_items[0]["selected"] is True


def test_planner_can_trigger_ack_scc_specialist_from_metadata_proxy():
    state = create_case_state(
        {
            "file": "planner2.png",
            "metadata": {"site": "hand", "clinical_history": "changed"},
            "text": "",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "SCC", "score": 0.52},
            {"name": "NEV", "score": 0.48},
        ],
        "uncertainty": {"level": "medium"},
        "risk_cues": {"malignant_cues": []},
    }
    state.retrieval = {
        "confusion_hits": [],
        "rule_hits": [],
        "retrieval_summary": {
            "retrieval_confidence": "low",
            "supports_top1": False,
            "has_confusion_support": False,
            "memory_recommended_skills": [],
            "recommended_skills": [],
        },
    }

    planner = ExperienceSkillPlanner(
        use_specialist=True,
        planning_mode="rules_only",
        enabled_skills={
            "uncertainty_assessment_skill",
            "compare_skill",
            "metadata_consistency_skill",
            "ack_scc_specialist_skill",
        },
    )
    result = planner.plan(state)

    assert "ack_scc_specialist_skill" in result["selected_skills"]
    specialist_items = [item for item in result["decision_trace"] if item["skill"] == "ack_scc_specialist_skill"]
    assert specialist_items and specialist_items[0]["trigger"] == "metadata_proxy_support"


def test_planner_does_not_trigger_specialist_for_low_uncertainty_large_gap_pair():
    state = create_case_state(
        {
            "file": "planner3.png",
            "metadata": {"age": 8, "site": "arm"},
            "text": "",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "SCC", "score": 0.62},
            {"name": "ACK", "score": 0.28},
            {"name": "BCC", "score": 0.10},
        ],
        "uncertainty": {"level": "low"},
        "risk_cues": {"malignant_cues": ["border irregularity"]},
    }
    state.retrieval = {
        "confusion_hits": [],
        "rule_hits": [],
        "retrieval_summary": {
            "retrieval_confidence": "low",
            "supports_top1": False,
            "has_confusion_support": False,
            "memory_recommended_skills": [],
            "recommended_skills": [],
        },
    }

    planner = ExperienceSkillPlanner(
        use_specialist=True,
        planning_mode="rules_only",
        enabled_skills={
            "uncertainty_assessment_skill",
            "metadata_consistency_skill",
            "ack_scc_specialist_skill",
            "bcc_scc_specialist_skill",
        },
    )
    result = planner.plan(state)

    assert "ack_scc_specialist_skill" not in result["selected_skills"]
    assert "bcc_scc_specialist_skill" not in result["selected_skills"]


def test_planner_only_triggers_top2_specialist_when_multiple_pairs_overlap():
    state = create_case_state(
        {
            "file": "planner4.png",
            "metadata": {"age": 72, "site": "face", "clinical_history": "changed"},
            "text": "",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "SCC", "score": 0.39},
            {"name": "ACK", "score": 0.36},
            {"name": "BCC", "score": 0.34},
        ],
        "uncertainty": {"level": "high"},
        "risk_cues": {"malignant_cues": []},
    }
    state.retrieval = {
        "confusion_hits": [],
        "rule_hits": [],
        "retrieval_summary": {
            "retrieval_confidence": "low",
            "supports_top1": False,
            "has_confusion_support": False,
            "memory_recommended_skills": [],
            "recommended_skills": [],
        },
    }

    planner = ExperienceSkillPlanner(
        use_specialist=True,
        planning_mode="rules_only",
        enabled_skills={
            "uncertainty_assessment_skill",
            "compare_skill",
            "ack_scc_specialist_skill",
            "bcc_scc_specialist_skill",
        },
    )
    result = planner.plan(state)

    assert "ack_scc_specialist_skill" in result["selected_skills"]
    assert "bcc_scc_specialist_skill" not in result["selected_skills"]


def test_planner_can_use_learned_specialist_threshold_from_calibrator(monkeypatch):
    state = create_case_state(
        {
            "file": "planner5.png",
            "metadata": {"age": 67, "site": "face"},
            "text": "",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "BCC", "score": 0.82},
            {"name": "SCC", "score": 0.14},
        ],
        "uncertainty": {"level": "low"},
        "risk_cues": {"malignant_cues": []},
    }
    state.retrieval = {
        "confusion_hits": [],
        "rule_hits": [],
        "retrieval_summary": {
            "retrieval_confidence": "low",
            "supports_top1": False,
            "has_confusion_support": False,
            "memory_recommended_skills": [],
            "recommended_skills": [],
        },
    }

    controller = LearnableSkillController(build_default_skill_index())
    monkeypatch.setattr(
        controller.target_learner,
        "predict_target",
        lambda skill_name, features, state: 0.8 if skill_name == "bcc_scc_specialist_skill" else 0.0,
    )
    calibrator = LearnableEvidenceCalibrator()
    calibrator.weights["planner_specialist_threshold"] = 0.75

    planner = ExperienceSkillPlanner(
        use_specialist=True,
        controller=controller,
        evidence_calibrator=calibrator,
        planning_mode="rules_only",
        enabled_skills={
            "uncertainty_assessment_skill",
            "bcc_scc_specialist_skill",
        },
    )
    result = planner.plan(state)

    assert "bcc_scc_specialist_skill" in result["selected_skills"]
