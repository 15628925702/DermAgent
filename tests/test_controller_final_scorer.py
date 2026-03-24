from agent.controller import LearnableSkillController
from agent.final_scorer import LearnableFinalScorer
from agent.state import create_case_state
from memory.skill_index import build_default_skill_index


def test_controller_uses_planner_context_features_and_biases():
    controller = LearnableSkillController(build_default_skill_index())
    state = create_case_state({"file": "ctrl.png", "metadata": {"site": "face"}, "text": ""})
    state.perception = {
        "ddx_candidates": [
            {"name": "BCC", "score": 0.55},
            {"name": "SCC", "score": 0.51},
        ],
        "uncertainty": {"level": "high"},
    }
    state.retrieval = {"retrieval_summary": {"retrieval_confidence": "medium", "recommended_skills": []}}

    result = controller.select_skills(
        state,
        rule_priors=[],
        planner_context={
            "recommended_skills": [],
            "flags": {"recommended_skills": [], "memory_recommended_skills": [], "rule_recommended_skills": []},
            "case_features": {
                "top_names": ["BCC", "SCC"],
                "top_gap": 0.04,
                "top_gap_small": True,
                "uncertainty": "high",
                "retrieval_confidence": "medium",
                "supports_top1": False,
                "has_confusion_support": True,
                "metadata_present": True,
                "has_malignant_candidate": True,
                "has_bcc_scc_pair": True,
                "sun_exposed_site": True,
                "strong_invasive_history": False,
            },
            "decision_trace": [
                {"skill": "compare_skill", "selected": True, "trigger": "small_top_gap"},
                {"skill": "bcc_scc_specialist_skill", "selected": True, "trigger": "pair_present_in_top_k"},
            ],
        },
    )

    debug_features = result["controller_debug"]["case_features"]
    assert debug_features["planner_compare_selected"] == 1.0
    assert debug_features["planner_bcc_scc_signal"] == 1.0
    compare_reasons = result["skill_scores"]["compare_skill"]["reasons"]
    specialist_reasons = result["skill_scores"]["bcc_scc_specialist_skill"]["reasons"]
    assert "planner:small_top_gap" in compare_reasons
    assert "planner:pair_present_in_top_k" in specialist_reasons


def test_final_scorer_prefers_anchor_plus_consistent_correction():
    scorer = LearnableFinalScorer()
    ranked, _ = scorer.rank_candidates(
        {
            "MEL": {
                "bias": 1.0,
                "base_total": 0.9,
                "perception_anchor": 0.72,
                "evidence_correction": 0.18,
                "skill_correction": 0.16,
                "retrieval_correction": 0.08,
            },
            "NEV": {
                "bias": 1.0,
                "base_total": 0.82,
                "perception_anchor": 0.46,
                "evidence_correction": 0.36,
                "skill_correction": 0.25,
                "retrieval_correction": 0.14,
            },
        }
    )

    assert ranked[0][0] == "MEL"


def test_controller_targets_follow_skill_evidence_not_just_rule_presence():
    controller = LearnableSkillController(build_default_skill_index())
    state = create_case_state(
        {
            "file": "evidence.png",
            "metadata": {"site": "face", "age": 68},
            "text": "",
            "true_label": "BCC",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "BCC", "score": 0.54},
            {"name": "SCC", "score": 0.51},
        ],
        "uncertainty": {"level": "high"},
    }
    state.retrieval = {
        "retrieval_summary": {
            "retrieval_confidence": "low",
            "supports_top1": False,
        }
    }
    state.planner = {
        "case_features": {
            "top_gap_small": True,
            "has_malignant_candidate": True,
        }
    }
    state.final_decision = {"final_label": "SCC"}
    state.skill_outputs = {
        "compare_skill": {
            "supports": "BCC",
            "confidence": 0.82,
            "applicable": 1.0,
            "local_decision": {
                "supports": "BCC",
                "opposes": "SCC",
                "confidence": 0.82,
                "applicable": 1.0,
            },
        },
        "bcc_scc_specialist_skill": {
            "supports": "SCC",
            "confidence": 0.76,
            "applicable": 1.0,
            "local_decision": {
                "supports": "SCC",
                "opposes": "BCC",
                "confidence": 0.76,
                "applicable": 1.0,
            },
        },
        "metadata_consistency_skill": {
            "supported_labels": ["BCC"],
            "penalized_labels": ["SCC"],
            "score": 0.7,
        },
        "malignancy_risk_skill": {
            "risk_level": "high",
            "preferred_label": "BCC",
        },
    }

    targets = controller._build_targets(state)

    assert targets["compare_skill"] > 0.85
    assert targets["metadata_consistency_skill"] > 0.8
    assert targets["malignancy_risk_skill"] > 0.9
    assert targets["bcc_scc_specialist_skill"] < 0.2


def test_controller_does_not_penalize_unselected_skills_as_false_negatives():
    controller = LearnableSkillController(build_default_skill_index())
    state = create_case_state(
        {
            "file": "no_false_negative.png",
            "metadata": {"site": "face", "age": 70},
            "text": "",
            "true_label": "BCC",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "BCC", "score": 0.58},
            {"name": "SCC", "score": 0.54},
        ],
        "uncertainty": {"level": "high"},
    }
    state.retrieval = {"retrieval_summary": {"retrieval_confidence": "low", "supports_top1": False}}
    state.planner = {"case_features": {"top_gap_small": True, "has_malignant_candidate": True}}
    state.selected_skills = ["compare_skill"]
    state.final_decision = {"final_label": "BCC", "confidence": "medium"}
    state.skill_outputs = {
        "compare_skill": {
            "supports": "BCC",
            "confidence": 0.8,
            "local_decision": {"supports": "BCC", "confidence": 0.8, "applicable": 1.0},
        }
    }

    untouched_before = controller.target_learner.base_targets["mel_nev_specialist_skill"]
    controller.update_from_case(state)
    untouched_after = controller.target_learner.base_targets["mel_nev_specialist_skill"]

    assert untouched_after == untouched_before
