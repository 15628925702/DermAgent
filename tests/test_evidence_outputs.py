from agent.state import create_case_state
from skills.compare import CompareSkill
from skills.specialists.mel_nev_specialist import MelNevSpecialistSkill


def test_compare_skill_emits_evidence_protocol_fields():
    state = create_case_state({"file": "cmp.png", "metadata": {}, "text": ""})
    state.perception = {
        "ddx_candidates": [
            {"name": "MEL", "score": 0.61},
            {"name": "NEV", "score": 0.42},
        ]
    }
    state.retrieval = {"retrieval_summary": {"support_labels": [], "retrieval_confidence": "low"}}

    result = CompareSkill().run(state)

    assert result["supports"] == result["winner"]
    assert result["supported_label"] == result["winner"]
    assert result["applicable"] == 1.0
    assert result["local_decision"]["supports"] == result["winner"]
    assert isinstance(result["evidence_items"], list)
    assert len(result["evidence_items"]) >= 1


def test_specialist_emits_evidence_protocol_fields():
    state = create_case_state({"file": "spec.png", "metadata": {"age": 62}, "text": ""})
    state.perception = {
        "ddx_candidates": [
            {"name": "MEL", "score": 0.57},
            {"name": "NEV", "score": 0.53},
        ],
        "visual_cues": ["irregular border", "multiple colors"],
        "risk_cues": {"malignant_cues": ["asymmetry"], "suspicious_cues": []},
    }
    state.retrieval = {
        "retrieval_summary": {
            "support_labels": ["MEL"],
            "retrieval_confidence": "medium",
        }
    }
    state.skill_outputs["compare_skill"] = {"winner": "MEL", "confidence": 0.8}
    state.skill_outputs["metadata_consistency_skill"] = {
        "supported_diagnoses": ["MEL"],
        "penalized_diagnoses": [],
    }

    result = MelNevSpecialistSkill().run(state)

    assert result["supports"] == result["recommendation"]
    assert result["supported_label"] == result["recommendation"]
    assert result["applicable"] == 1.0
    assert result["local_decision"]["supports"] == result["recommendation"]
    assert isinstance(result["evidence_items"], list)
    assert len(result["evidence_items"]) >= 1
