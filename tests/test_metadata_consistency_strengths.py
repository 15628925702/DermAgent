from agent.evidence_calibrator import LearnableEvidenceCalibrator
from agent.state import create_case_state
from skills.metadata_consistency import MetadataConsistencySkill


def test_metadata_skill_emits_weighted_strengths_for_pediatric_case():
    state = create_case_state(
        {
            "file": "meta1.png",
            "metadata": {"age": 8, "site": "arm", "clinical_history": "changed"},
            "text": "",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "SCC", "score": 0.51},
            {"name": "NEV", "score": 0.49},
        ],
        "risk_cues": {"malignant_cues": []},
    }

    output = MetadataConsistencySkill().run(state)

    assert "NEV" in output["supported_labels"]
    assert output["support_strengths"]["NEV"] > 0.0
    assert "SCC" in output["penalized_labels"]
    assert output["penalty_strengths"]["SCC"] > 0.0


def test_metadata_skill_uses_calibrator_weights_for_nev_rescue():
    state = create_case_state(
        {
            "file": "meta2.png",
            "metadata": {"age": 10, "site": "trunk"},
            "text": "",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "NEV", "score": 0.42},
            {"name": "ACK", "score": 0.38},
        ],
        "risk_cues": {"malignant_cues": []},
    }
    calibrator = LearnableEvidenceCalibrator()
    calibrator.weights["metadata_nev_rescue_bonus"] = 1.4

    output = MetadataConsistencySkill(evidence_calibrator=calibrator).run(state)

    assert output["support_strengths"]["NEV"] >= 1.4
