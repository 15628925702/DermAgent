from agent.state import create_case_state
from skills.metadata_consistency import MetadataConsistencySkill


def test_metadata_consistency_adds_pediatric_nevus_rescue_signal():
    state = create_case_state(
        {
            "file": "meta1.png",
            "metadata": {"age": 8, "site": "arm", "clinical_history": "stable"},
            "text": "",
        }
    )
    state.perception = {
        "ddx_candidates": [
            {"name": "SCC", "score": 0.52},
            {"name": "BCC", "score": 0.28},
        ],
        "risk_cues": {"malignant_cues": [], "suspicious_cues": []},
    }

    result = MetadataConsistencySkill().run(state)

    assert "NEV" in result["supported_labels"]
    assert "SCC" in result["penalized_labels"]
    assert "BCC" in result["penalized_labels"]
