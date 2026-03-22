from agent.aggregator import DecisionAggregator
from agent.state import create_case_state


def test_aggregator_keeps_strong_perception_anchor_when_correction_is_large():
    state = create_case_state({"file": "case1.png", "metadata": {}, "text": ""})
    state.perception = {
        "ddx_candidates": [
            {"name": "MEL", "score": 0.92},
            {"name": "NEV", "score": 0.35},
        ],
        "uncertainty": {"level": "medium"},
    }
    state.retrieval = {
        "retrieval_summary": {
            "support_labels": ["NEV"],
            "prototype_votes": {"NEV": 2},
            "confusion_votes": {},
            "memory_consensus_label": "NEV",
            "retrieval_confidence": "high",
            "supports_top1": False,
        }
    }
    state.skill_outputs = {
        "compare_skill": {"winner": "NEV", "confidence": 0.95},
        "mel_nev_specialist_skill": {"recommendation": "NEV", "confidence": 1.0},
        "metadata_consistency_skill": {"supported_diagnoses": ["NEV"], "penalized_diagnoses": []},
    }

    result = DecisionAggregator().aggregate(state)

    assert result["final_label"] == "MEL"
    debug = result["aggregator_debug"]["candidate_features"]
    assert debug["MEL"]["perception_anchor"] > debug["NEV"]["perception_anchor"]
    assert debug["NEV"]["evidence_correction"] > 0


def test_aggregator_allows_close_case_to_flip_with_consistent_support():
    state = create_case_state({"file": "case2.png", "metadata": {}, "text": ""})
    state.perception = {
        "ddx_candidates": [
            {"name": "MEL", "score": 0.58},
            {"name": "NEV", "score": 0.54},
        ],
        "uncertainty": {"level": "high"},
    }
    state.retrieval = {
        "retrieval_summary": {
            "support_labels": ["NEV"],
            "prototype_votes": {"NEV": 2},
            "confusion_votes": {"NEV": 1},
            "memory_consensus_label": "NEV",
            "retrieval_confidence": "high",
            "supports_top1": False,
        }
    }
    state.skill_outputs = {
        "compare_skill": {"winner": "NEV", "confidence": 0.95},
        "mel_nev_specialist_skill": {"recommendation": "NEV", "confidence": 1.0},
        "metadata_consistency_skill": {"supported_diagnoses": ["NEV"], "penalized_diagnoses": []},
    }

    result = DecisionAggregator().aggregate(state)

    assert result["final_label"] == "NEV"
    debug = result["aggregator_debug"]["candidate_features"]
    nev_correction = debug["NEV"].get("evidence_correction", 0.0)
    mel_correction = debug["MEL"].get("evidence_correction", 0.0)
    assert nev_correction > mel_correction


def test_aggregator_dampens_duplicate_specialist_support_for_same_label():
    state = create_case_state({"file": "case3.png", "metadata": {}, "text": ""})
    state.perception = {
        "ddx_candidates": [
            {"name": "SCC", "score": 0.62},
            {"name": "ACK", "score": 0.28},
            {"name": "BCC", "score": 0.10},
        ],
        "uncertainty": {"level": "low"},
    }
    state.retrieval = {"retrieval_summary": {"retrieval_confidence": "low"}}
    state.skill_outputs = {
        "ack_scc_specialist_skill": {"recommendation": "SCC", "confidence": 0.75},
        "bcc_scc_specialist_skill": {"recommendation": "SCC", "confidence": 0.75},
    }

    result = DecisionAggregator().aggregate(state)

    specialist_score = result["aggregator_debug"]["candidate_features"]["SCC"]["specialist_score"]
    assert specialist_score < 1.5
    assert specialist_score <= 1.0


def test_aggregator_strongly_dampens_overlapping_specialists_for_same_label():
    state = create_case_state({"file": "case4.png", "metadata": {}, "text": ""})
    state.perception = {
        "ddx_candidates": [
            {"name": "SCC", "score": 0.55},
            {"name": "ACK", "score": 0.53},
            {"name": "BCC", "score": 0.51},
        ],
        "uncertainty": {"level": "high"},
    }
    state.retrieval = {"retrieval_summary": {"retrieval_confidence": "low"}}
    state.skill_outputs = {
        "ack_scc_specialist_skill": {"recommendation": "SCC", "confidence": 0.9},
        "bcc_scc_specialist_skill": {"recommendation": "SCC", "confidence": 0.9},
    }

    result = DecisionAggregator().aggregate(state)

    specialist_score = result["aggregator_debug"]["candidate_features"]["SCC"]["specialist_score"]
    assert specialist_score < 0.95
