from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.state import CaseState


class LearnableFinalScorer:
    def __init__(
        self,
        *,
        learning_rate: float = 0.01,
        margin_target: float = 0.45,
    ) -> None:
        self.learning_rate = learning_rate
        self.margin_target = margin_target
        self.weights: Dict[str, float] = {
            "bias": 0.0,
            "base_total": 1.0,
            "raw_total": 0.1,
            "perception_score": 0.0,
            "perception_anchor": 0.35,
            "retrieval_score": 0.0,
            "prototype_score": 0.0,
            "confusion_score": 0.0,
            "compare_score": 0.0,
            "specialist_score": 0.0,
            "metadata_score": 0.0,
            "malignancy_score": 0.0,
            "memory_consensus_score": 0.0,
            "retrieval_correction": 0.12,
            "skill_correction": 0.18,
            "multi_source_bonus": 0.08,
            "off_perception_penalty": 0.22,
            "evidence_correction_raw": 0.05,
            "evidence_correction": 0.25,
            "is_perception_top1": 0.0,
            "in_support_labels": 0.0,
            "matches_memory_consensus": 0.0,
            "malignant_candidate": 0.0,
            "retrieval_confidence_high": 0.0,
            "retrieval_confidence_medium": 0.0,
            "retrieval_confidence_low": 0.0,
        }

    def rank_candidates(
        self,
        candidate_features: Dict[str, Dict[str, float]],
    ) -> Tuple[List[Tuple[str, float]], Dict[str, Dict[str, float]]]:
        scored: List[Tuple[str, float]] = []
        feature_debug: Dict[str, Dict[str, float]] = {}

        for label, features in candidate_features.items():
            score = self.score_candidate(features)
            scored.append((label, score))
            feature_debug[label] = {
                key: round(float(value), 4)
                for key, value in sorted(features.items())
                if abs(float(value)) > 1e-8
            }

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored, feature_debug

    def score_candidate(self, features: Dict[str, float]) -> float:
        score = 0.0
        for name, value in features.items():
            score += float(self.weights.get(name, 0.0)) * float(value)
        return score

    def update_from_case(self, state: CaseState) -> Dict[str, Any]:
        final_decision = state.final_decision or {}
        aggregator_debug = final_decision.get("aggregator_debug", {}) or {}
        candidate_features = aggregator_debug.get("candidate_features", {}) or {}
        if not candidate_features:
            return {"updated": False, "reason": "missing_candidate_features"}

        true_label = str((state.case_info or {}).get("true_label", "")).strip().upper()
        if not true_label:
            return {"updated": False, "reason": "missing_true_label"}
        if true_label not in candidate_features:
            return {
                "updated": False,
                "reason": "true_label_not_in_candidates",
                "true_label": true_label,
            }

        ranked, _ = self.rank_candidates(candidate_features)
        if not ranked:
            return {"updated": False, "reason": "no_candidates"}

        predicted_label = str(final_decision.get("final_label") or final_decision.get("diagnosis") or "").strip().upper()
        if not predicted_label:
            predicted_label = ranked[0][0]

        if predicted_label == true_label:
            return {
                "updated": False,
                "reason": "prediction_already_correct",
                "true_label": true_label,
                "predicted_label": predicted_label,
            }

        true_features = candidate_features.get(true_label, {}) or {}
        predicted_features = candidate_features.get(predicted_label, {}) or {}
        true_score = self.score_candidate(true_features)
        predicted_score = self.score_candidate(predicted_features)
        margin = true_score - predicted_score
        error = max(1.0, self.margin_target - margin)

        updates: Dict[str, float] = {}
        all_keys = set(true_features.keys()) | set(predicted_features.keys())
        for key in all_keys:
            delta = self.learning_rate * error * (
                float(true_features.get(key, 0.0)) - float(predicted_features.get(key, 0.0))
            )
            if abs(delta) <= 1e-8:
                continue
            self.weights[key] = float(self.weights.get(key, 0.0)) + delta
            updates[key] = round(delta, 6)

        return {
            "updated": True,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "true_score": round(true_score, 4),
            "predicted_score": round(predicted_score, 4),
            "margin": round(margin, 4),
            "target_margin": round(self.margin_target, 4),
            "error": round(error, 4),
            "weight_updates": updates,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 3,
            "enabled": True,
            "learning_rate": self.learning_rate,
            "margin_target": self.margin_target,
            "weights": {
                key: round(float(value), 6)
                for key, value in sorted(self.weights.items())
            },
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        version = int(payload.get("version", 0) or 0)
        if version not in {2, 3}:
            return
        if not payload.get("enabled", False):
            return
        if payload.get("learning_rate") is not None:
            self.learning_rate = float(payload["learning_rate"])
        if payload.get("margin_target") is not None:
            self.margin_target = float(payload["margin_target"])
        weights = payload.get("weights", {}) or {}
        if weights:
            merged = dict(self.weights)
            for key, value in weights.items():
                merged[str(key)] = float(value)
            self.weights = merged
