from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.state import CaseState
from memory.skill_index import sigmoid


class LearnableRuleScorer:
    def __init__(
        self,
        *,
        learning_rate: float = 0.03,
        threshold: float = 0.52,
    ) -> None:
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.weights: Dict[str, float] = {
            "bias": -0.1,
            "priority": 0.45,
            "source_count": 0.35,
            "compression_high": 0.3,
            "compression_medium": 0.15,
            "uncertainty_high": 0.25,
            "uncertainty_medium": 0.1,
            "retrieval_low": 0.18,
            "retrieval_medium": 0.06,
            "retrieval_high": -0.05,
            "supports_top1": -0.2,
            "has_confusion_support": 0.14,
            "requires_all_match": 0.5,
            "requires_any_overlap": 0.35,
            "specialist_hint": 0.12,
        }
        self.rule_biases: Dict[str, float] = {}

    def score_rules(
        self,
        state: CaseState,
        *,
        rule_hits: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        rule_hits = list(rule_hits or state.retrieval.get("rule_hits", []) or [])
        recommended_skill_scores: Dict[str, float] = {}
        scored_rules: Dict[str, Dict[str, Any]] = {}
        applied_rules: List[str] = []

        for item in rule_hits:
            rule_name = str(item.get("rule_name", "")).strip()
            if not rule_name:
                continue
            features = self.extract_features(state, item)
            bias = float(self.rule_biases.get(rule_name, 0.0))
            logit = sum(float(self.weights.get(k, 0.0)) * float(v) for k, v in features.items()) + bias
            probability = sigmoid(logit)
            suggested_skills = [
                str(x).strip()
                for x in ((item.get("action", {}) or {}).get("suggested_skills", []) or [])
                if str(x).strip()
            ]
            priority = self._safe_float(item.get("priority"), default=1.0)
            selected = probability >= self.threshold
            if selected:
                applied_rules.append(rule_name)
                base_gain = probability * (1.0 + 0.05 * priority)
                for skill_name in suggested_skills:
                    recommended_skill_scores[skill_name] = round(
                        recommended_skill_scores.get(skill_name, 0.0) + base_gain,
                        4,
                    )

            scored_rules[rule_name] = {
                "rule_name": rule_name,
                "probability": round(probability, 4),
                "logit": round(logit, 4),
                "selected": selected,
                "priority": round(priority, 4),
                "suggested_skills": suggested_skills,
                "features": {k: round(float(v), 4) for k, v in features.items() if abs(float(v)) > 1e-8},
                "bias": round(bias, 4),
            }

        ranked_skills = sorted(
            recommended_skill_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "rule_scores": scored_rules,
            "recommended_skill_scores": {k: round(v, 4) for k, v in ranked_skills},
            "recommended_skills": [name for name, _ in ranked_skills],
            "applied_rules": applied_rules,
        }

    def extract_features(self, state: CaseState, rule_item: Dict[str, Any]) -> Dict[str, float]:
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        retrieval_confidence = str(retrieval_summary.get("retrieval_confidence", "low")).lower()
        conditions = rule_item.get("trigger_conditions", {}) or {}
        action = rule_item.get("action", {}) or {}
        top_names = set(state.get_top_ddx_names(top_k=5))
        requires_all = {
            str(x).strip().upper() for x in conditions.get("requires_all_diseases", []) if str(x).strip()
        }
        requires_any = [
            str(x).strip().upper() for x in conditions.get("requires_any_disease", []) if str(x).strip()
        ]
        overlap_any = len(top_names.intersection(set(requires_any)))
        overlap_ratio = overlap_any / max(1, len(set(requires_any)))
        compression = str(rule_item.get("compression_level", "")).strip().lower()
        suggested_skills = [
            str(x).strip() for x in action.get("suggested_skills", []) if str(x).strip()
        ]

        return {
            "bias": 1.0,
            "priority": min(1.0, self._safe_float(rule_item.get("priority"), default=1.0) / 5.0),
            "source_count": min(1.0, self._safe_float(rule_item.get("source_count"), default=0.0) / 5.0),
            "compression_high": 1.0 if compression == "high" else 0.0,
            "compression_medium": 1.0 if compression == "medium" else 0.0,
            "uncertainty_high": 1.0 if state.get_uncertainty_level() == "high" else 0.0,
            "uncertainty_medium": 1.0 if state.get_uncertainty_level() == "medium" else 0.0,
            "retrieval_low": 1.0 if retrieval_confidence == "low" else 0.0,
            "retrieval_medium": 1.0 if retrieval_confidence == "medium" else 0.0,
            "retrieval_high": 1.0 if retrieval_confidence == "high" else 0.0,
            "supports_top1": 1.0 if retrieval_summary.get("supports_top1", False) else 0.0,
            "has_confusion_support": 1.0 if retrieval_summary.get("has_confusion_support", False) else 0.0,
            "requires_all_match": 1.0 if requires_all and requires_all.issubset(top_names) else 0.0,
            "requires_any_overlap": round(overlap_ratio, 4),
            "specialist_hint": 1.0 if any("specialist" in skill for skill in suggested_skills) else 0.0,
        }

    def update_from_case(self, state: CaseState) -> Dict[str, Any]:
        planner_payload = state.planner or {}
        rule_scores = planner_payload.get("rule_scores", {}) or {}
        if not rule_scores:
            return {"updated": False, "reason": "no_rule_scores"}

        true_label = str((state.case_info or {}).get("true_label", "")).strip().upper()
        pred_label = str(
            (state.final_decision or {}).get("final_label")
            or (state.final_decision or {}).get("diagnosis")
            or ""
        ).strip().upper()
        is_correct = bool(true_label) and pred_label == true_label
        selected_skills = set(state.selected_skills)

        updates: List[Dict[str, Any]] = []
        for rule_name, info in rule_scores.items():
            prediction = self._safe_float(info.get("probability"), default=0.0)
            suggested_skills = set(info.get("suggested_skills", []) or [])
            rule_applied = bool(suggested_skills.intersection(selected_skills)) or bool(info.get("selected", False))
            if not rule_applied:
                continue

            target = 1.0 if is_correct else 0.0
            error = target - prediction
            features = info.get("features", {}) or {}
            for key, value in features.items():
                self.weights[key] = float(self.weights.get(key, 0.0)) + self.learning_rate * error * float(value)
            self.rule_biases[rule_name] = float(self.rule_biases.get(rule_name, 0.0)) + self.learning_rate * error
            updates.append(
                {
                    "rule_name": rule_name,
                    "prediction": round(prediction, 4),
                    "target": round(target, 4),
                    "error": round(error, 4),
                }
            )

        return {
            "updated": bool(updates),
            "is_correct": is_correct,
            "updated_rules": updates,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "learning_rate": self.learning_rate,
            "threshold": self.threshold,
            "weights": {k: round(float(v), 6) for k, v in sorted(self.weights.items())},
            "rule_biases": {k: round(float(v), 6) for k, v in sorted(self.rule_biases.items())},
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if int(payload.get("version", 0) or 0) != 1:
            return
        if payload.get("learning_rate") is not None:
            self.learning_rate = float(payload["learning_rate"])
        if payload.get("threshold") is not None:
            self.threshold = float(payload["threshold"])
        weights = payload.get("weights", {}) or {}
        if weights:
            self.weights = {str(k): float(v) for k, v in weights.items()}
        rule_biases = payload.get("rule_biases", {}) or {}
        if rule_biases:
            self.rule_biases = {str(k): float(v) for k, v in rule_biases.items()}

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
