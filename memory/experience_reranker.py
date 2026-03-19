from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.state import CaseState


class UtilityAwareExperienceReranker:
    """
    Lightweight utility scorer for second-version retrieval.

    The interface stays simple on purpose so we can later swap this
    scorer with a learned neural reranker without changing callers.
    """

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = {
            "base_score": 0.65,
            "supports_top1": 0.9,
            "label_overlap": 0.55,
            "confusion_pair_match": 0.9,
            "recommended_skill_overlap": 0.4,
            "metadata_hint": 0.2,
            "source_count": 0.25,
            "compression_level": 0.2,
        }
        if weights:
            self.weights.update(weights)

    def rerank(
        self,
        state: CaseState,
        *,
        raw_case_hits: List[Dict[str, Any]],
        prototype_hits: List[Dict[str, Any]],
        confusion_hits: List[Dict[str, Any]],
        rule_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        rule_skill_hints = self._collect_rule_skill_hints(rule_hits)

        raw_case_hits = self._sort_hits(
            hits=raw_case_hits,
            state=state,
            experience_type="raw_case",
            rule_skill_hints=rule_skill_hints,
        )
        prototype_hits = self._sort_hits(
            hits=prototype_hits,
            state=state,
            experience_type="prototype",
            rule_skill_hints=rule_skill_hints,
        )
        confusion_hits = self._sort_hits(
            hits=confusion_hits,
            state=state,
            experience_type="confusion",
            rule_skill_hints=rule_skill_hints,
        )

        return {
            "raw_case_hits": raw_case_hits,
            "prototype_hits": prototype_hits,
            "confusion_hits": confusion_hits,
            "rule_hits": rule_hits,
            "reranker_debug": {
                "weights": dict(self.weights),
                "rule_skill_hints": rule_skill_hints,
            },
        }

    def _sort_hits(
        self,
        *,
        hits: List[Dict[str, Any]],
        state: CaseState,
        experience_type: str,
        rule_skill_hints: List[str],
    ) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in hits:
            utility = self._score_hit(
                state=state,
                item=item,
                experience_type=experience_type,
                rule_skill_hints=rule_skill_hints,
            )
            enriched = dict(item)
            enriched["utility_score"] = round(utility, 4)
            scored.append((utility, enriched))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored]

    def _score_hit(
        self,
        *,
        state: CaseState,
        item: Dict[str, Any],
        experience_type: str,
        rule_skill_hints: List[str],
    ) -> float:
        top_names = state.get_top_ddx_names(top_k=3)
        top_name = top_names[0] if top_names else ""
        metadata = state.get_metadata()

        base_score = self._safe_float(item.get("_score"), default=0.0)
        label_overlap = 0.0
        supports_top1 = 0.0
        confusion_pair_match = 0.0
        recommended_skill_overlap = 0.0
        metadata_hint = 0.0
        source_count_signal = self._support_count_signal(item.get("source_count"))
        compression_level_signal = self._compression_level_signal(item.get("compression_level"))

        if experience_type == "raw_case":
            label = str((item.get("final_decision", {}) or {}).get("diagnosis", "")).strip().upper()
            if label and label in top_names:
                label_overlap = 1.0
            if label and label == top_name:
                supports_top1 = 1.0

        elif experience_type == "prototype":
            label = str(item.get("disease", "")).strip().upper()
            if label and label in top_names:
                label_overlap = 1.0
            if label and label == top_name:
                supports_top1 = 1.0
            recommended = [str(x).strip() for x in item.get("recommended_skills", []) if str(x).strip()]
            if set(recommended).intersection(rule_skill_hints):
                recommended_skill_overlap = 1.0

        elif experience_type == "confusion":
            pair = {str(x).strip().upper() for x in item.get("pair", []) if str(x).strip()}
            if pair and pair.issubset(set(top_names)):
                confusion_pair_match = 1.0
            useful_skills = [str(x).strip() for x in item.get("useful_skills", []) if str(x).strip()]
            if set(useful_skills).intersection(rule_skill_hints):
                recommended_skill_overlap = 1.0
            if top_name and top_name in set((item.get("label_votes", {}) or {}).keys()):
                supports_top1 = 1.0

        site = str(
            metadata.get("location")
            or metadata.get("site")
            or metadata.get("anatomical_site")
            or ""
        ).strip()
        if site:
            metadata_hint = 1.0

        return (
            self.weights["base_score"] * base_score
            + self.weights["supports_top1"] * supports_top1
            + self.weights["label_overlap"] * label_overlap
            + self.weights["confusion_pair_match"] * confusion_pair_match
            + self.weights["recommended_skill_overlap"] * recommended_skill_overlap
            + self.weights["metadata_hint"] * metadata_hint
            + self.weights["source_count"] * source_count_signal
            + self.weights["compression_level"] * compression_level_signal
        )

    def _collect_rule_skill_hints(self, rule_hits: List[Dict[str, Any]]) -> List[str]:
        hints: List[str] = []
        for item in rule_hits:
            action = item.get("action", {}) or {}
            for skill_name in action.get("suggested_skills", []) or []:
                skill_name = str(skill_name).strip()
                if skill_name and skill_name not in hints:
                    hints.append(skill_name)
        return hints

    def _support_count_signal(self, value: Any) -> float:
        count = self._safe_float(value, default=0.0)
        if count >= 5:
            return 1.0
        if count >= 2:
            return 0.6
        if count >= 1:
            return 0.3
        return 0.0

    def _compression_level_signal(self, value: Any) -> float:
        level = str(value or "").strip().lower()
        if level == "high":
            return 1.0
        if level == "medium":
            return 0.6
        if level == "low":
            return 0.3
        return 0.0

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
