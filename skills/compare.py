from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState
from skills.base import BaseSkill


class CompareSkill(BaseSkill):
    name = "compare_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", []) or []
        candidates = ddx[:2]

        if len(candidates) == 0:
            result = {
                "pair": [],
                "winner": "UNKNOWN",
                "pair_scores": {},
                "reason": "no_candidates",
                "rationale": [],
            }
            state.skill_outputs[self.name] = result
            state.trace(self.name, "warning", "No candidates available for comparison")
            return result

        if len(candidates) == 1:
            only_name = self._norm_label(candidates[0].get("name"))
            result = {
                "pair": [only_name],
                "winner": only_name,
                "pair_scores": {only_name: 1.0},
                "reason": "single_candidate_only",
                "rationale": ["Only one candidate available."],
                "confidence": 0.6,
            }
            state.skill_outputs[self.name] = result
            state.trace(self.name, "success", "Single candidate fallback compare")
            return result

        c1 = self._norm_label(candidates[0].get("name"))
        c2 = self._norm_label(candidates[1].get("name"))

        scores: Dict[str, float] = {c1: 0.0, c2: 0.0}
        rationale: List[str] = []

        p1 = self._extract_candidate_score(candidates[0], default=1.0)
        p2 = self._extract_candidate_score(candidates[1], default=0.85)
        scores[c1] += p1
        scores[c2] += p2
        rationale.append(f"perception_support: {c1}={p1:.3f}, {c2}={p2:.3f}")

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = [
            self._norm_label(x) for x in retrieval_summary.get("support_labels", [])
        ]
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()

        retrieval_bonus_map = {"high": 0.6, "medium": 0.35, "low": 0.15}
        retrieval_bonus = retrieval_bonus_map.get(retrieval_confidence, 0.15)

        if c1 in support_labels:
            scores[c1] += retrieval_bonus
            rationale.append(
                f"retrieval_support: {c1} matched support_labels with bonus {retrieval_bonus:.2f}"
            )
        if c2 in support_labels:
            scores[c2] += retrieval_bonus
            rationale.append(
                f"retrieval_support: {c2} matched support_labels with bonus {retrieval_bonus:.2f}"
            )

        confusion_hits = state.retrieval.get("confusion_hits", []) or []
        for item in confusion_hits:
            pair = [self._norm_label(x) for x in item.get("pair", [])]
            if set(pair) == {c1, c2}:
                useful_skills = item.get("useful_skills", []) or []
                distinguishing_clues = item.get("distinguishing_clues", []) or []
                rationale.append(
                    f"confusion_memory_found: pair={pair}, useful_skills={useful_skills}, clues={distinguishing_clues[:3]}"
                )
                scores[c1] += 0.15
                scores[c2] += 0.15

        meta_scores = self._metadata_compare_bonus(state, c1, c2)
        scores[c1] += meta_scores[c1]
        scores[c2] += meta_scores[c2]
        if meta_scores[c1] != 0 or meta_scores[c2] != 0:
            rationale.append(
                f"metadata_adjustment: {c1}={meta_scores[c1]:+.2f}, {c2}={meta_scores[c2]:+.2f}"
            )

        planner_flags = state.planner.get("flags", {}) or {}
        if planner_flags.get("has_confusion_support", False):
            rationale.append("planner_flag: has_confusion_support=True")

        winner = c1 if scores[c1] >= scores[c2] else c2
        loser = c2 if winner == c1 else c1
        gap = abs(scores[c1] - scores[c2])

        if gap >= 0.75:
            confidence = 0.9
            reason = "strong_preference"
        elif gap >= 0.30:
            confidence = 0.75
            reason = "moderate_preference"
        else:
            confidence = 0.55
            reason = "weak_preference"

        result = {
            "pair": [c1, c2],
            "winner": winner,
            "loser": loser,
            "pair_scores": {
                c1: round(scores[c1], 4),
                c2: round(scores[c2], 4),
            },
            "gap": round(gap, 4),
            "reason": reason,
            "rationale": rationale,
            "confidence": round(confidence, 4),
        }

        state.skill_outputs[self.name] = result
        state.trace(
            self.name,
            "success",
            f"Compare completed: winner={winner}",
            payload={
                "pair": [c1, c2],
                "winner": winner,
                "gap": round(gap, 4),
                "reason": reason,
            },
        )
        return result

    def _metadata_compare_bonus(self, state: CaseState, c1: str, c2: str) -> Dict[str, float]:
        metadata = state.get_metadata()
        age = self._safe_int(metadata.get("age"))
        site = self._norm_text(
            metadata.get("location")
            or metadata.get("site")
            or metadata.get("anatomical_site")
        )
        history = self._norm_text(
            metadata.get("history")
            or metadata.get("clinical_history")
            or metadata.get("past_history")
        )
        has_bleed_or_elevation = any(
            token in history
            for token in ["bleed", "bleeding", "elevation"]
        )

        bonus = {c1: 0.0, c2: 0.0}
        if age is None and not site and not history:
            return bonus

        if {c1, c2} == {"MEL", "NEV"} and age is not None:
            if age >= 50:
                bonus["MEL"] += 0.15
            elif age <= 25:
                bonus["NEV"] += 0.10

        if {c1, c2} == {"ACK", "SCC"} and age is not None:
            if age >= 60:
                bonus["SCC"] += 0.12

        if {c1, c2} == {"BCC", "SCC"} and age is not None:
            if age >= 50:
                bonus["BCC"] += 0.08

        if site:
            if {c1, c2} == {"ACK", "SCC"} and self._site_matches(site, ["face", "scalp", "ear", "neck"]):
                bonus["SCC"] += 0.08

            if {c1, c2} == {"BCC", "SCC"}:
                if self._site_matches(site, ["nose", "face", "ear", "temple", "cheek"]):
                    bonus["BCC"] += 0.12 if has_bleed_or_elevation else 0.03
                if self._site_matches(site, ["trunk", "back", "chest", "abdomen"]):
                    bonus["BCC"] += 0.06 if has_bleed_or_elevation else 0.0

        return bonus

    def _extract_candidate_score(self, item: Dict[str, Any], default: float) -> float:
        for key in ["score", "probability", "confidence"]:
            value = item.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return default

    def _norm_label(self, value: Any) -> str:
        if value is None:
            return "UNKNOWN"
        text = str(value).strip().upper()
        return text if text else "UNKNOWN"

    def _norm_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def _safe_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _site_matches(self, site: str, keywords: List[str]) -> bool:
        if not site:
            return False
        normalized = site.replace("-", " ").replace("/", " ").strip().lower()
        tokens = [token for token in normalized.split() if token]
        return any(keyword in tokens for keyword in keywords)
