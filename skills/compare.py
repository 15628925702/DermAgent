from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState
from skills.base import BaseSkill


class CompareSkill(BaseSkill):
    name = "compare_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", []) or []
        candidates = ddx[:3]

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
                "candidates": [only_name],
                "winner": only_name,
                "pair_scores": {only_name: 1.0},
                "reason": "single_candidate_only",
                "rationale": ["Only one candidate available."],
                "confidence": 0.6,
            }
            state.skill_outputs[self.name] = result
            state.trace(self.name, "success", "Single candidate fallback compare")
            return result

        candidate_names: List[str] = []
        scores: Dict[str, float] = {}
        rationale: List[str] = []

        for idx, item in enumerate(candidates):
            name = self._norm_label(item.get("name"))
            if name == "UNKNOWN" or name in scores:
                continue
            default_score = max(0.55, 1.0 - 0.15 * idx)
            score = self._extract_candidate_score(item, default=default_score)
            candidate_names.append(name)
            scores[name] = score

        rationale.append(
            "perception_support: "
            + ", ".join(f"{name}={scores[name]:.3f}" for name in candidate_names)
        )

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = [
            self._norm_label(x) for x in retrieval_summary.get("support_labels", [])
        ]
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()

        retrieval_bonus_map = {"high": 0.6, "medium": 0.35, "low": 0.15}
        retrieval_bonus = retrieval_bonus_map.get(retrieval_confidence, 0.15)

        for name in candidate_names:
            if name in support_labels:
                scores[name] += retrieval_bonus
                rationale.append(
                    f"retrieval_support: {name} matched support_labels with bonus {retrieval_bonus:.2f}"
                )

        confusion_hits = state.retrieval.get("confusion_hits", []) or []
        for item in confusion_hits:
            pair = [self._norm_label(x) for x in item.get("pair", [])]
            pair_set = set(pair)
            overlap = pair_set.intersection(set(candidate_names))
            if len(overlap) >= 2:
                useful_skills = item.get("useful_skills", []) or []
                distinguishing_clues = item.get("distinguishing_clues", []) or []
                rationale.append(
                    f"confusion_memory_found: pair={pair}, useful_skills={useful_skills}, clues={distinguishing_clues[:3]}"
                )
                for name in overlap:
                    scores[name] += 0.15

        for idx, left in enumerate(candidate_names):
            for right in candidate_names[idx + 1:]:
                meta_scores = self._metadata_compare_bonus(state, left, right)
                scores[left] += meta_scores[left]
                scores[right] += meta_scores[right]
                if meta_scores[left] != 0 or meta_scores[right] != 0:
                    rationale.append(
                        f"metadata_adjustment: {left}={meta_scores[left]:+.2f}, {right}={meta_scores[right]:+.2f}"
                    )

        planner_flags = state.planner.get("flags", {}) or {}
        if planner_flags.get("has_confusion_support", False):
            rationale.append("planner_flag: has_confusion_support=True")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        winner = ranked[0][0]
        loser = ranked[1][0] if len(ranked) > 1 else ""
        gap = abs(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else ranked[0][1]

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
            "pair": candidate_names[:2],
            "candidates": candidate_names,
            "winner": winner,
            "loser": loser,
            "pair_scores": {
                name: round(scores[name], 4)
                for name in candidate_names
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
                "pair": candidate_names[:2],
                "candidates": candidate_names,
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
