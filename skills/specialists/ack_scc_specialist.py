from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState
from skills.base import BaseSkill


class AckSccSpecialistSkill(BaseSkill):
    name = "ack_scc_specialist_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [
            self._norm_label(x.get("name"))
            for x in ddx[:5]
            if self._norm_label(x.get("name")) != "UNKNOWN"
        ]
        metadata = state.get_metadata()
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
        risk_cues = state.perception.get("risk_cues", {}) or {}
        malignant_cues = self._normalize_text_list(risk_cues.get("malignant_cues", []))
        ack_proxy_allowed = self._should_enable_ack_proxy(
            top_names=top_names,
            site=site,
            history=history,
            malignant_cues=malignant_cues,
        )

        if "SCC" not in top_names or ("ACK" not in top_names and not ack_proxy_allowed):
            result = {
                "target_group": ["ACK", "SCC"],
                "recommendation": None,
                "group_scores": {"ACK": 0.0, "SCC": 0.0},
                "confidence": 0.0,
                "rationale": ["ACK/SCC review not triggered by current perception or metadata context."],
                "reason": "ack_scc_not_supported_by_context",
            }
            state.skill_outputs[self.name] = result
            state.trace(self.name, "warning", "ACK/SCC specialist skipped: insufficient context")
            return result

        scores = {"ACK": 0.0, "SCC": 0.0}
        rationale: List[str] = []

        ack_item = self._find_candidate(ddx, "ACK")
        scc_item = self._find_candidate(ddx, "SCC")
        ack_score = self._extract_candidate_score(
            ack_item,
            default=0.32 if ack_proxy_allowed else 0.8,
        )
        scc_score = self._extract_candidate_score(scc_item, default=0.8)
        scores["ACK"] += ack_score
        scores["SCC"] += scc_score
        rationale.append(f"perception_score_support: ACK={ack_score:.3f}, SCC={scc_score:.3f}")
        if ack_proxy_allowed and not ack_item:
            rationale.append("ack_proxy_enabled_from_metadata_context")

        visual_cues = self._normalize_text_list(state.perception.get("visual_cues", []))
        ack_keywords = [
            "scaly",
            "rough",
            "keratotic",
            "hyperkeratotic",
            "thin plaque",
            "erythematous patch",
            "flat",
            "sun damage",
            "actinic",
        ]
        scc_keywords = [
            "ulcer",
            "ulcerated",
            "bleeding",
            "indurated",
            "nodule",
            "rapid growth",
            "pain",
            "tender",
            "thick plaque",
            "crust",
            "crusted",
            "elevation",
        ]

        ack_hits = self._count_keyword_hits(visual_cues, ack_keywords)
        scc_hits = self._count_keyword_hits(visual_cues, scc_keywords)
        if ack_hits > 0:
            bonus = min(0.75, 0.16 * ack_hits)
            scores["ACK"] += bonus
            rationale.append(f"visual_cues_support_ack: hits={ack_hits}, bonus={bonus:.2f}")
        if scc_hits > 0:
            bonus = min(0.85, 0.16 * scc_hits)
            scores["SCC"] += bonus
            rationale.append(f"visual_cues_support_scc: hits={scc_hits}, bonus={bonus:.2f}")

        suspicious_cues = self._normalize_text_list(risk_cues.get("suspicious_cues", []))
        age = self._safe_int(metadata.get("age"))
        has_strong_invasive_history = any(
            token in history
            for token in ["bleed", "bleeding", "rapid growth", "pain", "hurt", "ulcer", "ulcerated"]
        )
        has_mild_change_history = any(
            token in history
            for token in ["grew", "growing", "changed", "elevation"]
        )
        invasive_signal_count = scc_hits + len(malignant_cues) + (1 if has_strong_invasive_history else 0)

        if age is not None:
            if age >= 75:
                if has_strong_invasive_history:
                    scores["SCC"] += 0.22
                    scores["ACK"] += 0.08
                    rationale.append("older_age_with_invasive_history_supports_scc")
                else:
                    scores["ACK"] += 0.16
                    scores["SCC"] += 0.08
                    rationale.append("older_age_without_invasive_history_keeps_ack_plausible")
            elif age >= 60:
                if has_strong_invasive_history:
                    scores["SCC"] += 0.14
                    scores["ACK"] += 0.06
                    rationale.append("older_age_plus_invasive_history_slightly_favors_scc")
                else:
                    scores["ACK"] += 0.12
                    scores["SCC"] += 0.06
                    rationale.append("older_age_without_invasive_history_slightly_favors_ack")
            elif age <= 40:
                scores["ACK"] -= 0.08
                scores["SCC"] -= 0.12
                rationale.append("younger_age_weakly_penalizes_ack_and_scc")

        if site:
            sun_exposed_keywords = [
                "face",
                "scalp",
                "ear",
                "neck",
                "forehead",
                "cheek",
                "nose",
                "temple",
                "lip",
                "hand",
                "forearm",
            ]
            trunk_keywords = ["trunk", "back", "chest", "abdomen"]

            if self._site_matches(site, sun_exposed_keywords):
                scores["ACK"] += 0.18
                scores["SCC"] += 0.10
                rationale.append("sun_exposed_site_favors_ack_unless_invasive_features_exist")

                if not has_strong_invasive_history and scc_hits == 0 and len(malignant_cues) == 0:
                    scores["ACK"] += 0.12
                    rationale.append("sun_exposed_site_without_invasive_features_favors_ack")

            if self._site_matches(site, ["lip", "ear"]):
                scores["SCC"] += 0.18
                rationale.append("high_risk_sun_exposed_site_further_supports_scc")

            if self._site_matches(site, trunk_keywords):
                scores["ACK"] -= 0.08
                scores["SCC"] -= 0.16
                rationale.append("trunk_site_penalizes_scc_more_than_ack")

        if ack_proxy_allowed and not has_strong_invasive_history and not malignant_cues:
            scores["ACK"] += 0.18
            rationale.append("metadata_proxy_context_adds_ack_support")
        if has_mild_change_history and not has_strong_invasive_history:
            scores["ACK"] += 0.06
            rationale.append("mild_change_history_keeps_ack_plausible")

        malignancy_output = state.skill_outputs.get("malignancy_risk_skill", {}) or {}
        risk_level = self._norm_text(
            malignancy_output.get("risk_level")
            or malignancy_output.get("malignancy_risk")
            or malignancy_output.get("level")
        )
        preferred_label = self._norm_label(
            malignancy_output.get("preferred_label")
            or malignancy_output.get("recommendation")
            or malignancy_output.get("winner")
        )

        if risk_level == "high":
            if invasive_signal_count >= 2:
                scores["SCC"] += 0.24
                rationale.append("high_malignancy_risk_with_invasive_signals_supports_scc")
            else:
                scores["SCC"] += 0.08
                scores["ACK"] += 0.06
                rationale.append("high_risk_without_clear_invasion_keeps_ack_in_play")
        elif risk_level == "medium":
            if invasive_signal_count >= 1:
                scores["SCC"] += 0.12
                scores["ACK"] += 0.04
                rationale.append("medium_risk_with_invasive_signal_slightly_favors_scc")
            else:
                scores["ACK"] += 0.08
                rationale.append("medium_risk_without_invasive_signal_slightly_favors_ack")
        elif risk_level == "low":
            scores["ACK"] += 0.08
            rationale.append("low_malignancy_risk_slightly_favors_ack")

        if preferred_label == "SCC" and invasive_signal_count >= 1:
            scores["SCC"] += 0.14
            rationale.append("malignancy_skill_preferred_label_supports_scc")
        elif preferred_label == "ACK":
            scores["ACK"] += 0.10
            rationale.append("malignancy_skill_preferred_label_supports_ack")

        if malignant_cues and invasive_signal_count >= 1:
            bonus = min(0.36, 0.10 * len(malignant_cues))
            scores["SCC"] += bonus
            rationale.append(f"perception_malignant_cues_support_scc: n={len(malignant_cues)}, bonus={bonus:.2f}")

        if suspicious_cues and invasive_signal_count >= 2:
            bonus = min(0.16, 0.04 * len(suspicious_cues))
            scores["SCC"] += bonus
            rationale.append(f"perception_suspicious_cues_support_scc: n={len(suspicious_cues)}, bonus={bonus:.2f}")

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = [self._norm_label(x) for x in retrieval_summary.get("support_labels", [])]
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()

        retrieval_bonus_map = {"high": 0.45, "medium": 0.25, "low": 0.10}
        retrieval_bonus = retrieval_bonus_map.get(retrieval_confidence, 0.10)
        if "ACK" in support_labels:
            scores["ACK"] += retrieval_bonus
            rationale.append(f"retrieval_supports_ack: bonus={retrieval_bonus:.2f}")
        if "SCC" in support_labels:
            scores["SCC"] += retrieval_bonus
            rationale.append(f"retrieval_supports_scc: bonus={retrieval_bonus:.2f}")

        confusion_hits = state.retrieval.get("confusion_hits", []) or []
        for item in confusion_hits:
            pair = {self._norm_label(x) for x in item.get("pair", [])}
            if pair == {"ACK", "SCC"}:
                scores["ACK"] += 0.08
                scores["SCC"] += 0.08
                rationale.append("ack_scc_confusion_memory_found")
                break

        compare_output = state.skill_outputs.get("compare_skill", {}) or {}
        compare_winner = self._norm_label(
            compare_output.get("winner")
            or compare_output.get("recommendation")
            or compare_output.get("final_choice")
        )
        compare_conf = self._safe_float(compare_output.get("confidence"), default=0.0)
        if compare_winner == "ACK":
            bonus = 0.10 + min(0.15, compare_conf * 0.15)
            scores["ACK"] += bonus
            rationale.append(f"compare_skill_supports_ack: bonus={bonus:.2f}")
        elif compare_winner == "NEV" and ack_proxy_allowed:
            bonus = 0.08 + min(0.12, compare_conf * 0.12)
            scores["ACK"] += bonus
            rationale.append(f"compare_skill_nev_fallback_keeps_ack_alive: bonus={bonus:.2f}")
        elif compare_winner == "SCC":
            bonus = 0.10 + min(0.15, compare_conf * 0.15)
            scores["SCC"] += bonus
            rationale.append(f"compare_skill_supports_scc: bonus={bonus:.2f}")

        meta_output = state.skill_outputs.get("metadata_consistency_skill", {}) or {}
        supported = [
            self._norm_label(x)
            for x in (meta_output.get("supported_diagnoses", []) or meta_output.get("supported_labels", []) or [])
        ]
        penalized = [
            self._norm_label(x)
            for x in (meta_output.get("penalized_diagnoses", []) or meta_output.get("penalized_labels", []) or [])
        ]

        if "ACK" in supported:
            scores["ACK"] += 0.16
            rationale.append("metadata_consistency_supports_ack")
        if "SCC" in supported:
            scores["SCC"] += 0.12
            rationale.append("metadata_consistency_supports_scc")
        if "ACK" in penalized:
            scores["ACK"] -= 0.10
            rationale.append("metadata_consistency_penalizes_ack")
        if "SCC" in penalized:
            scores["SCC"] -= 0.14
            rationale.append("metadata_consistency_penalizes_scc")

        recommendation = "ACK" if scores["ACK"] >= scores["SCC"] else "SCC"
        loser = "SCC" if recommendation == "ACK" else "ACK"
        gap = abs(scores["ACK"] - scores["SCC"])

        if gap >= 0.90:
            confidence = 0.90
            reason = "strong_specialist_preference"
        elif gap >= 0.35:
            confidence = 0.75
            reason = "moderate_specialist_preference"
        else:
            confidence = 0.58
            reason = "weak_specialist_preference"

        result = {
            "target_group": ["ACK", "SCC"],
            "recommendation": recommendation,
            "loser": loser,
            "group_scores": {
                "ACK": round(scores["ACK"], 4),
                "SCC": round(scores["SCC"], 4),
            },
            "confidence": round(confidence, 4),
            "gap": round(gap, 4),
            "reason": reason,
            "used_ack_proxy": bool(ack_proxy_allowed and not ack_item),
            "rationale": rationale[:24],
        }

        state.skill_outputs[self.name] = result
        state.trace(
            self.name,
            "success",
            f"ACK/SCC specialist completed: recommendation={recommendation}",
            payload={
                "recommendation": recommendation,
                "confidence": round(confidence, 4),
                "gap": round(gap, 4),
            },
        )
        return result

    def _find_candidate(self, ddx: List[Dict[str, Any]], target: str) -> Dict[str, Any]:
        target = self._norm_label(target)
        for item in ddx:
            if self._norm_label(item.get("name")) == target:
                return item
        return {}

    def _extract_candidate_score(self, item: Dict[str, Any], default: float) -> float:
        for key in ["score", "probability", "confidence"]:
            value = item.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return default

    def _count_keyword_hits(self, cues: List[str], keywords: List[str]) -> int:
        hits = 0
        for cue in cues:
            cue_lower = cue.lower()
            for kw in keywords:
                if kw in cue_lower:
                    hits += 1
                    break
        return hits

    def _normalize_text_list(self, items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for x in items:
            text = str(x).strip()
            if text and text not in out:
                out.append(text)
        return out

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

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _should_enable_ack_proxy(
        self,
        top_names: List[str],
        site: str,
        history: str,
        malignant_cues: List[str],
    ) -> bool:
        if "ACK" in top_names or "SCC" not in top_names:
            return False
        if not self._site_matches(
            site,
            [
                "neck",
                "hand",
                "forearm",
            ],
        ):
            return False
        strong_invasive_history = any(
            token in history
            for token in ["bleed", "bleeding", "rapid growth", "pain", "hurt", "ulcer", "ulcerated"]
        )
        return not strong_invasive_history and len(malignant_cues) == 0

    def _site_matches(self, site: str, keywords: List[str]) -> bool:
        if not site:
            return False
        normalized = site.replace("-", " ").replace("/", " ").strip().lower()
        tokens = [token for token in normalized.split() if token]
        return any(keyword in tokens for keyword in keywords)
