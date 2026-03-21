from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState
from skills.base import BaseSkill


class BccSekSpecialistSkill(BaseSkill):
    name = "bcc_sek_specialist_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [
            self._norm_label(item.get("name"))
            for item in ddx[:5]
            if self._norm_label(item.get("name")) != "UNKNOWN"
        ]

        if not {"BCC", "SEK"}.issubset(set(top_names)):
            result = {
                "target_group": ["BCC", "SEK"],
                "recommendation": None,
                "supports": None,
                "supported_label": None,
                "group_scores": {"BCC": 0.0, "SEK": 0.0},
                "applicable": 0.0,
                "evidence_items": [],
                "local_decision": {
                    "supports": None,
                    "opposes": None,
                    "confidence": 0.0,
                    "applicable": 0.0,
                    "reason": "bcc_sek_not_present",
                },
                "confidence": 0.0,
                "reason": "bcc_sek_not_present",
                "rationale": ["BCC/SEK specialist not triggered by current candidates."],
            }
            state.skill_outputs[self.name] = result
            state.trace(self.name, "warning", "BCC/SEK specialist skipped: pair not present")
            return result

        scores = {"BCC": 0.0, "SEK": 0.0}
        rationale: List[str] = []

        bcc_item = self._find_candidate(ddx, "BCC")
        sek_item = self._find_candidate(ddx, "SEK")
        bcc_score = self._extract_candidate_score(bcc_item, default=0.8)
        sek_score = self._extract_candidate_score(sek_item, default=0.8)
        scores["BCC"] += bcc_score
        scores["SEK"] += sek_score
        rationale.append(f"perception_score_support: BCC={bcc_score:.3f}, SEK={sek_score:.3f}")

        visual_cues = self._normalize_text_list(state.perception.get("visual_cues", []))
        bcc_keywords = [
            "pearly",
            "rolled border",
            "telangiectasia",
            "translucent",
            "ulcer",
            "shiny",
            "nodular",
        ]
        sek_keywords = [
            "waxy",
            "stuck on",
            "verrucous",
            "cerebriform",
            "keratotic",
            "milia like",
            "comedo like",
            "sharply demarcated",
        ]
        bcc_hits = self._count_keyword_hits(visual_cues, bcc_keywords)
        sek_hits = self._count_keyword_hits(visual_cues, sek_keywords)
        if bcc_hits:
            bonus = min(0.8, 0.16 * bcc_hits)
            scores["BCC"] += bonus
            rationale.append(f"visual_cues_support_bcc: hits={bcc_hits}, bonus={bonus:.2f}")
        if sek_hits:
            bonus = min(0.8, 0.16 * sek_hits)
            scores["SEK"] += bonus
            rationale.append(f"visual_cues_support_sek: hits={sek_hits}, bonus={bonus:.2f}")

        metadata = state.get_metadata()
        age = self._safe_int(metadata.get("age"))
        site = self._norm_text(metadata.get("location") or metadata.get("site") or metadata.get("anatomical_site"))
        history = self._norm_text(metadata.get("history") or metadata.get("clinical_history") or metadata.get("past_history"))
        if age is not None and age >= 55:
            scores["SEK"] += 0.1
            scores["BCC"] += 0.06
            rationale.append("older_age_keeps_sek_and_bcc_plausible")
        if self._site_matches(site, ["nose", "cheek", "face", "temple", "ear"]):
            scores["BCC"] += 0.16
            rationale.append("classic_bcc_site_supports_bcc")
        if self._site_matches(site, ["trunk", "back", "chest", "abdomen"]):
            scores["SEK"] += 0.14
            rationale.append("trunk_site_supports_sek")
        if any(token in history for token in ["itch", "changed"]):
            scores["SEK"] += 0.08
            rationale.append("benign_irritation_history_supports_sek")
        if any(token in history for token in ["bleed", "bleeding", "elevation"]):
            scores["BCC"] += 0.12
            rationale.append("bleeding_or_elevation_history_supports_bcc")

        risk_output = state.skill_outputs.get("malignancy_risk_skill", {}) or {}
        preferred_label = self._norm_label(
            risk_output.get("preferred_label")
            or risk_output.get("recommendation")
            or risk_output.get("winner")
        )
        risk_level = self._norm_text(risk_output.get("risk_level") or risk_output.get("malignancy_risk"))
        if preferred_label == "BCC":
            scores["BCC"] += 0.14
            rationale.append("malignancy_skill_supports_bcc")
        if risk_level == "low":
            scores["SEK"] += 0.08
            rationale.append("low_malignancy_risk_supports_sek")

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = [self._norm_label(x) for x in retrieval_summary.get("support_labels", [])]
        retrieval_confidence = self._norm_text(retrieval_summary.get("retrieval_confidence", "low"))
        retrieval_bonus = {"high": 0.42, "medium": 0.24, "low": 0.1}.get(retrieval_confidence, 0.1)
        if "BCC" in support_labels:
            scores["BCC"] += retrieval_bonus
            rationale.append(f"retrieval_supports_bcc: bonus={retrieval_bonus:.2f}")
        if "SEK" in support_labels:
            scores["SEK"] += retrieval_bonus
            rationale.append(f"retrieval_supports_sek: bonus={retrieval_bonus:.2f}")

        compare_output = state.skill_outputs.get("compare_skill", {}) or {}
        compare_winner = self._norm_label(compare_output.get("winner") or compare_output.get("recommendation"))
        compare_confidence = self._safe_float(compare_output.get("confidence"), default=0.0)
        if compare_winner == "BCC":
            scores["BCC"] += 0.1 + min(0.15, compare_confidence * 0.15)
            rationale.append("compare_skill_supports_bcc")
        elif compare_winner == "SEK":
            scores["SEK"] += 0.1 + min(0.15, compare_confidence * 0.15)
            rationale.append("compare_skill_supports_sek")

        meta_output = state.skill_outputs.get("metadata_consistency_skill", {}) or {}
        supported = [
            self._norm_label(x)
            for x in (meta_output.get("supported_diagnoses", []) or meta_output.get("supported_labels", []) or [])
        ]
        penalized = [
            self._norm_label(x)
            for x in (meta_output.get("penalized_diagnoses", []) or meta_output.get("penalized_labels", []) or [])
        ]
        if "BCC" in supported:
            scores["BCC"] += 0.14
            rationale.append("metadata_consistency_supports_bcc")
        if "SEK" in supported:
            scores["SEK"] += 0.14
            rationale.append("metadata_consistency_supports_sek")
        if "BCC" in penalized:
            scores["BCC"] -= 0.1
            rationale.append("metadata_consistency_penalizes_bcc")
        if "SEK" in penalized:
            scores["SEK"] -= 0.1
            rationale.append("metadata_consistency_penalizes_sek")

        recommendation = "BCC" if scores["BCC"] >= scores["SEK"] else "SEK"
        loser = "SEK" if recommendation == "BCC" else "BCC"
        gap = abs(scores["BCC"] - scores["SEK"])
        confidence = 0.9 if gap >= 0.9 else 0.75 if gap >= 0.35 else 0.58
        reason = "strong_specialist_preference" if gap >= 0.9 else "moderate_specialist_preference" if gap >= 0.35 else "weak_specialist_preference"

        result = {
            "target_group": ["BCC", "SEK"],
            "recommendation": recommendation,
            "supports": recommendation,
            "supported_label": recommendation,
            "loser": loser,
            "group_scores": {name: round(value, 4) for name, value in scores.items()},
            "applicable": 1.0,
            "supporting_evidence": rationale[:24],
            "evidence_items": self._build_evidence_items(
                scores=scores,
                recommendation=recommendation,
                loser=loser,
                confidence=confidence,
                gap=gap,
                rationale=rationale,
            ),
            "confidence": round(confidence, 4),
            "gap": round(gap, 4),
            "reason": reason,
            "rationale": rationale[:24],
            "local_decision": {
                "supports": recommendation,
                "opposes": loser,
                "confidence": round(confidence, 4),
                "applicable": 1.0,
                "gap": round(gap, 4),
                "reason": reason,
            },
        }
        state.skill_outputs[self.name] = result
        state.trace(self.name, "success", f"BCC/SEK specialist completed: recommendation={recommendation}")
        return result

    def _build_evidence_items(
        self,
        *,
        scores: Dict[str, float],
        recommendation: str,
        loser: str,
        confidence: float,
        gap: float,
        rationale: List[str],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = [
            {
                "source": self.name,
                "type": "group_summary",
                "supports": recommendation,
                "opposes": loser,
                "weight": round(confidence, 4),
                "gap": round(gap, 4),
                "group_scores": {name: round(value, 4) for name, value in scores.items()},
            }
        ]
        for line in rationale[:8]:
            items.append(
                {
                    "source": self.name,
                    "type": "rationale",
                    "supports": recommendation,
                    "detail": line,
                }
            )
        return items

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
            for keyword in keywords:
                if keyword in cue_lower:
                    hits += 1
                    break
        return hits

    def _normalize_text_list(self, items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        values: List[str] = []
        for item in items:
            text = str(item).strip()
            if text and text not in values:
                values.append(text)
        return values

    def _safe_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
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

    def _site_matches(self, site: str, keywords: List[str]) -> bool:
        if not site:
            return False
        normalized = site.replace("-", " ").replace("/", " ").strip().lower()
        tokens = [token for token in normalized.split() if token]
        return any(keyword in tokens for keyword in keywords)
