from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState
from skills.base import BaseSkill


class MalignancyRiskSkill(BaseSkill):
    name = "malignancy_risk_skill"

    MALIGNANT_CANDIDATES = {"MEL", "BCC", "SCC"}
    HIGH_RISK_CANDIDATES = {"MEL", "SCC"}
    MALIGNANCY_PRIORITY = ["MEL", "SCC", "BCC"]

    def run(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [
            self._norm_label(x.get("name"))
            for x in ddx[:5]
            if self._norm_label(x.get("name")) != "UNKNOWN"
        ]

        risk_cues = state.perception.get("risk_cues", {}) or {}
        malignant_cues = self._normalize_text_list(risk_cues.get("malignant_cues", []))
        suspicious_cues = self._normalize_text_list(risk_cues.get("suspicious_cues", []))
        visual_cues = self._normalize_text_list(state.perception.get("visual_cues", []))
        metadata = state.get_metadata()
        site = self._norm_text(
            metadata.get("location")
            or metadata.get("site")
            or metadata.get("anatomical_site")
        )
        history_text = self._norm_text(
            metadata.get("history")
            or metadata.get("clinical_history")
            or metadata.get("past_history")
        )

        rationale: List[str] = []
        score = 0.0
        invasive_signal_count = self._count_invasive_signals(
            visual_cues=visual_cues,
            malignant_cues=malignant_cues,
            suspicious_cues=suspicious_cues,
            history_text=history_text,
        )

        if not ddx and not malignant_cues and not suspicious_cues and not visual_cues:
            result = {
                "risk_level": "low",
                "suspicious_malignancy": False,
                "tendency": "likely_benign",
                "recommendation": None,
                "preferred_label": None,
                "malignant_candidates_in_top": [],
                "evidence": {
                    "malignant_cues": [],
                    "suspicious_cues": [],
                    "matched_visual_risk_keywords": [],
                    "support_labels": [],
                },
                "rationale": ["insufficient_inputs_for_malignancy_assessment"],
                "score": 0.0,
                "confidence": 0.35,
                "reason": "fallback_insufficient_inputs",
                "invasive_signal_count": 0,
                "ack_scc_ambiguous": False,
            }
            state.skill_outputs[self.name] = result
            state.trace(
                self.name,
                "warning",
                "Malignancy risk fallback: insufficient inputs",
                payload={"risk_level": "low", "score": 0.0},
            )
            return result

        if malignant_cues:
            score += 2.0
            rationale.append(f"malignant_cues_present={len(malignant_cues)}")

        if suspicious_cues:
            score += 1.0
            rationale.append(f"suspicious_cues_present={len(suspicious_cues)}")

        risk_keywords = [
            "asymmetry",
            "irregular border",
            "irregular pigmentation",
            "ulcer",
            "ulcerated",
            "bleeding",
            "rapid growth",
            "variegated",
            "atypical",
            "indurated",
            "crust",
            "crusted",
        ]
        matched_keywords = []
        for cue in visual_cues:
            cue_lower = cue.lower()
            for kw in risk_keywords:
                if kw in cue_lower:
                    matched_keywords.append(kw)

        unique_matched_keywords = sorted(set(matched_keywords))
        if unique_matched_keywords:
            bonus = min(1.0, 0.2 * len(unique_matched_keywords))
            score += bonus
            rationale.append(f"visual_risk_keywords={unique_matched_keywords}")

        if invasive_signal_count > 0:
            rationale.append(f"invasive_signal_count={invasive_signal_count}")

        malignant_in_top = [x for x in top_names if x in self.MALIGNANT_CANDIDATES]
        high_risk_in_top = [x for x in top_names if x in self.HIGH_RISK_CANDIDATES]

        if malignant_in_top:
            score += 1.0
            rationale.append(f"malignant_candidates_in_top={malignant_in_top}")

        if high_risk_in_top:
            score += 0.5
            rationale.append(f"high_risk_candidates_in_top={high_risk_in_top}")

        top1 = top_names[0] if top_names else "UNKNOWN"
        if top1 in self.MALIGNANT_CANDIDATES:
            score += 1.0
            rationale.append(f"top1_is_malignant_candidate={top1}")

        if top1 in self.HIGH_RISK_CANDIDATES:
            score += 0.5
            rationale.append(f"top1_is_high_risk_candidate={top1}")

        top2 = top_names[:2]
        if len(set(top2).intersection(self.MALIGNANT_CANDIDATES)) >= 2:
            score += 0.4
            rationale.append(f"multiple_malignant_candidates_in_top2={top2}")

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = [self._norm_label(x) for x in retrieval_summary.get("support_labels", [])]
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()

        malignant_supports = [x for x in support_labels if x in self.MALIGNANT_CANDIDATES]
        if malignant_supports:
            if retrieval_confidence == "high":
                score += 0.8
            elif retrieval_confidence == "medium":
                score += 0.4
            else:
                score += 0.2
            rationale.append(
                f"retrieval_supports_malignant={malignant_supports}, retrieval_confidence={retrieval_confidence}"
            )

        if score >= 3.5:
            risk_level = "high"
            suspicious_malignancy = True
            confidence = 0.88
            reason = "strong_malignancy_evidence"
        elif score >= 1.8:
            risk_level = "medium"
            suspicious_malignancy = True
            confidence = 0.72
            reason = "moderate_malignancy_evidence"
        else:
            risk_level = "low"
            suspicious_malignancy = False
            confidence = 0.58
            reason = "weak_malignancy_evidence"

        preferred_label = self._choose_preferred_label(
            top_names=top_names,
            malignant_supports=malignant_supports,
            invasive_signal_count=invasive_signal_count,
            visual_cues=visual_cues,
            site=site,
            history_text=history_text,
        )

        ack_scc_ambiguous = (
            "ACK" in top_names
            and "SCC" in top_names
            and invasive_signal_count < 2
        )

        result = {
            "risk_level": risk_level,
            "suspicious_malignancy": suspicious_malignancy,
            "tendency": self._build_tendency(risk_level, suspicious_malignancy),
            "recommendation": preferred_label,
            "preferred_label": preferred_label,
            "malignant_candidates_in_top": malignant_in_top,
            "evidence": {
                "malignant_cues": malignant_cues,
                "suspicious_cues": suspicious_cues,
                "matched_visual_risk_keywords": unique_matched_keywords,
                "support_labels": support_labels,
                "retrieval_confidence": retrieval_confidence,
            },
            "rationale": rationale,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "reason": reason,
            "invasive_signal_count": invasive_signal_count,
            "ack_scc_ambiguous": ack_scc_ambiguous,
        }

        state.skill_outputs[self.name] = result
        state.trace(
            self.name,
            "success",
            f"Malignancy risk assessed: level={risk_level}",
            payload={
                "risk_level": risk_level,
                "suspicious_malignancy": suspicious_malignancy,
                "preferred_label": preferred_label,
                "score": round(score, 4),
                "confidence": round(confidence, 4),
            },
        )
        return result

    def _choose_preferred_label(
        self,
        top_names: List[str],
        malignant_supports: List[str],
        invasive_signal_count: int,
        visual_cues: List[str],
        site: str,
        history_text: str,
    ) -> str | None:
        if "ACK" in top_names and "SCC" in top_names and invasive_signal_count < 2:
            return None

        bcc_clues = ["pearly", "rolled", "translucent", "shiny", "telangiect"]
        bcc_site = self._site_matches(site, ["nose", "face", "ear", "temple", "cheek"])
        bcc_signal_count = 0
        for cue in visual_cues:
            cue_lower = cue.lower()
            if any(token in cue_lower for token in bcc_clues):
                bcc_signal_count += 1
        if any(token in history_text for token in ["bleed", "bleeding", "elevation"]):
            bcc_signal_count += 1

        if "BCC" in top_names and (bcc_signal_count >= 2 or (bcc_site and invasive_signal_count <= 2)):
            return "BCC"

        for x in top_names:
            if x in self.MALIGNANT_CANDIDATES:
                return x

        for label in self.MALIGNANCY_PRIORITY:
            if label in malignant_supports:
                return label

        return None

    def _build_tendency(self, risk_level: str, suspicious_malignancy: bool) -> str:
        if risk_level == "high":
            return "malignant_like"
        if risk_level == "medium" and suspicious_malignancy:
            return "suspicious"
        return "likely_benign"

    def _count_invasive_signals(
        self,
        visual_cues: List[str],
        malignant_cues: List[str],
        suspicious_cues: List[str],
        history_text: str,
    ) -> int:
        indicators = [
            "ulcer",
            "ulcerated",
            "bleed",
            "bleeding",
            "indurated",
            "nodule",
            "rapid growth",
            "growing",
            "grew",
            "elevation",
            "pain",
            "tender",
            "crust",
            "crusted",
        ]

        hits = 0
        for bucket in [visual_cues, malignant_cues, suspicious_cues]:
            for cue in bucket:
                cue_lower = cue.lower()
                if any(token in cue_lower for token in indicators):
                    hits += 1

        if history_text and any(token in history_text for token in indicators):
            hits += 1

        return hits

    def _norm_label(self, value: Any) -> str:
        if value is None:
            return "UNKNOWN"
        text = str(value).strip().upper()
        return text if text else "UNKNOWN"

    def _norm_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def _normalize_text_list(self, items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for x in items:
            text = str(x).strip()
            if text and text not in out:
                out.append(text)
        return out

    def _site_matches(self, site: str, keywords: List[str]) -> bool:
        if not site:
            return False
        normalized = site.replace("-", " ").replace("/", " ").strip().lower()
        tokens = [token for token in normalized.split() if token]
        return any(keyword in tokens for keyword in keywords)
