from __future__ import annotations

from typing import Any, Dict, List

from agent.evidence_calibrator import LearnableEvidenceCalibrator
from agent.state import CaseState
from skills.base import BaseSkill


class MetadataConsistencySkill(BaseSkill):
    name = "metadata_consistency_skill"

    def __init__(self, evidence_calibrator: LearnableEvidenceCalibrator | None = None) -> None:
        self.evidence_calibrator = evidence_calibrator

    def run(self, state: CaseState) -> Dict[str, Any]:
        metadata = state.get_metadata()
        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [
            self._norm_label(x.get("name"))
            for x in ddx[:5]
            if self._norm_label(x.get("name")) != "UNKNOWN"
        ]

        age = self._safe_int(metadata.get("age"))
        sex = self._norm_text(metadata.get("sex"))
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
        suspicious_cues = self._normalize_text_list(risk_cues.get("suspicious_cues", []))
        has_strong_invasive_history = any(
            token in history
            for token in ["rapid growth", "pain", "hurt", "ulcer", "ulcerated"]
        )
        has_mild_change_history = any(
            token in history
            for token in ["grew", "growing", "changed", "elevation"]
        )
        has_bleed_or_elevation = any(
            token in history
            for token in ["bleed", "bleeding", "elevation"]
        )
        has_invasive_signal = has_strong_invasive_history or len(malignant_cues) > 0
        malignant_candidates_present = any(x in top_names for x in ["SCC", "BCC", "MEL"])
        ack_scc_pair_present = "ACK" in top_names and "SCC" in top_names
        top1 = top_names[0] if top_names else "UNKNOWN"

        support_scores: Dict[str, float] = {}
        penalty_scores: Dict[str, float] = {}
        rationale: List[str] = []
        perception_fallback = bool((state.perception or {}).get("fallback_reason"))

        meta_available = any([age is not None, bool(sex), bool(site), bool(history)])
        if not meta_available:
            result = {
                "consistency": "unknown",
                "supported_diagnoses": [],
                "penalized_diagnoses": [],
                "meta_keys": list(metadata.keys()),
                "rationale": ["No informative metadata available."],
                "support_strengths": {},
                "penalty_strengths": {},
                "score": 0.0,
            }
            state.skill_outputs[self.name] = result
            state.trace(self.name, "success", "Metadata consistency executed: no informative metadata")
            return result

        if age is not None:
            if age >= 55:
                for disease in ["ACK", "BCC", "MEL"]:
                    if disease in top_names:
                        self._add_support(
                            support_scores,
                            rationale,
                            disease,
                            f"Older age weakly supports {disease}.",
                            self._weight("metadata_invasive_malignant_bonus", 0.75 if disease in {"BCC", "MEL"} else 0.60),
                        )
                if "SCC" in top_names and has_strong_invasive_history:
                    self._add_support(
                        support_scores,
                        rationale,
                        "SCC",
                        "Older age plus invasive history supports SCC.",
                        self._weight("metadata_invasive_malignant_bonus", 0.90),
                    )
                if age >= 50 and has_bleed_or_elevation:
                    self._add_support(
                        support_scores,
                        rationale,
                        "BCC",
                        "Older age plus bleed/elevation supports BCC.",
                        self._weight("metadata_invasive_malignant_bonus", 0.80),
                    )

            if age <= 30:
                if "NEV" in top_names:
                    self._add_support(
                        support_scores,
                        rationale,
                        "NEV",
                        "Younger age weakly supports NEV.",
                        self._weight("metadata_nev_rescue_bonus", 0.65),
                    )
                for disease in ["ACK", "BCC"]:
                    if disease in top_names:
                        self._add_penalty(
                            penalty_scores,
                            rationale,
                            disease,
                            f"Younger age weakly penalizes {disease}.",
                            self._weight("metadata_pediatric_malignant_penalty", 0.65),
                        )

            if age <= 18:
                for disease in ["NEV", "SEK"]:
                    if disease in top_names:
                        self._add_support(
                            support_scores,
                            rationale,
                            disease,
                            f"Pediatric age weakly supports {disease}.",
                            self._weight("metadata_pediatric_benign_bonus", 1.00),
                        )
                if not has_invasive_signal:
                    self._add_support(
                        support_scores,
                        rationale,
                        "NEV",
                        "Pediatric age without invasive signals keeps NEV as a strong benign alternative.",
                        self._weight("metadata_nev_rescue_bonus", 0.85),
                    )
                for disease in ["BCC", "ACK"]:
                    if disease in top_names:
                        self._add_penalty(
                            penalty_scores,
                            rationale,
                            disease,
                            f"Pediatric age weakly penalizes {disease}.",
                            self._weight("metadata_pediatric_malignant_penalty", 1.00),
                        )
                if "SCC" in top_names and not has_invasive_signal:
                    self._add_penalty(
                        penalty_scores,
                        rationale,
                        "SCC",
                        "Pediatric age without invasive signals penalizes SCC.",
                        self._weight("metadata_pediatric_malignant_penalty", 1.10),
                    )

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
                "hand",
                "forearm",
                "lip",
            ]
            trunk_keywords = ["trunk", "back", "chest", "abdomen"]
            lower_limb_keywords = ["leg", "foot", "toe"]

            if self._site_matches(site, sun_exposed_keywords):
                for disease in ["ACK", "BCC", "SCC"]:
                    if disease in top_names:
                        self._add_support(
                            support_scores,
                            rationale,
                            disease,
                            "Sun-exposed site weakly supports ACK/BCC/SCC.",
                            self._weight("metadata_sun_exposed_keratinocytic_bonus", 0.70),
                        )

                if self._site_matches(site, ["nose", "face", "ear", "temple", "cheek"]):
                    self._add_support(
                        support_scores,
                        rationale,
                        "BCC",
                        "Classic BCC site supports BCC.",
                        self._weight("metadata_bcc_bonus", 0.60),
                    )

                if "ACK" in top_names and not has_strong_invasive_history:
                    self._add_support(
                        support_scores,
                        rationale,
                        "ACK",
                        "Sun-exposed site without invasive history favors ACK.",
                        self._weight("metadata_sun_exposed_keratinocytic_bonus", 0.65),
                    )

                if (
                    self._site_matches(site, ["neck", "hand", "forearm"])
                    and not has_strong_invasive_history
                    and len(malignant_cues) == 0
                    and (top1 in {"NEV", "SCC"} or "SCC" in top_names)
                ):
                    self._add_support(
                        support_scores,
                        rationale,
                        "ACK",
                        "Sun-exposed low-invasion context adds ACK as an alternative.",
                        self._weight("metadata_sun_exposed_keratinocytic_bonus", 0.55),
                    )

                if "SCC" in top_names and not has_invasive_signal:
                    self._add_penalty(
                        penalty_scores,
                        rationale,
                        "SCC",
                        "Low-invasion sun-exposed context weakly penalizes SCC.",
                        self._weight("metadata_pediatric_malignant_penalty", 0.45),
                    )

            if self._site_matches(site, ["lip", "ear"]) and "SCC" in top_names:
                self._add_support(
                    support_scores,
                    rationale,
                    "SCC",
                    "High-risk site supports SCC.",
                    self._weight("metadata_invasive_malignant_bonus", 0.80),
                )

            if self._site_matches(site, trunk_keywords):
                if (
                    "NEV" in top_names
                    and top1 == "NEV"
                    and not malignant_candidates_present
                    and len(suspicious_cues) == 0
                ):
                    self._add_support(
                        support_scores,
                        rationale,
                        "NEV",
                        "Trunk site weakly supports NEV in low-risk context.",
                        self._weight("metadata_trunk_benign_bonus", 0.75),
                    )
                if age is not None and age >= 55 and has_bleed_or_elevation and "NEV" in top_names:
                    self._add_penalty(
                        penalty_scores,
                        rationale,
                        "NEV",
                        "Older age with bleed/elevation penalizes NEV.",
                        self._weight("metadata_invasive_malignant_bonus", 0.55),
                    )
                if "ACK" in top_names and not has_invasive_signal:
                    self._add_penalty(
                        penalty_scores,
                        rationale,
                        "ACK",
                        "Trunk site weakly penalizes ACK.",
                        self._weight("metadata_trunk_benign_bonus", 0.45),
                    )
                if ack_scc_pair_present and "SCC" in top_names and not has_invasive_signal:
                    self._add_penalty(
                        penalty_scores,
                        rationale,
                        "SCC",
                        "Trunk site weakly penalizes SCC in low-invasion ACK/SCC ambiguity.",
                        self._weight("metadata_trunk_benign_bonus", 0.55),
                    )

            if self._site_matches(site, lower_limb_keywords) and "MEL" in top_names:
                self._add_support(
                    support_scores,
                    rationale,
                    "MEL",
                    "Lower limb site may weakly support MEL.",
                    self._weight("metadata_invasive_malignant_bonus", 0.65),
                )

            if self._site_matches(site, ["nose", "face", "ear"]) and has_bleed_or_elevation:
                self._add_support(
                    support_scores,
                    rationale,
                    "BCC",
                    "Bleeding/elevated lesion on classic BCC site strongly supports BCC.",
                    self._weight("metadata_bcc_bonus", 0.90),
                )

            if (
                perception_fallback
                and not has_strong_invasive_history
                and (
                    (self._site_matches(site, ["nose", "face", "ear", "temple", "cheek"]) and has_bleed_or_elevation)
                    or (self._site_matches(site, ["neck", "hand", "forearm"]) and age is not None and age >= 55 and has_bleed_or_elevation)
                    or (self._site_matches(site, ["trunk", "back", "chest", "abdomen"]) and age is not None and age >= 70 and has_bleed_or_elevation)
                )
            ):
                self._add_support(
                    support_scores,
                    rationale,
                    "BCC",
                    "Fallback metadata pattern strongly supports BCC.",
                    self._weight("metadata_bcc_bonus", 0.95),
                )
                if "SCC" in top_names:
                    self._add_penalty(
                        penalty_scores,
                        rationale,
                        "SCC",
                        "Fallback BCC pattern weakly penalizes SCC.",
                        self._weight("metadata_penalty_weight", 0.35),
                    )

        if history:
            history_map_support = {
                "bleeding": ["SCC", "MEL"],
                "bleed": ["SCC", "MEL"],
                "rapid growth": ["SCC", "MEL"],
                "pain": ["SCC", "MEL"],
                "hurt": ["SCC", "MEL"],
                "new lesion": ["MEL", "BCC"],
                "elevation": ["BCC"],
                "sun damage": ["ACK", "BCC", "SCC"],
                "sun exposure": ["ACK", "BCC", "SCC"],
                "chronic sun": ["ACK", "BCC", "SCC"],
            }

            for key, diseases in history_map_support.items():
                if key in history:
                    matched = [d for d in diseases if d in top_names]
                    if matched:
                        for disease in matched:
                            self._add_support(
                                support_scores,
                                rationale,
                                disease,
                                f"History keyword '{key}' weakly supports {matched}.",
                                self._weight("metadata_invasive_malignant_bonus", 0.55 if disease in {"MEL", "BCC", "SCC"} else 0.45),
                            )

            if has_bleed_or_elevation:
                self._add_support(
                    support_scores,
                    rationale,
                    "BCC",
                    "Bleed/elevation pattern supports BCC.",
                    self._weight("metadata_invasive_malignant_bonus", 0.70),
                )
                if "NEV" in top_names:
                    self._add_penalty(
                        penalty_scores,
                        rationale,
                        "NEV",
                        "Bleed/elevation pattern penalizes NEV.",
                        self._weight("metadata_penalty_weight", 0.40),
                    )
            if has_mild_change_history and not has_strong_invasive_history:
                self._add_support(
                    support_scores,
                    rationale,
                    "ACK",
                    "Mild change history without strong invasion keeps ACK plausible.",
                    self._weight("metadata_sun_exposed_keratinocytic_bonus", 0.45),
                )

        for label in list(set(support_scores) | set(penalty_scores)):
            support_value = support_scores.get(label, 0.0)
            penalty_value = penalty_scores.get(label, 0.0)
            if support_value > 0.0 and penalty_value > 0.0:
                net = support_value - penalty_value
                if net >= 0.10:
                    support_scores[label] = round(net, 4)
                    penalty_scores.pop(label, None)
                elif net <= -0.10:
                    penalty_scores[label] = round(abs(net), 4)
                    support_scores.pop(label, None)
                else:
                    support_scores.pop(label, None)
                    penalty_scores.pop(label, None)

        supported = [
            label for label, _ in sorted(support_scores.items(), key=lambda item: item[1], reverse=True)
            if label in top_names or support_scores[label] >= 0.75
        ][:5]
        penalized = [
            label for label, _ in sorted(penalty_scores.items(), key=lambda item: item[1], reverse=True)
            if label not in supported
        ][:5]

        score = round(
            sum(support_scores.get(label, 0.0) for label in supported)
            - sum(penalty_scores.get(label, 0.0) for label in penalized),
            4,
        )
        if supported and not penalized:
            consistency = "strong"
        elif supported and penalized:
            consistency = "partial"
        elif penalized and not supported:
            consistency = "weak"
        else:
            consistency = "unknown"

        result = {
            "consistency": consistency,
            "supported_diagnoses": supported,
            "supported_labels": supported,
            "penalized_diagnoses": penalized,
            "penalized_labels": penalized,
            "meta_keys": list(metadata.keys()),
            "metadata_snapshot": {
                "age": age,
                "sex": sex,
                "site": site,
                "history": history,
            },
            "support_strengths": {label: round(float(support_scores.get(label, 0.0)), 4) for label in supported},
            "penalty_strengths": {label: round(float(penalty_scores.get(label, 0.0)), 4) for label in penalized},
            "rationale": rationale[:12],
            "score": score,
        }

        state.skill_outputs[self.name] = result
        state.trace(
            self.name,
            "success",
            f"Metadata consistency executed: consistency={consistency}",
            payload={
                "consistency": consistency,
                "supported": supported,
                "penalized": penalized,
            },
        )
        return result

    def _add_support(
        self,
        scores: Dict[str, float],
        rationale: List[str],
        label: str,
        message: str,
        weight: float,
    ) -> None:
        name = self._norm_label(label)
        if name == "UNKNOWN":
            return
        scores[name] = round(scores.get(name, 0.0) + max(0.0, float(weight)), 4)
        if message and message not in rationale:
            rationale.append(message)

    def _add_penalty(
        self,
        scores: Dict[str, float],
        rationale: List[str],
        label: str,
        message: str,
        weight: float,
    ) -> None:
        name = self._norm_label(label)
        if name == "UNKNOWN":
            return
        scores[name] = round(scores.get(name, 0.0) + max(0.0, float(weight)), 4)
        if message and message not in rationale:
            rationale.append(message)

    def _weight(self, name: str, default: float) -> float:
        if self.evidence_calibrator is None:
            return float(default)
        return float(self.evidence_calibrator.get_weight(name, default))

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
