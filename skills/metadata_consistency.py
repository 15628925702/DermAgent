from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState
from skills.base import BaseSkill


class MetadataConsistencySkill(BaseSkill):
    name = "metadata_consistency_skill"

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

        supported: List[str] = []
        penalized: List[str] = []
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
                "score": 0.0,
            }
            state.skill_outputs[self.name] = result
            state.trace(self.name, "success", "Metadata consistency executed: no informative metadata")
            return result

        if age is not None:
            if age >= 55:
                for disease in ["ACK", "BCC", "MEL"]:
                    if disease in top_names and disease not in supported:
                        supported.append(disease)
                        rationale.append(f"Older age weakly supports {disease}.")
                if "SCC" in top_names and has_strong_invasive_history and "SCC" not in supported:
                    supported.append("SCC")
                    rationale.append("Older age plus invasive history supports SCC.")
                if age >= 50 and has_bleed_or_elevation and "BCC" not in supported:
                    supported.append("BCC")
                    rationale.append("Older age plus bleed/elevation supports BCC.")

            if age <= 30:
                if "NEV" in top_names and "NEV" not in supported:
                    supported.append("NEV")
                    rationale.append("Younger age weakly supports NEV.")
                for disease in ["ACK", "BCC"]:
                    if disease in top_names and disease not in penalized:
                        penalized.append(disease)
                        rationale.append(f"Younger age weakly penalizes {disease}.")

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
                    if disease in top_names and disease not in supported:
                        supported.append(disease)
                rationale.append("Sun-exposed site weakly supports ACK/BCC/SCC.")

                if self._site_matches(site, ["nose", "face", "ear", "temple", "cheek"]) and "BCC" not in supported:
                    supported.append("BCC")
                    rationale.append("Classic BCC site supports BCC.")

                if "ACK" in top_names and not has_strong_invasive_history and "ACK" not in supported:
                    supported.append("ACK")
                    rationale.append("Sun-exposed site without invasive history favors ACK.")

                if (
                    "ACK" not in supported
                    and self._site_matches(site, ["neck", "hand", "forearm"])
                    and not has_strong_invasive_history
                    and len(malignant_cues) == 0
                    and (top1 in {"NEV", "SCC"} or "SCC" in top_names)
                ):
                    supported.append("ACK")
                    rationale.append("Sun-exposed low-invasion context adds ACK as an alternative.")

                if (
                    "SCC" in top_names
                    and "SCC" in top_names
                    and not has_invasive_signal
                    and "SCC" not in penalized
                ):
                    penalized.append("SCC")
                    rationale.append("Low-invasion sun-exposed context weakly penalizes SCC.")

            if self._site_matches(site, ["lip", "ear"]):
                if "SCC" in top_names and "SCC" not in supported:
                    supported.append("SCC")
                    rationale.append("High-risk site supports SCC.")

            if self._site_matches(site, trunk_keywords):
                if (
                    "NEV" in top_names
                    and top1 == "NEV"
                    and not malignant_candidates_present
                    and len(suspicious_cues) == 0
                    and "NEV" not in supported
                ):
                    supported.append("NEV")
                    rationale.append("Trunk site weakly supports NEV in low-risk context.")
                if age is not None and age >= 55 and has_bleed_or_elevation and "NEV" in top_names and "NEV" not in penalized:
                    penalized.append("NEV")
                    rationale.append("Older age with bleed/elevation penalizes NEV.")
                if (
                    "ACK" in top_names
                    and not has_invasive_signal
                    and "ACK" not in penalized
                ):
                    penalized.append("ACK")
                    rationale.append("Trunk site weakly penalizes ACK.")
                if (
                    ack_scc_pair_present
                    and "SCC" in top_names
                    and not has_invasive_signal
                    and "SCC" not in penalized
                ):
                    penalized.append("SCC")
                    rationale.append("Trunk site weakly penalizes SCC in low-invasion ACK/SCC ambiguity.")

            if self._site_matches(site, lower_limb_keywords):
                if "MEL" in top_names and "MEL" not in supported:
                    supported.append("MEL")
                    rationale.append("Lower limb site may weakly support MEL.")

            if self._site_matches(site, ["nose", "face", "ear"]) and has_bleed_or_elevation and "BCC" not in supported:
                supported.append("BCC")
                rationale.append("Bleeding/elevated lesion on classic BCC site strongly supports BCC.")

            if (
                perception_fallback
                and "BCC" not in supported
                and not has_strong_invasive_history
                and (
                    (self._site_matches(site, ["nose", "face", "ear", "temple", "cheek"]) and has_bleed_or_elevation)
                    or (self._site_matches(site, ["neck", "hand", "forearm"]) and age is not None and age >= 55 and has_bleed_or_elevation)
                    or (self._site_matches(site, ["trunk", "back", "chest", "abdomen"]) and age is not None and age >= 70 and has_bleed_or_elevation)
                )
            ):
                supported.append("BCC")
                rationale.append("Fallback metadata pattern strongly supports BCC.")
                if "SCC" in top_names and "SCC" not in penalized:
                    penalized.append("SCC")
                    rationale.append("Fallback BCC pattern weakly penalizes SCC.")

        if history:
            history_map_support = {
                "bleeding": ["SCC", "MEL"],
                "bleed": ["SCC", "MEL"],
                "rapid growth": ["SCC", "MEL"],
                "pain": ["SCC", "MEL"],
                "hurt": ["SCC", "MEL"],
                "new lesion": ["MEL", "BCC"],
                "elevation": ["BCC"],
                "new lesion": ["MEL", "BCC"],
                "sun damage": ["ACK", "BCC", "SCC"],
                "sun exposure": ["ACK", "BCC", "SCC"],
                "chronic sun": ["ACK", "BCC", "SCC"],
            }

            for key, diseases in history_map_support.items():
                if key in history:
                    matched = [d for d in diseases if d in top_names]
                    for disease in matched:
                        if disease not in supported:
                            supported.append(disease)
                    if matched:
                        rationale.append(f"History keyword '{key}' weakly supports {matched}.")

            if has_bleed_or_elevation:
                if "BCC" not in supported:
                    supported.append("BCC")
                    rationale.append("Bleed/elevation pattern supports BCC.")
                if "NEV" in top_names and "NEV" not in penalized:
                    penalized.append("NEV")
                    rationale.append("Bleed/elevation pattern penalizes NEV.")
            if has_mild_change_history and not has_strong_invasive_history and "ACK" not in supported:
                supported.append("ACK")
                rationale.append("Mild change history without strong invasion keeps ACK plausible.")

        penalized = [x for x in penalized if x not in supported]

        score = 0.35 * len(supported) - 0.25 * len(penalized)
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
            "supported_diagnoses": supported[:5],
            "supported_labels": supported[:5],
            "penalized_diagnoses": penalized[:5],
            "penalized_labels": penalized[:5],
            "meta_keys": list(metadata.keys()),
            "metadata_snapshot": {
                "age": age,
                "sex": sex,
                "site": site,
                "history": history,
            },
            "rationale": rationale[:12],
            "score": round(score, 4),
        }

        state.skill_outputs[self.name] = result
        state.trace(
            self.name,
            "success",
            f"Metadata consistency executed: consistency={consistency}",
            payload={
                "consistency": consistency,
                "supported": supported[:5],
                "penalized": penalized[:5],
            },
        )
        return result

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
