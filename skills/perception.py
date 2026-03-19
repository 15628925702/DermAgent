from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agent.state import CaseState
from integrations.openai_client import OpenAICompatClient
from skills.base import BaseSkill


class PerceptionSkill(BaseSkill):
    """
    Perception skill.

    Output schema:
    {
        "ddx_candidates": [{"name": "SCC", "score": 0.62}, ...],
        "most_likely": {"name": "SCC", "score": 0.62},
        "visual_cues": [...],
        "risk_cues": {
            "malignant_cues": [...],
            "suspicious_cues": [...]
        },
        "uncertainty": {"level": "medium"}
    }
    """

    name = "perception_skill"

    def __init__(self, model: str = "gpt-4o") -> None:
        self.client = OpenAICompatClient(model=model)

    def run(self, state: CaseState) -> Dict[str, Any]:
        image_path = self._resolve_image_path(state)
        metadata = self._safe_metadata(state)

        if not image_path:
            result = self._fallback_perception(
                reason="missing_image_path",
                metadata=metadata,
            )
            state.perception = result
            state.trace(
                self.name,
                "warning",
                "Perception fallback: image path not found",
                payload={"reason": "missing_image_path"},
            )
            return result

        try:
            raw_text = self.client.infer_derm_perception(
                image_path=image_path,
                metadata=metadata,
            )
            parsed = self._parse_json(raw_text)
            result = self._normalize_perception(parsed)

            state.perception = result
            state.trace(
                self.name,
                "success",
                "Perception completed",
                payload={
                    "image_path": image_path,
                    "top_ddx": result.get("ddx_candidates", [])[:3],
                    "uncertainty": result.get("uncertainty", {}),
                },
            )
            return result

        except Exception as e:
            result = self._fallback_perception(
                reason=f"api_error:{type(e).__name__}",
                metadata=metadata,
            )
            state.perception = result
            state.trace(
                self.name,
                "error",
                f"Perception failed, fallback used: {e}",
                payload={"image_path": image_path, "error_type": type(e).__name__},
            )
            return result

    def _resolve_image_path(self, state: CaseState) -> str:
        candidates: List[Optional[str]] = [
            getattr(state, "image_path", None),
            getattr(state, "img_path", None),
        ]

        case_info = getattr(state, "case_info", None)
        if isinstance(case_info, dict):
            candidates.extend(
                [
                    case_info.get("image_path"),
                    case_info.get("img_path"),
                    case_info.get("image"),
                    case_info.get("path"),
                ]
            )

        metadata = self._safe_metadata(state)
        candidates.extend(
            [
                metadata.get("image_path"),
                metadata.get("img_path"),
                metadata.get("image"),
                metadata.get("path"),
            ]
        )

        for attr_name in ["input_case", "case", "case_data", "raw_input"]:
            obj = getattr(state, attr_name, None)
            if isinstance(obj, dict):
                candidates.extend(
                    [
                        obj.get("image_path"),
                        obj.get("img_path"),
                        obj.get("image"),
                        obj.get("path"),
                    ]
                )

        for x in candidates:
            text = str(x).strip() if x is not None else ""
            if text:
                return text

        return ""

    def _safe_metadata(self, state: CaseState) -> Dict[str, Any]:
        if hasattr(state, "get_metadata"):
            meta = state.get_metadata()
            if isinstance(meta, dict):
                return meta
        meta = getattr(state, "metadata", None)
        if isinstance(meta, dict):
            return meta
        return {}

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return {}

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

    def _normalize_perception(self, data: Dict[str, Any]) -> Dict[str, Any]:
        ddx_raw = data.get("ddx_candidates", []) or []
        ddx_candidates: List[Dict[str, Any]] = []

        for item in ddx_raw[:5]:
            if not isinstance(item, dict):
                continue
            name = self._norm_label(item.get("name"))
            if not name:
                continue
            score = self._safe_score(item.get("score"), default=None)
            entry: Dict[str, Any] = {"name": name}
            if score is not None:
                entry["score"] = score
            ddx_candidates.append(entry)

        if not ddx_candidates:
            most_likely = data.get("most_likely", {}) or {}
            name = self._norm_label(most_likely.get("name"))
            score = self._safe_score(most_likely.get("score"), default=0.5)
            if name:
                ddx_candidates = [{"name": name, "score": score}]

        most_likely = data.get("most_likely", {}) or {}
        most_likely_name = self._norm_label(most_likely.get("name"))
        most_likely_score = self._safe_score(most_likely.get("score"), default=None)

        if not most_likely_name and ddx_candidates:
            most_likely_name = ddx_candidates[0]["name"]
            most_likely_score = ddx_candidates[0].get("score")

        visual_cues = self._normalize_text_list(data.get("visual_cues", []))

        risk_cues_raw = data.get("risk_cues", {}) or {}
        malignant_cues = self._normalize_text_list(risk_cues_raw.get("malignant_cues", []))
        suspicious_cues = self._normalize_text_list(risk_cues_raw.get("suspicious_cues", []))

        uncertainty_raw = data.get("uncertainty", {}) or {}
        uncertainty_level = str(uncertainty_raw.get("level", "high")).strip().lower()
        if uncertainty_level not in {"low", "medium", "high"}:
            uncertainty_level = "high"

        return {
            "ddx_candidates": ddx_candidates,
            "most_likely": {
                "name": most_likely_name or "UNKNOWN",
                "score": most_likely_score if most_likely_score is not None else 0.0,
            },
            "visual_cues": visual_cues,
            "risk_cues": {
                "malignant_cues": malignant_cues,
                "suspicious_cues": suspicious_cues,
            },
            "uncertainty": {
                "level": uncertainty_level,
            },
        }

    def _fallback_perception(
        self,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        site = str(
            metadata.get("location")
            or metadata.get("site")
            or metadata.get("anatomical_site")
            or ""
        ).lower()
        history = str(
            metadata.get("history")
            or metadata.get("clinical_history")
            or metadata.get("past_history")
            or ""
        ).lower()
        age = self._safe_int(metadata.get("age"))

        classic_bcc_site = self._site_matches(site, ["nose", "face", "ear", "temple", "cheek"])
        sun_exposed_site = self._site_matches(
            site,
            ["face", "scalp", "ear", "neck", "forehead", "cheek", "nose", "temple", "hand", "forearm", "lip"],
        )
        trunk_site = self._site_matches(site, ["trunk", "back", "chest", "abdomen"])
        strong_scc_history = any(
            token in history
            for token in ["rapid growth", "pain", "hurt", "ulcer", "ulcerated"]
        )
        bleed_or_elevation = any(
            token in history
            for token in ["bleed", "bleeding", "elevation"]
        )
        mild_change = any(
            token in history
            for token in ["grew", "growing", "changed"]
        )

        if classic_bcc_site and bleed_or_elevation and not strong_scc_history:
            ddx_candidates = [
                {"name": "BCC", "score": 0.46},
                {"name": "SCC", "score": 0.28},
                {"name": "NEV", "score": 0.16},
            ]
        elif sun_exposed_site and bleed_or_elevation and age is not None and age >= 55 and not strong_scc_history:
            ddx_candidates = [
                {"name": "BCC", "score": 0.43},
                {"name": "SCC", "score": 0.29},
                {"name": "NEV", "score": 0.18},
            ]
        elif strong_scc_history or self._site_matches(site, ["lip"]):
            ddx_candidates = [
                {"name": "SCC", "score": 0.48},
                {"name": "BCC", "score": 0.22},
                {"name": "NEV", "score": 0.15},
            ]
        elif self._site_matches(site, ["nose", "ear", "lip"]) and age is not None and age >= 60 and not bleed_or_elevation:
            ddx_candidates = [
                {"name": "SCC", "score": 0.42},
                {"name": "BCC", "score": 0.27},
                {"name": "ACK", "score": 0.16},
            ]
        elif sun_exposed_site and not bleed_or_elevation and not strong_scc_history:
            ddx_candidates = [
                {"name": "ACK", "score": 0.42},
                {"name": "SCC", "score": 0.25},
                {"name": "NEV", "score": 0.18},
            ]
        elif trunk_site and bleed_or_elevation and age is not None and age >= 60:
            ddx_candidates = [
                {"name": "BCC", "score": 0.38},
                {"name": "SCC", "score": 0.30},
                {"name": "NEV", "score": 0.18},
            ]
        elif trunk_site and mild_change and not strong_scc_history:
            ddx_candidates = [
                {"name": "SEK", "score": 0.34},
                {"name": "NEV", "score": 0.30},
                {"name": "SCC", "score": 0.18},
            ]
        else:
            default_label = "ACK" if sun_exposed_site else "NEV"
            ddx_candidates = [
                {"name": default_label, "score": 0.4},
                {"name": "SCC", "score": 0.3},
            ]

        return {
            "ddx_candidates": ddx_candidates,
            "most_likely": dict(ddx_candidates[0]),
            "visual_cues": [],
            "risk_cues": {
                "malignant_cues": [],
                "suspicious_cues": [],
            },
            "uncertainty": {"level": "high"},
            "fallback_reason": reason,
        }

    def _norm_label(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().upper()
        allowed = {"MEL", "NEV", "SCC", "BCC", "ACK", "SEK"}
        return text if text in allowed else ""

    def _safe_score(self, value: Any, default: Optional[float]) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            score = float(value)
            if score < 0:
                score = 0.0
            if score > 1:
                score = 1.0
            return round(score, 4)
        except (TypeError, ValueError):
            return default

    def _normalize_text_list(self, items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for x in items:
            text = str(x).strip()
            if text and text not in out:
                out.append(text)
        return out

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _site_matches(self, site: str, keywords: List[str]) -> bool:
        if not site:
            return False
        normalized = site.replace("-", " ").replace("/", " ").strip().lower()
        tokens = [token for token in normalized.split() if token]
        return any(keyword in tokens for keyword in keywords)
