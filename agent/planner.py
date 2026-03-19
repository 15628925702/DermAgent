from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState


class ExperienceSkillPlanner:
    def __init__(self, use_specialist: bool = True) -> None:
        self.use_specialist = use_specialist

    def plan(self, state: CaseState) -> Dict[str, object]:
        selected: List[str] = ["uncertainty_assessment_skill"]
        routing_reasons: List[Dict[str, Any]] = []

        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [str(x.get("name", "")).strip().upper() for x in ddx[:3] if str(x.get("name", "")).strip()]
        uncertainty = str(
            (state.perception.get("uncertainty", {}) or {}).get("level", "high")
        ).lower()

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        confusion_hits = state.retrieval.get("confusion_hits", []) or []
        recommended_skills = [
            str(x).strip()
            for x in retrieval_summary.get("recommended_skills", [])
            if str(x).strip()
        ]
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()
        has_confusion_support = bool(retrieval_summary.get("has_confusion_support", False))
        supports_top1 = bool(retrieval_summary.get("supports_top1", False))
        confusion_pairs = [
            tuple(
                sorted(
                    str(x).strip().upper()
                    for x in item.get("pair", [])
                    if str(x).strip()
                )
            )
            for item in confusion_hits
            if isinstance(item, dict)
        ]

        if uncertainty in {"medium", "high"}:
            selected.append("compare_skill")
            routing_reasons.append(
                {
                    "skill": "compare_skill",
                    "trigger": "uncertainty_threshold",
                    "detail": {"uncertainty": uncertainty},
                }
            )

        if "compare_skill" in recommended_skills and "compare_skill" not in selected:
            selected.append("compare_skill")
            routing_reasons.append(
                {
                    "skill": "compare_skill",
                    "trigger": "retrieval_recommended_skill",
                    "detail": {"recommended_skills": recommended_skills},
                }
            )

        if {"MEL", "BCC", "SCC"}.intersection(top_names):
            selected.append("malignancy_risk_skill")
            routing_reasons.append(
                {
                    "skill": "malignancy_risk_skill",
                    "trigger": "malignant_candidate_in_top_k",
                    "detail": {"top_names": top_names},
                }
            )

        if self._should_add_metadata_skill(
            state=state,
            uncertainty=uncertainty,
            retrieval_confidence=retrieval_confidence,
            supports_top1=supports_top1,
        ):
            selected.append("metadata_consistency_skill")
            routing_reasons.append(
                {
                    "skill": "metadata_consistency_skill",
                    "trigger": "metadata_or_support_check",
                    "detail": {
                        "retrieval_confidence": retrieval_confidence,
                        "supports_top1": supports_top1,
                        "metadata_keys": list(state.get_metadata().keys()),
                    },
                }
            )

        if self.use_specialist and self._should_add_pair_specialist(
            target_pair=("ACK", "SCC"),
            top_names=top_names,
            confusion_pairs=confusion_pairs,
            recommended_skills=recommended_skills,
            state=state,
        ):
            selected.append("ack_scc_specialist_skill")
            routing_reasons.append(
                {
                    "skill": "ack_scc_specialist_skill",
                    "trigger": "pair_or_confusion_support",
                    "detail": {
                        "target_pair": ["ACK", "SCC"],
                        "top_names": top_names,
                        "confusion_pairs": [list(x) for x in confusion_pairs],
                        "recommended_skills": recommended_skills,
                    },
                }
            )

        if self.use_specialist and self._should_add_pair_specialist(
            target_pair=("MEL", "NEV"),
            top_names=top_names,
            confusion_pairs=confusion_pairs,
            recommended_skills=recommended_skills,
            state=state,
        ):
            selected.append("mel_nev_specialist_skill")
            routing_reasons.append(
                {
                    "skill": "mel_nev_specialist_skill",
                    "trigger": "pair_or_confusion_support",
                    "detail": {
                        "target_pair": ["MEL", "NEV"],
                        "top_names": top_names,
                        "confusion_pairs": [list(x) for x in confusion_pairs],
                        "recommended_skills": recommended_skills,
                    },
                }
            )

        state.selected_skills = list(dict.fromkeys(selected))
        state.planner = {
            "selected_skills": state.selected_skills,
            "reason": routing_reasons,
            "flags": {
                "top_names": top_names,
                "uncertainty": uncertainty,
                "retrieval_confidence": retrieval_confidence,
                "has_confusion_support": has_confusion_support,
                "supports_top1": supports_top1,
                "recommended_skills": recommended_skills,
                "use_specialist": self.use_specialist,
            },
        }
        state.trace("planner", "success", f"Selected {state.selected_skills}")
        return state.planner

    def _should_add_pair_specialist(
        self,
        target_pair: tuple[str, str],
        top_names: List[str],
        confusion_pairs: List[tuple[str, ...]],
        recommended_skills: List[str],
        state: CaseState,
    ) -> bool:
        pair_set = set(target_pair)
        specialist_name = (
            "ack_scc_specialist_skill"
            if pair_set == {"ACK", "SCC"}
            else "mel_nev_specialist_skill"
        )

        if pair_set.issubset(set(top_names)):
            return True

        for pair in confusion_pairs:
            if set(pair) == pair_set:
                return True

        if pair_set == {"ACK", "SCC"} and self._should_add_ack_scc_specialist_from_metadata(
            state=state,
            top_names=top_names,
        ):
            return True

        return specialist_name in recommended_skills

    def _should_add_metadata_skill(
        self,
        state: CaseState,
        uncertainty: str,
        retrieval_confidence: str,
        supports_top1: bool,
    ) -> bool:
        metadata = state.get_metadata()
        if not metadata:
            return False

        if uncertainty in {"medium", "high"}:
            return True

        if retrieval_confidence == "low" or not supports_top1:
            return True

        return True

    def _should_add_ack_scc_specialist_from_metadata(
        self,
        state: CaseState,
        top_names: List[str],
    ) -> bool:
        if "SCC" not in top_names:
            return False

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
        malignant_cues = [
            str(x).strip().lower()
            for x in ((state.perception.get("risk_cues", {}) or {}).get("malignant_cues", []) or [])
            if str(x).strip()
        ]
        sun_exposed = self._site_matches(
            site,
            [
                "neck",
                "hand",
                "forearm",
            ],
        )
        strong_invasive_history = any(
            token in history
            for token in ["bleed", "bleeding", "rapid growth", "pain", "hurt", "ulcer", "ulcerated"]
        )
        top1 = top_names[0] if top_names else "UNKNOWN"
        return sun_exposed and not strong_invasive_history and not malignant_cues and top1 in {"NEV", "SCC"}

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
