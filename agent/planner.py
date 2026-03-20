from __future__ import annotations

from typing import Any, Dict, List

from agent.controller import LearnableSkillController
from agent.rule_scorer import LearnableRuleScorer
from agent.state import CaseState


class ExperienceSkillPlanner:
    SPECIALIST_CONFIGS = [
        {
            "pair": ("ACK", "SCC"),
            "skill": "ack_scc_specialist_skill",
            "allow_metadata_proxy": True,
        },
        {
            "pair": ("BCC", "SCC"),
            "skill": "bcc_scc_specialist_skill",
            "allow_metadata_proxy": False,
        },
        {
            "pair": ("BCC", "SEK"),
            "skill": "bcc_sek_specialist_skill",
            "allow_metadata_proxy": False,
        },
        {
            "pair": ("MEL", "NEV"),
            "skill": "mel_nev_specialist_skill",
            "allow_metadata_proxy": False,
        },
    ]

    def __init__(
        self,
        use_specialist: bool = True,
        controller: LearnableSkillController | None = None,
        rule_scorer: LearnableRuleScorer | None = None,
        planning_mode: str = "learnable_hybrid",
        enabled_skills: set[str] | None = None,
    ) -> None:
        self.use_specialist = use_specialist
        self.controller = controller
        self.rule_scorer = rule_scorer
        self.planning_mode = planning_mode
        self.enabled_skills = enabled_skills or set()

    def _is_enabled(self, skill_name: str) -> bool:
        return not self.enabled_skills or skill_name in self.enabled_skills

    def plan(self, state: CaseState) -> Dict[str, object]:
        selected: List[str] = []
        routing_reasons: List[Dict[str, Any]] = []

        if self._is_enabled("uncertainty_assessment_skill"):
            selected.append("uncertainty_assessment_skill")

        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [
            str(x.get("name", "")).strip().upper()
            for x in ddx[:3]
            if str(x.get("name", "")).strip()
        ]
        uncertainty = str((state.perception.get("uncertainty", {}) or {}).get("level", "high")).lower()

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        confusion_hits = state.retrieval.get("confusion_hits", []) or []
        rule_hits = state.retrieval.get("rule_hits", []) or []
        memory_recommended_skills = [
            str(x).strip() for x in retrieval_summary.get("memory_recommended_skills", []) if str(x).strip()
        ]
        retrieval_recommended_skills = [
            str(x).strip() for x in retrieval_summary.get("recommended_skills", []) if str(x).strip()
        ]
        retrieval_confidence = str(retrieval_summary.get("retrieval_confidence", "low")).lower()
        has_confusion_support = bool(retrieval_summary.get("has_confusion_support", False))
        supports_top1 = bool(retrieval_summary.get("supports_top1", False))
        confusion_pairs = [
            tuple(sorted(str(x).strip().upper() for x in item.get("pair", []) if str(x).strip()))
            for item in confusion_hits
            if isinstance(item, dict)
        ]

        rule_payload: Dict[str, Any] = {
            "rule_scores": {},
            "recommended_skill_scores": {},
            "recommended_skills": [],
            "applied_rules": [],
        }
        if self.rule_scorer is not None and rule_hits:
            rule_payload = self.rule_scorer.score_rules(state, rule_hits=rule_hits)

        rule_recommended_skills = [
            str(x).strip() for x in rule_payload.get("recommended_skills", []) if str(x).strip()
        ]
        recommended_skills = list(dict.fromkeys(memory_recommended_skills + rule_recommended_skills))
        if not recommended_skills:
            recommended_skills = retrieval_recommended_skills

        if self._is_enabled("compare_skill") and uncertainty in {"medium", "high"}:
            selected.append("compare_skill")
            routing_reasons.append({
                "skill": "compare_skill",
                "trigger": "uncertainty_threshold",
                "detail": {"uncertainty": uncertainty},
            })

        if self._is_enabled("compare_skill") and "compare_skill" in recommended_skills and "compare_skill" not in selected:
            selected.append("compare_skill")
            routing_reasons.append({
                "skill": "compare_skill",
                "trigger": "memory_or_rule_recommended_skill",
                "detail": {
                    "memory_recommended_skills": memory_recommended_skills,
                    "rule_recommended_skills": rule_recommended_skills,
                },
            })

        if self._is_enabled("malignancy_risk_skill") and {"MEL", "BCC", "SCC"}.intersection(top_names):
            selected.append("malignancy_risk_skill")
            routing_reasons.append({
                "skill": "malignancy_risk_skill",
                "trigger": "malignant_candidate_in_top_k",
                "detail": {"top_names": top_names},
            })

        if self._is_enabled("metadata_consistency_skill") and self._should_add_metadata_skill(
            state=state,
            uncertainty=uncertainty,
            retrieval_confidence=retrieval_confidence,
            supports_top1=supports_top1,
        ):
            selected.append("metadata_consistency_skill")
            routing_reasons.append({
                "skill": "metadata_consistency_skill",
                "trigger": "metadata_or_support_check",
                "detail": {
                    "retrieval_confidence": retrieval_confidence,
                    "supports_top1": supports_top1,
                    "metadata_keys": list(state.get_metadata().keys()),
                },
            })

        if self.use_specialist:
            for config in self.SPECIALIST_CONFIGS:
                target_pair = tuple(config["pair"])
                skill_name = str(config["skill"])
                if not self._is_enabled(skill_name):
                    continue
                if not self._should_add_pair_specialist(
                    target_pair=target_pair,
                    top_names=top_names,
                    confusion_pairs=confusion_pairs,
                    recommended_skills=recommended_skills,
                    state=state,
                    allow_metadata_proxy=bool(config.get("allow_metadata_proxy", False)),
                ):
                    continue
                selected.append(skill_name)
                routing_reasons.append({
                    "skill": skill_name,
                    "trigger": "pair_or_scored_rule_support",
                    "detail": {
                        "target_pair": list(target_pair),
                        "rule_recommended_skills": rule_recommended_skills,
                    },
                })

        rule_selected = list(dict.fromkeys(selected))
        planner_flags = {
            "top_names": top_names,
            "uncertainty": uncertainty,
            "retrieval_confidence": retrieval_confidence,
            "has_confusion_support": has_confusion_support,
            "supports_top1": supports_top1,
            "memory_recommended_skills": memory_recommended_skills,
            "rule_recommended_skills": rule_recommended_skills,
            "recommended_skills": recommended_skills,
            "use_specialist": self.use_specialist,
            "num_rule_hits": len(rule_hits),
            "applied_rules": rule_payload.get("applied_rules", []),
        }

        final_selected = rule_selected
        controller_payload: Dict[str, Any] = {}
        if self.controller is not None and self.planning_mode in {"controller", "learnable_hybrid"}:
            controller_payload = self.controller.select_skills(
                state,
                rule_priors=rule_selected,
                planner_context={
                    "recommended_skills": recommended_skills,
                    "flags": planner_flags,
                },
                allowed_skills=self.enabled_skills,
            )
            final_selected = list(controller_payload.get("selected_skills", []) or [])
            if not final_selected:
                final_selected = rule_selected
            if self.planning_mode == "learnable_hybrid":
                mandatory_rule_skills = [
                    skill_name for skill_name in rule_selected
                    if self._is_enabled(skill_name) and (skill_name == "malignancy_risk_skill" or skill_name.endswith("_specialist_skill"))
                ]
                final_selected = list(dict.fromkeys(final_selected + mandatory_rule_skills))

        state.selected_skills = list(dict.fromkeys(final_selected))
        state.planner = {
            "selected_skills": state.selected_skills,
            "reason": routing_reasons,
            "flags": planner_flags,
            "rule_selected_skills": rule_selected,
            "skill_scores": controller_payload.get("skill_scores", {}),
            "stop_probability": controller_payload.get("stop_probability"),
            "controller_debug": controller_payload.get("controller_debug", {}),
            "planning_mode": controller_payload.get("controller_mode", self.planning_mode),
            "rule_scores": rule_payload.get("rule_scores", {}),
            "rule_skill_scores": rule_payload.get("recommended_skill_scores", {}),
        }
        state.trace(
            "planner",
            "success",
            f"Selected {state.selected_skills}",
            payload={
                "rule_selected_skills": rule_selected,
                "controller_selected_skills": controller_payload.get("selected_skills", rule_selected),
                "stop_probability": controller_payload.get("stop_probability"),
                "applied_rules": rule_payload.get("applied_rules", []),
            },
        )
        return state.planner

    def _should_add_pair_specialist(
        self,
        target_pair: tuple[str, str],
        top_names: List[str],
        confusion_pairs: List[tuple[str, ...]],
        recommended_skills: List[str],
        state: CaseState,
        allow_metadata_proxy: bool = False,
    ) -> bool:
        pair_set = set(target_pair)
        specialist_name = self._specialist_name_for_pair(pair_set)
        if pair_set.issubset(set(top_names)):
            return True
        for pair in confusion_pairs:
            if set(pair) == pair_set:
                return True
        if allow_metadata_proxy and pair_set == {"ACK", "SCC"} and self._should_add_ack_scc_specialist_from_metadata(state=state, top_names=top_names):
            return True
        return specialist_name in recommended_skills

    def _specialist_name_for_pair(self, pair_set: set[str]) -> str:
        if pair_set == {"ACK", "SCC"}:
            return "ack_scc_specialist_skill"
        if pair_set == {"BCC", "SCC"}:
            return "bcc_scc_specialist_skill"
        if pair_set == {"BCC", "SEK"}:
            return "bcc_sek_specialist_skill"
        return "mel_nev_specialist_skill"

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
        return False

    def _should_add_ack_scc_specialist_from_metadata(self, state: CaseState, top_names: List[str]) -> bool:
        if "SCC" not in top_names:
            return False
        metadata = state.get_metadata()
        site = self._norm_text(metadata.get("location") or metadata.get("site") or metadata.get("anatomical_site"))
        history = self._norm_text(metadata.get("history") or metadata.get("clinical_history") or metadata.get("past_history"))
        malignant_cues = [
            str(x).strip().lower()
            for x in ((state.perception.get("risk_cues", {}) or {}).get("malignant_cues", []) or [])
            if str(x).strip()
        ]
        sun_exposed = self._site_matches(site, ["neck", "hand", "forearm"])
        strong_invasive_history = any(token in history for token in ["bleed", "bleeding", "rapid growth", "pain", "hurt", "ulcer", "ulcerated"])
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
