from __future__ import annotations

from typing import Any, Dict, List

from agent.controller import LearnableSkillController
from agent.rule_scorer import LearnableRuleScorer
from agent.state import CaseState


class ExperienceSkillPlanner:
    LEARNED_SPECIALIST_THRESHOLD = 0.95
    LEARNED_METADATA_THRESHOLD = 0.92

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
        decision_trace: List[Dict[str, Any]] = []

        case_features = self.extract_case_features(state)
        top_names = list(case_features["top_names"])
        uncertainty = str(case_features["uncertainty"])
        retrieval_confidence = str(case_features["retrieval_confidence"])
        supports_top1 = bool(case_features["supports_top1"])
        has_confusion_support = bool(case_features["has_confusion_support"])

        if self._is_enabled("uncertainty_assessment_skill"):
            selected.append("uncertainty_assessment_skill")
            decision_trace.append({
                "skill": "uncertainty_assessment_skill",
                "selected": True,
                "trigger": "always_on_core_skill",
            })

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        confusion_hits = state.retrieval.get("confusion_hits", []) or []
        rule_hits = state.retrieval.get("rule_hits", []) or []
        memory_recommended_skills = [
            str(x).strip() for x in retrieval_summary.get("memory_recommended_skills", []) if str(x).strip()
        ]
        retrieval_recommended_skills = [
            str(x).strip() for x in retrieval_summary.get("recommended_skills", []) if str(x).strip()
        ]
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

        compare_selected, compare_trigger, compare_detail = self._should_add_compare_skill(
            case_features=case_features,
            recommended_skills=recommended_skills,
        )
        if self._is_enabled("compare_skill") and compare_selected:
            selected.append("compare_skill")
            routing_reasons.append({
                "skill": "compare_skill",
                "trigger": compare_trigger,
                "detail": compare_detail,
            })
        decision_trace.append({
            "skill": "compare_skill",
            "selected": self._is_enabled("compare_skill") and compare_selected,
            "trigger": compare_trigger,
            "detail": compare_detail,
        })

        if self._is_enabled("malignancy_risk_skill") and bool(case_features["has_malignant_candidate"]):
            selected.append("malignancy_risk_skill")
            routing_reasons.append({
                "skill": "malignancy_risk_skill",
                "trigger": "malignant_candidate_in_top_k",
                "detail": {"top_names": top_names},
            })
            decision_trace.append({
                "skill": "malignancy_risk_skill",
                "selected": True,
                "trigger": "malignant_candidate_in_top_k",
                "detail": {"top_names": top_names},
            })
        else:
            decision_trace.append({
                "skill": "malignancy_risk_skill",
                "selected": False,
                "trigger": "malignant_candidate_in_top_k",
                "detail": {"top_names": top_names},
            })

        if self._is_enabled("metadata_consistency_skill") and self._should_add_metadata_skill(
            case_features=case_features,
            state=state,
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
            decision_trace.append({
                "skill": "metadata_consistency_skill",
                "selected": True,
                "trigger": "metadata_or_support_check",
                "detail": {
                    "retrieval_confidence": retrieval_confidence,
                    "supports_top1": supports_top1,
                    "metadata_present": case_features["metadata_present"],
                },
            })
        else:
            decision_trace.append({
                "skill": "metadata_consistency_skill",
                "selected": False,
                "trigger": "metadata_or_support_check",
                "detail": {
                    "retrieval_confidence": retrieval_confidence,
                    "supports_top1": supports_top1,
                    "metadata_present": case_features["metadata_present"],
                },
            })

        if self.use_specialist:
            for config in self.SPECIALIST_CONFIGS:
                target_pair = tuple(config["pair"])
                skill_name = str(config["skill"])
                if not self._is_enabled(skill_name):
                    decision_trace.append({
                        "skill": skill_name,
                        "selected": False,
                        "trigger": "skill_disabled",
                        "detail": {"target_pair": list(target_pair)},
                    })
                    continue
                specialist_selected, specialist_reason = self._should_add_pair_specialist(
                    target_pair=target_pair,
                    skill_name=skill_name,
                    case_features=case_features,
                    confusion_pairs=confusion_pairs,
                    recommended_skills=recommended_skills,
                    state=state,
                    allow_metadata_proxy=bool(config.get("allow_metadata_proxy", False)),
                )
                if not specialist_selected:
                    decision_trace.append({
                        "skill": skill_name,
                        "selected": False,
                        "trigger": specialist_reason,
                        "detail": {
                            "target_pair": list(target_pair),
                            "recommended_skills": recommended_skills,
                        },
                    })
                    continue
                selected.append(skill_name)
                routing_reasons.append({
                    "skill": skill_name,
                    "trigger": specialist_reason,
                    "detail": {
                        "target_pair": list(target_pair),
                        "rule_recommended_skills": rule_recommended_skills,
                    },
                })
                decision_trace.append({
                    "skill": skill_name,
                    "selected": True,
                    "trigger": specialist_reason,
                    "detail": {
                        "target_pair": list(target_pair),
                        "recommended_skills": recommended_skills,
                    },
                })

        rule_selected = list(dict.fromkeys(selected))
        planner_flags = {
            "top_names": top_names,
            "uncertainty": uncertainty,
            "retrieval_confidence": retrieval_confidence,
            "has_confusion_support": has_confusion_support,
            "supports_top1": supports_top1,
            "top_gap": case_features["top_gap"],
            "top_gap_small": case_features["top_gap_small"],
            "metadata_present": case_features["metadata_present"],
            "sun_exposed_site": case_features["sun_exposed_site"],
            "strong_invasive_history": case_features["strong_invasive_history"],
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
                    "case_features": case_features,
                    "decision_trace": decision_trace,
                },
                allowed_skills=self.enabled_skills,
            )
            final_selected = list(controller_payload.get("selected_skills", []) or [])
            if not final_selected:
                final_selected = rule_selected
            if self.planning_mode == "learnable_hybrid":
                mandatory_rule_skills = [
                    skill_name for skill_name in rule_selected
                    if self._is_enabled(skill_name) and skill_name == "malignancy_risk_skill"
                ]
                final_selected = list(dict.fromkeys(final_selected + mandatory_rule_skills))

        state.selected_skills = list(dict.fromkeys(final_selected))
        state.planner = {
            "selected_skills": state.selected_skills,
            "reason": routing_reasons,
            "decision_trace": decision_trace,
            "flags": planner_flags,
            "case_features": case_features,
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

    def extract_case_features(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [
            str(x.get("name", "")).strip().upper()
            for x in ddx[:3]
            if str(x.get("name", "")).strip()
        ]
        top_scores = [
            self._safe_float(item.get("score") or item.get("probability") or item.get("confidence"))
            for item in ddx[:2]
        ]
        top_gap = round(max(0.0, (top_scores[0] if top_scores else 0.0) - (top_scores[1] if len(top_scores) > 1 else 0.0)), 4)

        metadata = state.get_metadata()
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        site = self._norm_text(metadata.get("location") or metadata.get("site") or metadata.get("anatomical_site"))
        history = self._norm_text(metadata.get("history") or metadata.get("clinical_history") or metadata.get("past_history"))
        age = self._safe_float(metadata.get("age"))
        malignant_cues = [
            str(x).strip().lower()
            for x in ((state.perception.get("risk_cues", {}) or {}).get("malignant_cues", []) or [])
            if str(x).strip()
        ]
        malignant_count = len({"MEL", "BCC", "SCC"}.intersection(set(top_names)))
        top1_uncertain = (
            str((state.perception.get("uncertainty", {}) or {}).get("level", "high")).lower() in {"medium", "high"}
            or top_gap <= 0.15
        )

        return {
            "top_names": top_names,
            "top_scores": [round(float(x), 4) for x in top_scores],
            "top_gap": top_gap,
            "top_gap_small": top_gap <= 0.15,
            "uncertainty": str((state.perception.get("uncertainty", {}) or {}).get("level", "high")).lower(),
            "retrieval_confidence": str(retrieval_summary.get("retrieval_confidence", "low")).lower(),
            "high_confidence": str(retrieval_summary.get("retrieval_confidence", "low")).lower() == "high" and top_gap >= 0.28,
            "low_confidence": str(retrieval_summary.get("retrieval_confidence", "low")).lower() == "low" or top_gap <= 0.15,
            "top1_uncertain": top1_uncertain,
            "supports_top1": bool(retrieval_summary.get("supports_top1", False)),
            "has_confusion_support": bool(retrieval_summary.get("has_confusion_support", False)),
            "metadata_present": bool(metadata),
            "metadata_keys": list(metadata.keys()),
            "has_malignant_candidate": bool({"MEL", "BCC", "SCC"}.intersection(top_names)),
            "multiple_malignant_candidates": malignant_count >= 2,
            "has_ack_scc_pair": {"ACK", "SCC"}.issubset(set(top_names)),
            "has_bcc_scc_pair": {"BCC", "SCC"}.issubset(set(top_names)),
            "has_bcc_sek_pair": {"BCC", "SEK"}.issubset(set(top_names)),
            "has_mel_nev_pair": {"MEL", "NEV"}.issubset(set(top_names)),
            "age_match": self._age_match_score(age=age, top_names=top_names),
            "site_match": self._site_match_score(site=site, top_names=top_names),
            "sun_exposed_site": self._site_matches(site, ["face", "scalp", "ear", "neck", "nose", "temple", "cheek", "hand", "forearm", "lip"]),
            "strong_invasive_history": any(token in history for token in ["bleed", "bleeding", "rapid growth", "pain", "hurt", "ulcer", "ulcerated"]),
            "has_malignant_cues": bool(malignant_cues),
        }

    def _should_add_compare_skill(
        self,
        *,
        case_features: Dict[str, Any],
        recommended_skills: List[str],
    ) -> tuple[bool, str, Dict[str, Any]]:
        if "compare_skill" in recommended_skills:
            return True, "memory_or_rule_recommended_skill", {
                "recommended_skills": recommended_skills,
                "top_gap": case_features["top_gap"],
            }
        if str(case_features["uncertainty"]) in {"medium", "high"}:
            return True, "uncertainty_threshold", {
                "uncertainty": case_features["uncertainty"],
                "top_gap": case_features["top_gap"],
            }
        if bool(case_features["top_gap_small"]):
            return True, "small_top_gap", {
                "top_gap": case_features["top_gap"],
            }
        return False, "not_triggered", {
            "uncertainty": case_features["uncertainty"],
            "top_gap": case_features["top_gap"],
        }

    def _should_add_pair_specialist(
        self,
        target_pair: tuple[str, str],
        skill_name: str,
        case_features: Dict[str, Any],
        confusion_pairs: List[tuple[str, ...]],
        recommended_skills: List[str],
        state: CaseState,
        allow_metadata_proxy: bool = False,
    ) -> tuple[bool, str]:
        pair_set = set(target_pair)
        specialist_name = skill_name or self._specialist_name_for_pair(pair_set)
        top_names = list(case_features["top_names"])
        ambiguous_context = (
            str(case_features.get("uncertainty", "high")) in {"medium", "high"}
            or bool(case_features.get("top_gap_small", False))
            or bool(case_features.get("has_confusion_support", False))
        )
        learned_score = self._learned_gate_score(
            skill_name=specialist_name,
            case_features=case_features,
            recommended_skills=recommended_skills,
            state=state,
        )
        if pair_set.issubset(set(top_names)):
            if learned_score >= self.LEARNED_SPECIALIST_THRESHOLD:
                return True, "learned_specialist_score"
            if ambiguous_context:
                return True, "pair_present_in_top_k"
            if specialist_name in recommended_skills:
                return True, "memory_or_rule_recommended_skill"
            return False, "pair_present_but_not_ambiguous_enough"
        for pair in confusion_pairs:
            if set(pair) == pair_set:
                return True, "confusion_memory_pair_match"
        if allow_metadata_proxy and pair_set == {"ACK", "SCC"} and self._should_add_ack_scc_specialist_from_metadata(state=state, top_names=top_names):
            return True, "metadata_proxy_support"
        if specialist_name in recommended_skills:
            return True, "memory_or_rule_recommended_skill"
        return False, "not_triggered"

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
        case_features: Dict[str, Any],
        state: CaseState,
    ) -> bool:
        if not bool(case_features["metadata_present"]):
            return False
        learned_score = self._learned_gate_score(
            skill_name="metadata_consistency_skill",
            case_features=case_features,
            recommended_skills=[],
            state=state,
        )
        if learned_score >= self.LEARNED_METADATA_THRESHOLD:
            return True
        if str(case_features["uncertainty"]) in {"medium", "high"}:
            return True
        if str(case_features["retrieval_confidence"]) == "low" or not bool(case_features["supports_top1"]):
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

    def _safe_float(self, value: Any) -> float:
        try:
            if value is None or value == "":
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _age_match_score(self, *, age: float, top_names: List[str]) -> float:
        if age <= 0:
            return 0.0
        if age <= 18 and {"NEV", "SEK"}.intersection(top_names):
            return 1.0
        if age <= 30 and "NEV" in top_names:
            return 1.0
        if age >= 55 and {"ACK", "BCC", "MEL", "SCC"}.intersection(top_names):
            return 1.0
        return 0.0

    def _site_match_score(self, *, site: str, top_names: List[str]) -> float:
        if not site:
            return 0.0
        if self._site_matches(site, ["face", "scalp", "ear", "neck", "nose", "temple", "cheek", "hand", "forearm", "lip"]) and {"ACK", "BCC", "SCC"}.intersection(top_names):
            return 1.0
        if self._site_matches(site, ["trunk", "back", "chest", "abdomen"]) and "NEV" in top_names:
            return 1.0
        if self._site_matches(site, ["leg", "foot", "toe"]) and "MEL" in top_names:
            return 1.0
        return 0.0

    def _learned_gate_score(
        self,
        *,
        skill_name: str,
        case_features: Dict[str, Any],
        recommended_skills: List[str],
        state: CaseState,
    ) -> float:
        if self.controller is None:
            return 0.0
        target_learner = getattr(self.controller, "target_learner", None)
        if target_learner is None:
            return 0.0

        features = {
            "uncertainty_medium": 1.0 if str(case_features.get("uncertainty", "")).lower() == "medium" else 0.0,
            "uncertainty_high": 1.0 if str(case_features.get("uncertainty", "")).lower() == "high" else 0.0,
            "top_gap_small": 1.0 if bool(case_features.get("top_gap_small", False)) else 0.0,
            "top1_uncertain": 1.0 if bool(case_features.get("top1_uncertain", False)) else 0.0,
            "high_confidence": 1.0 if bool(case_features.get("high_confidence", False)) else 0.0,
            "low_confidence": 1.0 if bool(case_features.get("low_confidence", False)) else 0.0,
            "metadata_present": 1.0 if bool(case_features.get("metadata_present", False)) else 0.0,
            "retrieval_low": 1.0 if str(case_features.get("retrieval_confidence", "")).lower() == "low" else 0.0,
            "supports_top1": 1.0 if bool(case_features.get("supports_top1", False)) else 0.0,
            "age_match": float(case_features.get("age_match", 0.0) or 0.0),
            "site_match": float(case_features.get("site_match", 0.0) or 0.0),
            "has_malignant_candidate": 1.0 if bool(case_features.get("has_malignant_candidate", False)) else 0.0,
            "multiple_malignant_candidates": 1.0 if bool(case_features.get("multiple_malignant_candidates", False)) else 0.0,
            "has_confusion_support": 1.0 if bool(case_features.get("has_confusion_support", False)) else 0.0,
            "has_ack_scc_pair": 1.0 if bool(case_features.get("has_ack_scc_pair", False)) else 0.0,
            "has_bcc_scc_pair": 1.0 if bool(case_features.get("has_bcc_scc_pair", False)) else 0.0,
            "has_bcc_sek_pair": 1.0 if bool(case_features.get("has_bcc_sek_pair", False)) else 0.0,
            "has_mel_nev_pair": 1.0 if bool(case_features.get("has_mel_nev_pair", False)) else 0.0,
            "retrieval_recommends_ack_scc": 1.0 if "ack_scc_specialist_skill" in recommended_skills else 0.0,
            "retrieval_recommends_bcc_scc": 1.0 if "bcc_scc_specialist_skill" in recommended_skills else 0.0,
            "retrieval_recommends_bcc_sek": 1.0 if "bcc_sek_specialist_skill" in recommended_skills else 0.0,
            "retrieval_recommends_mel_nev": 1.0 if "mel_nev_specialist_skill" in recommended_skills else 0.0,
        }
        try:
            return float(target_learner.predict_target(skill_name, features, state))
        except Exception:
            return 0.0

    def _site_matches(self, site: str, keywords: List[str]) -> bool:
        if not site:
            return False
        normalized = site.replace("-", " ").replace("/", " ").strip().lower()
        tokens = [token for token in normalized.split() if token]
        return any(keyword in tokens for keyword in keywords)
