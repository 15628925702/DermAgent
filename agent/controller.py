from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.state import CaseState
from memory.skill_index import SkillIndex, sigmoid


class LearnableSkillController:
    """
    Second-version controller.

    This controller keeps skill selection learnable while remaining lightweight:
    - skills are represented as structured objects with trainable weights
    - case state is mapped into a stable numeric feature space
    - controller performs online updates from heuristic targets and outcome signals
    """

    def __init__(
        self,
        skill_index: SkillIndex,
        *,
        learning_rate: float = 0.08,
        max_skills: int = 4,
    ) -> None:
        self.skill_index = skill_index
        self.learning_rate = learning_rate
        self.max_skills = max_skills
        self.stop_weights: Dict[str, float] = {
            "bias": 0.15,
            "uncertainty_low": 0.9,
            "uncertainty_medium": 0.15,
            "uncertainty_high": -0.75,
            "retrieval_high": 0.75,
            "retrieval_medium": 0.2,
            "retrieval_low": -0.6,
            "supports_top1": 0.85,
            "has_confusion_support": -0.7,
            "top_gap_small": -0.65,
        }

    def select_skills(
        self,
        state: CaseState,
        *,
        rule_priors: List[str] | None = None,
        planner_context: Dict[str, Any] | None = None,
        allowed_skills: set[str] | None = None,
    ) -> Dict[str, Any]:
        planner_context = planner_context or {}
        features = self.extract_features(state, planner_context=planner_context)
        rule_priors = list(rule_priors or [])
        allowed_skills = set(allowed_skills or [])
        recommended_skills = {
            str(x).strip()
            for x in planner_context.get("recommended_skills", [])
            if str(x).strip()
        }

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for spec in self.skill_index.routable_specs():
            if allowed_skills and spec.skill_id not in allowed_skills:
                continue
            extra_bias = 0.0
            reasons: List[str] = []

            if spec.skill_id in rule_priors:
                extra_bias += 0.75
                reasons.append("rule_prior")

            if spec.skill_id in recommended_skills:
                extra_bias += 0.45
                reasons.append("retrieval_recommendation")

            planner_bias, planner_reasons = self._planner_extra_bias(spec.skill_id, planner_context=planner_context)
            if abs(planner_bias) > 1e-8:
                extra_bias += planner_bias
                reasons.extend(planner_reasons)

            logit = spec.logit(features, extra_bias=extra_bias)
            probability = sigmoid(logit)
            score_item = {
                "skill_id": spec.skill_id,
                "logit": round(logit, 4),
                "probability": round(probability, 4),
                "threshold": spec.threshold,
                "selected": probability >= spec.threshold,
                "extra_bias": round(extra_bias, 4),
                "success_rate": round(spec.success_rate(), 4),
                "reasons": reasons,
            }
            scored.append((probability, spec.skill_id, score_item))

        scored.sort(key=lambda item: (item[0], item[2]["logit"]), reverse=True)

        selected = [
            skill_id
            for probability, skill_id, item in scored
            if item["selected"]
        ]
        if "uncertainty_assessment_skill" not in selected:
            selected.insert(0, "uncertainty_assessment_skill")

        if len(selected) == 1:
            for probability, skill_id, item in scored:
                if skill_id == "uncertainty_assessment_skill":
                    continue
                if skill_id not in selected:
                    selected.append(skill_id)
                    item["selected"] = True
                    item["reasons"].append("min_skill_floor")
                    break

        if len(selected) > self.max_skills:
            kept: List[str] = []
            if "uncertainty_assessment_skill" in selected:
                kept.append("uncertainty_assessment_skill")
            for probability, skill_id, item in scored:
                if skill_id in kept or skill_id not in selected:
                    continue
                kept.append(skill_id)
                if len(kept) >= self.max_skills:
                    break
            selected = kept

        stop_probability = self._estimate_stop_probability(features)
        controller_debug = {
            "case_features": {key: round(float(value), 4) for key, value in features.items()},
            "recommended_skills": sorted(recommended_skills),
            "rule_priors": rule_priors,
            "scored_skills": [item for _, _, item in scored],
        }

        result = {
            "selected_skills": selected,
            "skill_scores": {item["skill_id"]: item for _, _, item in scored},
            "stop_probability": round(stop_probability, 4),
            "controller_debug": controller_debug,
            "controller_mode": "learnable_hybrid",
        }

        state.controller = result
        return result

    def extract_features(
        self,
        state: CaseState,
        *,
        planner_context: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        planner_context = planner_context or {}
        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [str(item.get("name", "")).strip().upper() for item in ddx[:5] if str(item.get("name", "")).strip()]
        top_scores = [self._safe_float(item.get("score"), default=0.0) for item in ddx[:2]]
        top_gap = max(0.0, (top_scores[0] if top_scores else 0.0) - (top_scores[1] if len(top_scores) > 1 else 0.0))

        planner_case = planner_context.get("case_features", {}) or (state.planner.get("case_features", {}) or {})
        planner_trace = planner_context.get("decision_trace", []) or (state.planner.get("decision_trace", []) or [])
        planner_flags = planner_context.get("flags", {}) or (state.planner.get("flags", {}) or {})

        uncertainty = str(planner_case.get("uncertainty", state.get_uncertainty_level())).lower()
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        retrieval_confidence = str(planner_case.get("retrieval_confidence", retrieval_summary.get("retrieval_confidence", "low"))).lower()
        metadata = state.get_metadata()
        history_text = self._norm_text(
            metadata.get("history")
            or metadata.get("clinical_history")
            or metadata.get("past_history")
        )
        site = self._norm_text(
            metadata.get("location")
            or metadata.get("site")
            or metadata.get("anatomical_site")
        )
        risk_cues = state.perception.get("risk_cues", {}) or {}
        malignant_cues = risk_cues.get("malignant_cues", []) or []
        suspicious_cues = risk_cues.get("suspicious_cues", []) or []
        visual_cues = state.perception.get("visual_cues", []) or []
        support_strength = retrieval_summary.get("support_strength", {}) or {}
        recommended_skills = {
            str(x).strip()
            for x in (planner_flags.get("recommended_skills", []) or retrieval_summary.get("recommended_skills", []))
            if str(x).strip()
        }
        memory_recommended_skills = {
            str(x).strip()
            for x in (planner_flags.get("memory_recommended_skills", []) or [])
            if str(x).strip()
        }
        rule_recommended_skills = {
            str(x).strip()
            for x in (planner_flags.get("rule_recommended_skills", []) or [])
            if str(x).strip()
        }
        trace_map = self._decision_trace_map(planner_trace)
        top_names = list(planner_case.get("top_names", top_names))
        top_gap = float(planner_case.get("top_gap", round(top_gap, 4)))

        features = {
            "bias": 1.0,
            "uncertainty_low": 1.0 if uncertainty == "low" else 0.0,
            "uncertainty_medium": 1.0 if uncertainty == "medium" else 0.0,
            "uncertainty_high": 1.0 if uncertainty == "high" else 0.0,
            "top_gap_small": 1.0 if bool(planner_case.get("top_gap_small", top_gap <= 0.15)) else 0.0,
            "top_gap": round(top_gap, 4),
            "top_candidate_count": min(1.0, len(top_names) / 3.0),
            "has_malignant_candidate": 1.0 if bool(planner_case.get("has_malignant_candidate", bool({"MEL", "BCC", "SCC"}.intersection(top_names)))) else 0.0,
            "malignant_candidate_ratio": round(
                len({"MEL", "BCC", "SCC"}.intersection(set(top_names))) / 3.0,
                4,
            ),
            "has_ack_scc_pair": 1.0 if bool(planner_case.get("has_ack_scc_pair", {"ACK", "SCC"}.issubset(set(top_names)))) else 0.0,
            "has_bcc_scc_pair": 1.0 if bool(planner_case.get("has_bcc_scc_pair", {"BCC", "SCC"}.issubset(set(top_names)))) else 0.0,
            "has_bcc_sek_pair": 1.0 if bool(planner_case.get("has_bcc_sek_pair", {"BCC", "SEK"}.issubset(set(top_names)))) else 0.0,
            "has_mel_nev_pair": 1.0 if bool(planner_case.get("has_mel_nev_pair", {"MEL", "NEV"}.issubset(set(top_names)))) else 0.0,
            "retrieval_high": 1.0 if retrieval_confidence == "high" else 0.0,
            "retrieval_medium": 1.0 if retrieval_confidence == "medium" else 0.0,
            "retrieval_low": 1.0 if retrieval_confidence == "low" else 0.0,
            "supports_top1": 1.0 if bool(planner_case.get("supports_top1", retrieval_summary.get("supports_top1", False))) else 0.0,
            "has_confusion_support": 1.0 if bool(planner_case.get("has_confusion_support", retrieval_summary.get("has_confusion_support", False))) else 0.0,
            "metadata_present": 1.0 if bool(planner_case.get("metadata_present", bool(metadata))) else 0.0,
            "sun_exposed_site": 1.0 if bool(planner_case.get("sun_exposed_site", self._site_matches(site, ["face", "scalp", "ear", "neck", "nose", "temple", "cheek", "hand", "forearm", "lip"]))) else 0.0,
            "strong_invasive_history": 1.0 if bool(planner_case.get("strong_invasive_history", any(token in history_text for token in ["bleed", "bleeding", "rapid growth", "pain", "hurt", "ulcer", "ulcerated"]))) else 0.0,
            "num_visual_cues": min(1.0, len(visual_cues) / 6.0),
            "num_malignant_cues": min(1.0, len(malignant_cues) / 4.0),
            "num_suspicious_cues": min(1.0, len(suspicious_cues) / 4.0),
            "raw_case_support": min(1.0, self._safe_float(support_strength.get("raw_case"), default=0.0) / 3.0),
            "prototype_support": min(1.0, self._safe_float(support_strength.get("prototype"), default=0.0) / 3.0),
            "confusion_support": min(1.0, self._safe_float(support_strength.get("confusion"), default=0.0) / 2.0),
            "retrieval_recommends_ack_scc": 1.0 if "ack_scc_specialist_skill" in recommended_skills else 0.0,
            "retrieval_recommends_bcc_scc": 1.0 if "bcc_scc_specialist_skill" in recommended_skills else 0.0,
            "retrieval_recommends_bcc_sek": 1.0 if "bcc_sek_specialist_skill" in recommended_skills else 0.0,
            "retrieval_recommends_mel_nev": 1.0 if "mel_nev_specialist_skill" in recommended_skills else 0.0,
            "planner_compare_selected": 1.0 if trace_map.get("compare_skill", {}).get("selected", False) else 0.0,
            "planner_metadata_selected": 1.0 if trace_map.get("metadata_consistency_skill", {}).get("selected", False) else 0.0,
            "planner_malignancy_selected": 1.0 if trace_map.get("malignancy_risk_skill", {}).get("selected", False) else 0.0,
            "planner_ack_scc_signal": 1.0 if trace_map.get("ack_scc_specialist_skill", {}).get("selected", False) else 0.0,
            "planner_bcc_scc_signal": 1.0 if trace_map.get("bcc_scc_specialist_skill", {}).get("selected", False) else 0.0,
            "planner_bcc_sek_signal": 1.0 if trace_map.get("bcc_sek_specialist_skill", {}).get("selected", False) else 0.0,
            "planner_mel_nev_signal": 1.0 if trace_map.get("mel_nev_specialist_skill", {}).get("selected", False) else 0.0,
            "planner_memory_signal": 1.0 if bool(memory_recommended_skills or rule_recommended_skills) else 0.0,
            "planner_rule_density": min(1.0, len(rule_recommended_skills) / 3.0),
            "planner_memory_density": min(1.0, len(memory_recommended_skills) / 3.0),
            "planner_retrieval_rule_overlap": 1.0 if bool(memory_recommended_skills and rule_recommended_skills) else 0.0,
        }
        return features

    def update_from_case(self, state: CaseState) -> Dict[str, Any]:
        features = self.extract_features(
            state,
            planner_context={
                "case_features": state.planner.get("case_features", {}) or {},
                "decision_trace": state.planner.get("decision_trace", []) or [],
                "flags": state.planner.get("flags", {}) or {},
            },
        )
        targets = self._build_targets(state)
        predicted = (state.controller or {}).get("skill_scores", {})
        feedback: Dict[str, Any] = {
            "targets": targets,
            "updated_skills": [],
        }
        final_label = str(
            (state.final_decision or {}).get("final_label")
            or (state.final_decision or {}).get("diagnosis")
            or ""
        ).strip().upper()
        true_label = str(
            (state.case_info or {}).get("true_label", "")
        ).strip().upper()
        is_correct = bool(true_label) and final_label == true_label

        for spec in self.skill_index.routable_specs():
            prediction = self._safe_float(
                (predicted.get(spec.skill_id, {}) or {}).get("probability"),
                default=spec.probability(features),
            )
            target = float(targets.get(spec.skill_id, 0.0))
            spec.update_weights(
                features=features,
                target=target,
                prediction=prediction,
                learning_rate=self.learning_rate,
            )
            helpful = spec.skill_id in state.selected_skills and (is_correct or target >= 1.0)
            if spec.skill_id in state.selected_skills:
                spec.record_use(helpful=helpful)
            feedback["updated_skills"].append(
                {
                    "skill_id": spec.skill_id,
                    "target": round(target, 4),
                    "prediction": round(prediction, 4),
                    "success_rate": round(spec.success_rate(), 4),
                }
            )

        stop_target = 1.0 if is_correct and str((state.final_decision or {}).get("confidence", "low")).lower() in {"medium", "high"} else 0.0
        stop_prediction = self._estimate_stop_probability(features)
        error = stop_target - stop_prediction
        for name, value in features.items():
            self.stop_weights[name] = float(self.stop_weights.get(name, 0.0)) + self.learning_rate * error * float(value)
        feedback["stop_target"] = round(stop_target, 4)
        feedback["stop_prediction"] = round(stop_prediction, 4)
        feedback["is_correct"] = is_correct
        return feedback

    def _planner_extra_bias(
        self,
        skill_id: str,
        *,
        planner_context: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        trace_map = self._decision_trace_map(planner_context.get("decision_trace", []) or [])
        item = trace_map.get(skill_id, {}) or {}
        trigger = str(item.get("trigger", "")).strip()
        if not trigger or not bool(item.get("selected", False)):
            return 0.0, []

        bias_map = {
            "always_on_core_skill": 0.15,
            "uncertainty_threshold": 0.08,
            "small_top_gap": 0.1,
            "memory_or_rule_recommended_skill": 0.12,
            "pair_present_in_top_k": 0.14,
            "confusion_memory_pair_match": 0.12,
            "metadata_proxy_support": 0.1,
            "malignant_candidate_in_top_k": 0.1,
            "metadata_or_support_check": 0.08,
        }
        bias = float(bias_map.get(trigger, 0.0))
        if bias == 0.0:
            return 0.0, []
        return bias, [f"planner:{trigger}"]

    def _decision_trace_map(self, items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        mapped: Dict[str, Dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            skill = str(item.get("skill", "")).strip()
            if skill:
                mapped[skill] = item
        return mapped

    def _estimate_stop_probability(self, features: Dict[str, float]) -> float:
        logit = 0.0
        for name, weight in self.stop_weights.items():
            logit += float(weight) * float(features.get(name, 0.0))
        return sigmoid(logit)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_skills": self.max_skills,
            "stop_weights": {
                key: round(float(value), 6)
                for key, value in sorted(self.stop_weights.items())
            },
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("learning_rate") is not None:
            self.learning_rate = float(payload["learning_rate"])
        if payload.get("max_skills") is not None:
            self.max_skills = int(payload["max_skills"])
        stop_weights = payload.get("stop_weights", {}) or {}
        if stop_weights:
            self.stop_weights = {
                str(key): float(value)
                for key, value in stop_weights.items()
            }

    def _build_targets(self, state: CaseState) -> Dict[str, float]:
        top_names = state.get_top_ddx_names(top_k=3)
        metadata = state.get_metadata()
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        true_label = str((state.case_info or {}).get("true_label", "")).strip().upper()
        final_label = str(
            (state.final_decision or {}).get("final_label")
            or (state.final_decision or {}).get("diagnosis")
            or ""
        ).strip().upper()
        incorrect = bool(true_label) and final_label and true_label != final_label

        targets = {
            "uncertainty_assessment_skill": 1.0,
            "compare_skill": 1.0 if state.get_uncertainty_level() in {"medium", "high"} or incorrect or len(top_names) >= 2 else 0.0,
            "malignancy_risk_skill": 1.0 if true_label in {"MEL", "BCC", "SCC"} or {"MEL", "BCC", "SCC"}.intersection(top_names) else 0.0,
            "metadata_consistency_skill": 1.0 if metadata and (
                str(retrieval_summary.get("retrieval_confidence", "low")).lower() == "low"
                or not retrieval_summary.get("supports_top1", False)
                or incorrect
            ) else 0.0,
            "ack_scc_specialist_skill": 1.0 if {"ACK", "SCC"}.issubset(set(top_names)) or true_label in {"ACK", "SCC"} and "SCC" in top_names else 0.0,
            "bcc_scc_specialist_skill": 1.0 if {"BCC", "SCC"}.issubset(set(top_names)) or true_label in {"BCC", "SCC"} and bool({"BCC", "SCC"}.intersection(set(top_names))) else 0.0,
            "bcc_sek_specialist_skill": 1.0 if {"BCC", "SEK"}.issubset(set(top_names)) or true_label in {"BCC", "SEK"} and bool({"BCC", "SEK"}.intersection(set(top_names))) else 0.0,
            "mel_nev_specialist_skill": 1.0 if {"MEL", "NEV"}.issubset(set(top_names)) or true_label in {"MEL", "NEV"} and "NEV" in top_names else 0.0,
        }
        return targets

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

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
