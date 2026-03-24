from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.state import CaseState
from memory.skill_index import SkillIndex, sigmoid


class TargetLearner:
    """可学习的技能目标函数管理器"""

    def __init__(self, learning_rate: float = 0.05, use_adam: bool = True):
        self.learning_rate = learning_rate
        self.use_adam = use_adam

        # Adam优化器参数
        if use_adam:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.m = {}  # 一阶矩
            self.v = {}  # 二阶矩
            self.t = 0   # 时间步

        # 基础目标权重
        self.base_targets = {
            "uncertainty_assessment_skill": 1.0,  # 始终运行
            "compare_skill": 0.5,
            "malignancy_risk_skill": 0.4,
            "metadata_consistency_skill": 0.6,
            "ack_scc_specialist_skill": 0.3,
            "bcc_scc_specialist_skill": 0.3,
            "bcc_sek_specialist_skill": 0.3,
            "mel_nev_specialist_skill": 0.3,
        }

        # 条件权重 - 将启发式变成特征权重
        self.condition_weights = {
            # compare_skill 条件
            "compare_uncertainty_medium": 0.35,
            "compare_uncertainty_high": 0.5,
            "compare_incorrect": 0.4,
            "compare_top_gap_small": 0.3,
            "compare_has_multiple_candidates": 0.2,
            "compare_high_confidence": 0.4,
            "compare_low_confidence": -0.2,

            # malignancy_risk_skill 条件
            "malignancy_true_malignant": 0.4,
            "malignancy_has_malignant_candidate": 0.3,
            "malignancy_multiple_malignant": 0.5,

            # metadata_consistency_skill 条件
            "metadata_has_metadata": 0.3,
            "metadata_low_confidence": 0.3,
            "metadata_not_supports_top1": 0.2,
            "metadata_incorrect": 0.3,
            "metadata_age_match": 0.25,
            "metadata_site_match": 0.2,

            # specialist skills 条件
            "specialist_pair_present": 0.6,
            "specialist_retrieval_recommends": 0.4,
            "specialist_confusion_support": 0.3,
            "specialist_top1_uncertain": 0.35,
        }

    def predict_target(self, skill_id: str, features: Dict[str, float], state: CaseState) -> float:
        """基于特征预测技能目标值"""
        base = self.base_targets.get(skill_id, 0.0)

        if skill_id == "uncertainty_assessment_skill":
            return 1.0  # 始终运行

        elif skill_id == "compare_skill":
            # 条件累加
            conditions = [
                features.get("uncertainty_medium", 0.0) * self.condition_weights["compare_uncertainty_medium"],
                features.get("uncertainty_high", 0.0) * self.condition_weights["compare_uncertainty_high"],
                (1.0 if self._is_incorrect(state) else 0.0) * self.condition_weights["compare_incorrect"],
                features.get("top_gap_small", 0.0) * self.condition_weights["compare_top_gap_small"],
                (1.0 if len(state.get_top_ddx_names()) >= 2 else 0.0) * self.condition_weights["compare_has_multiple_candidates"],
                features.get("high_confidence", 0.0) * self.condition_weights["compare_high_confidence"],
                features.get("low_confidence", 0.0) * self.condition_weights["compare_low_confidence"],
            ]
            return base + sum(conditions)

        elif skill_id == "malignancy_risk_skill":
            true_malignant = 1.0 if self._is_true_malignant(state) else 0.0
            has_candidate = features.get("has_malignant_candidate", 0.0)
            multiple_malignant = features.get("multiple_malignant_candidates", 0.0)
            return base + (
                true_malignant * self.condition_weights["malignancy_true_malignant"] +
                has_candidate * self.condition_weights["malignancy_has_malignant_candidate"] +
                multiple_malignant * self.condition_weights["malignancy_multiple_malignant"]
            )

        elif skill_id == "metadata_consistency_skill":
            has_metadata = features.get("metadata_present", 0.0)
            low_conf = features.get("retrieval_low", 0.0)
            not_supports = 1.0 - features.get("supports_top1", 0.0)
            incorrect = 1.0 if self._is_incorrect(state) else 0.0
            age_match = features.get("age_match", 0.0)
            site_match = features.get("site_match", 0.0)
            return base + (
                has_metadata * self.condition_weights["metadata_has_metadata"] +
                low_conf * self.condition_weights["metadata_low_confidence"] +
                not_supports * self.condition_weights["metadata_not_supports_top1"] +
                incorrect * self.condition_weights["metadata_incorrect"] +
                age_match * self.condition_weights["metadata_age_match"] +
                site_match * self.condition_weights["metadata_site_match"]
            )

        elif skill_id in ["ack_scc_specialist_skill", "bcc_scc_specialist_skill",
                         "bcc_sek_specialist_skill", "mel_nev_specialist_skill"]:
            pair_present = self._has_specialist_pair(skill_id, features)
            retrieval_rec = self._retrieval_recommends(skill_id, features)
            confusion_sup = features.get("has_confusion_support", 0.0)
            top1_uncertain = features.get("top1_uncertain", 0.0)
            return base + (
                pair_present * self.condition_weights["specialist_pair_present"] +
                retrieval_rec * self.condition_weights["specialist_retrieval_recommends"] +
                confusion_sup * self.condition_weights["specialist_confusion_support"] +
                top1_uncertain * self.condition_weights["specialist_top1_uncertain"]
            )

        return base

    def update_from_case(
        self,
        skill_id: str,
        features: Dict[str, float],
        state: CaseState,
        actual_helpful: bool | None,
        learning_rate: float = None,
    ) -> None:
        """从案例中学习目标函数参数"""
        if actual_helpful is None:
            return
        lr = learning_rate or self.learning_rate
        predicted = self.predict_target(skill_id, features, state)
        target = 1.0 if actual_helpful else 0.0
        error = target - predicted

        # 更新基础目标
        if skill_id in self.base_targets:
            if self.use_adam:
                self._adam_update(skill_id, lr * error)
            else:
                self.base_targets[skill_id] += lr * error

        # 更新条件权重
        if skill_id == "compare_skill":
            self._update_condition_weights([
                ("compare_uncertainty_medium", features.get("uncertainty_medium", 0.0)),
                ("compare_uncertainty_high", features.get("uncertainty_high", 0.0)),
                ("compare_incorrect", 1.0 if self._is_incorrect(state) else 0.0),
                ("compare_top_gap_small", features.get("top_gap_small", 0.0)),
                ("compare_has_multiple_candidates", 1.0 if len(state.get_top_ddx_names()) >= 2 else 0.0),                ("compare_high_confidence", features.get("high_confidence", 0.0)),
                ("compare_low_confidence", features.get("low_confidence", 0.0)),            ], error, lr)

        elif skill_id == "malignancy_risk_skill":
            self._update_condition_weights([
                ("malignancy_true_malignant", 1.0 if self._is_true_malignant(state) else 0.0),
                ("malignancy_has_malignant_candidate", features.get("has_malignant_candidate", 0.0)),
                ("malignancy_multiple_malignant", features.get("multiple_malignant_candidates", 0.0)),
            ], error, lr)

        elif skill_id == "metadata_consistency_skill":
            self._update_condition_weights([
                ("metadata_has_metadata", features.get("metadata_present", 0.0)),
                ("metadata_low_confidence", features.get("retrieval_low", 0.0)),
                ("metadata_not_supports_top1", 1.0 - features.get("supports_top1", 0.0)),
                ("metadata_incorrect", 1.0 if self._is_incorrect(state) else 0.0),
                ("metadata_age_match", features.get("age_match", 0.0)),
                ("metadata_site_match", features.get("site_match", 0.0)),
            ], error, lr)

        elif skill_id in ["ack_scc_specialist_skill", "bcc_scc_specialist_skill",
                         "bcc_sek_specialist_skill", "mel_nev_specialist_skill"]:
            self._update_condition_weights([
                ("specialist_pair_present", self._has_specialist_pair(skill_id, features)),
                ("specialist_retrieval_recommends", self._retrieval_recommends(skill_id, features)),
                ("specialist_confusion_support", features.get("has_confusion_support", 0.0)),
                ("specialist_top1_uncertain", features.get("top1_uncertain", 0.0)),
            ], error, lr)

    def _update_condition_weights(self, conditions: List[Tuple[str, float]], error: float, lr: float) -> None:
        """更新条件权重"""
        for cond_name, cond_value in conditions:
            if cond_name in self.condition_weights:
                gradient = lr * error * cond_value
                if self.use_adam:
                    self._adam_update(cond_name, gradient)
                else:
                    self.condition_weights[cond_name] += gradient

    def _adam_update(self, param_name: str, gradient: float) -> None:
        """Adam优化器更新"""
        self.t += 1

        # 初始化矩
        if param_name not in self.m:
            self.m[param_name] = 0.0
            self.v[param_name] = 0.0

        # 更新一阶矩
        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradient
        # 更新二阶矩
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * gradient**2

        # 偏差校正
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)

        # 更新参数
        if param_name in self.condition_weights:
            self.condition_weights[param_name] += self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)
        elif param_name in self.base_targets:
            self.base_targets[param_name] += self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)

    def _is_incorrect(self, state: CaseState) -> bool:
        true_label = str((state.case_info or {}).get("true_label", "")).strip().upper()
        final_label = str(
            (state.final_decision or {}).get("final_label")
            or (state.final_decision or {}).get("diagnosis")
            or ""
        ).strip().upper()
        return bool(true_label) and final_label and true_label != final_label

    def _is_true_malignant(self, state: CaseState) -> bool:
        true_label = str((state.case_info or {}).get("true_label", "")).strip().upper()
        return true_label in {"MEL", "BCC", "SCC"}

    def _has_specialist_pair(self, skill_id: str, features: Dict[str, float]) -> float:
        pair_map = {
            "ack_scc_specialist_skill": "has_ack_scc_pair",
            "bcc_scc_specialist_skill": "has_bcc_scc_pair",
            "bcc_sek_specialist_skill": "has_bcc_sek_pair",
            "mel_nev_specialist_skill": "has_mel_nev_pair",
        }
        return features.get(pair_map.get(skill_id, ""), 0.0)

    def _retrieval_recommends(self, skill_id: str, features: Dict[str, float]) -> float:
        rec_map = {
            "ack_scc_specialist_skill": "retrieval_recommends_ack_scc",
            "bcc_scc_specialist_skill": "retrieval_recommends_bcc_scc",
            "bcc_sek_specialist_skill": "retrieval_recommends_bcc_sek",
            "mel_nev_specialist_skill": "retrieval_recommends_mel_nev",
        }
        return features.get(rec_map.get(skill_id, ""), 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "base_targets": {k: round(v, 4) for k, v in self.base_targets.items()},
            "condition_weights": {k: round(v, 4) for k, v in self.condition_weights.items()},
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if "base_targets" in payload:
            self.base_targets.update(payload["base_targets"])
        if "condition_weights" in payload:
            self.condition_weights.update(payload["condition_weights"])


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
        use_adam: bool = True,
    ) -> None:
        self.skill_index = skill_index
        self.learning_rate = learning_rate
        self.max_skills = max_skills
        self.target_learner = TargetLearner(learning_rate=learning_rate, use_adam=use_adam)
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
        stop_probability = self._estimate_stop_probability(features)

        selected = [
            skill_id
            for probability, skill_id, item in scored
            if item["selected"]
        ]
        if "uncertainty_assessment_skill" not in selected:
            selected.insert(0, "uncertainty_assessment_skill")

        desired_skill_floor = self._desired_skill_floor(
            features,
            rule_priors=rule_priors,
            recommended_skills=recommended_skills,
            stop_probability=stop_probability,
        )
        floor_retained: List[str] = []
        if len(selected) < desired_skill_floor:
            for probability, skill_id, item in scored:
                if skill_id == "uncertainty_assessment_skill":
                    continue
                if skill_id not in selected:
                    selected.append(skill_id)
                    item["selected"] = True
                    item["reasons"].append("dynamic_skill_floor")
                    floor_retained.append(skill_id)
                    if len(selected) >= desired_skill_floor:
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

        controller_debug = {
            "case_features": {key: round(float(value), 4) for key, value in features.items()},
            "recommended_skills": sorted(recommended_skills),
            "rule_priors": rule_priors,
            "desired_skill_floor": desired_skill_floor,
            "floor_retained_skills": floor_retained,
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
        age = self._safe_float(metadata.get("age"), default=-1.0)
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
        malignant_count = len({"MEL", "BCC", "SCC"}.intersection(set(top_names)))
        top1_uncertain = (
            str(uncertainty) in {"medium", "high"}
            or bool(planner_case.get("top_gap_small", top_gap <= 0.15))
        )
        age_match = self._age_match_score(age=age, top_names=top_names)
        site_match = self._site_match_score(site=site, top_names=top_names)
        high_confidence = (
            retrieval_confidence == "high"
            and bool(planner_case.get("supports_top1", retrieval_summary.get("supports_top1", False)))
            and top_gap >= 0.28
        )
        low_confidence = (
            retrieval_confidence == "low"
            or str(uncertainty) == "high"
            or bool(planner_case.get("top_gap_small", top_gap <= 0.15))
        )

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
                malignant_count / 3.0,
                4,
            ),
            "multiple_malignant_candidates": 1.0 if malignant_count >= 2 else 0.0,
            "has_ack_scc_pair": 1.0 if bool(planner_case.get("has_ack_scc_pair", {"ACK", "SCC"}.issubset(set(top_names)))) else 0.0,
            "has_bcc_scc_pair": 1.0 if bool(planner_case.get("has_bcc_scc_pair", {"BCC", "SCC"}.issubset(set(top_names)))) else 0.0,
            "has_bcc_sek_pair": 1.0 if bool(planner_case.get("has_bcc_sek_pair", {"BCC", "SEK"}.issubset(set(top_names)))) else 0.0,
            "has_mel_nev_pair": 1.0 if bool(planner_case.get("has_mel_nev_pair", {"MEL", "NEV"}.issubset(set(top_names)))) else 0.0,
            "retrieval_high": 1.0 if retrieval_confidence == "high" else 0.0,
            "retrieval_medium": 1.0 if retrieval_confidence == "medium" else 0.0,
            "retrieval_low": 1.0 if retrieval_confidence == "low" else 0.0,
            "high_confidence": 1.0 if high_confidence else 0.0,
            "low_confidence": 1.0 if low_confidence else 0.0,
            "top1_uncertain": 1.0 if top1_uncertain else 0.0,
            "supports_top1": 1.0 if bool(planner_case.get("supports_top1", retrieval_summary.get("supports_top1", False))) else 0.0,
            "has_confusion_support": 1.0 if bool(planner_case.get("has_confusion_support", retrieval_summary.get("has_confusion_support", False))) else 0.0,
            "metadata_present": 1.0 if bool(planner_case.get("metadata_present", bool(metadata))) else 0.0,
            "age_match": round(age_match, 4),
            "site_match": round(site_match, 4),
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
        selected_skills = set(state.selected_skills or [])
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
            was_executed = spec.skill_id in selected_skills
            helpful = was_executed and (is_correct or target >= 0.65)
            if was_executed:
                spec.record_use(helpful=helpful)

            # 更新目标学习器
            observed_helpfulness = helpful if was_executed else None
            self.target_learner.update_from_case(
                spec.skill_id,
                features,
                state,
                observed_helpfulness,
                self.learning_rate,
            )

            feedback["updated_skills"].append(
                {
                    "skill_id": spec.skill_id,
                    "target": round(target, 4),
                    "prediction": round(prediction, 4),
                    "success_rate": round(spec.success_rate(), 4),
                    "executed": was_executed,
                    "observed_helpful": observed_helpfulness,
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

    def _desired_skill_floor(
        self,
        features: Dict[str, float],
        *,
        rule_priors: List[str],
        recommended_skills: set[str],
        stop_probability: float,
    ) -> int:
        difficulty = 0.0
        difficulty += 1.0 * float(features.get("uncertainty_high", 0.0))
        difficulty += 0.55 * float(features.get("uncertainty_medium", 0.0))
        difficulty += 0.8 * float(features.get("top_gap_small", 0.0))
        difficulty += 0.65 * float(features.get("has_confusion_support", 0.0))
        difficulty += 0.45 * (1.0 - float(features.get("supports_top1", 0.0)))
        difficulty += 0.35 * float(features.get("retrieval_low", 0.0))
        difficulty += 0.15 * min(2, len(rule_priors))
        difficulty += 0.12 * min(2, len(recommended_skills))
        if stop_probability <= 0.18:
            difficulty += 0.3

        floor = 2
        if difficulty >= 1.6:
            floor += 1
        if difficulty >= 2.8 and self.max_skills >= 4:
            floor += 1
        return max(1, min(self.max_skills, floor))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "max_skills": self.max_skills,
            "target_learner": self.target_learner.to_dict(),
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
        if "target_learner" in payload:
            self.target_learner.load_state(payload["target_learner"])
        stop_weights = payload.get("stop_weights", {}) or {}
        if stop_weights:
            self.stop_weights = {
                str(key): float(value)
                for key, value in stop_weights.items()
            }

    def _build_targets(self, state: CaseState) -> Dict[str, float]:
        # 提取特征用于目标预测
        features = self.extract_features(state, planner_context={
            "case_features": state.planner.get("case_features", {}) or {},
            "decision_trace": state.planner.get("decision_trace", []) or [],
            "flags": state.planner.get("flags", {}) or {},
        })

        # 使用可学习的目标函数预测每个技能的目标值
        targets = {}
        for skill_id in [
            "uncertainty_assessment_skill",
            "compare_skill",
            "malignancy_risk_skill",
            "metadata_consistency_skill",
            "ack_scc_specialist_skill",
            "bcc_scc_specialist_skill",
            "bcc_sek_specialist_skill",
            "mel_nev_specialist_skill",
        ]:
            targets[skill_id] = self.target_learner.predict_target(skill_id, features, state)

        # 应用决策支持调整（保持原有逻辑）
        skill_outputs = state.skill_outputs or {}
        true_label = str((state.case_info or {}).get("true_label", "")).strip().upper()
        final_label = str(
            (state.final_decision or {}).get("final_label")
            or (state.final_decision or {}).get("diagnosis")
            or ""
        ).strip().upper()

        targets["compare_skill"] = self._decision_support_target(
            output=skill_outputs.get("compare_skill", {}) or {},
            true_label=true_label,
            fallback=targets["compare_skill"],
        )
        targets["ack_scc_specialist_skill"] = self._decision_support_target(
            output=skill_outputs.get("ack_scc_specialist_skill", {}) or {},
            true_label=true_label,
            fallback=targets["ack_scc_specialist_skill"],
        )
        targets["bcc_scc_specialist_skill"] = self._decision_support_target(
            output=skill_outputs.get("bcc_scc_specialist_skill", {}) or {},
            true_label=true_label,
            fallback=targets["bcc_scc_specialist_skill"],
        )
        targets["bcc_sek_specialist_skill"] = self._decision_support_target(
            output=skill_outputs.get("bcc_sek_specialist_skill", {}) or {},
            true_label=true_label,
            fallback=targets["bcc_sek_specialist_skill"],
        )
        targets["mel_nev_specialist_skill"] = self._decision_support_target(
            output=skill_outputs.get("mel_nev_specialist_skill", {}) or {},
            true_label=true_label,
            fallback=targets["mel_nev_specialist_skill"],
        )
        targets["metadata_consistency_skill"] = self._metadata_target(
            output=skill_outputs.get("metadata_consistency_skill", {}) or {},
            true_label=true_label,
            final_label=final_label,
            fallback=targets["metadata_consistency_skill"],
        )
        targets["malignancy_risk_skill"] = self._malignancy_target(
            output=skill_outputs.get("malignancy_risk_skill", {}) or {},
            true_label=true_label,
            fallback=targets["malignancy_risk_skill"],
        )
        return {
            skill_id: round(self._clamp01(value), 4)
            for skill_id, value in targets.items()
        }

    def _decision_support_target(
        self,
        *,
        output: Dict[str, Any],
        true_label: str,
        fallback: float,
    ) -> float:
        if not output:
            return fallback
        local = output.get("local_decision", {}) or {}
        supports = self._norm_label(
            local.get("supports")
            or output.get("supports")
            or output.get("supported_label")
            or output.get("winner")
            or output.get("recommendation")
        )
        opposes = self._norm_label(
            local.get("opposes")
            or output.get("loser")
        )
        confidence = self._safe_float(
            local.get("confidence", output.get("confidence")),
            default=0.0,
        )
        applicable = self._safe_float(
            local.get("applicable", output.get("applicable")),
            default=1.0,
        )
        if applicable <= 0.05:
            return min(fallback, 0.05)

        positive_target = min(1.0, 0.4 + 0.35 * confidence + 0.25 * applicable)
        negative_target = max(0.0, 0.2 - 0.15 * confidence)

        if true_label and supports == true_label:
            return max(fallback, positive_target)
        if true_label and opposes == true_label:
            return min(fallback, 0.05)
        if true_label and supports not in {"", "UNKNOWN"} and supports != true_label:
            return min(fallback, negative_target)
        return fallback

    def _metadata_target(
        self,
        *,
        output: Dict[str, Any],
        true_label: str,
        final_label: str,
        fallback: float,
    ) -> float:
        if not output:
            return fallback
        supported = {
            self._norm_label(x)
            for x in (output.get("supported_diagnoses", []) or output.get("supported_labels", []) or [])
        }
        penalized = {
            self._norm_label(x)
            for x in (output.get("penalized_diagnoses", []) or output.get("penalized_labels", []) or [])
        }
        score = self._safe_float(output.get("score"), default=0.0)

        if true_label and true_label in supported:
            return max(fallback, min(1.0, 0.78 + 0.12 * max(0.0, score)))
        if true_label and true_label in penalized:
            return min(fallback, 0.05)
        if final_label and final_label in supported and true_label and final_label != true_label:
            return min(fallback, 0.15)
        if supported or penalized:
            return max(fallback, min(0.6, 0.3 + 0.1 * len(supported)))
        return fallback

    def _malignancy_target(
        self,
        *,
        output: Dict[str, Any],
        true_label: str,
        fallback: float,
    ) -> float:
        if not output:
            return fallback
        risk_level = self._norm_text(output.get("risk_level") or output.get("level"))
        preferred = self._norm_label(
            output.get("preferred_label")
            or output.get("recommendation")
            or output.get("supports")
        )
        malignant_truth = true_label in {"MEL", "BCC", "SCC"}
        if malignant_truth:
            target = fallback
            if risk_level == "high":
                target = max(target, 0.92)
            elif risk_level == "medium":
                target = max(target, 0.74)
            elif risk_level == "low":
                target = min(target, 0.2)
            if preferred == true_label:
                target = max(target, 0.95)
            return target
        if risk_level == "high":
            return min(fallback, 0.08)
        if risk_level == "medium":
            return min(fallback, 0.2)
        if risk_level == "low":
            return max(0.12, min(fallback, 0.3))
        return fallback

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _age_match_score(self, *, age: float, top_names: List[str]) -> float:
        if age < 0:
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

    def _norm_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    def _norm_label(self, value: Any) -> str:
        if value is None:
            return "UNKNOWN"
        text = str(value).strip().upper()
        return text if text else "UNKNOWN"

    def _site_matches(self, site: str, keywords: List[str]) -> bool:
        if not site:
            return False
        normalized = site.replace("-", " ").replace("/", " ").strip().lower()
        tokens = [token for token in normalized.split() if token]
        return any(keyword in tokens for keyword in keywords)
