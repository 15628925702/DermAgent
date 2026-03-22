from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState


class LearnableEvidenceCalibrator:
    """Learnable calibration for specialist and metadata evidence."""

    STRONG_BCC_TOKENS = (
        "classic bcc site",
        "bleeding/elevated lesion on classic bcc site",
        "fallback metadata pattern strongly supports bcc",
    )

    def __init__(
        self,
        *,
        learning_rate: float = 0.02,
        use_adam: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.use_adam = use_adam

        if use_adam:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.m: Dict[str, float] = {}
            self.v: Dict[str, float] = {}
            self.t = 0

        self.weights: Dict[str, float] = {
            "specialist_support_scale": 1.0,
            "specialist_repeat_decay": 0.35,
            "specialist_overlap_decay": 0.15,
            "specialist_malignant_cap": 1.0,
            "specialist_benign_cap": 1.15,
            "ack_proxy_gap_high_floor": 1.05,
            "ack_proxy_gap_low_floor": 0.92,
            "metadata_support_weight": 0.35,
            "metadata_penalty_weight": 0.25,
            "metadata_bcc_bonus": 0.20,
            "skill_correction_specialist": 0.20,
            "skill_correction_metadata": 0.10,
        }
        self.bounds: Dict[str, tuple[float, float]] = {
            "specialist_support_scale": (0.35, 1.8),
            "specialist_repeat_decay": (0.10, 0.70),
            "specialist_overlap_decay": (0.05, 0.40),
            "specialist_malignant_cap": (0.45, 1.45),
            "specialist_benign_cap": (0.55, 1.65),
            "ack_proxy_gap_high_floor": (0.60, 1.40),
            "ack_proxy_gap_low_floor": (0.45, 1.20),
            "metadata_support_weight": (0.08, 0.90),
            "metadata_penalty_weight": (0.08, 0.90),
            "metadata_bcc_bonus": (0.00, 0.45),
            "skill_correction_specialist": (0.05, 0.45),
            "skill_correction_metadata": (0.02, 0.28),
        }

    def get_weight(self, name: str, default: float) -> float:
        return float(self.weights.get(name, default))

    def update_from_case(self, state: CaseState) -> Dict[str, Any]:
        true_label = self._norm_label((state.case_info or {}).get("true_label"))
        pred_label = self._norm_label(
            (state.final_decision or {}).get("final_label")
            or (state.final_decision or {}).get("diagnosis")
        )
        if not true_label or not pred_label:
            return {"updated": False, "reason": "missing_labels"}

        feedback: Dict[str, Any] = {
            "updated": False,
            "true_label": true_label,
            "predicted_label": pred_label,
            "weight_updates": {},
        }

        skill_outputs = state.skill_outputs or {}
        metadata_output = skill_outputs.get("metadata_consistency_skill", {}) or {}
        rationale_text = " ".join(str(x) for x in (metadata_output.get("rationale", []) or [])).lower()
        supported = {self._norm_label(x) for x in (metadata_output.get("supported_labels") or metadata_output.get("supported_diagnoses") or [])}
        penalized = {self._norm_label(x) for x in (metadata_output.get("penalized_labels") or metadata_output.get("penalized_diagnoses") or [])}

        metadata_support_signal = 0.0
        metadata_penalty_signal = 0.0
        metadata_bonus_signal = 0.0

        if true_label in supported:
            metadata_support_signal += 1.0
        if pred_label != true_label and pred_label in supported:
            metadata_support_signal -= 1.0
        if pred_label != true_label and pred_label in penalized:
            metadata_penalty_signal += 1.0
        if true_label in penalized:
            metadata_penalty_signal -= 1.0
        if any(token in rationale_text for token in self.STRONG_BCC_TOKENS):
            if true_label == "BCC":
                metadata_bonus_signal += 1.0
            if pred_label == "BCC" and pred_label != true_label:
                metadata_bonus_signal -= 1.0

        specialist_votes: Dict[str, int] = {}
        specialist_signal = 0.0
        duplicate_signal = 0.0
        malignant_cap_signal = 0.0
        for skill_name, output in skill_outputs.items():
            if not str(skill_name).endswith("_specialist_skill"):
                continue
            recommendation = self._norm_label(
                output.get("supports")
                or output.get("supported_label")
                or output.get("recommendation")
                or output.get("winner")
                or output.get("final_choice")
            )
            if not recommendation:
                continue
            confidence = max(0.25, self._safe_float(output.get("confidence") or output.get("score") or 0.5))
            specialist_votes[recommendation] = specialist_votes.get(recommendation, 0) + 1
            if recommendation == true_label:
                specialist_signal += confidence
                if recommendation in {"MEL", "BCC", "SCC"}:
                    malignant_cap_signal += 0.5 * confidence
            elif recommendation == pred_label and pred_label != true_label:
                specialist_signal -= confidence
                if recommendation in {"MEL", "BCC", "SCC"}:
                    malignant_cap_signal -= 0.7 * confidence

            if skill_name == "ack_scc_specialist_skill" and bool(output.get("used_ack_proxy")):
                if recommendation == true_label:
                    self._update_weight("ack_proxy_gap_high_floor", 0.25 * confidence, feedback["weight_updates"])
                    self._update_weight("ack_proxy_gap_low_floor", 0.20 * confidence, feedback["weight_updates"])
                elif recommendation == pred_label and pred_label != true_label:
                    self._update_weight("ack_proxy_gap_high_floor", -0.25 * confidence, feedback["weight_updates"])
                    self._update_weight("ack_proxy_gap_low_floor", -0.20 * confidence, feedback["weight_updates"])

        for label, count in specialist_votes.items():
            if count < 2:
                continue
            if label == true_label:
                duplicate_signal += 0.6
            elif label == pred_label and pred_label != true_label:
                duplicate_signal -= 0.8

        self._update_weight("metadata_support_weight", metadata_support_signal, feedback["weight_updates"])
        self._update_weight("metadata_penalty_weight", metadata_penalty_signal, feedback["weight_updates"])
        self._update_weight("metadata_bcc_bonus", metadata_bonus_signal, feedback["weight_updates"])
        self._update_weight("skill_correction_metadata", 0.35 * metadata_support_signal + 0.35 * metadata_penalty_signal, feedback["weight_updates"])
        self._update_weight("specialist_support_scale", specialist_signal, feedback["weight_updates"])
        self._update_weight("specialist_repeat_decay", duplicate_signal, feedback["weight_updates"])
        self._update_weight("specialist_overlap_decay", -0.6 * duplicate_signal, feedback["weight_updates"])
        self._update_weight("specialist_malignant_cap", malignant_cap_signal, feedback["weight_updates"])
        self._update_weight("specialist_benign_cap", 0.5 * specialist_signal, feedback["weight_updates"])
        self._update_weight("skill_correction_specialist", 0.35 * specialist_signal + 0.25 * duplicate_signal, feedback["weight_updates"])

        feedback["updated"] = bool(feedback["weight_updates"])
        if not feedback["updated"]:
            feedback["reason"] = "no_calibration_signal"
        return feedback

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "learning_rate": self.learning_rate,
            "weights": {
                key: round(float(value), 6)
                for key, value in sorted(self.weights.items())
            },
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if int(payload.get("version", 0) or 0) != 1:
            return
        if payload.get("learning_rate") is not None:
            self.learning_rate = float(payload["learning_rate"])
        weights = payload.get("weights", {}) or {}
        for key, value in weights.items():
            if key in self.weights:
                self.weights[str(key)] = float(value)

    def _update_weight(self, name: str, signal: float, updates: Dict[str, float]) -> None:
        if abs(signal) <= 1e-8 or name not in self.weights:
            return
        gradient = self.learning_rate * float(signal)
        if self.use_adam:
            self._adam_update(name, gradient)
        else:
            self.weights[name] = float(self.weights[name]) + gradient
        low, high = self.bounds[name]
        self.weights[name] = min(high, max(low, float(self.weights[name])))
        updates[name] = round(float(self.weights[name]), 6)

    def _adam_update(self, param_name: str, gradient: float) -> None:
        self.t += 1
        if param_name not in self.m:
            self.m[param_name] = 0.0
            self.v[param_name] = 0.0

        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradient
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * gradient ** 2

        m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
        self.weights[param_name] = float(self.weights.get(param_name, 0.0)) + (
            self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)
        )

    def _norm_label(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().upper()

    def _safe_float(self, value: Any) -> float:
        try:
            if value is None or value == "":
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
