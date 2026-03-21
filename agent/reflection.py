"""
Reflection 模块。

职责：
- 诊断结束后总结本病例经验
- 形成可写回 experience bank 的中间对象
- 给 writer 提供结构化反思结果

基础版目标：
- 总结最终诊断来源
- 标记 confusion / risk / weak-support 等标签
- 为后续 prototype / confusion 写回提供辅助字段

增强版补充：
- learning_signals：标记 hard case / confusion case / needs_more_experience
- confusion_info：结构化混淆信息
- prototype_features：为 prototype memory 提供可直接写回的特征对象

相关文件：
- memory/schema.py
- memory/writer.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.state import CaseState


KNOWN_CONFUSION_PAIRS = {
    frozenset({"ACK", "SCC"}),
    frozenset({"MEL", "NEV"}),
    frozenset({"BCC", "SEK"}),
    frozenset({"BCC", "SCC"}),
}

MAX_CONFUSION_GAP = 0.15
MAX_HARD_CASE_GAP = 0.2


class ReflectionEngine:
    """
    基础版 reflection。

    不做复杂“自进化设计”，只做结构化总结：
    - final diagnosis
    - key evidence sources
    - confusion tags
    - risk tags
    - retrieval support status

    增强版新增：
    - learning signals
    - confusion info
    - prototype features
    """

    def summarize(self, state: CaseState) -> Dict[str, Any]:
        case_id = state.get_case_id()
        final_label = self._get_final_label(state)
        top_ddx = state.get_top_ddx_names(top_k=3)
        uncertainty = state.get_uncertainty_level()

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        planner_flags = state.planner.get("flags", {}) or {}
        evidence_summary = state.final_decision.get("evidence_summary", {}) or {}
        risk_summary = state.final_decision.get("risk_summary", {}) or {}

        confusion_tag = self._build_confusion_tag(state, final_label)
        confusion_info = self._build_confusion_info(state, final_label)
        risk_tags = self._build_risk_tags(state, final_label)
        support_status = self._build_support_status(state)

        decisive_factors = self._build_decisive_factors(
            state=state,
            final_label=final_label,
        )

        learning_signals = self._build_learning_signals(state)
        prototype_features = self._build_prototype_features(state, final_label)

        result: Dict[str, Any] = {
            "case_id": case_id,
            "summary": self._build_summary_text(
                case_id=case_id,
                final_label=final_label,
                top_ddx=top_ddx,
                uncertainty=uncertainty,
                support_status=support_status,
            ),
            "final_label": final_label,
            "selected_skills": list(state.selected_skills),
            "top_ddx": top_ddx,
            "uncertainty": uncertainty,
            "confusion_tag": confusion_tag,
            "confusion_info": confusion_info,
            "risk_tags": risk_tags,
            "support_status": support_status,
            "decisive_factors": decisive_factors,
            "learning_signals": learning_signals,
            "prototype_features": prototype_features,
            "retrieval_observation": {
                "support_labels": retrieval_summary.get("support_labels", []),
                "retrieval_confidence": retrieval_summary.get("retrieval_confidence", "low"),
                "supports_top1": retrieval_summary.get("supports_top1", False),
                "has_confusion_support": retrieval_summary.get("has_confusion_support", False),
            },
            "planner_observation": {
                "selected_skills": list(state.selected_skills),
                "routing_flags": planner_flags,
            },
            "decision_observation": {
                "confidence": state.final_decision.get("confidence", "low"),
                "top_k": state.final_decision.get("top_k", []),
                "used_sources": evidence_summary.get("used_sources", []),
            },
            "risk_observation": risk_summary,
            "writeback_hints": {
                "should_write_prototype": self._should_write_prototype(state),
                "should_write_confusion": self._should_write_confusion(state),
                "should_write_hard_case": learning_signals.get("hard_case", False),
            },
        }

        state.reflection = result
        state.trace(
            "reflection",
            "success",
            "Reflection completed",
            payload={
                "final_label": final_label,
                "confusion_tag": confusion_tag,
                "risk_tags": risk_tags,
                "support_status": support_status,
                "hard_case": learning_signals.get("hard_case", False),
                "needs_more_experience": learning_signals.get("needs_more_experience", False),
            },
        )
        return result

    # =========================
    # Core builders
    # =========================

    def _build_summary_text(
        self,
        case_id: str,
        final_label: str,
        top_ddx: List[str],
        uncertainty: str,
        support_status: str,
    ) -> str:
        ddx_text = ", ".join(top_ddx) if top_ddx else "N/A"
        return (
            f"Case {case_id} finalized as {final_label}. "
            f"Top differential candidates were [{ddx_text}], "
            f"uncertainty={uncertainty}, support_status={support_status}."
        )

    def _build_confusion_tag(self, state: CaseState, final_label: str) -> Optional[str]:
        ddx = state.perception.get("ddx_candidates", []) or []
        if len(ddx) < 2:
            return None

        d1 = str(ddx[0].get("name", "")).strip().upper()
        d2 = str(ddx[1].get("name", "")).strip().upper()
        if not d1 or not d2 or d1 == d2:
            return None

        uncertainty = state.get_uncertainty_level()
        selected = set(state.selected_skills)
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}

        is_confusion_case = (
            uncertainty in {"medium", "high"}
            or "compare_skill" in selected
            or "mel_nev_specialist_skill" in selected
            or "ack_scc_specialist_skill" in selected
            or retrieval_summary.get("has_confusion_support", False)
        )
        if not is_confusion_case:
            return None

        pair = sorted([d1, d2])
        return f"{pair[0]}_VS_{pair[1]}"

    def _build_confusion_info(
        self,
        state: CaseState,
        final_label: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._should_write_confusion(state):
            return None

        ddx = state.perception.get("ddx_candidates", []) or []
        if len(ddx) < 2:
            return None

        d1 = str(ddx[0].get("name", "")).strip().upper()
        d2 = str(ddx[1].get("name", "")).strip().upper()

        if not d1 or not d2 or d1 == d2:
            return None

        s1 = self._extract_score(ddx[0])
        s2 = self._extract_score(ddx[1])
        gap = abs(s1 - s2) if s1 is not None and s2 is not None else None

        return {
            "pair": sorted([d1, d2]),
            "final_label": final_label,
            "top2_gap": round(gap, 4) if gap is not None else None,
            "selected_skills": list(state.selected_skills),
            "retrieval_has_confusion_support": bool(
                (state.retrieval.get("retrieval_summary", {}) or {}).get("has_confusion_support", False)
            ),
        }

    def _build_risk_tags(self, state: CaseState, final_label: str) -> List[str]:
        tags: List[str] = []
        final_label = str(final_label).strip().upper()

        if final_label in {"MEL", "BCC", "SCC"}:
            tags.append("final_malignant_or_high_risk")

        risk_summary = state.final_decision.get("risk_summary", {}) or {}
        risk_level = str(risk_summary.get("risk_level", "")).lower()
        suspicious = bool(risk_summary.get("suspicious_malignancy", False))

        if risk_level == "high":
            tags.append("high_malignancy_risk")
        elif risk_level == "medium":
            tags.append("medium_malignancy_risk")

        if suspicious:
            tags.append("suspicious_malignancy_signal")

        if state.get_uncertainty_level() == "high":
            tags.append("high_uncertainty")

        return tags

    def _build_support_status(self, state: CaseState) -> str:
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()
        supports_top1 = bool(retrieval_summary.get("supports_top1", False))

        if retrieval_confidence == "high" and supports_top1:
            return "strong_support"

        if retrieval_confidence in {"medium", "high"}:
            return "partial_support"

        return "weak_support"

    def _build_decisive_factors(
        self,
        state: CaseState,
        final_label: str,
    ) -> List[str]:
        factors: List[str] = []

        evidence_summary = state.final_decision.get("evidence_summary", {}) or {}
        used_sources = evidence_summary.get("used_sources", []) or []
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        final_label = str(final_label).strip().upper()

        if "perception" in used_sources:
            factors.append("perception_initial_support")

        if retrieval_summary.get("supports_top1", False):
            factors.append("retrieval_supports_final")

        if "compare_skill" in state.skill_outputs:
            compare_out = state.skill_outputs.get("compare_skill", {}) or {}
            compare_winner = str(
                compare_out.get("winner")
                or compare_out.get("recommendation")
                or compare_out.get("final_choice", "")
            ).strip().upper()
            if compare_winner == final_label:
                factors.append("compare_skill_supports_final")

        for specialist_key in ["mel_nev_specialist_skill", "ack_scc_specialist_skill"]:
            if specialist_key in state.skill_outputs:
                specialist_out = state.skill_outputs.get(specialist_key, {}) or {}
                specialist_label = str(
                    specialist_out.get("recommendation")
                    or specialist_out.get("winner")
                    or specialist_out.get("final_choice", "")
                ).strip().upper()
                if specialist_label == final_label:
                    factors.append(f"{specialist_key}_supports_final")

        if "metadata_consistency_skill" in state.skill_outputs:
            metadata_out = state.skill_outputs.get("metadata_consistency_skill", {}) or {}
            supported = metadata_out.get("supported_diagnoses", []) or metadata_out.get("supported_labels", []) or []
            supported = [str(x).strip().upper() for x in supported]
            if final_label in supported:
                factors.append("metadata_consistency_supports_final")

        if "malignancy_risk_skill" in state.skill_outputs:
            risk_out = state.skill_outputs.get("malignancy_risk_skill", {}) or {}
            preferred = str(
                risk_out.get("preferred_label")
                or risk_out.get("recommended_label", "")
            ).strip().upper()
            if preferred == final_label:
                factors.append("malignancy_risk_supports_final")

        if not factors:
            factors.append("weak_explicit_support")

        return factors

    def _build_learning_signals(self, state: CaseState) -> Dict[str, Any]:
        confidence = str(state.final_decision.get("confidence", "low")).lower()
        uncertainty = state.get_uncertainty_level()
        fallback_case = bool((state.perception or {}).get("fallback_reason"))
        is_top1_correct = self._is_top1_correct(state)
        true_in_top3 = self._is_true_in_top3(state)
        confusion_info = self._extract_top2_confusion_info(state)
        top2_gap = confusion_info.get("gap")
        known_confusion_pair = bool(confusion_info.get("is_known_pair", False))

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        supports_top1 = bool(retrieval_summary.get("supports_top1", False))
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()
        risk_summary = state.final_decision.get("risk_summary", {}) or {}
        risk_level = str(risk_summary.get("risk_level", "")).lower()
        suspicious_malignancy = bool(risk_summary.get("suspicious_malignancy", False))
        final_label = self._get_final_label(state)
        malignant_mismatch = (
            suspicious_malignancy
            and risk_level in {"medium", "high"}
            and final_label not in {"MEL", "BCC", "SCC"}
        )

        hard_case = (
            not fallback_case
            and not is_top1_correct
            and true_in_top3
            and (
                malignant_mismatch
                or (
                    known_confusion_pair
                    and top2_gap is not None
                    and top2_gap <= MAX_HARD_CASE_GAP
                )
                or (
                    top2_gap is not None
                    and top2_gap <= 0.12
                    and uncertainty in {"medium", "high"}
                )
            )
        )

        confusion_case = self._should_write_confusion(state)

        needs_more_experience = (
            hard_case
            or confusion_case
            or (
                not fallback_case
                and not is_top1_correct
                and true_in_top3
                and retrieval_confidence == "low"
                and confidence == "low"
            )
        )

        return {
            "hard_case": hard_case,
            "confusion_case": confusion_case,
            "needs_more_experience": needs_more_experience,
            "low_retrieval_support": retrieval_confidence == "low",
            "fallback_case": fallback_case,
            "malignant_mismatch": malignant_mismatch,
            "top1_correct": is_top1_correct,
            "true_in_top3": true_in_top3,
            "known_confusion_pair": known_confusion_pair,
            "top2_gap": round(top2_gap, 4) if top2_gap is not None else None,
        }

    def _build_prototype_features(
        self,
        state: CaseState,
        final_label: str,
    ) -> Optional[Dict[str, Any]]:
        if not self._should_write_prototype(state):
            return None

        visual_cues = state.perception.get("visual_cues", []) or []
        metadata = state.get_metadata()
        decisive = self._build_decisive_factors(state, final_label)

        return {
            "label": final_label,
            "visual_cues": visual_cues[:10],
            "decisive_factors": decisive,
            "age": metadata.get("age"),
            "site": (
                metadata.get("location")
                or metadata.get("site")
                or metadata.get("anatomical_site")
            ),
        }

    # =========================
    # Writeback hints
    # =========================

    def _should_write_prototype(self, state: CaseState) -> bool:
        """
        更适合作为 prototype 的病例：
        - final confidence 不是 low
        - uncertainty 不是 high
        - 有一定 visual cues
        """
        confidence = str(state.final_decision.get("confidence", "low")).lower()
        uncertainty = state.get_uncertainty_level()
        visual_cues = state.perception.get("visual_cues", []) or []
        if (state.perception or {}).get("fallback_reason"):
            return False

        return (
            self._is_top1_correct(state)
            confidence in {"medium", "high"}
            and uncertainty in {"low", "medium"}
            and (len(visual_cues) > 0 or len(self._build_decisive_factors(state, self._get_final_label(state))) >= 2)
        )

    def _should_write_confusion(self, state: CaseState) -> bool:
        """
        更适合作为 confusion experience 的病例：
        - top2 ddx 存在
        - 有 compare 或 specialist 或 confusion support
        """
        ddx = state.perception.get("ddx_candidates", []) or []
        if len(ddx) < 2:
            return False
        if (state.perception or {}).get("fallback_reason"):
            return False

        confusion_info = self._extract_top2_confusion_info(state)
        if not confusion_info.get("is_known_pair", False):
            return False
        gap = confusion_info.get("gap")
        if gap is not None and gap > MAX_CONFUSION_GAP:
            return False

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        selected = set(state.selected_skills)
        uncertainty = state.get_uncertainty_level()

        return (
            self._is_top1_correct(state)
            retrieval_summary.get("has_confusion_support", False)
            or (
                uncertainty in {"medium", "high"}
                and (
                    "compare_skill" in selected
                    or "mel_nev_specialist_skill" in selected
                    or "ack_scc_specialist_skill" in selected
                    or "bcc_scc_specialist_skill" in selected
                    or "bcc_sek_specialist_skill" in selected
                )
            )
        )

    # =========================
    # Helpers
    # =========================

    def _get_final_label(self, state: CaseState) -> str:
        for key in ["diagnosis", "final_label"]:
            value = str(state.final_decision.get(key, "")).strip().upper()
            if value:
                return value
        return "UNKNOWN"

    def _get_true_label(self, state: CaseState) -> str:
        return str((state.case_info or {}).get("true_label", "")).strip().upper()

    def _is_top1_correct(self, state: CaseState) -> bool:
        true_label = self._get_true_label(state)
        final_label = self._get_final_label(state)
        return bool(true_label) and true_label == final_label

    def _is_true_in_top3(self, state: CaseState) -> bool:
        true_label = self._get_true_label(state)
        if not true_label:
            return False

        labels: List[str] = []
        for item in (state.final_decision.get("top_k", []) or [])[:3]:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip().upper()
            else:
                name = str(item).strip().upper()
            if name:
                labels.append(name)
        return true_label in labels

    def _extract_top2_confusion_info(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", []) or []
        if len(ddx) < 2:
            return {"pair": [], "gap": None, "is_known_pair": False}

        first = ddx[0] if isinstance(ddx[0], dict) else {}
        second = ddx[1] if isinstance(ddx[1], dict) else {}
        d1 = str(first.get("name", "")).strip().upper()
        d2 = str(second.get("name", "")).strip().upper()
        if not d1 or not d2 or d1 == d2:
            return {"pair": [], "gap": None, "is_known_pair": False}

        s1 = self._extract_score(first)
        s2 = self._extract_score(second)
        gap = abs(s1 - s2) if s1 is not None and s2 is not None else None
        pair = sorted([d1, d2])
        return {
            "pair": pair,
            "gap": gap,
            "is_known_pair": frozenset(pair) in KNOWN_CONFUSION_PAIRS,
        }

    def _extract_score(self, item: Dict[str, Any]) -> Optional[float]:
        for key in ["score", "probability", "confidence"]:
            try:
                if item.get(key) is not None:
                    return float(item[key])
            except (TypeError, ValueError):
                continue
        return None
