"""
证据聚合器。

职责：
- 汇总 perception / retrieval / compare / risk / specialist 输出
- 进行基础版候选打分聚合
- 生成 final_decision

基础版原则：
- perception 是初始证据
- compare / specialist 可以增强或覆盖候选倾向
- retrieval 提供经验支持
- malignancy risk 提供安全约束
- 输出结构化 evidence，方便 reporter / reflection 使用

相关文件：
- skills/reporter.py
- agent/reflection.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.state import CaseState


class DecisionAggregator:
    """
    基础版证据聚合器。

    聚合优先级思路：
    1. perception 提供初始候选分
    2. compare skill 对 winner 加权
    3. specialist recommendation 高权重
    4. retrieval support_labels 提供经验支持
    5. malignancy risk 不一定直接改标签，但可加 risk flag
    """

    def aggregate(self, state: CaseState) -> Dict[str, Any]:
        candidate_scores: Dict[str, float] = {}
        evidence_log: List[Dict[str, Any]] = []

        # 1. perception 初始分
        self._add_perception_evidence(state, candidate_scores, evidence_log)

        # 2. retrieval 支持分
        self._add_retrieval_evidence(state, candidate_scores, evidence_log)

        # 3. compare skill
        self._add_compare_evidence(state, candidate_scores, evidence_log)

        # 4. specialist skills
        self._add_specialist_evidence(state, candidate_scores, evidence_log)

        # 5. metadata consistency skill
        self._add_metadata_consistency_evidence(state, candidate_scores, evidence_log)

        # 6. malignancy risk signal（安全标记，不直接强推标签，除非输出有 preferred_label）
        risk_summary = self._build_risk_summary(state)
        self._add_malignancy_evidence(state, candidate_scores, evidence_log)

        # 排序
        ranked_candidates = self._rank_candidates(candidate_scores)
        final_label = ranked_candidates[0][0] if ranked_candidates else "UNKNOWN"

        confidence = self._estimate_final_confidence(
            ranked_candidates=ranked_candidates,
            state=state,
        )

        result: Dict[str, Any] = {
            "diagnosis": final_label,
            "final_label": final_label,  # 兼容旧接口
            "top_k": [
                {"name": name, "score": round(score, 4)}
                for name, score in ranked_candidates[:5]
            ],
            "confidence": confidence,
            "risk_summary": risk_summary,
            "evidence_summary": self._build_evidence_summary(
                state=state,
                ranked_candidates=ranked_candidates,
                evidence_log=evidence_log,
            ),
            "aggregator_debug": {
                "candidate_scores": {
                    name: round(score, 4) for name, score in ranked_candidates
                },
                "num_evidence_items": len(evidence_log),
            },
        }

        state.final_decision = result
        state.trace(
            "aggregator",
            "success",
            f"Final diagnosis: {final_label}",
            payload={
                "final_label": final_label,
                "confidence": confidence,
                "top_candidates": result["top_k"][:3],
            },
        )
        return result

    # =========================
    # Evidence adders
    # =========================

    def _add_perception_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        ddx = state.perception.get("ddx_candidates", []) or []

        if ddx:
            for rank, item in enumerate(ddx[:5]):
                name = self._norm_label(item.get("name"))
                if not name:
                    continue

                raw_score = self._extract_candidate_score(item)
                if raw_score is None:
                    # 没显式分数时，按 rank 给默认分
                    base_score = max(0.1, 1.0 - 0.15 * rank)
                else:
                    base_score = float(raw_score)

                self._boost(candidate_scores, name, base_score)
                evidence_log.append(
                    {
                        "source": "perception",
                        "label": name,
                        "weight": round(base_score, 4),
                        "detail": f"ddx_rank={rank + 1}",
                    }
                )
            return

        most_likely = (state.perception.get("most_likely", {}) or {}).get("name")
        name = self._norm_label(most_likely)
        if name:
            self._boost(candidate_scores, name, 1.0)
            evidence_log.append(
                {
                    "source": "perception",
                    "label": name,
                    "weight": 1.0,
                    "detail": "fallback_most_likely",
                }
            )

    def _add_retrieval_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = retrieval_summary.get("support_labels", []) or []
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()

        confidence_weight = {
            "high": 0.9,
            "medium": 0.5,
            "low": 0.2,
        }.get(retrieval_confidence, 0.2)

        for idx, label in enumerate(support_labels[:5]):
            name = self._norm_label(label)
            if not name:
                continue

            # 越靠前支持越强
            w = max(0.1, confidence_weight - 0.1 * idx)
            self._boost(candidate_scores, name, w)
            evidence_log.append(
                {
                    "source": "retrieval",
                    "label": name,
                    "weight": round(w, 4),
                    "detail": f"retrieval_confidence={retrieval_confidence}",
                }
            )

    def _add_compare_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        compare_output = state.skill_outputs.get("compare_skill", {}) or {}
        winner = self._norm_label(
            compare_output.get("winner")
            or compare_output.get("recommendation")
            or compare_output.get("final_choice")
        )
        if not winner:
            return

        weight = self._extract_skill_weight(compare_output, default=0.8)
        self._boost(candidate_scores, winner, weight)

        evidence_log.append(
            {
                "source": "compare_skill",
                "label": winner,
                "weight": round(weight, 4),
                "detail": "winner_from_compare_skill",
            }
        )

    def _add_specialist_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        specialist_keys = [
            "mel_nev_specialist_skill",
            "ack_scc_specialist_skill",
        ]

        for key in specialist_keys:
            output = state.skill_outputs.get(key, {}) or {}
            recommendation = self._norm_label(
                output.get("recommendation")
                or output.get("winner")
                or output.get("final_choice")
            )
            if not recommendation:
                continue

            weight = self._extract_skill_weight(output, default=1.2)
            if (
                key == "ack_scc_specialist_skill"
                and recommendation == "ACK"
                and bool(output.get("used_ack_proxy"))
            ):
                gap = self._safe_float(output.get("gap"))
                weight = max(weight, 1.05 if gap >= 0.6 else 0.92)
            self._boost(candidate_scores, recommendation, weight)

            evidence_log.append(
                {
                    "source": key,
                    "label": recommendation,
                    "weight": round(weight, 4),
                    "detail": "specialist_recommendation",
                }
            )

    def _add_metadata_consistency_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        output = state.skill_outputs.get("metadata_consistency_skill", {}) or {}

        supported = output.get("supported_diagnoses", []) or output.get("supported_labels", []) or []
        penalized = output.get("penalized_diagnoses", []) or output.get("penalized_labels", []) or []
        rationale_text = " ".join(str(x) for x in (output.get("rationale", []) or [])).lower()

        for label in supported[:5]:
            name = self._norm_label(label)
            if not name:
                continue
            weight = 0.35
            if name == "BCC" and any(
                token in rationale_text
                for token in [
                    "classic bcc site",
                    "bleeding/elevated lesion on classic bcc site",
                    "fallback metadata pattern strongly supports bcc",
                ]
            ):
                weight = 0.55
            self._boost(candidate_scores, name, weight)
            evidence_log.append(
                {
                    "source": "metadata_consistency_skill",
                    "label": name,
                    "weight": round(weight, 4),
                    "detail": "metadata_supported",
                }
            )

        for label in penalized[:5]:
            name = self._norm_label(label)
            if not name:
                continue
            self._boost(candidate_scores, name, -0.25)
            evidence_log.append(
                {
                    "source": "metadata_consistency_skill",
                    "label": name,
                    "weight": -0.25,
                    "detail": "metadata_penalized",
                }
            )

    def _add_malignancy_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        output = state.skill_outputs.get("malignancy_risk_skill", {}) or {}

        preferred = self._norm_label(
            output.get("preferred_label")
            or output.get("recommended_label")
        )
        if preferred:
            weight = 0.5
            if bool(output.get("ack_scc_ambiguous", False)):
                weight = 0.12

            self._boost(candidate_scores, preferred, weight)
            evidence_log.append(
                {
                    "source": "malignancy_risk_skill",
                    "label": preferred,
                    "weight": round(weight, 4),
                    "detail": "preferred_label_from_risk_skill",
                }
            )

    # =========================
    # Risk / summary / confidence
    # =========================

    def _build_risk_summary(self, state: CaseState) -> Dict[str, Any]:
        output = state.skill_outputs.get("malignancy_risk_skill", {}) or {}

        risk_level = str(output.get("risk_level", "unknown")).lower()
        suspicious = bool(output.get("suspicious_malignancy", False))
        rationale = output.get("rationale", "") or output.get("reason", "")

        return {
            "risk_level": risk_level,
            "suspicious_malignancy": suspicious,
            "rationale": rationale,
        }

    def _build_evidence_summary(
        self,
        state: CaseState,
        ranked_candidates: List[Tuple[str, float]],
        evidence_log: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        top3 = [{"name": x[0], "score": round(x[1], 4)} for x in ranked_candidates[:3]]

        used_sources: List[str] = []
        for item in evidence_log:
            source = str(item.get("source", "")).strip()
            if source and source not in used_sources:
                used_sources.append(source)

        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}

        return {
            "top_candidates": top3,
            "used_sources": used_sources,
            "selected_skills": list(state.selected_skills),
            "retrieval_support_labels": retrieval_summary.get("support_labels", []),
            "supports_top1": retrieval_summary.get("supports_top1", False),
            "evidence_log": evidence_log[:20],
        }

    def _estimate_final_confidence(
        self,
        ranked_candidates: List[Tuple[str, float]],
        state: CaseState,
    ) -> str:
        if not ranked_candidates:
            return "low"

        planner_flags = state.planner.get("flags", {}) or {}
        uncertainty = str(planner_flags.get("uncertainty", state.get_uncertainty_level())).lower()
        retrieval_confidence = str(planner_flags.get("retrieval_confidence", "low")).lower()
        supports_top1 = bool(planner_flags.get("supports_top1", False))

        top1 = ranked_candidates[0][1]
        top2 = ranked_candidates[1][1] if len(ranked_candidates) >= 2 else 0.0
        gap = top1 - top2

        if uncertainty == "low" and retrieval_confidence == "high" and supports_top1 and gap >= 0.8:
            return "high"

        if gap >= 0.35 or retrieval_confidence in {"medium", "high"}:
            return "medium"

        return "low"

    # =========================
    # Helpers
    # =========================

    def _rank_candidates(self, candidate_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        ranked = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(name, score) for name, score in ranked if name]

    def _boost(self, candidate_scores: Dict[str, float], label: str, delta: float) -> None:
        if not label:
            return
        candidate_scores[label] = candidate_scores.get(label, 0.0) + float(delta)

    def _norm_label(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().upper()

    def _extract_candidate_score(self, item: Dict[str, Any]) -> float | None:
        for key in ["score", "probability", "confidence"]:
            value = item.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _extract_skill_weight(self, output: Dict[str, Any], default: float) -> float:
        for key in ["score", "confidence", "weight"]:
            value = output.get(key)
            try:
                if value is not None:
                    return max(0.1, float(value))
            except (TypeError, ValueError):
                continue
        return default

    def _safe_float(self, value: Any) -> float:
        try:
            if value is None or value == "":
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
