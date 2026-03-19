from __future__ import annotations

from typing import Any, Dict, List, Tuple

from agent.final_scorer import LearnableFinalScorer
from agent.state import CaseState


class DecisionAggregator:
    def __init__(self, final_scorer: LearnableFinalScorer | None = None) -> None:
        self.final_scorer = final_scorer

    def aggregate(self, state: CaseState) -> Dict[str, Any]:
        candidate_scores: Dict[str, float] = {}
        candidate_features: Dict[str, Dict[str, float]] = {}
        evidence_log: List[Dict[str, Any]] = []

        self._add_perception_evidence(state, candidate_scores, candidate_features, evidence_log)
        self._add_retrieval_evidence(state, candidate_scores, candidate_features, evidence_log)
        self._add_compare_evidence(state, candidate_scores, candidate_features, evidence_log)
        self._add_specialist_evidence(state, candidate_scores, candidate_features, evidence_log)
        self._add_metadata_consistency_evidence(state, candidate_scores, candidate_features, evidence_log)
        risk_summary = self._build_risk_summary(state)
        self._add_malignancy_evidence(state, candidate_scores, candidate_features, evidence_log)
        self._add_memory_consensus_evidence(state, candidate_scores, candidate_features, evidence_log)
        self._finalize_candidate_features(state, candidate_scores, candidate_features)

        if self.final_scorer is not None and candidate_features:
            ranked_candidates, feature_debug = self.final_scorer.rank_candidates(candidate_features)
            ranked_candidates = [(label, round(score, 6)) for label, score in ranked_candidates]
        else:
            ranked_candidates = self._rank_candidates(candidate_scores)
            feature_debug = {
                label: {
                    key: round(float(value), 4)
                    for key, value in sorted(features.items())
                    if abs(float(value)) > 1e-8
                }
                for label, features in candidate_features.items()
            }

        final_label = ranked_candidates[0][0] if ranked_candidates else "UNKNOWN"
        confidence = self._estimate_final_confidence(ranked_candidates=ranked_candidates, state=state)

        result: Dict[str, Any] = {
            "diagnosis": final_label,
            "final_label": final_label,
            "top_k": [{"name": name, "score": round(score, 4)} for name, score in ranked_candidates[:5]],
            "confidence": confidence,
            "risk_summary": risk_summary,
            "evidence_summary": self._build_evidence_summary(
                state=state,
                ranked_candidates=ranked_candidates,
                evidence_log=evidence_log,
            ),
            "aggregator_debug": {
                "candidate_scores": {name: round(score, 4) for name, score in ranked_candidates},
                "candidate_base_scores": {
                    name: round(score, 4) for name, score in self._rank_candidates(candidate_scores)
                },
                "candidate_features": feature_debug,
                "num_evidence_items": len(evidence_log),
                "memory_consensus_label": (state.retrieval.get("retrieval_summary", {}) or {}).get("memory_consensus_label", ""),
                "uses_learnable_final_scorer": self.final_scorer is not None,
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

    def _add_perception_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        ddx = state.perception.get("ddx_candidates", []) or []

        if ddx:
            for rank, item in enumerate(ddx[:5]):
                name = self._norm_label(item.get("name"))
                if not name:
                    continue
                raw_score = self._extract_candidate_score(item)
                base_score = max(0.1, 1.0 - 0.15 * rank) if raw_score is None else float(raw_score)
                self._apply_feature(candidate_scores, candidate_features, name, "perception_score", base_score)
                evidence_log.append({
                    "source": "perception",
                    "label": name,
                    "weight": round(base_score, 4),
                    "detail": f"ddx_rank={rank + 1}",
                })
            return

        most_likely = (state.perception.get("most_likely", {}) or {}).get("name")
        name = self._norm_label(most_likely)
        if name:
            self._apply_feature(candidate_scores, candidate_features, name, "perception_score", 1.0)
            evidence_log.append({
                "source": "perception",
                "label": name,
                "weight": 1.0,
                "detail": "fallback_most_likely",
            })

    def _add_retrieval_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = retrieval_summary.get("support_labels", []) or []
        prototype_votes = retrieval_summary.get("prototype_votes", {}) or {}
        confusion_votes = retrieval_summary.get("confusion_votes", {}) or {}
        retrieval_confidence = str(retrieval_summary.get("retrieval_confidence", "low")).lower()

        confidence_weight = {"high": 0.95, "medium": 0.55, "low": 0.2}.get(retrieval_confidence, 0.2)

        for idx, label in enumerate(support_labels[:5]):
            name = self._norm_label(label)
            if not name:
                continue
            weight = max(0.1, confidence_weight - 0.08 * idx)
            self._apply_feature(candidate_scores, candidate_features, name, "retrieval_score", weight)
            evidence_log.append({
                "source": "retrieval",
                "label": name,
                "weight": round(weight, 4),
                "detail": f"support_labels confidence={retrieval_confidence}",
            })

        for idx, (label, vote) in enumerate(list(prototype_votes.items())[:5]):
            name = self._norm_label(label)
            if not name:
                continue
            weight = min(1.1, 0.18 * float(vote))
            if idx == 0:
                weight += 0.08
            self._apply_feature(candidate_scores, candidate_features, name, "prototype_score", weight)
            evidence_log.append({
                "source": "prototype_memory",
                "label": name,
                "weight": round(weight, 4),
                "detail": f"prototype_vote={round(float(vote), 4)}",
            })

        for idx, (label, vote) in enumerate(list(confusion_votes.items())[:5]):
            name = self._norm_label(label)
            if not name:
                continue
            weight = min(0.9, 0.22 * float(vote))
            if idx == 0:
                weight += 0.05
            self._apply_feature(candidate_scores, candidate_features, name, "confusion_score", weight)
            evidence_log.append({
                "source": "confusion_memory",
                "label": name,
                "weight": round(weight, 4),
                "detail": f"confusion_vote={round(float(vote), 4)}",
            })

    def _add_compare_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
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
        self._apply_feature(candidate_scores, candidate_features, winner, "compare_score", weight)
        evidence_log.append({
            "source": "compare_skill",
            "label": winner,
            "weight": round(weight, 4),
            "detail": "winner_from_compare_skill",
        })

    def _add_specialist_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        for key, output in (state.skill_outputs or {}).items():
            if not str(key).endswith("_specialist_skill"):
                continue
            output = output or {}
            recommendation = self._norm_label(
                output.get("recommendation")
                or output.get("winner")
                or output.get("final_choice")
            )
            if not recommendation:
                continue

            weight = self._extract_skill_weight(output, default=1.2)
            if key == "ack_scc_specialist_skill" and recommendation == "ACK" and bool(output.get("used_ack_proxy")):
                gap = self._safe_float(output.get("gap"))
                weight = max(weight, 1.05 if gap >= 0.6 else 0.92)
            self._apply_feature(candidate_scores, candidate_features, recommendation, "specialist_score", weight)
            evidence_log.append({
                "source": key,
                "label": recommendation,
                "weight": round(weight, 4),
                "detail": "specialist_recommendation",
            })

    def _add_metadata_consistency_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
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
            self._apply_feature(candidate_scores, candidate_features, name, "metadata_score", weight)
            evidence_log.append({
                "source": "metadata_consistency_skill",
                "label": name,
                "weight": round(weight, 4),
                "detail": "metadata_supported",
            })

        for label in penalized[:5]:
            name = self._norm_label(label)
            if not name:
                continue
            self._apply_feature(candidate_scores, candidate_features, name, "metadata_score", -0.25)
            evidence_log.append({
                "source": "metadata_consistency_skill",
                "label": name,
                "weight": -0.25,
                "detail": "metadata_penalized",
            })

    def _add_malignancy_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        output = state.skill_outputs.get("malignancy_risk_skill", {}) or {}
        preferred = self._norm_label(output.get("preferred_label") or output.get("recommended_label"))
        if preferred:
            weight = 0.5
            if bool(output.get("ack_scc_ambiguous", False)):
                weight = 0.12
            self._apply_feature(candidate_scores, candidate_features, preferred, "malignancy_score", weight)
            evidence_log.append({
                "source": "malignancy_risk_skill",
                "label": preferred,
                "weight": round(weight, 4),
                "detail": "preferred_label_from_risk_skill",
            })

    def _add_memory_consensus_evidence(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
        evidence_log: List[Dict[str, Any]],
    ) -> None:
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        label = self._norm_label(retrieval_summary.get("memory_consensus_label"))
        confidence = str(retrieval_summary.get("retrieval_confidence", "low")).lower()
        if not label:
            return
        weight = {"high": 0.8, "medium": 0.45, "low": 0.18}.get(confidence, 0.18)
        self._apply_feature(candidate_scores, candidate_features, label, "memory_consensus_score", weight)
        evidence_log.append({
            "source": "memory_consensus",
            "label": label,
            "weight": round(weight, 4),
            "detail": f"memory_consensus retrieval_confidence={confidence}",
        })

    def _finalize_candidate_features(
        self,
        state: CaseState,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
    ) -> None:
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = {self._norm_label(x) for x in retrieval_summary.get("support_labels", []) or []}
        memory_consensus_label = self._norm_label(retrieval_summary.get("memory_consensus_label"))
        retrieval_confidence = str(retrieval_summary.get("retrieval_confidence", "low")).lower()
        perception_top1 = state.get_top_ddx_names(top_k=1)
        perception_top1 = perception_top1[0] if perception_top1 else ""

        for label, features in candidate_features.items():
            features["bias"] = 1.0
            features["base_total"] = round(float(candidate_scores.get(label, 0.0)), 6)
            features["is_perception_top1"] = 1.0 if label == perception_top1 else 0.0
            features["in_support_labels"] = 1.0 if label in support_labels else 0.0
            features["matches_memory_consensus"] = 1.0 if memory_consensus_label and label == memory_consensus_label else 0.0
            features["malignant_candidate"] = 1.0 if label in {"MEL", "BCC", "SCC"} else 0.0
            features["retrieval_confidence_high"] = 1.0 if retrieval_confidence == "high" else 0.0
            features["retrieval_confidence_medium"] = 1.0 if retrieval_confidence == "medium" else 0.0
            features["retrieval_confidence_low"] = 1.0 if retrieval_confidence == "low" else 0.0

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
            "prototype_votes": retrieval_summary.get("prototype_votes", {}),
            "confusion_votes": retrieval_summary.get("confusion_votes", {}),
            "memory_consensus_label": retrieval_summary.get("memory_consensus_label", ""),
            "supports_top1": retrieval_summary.get("supports_top1", False),
            "evidence_log": evidence_log[:25],
        }

    def _estimate_final_confidence(
        self,
        ranked_candidates: List[Tuple[str, float]],
        state: CaseState,
    ) -> str:
        if not ranked_candidates:
            return "low"

        planner_flags = state.planner.get("flags", {}) or {}
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        uncertainty = str(planner_flags.get("uncertainty", state.get_uncertainty_level())).lower()
        retrieval_confidence = str(planner_flags.get("retrieval_confidence", retrieval_summary.get("retrieval_confidence", "low"))).lower()
        supports_top1 = bool(planner_flags.get("supports_top1", retrieval_summary.get("supports_top1", False)))
        has_memory_consensus = bool(retrieval_summary.get("memory_consensus_label"))

        top1 = float(ranked_candidates[0][1])
        top2 = float(ranked_candidates[1][1]) if len(ranked_candidates) >= 2 else 0.0
        gap = top1 - top2

        if uncertainty == "low" and retrieval_confidence == "high" and supports_top1 and has_memory_consensus and gap >= 0.75:
            return "high"
        if gap >= 0.35 or retrieval_confidence in {"medium", "high"} or has_memory_consensus:
            return "medium"
        return "low"

    def _rank_candidates(self, candidate_scores: Dict[str, float]) -> List[Tuple[str, float]]:
        ranked = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [(name, score) for name, score in ranked if name]

    def _apply_feature(
        self,
        candidate_scores: Dict[str, float],
        candidate_features: Dict[str, Dict[str, float]],
        label: str,
        feature_name: str,
        delta: float,
    ) -> None:
        if not label:
            return
        candidate_scores[label] = candidate_scores.get(label, 0.0) + float(delta)
        features = candidate_features.setdefault(label, {})
        features[feature_name] = float(features.get(feature_name, 0.0)) + float(delta)

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
