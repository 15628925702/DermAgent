from __future__ import annotations

from typing import Any, Dict, List

from agent.state import CaseState
from skills.base import BaseSkill


class MelNevSpecialistSkill(BaseSkill):
    """
    MEL / NEV 混淆组 specialist。

    什么时候用：
    - perception top-k 同时出现 MEL 和 NEV
    - 存在色素不均、边界不规则等线索

    基础版策略：
    1. 仅处理 MEL vs NEV
    2. 结合视觉线索、年龄、retrieval、compare 输出
    3. 给出 recommendation + group_scores + supporting_evidence

    相关文件：
    - agent/planner.py
    - skills/compare.py
    - memory/retriever.py
    - agent/aggregator.py
    """

    name = "mel_nev_specialist_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        ddx = state.perception.get("ddx_candidates", []) or []
        top_names = [
            self._norm_label(x.get("name"))
            for x in ddx[:5]
            if self._norm_label(x.get("name")) != "UNKNOWN"
        ]

        if not {"MEL", "NEV"}.issubset(set(top_names)):
            result = {
                "target_group": ["MEL", "NEV"],
                "recommendation": None,
                "supports": None,
                "supported_label": None,
                "local_decision": {
                    "supports": None,
                    "opposes": None,
                    "confidence": 0.0,
                    "applicable": 0.0,
                    "reason": "mel_nev_not_both_present",
                },
                "supporting_evidence": [],
                "evidence_items": [],
                "applicable": 0.0,
                "confidence": 0.0,
                "reason": "mel_nev_not_both_present",
            }
            state.skill_outputs[self.name] = result
            state.trace(self.name, "warning", "MEL/NEV specialist skipped: pair not present")
            return result

        scores = {
            "MEL": 0.0,
            "NEV": 0.0,
        }
        evidence: List[str] = []

        # =========================
        # 1. perception 排名/分数
        # =========================
        mel_item = self._find_candidate(ddx, "MEL")
        nev_item = self._find_candidate(ddx, "NEV")

        mel_score = self._extract_candidate_score(mel_item, default=0.8)
        nev_score = self._extract_candidate_score(nev_item, default=0.8)

        scores["MEL"] += mel_score
        scores["NEV"] += nev_score
        evidence.append(f"perception_score_support: MEL={mel_score:.3f}, NEV={nev_score:.3f}")

        # =========================
        # 2. visual cues
        # =========================
        visual_cues = self._normalize_text_list(state.perception.get("visual_cues", []))
        mel_keywords = [
            "asymmetry",
            "irregular border",
            "irregular pigmentation",
            "variegated",
            "multiple colors",
            "atypical pigment",
            "blue white veil",
            "ulcer",
            "bleeding",
        ]
        nev_keywords = [
            "symmetric",
            "regular border",
            "uniform pigmentation",
            "stable appearance",
            "homogeneous",
            "well circumscribed",
        ]

        mel_hits = self._count_keyword_hits(visual_cues, mel_keywords)
        nev_hits = self._count_keyword_hits(visual_cues, nev_keywords)

        if mel_hits > 0:
            mel_bonus = min(0.9, 0.18 * mel_hits)
            scores["MEL"] += mel_bonus
            evidence.append(f"visual_cues_support_mel: hits={mel_hits}, bonus={mel_bonus:.2f}")

        if nev_hits > 0:
            nev_bonus = min(0.7, 0.15 * nev_hits)
            scores["NEV"] += nev_bonus
            evidence.append(f"visual_cues_support_nev: hits={nev_hits}, bonus={nev_bonus:.2f}")

        # =========================
        # 3. risk cues
        # =========================
        risk_cues = state.perception.get("risk_cues", {}) or {}
        malignant_cues = self._normalize_text_list(risk_cues.get("malignant_cues", []))
        suspicious_cues = self._normalize_text_list(risk_cues.get("suspicious_cues", []))

        if malignant_cues:
            bonus = min(0.8, 0.2 * len(malignant_cues))
            scores["MEL"] += bonus
            evidence.append(f"malignant_cues_support_mel: n={len(malignant_cues)}, bonus={bonus:.2f}")

        if suspicious_cues:
            bonus = min(0.4, 0.1 * len(suspicious_cues))
            scores["MEL"] += bonus
            evidence.append(f"suspicious_cues_lean_mel: n={len(suspicious_cues)}, bonus={bonus:.2f}")

        # =========================
        # 4. metadata
        # =========================
        metadata = state.get_metadata()
        age = self._safe_int(metadata.get("age"))

        if age is not None:
            if age >= 50:
                scores["MEL"] += 0.18
                evidence.append("older_age_weakly_supports_mel")
            elif age <= 25:
                scores["NEV"] += 0.12
                evidence.append("younger_age_weakly_supports_nev")

        # =========================
        # 5. retrieval support
        # =========================
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        support_labels = [self._norm_label(x) for x in retrieval_summary.get("support_labels", [])]
        retrieval_confidence = str(
            retrieval_summary.get("retrieval_confidence", "low")
        ).lower()

        retrieval_bonus_map = {
            "high": 0.45,
            "medium": 0.25,
            "low": 0.10,
        }
        retrieval_bonus = retrieval_bonus_map.get(retrieval_confidence, 0.10)

        if "MEL" in support_labels:
            scores["MEL"] += retrieval_bonus
            evidence.append(f"retrieval_supports_mel: bonus={retrieval_bonus:.2f}")

        if "NEV" in support_labels:
            scores["NEV"] += retrieval_bonus
            evidence.append(f"retrieval_supports_nev: bonus={retrieval_bonus:.2f}")

        # confusion memory 里如果正好 hit 到 MEL/NEV，也稍微提高 specialist 置信
        confusion_hits = state.retrieval.get("confusion_hits", []) or []
        for item in confusion_hits:
            pair = {self._norm_label(x) for x in item.get("pair", [])}
            if pair == {"MEL", "NEV"}:
                scores["MEL"] += 0.08
                scores["NEV"] += 0.08
                evidence.append("mel_nev_confusion_memory_found")
                break

        # =========================
        # 6. compare skill 输出作为参考
        # =========================
        compare_output = state.skill_outputs.get("compare_skill", {}) or {}
        compare_winner = self._norm_label(
            compare_output.get("winner")
            or compare_output.get("recommendation")
            or compare_output.get("final_choice")
        )
        compare_conf = self._safe_float(compare_output.get("confidence"), default=0.0)

        if compare_winner == "MEL":
            bonus = 0.10 + min(0.15, compare_conf * 0.15)
            scores["MEL"] += bonus
            evidence.append(f"compare_skill_supports_mel: bonus={bonus:.2f}")
        elif compare_winner == "NEV":
            bonus = 0.10 + min(0.15, compare_conf * 0.15)
            scores["NEV"] += bonus
            evidence.append(f"compare_skill_supports_nev: bonus={bonus:.2f}")

        # =========================
        # 7. metadata consistency skill
        # =========================
        meta_output = state.skill_outputs.get("metadata_consistency_skill", {}) or {}
        supported = [
            self._norm_label(x)
            for x in (meta_output.get("supported_diagnoses", []) or meta_output.get("supported_labels", []) or [])
        ]
        penalized = [
            self._norm_label(x)
            for x in (meta_output.get("penalized_diagnoses", []) or meta_output.get("penalized_labels", []) or [])
        ]

        if "MEL" in supported:
            scores["MEL"] += 0.12
            evidence.append("metadata_consistency_supports_mel")
        if "NEV" in supported:
            scores["NEV"] += 0.12
            evidence.append("metadata_consistency_supports_nev")
        if "MEL" in penalized:
            scores["MEL"] -= 0.10
            evidence.append("metadata_consistency_penalizes_mel")
        if "NEV" in penalized:
            scores["NEV"] -= 0.10
            evidence.append("metadata_consistency_penalizes_nev")

        # =========================
        # 8. 最终决策
        # =========================
        recommendation = "MEL" if scores["MEL"] >= scores["NEV"] else "NEV"
        loser = "NEV" if recommendation == "MEL" else "MEL"
        gap = abs(scores["MEL"] - scores["NEV"])

        if gap >= 0.90:
            confidence = 0.90
            reason = "strong_specialist_preference"
        elif gap >= 0.35:
            confidence = 0.75
            reason = "moderate_specialist_preference"
        else:
            confidence = 0.58
            reason = "weak_specialist_preference"

        result = {
            "target_group": ["MEL", "NEV"],
            "recommendation": recommendation,
            "supports": recommendation,
            "supported_label": recommendation,
            "loser": loser,
            "group_scores": {
                "MEL": round(scores["MEL"], 4),
                "NEV": round(scores["NEV"], 4),
            },
            "supporting_evidence": evidence[:20],
            "evidence_items": self._build_evidence_items(
                scores=scores,
                recommendation=recommendation,
                loser=loser,
                confidence=confidence,
                gap=gap,
                evidence=evidence,
            ),
            "applicable": 1.0,
            "confidence": round(confidence, 4),
            "gap": round(gap, 4),
            "reason": reason,
            "local_decision": {
                "supports": recommendation,
                "opposes": loser,
                "confidence": round(confidence, 4),
                "applicable": 1.0,
                "gap": round(gap, 4),
                "reason": reason,
            },
        }

        state.skill_outputs[self.name] = result
        state.trace(
            self.name,
            "success",
            f"MEL/NEV specialist completed: recommendation={recommendation}",
            payload={
                "recommendation": recommendation,
                "confidence": round(confidence, 4),
                "gap": round(gap, 4),
            },
        )
        return result

    def _build_evidence_items(
        self,
        *,
        scores: Dict[str, float],
        recommendation: str,
        loser: str,
        confidence: float,
        gap: float,
        evidence: List[str],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = [
            {
                "source": self.name,
                "type": "group_summary",
                "supports": recommendation,
                "opposes": loser,
                "weight": round(confidence, 4),
                "gap": round(gap, 4),
                "group_scores": {name: round(value, 4) for name, value in scores.items()},
            }
        ]
        for line in evidence[:8]:
            items.append(
                {
                    "source": self.name,
                    "type": "rationale",
                    "supports": recommendation,
                    "detail": line,
                }
            )
        return items

    def _find_candidate(self, ddx: List[Dict[str, Any]], target: str) -> Dict[str, Any]:
        target = self._norm_label(target)
        for item in ddx:
            if self._norm_label(item.get("name")) == target:
                return item
        return {}

    def _extract_candidate_score(self, item: Dict[str, Any], default: float) -> float:
        for key in ["score", "probability", "confidence"]:
            value = item.get(key)
            try:
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                continue
        return default

    def _count_keyword_hits(self, cues: List[str], keywords: List[str]) -> int:
        hits = 0
        for cue in cues:
            cue_lower = cue.lower()
            for kw in keywords:
                if kw in cue_lower:
                    hits += 1
                    break
        return hits

    def _normalize_text_list(self, items: Any) -> List[str]:
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for x in items:
            text = str(x).strip()
            if text and text not in out:
                out.append(text)
        return out

    def _norm_label(self, value: Any) -> str:
        if value is None:
            return "UNKNOWN"
        text = str(value).strip().upper()
        return text if text else "UNKNOWN"

    def _safe_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default
