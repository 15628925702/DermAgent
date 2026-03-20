from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from memory.experience_bank import ExperienceBank
from memory.skill_index import SkillIndex


PAIR_TO_SPECIALIST = {
    ("ACK", "SCC"): "ack_scc_specialist_skill",
    ("BCC", "SCC"): "bcc_scc_specialist_skill",
    ("BCC", "SEK"): "bcc_sek_specialist_skill",
    ("MEL", "NEV"): "mel_nev_specialist_skill",
}


class SkillEvolutionDesigner:
    """
    Lightweight designer that periodically reads accumulated hard cases and
    nudges the structured skill bank toward recurring failure modes.

    This is intentionally simple:
    - refine existing skill thresholds / priorities / feature weights
    - produce proposals for missing specialists
    - persist enough state to avoid reprocessing the same hard cases
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.035,
        min_pair_support: int = 2,
    ) -> None:
        self.learning_rate = learning_rate
        self.min_pair_support = max(1, int(min_pair_support))
        self.evolution_round = 0
        self.processed_case_ids: List[str] = []
        self.proposed_skills: List[Dict[str, Any]] = []

    def evolve(
        self,
        *,
        bank: ExperienceBank,
        skill_index: SkillIndex,
    ) -> Dict[str, Any]:
        hard_cases = bank.get_hard_cases()
        unseen_cases = [
            item for item in hard_cases
            if str(item.get("case_id", "")).strip()
            and str(item.get("case_id", "")).strip() not in set(self.processed_case_ids)
        ]
        if not unseen_cases:
            return {
                "updated": False,
                "reason": "no_new_hard_cases",
                "num_seen_hard_cases": len(hard_cases),
                "num_new_hard_cases": 0,
                "evolution_round": self.evolution_round,
            }

        self.evolution_round += 1
        updates: List[Dict[str, Any]] = []

        pair_counts = Counter()
        low_support_cases = 0
        fallback_cases = 0
        malignant_cases = 0
        confusion_cases = 0

        for item in unseen_cases:
            pair = self._extract_pair(item)
            if len(pair) == 2:
                pair_counts[tuple(pair)] += 1
            signals = item.get("learning_signals", {}) or {}
            if bool(signals.get("low_retrieval_support", False)):
                low_support_cases += 1
            if bool(signals.get("fallback_case", False)):
                fallback_cases += 1
            if bool(signals.get("confusion_case", False)):
                confusion_cases += 1
            final_label = str(item.get("final_label", "")).strip().upper()
            if final_label in {"MEL", "BCC", "SCC"}:
                malignant_cases += 1

        updates.extend(
            self._refine_generic_skills(
                skill_index=skill_index,
                total_cases=len(unseen_cases),
                low_support_cases=low_support_cases,
                fallback_cases=fallback_cases,
                malignant_cases=malignant_cases,
                confusion_cases=confusion_cases,
            )
        )
        updates.extend(
            self._refine_specialists(
                skill_index=skill_index,
                pair_counts=pair_counts,
            )
        )
        proposals = self._propose_missing_specialists(pair_counts=pair_counts, skill_index=skill_index)

        for item in unseen_cases:
            case_id = str(item.get("case_id", "")).strip()
            if case_id and case_id not in self.processed_case_ids:
                self.processed_case_ids.append(case_id)

        self._merge_proposals(proposals)

        return {
            "updated": bool(updates or proposals),
            "evolution_round": self.evolution_round,
            "num_seen_hard_cases": len(hard_cases),
            "num_new_hard_cases": len(unseen_cases),
            "generic_counts": {
                "low_support_cases": low_support_cases,
                "fallback_cases": fallback_cases,
                "malignant_cases": malignant_cases,
                "confusion_cases": confusion_cases,
            },
            "pair_counts": [
                {"pair": list(pair), "count": count}
                for pair, count in pair_counts.most_common()
            ],
            "skill_updates": updates,
            "proposed_skills": proposals,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "learning_rate": self.learning_rate,
            "min_pair_support": self.min_pair_support,
            "evolution_round": self.evolution_round,
            "processed_case_ids": list(self.processed_case_ids),
            "proposed_skills": list(self.proposed_skills),
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if int(payload.get("version", 0) or 0) != 1:
            return
        if payload.get("learning_rate") is not None:
            self.learning_rate = float(payload["learning_rate"])
        if payload.get("min_pair_support") is not None:
            self.min_pair_support = max(1, int(payload["min_pair_support"]))
        if payload.get("evolution_round") is not None:
            self.evolution_round = int(payload["evolution_round"])
        self.processed_case_ids = [
            str(x).strip()
            for x in (payload.get("processed_case_ids", []) or [])
            if str(x).strip()
        ]
        self.proposed_skills = [
            dict(item)
            for item in (payload.get("proposed_skills", []) or [])
            if isinstance(item, dict)
        ]

    def _refine_generic_skills(
        self,
        *,
        skill_index: SkillIndex,
        total_cases: int,
        low_support_cases: int,
        fallback_cases: int,
        malignant_cases: int,
        confusion_cases: int,
    ) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []
        total = max(1, total_cases)

        if confusion_cases > 0:
            compare_spec = skill_index.get("compare_skill")
            if compare_spec is not None:
                step = self.learning_rate * (confusion_cases / total)
                compare_spec.priority += step
                compare_spec.threshold = max(0.48, compare_spec.threshold - 0.12 * step)
                compare_spec.feature_weights["top_gap_small"] = float(compare_spec.feature_weights.get("top_gap_small", 0.0)) + 0.8 * step
                compare_spec.feature_weights["has_confusion_support"] = float(compare_spec.feature_weights.get("has_confusion_support", 0.0)) + 0.6 * step
                updates.append({
                    "skill_id": compare_spec.skill_id,
                    "reason": "recurring_confusion_cases",
                    "priority_delta": round(step, 6),
                    "new_threshold": round(compare_spec.threshold, 4),
                })

        if low_support_cases > 0 or fallback_cases > 0:
            metadata_spec = skill_index.get("metadata_consistency_skill")
            if metadata_spec is not None:
                step = self.learning_rate * ((low_support_cases + fallback_cases) / total)
                metadata_spec.priority += step
                metadata_spec.threshold = max(0.47, metadata_spec.threshold - 0.08 * step)
                metadata_spec.feature_weights["retrieval_low"] = float(metadata_spec.feature_weights.get("retrieval_low", 0.0)) + 0.7 * step
                metadata_spec.feature_weights["metadata_present"] = float(metadata_spec.feature_weights.get("metadata_present", 0.0)) + 0.45 * step
                updates.append({
                    "skill_id": metadata_spec.skill_id,
                    "reason": "recurring_low_support_cases",
                    "priority_delta": round(step, 6),
                    "new_threshold": round(metadata_spec.threshold, 4),
                })

        if malignant_cases > 0:
            risk_spec = skill_index.get("malignancy_risk_skill")
            if risk_spec is not None:
                step = self.learning_rate * (malignant_cases / total)
                risk_spec.priority += step
                risk_spec.threshold = max(0.45, risk_spec.threshold - 0.05 * step)
                risk_spec.feature_weights["has_malignant_candidate"] = float(risk_spec.feature_weights.get("has_malignant_candidate", 0.0)) + 0.65 * step
                risk_spec.feature_weights["num_malignant_cues"] = float(risk_spec.feature_weights.get("num_malignant_cues", 0.0)) + 0.5 * step
                updates.append({
                    "skill_id": risk_spec.skill_id,
                    "reason": "recurring_malignant_hard_cases",
                    "priority_delta": round(step, 6),
                    "new_threshold": round(risk_spec.threshold, 4),
                })

        uncertainty_spec = skill_index.get("uncertainty_assessment_skill")
        if uncertainty_spec is not None and total_cases > 0:
            step = self.learning_rate * min(1.0, total_cases / 10.0) * 0.2
            uncertainty_spec.priority += step
            uncertainty_spec.feature_weights["uncertainty_high"] = float(uncertainty_spec.feature_weights.get("uncertainty_high", 0.0)) + 0.3 * step
            updates.append({
                "skill_id": uncertainty_spec.skill_id,
                "reason": "maintain_calibration_anchor",
                "priority_delta": round(step, 6),
                "new_threshold": round(uncertainty_spec.threshold, 4),
            })

        return updates

    def _refine_specialists(
        self,
        *,
        skill_index: SkillIndex,
        pair_counts: Counter[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        updates: List[Dict[str, Any]] = []

        for pair, count in pair_counts.items():
            if count < self.min_pair_support:
                continue
            skill_id = PAIR_TO_SPECIALIST.get(pair)
            if not skill_id:
                continue
            spec = skill_index.get(skill_id)
            if spec is None:
                continue

            step = self.learning_rate * min(2.0, float(count))
            spec.priority += step
            spec.threshold = max(0.5, spec.threshold - 0.03 * min(3.0, float(count)))
            pair_feature_name = self._pair_feature_name(pair)
            if pair_feature_name:
                spec.feature_weights[pair_feature_name] = float(spec.feature_weights.get(pair_feature_name, 0.0)) + 0.55 * step
            spec.feature_weights["has_confusion_support"] = float(spec.feature_weights.get("has_confusion_support", 0.0)) + 0.35 * step
            spec.feature_weights["top_gap_small"] = float(spec.feature_weights.get("top_gap_small", 0.0)) + 0.2 * step
            updates.append({
                "skill_id": skill_id,
                "reason": f"repeated_hard_pair_{pair[0]}_{pair[1]}",
                "count": count,
                "priority_delta": round(step, 6),
                "new_threshold": round(spec.threshold, 4),
            })

        return updates

    def _propose_missing_specialists(
        self,
        *,
        pair_counts: Counter[Tuple[str, str]],
        skill_index: SkillIndex,
    ) -> List[Dict[str, Any]]:
        existing = {spec.skill_id for spec in skill_index.all_specs()}
        proposals: List[Dict[str, Any]] = []

        for pair, count in pair_counts.items():
            if count < max(3, self.min_pair_support + 1):
                continue
            mapped_skill = PAIR_TO_SPECIALIST.get(pair)
            if mapped_skill and mapped_skill in existing:
                continue

            proposal_id = f"{pair[0].lower()}_{pair[1].lower()}_specialist_skill"
            proposals.append({
                "skill_id": proposal_id,
                "pair": list(pair),
                "category": "specialist_proposal",
                "reason": f"Hard cases repeatedly show unresolved pair {pair[0]}/{pair[1]}",
                "support_count": count,
            })

        return proposals

    def _merge_proposals(self, proposals: List[Dict[str, Any]]) -> None:
        seen = {
            str(item.get("skill_id", "")).strip()
            for item in self.proposed_skills
            if str(item.get("skill_id", "")).strip()
        }
        for item in proposals:
            skill_id = str(item.get("skill_id", "")).strip()
            if not skill_id or skill_id in seen:
                continue
            self.proposed_skills.append(dict(item))
            seen.add(skill_id)

    def _extract_pair(self, item: Dict[str, Any]) -> List[str]:
        top_ddx = [
            str(x).strip().upper()
            for x in (item.get("top_ddx", []) or [])
            if str(x).strip()
        ]
        if len(top_ddx) >= 2:
            return sorted(top_ddx[:2])
        return []

    def _pair_feature_name(self, pair: Tuple[str, str]) -> str:
        if pair == ("ACK", "SCC"):
            return "has_ack_scc_pair"
        if pair == ("BCC", "SCC"):
            return "has_bcc_scc_pair"
        if pair == ("BCC", "SEK"):
            return "has_bcc_sek_pair"
        if pair == ("MEL", "NEV"):
            return "has_mel_nev_pair"
        return ""
