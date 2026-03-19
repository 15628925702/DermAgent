from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from agent.state import CaseState
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker


class ExperienceRetriever:
    def __init__(
        self,
        bank: ExperienceBank,
        reranker: UtilityAwareExperienceReranker | None = None,
    ) -> None:
        self.bank = bank
        self.reranker = reranker

    def retrieve(self, state: CaseState, top_k: int = 5) -> Dict[str, Any]:
        all_items = self.bank.list_all()

        raw_case_hits = self._retrieve_raw_cases(all_items, state, top_k=top_k)
        prototype_hits = self._retrieve_prototypes(all_items, state, top_k=max(3, min(top_k, 5)))
        confusion_hits = self._retrieve_confusions(all_items, state, top_k=max(2, min(top_k, 4)))
        rule_hits = self._retrieve_rules(all_items, state, top_k=max(2, min(top_k, 4)))

        reranker_debug: Dict[str, Any] = {}
        if self.reranker is not None:
            reranked = self.reranker.rerank(
                state,
                raw_case_hits=raw_case_hits,
                prototype_hits=prototype_hits,
                confusion_hits=confusion_hits,
                rule_hits=rule_hits,
            )
            raw_case_hits = reranked["raw_case_hits"]
            prototype_hits = reranked["prototype_hits"]
            confusion_hits = reranked["confusion_hits"]
            rule_hits = reranked["rule_hits"]
            reranker_debug = reranked.get("reranker_debug", {}) or {}

        retrieval_summary = self._build_retrieval_summary(
            state=state,
            raw_case_hits=raw_case_hits,
            prototype_hits=prototype_hits,
            confusion_hits=confusion_hits,
            rule_hits=rule_hits,
        )
        if reranker_debug:
            retrieval_summary["reranker_debug"] = reranker_debug

        result = {
            "top_k": top_k,
            "raw_case_hits": raw_case_hits,
            "prototype_hits": prototype_hits,
            "confusion_hits": confusion_hits,
            "rule_hits": rule_hits,
            "retrieval_summary": retrieval_summary,
        }

        state.retrieval = result
        state.trace(
            "retrieval",
            "success",
            "Experience retrieval completed",
            payload={
                "num_raw_case_hits": len(raw_case_hits),
                "num_prototype_hits": len(prototype_hits),
                "num_confusion_hits": len(confusion_hits),
                "num_rule_hits": len(rule_hits),
                "retrieval_confidence": retrieval_summary.get("retrieval_confidence", "low"),
                "memory_consensus_label": retrieval_summary.get("memory_consensus_label", ""),
            },
        )
        return result

    def _retrieve_raw_cases(self, all_items: List[Dict[str, Any]], state: CaseState, top_k: int) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in all_items:
            if item.get("experience_type") != "raw_case":
                continue
            score = self._score_raw_case(item, state)
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["_score"] = round(score, 4)
            scored.append((score, enriched))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def _score_raw_case(self, item: Dict[str, Any], state: CaseState) -> float:
        score = 0.0
        current_top_names = set(state.get_top_ddx_names(top_k=5))
        metadata = state.get_metadata()

        final_diag = str(item.get("final_decision", {}).get("diagnosis", "")).upper()
        if final_diag and final_diag in current_top_names:
            score += 4.0

        hist_ddx = {
            str(x.get("name", "")).upper()
            for x in item.get("perception", {}).get("ddx", [])
            if str(x.get("name", "")).strip()
        }
        score += 1.5 * len(current_top_names.intersection(hist_ddx))
        score += self._score_metadata_similarity(metadata, item.get("metadata", {}) or {})

        uncertainty_level = str(item.get("tags", {}).get("uncertainty_level", "")).lower()
        if uncertainty_level == "low":
            score += 0.5
        return score

    def _retrieve_prototypes(self, all_items: List[Dict[str, Any]], state: CaseState, top_k: int) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        current_top_names = set(state.get_top_ddx_names(top_k=5))

        for item in all_items:
            if item.get("experience_type") != "prototype":
                continue
            disease = str(item.get("disease", "")).upper()
            score = 0.0
            if disease in current_top_names:
                score += 5.0
            common_confusions = {str(x).upper() for x in item.get("common_confusions", [])}
            score += 1.0 * len(current_top_names.intersection(common_confusions))
            score += self._score_metadata_similarity(state.get_metadata(), item.get("typical_metadata", {}) or {})
            score += 0.35 * self._support_strength_bonus(item)
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["_score"] = round(score, 4)
            scored.append((score, enriched))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def _retrieve_confusions(self, all_items: List[Dict[str, Any]], state: CaseState, top_k: int) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        current_top_names = set(state.get_top_ddx_names(top_k=5))

        for item in all_items:
            if item.get("experience_type") != "confusion":
                continue
            pair = [str(x).upper() for x in item.get("pair", [])]
            pair_set = set(pair)
            overlap = len(current_top_names.intersection(pair_set))
            if overlap == 2:
                score = 5.0
            elif overlap == 1:
                score = 2.0
            else:
                score = 0.0
            score += 0.4 * self._support_strength_bonus(item)
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["_score"] = round(score, 4)
            scored.append((score, enriched))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def _retrieve_rules(self, all_items: List[Dict[str, Any]], state: CaseState, top_k: int) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for item in all_items:
            if item.get("experience_type") != "rule":
                continue
            score = self._score_rule_match(item, state)
            if score <= 0:
                continue
            enriched = dict(item)
            enriched["_score"] = round(score, 4)
            scored.append((score, enriched))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def _score_rule_match(self, item: Dict[str, Any], state: CaseState) -> float:
        conditions = item.get("trigger_conditions", {}) or {}
        current_top_names = set(state.get_top_ddx_names(top_k=5))
        current_uncertainty = state.get_uncertainty_level()
        score = 0.0

        min_uncertainty_level = str(conditions.get("min_uncertainty_level", "")).lower()
        if min_uncertainty_level:
            if self._uncertainty_rank(current_uncertainty) >= self._uncertainty_rank(min_uncertainty_level):
                score += 2.0
            else:
                return 0.0

        requires_any = {str(x).upper() for x in conditions.get("requires_any_disease", [])}
        if requires_any:
            overlap = current_top_names.intersection(requires_any)
            if not overlap:
                return 0.0
            score += 1.0 * len(overlap)

        requires_all = {str(x).upper() for x in conditions.get("requires_all_diseases", [])}
        if requires_all:
            if not requires_all.issubset(current_top_names):
                return 0.0
            score += 2.0

        priority = int(item.get("priority", 1))
        score += 0.1 * priority
        return score

    def _build_retrieval_summary(
        self,
        state: CaseState,
        raw_case_hits: List[Dict[str, Any]],
        prototype_hits: List[Dict[str, Any]],
        confusion_hits: List[Dict[str, Any]],
        rule_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        support_labels = self._collect_support_labels(raw_case_hits, prototype_hits)
        prototype_votes = self._collect_prototype_votes(prototype_hits)
        confusion_votes = self._collect_confusion_votes(confusion_hits)
        confusion_pairs = [x.get("pair", []) for x in confusion_hits]
        memory_recommended_skills = self._collect_memory_recommended_skills(prototype_hits, confusion_hits)
        rule_recommended_skills = self._collect_rule_recommended_skills(rule_hits)
        combined_recommended = list(dict.fromkeys(memory_recommended_skills + rule_recommended_skills))
        memory_consensus_label = self._select_memory_consensus_label(
            state=state,
            support_labels=support_labels,
            prototype_votes=prototype_votes,
            confusion_votes=confusion_votes,
        )
        retrieval_confidence = self._estimate_retrieval_confidence(
            raw_case_hits=raw_case_hits,
            prototype_hits=prototype_hits,
            support_labels=support_labels,
            prototype_votes=prototype_votes,
            confusion_votes=confusion_votes,
        )

        return {
            "support_labels": support_labels,
            "prototype_votes": prototype_votes,
            "confusion_votes": confusion_votes,
            "memory_consensus_label": memory_consensus_label,
            "retrieval_confidence": retrieval_confidence,
            "has_confusion_support": len(confusion_hits) > 0,
            "confusion_pairs": confusion_pairs,
            "memory_recommended_skills": memory_recommended_skills,
            "rule_recommended_skills": rule_recommended_skills,
            "recommended_skills": combined_recommended,
            "support_strength": {
                "raw_case": len(raw_case_hits),
                "prototype": len(prototype_hits),
                "confusion": len(confusion_hits),
                "rule": len(rule_hits),
            },
            "top_support_case_ids": [x.get("case_id", "") for x in raw_case_hits[:3] if x.get("case_id")],
            "supports_top1": self._supports_top1(state, support_labels, prototype_votes, confusion_votes),
        }

    def _collect_support_labels(self, raw_case_hits: List[Dict[str, Any]], prototype_hits: List[Dict[str, Any]]) -> List[str]:
        counter: Counter[str] = Counter()
        for item in raw_case_hits:
            diagnosis = str(item.get("final_decision", {}).get("diagnosis", "")).upper()
            if diagnosis:
                counter[diagnosis] += 2
        for item in prototype_hits:
            disease = str(item.get("disease", "")).upper()
            if disease:
                bonus = 1 + min(2, int(self._safe_int(item.get("source_count")) or 0) // 2)
                counter[disease] += bonus
        return [name for name, _ in counter.most_common(5)]

    def _collect_prototype_votes(self, prototype_hits: List[Dict[str, Any]]) -> Dict[str, float]:
        votes: Dict[str, float] = {}
        for item in prototype_hits:
            label = str(item.get("disease", "")).strip().upper()
            if not label:
                continue
            utility = self._safe_float(item.get("utility_score"), default=0.0)
            support = self._support_strength_bonus(item)
            votes[label] = round(votes.get(label, 0.0) + utility + 0.2 * support, 4)
        return dict(sorted(votes.items(), key=lambda pair: pair[1], reverse=True))

    def _collect_confusion_votes(self, confusion_hits: List[Dict[str, Any]]) -> Dict[str, float]:
        votes: Dict[str, float] = {}
        for item in confusion_hits:
            utility = self._safe_float(item.get("utility_score"), default=0.0)
            support = self._support_strength_bonus(item)
            label_votes = item.get("label_votes", {}) or {}
            if label_votes:
                total = sum(max(0, int(v)) for v in label_votes.values())
                if total > 0:
                    for label, count in label_votes.items():
                        name = str(label).strip().upper()
                        if not name:
                            continue
                        share = max(0, int(count)) / total
                        votes[name] = round(votes.get(name, 0.0) + utility * share + 0.15 * support * share, 4)
                    continue
            pair = [str(x).strip().upper() for x in item.get("pair", []) if str(x).strip()]
            if not pair:
                continue
            split_vote = (utility + 0.1 * support) / len(pair)
            for name in pair:
                votes[name] = round(votes.get(name, 0.0) + split_vote, 4)
        return dict(sorted(votes.items(), key=lambda pair: pair[1], reverse=True))

    def _collect_memory_recommended_skills(
        self,
        prototype_hits: List[Dict[str, Any]],
        confusion_hits: List[Dict[str, Any]],
    ) -> List[str]:
        counter: Counter[str] = Counter()
        for item in prototype_hits:
            for skill in item.get("recommended_skills", []):
                skill_name = str(skill).strip()
                if skill_name:
                    counter[skill_name] += 1
        for item in confusion_hits:
            for skill in item.get("useful_skills", []):
                skill_name = str(skill).strip()
                if skill_name:
                    counter[skill_name] += 2
        return [name for name, _ in counter.most_common(5)]

    def _collect_rule_recommended_skills(self, rule_hits: List[Dict[str, Any]]) -> List[str]:
        counter: Counter[str] = Counter()
        for item in rule_hits:
            action = item.get("action", {}) or {}
            for skill in action.get("suggested_skills", []):
                skill_name = str(skill).strip()
                if skill_name:
                    counter[skill_name] += max(1, int(self._safe_int(item.get("priority")) or 1))
        return [name for name, _ in counter.most_common(5)]

    def _estimate_retrieval_confidence(
        self,
        raw_case_hits: List[Dict[str, Any]],
        prototype_hits: List[Dict[str, Any]],
        support_labels: List[str],
        prototype_votes: Dict[str, float],
        confusion_votes: Dict[str, float],
    ) -> str:
        strong_prototype = next(iter(prototype_votes.values()), 0.0)
        strong_confusion = next(iter(confusion_votes.values()), 0.0)
        if len(raw_case_hits) >= 3 and len(support_labels) >= 1:
            return "high"
        if strong_prototype >= 3.0 and len(prototype_hits) >= 1:
            return "high"
        if len(raw_case_hits) >= 1 or len(prototype_hits) >= 2 or strong_confusion >= 1.2:
            return "medium"
        return "low"

    def _select_memory_consensus_label(
        self,
        *,
        state: CaseState,
        support_labels: List[str],
        prototype_votes: Dict[str, float],
        confusion_votes: Dict[str, float],
    ) -> str:
        combined: Counter[str] = Counter()
        for idx, label in enumerate(support_labels[:5]):
            name = str(label).strip().upper()
            if name:
                combined[name] += max(1, 5 - idx)
        for label, score in prototype_votes.items():
            combined[str(label).strip().upper()] += float(score)
        for label, score in confusion_votes.items():
            combined[str(label).strip().upper()] += float(score)
        top_names = set(state.get_top_ddx_names(top_k=5))
        filtered = [(label, score) for label, score in combined.items() if label in top_names]
        if filtered:
            filtered.sort(key=lambda pair: pair[1], reverse=True)
            return filtered[0][0]
        return support_labels[0] if support_labels else ""

    def _supports_top1(
        self,
        state: CaseState,
        support_labels: List[str],
        prototype_votes: Dict[str, float],
        confusion_votes: Dict[str, float],
    ) -> bool:
        top_names = state.get_top_ddx_names(top_k=1)
        if not top_names:
            return False
        top1 = top_names[0]
        return top1 in set(support_labels) or top1 in prototype_votes or top1 in confusion_votes

    def _score_metadata_similarity(self, current_meta: Dict[str, Any], stored_meta: Dict[str, Any]) -> float:
        score = 0.0
        current_age = self._safe_int(current_meta.get("age"))
        stored_age = self._safe_int(stored_meta.get("age"))
        if current_age is not None and stored_age is not None:
            if abs(current_age - stored_age) <= 10:
                score += 1.0
            elif abs(current_age - stored_age) <= 20:
                score += 0.5
        current_age_group = self._norm_str(current_meta.get("age_group"))
        stored_age_group = self._norm_str(stored_meta.get("age_group"))
        if current_age_group and stored_age_group and current_age_group == stored_age_group:
            score += 0.5
        current_sex = self._norm_str(current_meta.get("sex"))
        stored_sex = self._norm_str(stored_meta.get("sex"))
        if current_sex and stored_sex and current_sex == stored_sex:
            score += 0.3
        current_site = self._norm_str(current_meta.get("location") or current_meta.get("site") or current_meta.get("anatomical_site"))
        stored_site = self._norm_str(stored_meta.get("location") or stored_meta.get("site") or stored_meta.get("anatomical_site"))
        if current_site and stored_site:
            if current_site == stored_site:
                score += 1.0
            elif current_site in stored_site or stored_site in current_site:
                score += 0.5
        return score

    def _support_strength_bonus(self, item: Dict[str, Any]) -> float:
        count = self._safe_int(item.get("source_count")) or 0
        level = str(item.get("compression_level", "")).strip().lower()
        bonus = 0.0
        if count >= 5:
            bonus += 2.0
        elif count >= 2:
            bonus += 1.0
        elif count >= 1:
            bonus += 0.4
        if level == "high":
            bonus += 1.0
        elif level == "medium":
            bonus += 0.5
        elif level == "low":
            bonus += 0.2
        return bonus

    def _uncertainty_rank(self, level: str) -> int:
        return {"low": 1, "medium": 2, "high": 3}.get(str(level).lower(), 3)

    def _safe_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _norm_str(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()
