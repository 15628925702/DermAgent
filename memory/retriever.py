"""
经验检索器。

基础版目标：
- 从经验库中分类型检索 raw_case / prototype / confusion / rule
- 用简单规则分数做初版检索
- 给 planner / aggregator 提供稳定的 retrieval_summary

后续可升级方向：
- 替换为 embedding retriever
- 增加 utility reranker
- 引入 learned retrieval confidence

相关文件：
- memory/experience_bank.py
- skills/retrieval.py
- agent/planner.py
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from agent.state import CaseState
from memory.experience_bank import ExperienceBank


class ExperienceRetriever:
    def __init__(self, bank: ExperienceBank) -> None:
        self.bank = bank

    def retrieve(self, state: CaseState, top_k: int = 5) -> Dict[str, Any]:
        """
        基础版经验检索入口。

        返回：
        {
            "top_k": ...,
            "raw_case_hits": [...],
            "prototype_hits": [...],
            "confusion_hits": [...],
            "rule_hits": [...],
            "retrieval_summary": {...}
        }
        """
        all_items = self.bank.list_all()

        raw_case_hits = self._retrieve_raw_cases(all_items, state, top_k=top_k)
        prototype_hits = self._retrieve_prototypes(all_items, state, top_k=max(3, min(top_k, 5)))
        confusion_hits = self._retrieve_confusions(all_items, state, top_k=max(2, min(top_k, 4)))
        rule_hits = self._retrieve_rules(all_items, state, top_k=max(2, min(top_k, 4)))

        retrieval_summary = self._build_retrieval_summary(
            state=state,
            raw_case_hits=raw_case_hits,
            prototype_hits=prototype_hits,
            confusion_hits=confusion_hits,
            rule_hits=rule_hits,
        )

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
            },
        )
        return result

    # =========================
    # Raw case retrieval
    # =========================

    def _retrieve_raw_cases(
        self,
        all_items: List[Dict[str, Any]],
        state: CaseState,
        top_k: int,
    ) -> List[Dict[str, Any]]:
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
        """
        基础版 raw case 打分：
        - final diagnosis 与当前 top-k 候选重合
        - 历史 ddx 与当前 ddx 重合
        - metadata 相似（年龄段、部位、性别）
        """
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
        ddx_overlap = len(current_top_names.intersection(hist_ddx))
        score += 1.5 * ddx_overlap

        score += self._score_metadata_similarity(
            current_meta=metadata,
            stored_meta=item.get("metadata", {}) or {},
        )

        uncertainty_level = str(
            item.get("tags", {}).get("uncertainty_level", "")
        ).lower()
        if uncertainty_level == "low":
            score += 0.5

        return score

    # =========================
    # Prototype retrieval
    # =========================

    def _retrieve_prototypes(
        self,
        all_items: List[Dict[str, Any]],
        state: CaseState,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        current_top_names = set(state.get_top_ddx_names(top_k=5))

        for item in all_items:
            if item.get("experience_type") != "prototype":
                continue

            disease = str(item.get("disease", "")).upper()
            score = 0.0

            if disease in current_top_names:
                score += 5.0

            common_confusions = {
                str(x).upper() for x in item.get("common_confusions", [])
            }
            score += 1.0 * len(current_top_names.intersection(common_confusions))

            score += self._score_metadata_similarity(
                current_meta=state.get_metadata(),
                stored_meta=item.get("typical_metadata", {}) or {},
            )

            if score <= 0:
                continue

            enriched = dict(item)
            enriched["_score"] = round(score, 4)
            scored.append((score, enriched))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    # =========================
    # Confusion retrieval
    # =========================

    def _retrieve_confusions(
        self,
        all_items: List[Dict[str, Any]],
        state: CaseState,
        top_k: int,
    ) -> List[Dict[str, Any]]:
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

            if score <= 0:
                continue

            enriched = dict(item)
            enriched["_score"] = round(score, 4)
            scored.append((score, enriched))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    # =========================
    # Rule retrieval
    # =========================

    def _retrieve_rules(
        self,
        all_items: List[Dict[str, Any]],
        state: CaseState,
        top_k: int,
    ) -> List[Dict[str, Any]]:
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
        """
        基础版 rule 只做轻量触发：
        trigger_conditions 支持：
        - min_uncertainty_level: low / medium / high
        - requires_any_disease: [...]
        - requires_all_diseases: [...]
        """
        conditions = item.get("trigger_conditions", {}) or {}
        current_top_names = set(state.get_top_ddx_names(top_k=5))
        current_uncertainty = state.get_uncertainty_level()

        score = 0.0

        min_uncertainty_level = str(
            conditions.get("min_uncertainty_level", "")
        ).lower()
        if min_uncertainty_level:
            if self._uncertainty_rank(current_uncertainty) >= self._uncertainty_rank(
                min_uncertainty_level
            ):
                score += 2.0
            else:
                return 0.0

        requires_any = {
            str(x).upper() for x in conditions.get("requires_any_disease", [])
        }
        if requires_any:
            overlap = current_top_names.intersection(requires_any)
            if not overlap:
                return 0.0
            score += 1.0 * len(overlap)

        requires_all = {
            str(x).upper() for x in conditions.get("requires_all_diseases", [])
        }
        if requires_all:
            if not requires_all.issubset(current_top_names):
                return 0.0
            score += 2.0

        priority = int(item.get("priority", 1))
        score += 0.1 * priority
        return score

    # =========================
    # Summary builder
    # =========================

    def _build_retrieval_summary(
        self,
        state: CaseState,
        raw_case_hits: List[Dict[str, Any]],
        prototype_hits: List[Dict[str, Any]],
        confusion_hits: List[Dict[str, Any]],
        rule_hits: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        support_labels = self._collect_support_labels(raw_case_hits, prototype_hits)
        confusion_pairs = [x.get("pair", []) for x in confusion_hits]
        recommended_skills = self._collect_recommended_skills(
            prototype_hits=prototype_hits,
            confusion_hits=confusion_hits,
            rule_hits=rule_hits,
        )

        retrieval_confidence = self._estimate_retrieval_confidence(
            raw_case_hits=raw_case_hits,
            prototype_hits=prototype_hits,
            support_labels=support_labels,
        )

        summary = {
            "support_labels": support_labels,
            "retrieval_confidence": retrieval_confidence,
            "has_confusion_support": len(confusion_hits) > 0,
            "confusion_pairs": confusion_pairs,
            "recommended_skills": recommended_skills,
            "support_strength": {
                "raw_case": len(raw_case_hits),
                "prototype": len(prototype_hits),
                "confusion": len(confusion_hits),
                "rule": len(rule_hits),
            },
            "top_support_case_ids": [
                x.get("case_id", "") for x in raw_case_hits[:3] if x.get("case_id")
            ],
            "supports_top1": self._supports_top1(state, support_labels),
        }

        return summary

    def _collect_support_labels(
        self,
        raw_case_hits: List[Dict[str, Any]],
        prototype_hits: List[Dict[str, Any]],
    ) -> List[str]:
        counter: Counter[str] = Counter()

        for item in raw_case_hits:
            diagnosis = str(item.get("final_decision", {}).get("diagnosis", "")).upper()
            if diagnosis:
                counter[diagnosis] += 2

        for item in prototype_hits:
            disease = str(item.get("disease", "")).upper()
            if disease:
                counter[disease] += 1

        return [name for name, _ in counter.most_common(5)]

    def _collect_recommended_skills(
        self,
        prototype_hits: List[Dict[str, Any]],
        confusion_hits: List[Dict[str, Any]],
        rule_hits: List[Dict[str, Any]],
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

        for item in rule_hits:
            action = item.get("action", {}) or {}
            for skill in action.get("suggested_skills", []):
                skill_name = str(skill).strip()
                if skill_name:
                    counter[skill_name] += 2

        return [name for name, _ in counter.most_common(5)]

    def _estimate_retrieval_confidence(
        self,
        raw_case_hits: List[Dict[str, Any]],
        prototype_hits: List[Dict[str, Any]],
        support_labels: List[str],
    ) -> str:
        """
        基础版置信度估计：
        high:
            - raw case 命中 >= 3
            - 且 support labels 不空
        medium:
            - raw case 命中 >= 1 或 prototype 命中 >= 2
        low:
            - 其他
        """
        if len(raw_case_hits) >= 3 and len(support_labels) >= 1:
            return "high"

        if len(raw_case_hits) >= 1 or len(prototype_hits) >= 2:
            return "medium"

        return "low"

    def _supports_top1(self, state: CaseState, support_labels: List[str]) -> bool:
        top_names = state.get_top_ddx_names(top_k=1)
        if not top_names:
            return False
        return top_names[0] in set(support_labels)

    # =========================
    # Similarity helpers
    # =========================

    def _score_metadata_similarity(
        self,
        current_meta: Dict[str, Any],
        stored_meta: Dict[str, Any],
    ) -> float:
        """
        极简 metadata 相似度：
        - age / age_group
        - sex
        - location / site / anatomical_site
        """
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

        current_site = self._norm_str(
            current_meta.get("location")
            or current_meta.get("site")
            or current_meta.get("anatomical_site")
        )
        stored_site = self._norm_str(
            stored_meta.get("location")
            or stored_meta.get("site")
            or stored_meta.get("anatomical_site")
        )
        if current_site and stored_site:
            if current_site == stored_site:
                score += 1.0
            elif current_site in stored_site or stored_site in current_site:
                score += 0.5

        return score

    def _uncertainty_rank(self, level: str) -> int:
        level = str(level).lower()
        mapping = {
            "low": 1,
            "medium": 2,
            "high": 3,
        }
        return mapping.get(level, 3)

    def _safe_int(self, value: Any) -> int | None:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _norm_str(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()