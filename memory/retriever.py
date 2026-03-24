from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Tuple

from agent.state import CaseState
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker


class LearnableRetrievalScorer:
    """可学习的检索打分器，将规则式打分变成参数化学习"""

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

        # 打分权重 - 将硬编码的分数变成可学习的参数
        self.score_weights = {
            # 疾病匹配相关
            "disease_exact_match": 4.0,
            "disease_partial_match": 2.0,

            # 元数据相似性
            "metadata_age_match": 1.5,
            "metadata_site_match": 1.0,
            "metadata_history_match": 0.8,

            # 置信度和不确定性
            "confidence_match": 0.8,
            "uncertainty_match": 0.6,

            # 视觉特征相似性
            "visual_cues_overlap": 1.2,
            "risk_cues_match": 0.7,

            # 时间和使用频率
            "recency_bonus": 0.3,
            "frequency_penalty": -0.1,
        }

        # 特征权重 - 用于学习哪些特征更重要
        self.feature_importance = {
            "disease_match": 1.0,
            "metadata_similarity": 0.8,
            "confidence_alignment": 0.6,
            "visual_similarity": 0.7,
            "temporal_factors": 0.3,
        }

    def score_raw_case(self, case_item: Dict[str, Any], query_state: CaseState) -> float:
        """计算原始病例的匹配分数"""
        score = 0.0
        features_used = {}

        # 疾病匹配
        case_disease = str(case_item.get("disease", "")).strip().upper()
        query_diseases = set(query_state.get_top_ddx_names(top_k=5))
        if case_disease in query_diseases:
            score += self.score_weights["disease_exact_match"]
            features_used["disease_match"] = 1.0
        elif case_disease:  # 部分匹配逻辑可以扩展
            score += self.score_weights["disease_partial_match"] * 0.5
            features_used["disease_match"] = 0.5

        # 元数据相似性
        metadata_score = self._score_metadata_similarity(case_item, query_state)
        score += metadata_score
        features_used["metadata_similarity"] = min(1.0, metadata_score / 2.0)

        # 置信度对齐
        confidence_score = self._score_confidence_alignment(case_item, query_state)
        score += confidence_score
        features_used["confidence_alignment"] = min(1.0, confidence_score / 1.0)

        # 视觉相似性
        visual_score = self._score_visual_similarity(case_item, query_state)
        score += visual_score
        features_used["visual_similarity"] = min(1.0, visual_score / 2.0)

        # 时间因素
        temporal_score = self._score_temporal_factors(case_item)
        score += temporal_score
        features_used["temporal_factors"] = min(1.0, temporal_score / 0.5)

        case_item["_retrieval_score"] = score
        case_item["_score_features"] = features_used
        return score

    def _score_metadata_similarity(self, case_item: Dict[str, Any], query_state: CaseState) -> float:
        """计算元数据相似性分数"""
        score = 0.0
        case_metadata = case_item.get("metadata", {}) or {}
        query_metadata = query_state.get_metadata()

        # 年龄匹配
        case_age = case_metadata.get("age")
        query_age = query_metadata.get("age")
        if case_age and query_age and abs(case_age - query_age) <= 5:
            score += self.score_weights["metadata_age_match"]

        # 部位匹配
        case_site = str(case_metadata.get("site", "")).lower()
        query_site = str(query_metadata.get("site", "")).lower()
        if case_site and query_site and case_site in query_site or query_site in case_site:
            score += self.score_weights["metadata_site_match"]

        # 病史匹配
        case_history = str(case_metadata.get("history", "")).lower()
        query_history = str(query_metadata.get("history", "")).lower()
        if case_history and query_history:
            common_words = set(case_history.split()) & set(query_history.split())
            if len(common_words) > 2:
                score += self.score_weights["metadata_history_match"]

        return score

    def _score_confidence_alignment(self, case_item: Dict[str, Any], query_state: CaseState) -> float:
        """计算置信度对齐分数"""
        score = 0.0

        # 比较不确定性水平
        case_uncertainty = case_item.get("perception", {}).get("uncertainty_level", "medium")
        query_uncertainty = query_state.get_uncertainty_level()

        if case_uncertainty == query_uncertainty:
            score += self.score_weights["uncertainty_match"]

        # 比较置信度
        case_confidence = case_item.get("final_decision", {}).get("confidence", "medium")
        query_confidence = query_state.retrieval.get("retrieval_summary", {}).get("retrieval_confidence", "medium")

        if case_confidence == query_confidence:
            score += self.score_weights["confidence_match"]

        return score

    def _score_visual_similarity(self, case_item: Dict[str, Any], query_state: CaseState) -> float:
        """计算视觉相似性分数"""
        score = 0.0

        case_cues = set(case_item.get("perception", {}).get("visual_cues", []))
        query_cues = set(query_state.perception.get("visual_cues", []))

        if case_cues and query_cues:
            overlap = len(case_cues & query_cues)
            union = len(case_cues | query_cues)
            if union > 0:
                jaccard = overlap / union
                score += self.score_weights["visual_cues_overlap"] * jaccard

        # 风险线索匹配
        case_risks = set(case_item.get("perception", {}).get("risk_cues", {}).get("malignant_cues", []))
        query_risks = set(query_state.perception.get("risk_cues", {}).get("malignant_cues", []))

        if case_risks and query_risks:
            risk_overlap = len(case_risks & query_risks)
            if risk_overlap > 0:
                score += self.score_weights["risk_cues_match"]

        return score

    def _score_temporal_factors(self, case_item: Dict[str, Any]) -> float:
        """计算时间相关因素"""
        score = 0.0

        # 最近使用奖励
        last_used = case_item.get("last_accessed")
        if last_used:
            # 简单的时效性奖励，可以扩展为更复杂的逻辑
            score += self.score_weights["recency_bonus"]

        # 使用频率惩罚（避免过度依赖热门案例）
        access_count = case_item.get("access_count", 0)
        if access_count > 10:
            score += self.score_weights["frequency_penalty"] * min(5, access_count / 10)

        return score

    def update_from_feedback(
        self,
        case_item: Dict[str, Any],
        helpful_signal: float | bool,
        learning_rate: float = None,
    ) -> None:
        """从检索结果的反馈中学习"""
        lr = learning_rate or self.learning_rate
        features = case_item.get("_score_features", {})
        signal = 1.0 if helpful_signal is True else -1.0 if helpful_signal is False else float(helpful_signal)
        signal = max(-1.0, min(1.0, signal))

        for feature_name, feature_value in features.items():
            if feature_name in self.feature_importance:
                # 如果这个案例有帮助，增加相应特征的权重
                # 如果没有帮助，减少权重
                adjustment = lr * feature_value * signal

                if self.use_adam:
                    self._adam_update(f"importance_{feature_name}", adjustment)
                    # 同时调整具体的打分权重
                    related_weights = self._get_related_weights(feature_name)
                    for weight_name in related_weights:
                        self._adam_update(f"weight_{weight_name}", adjustment * 0.1)
                else:
                    self.feature_importance[feature_name] += adjustment
                    # 同时调整具体的打分权重
                    related_weights = self._get_related_weights(feature_name)
                    for weight_name in related_weights:
                        if weight_name in self.score_weights:
                            self.score_weights[weight_name] += adjustment * 0.1  # 更保守的调整

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
        if param_name.startswith("importance_"):
            feature_name = param_name[len("importance_"):]
            if feature_name in self.feature_importance:
                self.feature_importance[feature_name] += self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)
        elif param_name.startswith("weight_"):
            weight_name = param_name[len("weight_"):]
            if weight_name in self.score_weights:
                self.score_weights[weight_name] += self.learning_rate * m_hat / (v_hat**0.5 + self.epsilon)

    def _get_related_weights(self, feature_name: str) -> List[str]:
        """获取与特征相关的权重名称"""
        mapping = {
            "disease_match": ["disease_exact_match", "disease_partial_match"],
            "metadata_similarity": ["metadata_age_match", "metadata_site_match", "metadata_history_match"],
            "confidence_alignment": ["confidence_match", "uncertainty_match"],
            "visual_similarity": ["visual_cues_overlap", "risk_cues_match"],
            "temporal_factors": ["recency_bonus", "frequency_penalty"],
        }
        return mapping.get(feature_name, [])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "score_weights": {k: round(v, 4) for k, v in self.score_weights.items()},
            "feature_importance": {k: round(v, 4) for k, v in self.feature_importance.items()},
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if "score_weights" in payload:
            self.score_weights.update(payload["score_weights"])
        if "feature_importance" in payload:
            self.feature_importance.update(payload["feature_importance"])


class ExperienceRetriever:
    def __init__(
        self,
        bank: ExperienceBank,
        reranker: UtilityAwareExperienceReranker | None = None,
        scorer: LearnableRetrievalScorer | None = None,
    ) -> None:
        self.bank = bank
        self.reranker = reranker
        self.scorer = scorer or LearnableRetrievalScorer()

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
        # 使用可学习的打分器
        return self.scorer.score_raw_case(item, state)

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

    def update_from_feedback(self, retrieved_cases: List[Dict[str, Any]], helpful_signal: float | bool) -> None:
        """从检索结果的反馈中更新打分器"""
        for case_item in retrieved_cases:
            if "_score_features" in case_item:
                self.scorer.update_from_feedback(case_item, helpful_signal)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scorer": self.scorer.to_dict(),
        }

    def load_state(self, payload: Dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        if "scorer" in payload:
            self.scorer.load_state(payload["scorer"])
