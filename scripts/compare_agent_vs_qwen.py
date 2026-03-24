#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import LearnableSkillController
from agent.evidence_calibrator import LearnableEvidenceCalibrator
from agent.final_scorer import LearnableFinalScorer
from agent.rule_scorer import LearnableRuleScorer
from agent.run_agent import run_agent
from datasets.splits import load_or_create_split_manifest, select_split_cases
from evaluation.run_eval import load_dataset_cases, normalize_dataset_type, stratified_subsample_cases
from integrations.openai_client import OpenAICompatClient, OpenAIClient
from memory.controller_store import load_controller_checkpoint
from memory.experience_bank import ExperienceBank
from memory.experience_reranker import UtilityAwareExperienceReranker
from memory.skill_index import build_default_skill_index


VALID_LABELS = {"MEL", "BCC", "SCC", "NEV", "ACK", "SEK"}
MALIGNANT_LABELS = {"MEL", "BCC", "SCC"}


def _norm_label(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def _extract_top_k_labels(final_decision: Dict[str, Any], top_n: int) -> List[str]:
    labels: List[str] = []
    for item in (final_decision or {}).get("top_k", [])[:top_n]:
        if isinstance(item, dict):
            label = _norm_label(item.get("name"))
        else:
            label = _norm_label(item)
        if label:
            labels.append(label)
    return labels


class QwenDirectInference:
    """Direct VLM baseline using the same OpenAI-compatible server."""

    def __init__(self) -> None:
        try:
            self.client = OpenAIClient()
            self.available = True
        except Exception as exc:  # pragma: no cover - runtime dependent
            print(f"初始化 Direct Qwen 客户端失败: {exc}")
            self.client = None
            self.available = False

    def infer_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available or self.client is None:
            return {
                "diagnosis": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "",
                "raw_text": "",
                "raw_response": {},
                "error": "Qwen client not available",
            }

        try:
            raw_response = self.client.infer_derm_direct_diagnosis(
                image_path=case.get("image_path", ""),
                metadata=case.get("metadata", {}) or {},
            )
            parsed_response, raw_text = self._normalize_response(raw_response)
            diagnosis = self._parse_diagnosis(parsed_response, raw_text)
            confidence = self._extract_confidence(parsed_response, raw_text)
            return {
                "diagnosis": diagnosis,
                "confidence": confidence,
                "reasoning": parsed_response.get("reasoning", raw_text),
                "raw_text": raw_text,
                "raw_response": parsed_response,
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - runtime dependent
            return {
                "diagnosis": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "",
                "raw_text": "",
                "raw_response": {},
                "error": str(exc),
            }

    def batch_infer(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx, case in enumerate(cases):
            if idx % 10 == 0:
                print(f"  Direct Qwen 推理进度: {idx}/{len(cases)}")
            results.append(self.infer_case(case))
        return results

    def _normalize_response(self, raw_response: Any) -> Tuple[Dict[str, Any], str]:
        if isinstance(raw_response, dict):
            return raw_response, json.dumps(raw_response, ensure_ascii=False)

        raw_text = str(raw_response).strip() if raw_response is not None else ""
        if not raw_text:
            return {}, ""

        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            raw_text = fenced.group(1).strip()

        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                return obj, raw_text
        except Exception:
            pass

        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict):
                    return obj, raw_text
            except Exception:
                pass

        return {}, raw_text

    def _parse_diagnosis(self, response: Dict[str, Any], raw_text: str = "") -> str:
        candidates: List[str] = []

        diagnosis = response.get("diagnosis")
        if diagnosis is not None:
            candidates.append(str(diagnosis))

        top_k = response.get("top_k")
        if isinstance(top_k, list):
            candidates.extend([str(x) for x in top_k if x is not None])

        if raw_text:
            candidates.append(raw_text)

        mappings = {
            "MELANOMA": "MEL",
            "MEL": "MEL",
            "BASAL CELL CARCINOMA": "BCC",
            "BASAL_CELL": "BCC",
            "BCC": "BCC",
            "SQUAMOUS CELL CARCINOMA": "SCC",
            "SQUAMOUS_CELL": "SCC",
            "SCC": "SCC",
            "NEVUS": "NEV",
            "NEVI": "NEV",
            "MOLE": "NEV",
            "NEV": "NEV",
            "ACTINIC KERATOSIS": "ACK",
            "KERATOSIS": "ACK",
            "ACK": "ACK",
            "SEBORRHEIC KERATOSIS": "SEK",
            "SEBORRHEIC": "SEK",
            "SEK": "SEK",
        }

        for text in candidates:
            upper = str(text).strip().upper()
            if upper in VALID_LABELS:
                return upper
            for key, val in mappings.items():
                if key in upper:
                    return val

        return "UNKNOWN"

    def _extract_confidence(self, response: Dict[str, Any], raw_text: str = "") -> float:
        confidence = response.get("confidence")

        if isinstance(confidence, (int, float)):
            return max(0.0, min(1.0, float(confidence)))

        if isinstance(confidence, str):
            lowered = confidence.strip().lower()
            mapping = {"low": 0.33, "medium": 0.66, "high": 0.90}
            if lowered in mapping:
                return mapping[lowered]
            try:
                return max(0.0, min(1.0, float(lowered)))
            except Exception:
                pass

        if raw_text:
            match = re.search(
                r'"confidence"\s*:\s*"?(low|medium|high|0?\.\d+|1(?:\.0+)?)"?',
                raw_text,
                re.IGNORECASE,
            )
            if match:
                value = match.group(1).strip().lower()
                mapping = {"low": 0.33, "medium": 0.66, "high": 0.90}
                if value in mapping:
                    return mapping[value]
                try:
                    return max(0.0, min(1.0, float(value)))
                except Exception:
                    pass

        return 0.5


class ComparisonFramework:
    def __init__(
        self,
        test_limit: int = 100,
        *,
        dataset_type: str | None = None,
        dataset_root: str = "data/pad_ufes_20",
        split_json: str | None = None,
        split_name: str | None = "test",
        seed: int = 42,
        controller_checkpoint: str | None = None,
        bank_state_in: str | None = None,
        online_learning: bool = False,
        use_retrieval: bool = True,
        use_specialist: bool = True,
        use_controller: bool | None = None,
    ) -> None:
        self.test_limit = test_limit
        self.dataset_type = normalize_dataset_type(dataset_type, dataset_root)
        self.dataset_root = dataset_root
        self.split_json = split_json
        self.split_name = split_name
        self.seed = seed
        self.controller_checkpoint = controller_checkpoint
        self.bank_state_in = bank_state_in
        self.online_learning = bool(online_learning)
        self.use_retrieval = bool(use_retrieval)
        self.use_specialist = bool(use_specialist)
        self.use_controller = bool(controller_checkpoint) if use_controller is None else bool(use_controller)
        self.results_dir = Path("outputs/comparison")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.qwen_direct = QwenDirectInference()
        self.direct_model = self._resolve_direct_model_name()
        self.agent_perception_model = self.direct_model
        self.agent_report_model = self.direct_model
        self.agent_base_url = self._resolve_base_url()

    def _resolve_direct_model_name(self) -> str:
        if self.qwen_direct.available and self.qwen_direct.client is not None:
            model_name = str(getattr(self.qwen_direct.client, "model", "")).strip()
            if model_name:
                return model_name
        return OpenAICompatClient().model

    def _resolve_base_url(self) -> str:
        if self.qwen_direct.available and self.qwen_direct.client is not None:
            base_url = str(getattr(self.qwen_direct.client, "base_url", "")).strip()
            if base_url:
                return base_url
        return OpenAICompatClient().base_url

    def _count_items(self, rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in rows:
            values = row.get(key, []) or []
            for value in values:
                name = str(value).strip()
                if not name:
                    continue
                counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def _summarize_module_participation(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "cases_with_memory_recommendations": sum(1 for row in rows if row.get("memory_recommended_skills")),
            "cases_with_rule_recommendations": sum(1 for row in rows if row.get("rule_recommended_skills")),
            "cases_with_applied_rules": sum(1 for row in rows if row.get("applied_rules")),
            "cases_with_hybrid_retention": sum(1 for row in rows if row.get("hybrid_retained_skills")),
            "cases_with_confusion_support": sum(1 for row in rows if row.get("has_confusion_support")),
            "cases_with_supports_top1": sum(1 for row in rows if row.get("supports_top1")),
            "selected_skill_counts": self._count_items(rows, "selected_skills"),
            "rule_selected_skill_counts": self._count_items(rows, "rule_selected_skills"),
            "memory_recommended_skill_counts": self._count_items(rows, "memory_recommended_skills"),
            "rule_recommended_skill_counts": self._count_items(rows, "rule_recommended_skills"),
            "hybrid_retained_skill_counts": self._count_items(rows, "hybrid_retained_skills"),
        }

    def run_full_comparison(self) -> Dict[str, Any]:
        print("🚀 开始 Agent vs 直接 Qwen 对比实验")
        print(f"测试集大小: {self.test_limit} 案例\n")

        started_at = datetime.now()

        print("📊 加载测试数据...")
        all_cases = load_dataset_cases(dataset_type=self.dataset_type, dataset_root=self.dataset_root, limit=None)
        if self.split_name:
            split_path = self.split_json or str(Path("outputs/splits") / f"{Path(self.dataset_root).name}_seed{self.seed}.json")
            split_payload = load_or_create_split_manifest(all_cases, split_path, seed=self.seed)
            all_cases = select_split_cases(all_cases, split_payload, self.split_name)
        all_cases = stratified_subsample_cases(all_cases, self.test_limit, seed=self.seed)
        print(f"  已加载 {len(all_cases)} 个测试案例\n")

        print("🤖 方案1: Agent+Qwen 框架")
        agent_results = self._run_agent_pipeline(all_cases)

        print("\n🔍 方案2: 直接调用 Qwen VLM")
        qwen_results = self._run_qwen_direct(all_cases)

        print("\n📈 计算对比指标...")
        comparison = self._compute_comparison(all_cases, agent_results, qwen_results)

        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_type": self.dataset_type,
            "dataset_root": self.dataset_root,
            "split_json": self.split_json,
            "split_name": self.split_name,
            "seed": self.seed,
            "test_count": len(all_cases),
            "requested_limit": self.test_limit,
            "agent_results": agent_results,
            "qwen_direct_results": qwen_results,
            "comparison": comparison,
            "duration_seconds": (datetime.now() - started_at).total_seconds(),
        }

        self._save_report(report)
        self._print_summary(comparison)
        return report

    def _run_agent_pipeline(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        print("  启动 Agent 评估流程...")

        try:
            bank = ExperienceBank.from_json(self.bank_state_in) if self.bank_state_in else ExperienceBank()
            checkpoint_loaded = False

            if self.controller_checkpoint:
                (
                    skill_index,
                    controller_payload,
                    final_scorer_payload,
                    rule_scorer_payload,
                    evidence_calibrator_payload,
                ) = load_controller_checkpoint(self.controller_checkpoint)
                checkpoint_loaded = True
            else:
                skill_index = build_default_skill_index()
                controller_payload = {}
                final_scorer_payload = {}
                rule_scorer_payload = {}
                evidence_calibrator_payload = {}

            reranker = UtilityAwareExperienceReranker()
            controller = LearnableSkillController(skill_index) if self.use_controller else None
            final_scorer = LearnableFinalScorer() if self.use_controller else None
            rule_scorer = LearnableRuleScorer() if self.use_controller else None
            evidence_calibrator = LearnableEvidenceCalibrator() if self.use_controller else None

            if controller is not None and controller_payload:
                controller.load_state(controller_payload)
            if final_scorer is not None and final_scorer_payload:
                final_scorer.load_state(final_scorer_payload)
            if rule_scorer is not None and rule_scorer_payload:
                rule_scorer.load_state(rule_scorer_payload)
            if evidence_calibrator is not None and evidence_calibrator_payload:
                evidence_calibrator.load_state(evidence_calibrator_payload)

            correct_top1 = 0
            malignant_total = 0
            malignant_hit = 0
            errors = 0
            perception_fallback_cases = 0
            report_fallback_cases = 0
            perception_model_success_cases = 0
            report_model_success_cases = 0
            model_success_cases = 0
            per_case_results: List[Dict[str, Any]] = []

            for idx, case in enumerate(cases):
                if idx % 10 == 0:
                    print(f"  Agent 推理进度: {idx}/{len(cases)}")

                result = run_agent(
                    case=case,
                    bank=bank,
                    skill_index=skill_index,
                    reranker=reranker,
                    learning_components={
                        "controller": controller,
                        "final_scorer": final_scorer,
                        "rule_scorer": rule_scorer,
                        "evidence_calibrator": evidence_calibrator,
                    } if self.use_controller else {},
                    use_retrieval=self.use_retrieval,
                    use_specialist=self.use_specialist,
                    use_reflection=False,
                    use_controller=self.use_controller,
                    use_final_scorer=self.use_controller,
                    update_online=self.online_learning,
                    use_rule_memory=True,
                    enable_rule_compression=True,
                    perception_model=self.agent_perception_model,
                    report_model=self.agent_report_model,
                )

                true_label = _norm_label(case.get("label"))
                final_decision = result.get("final_decision", {}) or {}
                planner = result.get("planner", {}) or {}
                retrieval = result.get("retrieval", {}) or {}
                retrieval_summary = retrieval.get("retrieval_summary", {}) or {}
                aggregator_debug = final_decision.get("aggregator_debug", {}) or {}
                evidence_summary = final_decision.get("evidence_summary", {}) or {}
                pred_label = _norm_label(final_decision.get("final_label") or final_decision.get("diagnosis"))
                top3 = _extract_top_k_labels(final_decision, top_n=3)
                has_error = result.get("error") is not None

                perception = result.get("perception", {}) or {}
                report = result.get("report", {}) or {}
                perception_fallback = bool(perception.get("fallback_reason"))
                report_generation_mode = str(report.get("generation_mode", "")).strip().lower()
                report_fallback = report_generation_mode == "fallback"

                if perception_fallback:
                    perception_fallback_cases += 1
                else:
                    perception_model_success_cases += 1

                if report_fallback:
                    report_fallback_cases += 1
                elif report_generation_mode == "gpt":
                    report_model_success_cases += 1

                if (not perception_fallback) and report_generation_mode == "gpt":
                    model_success_cases += 1

                if has_error:
                    errors += 1

                is_top1_correct = pred_label == true_label
                if not has_error:
                    correct_top1 += int(is_top1_correct)

                is_malignant_true = true_label in MALIGNANT_LABELS
                is_malignant_pred = pred_label in MALIGNANT_LABELS
                if is_malignant_true and not has_error:
                    malignant_total += 1
                    if is_malignant_pred:
                        malignant_hit += 1

                per_case_results.append(
                    {
                        "case_id": case.get("file", f"case_{idx}"),
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "top3": top3,
                        "confidence": final_decision.get("confidence", "low"),
                        "is_top1_correct": is_top1_correct if not has_error else False,
                        "is_top3_correct": true_label in top3 if not has_error else False,
                        "is_malignant_case": is_malignant_true,
                        "malignant_recalled": is_malignant_true and is_malignant_pred and not has_error,
                        "perception_used_model": not perception_fallback,
                        "perception_fallback_reason": perception.get("fallback_reason"),
                        "report_used_model": report_generation_mode == "gpt",
                        "report_generation_mode": report_generation_mode or "unknown",
                        "report_fallback_reason": report.get("fallback_reason"),
                        "selected_skills": result.get("selected_skills", []),
                        "rule_selected_skills": planner.get("rule_selected_skills", []),
                        "controller_selected_skills": planner.get("controller_selected_skills", []),
                        "hybrid_retained_skills": planner.get("hybrid_retained_skills", []),
                        "hybrid_dropped_rule_skills": planner.get("hybrid_dropped_rule_skills", []),
                        "planner_mode": planner.get("planning_mode"),
                        "stop_probability": planner.get("stop_probability"),
                        "retrieval_confidence": retrieval_summary.get("retrieval_confidence"),
                        "supports_top1": retrieval_summary.get("supports_top1", False),
                        "has_confusion_support": retrieval_summary.get("has_confusion_support", False),
                        "memory_consensus_label": retrieval_summary.get("memory_consensus_label"),
                        "memory_recommended_skills": retrieval_summary.get("memory_recommended_skills", []),
                        "rule_recommended_skills": retrieval_summary.get("rule_recommended_skills", []),
                        "recommended_skills": retrieval_summary.get("recommended_skills", []),
                        "applied_rules": (planner.get("flags", {}) or {}).get("applied_rules", []),
                        "retrieval_learning_feedback": retrieval.get("learning_feedback", {}),
                        "aggregator_used_sources": evidence_summary.get("used_sources", []),
                        "aggregator_top_candidates": evidence_summary.get("top_candidates", []),
                        "candidate_features": aggregator_debug.get("candidate_features", {}),
                        "error": result.get("error"),
                    }
                )

            total_valid = len(cases) - errors
            same_model = self.agent_perception_model == self.direct_model == self.agent_report_model
            module_participation = self._summarize_module_participation(per_case_results)

            return {
                "success": True,
                "metrics": {
                    "accuracy_top1": correct_top1 / total_valid if total_valid > 0 else 0.0,
                    "accuracy_top3": sum(1 for row in per_case_results if row.get("is_top3_correct")) / total_valid if total_valid > 0 else 0.0,
                    "malignant_recall": malignant_hit / malignant_total if malignant_total > 0 else 0.0,
                },
                "per_case": per_case_results,
                "total_cases": len(cases),
                "valid_cases": total_valid,
                "errors": errors,
                "bank_stats": bank.stats(),
                "model_info": {
                    "perception_model": self.agent_perception_model,
                    "report_model": self.agent_report_model,
                    "base_url": self.agent_base_url,
                    "controller_enabled": self.use_controller,
                    "checkpoint_loaded": checkpoint_loaded,
                    "checkpoint_path": self.controller_checkpoint,
                },
                "model_usage": {
                    "model_success_cases": model_success_cases,
                    "perception_model_success_cases": perception_model_success_cases,
                    "report_model_success_cases": report_model_success_cases,
                    "perception_fallback_cases": perception_fallback_cases,
                    "report_fallback_cases": report_fallback_cases,
                },
                "module_participation": module_participation,
                "runtime_flags": {
                    "use_retrieval": self.use_retrieval,
                    "use_specialist": self.use_specialist,
                    "use_controller": self.use_controller,
                    "update_online": self.online_learning,
                    "frozen_benchmark": not self.online_learning,
                    "same_model_as_direct": same_model,
                },
            }
        except Exception as exc:
            print(f"  Agent 评估失败: {exc}")
            return {
                "success": False,
                "error": str(exc),
                "metrics": {},
                "per_case": [],
                "total_cases": len(cases),
                "valid_cases": 0,
                "errors": len(cases),
            }

    def _run_qwen_direct(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        print("  启动 Qwen 直接推理...")

        try:
            predictions = self.qwen_direct.batch_infer(cases)
            correct_top1 = 0
            malignant_total = 0
            malignant_hit = 0
            errors = 0
            per_case_results: List[Dict[str, Any]] = []

            for idx, (case, pred) in enumerate(zip(cases, predictions)):
                true_label = _norm_label(case.get("label"))
                pred_label = _norm_label(pred.get("diagnosis"))
                has_error = pred.get("error") is not None
                top3 = []
                raw_top_k = pred.get("raw_response", {}).get("top_k")
                if isinstance(raw_top_k, list):
                    top3 = [_norm_label(item) for item in raw_top_k[:3] if _norm_label(item)]
                elif pred_label:
                    top3 = [pred_label]

                if has_error:
                    errors += 1
                if not has_error and pred_label == true_label:
                    correct_top1 += 1

                is_malignant_true = true_label in MALIGNANT_LABELS
                is_malignant_pred = pred_label in MALIGNANT_LABELS
                if is_malignant_true and not has_error:
                    malignant_total += 1
                    if is_malignant_pred:
                        malignant_hit += 1

                per_case_results.append(
                    {
                        "case_id": case.get("file", f"case_{idx}"),
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "top3": top3,
                        "confidence": pred.get("confidence", 0.0),
                        "is_top1_correct": (pred_label == true_label) if not has_error else False,
                        "is_top3_correct": (true_label in top3) if not has_error else False,
                        "is_malignant_case": is_malignant_true,
                        "malignant_recalled": is_malignant_true and is_malignant_pred and not has_error,
                        "reasoning": pred.get("reasoning", ""),
                        "error": pred.get("error"),
                    }
                )

            total_valid = len(cases) - errors
            return {
                "success": True,
                "metrics": {
                    "accuracy_top1": correct_top1 / total_valid if total_valid > 0 else 0.0,
                    "accuracy_top3": sum(1 for row in per_case_results if row.get("is_top3_correct")) / total_valid if total_valid > 0 else 0.0,
                    "malignant_recall": malignant_hit / malignant_total if malignant_total > 0 else 0.0,
                },
                "per_case": per_case_results,
                "total_cases": len(cases),
                "valid_cases": total_valid,
                "errors": errors,
                "model_info": {
                    "direct_model": self.direct_model,
                    "base_url": self.agent_base_url,
                },
            }
        except Exception as exc:
            print(f"  Direct Qwen 评估失败: {exc}")
            return {
                "success": False,
                "error": str(exc),
                "metrics": {},
                "per_case": [],
                "total_cases": len(cases),
                "valid_cases": 0,
                "errors": len(cases),
            }

    def _safe_relative_improvement(self, better: float, base: float) -> float:
        if abs(base) < 1e-8:
            return 0.0 if abs(better) < 1e-8 else 100.0
        return ((better - base) / abs(base)) * 100.0

    def _compute_comparison(
        self,
        cases: List[Dict[str, Any]],
        agent_results: Dict[str, Any],
        qwen_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_metrics = agent_results.get("metrics", {})
        qwen_metrics = qwen_results.get("metrics", {})
        agent_acc = float(agent_metrics.get("accuracy_top1", 0.0))
        agent_top3 = float(agent_metrics.get("accuracy_top3", 0.0))
        qwen_acc = float(qwen_metrics.get("accuracy_top1", 0.0))
        qwen_top3 = float(qwen_metrics.get("accuracy_top3", 0.0))
        agent_recall = float(agent_metrics.get("malignant_recall", 0.0))
        qwen_recall = float(qwen_metrics.get("malignant_recall", 0.0))

        agent_case_ids = [str(item.get("case_id", "")).strip() for item in agent_results.get("per_case", [])]
        qwen_case_ids = [str(item.get("case_id", "")).strip() for item in qwen_results.get("per_case", [])]
        requested_case_ids = [str(case.get("file", "")).strip() for case in cases]

        sample_alignment = {
            "requested_cases": len(cases),
            "agent_total_cases": agent_results.get("total_cases", 0),
            "qwen_total_cases": qwen_results.get("total_cases", 0),
            "agent_matches_requested_order": agent_case_ids == requested_case_ids,
            "qwen_matches_requested_order": qwen_case_ids == requested_case_ids,
        }
        sample_alignment["fully_aligned"] = (
            sample_alignment["agent_total_cases"] == sample_alignment["requested_cases"]
            and sample_alignment["qwen_total_cases"] == sample_alignment["requested_cases"]
            and sample_alignment["agent_matches_requested_order"]
            and sample_alignment["qwen_matches_requested_order"]
        )

        fairness = {
            "same_model_as_direct": agent_results.get("runtime_flags", {}).get("same_model_as_direct", False),
            "frozen_benchmark": agent_results.get("runtime_flags", {}).get("frozen_benchmark", False),
        }

        return {
            "accuracy_improvement": {
                "agent": agent_acc,
                "qwen_direct": qwen_acc,
                "improvement_percentage": self._safe_relative_improvement(agent_acc, qwen_acc),
                "absolute_improvement": agent_acc - qwen_acc,
            },
            "top3_improvement": {
                "agent": agent_top3,
                "qwen_direct": qwen_top3,
                "improvement_percentage": self._safe_relative_improvement(agent_top3, qwen_top3),
                "absolute_improvement": agent_top3 - qwen_top3,
            },
            "malignant_recall_improvement": {
                "agent": agent_recall,
                "qwen_direct": qwen_recall,
                "improvement_percentage": self._safe_relative_improvement(agent_recall, qwen_recall),
                "absolute_improvement": agent_recall - qwen_recall,
            },
            "agent_efficiency": {
                "total_cases": agent_results.get("total_cases", 0),
                "errors": agent_results.get("errors", 0),
                "error_rate": agent_results.get("errors", 0) / max(agent_results.get("total_cases", 1), 1),
            },
            "qwen_efficiency": {
                "total_cases": qwen_results.get("total_cases", 0),
                "errors": qwen_results.get("errors", 0),
                "error_rate": qwen_results.get("errors", 0) / max(qwen_results.get("total_cases", 1), 1),
            },
            "sample_alignment": sample_alignment,
            "model_info": {
                "agent": agent_results.get("model_info", {}),
                "qwen_direct": qwen_results.get("model_info", {}),
            },
            "model_usage": {
                "agent": agent_results.get("model_usage", {}),
            },
            "module_participation": {
                "agent": agent_results.get("module_participation", {}),
            },
            "fairness": fairness,
            "conclusion": self._generate_conclusion(
                agent_acc=agent_acc,
                qwen_acc=qwen_acc,
                agent_recall=agent_recall,
                qwen_recall=qwen_recall,
            ),
        }

    def _generate_conclusion(
        self,
        *,
        agent_acc: float,
        qwen_acc: float,
        agent_recall: float,
        qwen_recall: float,
    ) -> str:
        if agent_acc > qwen_acc and agent_recall > qwen_recall:
            improvement = ((agent_acc - qwen_acc) + (agent_recall - qwen_recall)) / 2
            return f"✅ Agent 框架全面优于直接 Qwen（平均绝对提升 {improvement * 100:.1f}%）"
        if agent_acc > qwen_acc:
            return f"✅ Agent 框架在准确率上优于直接 Qwen（+{(agent_acc - qwen_acc) * 100:.1f}%）"
        if agent_recall > qwen_recall:
            return f"✅ Agent 框架在恶性召回上优于直接 Qwen（+{(agent_recall - qwen_recall) * 100:.1f}%）"
        if agent_acc == qwen_acc and agent_recall == qwen_recall:
            return "⚠️ Agent 与直接 Qwen 表现持平，需要更多数据或更强建模后再比较。"
        return "⚠️ 直接 Qwen 表现相近或更优，Agent 架构仍需继续优化。"

    def _save_report(self, report: Dict[str, Any]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = self.results_dir / f"comparison_report_{timestamp}.json"
        json_file.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"  💾 JSON 报告: {json_file}")

        csv_file = self.results_dir / f"comparison_metrics_{timestamp}.csv"
        self._save_comparison_csv(report, csv_file)
        print(f"  📊 对比表格: {csv_file}")

        txt_file = self.results_dir / f"comparison_summary_{timestamp}.txt"
        self._save_summary_txt(report, txt_file)
        print(f"  📄 摘要报告: {txt_file}")

    def _save_comparison_csv(self, report: Dict[str, Any], filepath: Path) -> None:
        comparison = report.get("comparison", {})
        acc_imp = comparison.get("accuracy_improvement", {})
        recall_imp = comparison.get("malignant_recall_improvement", {})
        agent_eff = comparison.get("agent_efficiency", {})
        qwen_eff = comparison.get("qwen_efficiency", {})
        fairness = comparison.get("fairness", {})
        alignment = comparison.get("sample_alignment", {})

        with filepath.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["指标", "Agent+Qwen", "直接Qwen", "补充信息"])
            writer.writerow(["Top1准确率", f"{acc_imp.get('agent', 0):.4f}", f"{acc_imp.get('qwen_direct', 0):.4f}", f"{acc_imp.get('improvement_percentage', 0):+.1f}%"])
            writer.writerow(["恶性召回率", f"{recall_imp.get('agent', 0):.4f}", f"{recall_imp.get('qwen_direct', 0):.4f}", f"{recall_imp.get('improvement_percentage', 0):+.1f}%"])
            writer.writerow(["错误率", f"{agent_eff.get('error_rate', 0):.4f}", f"{qwen_eff.get('error_rate', 0):.4f}", "Agent更低" if agent_eff.get("error_rate", 0) < qwen_eff.get("error_rate", 0) else "Qwen更低或持平"])
            writer.writerow(["样本完全对齐", str(alignment.get("fully_aligned", False)), "", ""])
            writer.writerow(["同模公平性", str(fairness.get("same_model_as_direct", False)), "", ""])
            writer.writerow(["冻结评估", str(fairness.get("frozen_benchmark", False)), "", ""])

    def _save_summary_txt(self, report: Dict[str, Any], filepath: Path) -> None:
        comparison = report.get("comparison", {})
        acc_imp = comparison.get("accuracy_improvement", {})
        recall_imp = comparison.get("malignant_recall_improvement", {})
        agent_eff = comparison.get("agent_efficiency", {})
        qwen_eff = comparison.get("qwen_efficiency", {})
        alignment = comparison.get("sample_alignment", {})
        model_info = comparison.get("model_info", {})
        agent_model = model_info.get("agent", {})
        qwen_model = model_info.get("qwen_direct", {})
        agent_usage = comparison.get("model_usage", {}).get("agent", {})
        fairness = comparison.get("fairness", {})

        with filepath.open("w", encoding="utf-8") as handle:
            handle.write("=" * 60 + "\n")
            handle.write("Agent+Qwen vs 直接Qwen 对比摘要\n")
            handle.write("=" * 60 + "\n\n")
            handle.write(f"时间: {report.get('timestamp')}\n")
            handle.write(f"样本数: {report.get('test_count', 0)}\n")
            handle.write(f"总耗时: {report.get('duration_seconds', 0.0):.1f} 秒\n\n")

            handle.write("Top1准确率:\n")
            handle.write(f"  Agent+Qwen: {acc_imp.get('agent', 0):.4f}\n")
            handle.write(f"  直接Qwen:   {acc_imp.get('qwen_direct', 0):.4f}\n")
            handle.write(f"  改进:       {acc_imp.get('improvement_percentage', 0):+.1f}%\n\n")

            handle.write("恶性症状召回率:\n")
            handle.write(f"  Agent+Qwen: {recall_imp.get('agent', 0):.4f}\n")
            handle.write(f"  直接Qwen:   {recall_imp.get('qwen_direct', 0):.4f}\n")
            handle.write(f"  改进:       {recall_imp.get('improvement_percentage', 0):+.1f}%\n\n")

            handle.write("错误率:\n")
            handle.write(f"  Agent+Qwen: {agent_eff.get('error_rate', 0):.4f}\n")
            handle.write(f"  直接Qwen:   {qwen_eff.get('error_rate', 0):.4f}\n\n")

            handle.write("样本对齐:\n")
            handle.write(f"  请求样本数: {alignment.get('requested_cases', 0)}\n")
            handle.write(f"  Agent样本数: {alignment.get('agent_total_cases', 0)}\n")
            handle.write(f"  直接Qwen样本数: {alignment.get('qwen_total_cases', 0)}\n")
            handle.write(f"  完全对齐: {alignment.get('fully_aligned', False)}\n\n")

            handle.write("公平性检查:\n")
            handle.write(f"  同模对比: {fairness.get('same_model_as_direct', False)}\n")
            handle.write(f"  冻结评估: {fairness.get('frozen_benchmark', False)}\n\n")

            handle.write("模型调用诊断:\n")
            handle.write(f"  Agent perception model: {agent_model.get('perception_model', 'unknown')}\n")
            handle.write(f"  Agent report model:     {agent_model.get('report_model', 'unknown')}\n")
            handle.write(f"  Direct Qwen model:      {qwen_model.get('direct_model', 'unknown')}\n")
            handle.write(f"  Agent模型完整成功case数: {agent_usage.get('model_success_cases', 0)}\n")
            handle.write(f"  perception成功case数:   {agent_usage.get('perception_model_success_cases', 0)}\n")
            handle.write(f"  report成功case数:       {agent_usage.get('report_model_success_cases', 0)}\n")
            handle.write(f"  perception fallback数:  {agent_usage.get('perception_fallback_cases', 0)}\n")
            handle.write(f"  report fallback数:      {agent_usage.get('report_fallback_cases', 0)}\n\n")

            handle.write("结论:\n")
            handle.write(comparison.get("conclusion", "") + "\n")

    def _print_summary(self, comparison: Dict[str, Any]) -> None:
        acc_imp = comparison.get("accuracy_improvement", {})
        recall_imp = comparison.get("malignant_recall_improvement", {})
        agent_eff = comparison.get("agent_efficiency", {})
        qwen_eff = comparison.get("qwen_efficiency", {})
        alignment = comparison.get("sample_alignment", {})
        model_info = comparison.get("model_info", {})
        agent_model = model_info.get("agent", {})
        qwen_model = model_info.get("qwen_direct", {})
        agent_usage = comparison.get("model_usage", {}).get("agent", {})
        fairness = comparison.get("fairness", {})

        print("\n" + "=" * 60)
        print("📊 对比结果汇总")
        print("=" * 60)

        print("\n🎯 Top1准确率:")
        print(f"  Agent+Qwen: {acc_imp.get('agent', 0):.4f}")
        print(f"  直接Qwen:   {acc_imp.get('qwen_direct', 0):.4f}")
        print(f"  改进:       {acc_imp.get('improvement_percentage', 0):+.1f}%")

        print("\n🔴 恶性症状召回率:")
        print(f"  Agent+Qwen: {recall_imp.get('agent', 0):.4f}")
        print(f"  直接Qwen:   {recall_imp.get('qwen_direct', 0):.4f}")
        print(f"  改进:       {recall_imp.get('improvement_percentage', 0):+.1f}%")

        print("\n⚠️ 错误率:")
        print(f"  Agent+Qwen: {agent_eff.get('error_rate', 0):.4f}")
        print(f"  直接Qwen:   {qwen_eff.get('error_rate', 0):.4f}")

        print("\n🧪 样本对齐:")
        print(f"  请求样本数: {alignment.get('requested_cases', 0)}")
        print(f"  Agent样本数: {alignment.get('agent_total_cases', 0)}")
        print(f"  直接Qwen样本数: {alignment.get('qwen_total_cases', 0)}")
        print(f"  完全对齐: {alignment.get('fully_aligned', False)}")

        print("\n⚖️ 公平性检查:")
        print(f"  同模对比:   {fairness.get('same_model_as_direct', False)}")
        print(f"  冻结评估:   {fairness.get('frozen_benchmark', False)}")

        print("\n🧠 模型调用诊断:")
        print(f"  Agent perception model: {agent_model.get('perception_model', 'unknown')}")
        print(f"  Agent report model:     {agent_model.get('report_model', 'unknown')}")
        print(f"  Direct Qwen model:      {qwen_model.get('direct_model', 'unknown')}")
        print(f"  Agent模型完整成功case数: {agent_usage.get('model_success_cases', 0)}")
        print(f"  perception成功case数:   {agent_usage.get('perception_model_success_cases', 0)}")
        print(f"  report成功case数:       {agent_usage.get('report_model_success_cases', 0)}")
        print(f"  perception fallback数:  {agent_usage.get('perception_fallback_cases', 0)}")
        print(f"  report fallback数:      {agent_usage.get('report_fallback_cases', 0)}")

        print(f"\n{comparison.get('conclusion', '')}")
        print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DermAgent against direct Qwen VLM.")
    parser.add_argument("--test-limit", type=int, default=100, help="Number of cases to compare.")
    parser.add_argument("--dataset-type", default=None, choices=["pad_ufes20", "pad_ufes_20", "ham10000"])
    parser.add_argument("--dataset-root", default="data/pad_ufes_20")
    parser.add_argument("--split-json", default=None)
    parser.add_argument("--split-name", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--controller-checkpoint", default=None, help="Checkpoint for controller/final scorer/rule scorer/evidence calibrator.")
    parser.add_argument("--bank-state-in", default=None, help="Optional experience bank checkpoint.")
    parser.add_argument("--online-learning", action="store_true", help="Enable online updates during compare. Off by default for frozen benchmarking.")
    parser.add_argument("--disable-retrieval", action="store_true", help="Disable retrieval in the agent branch.")
    parser.add_argument("--disable-specialist", action="store_true", help="Disable specialist skills in the agent branch.")
    parser.add_argument("--enable-controller", action="store_true", help="Force enable learned controller/scorers.")
    parser.add_argument("--disable-controller", action="store_true", help="Force disable learned controller/scorers.")
    args = parser.parse_args()

    use_controller = None
    if args.enable_controller:
        use_controller = True
    if args.disable_controller:
        use_controller = False

    framework = ComparisonFramework(
        test_limit=args.test_limit,
        dataset_type=args.dataset_type,
        dataset_root=args.dataset_root,
        split_json=args.split_json,
        split_name=args.split_name,
        seed=args.seed,
        controller_checkpoint=args.controller_checkpoint,
        bank_state_in=args.bank_state_in,
        online_learning=args.online_learning,
        use_retrieval=not args.disable_retrieval,
        use_specialist=not args.disable_specialist,
        use_controller=use_controller,
    )
    framework.run_full_comparison()

    print("✅ 对比实验完成！")
    print(f"   结果已保存到: {framework.results_dir}")


if __name__ == "__main__":
    main()
