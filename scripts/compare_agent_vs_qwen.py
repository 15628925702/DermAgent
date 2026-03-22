#!/usr/bin/env python3
"""
对比脚本：Agent+Qwen vs 直接Qwen
用于评估Agent框架相对于直接VLM推理的性能提升
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.controller import LearnableSkillController
from agent.final_scorer import LearnableFinalScorer
from agent.rule_scorer import LearnableRuleScorer
from agent.run_agent import run_agent
from evaluation.run_eval import load_pad_ufes20_cases
from integrations.openai_client import OpenAICompatClient
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
        label = _norm_label(item.get("name")) if isinstance(item, dict) else _norm_label(item)
        if label:
            labels.append(label)
    return labels


class QwenDirectInference:
    """直接Qwen VLM推理模块"""

    def __init__(self) -> None:
        """初始化Qwen推理器"""
        try:
            from integrations.openai_client import OpenAIClient

            self.client = OpenAIClient()
            self.available = True
        except Exception as e:
            print(f"⚠️ Qwen客户端初始化失败: {e}")
            self.available = False

    def infer_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """直接用Qwen推理单个案例"""
        if not self.available:
            return {
                "diagnosis": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "",
                "raw_text": "",
                "raw_response": {},
                "error": "Qwen client not available",
            }

        try:
            image_path = case.get("image_path", "")
            metadata = case.get("metadata", {}) or {}

            raw_response = self.client.infer_derm_direct_diagnosis(
                image_path=image_path,
                metadata=metadata,
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

        except Exception as e:
            return {
                "diagnosis": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": "",
                "raw_text": "",
                "raw_response": {},
                "error": str(e),
            }

    def batch_infer(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量推理"""
        results = []
        for i, case in enumerate(cases):
            if i % 10 == 0:
                print(f"  Qwen直接推理进度: {i}/{len(cases)}")
            result = self.infer_case(case)
            results.append(result)
        return results

    def _normalize_response(self, raw_response: Any) -> tuple[Dict[str, Any], str]:
        """
        兼容多种返回格式：
        1. 已经是 dict
        2. JSON 字符串
        3. 带 ```json ... ``` 的字符串
        4. 普通文本
        """
        if isinstance(raw_response, dict):
            return raw_response, json.dumps(raw_response, ensure_ascii=False)

        raw_text = str(raw_response).strip() if raw_response is not None else ""

        if not raw_text:
            return {}, ""

        # 去掉 markdown fence
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            raw_text = fenced.group(1).strip()

        # 先直接按 JSON 解析
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, dict):
                return obj, raw_text
        except Exception:
            pass

        # 从文本中提取第一个 JSON 对象
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            json_candidate = match.group(0)
            try:
                obj = json.loads(json_candidate)
                if isinstance(obj, dict):
                    return obj, raw_text
            except Exception:
                pass

        # 实在不行，当普通文本处理
        return {}, raw_text

    def _parse_diagnosis(self, response: Dict[str, Any], raw_text: str = "") -> str:
        """从响应中解析诊断"""
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
        """从响应中提取置信度"""
        confidence = response.get("confidence", None)

        if isinstance(confidence, (int, float)):
            conf = float(confidence)
            return max(0.0, min(1.0, conf))

        if isinstance(confidence, str):
            conf_str = confidence.strip().lower()
            mapping = {
                "low": 0.33,
                "medium": 0.66,
                "high": 0.90,
            }
            if conf_str in mapping:
                return mapping[conf_str]
            try:
                conf = float(conf_str)
                return max(0.0, min(1.0, conf))
            except Exception:
                pass

        if raw_text:
            m = re.search(r'"confidence"\s*:\s*"?(low|medium|high|0?\.\d+|1(?:\.0+)?)"?', raw_text, re.IGNORECASE)
            if m:
                value = m.group(1).strip().lower()
                mapping = {
                    "low": 0.33,
                    "medium": 0.66,
                    "high": 0.90,
                }
                if value in mapping:
                    return mapping[value]
                try:
                    return max(0.0, min(1.0, float(value)))
                except Exception:
                    pass

        return 0.5


class ComparisonFramework:
    """对比框架"""

    def __init__(
        self,
        test_limit: int = 100,
        *,
        controller_checkpoint: str | None = None,
        bank_state_in: str | None = None,
        online_learning: bool = False,
        use_retrieval: bool = True,
        use_specialist: bool = True,
        use_controller: bool | None = None,
    ):
        self.test_limit = test_limit
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
        if self.qwen_direct.available:
            model_name = str(getattr(self.qwen_direct.client, "model", "")).strip()
            if model_name:
                return model_name
        return OpenAICompatClient().model

    def _resolve_base_url(self) -> str:
        if self.qwen_direct.available:
            base_url = str(getattr(self.qwen_direct.client, "base_url", "")).strip()
            if base_url:
                return base_url
        return OpenAICompatClient().base_url

    def run_full_comparison(self) -> Dict[str, Any]:
        """运行完整对比实验"""
        print("🚀 开始Agent vs 直接Qwen对比实验")
        print(f"测试集大小: {self.test_limit} 案例\n")

        start_time = datetime.now()

        print("📊 加载测试数据...")
        all_cases = load_pad_ufes20_cases(
            dataset_root="data/pad_ufes_20",
            limit=self.test_limit,
        )
        print(f"  已加载 {len(all_cases)} 个测试案例\n")

        print("🤖 方案1: Agent+Qwen框架")
        agent_results = self._run_agent_pipeline(all_cases)

        print("\n🔍 方案2: 直接调用Qwen VLM")
        qwen_results = self._run_qwen_direct(all_cases)

        print("\n📈 计算对比指标...")
        comparison = self._compute_comparison(all_cases, agent_results, qwen_results)

        report = {
            "timestamp": datetime.now().isoformat(),
            "test_count": len(all_cases),
            "agent_results": agent_results,
            "qwen_direct_results": qwen_results,
            "comparison": comparison,
            "duration_seconds": (datetime.now() - start_time).total_seconds(),
        }

        self._save_report(report)
        self._print_summary(comparison)
        return report

    def _run_agent_pipeline(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行Agent推理"""
        print("  启动Agent评估流程...")

        try:
            bank = ExperienceBank.from_json(self.bank_state_in) if self.bank_state_in else ExperienceBank()
            checkpoint_loaded = False
            if self.controller_checkpoint:
                skill_index, controller_payload, final_scorer_payload, rule_scorer_payload = load_controller_checkpoint(self.controller_checkpoint)
                checkpoint_loaded = True
            else:
                skill_index = build_default_skill_index()
                controller_payload = {}
                final_scorer_payload = {}
                rule_scorer_payload = {}
            reranker = UtilityAwareExperienceReranker()
            controller = LearnableSkillController(skill_index) if self.use_controller else None
            final_scorer = LearnableFinalScorer() if self.use_controller else None
            rule_scorer = LearnableRuleScorer() if self.use_controller else None
            if controller is not None and controller_payload:
                controller.load_state(controller_payload)
            if final_scorer is not None and final_scorer_payload:
                final_scorer.load_state(final_scorer_payload)
            if rule_scorer is not None and rule_scorer_payload:
                rule_scorer.load_state(rule_scorer_payload)

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

            for i, case in enumerate(cases):
                if i % 10 == 0:
                    print(f"  Agent推理进度: {i}/{len(cases)}")

                result = run_agent(
                    case=case,
                    bank=bank,
                    skill_index=skill_index,
                    reranker=reranker,
                    learning_components={
                        "controller": controller,
                        "final_scorer": final_scorer,
                        "rule_scorer": rule_scorer,
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
                        "case_id": case.get("file", f"case_{i}"),
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
                        "planner_mode": (result.get("planner", {}) or {}).get("planning_mode"),
                        "stop_probability": (result.get("planner", {}) or {}).get("stop_probability"),
                        "error": result.get("error"),
                    }
                )

            total_valid = len(cases) - errors

            return {
                "success": True,
                "metrics": {
                    "accuracy_top1": correct_top1 / total_valid if total_valid > 0 else 0.0,
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
                "runtime_flags": {
                    "use_retrieval": self.use_retrieval,
                    "use_specialist": self.use_specialist,
                    "use_controller": self.use_controller,
                    "update_online": self.online_learning,
                },
            }

        except Exception as e:
            print(f"  ❌ Agent流程异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {},
                "per_case": [],
                "total_cases": len(cases),
                "errors": len(cases),
            }

    def _run_qwen_direct(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """直接Qwen推理"""
        print("  启动Qwen直接推理...")

        try:
            predictions = self.qwen_direct.batch_infer(cases)

            correct_top1 = 0
            malignant_total = 0
            malignant_hit = 0
            errors = 0

            per_case_results = []
            for i, (case, pred) in enumerate(zip(cases, predictions)):
                true_label = str(case.get("label", "")).strip().upper()
                pred_label = str(pred.get("diagnosis", "UNKNOWN")).strip().upper()

                has_error = pred.get("error") is not None
                if has_error:
                    errors += 1

                is_correct = pred_label == true_label
                if not has_error:
                    correct_top1 += int(is_correct)

                is_malignant_true = true_label in MALIGNANT_LABELS
                is_malignant_pred = pred_label in MALIGNANT_LABELS

                if is_malignant_true and not has_error:
                    malignant_total += 1
                    if is_malignant_pred:
                        malignant_hit += 1

                per_case_results.append(
                    {
                        "case_id": case.get("file", f"case_{i}"),
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": pred.get("confidence", 0.0),
                        "is_correct": is_correct if not has_error else False,
                        "raw_text": pred.get("raw_text", ""),
                        "raw_response": pred.get("raw_response", {}),
                        "reasoning": pred.get("reasoning", ""),
                        "error": pred.get("error"),
                    }
                )

            total_valid = len(cases) - errors
            accuracy_top1 = correct_top1 / total_valid if total_valid > 0 else 0.0
            malignant_recall = malignant_hit / malignant_total if malignant_total > 0 else 0.0

            return {
                "success": True,
                "metrics": {
                    "accuracy_top1": accuracy_top1,
                    "malignant_recall": malignant_recall,
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

        except Exception as e:
            print(f"  ❌ Qwen直接推理异常: {e}")
            return {
                "success": False,
                "error": str(e),
                "metrics": {},
                "per_case": [],
                "total_cases": len(cases),
                "valid_cases": 0,
                "errors": len(cases),
            }

    def _safe_relative_improvement(self, better: float, base: float) -> float:
        """避免 baseline 接近 0 时出现夸张百分比"""
        if abs(base) < 1e-8:
            if abs(better) < 1e-8:
                return 0.0
            return 100.0
        return ((better - base) / abs(base)) * 100.0

    def _compute_comparison(
        self,
        cases: List[Dict[str, Any]],
        agent_results: Dict[str, Any],
        qwen_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """计算对比指标"""
        agent_metrics = agent_results.get("metrics", {})
        qwen_metrics = qwen_results.get("metrics", {})

        agent_acc = float(agent_metrics.get("accuracy_top1", 0.0))
        qwen_acc = float(qwen_metrics.get("accuracy_top1", 0.0))

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

        comparison = {
            "accuracy_improvement": {
                "agent": agent_acc,
                "qwen_direct": qwen_acc,
                "improvement_percentage": self._safe_relative_improvement(agent_acc, qwen_acc),
                "absolute_improvement": agent_acc - qwen_acc,
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
                "error_rate": agent_results.get("errors", 0)
                / max(agent_results.get("total_cases", 1), 1),
            },
            "qwen_efficiency": {
                "total_cases": qwen_results.get("total_cases", 0),
                "errors": qwen_results.get("errors", 0),
                "error_rate": qwen_results.get("errors", 0)
                / max(qwen_results.get("total_cases", 1), 1),
            },
            "sample_alignment": sample_alignment,
            "model_info": {
                "agent": agent_results.get("model_info", {}),
                "qwen_direct": qwen_results.get("model_info", {}),
            },
            "model_usage": {
                "agent": agent_results.get("model_usage", {}),
            },
            "conclusion": self._generate_conclusion(
                agent_acc, qwen_acc, agent_recall, qwen_recall
            ),
        }

        return comparison

    def _generate_conclusion(
        self,
        agent_acc: float,
        qwen_acc: float,
        agent_recall: float,
        qwen_recall: float,
    ) -> str:
        """生成对比结论"""
        if agent_acc > qwen_acc and agent_recall > qwen_recall:
            improvement = ((agent_acc - qwen_acc) + (agent_recall - qwen_recall)) / 2
            return f"✅ Agent框架全面优于直接Qwen（平均绝对提升 {improvement * 100:.1f}%）"
        if agent_acc > qwen_acc:
            return f"✅ Agent框架在准确率上优于直接Qwen（+{(agent_acc - qwen_acc) * 100:.1f}%）"
        if agent_recall > qwen_recall:
            return f"✅ Agent框架在恶性召回上优于直接Qwen（+{(agent_recall - qwen_recall) * 100:.1f}%）"
        if agent_acc == qwen_acc and agent_recall == qwen_recall:
            return "⚠️ Agent与直接Qwen表现接近，需扩大测试集进一步验证"
        return "⚠️ 直接Qwen表现相近或更优，Agent架构优势需进一步优化"

    def _save_report(self, report: Dict[str, Any]) -> None:
        """保存对比报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        json_file = self.results_dir / f"comparison_report_{timestamp}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"  💾 JSON报告: {json_file}")

        csv_file = self.results_dir / f"comparison_metrics_{timestamp}.csv"
        self._save_comparison_csv(report, csv_file)
        print(f"  📊 对比表格: {csv_file}")

        txt_file = self.results_dir / f"comparison_summary_{timestamp}.txt"
        self._save_summary_txt(report, txt_file)
        print(f"  📄 摘要报告: {txt_file}")

    def _save_comparison_csv(self, report: Dict[str, Any], filepath: Path) -> None:
        """保存对比CSV"""
        comparison = report.get("comparison", {})

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["指标", "Agent+Qwen", "直接Qwen", "优势"])

            acc_agent = comparison.get("accuracy_improvement", {}).get("agent", 0)
            acc_qwen = comparison.get("accuracy_improvement", {}).get("qwen_direct", 0)
            acc_imp = comparison.get("accuracy_improvement", {}).get("improvement_percentage", 0)
            writer.writerow(["Top1准确率", f"{acc_agent:.4f}", f"{acc_qwen:.4f}", f"{acc_imp:+.1f}%"])

            recall_agent = comparison.get("malignant_recall_improvement", {}).get("agent", 0)
            recall_qwen = comparison.get("malignant_recall_improvement", {}).get("qwen_direct", 0)
            recall_imp = comparison.get("malignant_recall_improvement", {}).get("improvement_percentage", 0)
            writer.writerow(["恶性症状召回", f"{recall_agent:.4f}", f"{recall_qwen:.4f}", f"{recall_imp:+.1f}%"])

            agent_error_rate = comparison.get("agent_efficiency", {}).get("error_rate", 0)
            qwen_error_rate = comparison.get("qwen_efficiency", {}).get("error_rate", 0)
            writer.writerow(
                [
                    "错误率",
                    f"{agent_error_rate:.4f}",
                    f"{qwen_error_rate:.4f}",
                    "✓" if agent_error_rate < qwen_error_rate else "✗",
                ]
            )

    def _save_summary_txt(self, report: Dict[str, Any], filepath: Path) -> None:
        """保存文本总结"""
        comparison = report.get("comparison", {})

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("Agent+Qwen vs 直接Qwen 对比报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"生成时间: {report.get('timestamp')}\n")
            f.write(f"测试集规模: {report.get('test_count')} 案例\n")
            f.write(f"实验耗时: {report.get('duration_seconds'):.1f} 秒\n\n")

            f.write("📊 核心指标对比\n")
            f.write("-" * 60 + "\n")

            acc_imp = comparison.get("accuracy_improvement", {})
            f.write("Top1准确率:\n")
            f.write(f"  - Agent+Qwen: {acc_imp.get('agent', 0):.4f}\n")
            f.write(f"  - 直接Qwen:   {acc_imp.get('qwen_direct', 0):.4f}\n")
            f.write(f"  - 改进:       {acc_imp.get('improvement_percentage', 0):+.1f}%\n\n")

            recall_imp = comparison.get("malignant_recall_improvement", {})
            f.write("恶性症状召回率:\n")
            f.write(f"  - Agent+Qwen: {recall_imp.get('agent', 0):.4f}\n")
            f.write(f"  - 直接Qwen:   {recall_imp.get('qwen_direct', 0):.4f}\n")
            f.write(f"  - 改进:       {recall_imp.get('improvement_percentage', 0):+.1f}%\n\n")

            agent_eff = comparison.get("agent_efficiency", {})
            qwen_eff = comparison.get("qwen_efficiency", {})
            f.write("错误率:\n")
            f.write(f"  - Agent+Qwen: {agent_eff.get('error_rate', 0):.4f}\n")
            f.write(f"  - 直接Qwen:   {qwen_eff.get('error_rate', 0):.4f}\n\n")

            alignment = comparison.get("sample_alignment", {})
            f.write("样本对齐:\n")
            f.write(f"  - 请求样本数: {alignment.get('requested_cases', 0)}\n")
            f.write(f"  - Agent样本数: {alignment.get('agent_total_cases', 0)}\n")
            f.write(f"  - Qwen样本数:  {alignment.get('qwen_total_cases', 0)}\n")
            f.write(f"  - 完全对齐:    {alignment.get('fully_aligned', False)}\n\n")

            model_info = comparison.get("model_info", {})
            agent_model = model_info.get("agent", {})
            qwen_model = model_info.get("qwen_direct", {})
            agent_usage = comparison.get("model_usage", {}).get("agent", {})
            f.write("模型调用诊断:\n")
            f.write(f"  - Agent perception model: {agent_model.get('perception_model', 'unknown')}\n")
            f.write(f"  - Agent report model:     {agent_model.get('report_model', 'unknown')}\n")
            f.write(f"  - Direct Qwen model:      {qwen_model.get('direct_model', 'unknown')}\n")
            f.write(f"  - Agent模型完整成功case数: {agent_usage.get('model_success_cases', 0)}\n")
            f.write(f"  - perception成功case数:   {agent_usage.get('perception_model_success_cases', 0)}\n")
            f.write(f"  - report成功case数:       {agent_usage.get('report_model_success_cases', 0)}\n")
            f.write(f"  - perception fallback数:  {agent_usage.get('perception_fallback_cases', 0)}\n")
            f.write(f"  - report fallback数:      {agent_usage.get('report_fallback_cases', 0)}\n\n")

            f.write("🎯 结论\n")
            f.write("-" * 60 + "\n")
            f.write(comparison.get("conclusion", "") + "\n")

    def _print_summary(self, comparison: Dict[str, Any]) -> None:
        """打印总结"""
        print("\n" + "=" * 60)
        print("📊 对比结果汇总")
        print("=" * 60)

        acc_imp = comparison.get("accuracy_improvement", {})
        print("\n🎯 Top1准确率:")
        print(f"  Agent+Qwen: {acc_imp.get('agent', 0):.4f}")
        print(f"  直接Qwen:   {acc_imp.get('qwen_direct', 0):.4f}")
        print(f"  改进:       {acc_imp.get('improvement_percentage', 0):+.1f}%")

        recall_imp = comparison.get("malignant_recall_improvement", {})
        print("\n🔴 恶性症状召回率:")
        print(f"  Agent+Qwen: {recall_imp.get('agent', 0):.4f}")
        print(f"  直接Qwen:   {recall_imp.get('qwen_direct', 0):.4f}")
        print(f"  改进:       {recall_imp.get('improvement_percentage', 0):+.1f}%")

        agent_eff = comparison.get("agent_efficiency", {})
        qwen_eff = comparison.get("qwen_efficiency", {})
        print("\n⚠️ 错误率:")
        print(f"  Agent+Qwen: {agent_eff.get('error_rate', 0):.4f}")
        print(f"  直接Qwen:   {qwen_eff.get('error_rate', 0):.4f}")

        alignment = comparison.get("sample_alignment", {})
        print("\n🧪 样本对齐:")
        print(f"  请求样本数: {alignment.get('requested_cases', 0)}")
        print(f"  Agent样本数: {alignment.get('agent_total_cases', 0)}")
        print(f"  直接Qwen样本数: {alignment.get('qwen_total_cases', 0)}")
        print(f"  完全对齐: {alignment.get('fully_aligned', False)}")

        model_info = comparison.get("model_info", {})
        agent_model = model_info.get("agent", {})
        qwen_model = model_info.get("qwen_direct", {})
        agent_usage = comparison.get("model_usage", {}).get("agent", {})
        print("\n🧠 模型调用诊断:")
        print(f"  Agent perception model: {agent_model.get('perception_model', 'unknown')}")
        print(f"  Agent report model:     {agent_model.get('report_model', 'unknown')}")
        print(f"  Direct Qwen model:      {qwen_model.get('direct_model', 'unknown')}")
        print(f"  Agent模型完整成功case数: {agent_usage.get('model_success_cases', 0)}")
        print(f"  perception成功case数:   {agent_usage.get('perception_model_success_cases', 0)}")
        print(f"  report成功case数:       {agent_usage.get('report_model_success_cases', 0)}")
        print(f"  perception fallback数:  {agent_usage.get('perception_fallback_cases', 0)}")
        print(f"  report fallback数:      {agent_usage.get('report_fallback_cases', 0)}")

        print(f"\n✅ 结论: {comparison.get('conclusion', '')}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent vs 直接Qwen对比实验")
    parser.add_argument("--test-limit", type=int, default=100, help="测试集大小")
    parser.add_argument("--controller-checkpoint", default=None, help="冻结评估时加载训练好的controller/final_scorer/rule_scorer checkpoint")
    parser.add_argument("--bank-state-in", default=None, help="可选经验库checkpoint")
    parser.add_argument("--online-learning", action="store_true", help="在compare过程中允许在线学习；默认关闭以保证冻结benchmark")
    parser.add_argument("--disable-retrieval", action="store_true", help="关闭retrieval")
    parser.add_argument("--disable-specialist", action="store_true", help="关闭specialist")
    parser.add_argument("--enable-controller", action="store_true", help="即使没有checkpoint也启用controller")
    parser.add_argument("--disable-controller", action="store_true", help="强制关闭controller")
    args = parser.parse_args()

    use_controller = None
    if args.enable_controller:
        use_controller = True
    if args.disable_controller:
        use_controller = False

    framework = ComparisonFramework(
        test_limit=args.test_limit,
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
