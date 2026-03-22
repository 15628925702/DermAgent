#!/usr/bin/env python3
"""
对比脚本：Agent+Qwen vs 直接Qwen
用于评估Agent框架相对于直接VLM推理的性能提升
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import csv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.run_eval import run_evaluation, load_pad_ufes20_cases


class QwenDirectInference:
    """直接Qwen VLM推理模块"""

    def __init__(self):
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
                'diagnosis': 'UNKNOWN',
                'confidence': 0.0,
                'error': 'Qwen client not available'
            }

        try:
            image_path = case.get('image_path', '')
            metadata = case.get('metadata', {})

            # 构建提示
            prompt = self._build_prompt(case, metadata)

            # 调用Qwen VLM
            response = self.client.analyze_image_with_metadata(
                image_path=image_path,
                prompt=prompt,
                metadata=metadata
            )

            # 解析诊断结果
            diagnosis = self._parse_diagnosis(response)

            return {
                'diagnosis': diagnosis,
                'confidence': self._extract_confidence(response),
                'reasoning': response.get('reasoning', ''),
                'error': None
            }

        except Exception as e:
            return {
                'diagnosis': 'UNKNOWN',
                'confidence': 0.0,
                'error': str(e)
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

    def _build_prompt(self, case: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """构建提示"""
        age = metadata.get('age', '未知')
        sex = metadata.get('sex', '未知')
        location = metadata.get('location', metadata.get('region', '未知'))

        prompt = f"""你是一位皮肤病诊断专家。请分析这张皮肤病变图像并提供诊断。

患者信息：
- 年龄: {age}
- 性别: {sex}
- 病变位置: {location}

请从以下类别中选择最可能的诊断：
- MEL (黑色素瘤 - 恶性)
- BCC (基底细胞癌 - 恶性)
- SCC (鳞状细胞癌 - 恶性)
- NEV (痣 - 良性)
- ACK (角化病 - 良性)
- SEK (脂溢性角化病 - 良性)

请提供：
1. 诊断结果
2. 置信度（0-1）
3. 临床推理过程"""

        return prompt

    def _parse_diagnosis(self, response: Dict[str, Any]) -> str:
        """从响应中解析诊断"""
        diagnosis = response.get('diagnosis', 'UNKNOWN')
        
        # 标准化为大写
        diagnosis_upper = str(diagnosis).strip().upper()
        
        # 映射常见变体
        mappings = {
            'MELANOMA': 'MEL',
            'SKIN_CANCER': 'MEL',
            'BASAL_CELL': 'BCC',
            'SQUAMOUS_CELL': 'SCC',
            'NEVUS': 'NEV',
            'MOLE': 'NEV',
            'KERATOSIS': 'ACK',
            'SEBORRHEIC': 'SEK'
        }
        
        for key, val in mappings.items():
            if key in diagnosis_upper:
                return val
        
        # 如果已是标准类别直接返回
        if diagnosis_upper in ['MEL', 'BCC', 'SCC', 'NEV', 'ACK', 'SEK']:
            return diagnosis_upper
        
        return 'UNKNOWN'

    def _extract_confidence(self, response: Dict[str, Any]) -> float:
        """从响应中提取置信度"""
        confidence = response.get('confidence', 0.5)
        try:
            return float(confidence)
        except:
            return 0.5


class ComparisonFramework:
    """对比框架"""

    def __init__(self, test_limit: int = 100):
        self.test_limit = test_limit
        self.results_dir = Path('outputs/comparison')
        self.results_dir.mkdir(exist_ok=True)

        self.qwen_direct = QwenDirectInference()

    def run_full_comparison(self) -> Dict[str, Any]:
        """运行完整对比实验"""
        print("🚀 开始Agent vs 直接Qwen对比实验")
        print(f"测试集大小: {self.test_limit} 案例\n")

        start_time = datetime.now()

        # 加载测试数据
        print("📊 加载测试数据...")
        all_cases = load_pad_ufes20_cases(dataset_root='data/pad_ufes_20', limit=self.test_limit)
        print(f"  已加载 {len(all_cases)} 个测试案例\n")

        # 方案1: Agent框架 (基于run_eval)
        print("🤖 方案1: Agent+Qwen框架")
        agent_results = self._run_agent_pipeline(all_cases)

        # 方案2: 直接Qwen
        print("\n🔍 方案2: 直接调用Qwen VLM")
        qwen_results = self._run_qwen_direct(all_cases)

        # 对比和统计
        print("\n📈 计算对比指标...")
        comparison = self._compute_comparison(all_cases, agent_results, qwen_results)

        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_count': len(all_cases),
            'agent_results': agent_results,
            'qwen_direct_results': qwen_results,
            'comparison': comparison,
            'duration_seconds': (datetime.now() - start_time).total_seconds()
        }

        self._save_report(report)
        self._print_summary(comparison)

        return report

    def _run_agent_pipeline(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """运行Agent推理"""
        print("  启动Agent评估流程...")

        try:
            # 使用现有的run_eval框架
            result = run_evaluation(
                dataset_root='data/pad_ufes_20',
                limit=self.test_limit,
                split_name='test',
                use_retrieval=True,
                use_specialist=True,
                use_reflection=True,
                use_controller=True
            )

            agent_metrics = result.get('metrics', {})
            per_case_results = result.get('per_case', [])

            return {
                'success': True,
                'metrics': agent_metrics,
                'per_case': per_case_results,
                'total_cases': result.get('num_cases', 0),
                'errors': result.get('counts', {}).get('errors', 0)
            }

        except Exception as e:
            print(f"  ❌ Agent流程异常: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }

    def _run_qwen_direct(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """直接Qwen推理"""
        print("  启动Qwen直接推理...")

        try:
            # 批量推理
            predictions = self.qwen_direct.batch_infer(cases)

            # 计算指标
            correct_top1 = 0
            malignant_total = 0
            malignant_hit = 0
            errors = 0

            MALIGNANT_LABELS = {'MEL', 'BCC', 'SCC'}

            per_case_results = []
            for i, (case, pred) in enumerate(zip(cases, predictions)):
                if pred.get('error'):
                    errors += 1
                    continue

                true_label = str(case.get('label', '')).strip().upper()
                pred_label = pred.get('diagnosis', 'UNKNOWN')

                is_correct = pred_label == true_label
                correct_top1 += int(is_correct)

                is_malignant_true = true_label in MALIGNANT_LABELS
                is_malignant_pred = pred_label in MALIGNANT_LABELS

                if is_malignant_true:
                    malignant_total += 1
                    if is_malignant_pred:
                        malignant_hit += 1

                per_case_results.append({
                    'case_id': case.get('file', f'case_{i}'),
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': pred.get('confidence', 0.0),
                    'is_correct': is_correct,
                    'error': pred.get('error')
                })

            total_valid = len(cases) - errors
            accuracy_top1 = correct_top1 / total_valid if total_valid > 0 else 0.0
            malignant_recall = malignant_hit / malignant_total if malignant_total > 0 else 0.0

            return {
                'success': True,
                'metrics': {
                    'accuracy_top1': accuracy_top1,
                    'malignant_recall': malignant_recall
                },
                'per_case': per_case_results,
                'total_cases': len(cases),
                'valid_cases': total_valid,
                'errors': errors
            }

        except Exception as e:
            print(f"  ❌ Qwen直接推理异常: {e}")
            return {
                'success': False,
                'error': str(e),
                'metrics': {}
            }

    def _compute_comparison(self, cases: List[Dict[str, Any]],
                           agent_results: Dict[str, Any],
                           qwen_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算对比指标"""
        agent_metrics = agent_results.get('metrics', {})
        qwen_metrics = qwen_results.get('metrics', {})

        agent_acc = agent_metrics.get('accuracy_top1', 0)
        qwen_acc = qwen_metrics.get('accuracy_top1', 0)

        agent_recall = agent_metrics.get('malignant_recall', 0)
        qwen_recall = qwen_metrics.get('malignant_recall', 0)

        comparison = {
            'accuracy_improvement': {
                'agent': agent_acc,
                'qwen_direct': qwen_acc,
                'improvement_percentage': ((agent_acc - qwen_acc) / max(qwen_acc, 1e-6)) * 100,
                'absolute_improvement': agent_acc - qwen_acc
            },
            'malignant_recall_improvement': {
                'agent': agent_recall,
                'qwen_direct': qwen_recall,
                'improvement_percentage': ((agent_recall - qwen_recall) / max(qwen_recall, 1e-6)) * 100,
                'absolute_improvement': agent_recall - qwen_recall
            },
            'agent_efficiency': {
                'total_cases': agent_results.get('total_cases', 0),
                'errors': agent_results.get('errors', 0),
                'error_rate': agent_results.get('errors', 0) / max(agent_results.get('total_cases', 1), 1)
            },
            'qwen_efficiency': {
                'total_cases': qwen_results.get('total_cases', 0),
                'errors': qwen_results.get('errors', 0),
                'error_rate': qwen_results.get('errors', 0) / max(qwen_results.get('total_cases', 1), 1)
            },
            'conclusion': self._generate_conclusion(agent_acc, qwen_acc, agent_recall, qwen_recall)
        }

        return comparison

    def _generate_conclusion(self, agent_acc: float, qwen_acc: float,
                           agent_recall: float, qwen_recall: float) -> str:
        """生成对比结论"""
        if agent_acc > qwen_acc and agent_recall > qwen_recall:
            improvement = ((agent_acc - qwen_acc) + (agent_recall - qwen_recall)) / 2
            return f"✅ Agent框架全面优于直接Qwen（平均提升 {improvement*100:.1f}%）"
        elif agent_acc > qwen_acc:
            return f"✅ Agent框架在准确率上优于直接Qwen（+{(agent_acc-qwen_acc)*100:.1f}%）"
        elif agent_recall > qwen_recall:
            return f"✅ Agent框架在恶性召回上优于直接Qwen（+{(agent_recall-qwen_recall)*100:.1f}%）"
        else:
            return "⚠️ 直接Qwen表现相近或更优，Agent架构优势需进一步优化"

    def _save_report(self, report: Dict[str, Any]):
        """保存对比报告"""
        # 按时间创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存JSON报告
        json_file = self.results_dir / f"comparison_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  💾 JSON报告: {json_file}")

        # 保存CSV对比表
        csv_file = self.results_dir / f"comparison_metrics_{timestamp}.csv"
        self._save_comparison_csv(report, csv_file)
        print(f"  📊 对比表格: {csv_file}")

        # 保存文本摘要
        txt_file = self.results_dir / f"comparison_summary_{timestamp}.txt"
        self._save_summary_txt(report, txt_file)
        print(f"  📄 摘要报告: {txt_file}")

    def _save_comparison_csv(self, report: Dict[str, Any], filepath: Path):
        """保存对比CSV"""
        comparison = report.get('comparison', {})

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # 标题
            writer.writerow(['指标', 'Agent+Qwen', '直接Qwen', '优势'])

            # 准确率
            acc_agent = comparison.get('accuracy_improvement', {}).get('agent', 0)
            acc_qwen = comparison.get('accuracy_improvement', {}).get('qwen_direct', 0)
            acc_imp = comparison.get('accuracy_improvement', {}).get('improvement_percentage', 0)
            writer.writerow(['Top1准确率', f'{acc_agent:.4f}', f'{acc_qwen:.4f}', f'+{acc_imp:.1f}%'])

            # 恶性召回
            recall_agent = comparison.get('malignant_recall_improvement', {}).get('agent', 0)
            recall_qwen = comparison.get('malignant_recall_improvement', {}).get('qwen_direct', 0)
            recall_imp = comparison.get('malignant_recall_improvement', {}).get('improvement_percentage', 0)
            writer.writerow(['恶性症状召回', f'{recall_agent:.4f}', f'{recall_qwen:.4f}', f'+{recall_imp:.1f}%'])

            # 错误率
            agent_error_rate = comparison.get('agent_efficiency', {}).get('error_rate', 0)
            qwen_error_rate = comparison.get('qwen_efficiency', {}).get('error_rate', 0)
            writer.writerow(['错误率', f'{agent_error_rate:.4f}', f'{qwen_error_rate:.4f}', 
                           '✓' if agent_error_rate < qwen_error_rate else '✗'])

    def _save_summary_txt(self, report: Dict[str, Any], filepath: Path):
        """保存文本总结"""
        comparison = report.get('comparison', {})

        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Agent+Qwen vs 直接Qwen 对比报告\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"生成时间: {report.get('timestamp')}\n")
            f.write(f"测试集规模: {report.get('test_count')} 案例\n")
            f.write(f"实验耗时: {report.get('duration_seconds'):.1f} 秒\n\n")

            f.write("📊 核心指标对比\n")
            f.write("-" * 60 + "\n")

            acc_imp = comparison.get('accuracy_improvement', {})
            f.write(f"Top1准确率:\n")
            f.write(f"  - Agent+Qwen: {acc_imp.get('agent', 0):.4f}\n")
            f.write(f"  - 直接Qwen:   {acc_imp.get('qwen_direct', 0):.4f}\n")
            f.write(f"  - 改进:       {acc_imp.get('improvement_percentage', 0):.1f}%\n\n")

            recall_imp = comparison.get('malignant_recall_improvement', {})
            f.write(f"恶性症状召回率:\n")
            f.write(f"  - Agent+Qwen: {recall_imp.get('agent', 0):.4f}\n")
            f.write(f"  - 直接Qwen:   {recall_imp.get('qwen_direct', 0):.4f}\n")
            f.write(f"  - 改进:       {recall_imp.get('improvement_percentage', 0):.1f}%\n\n")

            f.write("🎯 结论\n")
            f.write("-" * 60 + "\n")
            f.write(comparison.get('conclusion', '') + "\n")

    def _print_summary(self, comparison: Dict[str, Any]):
        """打印总结"""
        print("\n" + "=" * 60)
        print("📊 对比结果汇总")
        print("=" * 60)

        acc_imp = comparison.get('accuracy_improvement', {})
        print(f"\n🎯 Top1准确率:")
        print(f"  Agent+Qwen: {acc_imp.get('agent', 0):.4f}")
        print(f"  直接Qwen:   {acc_imp.get('qwen_direct', 0):.4f}")
        print(f"  改进:       {acc_imp.get('improvement_percentage', 0):+.1f}%")

        recall_imp = comparison.get('malignant_recall_improvement', {})
        print(f"\n🔴 恶性症状召回率:")
        print(f"  Agent+Qwen: {recall_imp.get('agent', 0):.4f}")
        print(f"  直接Qwen:   {recall_imp.get('qwen_direct', 0):.4f}")
        print(f"  改进:       {recall_imp.get('improvement_percentage', 0):+.1f}%")

        print(f"\n✅ 结论: {comparison.get('conclusion', '')}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent vs 直接Qwen对比实验")
    parser.add_argument("--test-limit", type=int, default=100, help="测试集大小")
    args = parser.parse_args()

    framework = ComparisonFramework(test_limit=args.test_limit)
    report = framework.run_full_comparison()

    print("✅ 对比实验完成！")
    print(f"   结果已保存到: {framework.results_dir}")
