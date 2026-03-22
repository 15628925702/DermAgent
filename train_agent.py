#!/usr/bin/env python3
"""
DermAgent 训练脚本 - 使用统一权重管理

支持：
- 增量学习训练
- 权重检查点保存
- 自动评估和baseline对比
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from evaluation.run_eval import load_pad_ufes20_cases
from evaluation.run_eval import run_evaluation
from memory.weights_manager import weights_manager
from memory.skill_index import build_default_skill_index
from agent.run_agent import run_agent


def _extract_top_k_labels(final_decision: Dict[str, Any], top_n: int) -> List[str]:
    """从final_decision中提取前k个标签"""
    labels: List[str] = []
    for item in (final_decision or {}).get("top_k", [])[:top_n]:
        label = item.get("name") if isinstance(item, dict) else str(item)
        if label:
            labels.append(label)
    return labels


def train_epoch(cases: List[Dict[str, Any]], learning_components: Dict[str, Any],
               skill_index: Any, epoch: int, config: Dict[str, Any]) -> Dict[str, Any]:
    """训练一个epoch"""
    print(f"Training epoch {epoch}...")

    results = []
    for i, case in enumerate(cases):
        if i % 50 == 0:
            print(f"  Processing case {i}/{len(cases)}")

        try:
            result = run_agent(
                case=case,
                skill_index=skill_index,
                learning_components=learning_components,
                use_controller=config.get("use_controller", True),
                use_final_scorer=config.get("use_final_scorer", True),
                update_online=True
            )
            results.append(result)
        except Exception as e:
            print(f"  Error processing case {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return {"results": results, "epoch": epoch}


def evaluate_model(cases: List[Dict[str, Any]], learning_components: Dict[str, Any],
                  skill_index: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """评估模型性能"""
    print("Evaluating model...")

    predictions = []
    for case in cases:
        try:
            result = run_agent(
                case=case,
                skill_index=skill_index,
                learning_components=learning_components,
                use_controller=config.get("use_controller", True),
                use_final_scorer=config.get("use_final_scorer", True),
                update_online=False  # 评估时不更新权重
            )

            final_decision = result.get("final_decision", {})
            pred_label = final_decision.get("final_label", "")
            top3 = _extract_top_k_labels(final_decision, top_n=3)

            predictions.append({
                "case_id": case.get("case_id", ""),
                "true_label": case.get("label", ""),
                "predicted_label": pred_label,
                "top3": top3,
                "confidence": final_decision.get("confidence", "low")
            })
        except Exception as e:
            print(f"  Error evaluating case: {e}")
            continue

    # 计算完整指标
    correct_top1 = sum(1 for p in predictions if p["true_label"] == p["predicted_label"])
    correct_top3 = sum(1 for p in predictions if p["true_label"] in p["top3"])
    malignant_cases = [p for p in predictions if p["true_label"] in ["MEL", "BCC", "SCC"]]
    malignant_recall = sum(1 for p in malignant_cases if p["predicted_label"] in ["MEL", "BCC", "SCC"]) / len(malignant_cases) if malignant_cases else 0

    total = len(predictions)
    accuracy = correct_top1 / total if total > 0 else 0
    top3_accuracy = correct_top3 / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "top3_accuracy": top3_accuracy,
        "malignant_recall": malignant_recall,
        "total_cases": total,
        "predictions": predictions
    }


def main():
    parser = argparse.ArgumentParser(description="Train DermAgent")
    parser.add_argument("--run-name", type=str, default="train_run", help="Training run name")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--eval-interval", type=int, default=1, help="Evaluation interval")
    parser.add_argument("--save-interval", type=int, default=1, help="Checkpoint save interval")
    parser.add_argument("--data-split", type=str, default="train", help="Data split to use")
    parser.add_argument("--config", type=str, default="config.ini", help="Configuration file path")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # 创建输出目录
    run_dir = Path(f"outputs/train_runs/{args.run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 首先创建权重管理器实例来获取配置
    from memory.weights_manager import WeightsManager
    weights_manager_instance = WeightsManager(config_file=args.config)
    temp_config = {
        "learning_config": weights_manager_instance.config.to_dict()
    }

    # 加载数据
    print("Loading data...")
    data_limit = temp_config.get("learning_config", {}).get("data_limit")
    if data_limit and str(data_limit).lower() != "null":
        limit = int(data_limit)
    else:
        limit = None  # 使用全数据集
    all_cases = load_pad_ufes20_cases(limit=limit)

    # 简单的数据分割（实际应该使用预定义的分割）
    train_size = int(0.7 * len(all_cases))
    val_size = int(0.2 * len(all_cases))

    # 首先创建权重管理器实例来获取配置
    from memory.weights_manager import WeightsManager
    weights_manager_instance = WeightsManager(config_file=args.config)
    temp_config = {
        "learning_config": weights_manager_instance.config.to_dict()
    }

    # 使用配置中的批次大小，如果没有则使用默认值
    batch_size = temp_config.get("learning_config", {}).get("batch_size", args.batch_size)
    cases = all_cases[:train_size][:batch_size]  # 训练集
    eval_cases = all_cases[train_size:train_size+val_size][:100]  # 验证集

    # 初始化组件
    skill_index = build_default_skill_index()

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        weights_manager_instance = weights_manager  # 使用默认配置
        learning_components = weights_manager_instance.initialize_components(skill_index)
        weights_manager_instance.load_checkpoint(learning_components, args.resume_from)
    else:
        print("Starting fresh training")
        # 使用指定的配置文件
        from memory.weights_manager import WeightsManager
        weights_manager_instance = WeightsManager(config_file=args.config)
        learning_components = weights_manager_instance.initialize_components(skill_index)

    # 从权重管理器获取配置
    config = {
        "use_controller": True,
        "use_final_scorer": True,
        "use_retrieval": True,
        "use_specialist": True,
        "use_reflection": True,
        "learning_config": weights_manager_instance.config.to_dict()
    }

    # 训练历史
    history = {
        "run_name": args.run_name,
        "config": config,
        "epochs": [],
        "best_accuracy": 0.0
    }

    for epoch in range(args.epochs if args.epochs > 0 else config.get("learning_config", {}).get("max_epochs", 5)):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        # 训练
        train_result = train_epoch(cases, learning_components, skill_index, epoch, config)

        # 定期评估
        eval_interval = config.get("learning_config", {}).get("eval_interval", args.eval_interval)
        if epoch % eval_interval == 0:
            eval_result = evaluate_model(eval_cases, learning_components, skill_index, config)
            accuracy = eval_result["accuracy"]
            top3_accuracy = eval_result["top3_accuracy"]
            malignant_recall = eval_result["malignant_recall"]
            print(".3f")

            # 更新历史
            epoch_data = {
                "epoch": epoch,
                "train_cases": len(train_result["results"]),
                "eval_accuracy": accuracy,
                "eval_top3_accuracy": top3_accuracy,
                "eval_malignant_recall": malignant_recall,
                "eval_cases": eval_result["total_cases"]
            }
            history["epochs"].append(epoch_data)

            # 保存最佳模型
            if accuracy > history["best_accuracy"]:
                history["best_accuracy"] = accuracy
                weights_manager.save_checkpoint(
                    learning_components,
                    args.run_name,
                    epoch=epoch,
                    metadata={"accuracy": accuracy, "is_best": True}
                )
                print("  ✓ Saved best model")

        # 定期保存检查点
        save_interval = config.get("learning_config", {}).get("save_interval", args.save_interval)
        if epoch % save_interval == 0:
            weights_manager.save_checkpoint(
                learning_components,
                args.run_name,
                epoch=epoch,
                metadata={"accuracy": history["epochs"][-1]["eval_accuracy"] if history["epochs"] else 0}
            )

    # 保存最终模型和历史
    weights_manager.save_checkpoint(learning_components, args.run_name, metadata=history)
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining completed!")
    print(f"Best accuracy: {history['best_accuracy']:.3f}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()