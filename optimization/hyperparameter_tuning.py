#!/usr/bin/env python3
"""
自动超参数调优系统 - 为A刊发表优化

使用贝叶斯优化和网格搜索结合的策略
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.model_selection import ParameterGrid
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt

@dataclass
class HyperparameterSpace:
    """超参数空间定义"""
    learning_rates: List[float] = None
    weight_decays: List[float] = None
    batch_sizes: List[int] = None
    attention_heads: List[int] = None
    hidden_dims: List[int] = None
    dropout_rates: List[float] = None
    fusion_methods: List[str] = None

    def __post_init__(self):
        if self.learning_rates is None:
            self.learning_rates = [0.005, 0.01, 0.02, 0.05]
        if self.weight_decays is None:
            self.weight_decays = [0.001, 0.01, 0.1]
        if self.batch_sizes is None:
            self.batch_sizes = [100, 200, 500, 1000]
        if self.attention_heads is None:
            self.attention_heads = [4, 8, 12, 16]
        if self.hidden_dims is None:
            self.hidden_dims = [256, 512, 768, 1024]
        if self.dropout_rates is None:
            self.dropout_rates = [0.1, 0.2, 0.3, 0.5]
        if self.fusion_methods is None:
            self.fusion_methods = ['attention', 'gated', 'concat']

class HyperparameterOptimizer:
    """超参数优化器"""

    def __init__(self, space: HyperparameterSpace, n_trials: int = 50):
        self.space = space
        self.n_trials = n_trials
        self.results = []
        self.best_params = None
        self.best_score = float('-inf')

    def objective(self, trial: optuna.Trial) -> float:
        """目标函数 - 由具体训练过程实现"""
        # 这里定义超参数搜索空间
        params = {
            'learning_rate': trial.suggest_categorical('learning_rate', self.space.learning_rates),
            'weight_decay': trial.suggest_categorical('weight_decay', self.space.weight_decays),
            'batch_size': trial.suggest_categorical('batch_size', self.space.batch_sizes),
            'attention_heads': trial.suggest_categorical('attention_heads', self.space.attention_heads),
            'hidden_dim': trial.suggest_categorical('hidden_dim', self.space.hidden_dims),
            'dropout_rate': trial.suggest_categorical('dropout_rate', self.space.dropout_rates),
            'fusion_method': trial.suggest_categorical('fusion_method', self.space.fusion_methods)
        }

        # 模拟训练过程 (实际使用时替换为真实训练)
        score = self._simulate_training(params)

        # 记录结果
        self.results.append({
            'params': params,
            'score': score,
            'trial': trial.number
        })

        if score > self.best_score:
            self.best_score = score
            self.best_params = params

        return score

    def _simulate_training(self, params: Dict[str, Any]) -> float:
        """模拟训练过程 - 返回准确率"""
        # 基于参数的性能预测模型
        base_score = 0.45  # 基础准确率

        # 学习率影响
        lr_factor = 1.0 + 0.1 * (params['learning_rate'] - 0.02) / 0.02

        # 批量大小影响
        batch_factor = 1.0 + 0.05 * np.log(params['batch_size'] / 200) / np.log(2)

        # 隐藏维度影响
        hidden_factor = 1.0 + 0.03 * np.log(params['hidden_dim'] / 512) / np.log(2)

        # 注意头数影响
        attention_factor = 1.0 + 0.02 * (params['attention_heads'] - 8) / 8

        # Dropout影响
        dropout_factor = 1.0 - 0.1 * (params['dropout_rate'] - 0.1) / 0.4

        # 融合方法影响
        fusion_factors = {'attention': 1.05, 'gated': 1.02, 'concat': 1.0}
        fusion_factor = fusion_factors.get(params['fusion_method'], 1.0)

        # 正则化影响
        reg_factor = 1.0 - 0.05 * (params['weight_decay'] - 0.001) / 0.099

        # 计算最终分数
        score = base_score * lr_factor * batch_factor * hidden_factor * \
                attention_factor * dropout_factor * fusion_factor * reg_factor

        # 添加随机噪声
        score += np.random.normal(0, 0.02)

        return max(0.1, min(0.8, score))  # 限制在合理范围内

    def optimize(self) -> Dict[str, Any]:
        """执行超参数优化"""
        print(f"开始超参数优化，共 {self.n_trials} 次试验...")

        # 创建Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        # 执行优化
        study.optimize(self.objective, n_trials=self.n_trials)

        # 保存结果
        self._save_results(study)

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'all_results': self.results,
            'study': study
        }

    def _save_results(self, study: optuna.Study):
        """保存优化结果"""
        results_dir = Path('outputs/hyperparameter_tuning')
        results_dir.mkdir(exist_ok=True)

        # 保存最佳参数
        with open(results_dir / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=2)

        # 保存所有试验结果
        with open(results_dir / 'all_trials.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        # 生成可视化
        self._create_visualizations(study, results_dir)

    def _create_visualizations(self, study: optuna.Study, save_dir: Path):
        """创建优化结果可视化"""
        try:
            # 参数重要性图
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(str(save_dir / 'param_importances.png'))

            # 优化历史
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(str(save_dir / 'optimization_history.png'))

            # 平行坐标图
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(str(save_dir / 'parallel_coordinate.png'))

        except Exception as e:
            print(f"可视化生成失败: {e}")

class GridSearchOptimizer:
    """网格搜索优化器 - 用于精确调优"""

    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self.results = []

    def optimize(self) -> Dict[str, Any]:
        """执行网格搜索"""
        print("开始网格搜索...")

        grid = ParameterGrid(self.param_grid)
        best_score = float('-inf')
        best_params = None

        for i, params in enumerate(grid):
            print(f"网格搜索 {i+1}/{len(grid)}: {params}")

            # 模拟训练
            score = self._evaluate_params(params)

            self.results.append({
                'params': params,
                'score': score
            })

            if score > best_score:
                best_score = score
                best_params = params

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': self.results
        }

    def _evaluate_params(self, params: Dict[str, Any]) -> float:
        """评估参数组合"""
        # 使用与贝叶斯优化相同的模拟函数
        optimizer = HyperparameterOptimizer(HyperparameterSpace())
        return optimizer._simulate_training(params)

class EnsembleOptimizer:
    """集成优化策略"""

    def __init__(self):
        self.optimizers = {
            'bayesian': HyperparameterOptimizer(HyperparameterSpace(), n_trials=30),
            'grid': GridSearchOptimizer({
                'learning_rate': [0.01, 0.02, 0.05],
                'attention_heads': [8, 12, 16],
                'hidden_dim': [512, 768],
                'fusion_method': ['attention', 'gated']
            })
        }

    def optimize(self) -> Dict[str, Any]:
        """执行集成优化"""
        print("开始集成超参数优化...")

        results = {}

        # 贝叶斯优化
        print("\\n=== 阶段1: 贝叶斯优化 ===")
        bayesian_result = self.optimizers['bayesian'].optimize()
        results['bayesian'] = bayesian_result

        # 基于贝叶斯结果的网格搜索
        print("\\n=== 阶段2: 精确网格搜索 ===")
        # 使用贝叶斯结果作为起点创建更小的搜索空间
        best_bayesian = bayesian_result['best_params']

        refined_grid = {
            'learning_rate': [best_bayesian['learning_rate'] * 0.8, best_bayesian['learning_rate'], best_bayesian['learning_rate'] * 1.2],
            'attention_heads': [max(4, best_bayesian['attention_heads']-2), best_bayesian['attention_heads'], min(16, best_bayesian['attention_heads']+2)],
            'hidden_dim': [best_bayesian['hidden_dim']],
            'fusion_method': [best_bayesian['fusion_method']]
        }

        grid_optimizer = GridSearchOptimizer(refined_grid)
        grid_result = grid_optimizer.optimize()
        results['grid'] = grid_result

        # 选择最佳结果
        bayesian_score = bayesian_result['best_score']
        grid_score = grid_result['best_score']

        if grid_score > bayesian_score:
            final_result = grid_result
            method = 'grid'
        else:
            final_result = bayesian_result
            method = 'bayesian'

        print(f"\\n最佳方法: {method}, 得分: {final_result['best_score']:.4f}")

        # 保存最终结果
        self._save_ensemble_results(results, final_result)

        return final_result

    def _save_ensemble_results(self, all_results: Dict[str, Any], final_result: Dict[str, Any]):
        """保存集成优化结果"""
        results_dir = Path('outputs/hyperparameter_tuning')
        results_dir.mkdir(exist_ok=True)

        with open(results_dir / 'ensemble_results.json', 'w') as f:
            json.dump({
                'final_best_params': final_result['best_params'],
                'final_best_score': final_result['best_score'],
                'all_methods': all_results
            }, f, indent=2)

if __name__ == "__main__":
    print("=== 超参数优化系统测试 ===")

    # 测试集成优化
    ensemble_opt = EnsembleOptimizer()
    result = ensemble_opt.optimize()

    print("\\n=== 最终结果 ===")
    print(f"最佳参数: {result['best_params']}")
    print(f"最佳得分: {result['best_score']:.4f}")

    # 保存到配置文件
    config_path = Path('outputs/hyperparameter_tuning/optimized_config.json')
    with open(config_path, 'w') as f:
        json.dump(result['best_params'], f, indent=2)

    print(f"\\n优化配置已保存到: {config_path}")