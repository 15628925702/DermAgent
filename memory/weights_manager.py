"""
统一的学习参数和权重管理器

负责：
- 所有学习组件的权重初始化和保存
- 版本控制和baseline管理
- 学习超参数的统一配置
"""

from __future__ import annotations

import json
import os
import configparser
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from agent.controller import LearnableSkillController, TargetLearner
from agent.evidence_calibrator import LearnableEvidenceCalibrator
from agent.final_scorer import LearnableFinalScorer
from agent.rule_scorer import LearnableRuleScorer
from memory.retriever import LearnableRetrievalScorer
from memory.skill_index import SkillIndex


@dataclass
class LearningConfig:
    """学习超参数配置"""
    # 全局学习设置
    global_learning_rate: float = 0.05
    weight_decay: float = 0.001
    momentum: float = 0.9

    # 组件特定设置
    controller_learning_rate: float = 0.08
    final_scorer_learning_rate: float = 0.01
    rule_scorer_learning_rate: float = 0.05
    retrieval_scorer_learning_rate: float = 0.03
    evidence_calibrator_learning_rate: float = 0.02

    # 优化器设置
    use_adam: bool = False

    # 训练设置
    max_epochs: int = 50
    early_stopping_patience: int = 10
    validation_interval: int = 5

    # 权重初始化版本
    weights_version: str = "v1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_learning_rate": self.global_learning_rate,
            "weight_decay": self.weight_decay,
            "momentum": self.momentum,
            "controller_learning_rate": self.controller_learning_rate,
            "final_scorer_learning_rate": self.final_scorer_learning_rate,
            "rule_scorer_learning_rate": self.rule_scorer_learning_rate,
            "retrieval_scorer_learning_rate": self.retrieval_scorer_learning_rate,
            "evidence_calibrator_learning_rate": self.evidence_calibrator_learning_rate,
            "max_epochs": self.max_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "validation_interval": self.validation_interval,
            "weights_version": self.weights_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LearningConfig:
        return cls(**data)


class WeightsManager:
    """统一权重管理器"""

    def __init__(self, weights_dir: str = "outputs/weights", config_file: str = "config.ini"):
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> LearningConfig:
        """从配置文件加载学习配置"""
        config = configparser.ConfigParser()
        config.read(self.config_file)

        learning_config = LearningConfig()

        if config.has_section('learning'):
            learning = config['learning']
            learning_config.global_learning_rate = learning.getfloat('global_learning_rate', 0.05)
            learning_config.weight_decay = learning.getfloat('weight_decay', 0.001)
            learning_config.momentum = learning.getfloat('momentum', 0.9)
            learning_config.max_epochs = learning.getint('max_epochs', 50)
            learning_config.early_stopping_patience = learning.getint('early_stopping_patience', 10)
            learning_config.validation_interval = learning.getint('validation_interval', 5)
            learning_config.controller_learning_rate = learning.getfloat('controller_learning_rate', 0.08)
            learning_config.final_scorer_learning_rate = learning.getfloat('final_scorer_learning_rate', 0.01)
            learning_config.rule_scorer_learning_rate = learning.getfloat('rule_scorer_learning_rate', 0.05)
            learning_config.retrieval_scorer_learning_rate = learning.getfloat('retrieval_scorer_learning_rate', 0.03)
            learning_config.evidence_calibrator_learning_rate = learning.getfloat('evidence_calibrator_learning_rate', 0.02)

        if config.has_section('global'):
            global_section = config['global']
            learning_config.weights_version = global_section.get('version', 'v1.0')

        return learning_config

    def get_baseline_path(self, version: str = "latest") -> Path:
        """获取baseline权重文件路径"""
        if version == "latest":
            version = self.config.weights_version
        return self.weights_dir / f"baseline_{version}.json"

    def get_checkpoint_path(self, run_name: str, epoch: int = -1) -> Path:
        """获取检查点文件路径"""
        if epoch >= 0:
            return self.weights_dir / f"checkpoint_{run_name}_epoch_{epoch}.json"
        else:
            return self.weights_dir / f"checkpoint_{run_name}_final.json"

    def save_baseline_weights(self, components: Dict[str, Any], version: str = None) -> None:
        """保存baseline权重"""
        if version:
            self.config.weights_version = version

        data = {
            "config": self.config.to_dict(),
            "components": {},
            "metadata": {
                "version": self.config.weights_version,
                "created_at": "2024-01-01T00:00:00Z",  # 实际使用时会更新
                "description": "Baseline weights for DermAgent"
            }
        }

        for name, component in components.items():
            if hasattr(component, 'to_dict'):
                data["components"][name] = component.to_dict()

        path = self.get_baseline_path()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved baseline weights to {path}")

    def load_baseline_weights(self, components: Dict[str, Any]) -> None:
        """加载baseline权重"""
        path = self.get_baseline_path()
        if not path.exists():
            print(f"Baseline weights not found at {path}, using defaults")
            return

        with open(path, 'r') as f:
            data = json.load(f)

        # 加载配置
        if "config" in data:
            self.config = LearningConfig.from_dict(data["config"])

        # 加载组件权重
        component_data = data.get("components", {})
        for name, component in components.items():
            if name in component_data and hasattr(component, 'load_state'):
                component.load_state(component_data[name])
                print(f"Loaded {name} weights from baseline")

    def save_checkpoint(self, components: Dict[str, Any], run_name: str, epoch: int = -1,
                       metadata: Dict[str, Any] = None) -> None:
        """保存训练检查点"""
        data = {
            "config": self.config.to_dict(),
            "components": {},
            "metadata": metadata or {},
            "epoch": epoch,
            "run_name": run_name
        }

        for name, component in components.items():
            if hasattr(component, 'to_dict'):
                data["components"][name] = component.to_dict()

        path = self.get_checkpoint_path(run_name, epoch)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, components: Dict[str, Any], run_name: str, epoch: int = -1) -> Dict[str, Any]:
        """加载训练检查点"""
        path = self.get_checkpoint_path(run_name, epoch)
        if not path.exists():
            print(f"Checkpoint not found at {path}")
            return {}

        with open(path, 'r') as f:
            data = json.load(f)

        # 加载配置
        if "config" in data:
            self.config = LearningConfig.from_dict(data["config"])

        # 加载组件权重
        component_data = data.get("components", {})
        for name, component in components.items():
            if name in component_data and hasattr(component, 'load_state'):
                component.load_state(component_data[name])

        print(f"Loaded checkpoint from {path}")
        return data.get("metadata", {})

    def initialize_components(self, skill_index: SkillIndex) -> Dict[str, Any]:
        """初始化所有学习组件"""
        components = {
            "controller": LearnableSkillController(skill_index, learning_rate=self.config.controller_learning_rate, use_adam=self.config.use_adam),
            "final_scorer": LearnableFinalScorer(learning_rate=self.config.final_scorer_learning_rate, use_adam=self.config.use_adam),
            "rule_scorer": LearnableRuleScorer(learning_rate=self.config.rule_scorer_learning_rate, use_adam=self.config.use_adam),
            "retrieval_scorer": LearnableRetrievalScorer(learning_rate=self.config.retrieval_scorer_learning_rate, use_adam=self.config.use_adam),
            "evidence_calibrator": LearnableEvidenceCalibrator(learning_rate=self.config.evidence_calibrator_learning_rate, use_adam=self.config.use_adam),
        }

        # 加载baseline权重
        self.load_baseline_weights(components)

        return components

    def create_default_baseline(self, skill_index: SkillIndex) -> None:
        """创建默认baseline权重（如果不存在）"""
        baseline_path = self.get_baseline_path()
        if baseline_path.exists():
            return

        print("Creating default baseline weights...")

        # 初始化组件
        components = self.initialize_components(skill_index)

        # 保存为baseline
        self.save_baseline_weights(components, "v1.0")

        print("Default baseline weights created")


# 全局权重管理器实例
weights_manager = WeightsManager()
