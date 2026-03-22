#!/usr/bin/env python3
"""
增强版Agent架构 - 为A刊发表优化

包括：
- 多头注意力机制
- 动态特征融合
- 高级优化策略
- 自适应正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class EnhancedConfig:
    """增强配置"""
    attention_heads: int = 8
    hidden_dim: int = 512
    dropout_rate: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True
    fusion_method: str = 'attention'  # 'attention', 'concat', 'gated'
    adaptive_lr: bool = True
    curriculum_learning: bool = True

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # 线性变换并重塑
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, V)

        # 重塑并输出
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)

        return output

class DynamicFeatureFusion(nn.Module):
    """动态特征融合模块"""

    def __init__(self, config: EnhancedConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim

        # 不同模态的特征投影
        self.vision_proj = nn.Linear(1024, self.embed_dim)  # CLIP视觉特征
        self.text_proj = nn.Linear(768, self.embed_dim)    # 文本特征
        self.metadata_proj = nn.Linear(64, self.embed_dim)  # 元数据特征

        # 融合方法
        if config.fusion_method == 'attention':
            self.fusion = MultiHeadAttention(self.embed_dim, config.attention_heads, config.dropout_rate)
            self.fusion_norm = nn.LayerNorm(self.embed_dim) if config.use_layer_norm else nn.Identity()
        elif config.fusion_method == 'gated':
            self.gate_net = nn.Sequential(
                nn.Linear(self.embed_dim * 3, self.embed_dim),
                nn.Sigmoid()
            )
        else:  # concat
            self.fusion_proj = nn.Linear(self.embed_dim * 3, self.embed_dim)

        # 残差连接和层归一化
        self.residual = config.use_residual
        self.layer_norm = nn.LayerNorm(self.embed_dim) if config.use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, vision_features: torch.Tensor,
                text_features: torch.Tensor,
                metadata_features: torch.Tensor) -> torch.Tensor:
        # 投影到统一空间
        v_proj = self.vision_proj(vision_features)
        t_proj = self.text_proj(text_features)
        m_proj = self.metadata_proj(metadata_features)

        if self.config.fusion_method == 'attention':
            # 使用注意力融合
            # 将所有特征拼接作为key和value，vision作为query
            combined = torch.stack([v_proj, t_proj, m_proj], dim=1)  # [batch, 3, dim]
            fused = self.fusion(v_proj.unsqueeze(1), combined, combined)
            fused = fused.squeeze(1)

            if self.residual:
                fused = fused + v_proj  # 残差连接
            fused = self.fusion_norm(fused)

        elif self.config.fusion_method == 'gated':
            # 门控融合
            combined = torch.cat([v_proj, t_proj, m_proj], dim=-1)
            gates = self.gate_net(combined)
            fused = gates * v_proj + (1 - gates) * t_proj + gates * m_proj

        else:  # concat
            combined = torch.cat([v_proj, t_proj, m_proj], dim=-1)
            fused = self.fusion_proj(combined)

        # 最终处理
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused

class EnhancedTargetLearner(nn.Module):
    """增强版目标学习器"""

    def __init__(self, config: EnhancedConfig):
        super().__init__()
        self.config = config

        # 特征融合
        self.feature_fusion = DynamicFeatureFusion(config)

        # 增强的决策网络
        self.decision_net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # 注意力权重预测
        self.attention_weights = nn.Linear(config.hidden_dim, 15)  # 15个条件权重

    def forward(self, vision_features: torch.Tensor,
                text_features: torch.Tensor,
                metadata_features: torch.Tensor,
                condition_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 特征融合
        fused_features = self.feature_fusion(vision_features, text_features, metadata_features)

        # 决策预测
        decision = self.decision_net(fused_features)

        # 注意力权重
        attention_w = self.attention_weights(fused_features)

        return decision, attention_w

class AdaptiveOptimizer:
    """自适应优化器"""

    def __init__(self, base_lr: float = 0.02, adaptive: bool = True):
        self.base_lr = base_lr
        self.adaptive = adaptive
        self.lr_history = []

    def get_learning_rate(self, epoch: int, loss_history: List[float]) -> float:
        """自适应学习率调整"""
        if not self.adaptive:
            return self.base_lr

        # 基于损失历史调整学习率
        if len(loss_history) > 5:
            recent_loss = np.mean(loss_history[-5:])
            prev_loss = np.mean(loss_history[-10:-5]) if len(loss_history) > 10 else recent_loss

            if recent_loss > prev_loss * 1.01:  # 损失上升
                lr = self.base_lr * 0.8
            elif recent_loss < prev_loss * 0.99:  # 损失下降
                lr = self.base_lr * 1.05
            else:
                lr = self.base_lr
        else:
            lr = self.base_lr

        # 应用余弦退火
        cosine_factor = 0.5 * (1 + np.cos(np.pi * epoch / 100))
        lr = lr * cosine_factor

        self.lr_history.append(lr)
        return lr

class CurriculumLearningScheduler:
    """课程学习调度器"""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.phases = [
            {'name': 'easy', 'start': 0, 'end': 0.3, 'difficulty': 0.3},
            {'name': 'medium', 'start': 0.3, 'end': 0.7, 'difficulty': 0.6},
            {'name': 'hard', 'start': 0.7, 'end': 1.0, 'difficulty': 1.0}
        ]

    def get_difficulty(self, epoch: int) -> float:
        """获取当前epoch的难度级别"""
        progress = epoch / self.total_epochs

        for phase in self.phases:
            if phase['start'] <= progress < phase['end']:
                return phase['difficulty']

        return 1.0

    def should_include_case(self, case_difficulty: float, current_difficulty: float) -> bool:
        """决定是否包含某个案例"""
        return case_difficulty <= current_difficulty

class EnhancedAgentTrainer:
    """增强版Agent训练器"""

    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.model = EnhancedTargetLearner(config)
        self.optimizer = AdaptiveOptimizer(adaptive=config.adaptive_lr)
        self.curriculum = CurriculumLearningScheduler(100) if config.curriculum_learning else None

        # 损失函数
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def train_step(self, batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
        """训练一步"""
        # 获取当前难度
        difficulty = self.curriculum.get_difficulty(epoch) if self.curriculum else 1.0

        # 前向传播
        decision_pred, attention_pred = self.model(
            batch['vision_features'],
            batch['text_features'],
            batch['metadata_features'],
            batch['condition_features']
        )

        # 计算损失
        decision_loss = self.bce_loss(decision_pred.squeeze(), batch['target'])
        attention_loss = self.mse_loss(attention_pred, batch['attention_target'])

        total_loss = decision_loss + 0.1 * attention_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'decision_loss': decision_loss.item(),
            'attention_loss': attention_loss.item(),
            'learning_rate': self.optimizer.get_learning_rate(epoch, [total_loss.item()])
        }

    def predict(self, features: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """预测"""
        with torch.no_grad():
            decision, attention = self.model(
                features['vision_features'],
                features['text_features'],
                features['metadata_features'],
                features['condition_features']
            )

            return {
                'decision': decision.item(),
                'attention_weights': attention.squeeze().tolist(),
                'confidence': min(1.0, max(0.0, decision.item()))
            }

if __name__ == "__main__":
    # 测试增强架构
    config = EnhancedConfig()
    trainer = EnhancedAgentTrainer(config)

    print("增强版Agent架构已创建")
    print(f"配置: attention_heads={config.attention_heads}, hidden_dim={config.hidden_dim}")
    print(f"融合方法: {config.fusion_method}, 自适应LR: {config.adaptive_lr}")

    # 测试前向传播
    batch_size = 4
    test_batch = {
        'vision_features': torch.randn(batch_size, 1024),
        'text_features': torch.randn(batch_size, 768),
        'metadata_features': torch.randn(batch_size, 64),
        'condition_features': torch.randn(batch_size, 15),
        'target': torch.randn(batch_size),
        'attention_target': torch.randn(batch_size, 15)
    }

    result = trainer.train_step(test_batch, epoch=1)
    print(f"训练测试结果: loss={result['total_loss']:.4f}, lr={result['learning_rate']:.6f}")