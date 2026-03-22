#!/usr/bin/env python3
"""
数据增强系统 - 解决数据集单一问题

包括：
- 图像增强
- 文本增强
- 元数据合成
- 跨数据集迁移
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import random

class ImageAugmentation:
    """图像增强"""

    def __init__(self):
        # 定义增强管道
        self.augmentation_pipeline = A.Compose([
            # 几何变换
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

            # 颜色变换
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),

            # 噪声和模糊
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),

            # 其他变换
            A.RandomShadow(p=0.2),
            A.RandomSnow(p=0.1),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2)
        ])

        # 严重病变特化增强
        self.severe_augmentation = A.Compose([
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            A.GridDistortion(p=0.5),
        ])

    def augment_image(self, image: np.ndarray, severity: str = 'normal') -> np.ndarray:
        """增强单张图像"""
        if severity == 'severe':
            augmented = self.severe_augmentation(image=image)
        else:
            augmented = self.augmentation_pipeline(image=image)

        return augmented['image']

    def augment_batch(self, images: List[np.ndarray], severities: List[str] = None) -> List[np.ndarray]:
        """批量增强"""
        if severities is None:
            severities = ['normal'] * len(images)

        augmented = []
        for img, sev in zip(images, severities):
            aug_img = self.augment_image(img, sev)
            augmented.append(aug_img)

        return augmented

class TextAugmentation:
    """文本增强"""

    def __init__(self):
        # 医疗文本增强规则
        self.medical_synonyms = {
            'red': ['reddish', 'erythematous', 'crimson'],
            'swollen': ['edematous', 'inflamed', 'puffy'],
            'painful': ['tender', 'sore', 'aching'],
            'large': ['prominent', 'significant', 'substantial'],
            'small': ['minimal', 'tiny', 'diminutive']
        }

        self.location_synonyms = {
            'arm': ['upper extremity', 'brachium'],
            'face': ['facial region', 'facial area'],
            'back': ['dorsal region', 'posterior aspect'],
            'chest': ['thoracic region', 'anterior chest'],
            'leg': ['lower extremity', 'crural region']
        }

    def augment_text(self, text: str, n_augmentations: int = 3) -> List[str]:
        """文本增强"""
        augmented_texts = [text]  # 保留原文本

        for _ in range(n_augmentations):
            aug_text = text

            # 同义词替换
            for word, synonyms in {**self.medical_synonyms, **self.location_synonyms}.items():
                if word in aug_text.lower():
                    synonym = random.choice(synonyms)
                    aug_text = aug_text.replace(word, synonym)
                    break  # 只替换一个词

            # 随机删除一些词 (模拟描述不完整)
            words = aug_text.split()
            if len(words) > 3:
                n_remove = random.randint(0, min(2, len(words)//4))
                indices_to_remove = random.sample(range(len(words)), n_remove)
                words = [w for i, w in enumerate(words) if i not in indices_to_remove]
                aug_text = ' '.join(words)

            augmented_texts.append(aug_text)

        return augmented_texts

class MetadataAugmentation:
    """元数据增强"""

    def __init__(self):
        # 基于真实分布的元数据生成
        self.age_distributions = {
            'child': {'mean': 8, 'std': 3, 'range': (1, 15)},
            'young': {'mean': 35, 'std': 10, 'range': (16, 50)},
            'middle': {'mean': 55, 'std': 8, 'range': (51, 70)},
            'old': {'mean': 75, 'std': 8, 'range': (71, 100)}
        }

        self.location_frequencies = {
            'face': 0.25, 'arm': 0.20, 'back': 0.15, 'chest': 0.12,
            'leg': 0.10, 'neck': 0.08, 'hand': 0.05, 'foot': 0.03,
            'scalp': 0.02
        }

    def augment_metadata(self, metadata: Dict[str, Any], n_augmentations: int = 3) -> List[Dict[str, Any]]:
        """元数据增强"""
        augmented_metadata = [metadata.copy()]  # 保留原数据

        for _ in range(n_augmentations):
            aug_meta = metadata.copy()

            # 年龄扰动
            if 'age' in aug_meta and aug_meta['age']:
                age = aug_meta['age']
                # 基于年龄段添加噪声
                if age < 16:
                    noise = np.random.normal(0, 2)
                elif age < 50:
                    noise = np.random.normal(0, 5)
                else:
                    noise = np.random.normal(0, 3)

                aug_meta['age'] = max(1, min(100, int(age + noise)))

            # 性别保持或轻微变化 (模拟录入错误)
            if random.random() < 0.05:  # 5%概率改变性别
                aug_meta['sex'] = 'FEMALE' if aug_meta.get('sex') == 'MALE' else 'MALE'

            # 位置保持 (位置通常是确定的)
            # 可以添加位置描述的变体
            if 'location' in aug_meta and random.random() < 0.3:
                location = aug_meta['location'].lower()
                if location in ['arm', 'leg']:
                    aug_meta['location'] = aug_meta['location'] + ' (unspecified side)'

            augmented_metadata.append(aug_meta)

        return augmented_metadata

class CrossDatasetAugmentation:
    """跨数据集增强"""

    def __init__(self):
        # 数据集特征映射
        self.dataset_mappings = {
            'pad_ufes20': {
                'labels': ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK'],
                'modalities': ['clinical_image', 'dermoscopic_image', 'metadata'],
                'quality': 'high'
            },
            'ham10000': {
                'labels': ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
                'modalities': ['clinical_image', 'metadata'],
                'quality': 'high'
            },
            'isic2019': {
                'labels': ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC'],
                'modalities': ['dermoscopic_image', 'metadata'],
                'quality': 'very_high'
            }
        }

        # 标签映射 (跨数据集标签对齐)
        self.label_mappings = {
            'ACK': ['akiec', 'AKIEC'],
            'BCC': ['bcc', 'BCC'],
            'MEL': ['mel', 'MEL'],
            'NEV': ['nv', 'NV'],
            'SCC': ['SCC'],  # PAD-UFES-20特有
            'SEK': ['bkl', 'BKL']  # 近似映射
        }

    def generate_synthetic_case(self, base_case: Dict[str, Any],
                               target_dataset: str) -> Dict[str, Any]:
        """生成合成案例"""
        synthetic_case = base_case.copy()

        # 调整标签分布以匹配目标数据集
        target_dist = self._get_target_distribution(target_dataset)
        synthetic_case['label'] = self._sample_from_distribution(target_dist)

        # 调整元数据分布
        if 'metadata' in synthetic_case:
            synthetic_case['metadata'] = self._adjust_metadata_distribution(
                synthetic_case['metadata'], target_dataset
            )

        # 添加数据集标记
        synthetic_case['original_dataset'] = base_case.get('dataset', 'pad_ufes20')
        synthetic_case['synthetic'] = True

        return synthetic_case

    def _get_target_distribution(self, dataset: str) -> Dict[str, float]:
        """获取目标数据集的标签分布"""
        # 基于真实数据集的近似分布
        distributions = {
            'ham10000': {
                'akiec': 0.03, 'bcc': 0.05, 'bkl': 0.11,
                'df': 0.02, 'mel': 0.11, 'nv': 0.67, 'vasc': 0.01
            },
            'isic2019': {
                'MEL': 0.18, 'NV': 0.50, 'BCC': 0.13,
                'AKIEC': 0.04, 'BKL': 0.10, 'DF': 0.02, 'VASC': 0.03
            }
        }

        return distributions.get(dataset, {})

    def _sample_from_distribution(self, distribution: Dict[str, float]) -> str:
        """从分布中采样"""
        labels = list(distribution.keys())
        probs = list(distribution.values())
        return np.random.choice(labels, p=probs)

    def _adjust_metadata_distribution(self, metadata: Dict[str, Any], target_dataset: str) -> Dict[str, Any]:
        """调整元数据分布"""
        adjusted = metadata.copy()

        # 年龄分布调整
        if target_dataset == 'ham10000':
            # HAM10000年龄分布更广
            if 'age' in adjusted and adjusted['age']:
                age = adjusted['age']
                # 添加年龄噪声使分布更广
                noise = np.random.normal(0, 10)
                adjusted['age'] = max(1, min(100, int(age + noise)))

        elif target_dataset == 'isic2019':
            # ISIC2019更注重老年患者
            if 'age' in adjusted and adjusted['age']:
                age = adjusted['age']
                # 向更高年龄偏移
                age_boost = np.random.normal(5, 3)
                adjusted['age'] = max(1, min(100, int(age + age_boost)))

        return adjusted

class DataAugmentationPipeline:
    """完整的数据增强管道"""

    def __init__(self):
        self.image_aug = ImageAugmentation()
        self.text_aug = TextAugmentation()
        self.metadata_aug = MetadataAugmentation()
        self.cross_dataset_aug = CrossDatasetAugmentation()

    def augment_case(self, case: Dict[str, Any], augmentation_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """增强单个案例"""
        if augmentation_config is None:
            augmentation_config = {
                'image_augmentations': 2,
                'text_augmentations': 2,
                'metadata_augmentations': 1,
                'cross_dataset': False
            }

        augmented_cases = [case]  # 保留原案例

        # 图像增强
        if 'image_path' in case and augmentation_config['image_augmentations'] > 0:
            try:
                # 这里需要实际加载图像
                # image = cv2.imread(case['image_path'])
                # aug_images = self.image_aug.augment_batch([image] * augmentation_config['image_augmentations'])

                # 暂时用占位符
                for i in range(augmentation_config['image_augmentations']):
                    aug_case = case.copy()
                    aug_case['augmentation_type'] = 'image'
                    aug_case['augmentation_id'] = i
                    augmented_cases.append(aug_case)

            except Exception as e:
                print(f"图像增强失败: {e}")

        # 文本增强
        if 'text' in case and augmentation_config['text_augmentations'] > 0:
            aug_texts = self.text_aug.augment_text(case['text'], augmentation_config['text_augmentations'])

            for i, aug_text in enumerate(aug_texts[1:], 1):  # 跳过原文本
                aug_case = case.copy()
                aug_case['text'] = aug_text
                aug_case['augmentation_type'] = 'text'
                aug_case['augmentation_id'] = i
                augmented_cases.append(aug_case)

        # 元数据增强
        if 'metadata' in case and augmentation_config['metadata_augmentations'] > 0:
            aug_metadatas = self.metadata_aug.augment_metadata(case['metadata'], augmentation_config['metadata_augmentations'])

            for i, aug_meta in enumerate(aug_metadatas[1:], 1):  # 跳过原数据
                aug_case = case.copy()
                aug_case['metadata'] = aug_meta
                aug_case['augmentation_type'] = 'metadata'
                aug_case['augmentation_id'] = i
                augmented_cases.append(aug_case)

        # 跨数据集增强
        if augmentation_config['cross_dataset']:
            for target_dataset in ['ham10000', 'isic2019']:
                synthetic_case = self.cross_dataset_aug.generate_synthetic_case(case, target_dataset)
                synthetic_case['augmentation_type'] = f'cross_dataset_{target_dataset}'
                augmented_cases.append(synthetic_case)

        return augmented_cases

    def augment_dataset(self, cases: List[Dict[str, Any]],
                       augmentation_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """增强整个数据集"""
        augmented_dataset = []

        print(f"开始数据增强，原数据集大小: {len(cases)}")

        for i, case in enumerate(cases):
            if i % 100 == 0:
                print(f"增强进度: {i}/{len(cases)}")

            aug_cases = self.augment_case(case, augmentation_config)
            augmented_dataset.extend(aug_cases)

        print(f"增强完成，最终数据集大小: {len(augmented_dataset)}")
        print(f"扩增倍数: {len(augmented_dataset) / len(cases):.1f}x")

        return augmented_dataset

    def get_augmentation_stats(self, original_cases: List[Dict[str, Any]],
                              augmented_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取增强统计"""
        stats = {
            'original_size': len(original_cases),
            'augmented_size': len(augmented_cases),
            'expansion_factor': len(augmented_cases) / len(original_cases),
            'augmentation_types': {}
        }

        for case in augmented_cases:
            aug_type = case.get('augmentation_type', 'original')
            if aug_type not in stats['augmentation_types']:
                stats['augmentation_types'][aug_type] = 0
            stats['augmentation_types'][aug_type] += 1

        return stats

if __name__ == "__main__":
    # 测试数据增强系统
    pipeline = DataAugmentationPipeline()

    # 测试案例
    test_case = {
        'case_id': 'test_001',
        'label': 'BCC',
        'text': 'red swollen lesion on arm',
        'metadata': {
            'age': 55,
            'sex': 'MALE',
            'location': 'arm'
        }
    }

    # 增强配置
    config = {
        'image_augmentations': 0,  # 暂时跳过图像增强
        'text_augmentations': 2,
        'metadata_augmentations': 1,
        'cross_dataset': True
    }

    # 执行增强
    augmented_cases = pipeline.augment_case(test_case, config)

    print(f"原始案例: 1个")
    print(f"增强后案例: {len(augmented_cases)}个")

    for case in augmented_cases:
        print(f"- {case.get('augmentation_type', 'original')}: {case.get('text', 'N/A')[:50]}...")

    print("\\n数据增强系统测试完成！")