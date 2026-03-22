#!/usr/bin/env python3
"""
多数据集加载器 - 支持PAD-UFES-20, HAM10000, ISIC等皮肤病数据集

用于提升实验规模和发表等级
"""

import os
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    root_path: str
    metadata_file: str
    image_column: str = 'image_id'
    label_column: str = 'label'
    metadata_columns: List[str] = None

    def __post_init__(self):
        if self.metadata_columns is None:
            self.metadata_columns = ['age', 'sex', 'location']

class MultiDatasetLoader:
    """多数据集加载器"""

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.external_root = self.data_root / "external_datasets"

        # 配置各个数据集
        self.dataset_configs = {
            'pad_ufes20': DatasetConfig(
                name='PAD-UFES-20',
                root_path='pad_ufes_20',
                metadata_file='metadata.json',
                image_column='file',
                label_column='label',
                metadata_columns=['age', 'sex', 'location', 'site', 'clinical_history']
            ),
            'ham10000': DatasetConfig(
                name='HAM10000',
                root_path='external_datasets/ham10000',
                metadata_file='HAM10000_metadata.csv',
                image_column='image_id',
                label_column='dx',
                metadata_columns=['age', 'sex', 'localization']
            ),
            'isic2019': DatasetConfig(
                name='ISIC2019',
                root_path='external_datasets/isic2019',
                metadata_file='ISIC_2019_Training_Metadata.csv',
                image_column='image',
                label_column='diagnosis',
                metadata_columns=['age_approx', 'sex', 'anatom_site_general']
            )
        }

    def load_dataset(self, dataset_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """加载单个数据集"""
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        config = self.dataset_configs[dataset_name]
        dataset_path = self.data_root / config.root_path

        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset_name} not found at {dataset_path}")
            print("Please download the dataset manually to the specified location")
            return []

        # 加载元数据
        metadata_path = dataset_path / config.metadata_file
        if not metadata_path.exists():
            print(f"Warning: Metadata file not found: {metadata_path}")
            return []

        try:
            if config.metadata_file.endswith('.csv'):
                df = pd.read_csv(metadata_path)
            elif config.metadata_file.endswith('.json'):
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    df = pd.DataFrame(data)
            else:
                print(f"Unsupported metadata format: {config.metadata_file}")
                return []

            cases = []
            for _, row in df.iterrows():
                case = {
                    'dataset': dataset_name,
                    'case_id': f"{dataset_name}_{row[config.image_column]}",
                    'image_path': str(dataset_path / 'images' / f"{row[config.image_column]}.jpg"),
                    'label': row[config.label_column],
                    'metadata': {}
                }

                # 添加元数据
                for col in config.metadata_columns:
                    if col in row and pd.notna(row[col]):
                        case['metadata'][col] = row[col]

                cases.append(case)

                if limit and len(cases) >= limit:
                    break

            print(f"Loaded {len(cases)} cases from {dataset_name}")
            return cases

        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            return []

    def load_multiple_datasets(self, dataset_names: List[str], limit_per_dataset: Optional[int] = None) -> List[Dict[str, Any]]:
        """加载多个数据集"""
        all_cases = []

        for name in dataset_names:
            cases = self.load_dataset(name, limit_per_dataset)
            all_cases.extend(cases)

        print(f"\\nTotal cases loaded: {len(all_cases)}")
        return all_cases

    def get_dataset_stats(self, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """获取数据集统计信息"""
        stats = {
            'total_cases': len(cases),
            'datasets': {},
            'labels': {},
            'metadata_stats': {}
        }

        # 按数据集统计
        for case in cases:
            dataset = case.get('dataset', 'unknown')
            label = case.get('label', 'unknown')

            if dataset not in stats['datasets']:
                stats['datasets'][dataset] = 0
            stats['datasets'][dataset] += 1

            if label not in stats['labels']:
                stats['labels'][label] = 0
            stats['labels'][label] += 1

        return stats

    def create_balanced_split(self, cases: List[Dict[str, Any]],
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.2) -> Dict[str, List[Dict[str, Any]]]:
        """创建平衡的数据分割"""
        from collections import defaultdict

        # 按标签分组
        label_groups = defaultdict(list)
        for case in cases:
            label_groups[case['label']].append(case)

        train_cases, val_cases, test_cases = [], [], []

        for label, group_cases in label_groups.items():
            n_total = len(group_cases)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            n_test = n_total - n_train - n_val

            # 随机打乱
            np.random.shuffle(group_cases)

            train_cases.extend(group_cases[:n_train])
            val_cases.extend(group_cases[n_train:n_train+n_val])
            test_cases.extend(group_cases[n_train+n_val:])

        return {
            'train': train_cases,
            'val': val_cases,
            'test': test_cases
        }

# 下载指导
DOWNLOAD_GUIDE = """
=== 数据集下载指南 ===

1. HAM10000数据集 (推荐优先下载):
   - 官网: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T
   - 下载 metadata: HAM10000_metadata.csv
   - 下载 images: HAM10000_images_part_1.zip, HAM10000_images_part_2.zip
   - 放置位置: data/external_datasets/ham10000/

2. ISIC 2019数据集:
   - 官网: https://challenge.isic-archive.com/data/#2019
   - 下载 training data和metadata
   - 放置位置: data/external_datasets/isic2019/

3. 其他数据集:
   - BCN20000: https://www.bcndataset.com/
   - PH2: https://www.fc.up.pt/addi/ph2%20database.html

下载后运行:
python datasets/multi_dataset_loader.py --test-load
"""

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-dataset loader for DermAgent")
    parser.add_argument("--test-load", action="store_true", help="Test loading available datasets")
    parser.add_argument("--datasets", nargs="+", help="Datasets to load")
    parser.add_argument("--stats", action="store_true", help="Show dataset statistics")

    args = parser.parse_args()

    loader = MultiDatasetLoader()

    if args.test_load:
        print("Testing dataset loading...")

        # 测试PAD-UFES-20
        pad_cases = loader.load_dataset('pad_ufes20', limit=100)
        print(f"PAD-UFES-20: {len(pad_cases)} cases loaded")

        # 测试其他数据集
        available_datasets = ['ham10000', 'isic2019']
        for dataset in available_datasets:
            cases = loader.load_dataset(dataset, limit=50)
            print(f"{dataset}: {len(cases)} cases loaded")

    elif args.datasets:
        cases = loader.load_multiple_datasets(args.datasets)
        if args.stats:
            stats = loader.get_dataset_stats(cases)
            print("\\nDataset Statistics:")
            print(json.dumps(stats, indent=2))

    else:
        print(DOWNLOAD_GUIDE)