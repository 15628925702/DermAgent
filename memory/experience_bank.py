"""
经验库容器。

基础版使用内存列表，但接口尽量设计稳定，方便后续替换成：
- JSON 文件
- sqlite
- embedding index
- faiss / vector db

基础版目标：
- 支持经验写入 / 批量写入
- 支持按类型过滤
- 支持简单查询
- 支持去重与统计
- 给 retriever / writer / reflection 提供稳定接口

相关文件：
- memory/schema.py
- memory/retriever.py
- memory/writer.py
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ExperienceBank:
    def __init__(self, initial_items: Optional[List[Dict[str, Any]]] = None) -> None:
        self.items: List[Dict[str, Any]] = []
        if initial_items:
            self.extend(initial_items)

    # =========================
    # 基础写入
    # =========================

    def add(self, item: Dict[str, Any]) -> None:
        """
        添加单条经验。

        基础版做轻量校验：
        - 必须是 dict
        - 必须包含 experience_type
        """
        self._validate_item(item)
        self.items.append(dict(item))

    def extend(self, items: List[Dict[str, Any]]) -> None:
        """
        批量添加经验。
        """
        for item in items:
            self.add(item)

    def add_if_not_exists(self, item: Dict[str, Any]) -> bool:
        """
        若经验不存在则添加，返回是否成功添加。

        基础版去重规则：
        - raw_case: case_id 相同视为重复
        - prototype: disease 相同视为重复
        - confusion: pair 相同视为重复
        - rule: rule_name 相同视为重复
        - 其他：完全相同 key 不保证去重
        """
        self._validate_item(item)

        if self.exists(item):
            return False

        self.items.append(dict(item))
        return True

    # =========================
    # 基础读取
    # =========================

    def list_all(self) -> List[Dict[str, Any]]:
        """
        返回全部经验的浅拷贝列表。
        """
        return [dict(x) for x in self.items]

    def list_by_type(self, experience_type: str) -> List[Dict[str, Any]]:
        """
        按经验类型列出。
        """
        experience_type = str(experience_type).strip()
        return [
            dict(x)
            for x in self.items
            if str(x.get("experience_type", "")).strip() == experience_type
        ]

    def get_raw_cases(self) -> List[Dict[str, Any]]:
        return self.list_by_type("raw_case")

    def get_prototypes(self) -> List[Dict[str, Any]]:
        return self.list_by_type("prototype")

    def get_confusions(self) -> List[Dict[str, Any]]:
        return self.list_by_type("confusion")

    def get_rules(self) -> List[Dict[str, Any]]:
        return self.list_by_type("rule")

    def get_hard_cases(self) -> List[Dict[str, Any]]:
        return self.list_by_type("hard_case")

    # =========================
    # 简单查询
    # =========================

    def find_raw_case_by_case_id(self, case_id: str) -> Optional[Dict[str, Any]]:
        case_id = str(case_id).strip()
        if not case_id:
            return None

        for item in self.items:
            if item.get("experience_type") != "raw_case":
                continue
            if str(item.get("case_id", "")).strip() == case_id:
                return dict(item)
        return None

    def find_prototype_by_disease(self, disease: str) -> Optional[Dict[str, Any]]:
        disease = str(disease).strip().upper()
        if not disease:
            return None

        for item in self.items:
            if item.get("experience_type") != "prototype":
                continue
            if str(item.get("disease", "")).strip().upper() == disease:
                return dict(item)
        return None

    def find_confusion_by_pair(self, disease_a: str, disease_b: str) -> Optional[Dict[str, Any]]:
        pair = sorted([
            str(disease_a).strip().upper(),
            str(disease_b).strip().upper(),
        ])
        if not pair[0] or not pair[1]:
            return None

        for item in self.items:
            if item.get("experience_type") != "confusion":
                continue
            stored_pair = sorted([str(x).strip().upper() for x in item.get("pair", [])])
            if stored_pair == pair:
                return dict(item)
        return None

    def find_rule_by_name(self, rule_name: str) -> Optional[Dict[str, Any]]:
        rule_name = str(rule_name).strip()
        if not rule_name:
            return None

        for item in self.items:
            if item.get("experience_type") != "rule":
                continue
            if str(item.get("rule_name", "")).strip() == rule_name:
                return dict(item)
        return None

    # =========================
    # 去重判断
    # =========================

    def exists(self, item: Dict[str, Any]) -> bool:
        """
        判断某条经验是否已存在。
        """
        exp_type = str(item.get("experience_type", "")).strip()

        if exp_type == "raw_case":
            case_id = str(item.get("case_id", "")).strip()
            if not case_id:
                return False
            return self.find_raw_case_by_case_id(case_id) is not None

        if exp_type == "prototype":
            disease = str(item.get("disease", "")).strip().upper()
            if not disease:
                return False
            return self.find_prototype_by_disease(disease) is not None

        if exp_type == "confusion":
            pair = [str(x).strip().upper() for x in item.get("pair", [])]
            if len(pair) != 2 or not pair[0] or not pair[1]:
                return False
            return self.find_confusion_by_pair(pair[0], pair[1]) is not None

        if exp_type == "rule":
            rule_name = str(item.get("rule_name", "")).strip()
            if not rule_name:
                return False
            return self.find_rule_by_name(rule_name) is not None

        if exp_type == "hard_case":
            case_id = str(item.get("case_id", "")).strip()
            if not case_id:
                return False
            for stored in self.items:
                if stored.get("experience_type") != "hard_case":
                    continue
                if str(stored.get("case_id", "")).strip() == case_id:
                    return True
            return False

        return False

    # =========================
    # 统计与调试
    # =========================

    def size(self) -> int:
        return len(self.items)

    def stats(self) -> Dict[str, Any]:
        """
        返回经验库统计信息。
        """
        raw_case_count = 0
        prototype_count = 0
        confusion_count = 0
        rule_count = 0
        hard_case_count = 0
        unknown_count = 0

        for item in self.items:
            exp_type = str(item.get("experience_type", "")).strip()
            if exp_type == "raw_case":
                raw_case_count += 1
            elif exp_type == "prototype":
                prototype_count += 1
            elif exp_type == "confusion":
                confusion_count += 1
            elif exp_type == "rule":
                rule_count += 1
            elif exp_type == "hard_case":
                hard_case_count += 1
            else:
                unknown_count += 1

        return {
            "total": len(self.items),
            "raw_case": raw_case_count,
            "prototype": prototype_count,
            "confusion": confusion_count,
            "rule": rule_count,
            "hard_case": hard_case_count,
            "unknown": unknown_count,
        }

    def clear(self) -> None:
        """
        清空经验库。
        """
        self.items = []

    # =========================
    # 内部工具
    # =========================

    def _validate_item(self, item: Dict[str, Any]) -> None:
        if not isinstance(item, dict):
            raise TypeError("Experience item must be a dict.")

        exp_type = str(item.get("experience_type", "")).strip()
        if not exp_type:
            raise ValueError("Experience item must contain 'experience_type'.")
