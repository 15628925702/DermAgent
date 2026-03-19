from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperienceBank:
    def __init__(self, initial_items: Optional[List[Dict[str, Any]]] = None) -> None:
        self.items: List[Dict[str, Any]] = []
        if initial_items:
            self.extend(initial_items)

    def add(self, item: Dict[str, Any]) -> None:
        self._validate_item(item)
        self.items.append(dict(item))

    def extend(self, items: List[Dict[str, Any]]) -> None:
        for item in items:
            self.add(item)

    def add_if_not_exists(self, item: Dict[str, Any]) -> bool:
        self._validate_item(item)
        if self.exists(item):
            return False
        self.items.append(dict(item))
        return True

    def replace_type(self, experience_type: str, items: List[Dict[str, Any]]) -> Dict[str, int]:
        experience_type = str(experience_type).strip()
        validated: List[Dict[str, Any]] = []
        for item in items:
            self._validate_item(item)
            if str(item.get("experience_type", "")).strip() != experience_type:
                raise ValueError(
                    f"replace_type expected '{experience_type}' items, got '{item.get('experience_type')}'."
                )
            validated.append(dict(item))

        removed = sum(
            1 for item in self.items if str(item.get("experience_type", "")).strip() == experience_type
        )
        kept = [
            dict(item)
            for item in self.items
            if str(item.get("experience_type", "")).strip() != experience_type
        ]
        self.items = kept + validated
        return {
            "removed": removed,
            "added": len(validated),
        }

    def list_all(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.items]

    def list_by_type(self, experience_type: str) -> List[Dict[str, Any]]:
        experience_type = str(experience_type).strip()
        return [
            dict(item)
            for item in self.items
            if str(item.get("experience_type", "")).strip() == experience_type
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

    def exists(self, item: Dict[str, Any]) -> bool:
        exp_type = str(item.get("experience_type", "")).strip()

        if exp_type == "raw_case":
            case_id = str(item.get("case_id", "")).strip()
            return bool(case_id) and self.find_raw_case_by_case_id(case_id) is not None

        if exp_type == "prototype":
            disease = str(item.get("disease", "")).strip().upper()
            return bool(disease) and self.find_prototype_by_disease(disease) is not None

        if exp_type == "confusion":
            pair = [str(x).strip().upper() for x in item.get("pair", [])]
            return len(pair) == 2 and bool(pair[0]) and bool(pair[1]) and self.find_confusion_by_pair(pair[0], pair[1]) is not None

        if exp_type == "rule":
            rule_name = str(item.get("rule_name", "")).strip()
            return bool(rule_name) and self.find_rule_by_name(rule_name) is not None

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

    def size(self) -> int:
        return len(self.items)

    def stats(self) -> Dict[str, Any]:
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
        self.items = []

    def save_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "items": self.list_all(),
            "stats": self.stats(),
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return output_path

    def load_json(self, path: str | Path) -> None:
        input_path = Path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        items = payload.get("items", []) or []
        self.items = []
        for item in items:
            self.add(item)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperienceBank":
        bank = cls()
        bank.load_json(path)
        return bank

    def _validate_item(self, item: Dict[str, Any]) -> None:
        if not isinstance(item, dict):
            raise TypeError("Experience item must be a dict.")
        exp_type = str(item.get("experience_type", "")).strip()
        if not exp_type:
            raise ValueError("Experience item must contain 'experience_type'.")
