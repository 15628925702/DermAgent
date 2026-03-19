"""
Experience schema。

基础版先统一四类经验：
- raw_case
- prototype
- confusion
- rule

相关文件：
- memory/writer.py
- memory/experience_bank.py
- agent/reflection.py
"""

from __future__ import annotations

from typing import Any, Dict


def build_raw_case_experience(case_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experience_type": "raw_case",
        "case_id": case_id,
        "payload": payload,
    }
