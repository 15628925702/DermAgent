"""
经验库容器。

基础版先用内存列表。
后面你可以替换成：
- JSON 文件
- sqlite
- embedding index
- faiss / vector db

相关文件：
- memory/schema.py
- memory/retriever.py
- memory/writer.py
"""

from __future__ import annotations

from typing import Dict, List


class ExperienceBank:
    def __init__(self) -> None:
        self.items: List[Dict] = []

    def add(self, item: Dict) -> None:
        self.items.append(item)

    def list_all(self) -> List[Dict]:
        return list(self.items)
