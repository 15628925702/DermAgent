"""
Skill 基类。

以后所有 skill 都统一为：
`run(state: CaseState) -> dict`

这样你未来无论是自己手写 router，还是接 LangChain tool，都能共用同一套接口。
"""

from __future__ import annotations

from typing import Any, Dict

from agent.state import CaseState


class BaseSkill:
    name = "base_skill"

    def run(self, state: CaseState) -> Dict[str, Any]:
        raise NotImplementedError
