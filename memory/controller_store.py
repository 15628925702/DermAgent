from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from agent.controller import LearnableSkillController
from agent.final_scorer import LearnableFinalScorer
from agent.rule_scorer import LearnableRuleScorer
from memory.skill_index import SkillIndex, build_default_skill_index


def save_controller_checkpoint(
    path: str | Path,
    *,
    skill_index: SkillIndex,
    controller: LearnableSkillController | None,
    final_scorer: LearnableFinalScorer | None = None,
    rule_scorer: LearnableRuleScorer | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format_version": 3,
        "metadata": metadata or {},
        "skill_index": skill_index.as_dict(),
        "controller": controller.to_dict() if controller is not None else None,
        "final_scorer": final_scorer.to_dict() if final_scorer is not None else None,
        "rule_scorer": rule_scorer.to_dict() if rule_scorer is not None else None,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def load_controller_checkpoint(path: str | Path) -> Tuple[SkillIndex, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    default_index = build_default_skill_index()
    skill_payload = payload.get("skill_index", {}) or {}
    if skill_payload:
        loaded_index = SkillIndex.from_dict(skill_payload)
        for spec in loaded_index.all_specs():
            default_index.register(spec)
    skill_index = default_index
    controller_payload = payload.get("controller", {}) or {}
    final_scorer_payload = payload.get("final_scorer", {}) or {}
    rule_scorer_payload = payload.get("rule_scorer", {}) or {}
    return skill_index, controller_payload, final_scorer_payload, rule_scorer_payload
