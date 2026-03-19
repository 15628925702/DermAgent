from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.state import CaseState
from memory.experience_bank import ExperienceBank
from memory.schema import (
    build_hard_case_experience,
    build_confusion_experience,
    build_prototype_experience,
    build_raw_case_experience,
)


class ExperienceWriter:
    """
    升级版经验写回器（Reflection-driven）

    新增能力：
    - 基于 reflection 自动决定写回策略
    - 支持 hard case 写回
    - 使用 structured reflection（prototype / confusion）
    """

    def write_case(
        self,
        state: CaseState,
        bank: ExperienceBank,
        auto: bool = True,
    ) -> Dict[str, Any]:
        reflection = state.reflection or {}
        fallback_case = self._is_fallback_case(state)

        summary: Dict[str, Any] = {
            "case_id": state.get_case_id(),
            "raw_case_written": False,
            "prototype_written": False,
            "confusion_written": False,
            "hard_case_written": False,
            "fallback_case": fallback_case,
            "skipped_reason": None,
        }

        # =========================
        # 1. raw case（始终写）
        # =========================
        raw_case_item = self._build_raw_case_item(state)
        summary["raw_case_written"] = bank.add_if_not_exists(raw_case_item)

        # =========================
        # 2. 自动策略（核心🔥）
        # =========================
        write_prototype = False
        write_confusion = False
        write_hard_case = False

        if auto:
            hints = reflection.get("writeback_hints", {}) or {}
            signals = reflection.get("learning_signals", {}) or {}

            write_prototype = hints.get("should_write_prototype", False)
            write_confusion = hints.get("should_write_confusion", False)
            write_hard_case = signals.get("hard_case", False)

        if fallback_case:
            write_prototype = False
            write_confusion = False
            write_hard_case = True
            summary["skipped_reason"] = "fallback_perception_only_raw_and_hard_case"

        # =========================
        # 3. prototype
        # =========================
        if write_prototype:
            prototype_item = self._build_prototype_item(state)
            if prototype_item:
                summary["prototype_written"] = bank.add_if_not_exists(prototype_item)

        # =========================
        # 4. confusion
        # =========================
        if write_confusion:
            confusion_item = self._build_confusion_item(state)
            if confusion_item:
                summary["confusion_written"] = bank.add_if_not_exists(confusion_item)

        # =========================
        # 5. hard case（新增🔥）
        # =========================
        if write_hard_case:
            hard_case_item = self._build_hard_case_item(state)
            if hard_case_item:
                summary["hard_case_written"] = bank.add_if_not_exists(hard_case_item)

        state.trace(
            "experience_writeback",
            "success",
            "Experience writeback completed",
            payload=summary,
        )
        return summary

    # =========================
    # Raw case
    # =========================

    def _build_raw_case_item(self, state: CaseState) -> Dict[str, Any]:
        return build_raw_case_experience(
            case_id=state.get_case_id(),
            perception=state.perception,
            final_decision=state.final_decision,
            selected_skills=list(state.selected_skills),
            retrieval=state.retrieval,
            metadata=state.get_metadata(),
        )

    # =========================
    # Prototype（用 reflection🔥）
    # =========================

    def _build_prototype_item(self, state: CaseState) -> Optional[Dict[str, Any]]:
        reflection = state.reflection or {}
        proto = reflection.get("prototype_features")

        if not proto:
            return None

        return build_prototype_experience(
            disease=proto.get("label"),
            typical_cues=proto.get("visual_cues", [])[:8],
            typical_metadata=state.get_metadata(),
            common_confusions=self._extract_other_ddx(state, proto.get("label", ""))[:5],
            recommended_skills=list(state.selected_skills)[:5],
        )

    # =========================
    # Confusion（用 reflection🔥）
    # =========================

    def _build_confusion_item(self, state: CaseState) -> Optional[Dict[str, Any]]:
        reflection = state.reflection or {}
        info = reflection.get("confusion_info")

        if not info:
            return None

        pair = info.get("pair", [])
        if len(pair) != 2:
            return None

        d1, d2 = pair

        return build_confusion_experience(
            disease_a=d1,
            disease_b=d2,
            distinguishing_clues=self._extract_visual_cues(state)[:8],
            useful_skills=list(state.selected_skills)[:6],
            failure_modes=self._build_failure_modes(state),
        )

    # =========================
    # Hard case（新增🔥）
    # =========================

    def _build_hard_case_item(self, state: CaseState) -> Optional[Dict[str, Any]]:
        reflection = state.reflection or {}

        return build_hard_case_experience(
            case_id=state.get_case_id(),
            final_label=reflection.get("final_label", ""),
            top_ddx=reflection.get("top_ddx", []),
            uncertainty=reflection.get("uncertainty", ""),
            learning_signals=reflection.get("learning_signals", {}) or {},
            selected_skills=list(state.selected_skills),
        )

    # =========================
    # Helpers
    # =========================

    def _extract_visual_cues(self, state: CaseState) -> List[str]:
        cues = state.perception.get("visual_cues", []) or []
        clean: List[str] = []

        for cue in cues:
            text = str(cue).strip()
            if text and text not in clean:
                clean.append(text)

        return clean

    def _extract_other_ddx(self, state: CaseState, exclude: str) -> List[str]:
        exclude = str(exclude).strip().upper()
        ddx = state.perception.get("ddx_candidates", []) or []

        names: List[str] = []
        for item in ddx:
            name = str(item.get("name", "")).strip().upper()
            if not name or name == exclude:
                continue
            if name not in names:
                names.append(name)

        return names

    def _build_failure_modes(self, state: CaseState) -> List[str]:
        modes: List[str] = []

        reflection = state.reflection or {}
        signals = reflection.get("learning_signals", {}) or {}

        if signals.get("hard_case"):
            modes.append("hard_case")

        if signals.get("low_retrieval_support"):
            modes.append("weak_retrieval_support")

        if state.get_uncertainty_level() == "high":
            modes.append("high_uncertainty")

        if not state.perception.get("visual_cues"):
            modes.append("limited_visual_clues")

        return modes

    def _is_fallback_case(self, state: CaseState) -> bool:
        return bool((state.perception or {}).get("fallback_reason"))
