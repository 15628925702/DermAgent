from __future__ import annotations

import json
from typing import Any, Dict

from agent.state import CaseState
from integrations.openai_client import OpenAICompatClient
from skills.base import BaseSkill


class ReportSkill(BaseSkill):
    name = "report_skill"

    def __init__(self, model: str = "gpt-4o") -> None:
        self.client = OpenAICompatClient(model=model)

    def run(self, state: CaseState) -> Dict[str, Any]:
        final_decision = state.final_decision or {}
        if not final_decision:
            result = self._fallback_report(state, reason="missing_final_decision")
            state.report = result
            state.trace(self.name, "warning", "Report fallback used: missing final decision")
            return result

        try:
            raw_text = self.client.infer_derm_report(
                final_decision=final_decision,
                visual_cues=state.perception.get("visual_cues", []) or [],
                retrieval_summary=state.retrieval.get("retrieval_summary", {}) or {},
                metadata=state.get_metadata(),
            )
            parsed = self._parse_json(raw_text)
            result = self._normalize_report(parsed, state, generation_mode="gpt")
            state.report = result
            state.trace(
                self.name,
                "success",
                "Structured report generated",
                payload={
                    "diagnosis": result.get("diagnosis", "UNKNOWN"),
                    "top_k": result.get("top_k", [])[:3],
                    "generation_mode": result.get("generation_mode", "gpt"),
                },
            )
            return result
        except Exception as e:
            result = self._fallback_report(state, reason=f"api_error:{type(e).__name__}")
            state.report = result
            state.trace(
                self.name,
                "warning",
                f"Report generation failed, fallback used: {e}",
                payload={"error_type": type(e).__name__},
            )
            return result

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return {}

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            cleaned = raw_text.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)

    def _normalize_report(
        self,
        data: Dict[str, Any],
        state: CaseState,
        generation_mode: str,
    ) -> Dict[str, Any]:
        fallback = self._fallback_report(state, reason="normalization_fallback")
        final_decision = state.final_decision or {}

        diagnosis = self._norm_label(
            data.get("diagnosis")
            or final_decision.get("diagnosis")
            or final_decision.get("final_label")
        )
        top_k = self._normalize_top_k(data.get("top_k"), fallback.get("top_k", []))
        reasoning = self._safe_text(data.get("reasoning"), fallback.get("reasoning", ""))
        evidence = self._normalize_text_list(data.get("evidence"), fallback.get("evidence", []))
        risk_assessment = self._safe_text(
            data.get("risk_assessment"),
            fallback.get("risk_assessment", ""),
        )
        natural_language_report = self._safe_text(
            data.get("natural_language_report"),
            fallback.get("natural_language_report", ""),
        )

        return {
            "diagnosis": diagnosis or fallback["diagnosis"],
            "top_k": top_k,
            "reasoning": reasoning,
            "evidence": evidence,
            "risk_assessment": risk_assessment,
            "natural_language_report": natural_language_report,
            "generation_mode": generation_mode,
        }

    def _fallback_report(self, state: CaseState, reason: str) -> Dict[str, Any]:
        final_decision = state.final_decision or {}
        retrieval_summary = state.retrieval.get("retrieval_summary", {}) or {}
        risk_summary = final_decision.get("risk_summary", {}) or {}

        diagnosis = self._norm_label(
            final_decision.get("diagnosis")
            or final_decision.get("final_label")
            or state.perception.get("most_likely", {}).get("name")
            or "UNKNOWN"
        )
        top_k = self._normalize_top_k(final_decision.get("top_k", []), [diagnosis or "UNKNOWN"])
        visual_cues = self._normalize_text_list(state.perception.get("visual_cues", []), [])
        support_labels = self._normalize_top_k(retrieval_summary.get("support_labels", []), [])

        evidence = []
        for cue in visual_cues[:3]:
            evidence.append(f"visual cue: {cue}")
        for label in support_labels[:2]:
            evidence.append(f"retrieval support: {label}")
        if not evidence:
            evidence.append("limited explicit evidence available")

        reasoning_parts = []
        if diagnosis:
            reasoning_parts.append(f"Final aggregated diagnosis favors {diagnosis}.")
        if visual_cues:
            reasoning_parts.append(f"Visual cues included {', '.join(visual_cues[:3])}.")
        if support_labels:
            reasoning_parts.append(f"Retrieved experience supported {', '.join(support_labels[:3])}.")
        if not reasoning_parts:
            reasoning_parts.append("Fallback report generated from available structured outputs.")

        risk_level = str(risk_summary.get("risk_level", "unknown")).lower()
        suspicious = bool(risk_summary.get("suspicious_malignancy", False))
        if suspicious:
            risk_text = f"Risk level is {risk_level}; malignancy-oriented caution is advised."
        else:
            risk_text = f"Risk level is {risk_level}."

        return {
            "diagnosis": diagnosis or "UNKNOWN",
            "top_k": top_k,
            "reasoning": " ".join(reasoning_parts),
            "evidence": evidence,
            "risk_assessment": risk_text,
            "natural_language_report": (
                f"Most likely diagnosis is {diagnosis or 'UNKNOWN'}. "
                f"Key evidence includes {', '.join(evidence[:3])}. "
                f"{risk_text}"
            ),
            "generation_mode": "fallback",
            "fallback_reason": reason,
        }

    def _normalize_top_k(self, value: Any, default: list[str]) -> list[str]:
        items = value if isinstance(value, list) else default
        out: list[str] = []
        for item in items:
            if isinstance(item, dict):
                name = self._norm_label(item.get("name"))
            else:
                name = self._norm_label(item)
            if name and name not in out:
                out.append(name)
        return out or list(default)

    def _normalize_text_list(self, value: Any, default: list[str]) -> list[str]:
        if not isinstance(value, list):
            return list(default)
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
        return out or list(default)

    def _safe_text(self, value: Any, default: str) -> str:
        text = str(value).strip() if value is not None else ""
        return text or default

    def _norm_label(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().upper()
