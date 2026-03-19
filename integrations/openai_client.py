from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment-dependent fallback
    OpenAI = None


API_KEY = "sk-fh2OL0ZLDNzNXxV6IzznQLFfUn811UeixqCOupN0VHeSax6f"
BASE_URL = "https://ai.nengyongai.cn/v1"
DEFAULT_MODEL = "gpt-4o"


class OpenAICompatClient:
    """
    Lightweight OpenAI-compatible client used by perception and reporting skills.
    """

    def __init__(
        self,
        api_key: str = API_KEY,
        base_url: str = BASE_URL,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.client = (
            OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            if OpenAI is not None
            else None
        )

    def infer_derm_perception(
        self,
        image_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        if self.client is None:
            raise RuntimeError("openai package is not installed")
        metadata = metadata or {}
        image_data_url = self._to_data_url(image_path)

        system_prompt = (
            "You are a dermatology perception assistant. "
            "Your task is NOT to produce a final diagnosis report, "
            "but to extract structured perception signals from a skin lesion image. "
            "Return STRICT JSON only. No markdown, no explanations."
        )

        user_prompt = self._build_perception_prompt(metadata)

        response = self.client.chat.completions.create(
            model=model or self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url,
                            },
                        },
                    ],
                },
            ],
        )

        return response.choices[0].message.content or "{}"

    def infer_derm_report(
        self,
        final_decision: Dict[str, Any],
        visual_cues: Optional[list[str]] = None,
        retrieval_summary: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 900,
    ) -> str:
        if self.client is None:
            raise RuntimeError("openai package is not installed")
        system_prompt = (
            "You are a dermatology reporting assistant. "
            "You receive structured decision evidence from an agent pipeline. "
            "Return STRICT JSON only. No markdown, no explanations outside JSON."
        )

        user_prompt = self._build_report_prompt(
            final_decision=final_decision,
            visual_cues=visual_cues or [],
            retrieval_summary=retrieval_summary or {},
            metadata=metadata or {},
        )

        response = self.client.chat.completions.create(
            model=model or self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content or "{}"

    def _build_perception_prompt(self, metadata: Dict[str, Any]) -> str:
        age = metadata.get("age")
        sex = metadata.get("sex")
        site = (
            metadata.get("location")
            or metadata.get("site")
            or metadata.get("anatomical_site")
            or metadata.get("region")
        )
        extra = metadata.get("clinical_history") or metadata.get("history") or ""

        return f"""
Analyze this dermatology image and produce STRICT JSON with exactly this schema:

{{
  "ddx_candidates": [
    {{"name": "SCC", "score": 0.62}},
    {{"name": "ACK", "score": 0.28}},
    {{"name": "BCC", "score": 0.10}}
  ],
  "most_likely": {{"name": "SCC", "score": 0.62}},
  "visual_cues": [
    "free-text cue 1",
    "free-text cue 2"
  ],
  "risk_cues": {{
    "malignant_cues": [
      "free-text malignant cue"
    ],
    "suspicious_cues": [
      "free-text suspicious cue"
    ]
  }},
  "uncertainty": {{
    "level": "low"
  }}
}}

Rules:
1. Use ONLY these diagnosis labels when relevant:
MEL, NEV, SCC, BCC, ACK, SEK.
2. ddx_candidates must contain 1 to 5 items.
3. Scores must be floats between 0 and 1, roughly descending.
4. uncertainty.level must be one of: low, medium, high.
5. visual_cues should be short, concrete dermatology phrases.
6. Do not include any text outside JSON.

Clinical metadata:
- age: {age}
- sex: {sex}
- site: {site}
- extra_history: {extra}
""".strip()

    def _build_report_prompt(
        self,
        final_decision: Dict[str, Any],
        visual_cues: list[str],
        retrieval_summary: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        return f"""
Create a concise dermatology report from the structured agent outputs below.

Return STRICT JSON with exactly this schema:
{{
  "diagnosis": "SCC",
  "top_k": ["SCC", "ACK", "BCC"],
  "reasoning": "Short evidence-based reasoning.",
  "evidence": [
    "Visual cue 1",
    "Retrieval support 1"
  ],
  "risk_assessment": "Short risk summary.",
  "natural_language_report": "A short clinician-facing report."
}}

Rules:
1. Use diagnosis labels grounded in the supplied evidence.
2. Keep reasoning concise.
3. evidence must be a short list of concrete points.
4. Do not invent image findings beyond the provided cues.
5. Do not include markdown or any extra text outside JSON.

Final decision:
{json.dumps(final_decision, ensure_ascii=False)}

Visual cues:
{json.dumps(visual_cues, ensure_ascii=False)}

Retrieval summary:
{json.dumps(retrieval_summary, ensure_ascii=False)}

Metadata:
{json.dumps(metadata, ensure_ascii=False)}
""".strip()

    def _to_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "image/jpeg"

        image_bytes = path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
